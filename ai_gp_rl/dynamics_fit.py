"""Fit the AI-GP surrogate to synchronized command and telemetry sessions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares


GRAVITY_MPS2 = 9.81
DEFAULT_SAMPLE_DT_S = 0.02


@dataclass(frozen=True)
class TranslationParameters:
    thrust_acceleration_bias_mps2: float
    thrust_acceleration_gain_mps2: float
    linear_drag_xyz: tuple[float, float, float]
    quadratic_drag_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class RateParameters:
    command_latency_s: float
    rate_response_gain: tuple[float, float, float]
    yaw_gain_inferred: bool = True


@dataclass(frozen=True)
class Trace:
    session_id: str
    time_s: np.ndarray
    position_m: np.ndarray
    absolute_attitude_rad: np.ndarray
    internal_attitude_rad: np.ndarray
    commands: np.ndarray


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_trace(
    session_dir: Path,
    *,
    duration_s: float,
    sample_dt_s: float = DEFAULT_SAMPLE_DT_S,
) -> Trace:
    commands = [
        row
        for row in read_jsonl(session_dir / "commands.jsonl")
        if row.get("source") == "controller"
    ]
    telemetry = [
        row
        for row in read_jsonl(session_dir / "telemetry.jsonl")
        if isinstance(row.get("position_m"), dict)
        and isinstance(row.get("attitude_rad"), dict)
    ]
    if not commands or not telemetry:
        raise ValueError(f"{session_dir.name} has no synchronized command trace")

    command_start_s = float(commands[0]["monotonic_time_s"])
    command_end_s = float(commands[-1]["monotonic_time_s"])
    end_s = min(float(duration_s), command_end_s - command_start_s)
    if end_s <= sample_dt_s:
        raise ValueError(f"{session_dir.name} command trace is too short")
    time_s = np.arange(0.0, end_s + sample_dt_s * 0.25, sample_dt_s)

    telemetry_time_s = np.asarray(
        [
            float(row["monotonic_time_s"]) - command_start_s
            for row in telemetry
        ],
        dtype=float,
    )
    unique = np.r_[True, np.diff(telemetry_time_s) > 1e-7]
    telemetry_time_s = telemetry_time_s[unique]
    telemetry = [row for row, keep in zip(telemetry, unique) if keep]

    position_ned = np.asarray(
        [
            (
                float(row["position_m"]["x"]),
                float(row["position_m"]["y"]),
                float(row["position_m"]["z"]),
            )
            for row in telemetry
        ]
    )
    position_surrogate = np.column_stack(
        (-position_ned[:, 0], position_ned[:, 1], -position_ned[:, 2])
    )
    position_m = _interpolate_columns(
        telemetry_time_s, position_surrogate, time_s
    )
    position_m -= position_m[0]

    attitude = np.unwrap(
        np.asarray(
            [
                (
                    float(row["attitude_rad"]["roll_rad"]),
                    float(row["attitude_rad"]["pitch_rad"]),
                    float(row["attitude_rad"]["yaw_rad"]),
                )
                for row in telemetry
            ]
        ),
        axis=0,
    )
    absolute_attitude = _interpolate_columns(
        telemetry_time_s, attitude, time_s
    )
    attitude_delta = absolute_attitude - absolute_attitude[0]
    internal_attitude = attitude_delta.copy()
    internal_attitude[:, 1] *= -1.0

    command_time_s = np.asarray(
        [
            float(row["monotonic_time_s"]) - command_start_s
            for row in commands
        ]
    )
    command_values = np.asarray(
        [
            (
                float(row["thrust_normalized"]),
                float(row.get("roll_rate_radps", 0.0)),
                float(row.get("pitch_rate_radps", 0.0)),
                float(row.get("yaw_rate_radps", 0.0)),
            )
            for row in commands
        ]
    )
    sampled_commands = _sample_held_commands(
        command_time_s, command_values, time_s
    )
    return Trace(
        session_id=session_dir.name,
        time_s=time_s,
        position_m=position_m,
        absolute_attitude_rad=absolute_attitude,
        internal_attitude_rad=internal_attitude,
        commands=sampled_commands,
    )


def fit_rate_parameters(
    roll_traces: Iterable[Trace],
    pitch_traces: Iterable[Trace],
    *,
    max_latency_s: float = 0.10,
    latency_step_s: float = 0.001,
) -> tuple[RateParameters, dict[str, float]]:
    roll_traces = tuple(roll_traces)
    pitch_traces = tuple(pitch_traces)
    if not roll_traces or not pitch_traces:
        raise ValueError("roll and pitch traces are both required")

    best: tuple[float, float, float, float, float, float] | None = None
    for latency_s in np.arange(
        0.0, max_latency_s + latency_step_s * 0.25, latency_step_s
    ):
        roll_gain, roll_rmse = _fit_axis_gain(
            roll_traces, axis=0, latency_s=latency_s
        )
        pitch_gain, pitch_rmse = _fit_axis_gain(
            pitch_traces, axis=1, latency_s=latency_s
        )
        objective = roll_rmse * roll_rmse + pitch_rmse * pitch_rmse
        candidate = (
            objective,
            latency_s,
            roll_gain,
            pitch_gain,
            roll_rmse,
            pitch_rmse,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, latency_s, roll_gain, pitch_gain, roll_rmse, pitch_rmse = best
    yaw_gain = (roll_gain + pitch_gain) / 2.0
    return (
        RateParameters(
            command_latency_s=float(latency_s),
            rate_response_gain=(
                float(roll_gain),
                float(pitch_gain),
                float(yaw_gain),
            ),
        ),
        {
            "roll_angle_rmse_rad": float(roll_rmse),
            "pitch_angle_rmse_rad": float(pitch_rmse),
        },
    )


def fit_translation_parameters(
    traces: Iterable[Trace],
    *,
    sample_stride: int = 3,
) -> TranslationParameters:
    traces = tuple(traces)
    if not traces:
        raise ValueError("at least one translation trace is required")

    sample_indices = [
        np.arange(5, len(trace.time_s), sample_stride) for trace in traces
    ]

    def residual(parameters: np.ndarray) -> np.ndarray:
        errors: list[np.ndarray] = []
        for trace, indices in zip(traces, sample_indices):
            predicted = simulate_translation_conditioned_on_attitude(
                trace, _translation_parameters_from_array(parameters)
            )
            errors.append(
                (predicted[indices] - trace.position_m[indices])[:, (0, 2)]
                .reshape(-1)
            )
        return np.concatenate(errors)

    result = least_squares(
        residual,
        np.asarray((-5.0, 55.0, 0.15, 0.35, 0.02, 0.01)),
        bounds=(
            np.asarray((-30.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            np.asarray((10.0, 150.0, 3.0, 3.0, 0.30, 0.30)),
        ),
        loss="soft_l1",
        f_scale=0.30,
        max_nfev=1500,
    )
    if not result.success:
        raise RuntimeError(f"translation fit failed: {result.message}")
    return _translation_parameters_from_array(result.x)


def simulate_translation_conditioned_on_attitude(
    trace: Trace,
    parameters: TranslationParameters,
) -> np.ndarray:
    direction = _thrust_directions(trace.absolute_attitude_rad)
    return _integrate_translation(
        trace.time_s,
        trace.commands[:, 0],
        direction,
        parameters,
    )


def simulate_closed_loop(
    trace: Trace,
    translation: TranslationParameters,
    rates: RateParameters,
) -> tuple[np.ndarray, np.ndarray]:
    delayed_commands = _delay_commands(
        trace.time_s, trace.commands, rates.command_latency_s
    )
    predicted_attitude = np.zeros_like(trace.internal_attitude_rad)
    dt = float(trace.time_s[1] - trace.time_s[0])
    rate_gain = np.asarray(rates.rate_response_gain)
    for index in range(len(trace.time_s) - 1):
        predicted_attitude[index + 1] = (
            predicted_attitude[index]
            + rate_gain * delayed_commands[index, 1:] * dt
        )

    base_attitude = trace.absolute_attitude_rad[0]
    physical_attitude = predicted_attitude.copy()
    physical_attitude[:, 0] += base_attitude[0]
    physical_attitude[:, 1] = base_attitude[1] - predicted_attitude[:, 1]
    physical_attitude[:, 2] += base_attitude[2]
    direction = _thrust_directions(physical_attitude)
    predicted_position = _integrate_translation(
        trace.time_s,
        delayed_commands[:, 0],
        direction,
        translation,
    )
    return predicted_position, predicted_attitude


def trace_validation(
    trace: Trace,
    predicted_position: np.ndarray,
    predicted_attitude: np.ndarray | None = None,
) -> dict[str, Any]:
    position_error = predicted_position - trace.position_m
    result: dict[str, Any] = {
        "session_id": trace.session_id,
        "duration_s": float(trace.time_s[-1]),
        "position_rmse_m": _float_list(
            np.sqrt(np.mean(position_error * position_error, axis=0))
        ),
        "position_final_error_m": _float_list(position_error[-1]),
    }
    if predicted_attitude is not None:
        attitude_error = predicted_attitude - trace.internal_attitude_rad
        result["attitude_rmse_rad"] = _float_list(
            np.sqrt(np.mean(attitude_error * attitude_error, axis=0))
        )
        result["attitude_final_error_rad"] = _float_list(attitude_error[-1])
    return result


def build_fit_report(
    *,
    sessions_root: Path,
    translation_training_ids: Iterable[str],
    rate_roll_training_ids: Iterable[str],
    rate_pitch_training_ids: Iterable[str],
    held_out_calibration_ids: Iterable[str],
    held_out_policy_ids: Iterable[str],
    fit_duration_s: float = 3.0,
) -> dict[str, Any]:
    translation_training_ids = tuple(translation_training_ids)
    rate_roll_training_ids = tuple(rate_roll_training_ids)
    rate_pitch_training_ids = tuple(rate_pitch_training_ids)
    held_out_calibration_ids = tuple(held_out_calibration_ids)
    held_out_policy_ids = tuple(held_out_policy_ids)

    translation_traces = [
        load_trace(
            sessions_root / session_id,
            duration_s=fit_duration_s,
        )
        for session_id in translation_training_ids
    ]
    roll_traces = [
        load_trace(
            sessions_root / session_id,
            duration_s=0.34,
            sample_dt_s=0.005,
        )
        for session_id in rate_roll_training_ids
    ]
    pitch_traces = [
        load_trace(
            sessions_root / session_id,
            duration_s=1.5,
            sample_dt_s=0.005,
        )
        for session_id in rate_pitch_training_ids
    ]

    rates, rate_training_error = fit_rate_parameters(
        roll_traces, pitch_traces
    )
    translation = fit_translation_parameters(translation_traces)

    training_validation = [
        trace_validation(
            trace,
            simulate_translation_conditioned_on_attitude(trace, translation),
        )
        for trace in translation_traces
    ]
    held_out_calibration_validation = []
    for session_id in held_out_calibration_ids:
        trace = load_trace(
            sessions_root / session_id,
            duration_s=fit_duration_s,
        )
        predicted_position, predicted_attitude = simulate_closed_loop(
            trace, translation, rates
        )
        held_out_calibration_validation.append(
            trace_validation(trace, predicted_position, predicted_attitude)
        )

    held_out_policy_validation = []
    for session_id in held_out_policy_ids:
        trace = load_trace(
            sessions_root / session_id,
            duration_s=fit_duration_s,
            sample_dt_s=0.005,
        )
        predicted_position, predicted_attitude = simulate_closed_loop(
            trace, translation, rates
        )
        held_out_policy_validation.append(
            trace_validation(trace, predicted_position, predicted_attitude)
        )

    return {
        "schema_version": 1,
        "profile_name": "measured_ai_gp_v1_20260614",
        "coordinate_contract": (
            "telemetry NED position -> surrogate (-north, east, -down); "
            "attitude is relative to launch except physical thrust retains the "
            "measured launch pitch"
        ),
        "fit": {
            "sample_duration_s": fit_duration_s,
            "translation_training_sessions": list(translation_training_ids),
            "rate_roll_training_sessions": list(rate_roll_training_ids),
            "rate_pitch_training_sessions": list(rate_pitch_training_ids),
            "held_out_calibration_sessions": list(held_out_calibration_ids),
            "held_out_policy_sessions": list(held_out_policy_ids),
        },
        "parameters": {
            **asdict(translation),
            **asdict(rates),
            "base_pitch_offset_rad": float(
                np.median(
                    [
                        trace.absolute_attitude_rad[0, 1]
                        for trace in translation_traces
                    ]
                )
            ),
        },
        "rate_training_error": rate_training_error,
        "validation": {
            "translation_training": training_validation,
            "held_out_calibration": held_out_calibration_validation,
            "held_out_policy": held_out_policy_validation,
        },
    }


def write_fit_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fit_axis_gain(
    traces: tuple[Trace, ...],
    *,
    axis: int,
    latency_s: float,
) -> tuple[float, float]:
    integrated_commands: list[np.ndarray] = []
    measured_angles: list[np.ndarray] = []
    for trace in traces:
        delayed = _delay_commands(
            trace.time_s, trace.commands, latency_s
        )[:, axis + 1]
        dt = np.diff(trace.time_s)
        integrated = np.r_[
            0.0,
            np.cumsum((delayed[:-1] + delayed[1:]) * 0.5 * dt),
        ]
        integrated_commands.append(integrated)
        measured_angles.append(trace.internal_attitude_rad[:, axis])
    command = np.concatenate(integrated_commands)
    angle = np.concatenate(measured_angles)
    denominator = float(command @ command)
    if denominator <= 1e-12:
        raise ValueError("rate trace has no command excitation")
    gain = float(command @ angle) / denominator
    error = gain * command - angle
    return gain, float(np.sqrt(np.mean(error * error)))


def _translation_parameters_from_array(
    values: np.ndarray,
) -> TranslationParameters:
    bias, gain, linear_x, linear_z, quadratic_x, quadratic_z = (
        float(value) for value in values
    )
    return TranslationParameters(
        thrust_acceleration_bias_mps2=bias,
        thrust_acceleration_gain_mps2=gain,
        linear_drag_xyz=(linear_x, linear_x, linear_z),
        quadratic_drag_xyz=(quadratic_x, quadratic_x, quadratic_z),
    )


def _integrate_translation(
    time_s: np.ndarray,
    thrust_command: np.ndarray,
    direction: np.ndarray,
    parameters: TranslationParameters,
) -> np.ndarray:
    position = np.zeros((len(time_s), 3), dtype=float)
    velocity = np.zeros(3, dtype=float)
    linear_drag = np.asarray(parameters.linear_drag_xyz)
    quadratic_drag = np.asarray(parameters.quadratic_drag_xyz)
    for index, dt in enumerate(np.diff(time_s)):
        collective_acceleration = max(
            0.0,
            parameters.thrust_acceleration_bias_mps2
            + parameters.thrust_acceleration_gain_mps2
            * thrust_command[index],
        )
        drag = (
            linear_drag * velocity
            + quadratic_drag * np.abs(velocity) * velocity
        )
        acceleration = (
            direction[index] * collective_acceleration
            + np.asarray((0.0, 0.0, -GRAVITY_MPS2))
            - drag
        )
        velocity += acceleration * dt
        position[index + 1] = position[index] + velocity * dt
    return position


def _thrust_directions(attitude_rad: np.ndarray) -> np.ndarray:
    roll = attitude_rad[:, 0]
    pitch = attitude_rad[:, 1]
    yaw = attitude_rad[:, 2] - attitude_rad[0, 2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.column_stack(
        (
            cy * sp * cr + sy * sr,
            sy * sp * cr - cy * sr,
            cp * cr,
        )
    )


def _delay_commands(
    time_s: np.ndarray,
    commands: np.ndarray,
    latency_s: float,
) -> np.ndarray:
    delayed_time = time_s - latency_s
    delayed = np.column_stack(
        [
            np.interp(
                delayed_time,
                time_s,
                commands[:, column],
                left=commands[0, column] if column == 0 else 0.0,
                right=commands[-1, column],
            )
            for column in range(commands.shape[1])
        ]
    )
    return delayed


def _sample_held_commands(
    command_time_s: np.ndarray,
    command_values: np.ndarray,
    sample_time_s: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(
        command_time_s, sample_time_s, side="right"
    ) - 1
    indices = np.maximum(indices, 0)
    return command_values[indices]


def _interpolate_columns(
    source_time_s: np.ndarray,
    values: np.ndarray,
    target_time_s: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_time_s, source_time_s, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]
