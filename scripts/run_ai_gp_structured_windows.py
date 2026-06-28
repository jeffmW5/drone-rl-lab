from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


LAB_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = LAB_ROOT.parent
STACK_ROOT = Path(
    os.environ.get(
        "AI_GP_RUNTIME_ROOT",
        LAB_ROOT / "tmp" / "ai-grand-prix-stack-remote",
    )
).expanduser().resolve()
if not (STACK_ROOT / "adapter" / "dcl_runtime.py").exists():
    raise RuntimeError(
        "AI-GP operational runtime not found. Set AI_GP_RUNTIME_ROOT to the "
        "Windows execution worktree."
    )
for path in (LAB_ROOT, STACK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapter.commands import CommandSource, ControlCommand, RuntimeAction
from adapter.dcl_runtime import DclRuntime
from adapter.runtime import AdapterConfig, RuntimeEvent
from adapter.telemetry import TelemetrySample
from ai_gp_rl.session_dataset import _rotation_matrix, _transpose_matvec
from calibration.run_thrust_sweep import _reset_simulator
from replay.recording import SessionEvaluator, SessionRecorder


ACTION_NAMES = (
    "collective_offset",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
)
STRUCTURED_TEACHER_FEATURE_NAMES = (
    "active_gate_position_body_x",
    "active_gate_position_body_y",
    "active_gate_position_body_z",
    "active_gate_normal_body_x",
    "active_gate_normal_body_y",
    "active_gate_normal_body_z",
    "next_gate_position_body_x",
    "next_gate_position_body_y",
    "next_gate_position_body_z",
    "next_gate_normal_body_x",
    "next_gate_normal_body_y",
    "next_gate_normal_body_z",
    "body_velocity_x",
    "body_velocity_y",
    "body_velocity_z",
    "gravity_body_x",
    "gravity_body_y",
    "gravity_body_z",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
    "previous_collective_offset",
    "previous_roll_rate",
    "previous_pitch_rate",
    "previous_yaw_rate",
    "gate_index",
)
VELOCITY_SCALE_MPS = 8.0
RATE_SCALES_RADPS = (3.0, 3.0, 2.0)


@dataclass(frozen=True)
class StructuredPolicy:
    artifact: dict[str, Any]
    layers: list[dict[str, Any]]
    actor_features: tuple[str, ...]
    gates_ned: tuple[tuple[float, float, float], ...]
    gates_surrogate: tuple[tuple[float, float, float], ...]
    gate_normals_surrogate: tuple[tuple[float, float, float], ...]
    surrogate_altitude_offset_m: float
    gate_width_m: float
    gate_height_m: float
    validation_status: str
    action_calibration: dict[str, float]

    @classmethod
    def load(cls, path: str | Path) -> "StructuredPolicy":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        _validate_artifact(artifact)
        track = artifact["track"]
        gates_ned = tuple(
            tuple(float(value) for value in gate["center_ned_m"])
            for gate in sorted(track["gates_ned"], key=lambda item: int(item["index"]))
        )
        gates_surrogate_payload = sorted(
            track["gates_surrogate_flu"], key=lambda item: int(item["index"])
        )
        gates_surrogate = tuple(
            tuple(float(value) for value in gate["center_m"])
            for gate in gates_surrogate_payload
        )
        gate_normals = tuple(
            tuple(float(value) for value in gate["normal_unit"])
            for gate in gates_surrogate_payload
        )
        size = track["gate_size_m"]
        calibration = {
            key: float(artifact["action_calibration"][key])
            for key in (
                "thrust_command_center",
                "thrust_span_up",
                "thrust_span_down",
                "max_roll_rate_radps",
                "max_pitch_rate_radps",
                "max_yaw_rate_radps",
            )
        }
        return cls(
            artifact=artifact,
            layers=artifact["layers"],
            actor_features=tuple(artifact["actor_features"]),
            gates_ned=gates_ned,
            gates_surrogate=gates_surrogate,
            gate_normals_surrogate=gate_normals,
            surrogate_altitude_offset_m=float(
                track["surrogate_altitude_offset_m"]
            ),
            gate_width_m=float(size["width"]),
            gate_height_m=float(size["height"]),
            validation_status=str(artifact.get("validation_status", "")),
            action_calibration=calibration,
        )

    @property
    def gate_count(self) -> int:
        return len(self.gates_ned)

    def verify_test_vectors(self, tolerance: float = 1e-5) -> None:
        vectors = self.artifact.get("test_vectors") or []
        if not vectors:
            raise ValueError("artifact has no inference test vectors")
        for vector in vectors:
            action = self.act(vector["observation"])
            expected = tuple(float(value) for value in vector["expected_action"])
            if any(
                abs(actual_value - expected_value) > tolerance
                for actual_value, expected_value in zip(action, expected)
            ):
                raise ValueError("artifact inference self-test failed")
            command = self.map_action(action)
            expected_command = vector.get("expected_command") or {}
            for key, value in expected_command.items():
                if abs(command[key] - float(value)) > tolerance:
                    raise ValueError("artifact command-map self-test failed")

    def act(self, observation: Sequence[float]) -> tuple[float, ...]:
        if len(observation) != len(self.actor_features):
            raise ValueError(
                f"expected {len(self.actor_features)} observations, "
                f"received {len(observation)}"
            )
        values = [float(value) for value in observation]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("policy observation contains non-finite values")
        for layer_index, layer in enumerate(self.layers):
            output = [
                float(bias)
                + sum(float(weight) * value for weight, value in zip(row, values))
                for row, bias in zip(layer["weight"], layer["bias"])
            ]
            if layer_index < len(self.layers) - 1:
                values = [value if value >= 0.0 else 0.2 * value for value in output]
            else:
                values = [math.tanh(value) for value in output]
        if len(values) != len(ACTION_NAMES):
            raise ValueError("artifact produced the wrong action dimension")
        return tuple(values)

    def map_action(self, action: Sequence[float]) -> dict[str, float]:
        collective, roll, pitch, yaw = (
            _clamp(float(value), -1.0, 1.0) for value in action
        )
        thrust_span = (
            self.action_calibration["thrust_span_up"]
            if collective >= 0.0
            else self.action_calibration["thrust_span_down"]
        )
        return {
            "thrust_normalized": _clamp(
                self.action_calibration["thrust_command_center"]
                + collective * thrust_span,
                0.0,
                1.0,
            ),
            "roll_rate_radps": roll
            * self.action_calibration["max_roll_rate_radps"],
            "pitch_rate_radps": pitch
            * self.action_calibration["max_pitch_rate_radps"],
            "yaw_rate_radps": yaw * self.action_calibration["max_yaw_rate_radps"],
        }


def run(
    policy_path: Path,
    run_id: str,
    duration_s: float,
    *,
    target_gates: int,
    control_rate_hz: float,
    shadow: bool,
    reset_first: bool,
    allow_gate_plane_miss: bool,
    thrust_multiplier: float,
    roll_rate_multiplier: float,
    pitch_rate_multiplier: float,
    yaw_rate_multiplier: float,
    use_sim_gate_normals: bool,
    sim_gate_normal_axis: str,
) -> dict[str, object]:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if control_rate_hz <= 0.0 or control_rate_hz >= 90.0:
        raise ValueError("control_rate_hz must be greater than 0 and less than 90")
    if target_gates < 0:
        raise ValueError("target_gates cannot be negative")
    if thrust_multiplier <= 0.0:
        raise ValueError("thrust_multiplier must be positive")
    for name, multiplier in (
        ("roll_rate_multiplier", roll_rate_multiplier),
        ("pitch_rate_multiplier", pitch_rate_multiplier),
        ("yaw_rate_multiplier", yaw_rate_multiplier),
    ):
        if multiplier <= 0.0:
            raise ValueError(f"{name} must be positive")
    if sim_gate_normal_axis not in {"x", "y", "neg-x", "neg-y"}:
        raise ValueError("sim_gate_normal_axis must be x, y, neg-x, or neg-y")
    if reset_first:
        _reset_simulator(1.0)

    policy = StructuredPolicy.load(policy_path)
    policy.verify_test_vectors()
    recorder = SessionRecorder(
        STACK_ROOT / "replay" / "sessions",
        run_id,
        {
            "created_at_epoch_s": time.time(),
            "mode": "structured_state_policy",
            "policy_path": str(policy_path),
            "policy_validation_status": policy.validation_status,
            "policy_observation_contract": policy.artifact["observation_contract"],
            "duration_s": duration_s,
            "target_gates": target_gates,
            "control_rate_hz": control_rate_hz,
            "shadow": shadow,
            "allow_gate_plane_miss": allow_gate_plane_miss,
            "thrust_multiplier": thrust_multiplier,
            "roll_rate_multiplier": roll_rate_multiplier,
            "pitch_rate_multiplier": pitch_rate_multiplier,
            "yaw_rate_multiplier": yaw_rate_multiplier,
            "use_sim_gate_normals": use_sim_gate_normals,
            "sim_gate_normal_axis": sim_gate_normal_axis,
        },
    )
    policy_log_path = recorder.layout.session_root / "policy.jsonl"
    if policy_log_path.exists():
        policy_log_path.unlink()

    state: dict[str, object] = {
        "latest_telemetry": None,
        "telemetry_count": 0,
        "collision_count": 0,
        "active_gate_index": 0,
        "max_active_gate_index": 0,
        "race_finished": False,
        "race_finish_time_ns": -1,
        "race_start_boot_time_ms": None,
        "sim_boot_time_ms": None,
        "sim_gate_normals_ned": {},
    }
    runtime = DclRuntime()
    runtime.configure(
        AdapterConfig(
            mission_name=run_id,
            transport_uri="udp://127.0.0.1:14550",
            vision_transport_uri=None,
            enable_vision=False,
            command_rate_hz=max(control_rate_hz + 5.0, 50.0),
        )
    )

    def on_event(event: RuntimeEvent) -> None:
        if event.event_type == "collision":
            state["collision_count"] = int(state["collision_count"]) + 1
        if event.event_type == "track.gate":
            gate_id = event.fields.get("gate_id")
            if isinstance(gate_id, int):
                normal = _normal_from_gate_orientation(
                    event.fields,
                    sim_gate_normal_axis,
                )
                if normal is not None:
                    dict(state["sim_gate_normals_ned"])[gate_id] = normal
        if event.event_type == "race.status":
            if isinstance(event.fields.get("active_gate_index"), int):
                active = int(event.fields["active_gate_index"])
                state["active_gate_index"] = active
                state["max_active_gate_index"] = max(
                    int(state["max_active_gate_index"]),
                    active,
                )
            if isinstance(event.fields.get("race_finish_time_ns"), int):
                state["race_finish_time_ns"] = int(
                    event.fields["race_finish_time_ns"]
                )
                state["race_finished"] = int(event.fields["race_finish_time_ns"]) >= 0
            if isinstance(event.fields.get("race_start_boot_time_ms"), int):
                state["race_start_boot_time_ms"] = int(
                    event.fields["race_start_boot_time_ms"]
                )
            if isinstance(event.fields.get("sim_boot_time_ms"), int):
                state["sim_boot_time_ms"] = int(event.fields["sim_boot_time_ms"])
        recorder.record_event(event)

    def on_telemetry(sample: TelemetrySample) -> None:
        state["latest_telemetry"] = sample
        state["telemetry_count"] = int(state["telemetry_count"]) + 1
        recorder.record_telemetry(sample)

    runtime.set_event_handler(on_event)
    runtime.set_telemetry_handler(on_telemetry)
    runtime.start()

    previous_action = (0.0, 0.0, 0.0, 0.0)
    initial_position: tuple[float, float, float] | None = None
    initial_attitude: tuple[float, float, float] | None = None
    pitch_reference_rad = 0.0
    command_count = 0
    policy_steps = 0
    abort_reason: str | None = None
    target_reached = False
    armed_by_runner = False

    try:
        preflight_deadline = time.monotonic() + 25.0
        stable_since: float | None = None
        while time.monotonic() < preflight_deadline:
            runtime.poll(max_telemetry_packets=100, max_vision_packets=0)
            sample = state["latest_telemetry"]
            if isinstance(sample, TelemetrySample):
                speed = _vector_norm(sample.velocity_mps)
                rates = _vector_norm(sample.angular_rate_radps)
                stable = (
                    not sample.is_armed
                    and speed <= 0.10
                    and rates <= 0.20
                    and _sample_has_structured_state(sample)
                    and _race_ready(state)
                )
                stable_since = (
                    stable_since
                    if stable and stable_since is not None
                    else time.monotonic() if stable else None
                )
                if stable_since is not None and time.monotonic() - stable_since >= 0.35:
                    break
            time.sleep(0.002)
        else:
            raise RuntimeError("preflight timed out waiting for stable race start")

        sample = state["latest_telemetry"]
        assert isinstance(sample, TelemetrySample)
        initial_position = _vector_tuple(sample.position_m)
        initial_attitude = _attitude_tuple(sample)
        pitch_reference_rad = initial_attitude[1] if initial_attitude else 0.0

        if not shadow:
            runtime.send_action(RuntimeAction.ARM)
            arm_deadline = time.monotonic() + 0.60
            next_arm_retry_s = time.monotonic() + 0.10
            while time.monotonic() < arm_deadline:
                runtime.poll(max_telemetry_packets=50, max_vision_packets=0)
                sample = state["latest_telemetry"]
                if isinstance(sample, TelemetrySample) and sample.is_armed:
                    armed_by_runner = True
                    break
                if time.monotonic() >= next_arm_retry_s:
                    runtime.send_action(RuntimeAction.ARM)
                    next_arm_retry_s = time.monotonic() + 0.10
                time.sleep(0.002)

        command_start_s = time.monotonic()
        next_policy_s = command_start_s
        next_command_s = command_start_s
        next_arm_retry_s = command_start_s + 0.10
        control_period_s = 1.0 / control_rate_hz
        latest_command: dict[str, float] | None = None
        while time.monotonic() - command_start_s < duration_s:
            runtime.poll(max_telemetry_packets=50, max_vision_packets=0)
            now_s = time.monotonic()
            sample = state["latest_telemetry"]
            if not isinstance(sample, TelemetrySample):
                continue
            if not shadow:
                if sample.is_armed:
                    armed_by_runner = True
                elif now_s >= next_arm_retry_s:
                    runtime.send_action(RuntimeAction.ARM)
                    next_arm_retry_s = now_s + 0.10

            abort_reason = _safety_abort(
                sample,
                initial_position,
                initial_attitude,
                int(state["collision_count"]),
            )
            if abort_reason is not None:
                break
            if bool(state["race_finished"]):
                target_reached = True
                break
            if (
                target_gates > 0
                and int(state["max_active_gate_index"]) >= target_gates
            ):
                target_reached = True
                break

            active_gate_index = _clamp_gate_index(
                int(state["active_gate_index"]),
                policy.gate_count,
            )
            gate_normals_ned = _runtime_gate_normals_ned(policy, state, use_sim_gate_normals)
            gate_metrics = gate_plane_metrics(
                policy,
                sample,
                active_gate_index,
                gate_normals_ned=gate_normals_ned,
            )
            if (
                not allow_gate_plane_miss
                and gate_metrics is not None
                and gate_metrics["signed_distance_m"] > 5.0
                and int(state["max_active_gate_index"]) <= active_gate_index
            ):
                abort_reason = f"gate_{active_gate_index}_plane_miss"
                break

            if now_s >= next_policy_s:
                observation = build_structured_observation(
                    policy,
                    sample,
                    active_gate_index,
                    previous_action,
                    pitch_reference_rad=pitch_reference_rad,
                    gate_normals_ned=gate_normals_ned,
                )
                if observation is None:
                    time.sleep(0.001)
                    continue
                action = policy.act(observation)
                raw_command = policy.map_action(action)
                command = dict(raw_command)
                command["thrust_normalized"] = _clamp(
                    command["thrust_normalized"] * thrust_multiplier,
                    0.0,
                    1.0,
                )
                command["roll_rate_radps"] *= roll_rate_multiplier
                command["pitch_rate_radps"] *= pitch_rate_multiplier
                command["yaw_rate_radps"] *= yaw_rate_multiplier
                previous_action = action
                latest_command = command
                policy_steps += 1
                next_policy_s = now_s + control_period_s
                _append_jsonl(
                    policy_log_path,
                    {
                        "monotonic_time_s": now_s,
                        "active_gate_index": active_gate_index,
                        "position_m": _telemetry_vector(sample.position_m),
                        "velocity_mps": _telemetry_vector(sample.velocity_mps),
                        "attitude_rad": _telemetry_attitude(sample),
                        "angular_rate_radps": _telemetry_vector(
                            sample.angular_rate_radps
                        ),
                        "observation": observation,
                        "observation_features": dict(
                            zip(policy.actor_features, observation)
                        ),
                        "normalized_action": dict(zip(ACTION_NAMES, action)),
                        "raw_mapped_command": raw_command,
                        "mapped_command": command,
                        "gate_plane": gate_metrics,
                        "use_sim_gate_normals": use_sim_gate_normals,
                        "sim_gate_normal_axis": sim_gate_normal_axis,
                        "shadow": shadow,
                    },
                )

            if not shadow and latest_command is not None and now_s >= next_command_s:
                command = ControlCommand(
                    monotonic_time_s=now_s,
                    roll_rate_radps=latest_command["roll_rate_radps"],
                    pitch_rate_radps=latest_command["pitch_rate_radps"],
                    yaw_rate_radps=latest_command["yaw_rate_radps"],
                    thrust_normalized=latest_command["thrust_normalized"],
                    source=CommandSource.CONTROLLER,
                )
                runtime.send_command(command)
                recorder.record_command(command, "attitude_rate")
                command_count += 1
                next_command_s = now_s + control_period_s
            time.sleep(0.001)
    except Exception as exc:
        abort_reason = str(exc)
    finally:
        if not shadow:
            try:
                for _ in range(3):
                    now_s = time.monotonic()
                    stop = ControlCommand(
                        monotonic_time_s=now_s,
                        roll_rate_radps=0.0,
                        pitch_rate_radps=0.0,
                        yaw_rate_radps=0.0,
                        thrust_normalized=0.0,
                        source=CommandSource.FAILSAFE,
                    )
                    runtime.send_command(stop)
                    recorder.record_command(stop, "attitude_rate")
                    time.sleep(0.025)
                for _ in range(10):
                    runtime.send_action(RuntimeAction.DISARM)
                    runtime.poll(max_telemetry_packets=50, max_vision_packets=0)
                    time.sleep(0.03)
                runtime.send_action(RuntimeAction.SIM_RESET)
                reset_deadline = time.monotonic() + 1.5
                while time.monotonic() < reset_deadline:
                    runtime.send_action(RuntimeAction.DISARM)
                    runtime.poll(max_telemetry_packets=50, max_vision_packets=0)
                    time.sleep(0.03)
            finally:
                runtime.stop()
        else:
            runtime.stop()

    summary = SessionEvaluator(recorder.layout).summarize(abort_reason=abort_reason)
    summary.update(
        {
            "policy_steps": policy_steps,
            "policy_command_count": command_count,
            "gate0_passed": int(state["max_active_gate_index"]) >= 1,
            "target_gates": target_gates,
            "target_reached": target_reached,
            "race_finished": bool(state["race_finished"]),
            "race_finish_time_ns": int(state["race_finish_time_ns"]),
            "max_active_gate_index": int(state["max_active_gate_index"]),
            "final_active_gate_index": int(state["active_gate_index"]),
            "policy_validation_status": policy.validation_status,
            "policy_observation_contract": policy.artifact["observation_contract"],
            "shadow": shadow,
            "armed_by_runner": armed_by_runner,
            "thrust_multiplier": thrust_multiplier,
            "roll_rate_multiplier": roll_rate_multiplier,
            "pitch_rate_multiplier": pitch_rate_multiplier,
            "yaw_rate_multiplier": yaw_rate_multiplier,
            "use_sim_gate_normals": use_sim_gate_normals,
            "sim_gate_normal_axis": sim_gate_normal_axis,
            "session_path": str(recorder.layout.session_root),
        }
    )
    recorder.write_summary(summary)
    return summary


def build_structured_observation(
    policy: StructuredPolicy,
    sample: TelemetrySample,
    active_gate_index: int,
    previous_action: Sequence[float],
    *,
    pitch_reference_rad: float = 0.0,
    gate_normals_ned: Sequence[Sequence[float]] | None = None,
) -> list[float] | None:
    if not _sample_has_structured_state(sample):
        return None
    if len(previous_action) != len(ACTION_NAMES):
        raise ValueError("previous_action must have four values")
    assert sample.position_m is not None
    assert sample.velocity_mps is not None
    assert sample.attitude_rad is not None
    assert sample.angular_rate_radps is not None

    gate_index = _clamp_gate_index(active_gate_index, policy.gate_count)
    next_gate_index = min(gate_index + 1, policy.gate_count - 1)
    rotation_body_frd_to_ned = _rotation_matrix(
        sample.attitude_rad.roll_rad,
        -(sample.attitude_rad.pitch_rad - pitch_reference_rad),
        sample.attitude_rad.yaw_rad,
    )

    position_ned = _vector_tuple(sample.position_m)
    velocity_ned = _vector_tuple(sample.velocity_mps)
    assert position_ned is not None and velocity_ned is not None

    def body_flu_from_ned(vector_ned: Sequence[float]) -> tuple[float, float, float]:
        vector_frd = _transpose_matvec(
            rotation_body_frd_to_ned,
            tuple(float(value) for value in vector_ned),
        )
        return vector_frd[0], -vector_frd[1], -vector_frd[2]

    def relative_body(target_ned: Sequence[float]) -> tuple[float, float, float]:
        return body_flu_from_ned(
            (
                float(target_ned[0]) - position_ned[0],
                float(target_ned[1]) - position_ned[1],
                float(target_ned[2]) - position_ned[2],
            )
        )

    active_gate_body = relative_body(policy.gates_ned[gate_index])
    next_gate_body = relative_body(policy.gates_ned[next_gate_index])
    active_normal_ned = (
        gate_normals_ned[gate_index]
        if gate_normals_ned is not None
        else _surrogate_vector_to_ned(policy.gate_normals_surrogate[gate_index])
    )
    next_normal_ned = (
        gate_normals_ned[next_gate_index]
        if gate_normals_ned is not None
        else _surrogate_vector_to_ned(policy.gate_normals_surrogate[next_gate_index])
    )
    active_normal_body = body_flu_from_ned(active_normal_ned)
    next_normal_body = body_flu_from_ned(next_normal_ned)
    body_velocity = body_flu_from_ned(velocity_ned)
    gravity_body = body_flu_from_ned((0.0, 0.0, 1.0))
    angular_rate = (
        float(sample.angular_rate_radps.x),
        -float(sample.angular_rate_radps.y),
        float(sample.angular_rate_radps.z),
    )
    observation = [
        *(value / 30.0 for value in active_gate_body),
        *active_normal_body,
        *(value / 30.0 for value in next_gate_body),
        *next_normal_body,
        *(value / VELOCITY_SCALE_MPS for value in body_velocity),
        *gravity_body,
        *(value / scale for value, scale in zip(angular_rate, RATE_SCALES_RADPS)),
        *(_clamp(float(value), -1.0, 1.0) for value in previous_action),
        gate_index / max(policy.gate_count - 1, 1),
    ]
    if len(observation) != len(STRUCTURED_TEACHER_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in observation
    ):
        raise ValueError("invalid structured_teacher_v2 observation")
    return observation


def gate_plane_metrics(
    policy: StructuredPolicy,
    sample: TelemetrySample,
    active_gate_index: int,
    *,
    gate_normals_ned: Sequence[Sequence[float]] | None = None,
) -> dict[str, float] | None:
    if sample.position_m is None:
        return None
    gate_index = _clamp_gate_index(active_gate_index, policy.gate_count)
    position = _ned_position_to_surrogate(
        _vector_tuple(sample.position_m),
        altitude_offset_m=policy.surrogate_altitude_offset_m,
    )
    center = policy.gates_surrogate[gate_index]
    normal = (
        _ned_vector_to_surrogate(gate_normals_ned[gate_index])
        if gate_normals_ned is not None
        else policy.gate_normals_surrogate[gate_index]
    )
    relative = tuple(position[index] - center[index] for index in range(3))
    signed_distance = _dot(relative, normal)
    lateral_axis = (-normal[1], normal[0], 0.0)
    lateral_norm = math.sqrt(_dot(lateral_axis, lateral_axis))
    if lateral_norm <= 1e-9:
        lateral_axis = (0.0, 1.0, 0.0)
    else:
        lateral_axis = tuple(value / lateral_norm for value in lateral_axis)
    lateral_offset = _dot(relative, lateral_axis)
    vertical_offset = relative[2]
    half_width = policy.gate_width_m / 2.0
    half_height = policy.gate_height_m / 2.0
    rectangular_margin = min(
        half_width - abs(lateral_offset),
        half_height - abs(vertical_offset),
    )
    return {
        "signed_distance_m": signed_distance,
        "lateral_offset_m": lateral_offset,
        "vertical_offset_m": vertical_offset,
        "rectangular_margin_m": rectangular_margin,
    }


def _validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported policy artifact schema")
    if artifact.get("policy_role") != "structured_state_sim_teacher":
        raise ValueError("artifact is not a structured-state simulator teacher")
    if artifact.get("policy_architecture") != "mlp":
        raise ValueError("artifact is not an MLP policy")
    if artifact.get("observation_contract") != "structured_teacher_v2":
        raise ValueError("artifact observation contract is not structured_teacher_v2")
    actor_features = tuple(artifact.get("actor_features", ()))
    if actor_features != STRUCTURED_TEACHER_FEATURE_NAMES:
        raise ValueError("artifact actor feature order does not match")
    if int(artifact.get("actor_observation_dim", -1)) != len(actor_features):
        raise ValueError("artifact actor observation dimension does not match")
    if tuple(artifact.get("action_names", ())) != ACTION_NAMES:
        raise ValueError("artifact action names do not match")
    if artifact.get("hidden_activation") != {
        "name": "leaky_relu",
        "negative_slope": 0.2,
    }:
        raise ValueError("artifact hidden activation does not match")
    if artifact.get("output_activation") != "tanh":
        raise ValueError("artifact output activation does not match")
    calibration = artifact.get("action_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("artifact action calibration is missing")
    for key in (
        "thrust_command_center",
        "thrust_span_up",
        "thrust_span_down",
        "max_roll_rate_radps",
        "max_pitch_rate_radps",
        "max_yaw_rate_radps",
    ):
        if key not in calibration:
            raise ValueError(f"artifact action calibration is missing {key}")
    track = artifact.get("track")
    if not isinstance(track, dict) or int(track.get("gate_count", 0)) < 1:
        raise ValueError("artifact track payload is missing")
    if len(track.get("gates_ned", ())) != int(track["gate_count"]):
        raise ValueError("artifact NED gate count does not match")
    if len(track.get("gates_surrogate_flu", ())) != int(track["gate_count"]):
        raise ValueError("artifact surrogate gate count does not match")
    layers = artifact.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("artifact has no MLP layers")
    input_width = len(actor_features)
    for layer in layers:
        weights = layer.get("weight")
        biases = layer.get("bias")
        if not isinstance(weights, list) or not isinstance(biases, list):
            raise ValueError("artifact layer is malformed")
        if len(weights) != len(biases) or any(
            len(row) != input_width for row in weights
        ):
            raise ValueError("artifact layer dimensions do not match")
        input_width = len(biases)
    if input_width != len(ACTION_NAMES):
        raise ValueError("artifact output dimension does not match")


def _sample_has_structured_state(sample: TelemetrySample) -> bool:
    return (
        sample.position_m is not None
        and sample.velocity_mps is not None
        and sample.attitude_rad is not None
        and sample.angular_rate_radps is not None
    )


def _race_ready(state: dict[str, object]) -> bool:
    race_start = state["race_start_boot_time_ms"]
    sim_boot = state["sim_boot_time_ms"]
    return (
        isinstance(race_start, int)
        and isinstance(sim_boot, int)
        and race_start >= 0
        and sim_boot >= race_start
    )


def _safety_abort(
    sample: TelemetrySample,
    initial_position: tuple[float, float, float] | None,
    initial_attitude: tuple[float, float, float] | None,
    collision_count: int,
) -> str | None:
    if collision_count > 0:
        return "collision_abort"
    position = _vector_tuple(sample.position_m)
    velocity = _vector_tuple(sample.velocity_mps)
    attitude = _attitude_tuple(sample)
    if position is None or velocity is None or attitude is None:
        return "telemetry_incomplete"
    if initial_position is not None and initial_position[2] - position[2] > 5.0:
        return "altitude_abort"
    if velocity[2] < -12.0:
        return "upward_speed_abort"
    if math.sqrt(sum(value * value for value in velocity)) > 35.0:
        return "speed_abort"
    if initial_attitude is not None:
        tilt_change = math.sqrt(
            _angle_delta(attitude[0], initial_attitude[0]) ** 2
            + _angle_delta(attitude[1], initial_attitude[1]) ** 2
        )
        if tilt_change > 1.0:
            return "tilt_abort"
    return None


def _vector_tuple(value: object) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return (float(value.x), float(value.y), float(value.z))


def _attitude_tuple(sample: TelemetrySample) -> tuple[float, float, float] | None:
    if sample.attitude_rad is None:
        return None
    return (
        float(sample.attitude_rad.roll_rad),
        float(sample.attitude_rad.pitch_rad),
        float(sample.attitude_rad.yaw_rad),
    )


def _vector_norm(value: object) -> float:
    vector = _vector_tuple(value)
    if vector is None:
        return math.inf
    return math.sqrt(sum(component * component for component in vector))


def _angle_delta(value: float, reference: float) -> float:
    return (value - reference + math.pi) % (2.0 * math.pi) - math.pi


def _surrogate_vector_to_ned(
    vector: Sequence[float],
) -> tuple[float, float, float]:
    return -float(vector[0]), float(vector[1]), -float(vector[2])


def _ned_vector_to_surrogate(
    vector: Sequence[float],
) -> tuple[float, float, float]:
    return -float(vector[0]), float(vector[1]), -float(vector[2])


def _runtime_gate_normals_ned(
    policy: StructuredPolicy,
    state: dict[str, object],
    use_sim_gate_normals: bool,
) -> tuple[tuple[float, float, float], ...] | None:
    if not use_sim_gate_normals:
        return None
    normals = state["sim_gate_normals_ned"]
    if not isinstance(normals, dict):
        return None
    if any(index not in normals for index in range(policy.gate_count)):
        return None
    return tuple(
        tuple(float(value) for value in normals[index])
        for index in range(policy.gate_count)
    )


def _normal_from_gate_orientation(
    fields: dict[str, object],
    axis: str,
) -> tuple[float, float, float] | None:
    keys = (
        "orientation_ned_w",
        "orientation_ned_x",
        "orientation_ned_y",
        "orientation_ned_z",
    )
    if any(key not in fields for key in keys):
        return None
    w, x, y, z = (float(fields[key]) for key in keys)
    local = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "neg-x": (-1.0, 0.0, 0.0),
        "neg-y": (0.0, -1.0, 0.0),
    }[axis]
    vector = _quat_rotate((w, x, y, z), local)
    norm = math.sqrt(_dot(vector, vector))
    if norm <= 1e-9:
        return None
    return tuple(value / norm for value in vector)


def _quat_rotate(
    quat_wxyz: Sequence[float],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quat_wxyz)
    vx, vy, vz = (float(value) for value in vector)
    # q * v * q^-1, expanded for a unit quaternion.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _ned_position_to_surrogate(
    position_ned: tuple[float, float, float] | None,
    *,
    altitude_offset_m: float,
) -> tuple[float, float, float]:
    if position_ned is None:
        raise ValueError("position_ned is required")
    north, east, down = position_ned
    return -north, east, altitude_offset_m - down


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(first, second))


def _clamp_gate_index(gate_index: int, gate_count: int) -> int:
    return max(0, min(int(gate_index), gate_count - 1))


def _telemetry_vector(value: object) -> dict[str, float] | None:
    vector = _vector_tuple(value)
    if vector is None:
        return None
    return {"x": vector[0], "y": vector[1], "z": vector[2]}


def _telemetry_attitude(sample: TelemetrySample) -> dict[str, float] | None:
    attitude = _attitude_tuple(sample)
    if attitude is None:
        return None
    return {
        "roll_rad": attitude[0],
        "pitch_rad": attitude[1],
        "yaw_rad": attitude[2],
    }


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=(
            LAB_ROOT
            / "exports"
            / "ai_gp"
            / "ai_gp_040_near_gate_teacher_structured_policy.json"
        ),
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--target-gates",
        type=int,
        default=1,
        help="stop after this many gates; use 0 to run until duration/race finish",
    )
    parser.add_argument("--control-rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="compute observations/actions and logs without arming or commanding",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="do not reset the simulator before the run",
    )
    parser.add_argument(
        "--allow-gate-plane-miss",
        action="store_true",
        help="continue after crossing an uncompleted active-gate plane",
    )
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="repeat attempts until interrupted and stream one JSON summary per attempt",
    )
    parser.add_argument(
        "--attempt-delay-s",
        type=float,
        default=0.25,
        help="pause between repeated attempts",
    )
    parser.add_argument(
        "--thrust-multiplier",
        type=float,
        default=1.0,
        help="multiply mapped thrust before sending commands",
    )
    parser.add_argument(
        "--roll-rate-multiplier",
        type=float,
        default=1.0,
        help="multiply mapped roll-rate commands before sending",
    )
    parser.add_argument(
        "--pitch-rate-multiplier",
        type=float,
        default=1.0,
        help="multiply mapped pitch-rate commands before sending",
    )
    parser.add_argument(
        "--yaw-rate-multiplier",
        type=float,
        default=1.0,
        help="multiply mapped yaw-rate commands before sending",
    )
    parser.add_argument(
        "--use-sim-gate-normals",
        action="store_true",
        help="use simulator track.gate quaternion normals instead of exported normals",
    )
    parser.add_argument(
        "--sim-gate-normal-axis",
        choices=("x", "y", "neg-x", "neg-y"),
        default="y",
        help="local gate axis to rotate by simulator gate quaternion",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.attempts < 1 and not args.continuous:
        raise ValueError("--attempts must be positive")
    if args.attempt_delay_s < 0.0:
        raise ValueError("--attempt-delay-s cannot be negative")
    if args.thrust_multiplier <= 0.0:
        raise ValueError("--thrust-multiplier must be positive")
    for arg_name in (
        "roll_rate_multiplier",
        "pitch_rate_multiplier",
        "yaw_rate_multiplier",
    ):
        if getattr(args, arg_name) <= 0.0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be positive")
    run_id = args.run_id or (
        "structured_040_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    summaries = []
    attempt_index = 0
    while args.continuous or attempt_index < args.attempts:
        attempt_index += 1
        if args.continuous:
            attempt_run_id = f"{run_id}_{attempt_index:04d}"
        else:
            attempt_run_id = (
                run_id if args.attempts == 1 else f"{run_id}_{attempt_index:02d}"
            )
        summary = run(
            args.policy,
            attempt_run_id,
            args.duration,
            target_gates=args.target_gates,
            control_rate_hz=args.control_rate_hz,
            shadow=args.shadow,
            reset_first=not args.no_reset,
            allow_gate_plane_miss=args.allow_gate_plane_miss,
            thrust_multiplier=args.thrust_multiplier,
            roll_rate_multiplier=args.roll_rate_multiplier,
            pitch_rate_multiplier=args.pitch_rate_multiplier,
            yaw_rate_multiplier=args.yaw_rate_multiplier,
            use_sim_gate_normals=args.use_sim_gate_normals,
            sim_gate_normal_axis=args.sim_gate_normal_axis,
        )
        if args.continuous:
            print(
                json.dumps(
                    {
                        "attempt": attempt_index,
                        "run_id": attempt_run_id,
                        "summary": summary,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            summaries.append(summary)
        if args.continuous and args.attempt_delay_s > 0.0:
            time.sleep(args.attempt_delay_s)
    if args.continuous:
        return
    output = (
        summaries[0]
        if len(summaries) == 1
        else {
            "attempt_count": len(summaries),
            "target_reached_count": sum(
                bool(summary["target_reached"]) for summary in summaries
            ),
            "gate0_pass_count": sum(
                bool(summary["gate0_passed"]) for summary in summaries
            ),
            "race_finish_count": sum(
                bool(summary["race_finished"]) for summary in summaries
            ),
            "max_gate_index": max(
                int(summary["max_active_gate_index"]) for summary in summaries
            ),
            "collision_attempt_count": sum(
                int(summary["collision_count"]) > 0 for summary in summaries
            ),
            "attempts": summaries,
        }
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
