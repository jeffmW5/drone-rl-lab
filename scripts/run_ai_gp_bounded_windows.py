from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = LAB_ROOT.parent
STACK_ROOT = Path(
    os.environ.get(
        "AI_GP_RUNTIME_ROOT",
        LAB_ROOT / "tmp" / "ai-grand-prix-stack-remote",
    )
).expanduser().resolve()
EXECUTION_ROOT = SHARED_ROOT / "ai-grand-prix-stack"
if not (STACK_ROOT / "adapter" / "vision_udp.py").exists():
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
from adapter.vision_udp import VisionUdpClient, VisionUdpConfig
from ai_gp_rl.contract import (
    ActionCalibration,
    TemporalLivePolicyFeatures,
    build_motion_observation,
    build_temporal_base_observation,
)
from ai_gp_rl.session_dataset import _telemetry_body_features
from ai_gp_rl.session_dataset import _rotation_matrix, _transpose_matvec
from calibration.run_thrust_sweep import _reset_simulator
from perception.gates import GateObservation, NormalizedBox, NormalizedPoint
if str(EXECUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXECUTION_ROOT))
from policy.mlp_policy import (
    ACTION_NAMES,
    PolicyObservationStack,
    apply_action_governor,
    load_policy,
)
from replay.recording import SessionEvaluator, SessionRecorder


KNOWN_TRACK_GATES_NED = {
    0: (-23.2979679107666, -0.39990234375, -0.03195800632238388, 2.72, 2.72),
    1: (-46.89374923706055, -2.499990224838257, 5.068041801452637, 2.72, 2.72),
    2: (-74.59375, 1.2000097036361694, 13.668041229248047, 2.72, 2.72),
    3: (-111.49374389648438, -5.099989891052246, 24.56804084777832, 2.72, 2.72),
    4: (-135.49374389648438, -0.7999902367591858, 25.355653762817383, 2.72, 2.72),
    5: (-159.19374084472656, -4.399990081787109, 25.968040466308594, 2.72, 2.72),
}
MEASURED_ROLL_ACTION_SIGN = 1.0


def run(
    policy_path: Path,
    run_id: str,
    duration_s: float,
    target_gates: int = 1,
    control_rate_hz: float = 12.5,
    lateral_authority_scale: float = 1.0,
    pitch_authority_scale: float = 1.0,
    governor_slew_scale: float = 1.0,
    gate_source: str = "vision",
    post_control_rate_hz: float = 12.5,
    post_max_roll_rate_radps: float = 0.005,
    post_max_pitch_rate_radps: float = 0.07,
    post_max_yaw_rate_radps: float = 0.01,
    post_thrust_span_up: float = 0.008,
    post_thrust_span_down: float = 0.008,
    uniform_authority: bool = False,
    launch_phase_s: float | None = None,
    authority_ramp_s: float = 0.0,
    thrust_ungoverned_fraction: float = 0.0,
    direction_ungoverned_fraction: float = 0.0,
    post_thrust_gain: float = 1.0,
    allow_gate_plane_miss: bool = False,
) -> dict[str, object]:
    if min(
        control_rate_hz,
        post_control_rate_hz,
        lateral_authority_scale,
        pitch_authority_scale,
        governor_slew_scale,
        post_max_roll_rate_radps,
        post_max_pitch_rate_radps,
        post_max_yaw_rate_radps,
        post_thrust_span_up,
        post_thrust_span_down,
    ) <= 0.0:
        raise ValueError("control and authority scales must be positive")
    if gate_source not in {"vision", "track_pose"}:
        raise ValueError("gate_source must be vision or track_pose")
    if launch_phase_s is not None and launch_phase_s < 0.0:
        raise ValueError("launch_phase_s cannot be negative")
    if authority_ramp_s < 0.0:
        raise ValueError("authority_ramp_s cannot be negative")
    if not 0.0 <= thrust_ungoverned_fraction <= 1.0:
        raise ValueError("thrust_ungoverned_fraction must be within [0, 1]")
    if not 0.0 <= direction_ungoverned_fraction <= 1.0:
        raise ValueError("direction_ungoverned_fraction must be within [0, 1]")
    if post_thrust_gain <= 0.0:
        raise ValueError("post_thrust_gain must be positive")
    _reset_simulator(1.0)

    policy = load_policy(policy_path)
    policy.verify_test_vectors()
    if policy.observation_contract != "motion_live_v1":
        raise ValueError("bounded runner currently requires motion_live_v1")

    launch_calibration = ActionCalibration(
        hover_thrust=0.295,
        thrust_span_up=0.008,
        thrust_span_down=0.008,
        max_roll_rate_radps=0.005,
        max_pitch_rate_radps=0.07,
        max_yaw_rate_radps=0.01,
    )
    post_gate_calibration = ActionCalibration(
        hover_thrust=_lerp(
            0.295,
            0.5,
            thrust_ungoverned_fraction,
        ),
        thrust_span_up=_lerp(
            post_thrust_span_up,
            0.5,
            thrust_ungoverned_fraction,
        ),
        thrust_span_down=_lerp(
            post_thrust_span_down,
            0.5,
            thrust_ungoverned_fraction,
        ),
        max_roll_rate_radps=_lerp(
            post_max_roll_rate_radps,
            3.0,
            direction_ungoverned_fraction,
        ),
        max_pitch_rate_radps=_lerp(
            post_max_pitch_rate_radps,
            3.0,
            direction_ungoverned_fraction,
        ),
        max_yaw_rate_radps=_lerp(
            post_max_yaw_rate_radps,
            2.0,
            direction_ungoverned_fraction,
        ),
    )
    if uniform_authority:
        launch_calibration = post_gate_calibration
    launch_governor = policy.action_governor
    post_gate_governor = launch_governor
    if post_gate_governor is not None:
        post_gate_governor = {
            **post_gate_governor,
            "slew_limits": [
                min(1.0, limit * governor_slew_scale)
                for limit in post_gate_governor["slew_limits"]
            ],
        }
        post_gate_governor["slew_limits"][0] = _lerp(
            float(post_gate_governor["slew_limits"][0]),
            1.0,
            thrust_ungoverned_fraction,
        )
        for index in (1, 2, 3):
            post_gate_governor["slew_limits"][index] = _lerp(
                float(post_gate_governor["slew_limits"][index]),
                1.0,
                direction_ungoverned_fraction,
            )
        post_gate_governor["upward_brake_gain"] = _lerp(
            float(post_gate_governor["upward_brake_gain"]),
            0.0,
            thrust_ungoverned_fraction,
        )
    if uniform_authority:
        launch_governor = post_gate_governor
    recorder = SessionRecorder(
        STACK_ROOT / "replay" / "sessions",
        run_id,
        {
            "created_at_epoch_s": time.time(),
            "mode": "bounded_learned_policy",
            "policy_path": str(policy_path),
            "policy_validation_status": policy.status.validation_status,
            "policy_observation_contract": policy.observation_contract,
            "launch_calibration": asdict(launch_calibration),
            "post_gate_calibration": asdict(post_gate_calibration),
            "duration_s": duration_s,
            "target_gates": target_gates,
            "control_rate_hz": control_rate_hz,
            "lateral_authority_scale": lateral_authority_scale,
            "pitch_authority_scale": pitch_authority_scale,
            "governor_slew_scale": governor_slew_scale,
            "gate_source": gate_source,
            "post_control_rate_hz": post_control_rate_hz,
            "roll_action_sign": MEASURED_ROLL_ACTION_SIGN,
            "roll_action_sign_basis": "symmetric_live_pulses_20260614",
            "uniform_authority": uniform_authority,
            "launch_phase_s": launch_phase_s,
            "authority_ramp_s": authority_ramp_s,
            "thrust_ungoverned_fraction": thrust_ungoverned_fraction,
            "direction_ungoverned_fraction": direction_ungoverned_fraction,
            "post_thrust_gain": post_thrust_gain,
            "allow_gate_plane_miss": allow_gate_plane_miss,
        },
    )
    policy_log_path = recorder.layout.session_root / "policy.jsonl"
    if policy_log_path.exists():
        policy_log_path.unlink()

    state: dict[str, object] = {
        "latest_telemetry": None,
        "latest_detection": None,
        "latest_detection_time_s": None,
        "latest_frame_id": None,
        "new_detection": False,
        "telemetry_count": 0,
        "vision_count": 0,
        "detection_count": 0,
        "collision_count": 0,
        "active_gate_index": 0,
        "max_active_gate_index": 0,
        "race_finished": False,
        "race_finish_time_ns": -1,
        "track_gate_count": None,
        "track_gates": dict(KNOWN_TRACK_GATES_NED),
        "race_start_boot_time_ms": None,
        "sim_boot_time_ms": None,
    }
    vision_clients: list[VisionUdpClient] = []

    def vision_factory(config: VisionUdpConfig) -> VisionUdpClient:
        client = VisionUdpClient(config)
        vision_clients.append(client)
        return client

    runtime = DclRuntime(vision_factory=vision_factory)
    runtime.configure(
        AdapterConfig(
            mission_name=run_id,
            transport_uri="udp://127.0.0.1:14550",
            vision_transport_uri="udp://0.0.0.0:5600",
            command_rate_hz=50.0,
        )
    )
    seen_frames: set[str] = set()

    def on_event(event: RuntimeEvent) -> None:
        if event.event_type == "collision":
            state["collision_count"] = int(state["collision_count"]) + 1
        if event.event_type == "race.status":
            if isinstance(event.fields.get("active_gate_index"), int):
                state["active_gate_index"] = int(event.fields["active_gate_index"])
                state["max_active_gate_index"] = max(
                    int(state["max_active_gate_index"]),
                    int(event.fields["active_gate_index"]),
                )
            if isinstance(event.fields.get("race_finish_time_ns"), int):
                state["race_finish_time_ns"] = int(
                    event.fields["race_finish_time_ns"]
                )
                if int(event.fields["race_finish_time_ns"]) >= 0:
                    state["race_finished"] = True
            if isinstance(event.fields.get("race_start_boot_time_ms"), int):
                state["race_start_boot_time_ms"] = int(
                    event.fields["race_start_boot_time_ms"]
                )
            if isinstance(event.fields.get("sim_boot_time_ms"), int):
                state["sim_boot_time_ms"] = int(event.fields["sim_boot_time_ms"])
        if event.event_type == "track.info":
            if isinstance(event.fields.get("gate_count"), int):
                state["track_gate_count"] = int(event.fields["gate_count"])
        if event.event_type == "track.gate":
            gate_id = event.fields.get("gate_id")
            gate_values = (
                event.fields.get("position_ned_x"),
                event.fields.get("position_ned_y"),
                event.fields.get("position_ned_z"),
                event.fields.get("width_m"),
                event.fields.get("height_m"),
            )
            if isinstance(gate_id, int) and all(
                isinstance(value, (int, float)) for value in gate_values
            ):
                state["track_gates"][gate_id] = tuple(
                    float(value) for value in gate_values
                )
        recorder.record_event(event)

    def on_telemetry(sample: TelemetrySample) -> None:
        state["latest_telemetry"] = sample
        state["telemetry_count"] = int(state["telemetry_count"]) + 1
        recorder.record_telemetry(sample)

    def on_vision(frame) -> None:
        if frame.frame_id in seen_frames:
            return
        seen_frames.add(frame.frame_id)
        state["vision_count"] = int(state["vision_count"]) + 1
        frame_bytes = (
            vision_clients[0].frame_store.get(frame.storage_ref)
            if vision_clients
            else None
        )
        recorder.record_vision(frame, frame_bytes=frame_bytes)
        detections = _fast_gate_detections(frame, frame_bytes)
        recorder.record_detection(
            monotonic_time_s=frame.monotonic_time_s,
            frame_id=frame.frame_id,
            detections=detections,
        )
        if detections:
            state["latest_detection"] = detections[0]
            state["latest_detection_time_s"] = frame.monotonic_time_s
            state["latest_frame_id"] = frame.frame_id
            state["new_detection"] = True
            state["detection_count"] = int(state["detection_count"]) + 1

    runtime.set_event_handler(on_event)
    runtime.set_telemetry_handler(on_telemetry)
    runtime.set_vision_handler(on_vision)
    runtime.start()

    observation_stack = PolicyObservationStack(policy)
    previous_action = (0.0, 0.0, 0.0, 0.0)
    previous_temporal: list[float] | None = None
    policy_gate_index = 0
    latest_action: tuple[float, ...] | None = None
    initial_position: tuple[float, float, float] | None = None
    initial_attitude: tuple[float, float, float] | None = None
    command_count = 0
    policy_steps = 0
    abort_reason: str | None = None
    success = False
    armed_by_runner = False

    try:
        preflight_deadline = time.monotonic() + 25.0
        stable_since: float | None = None
        while time.monotonic() < preflight_deadline:
            runtime.poll(max_telemetry_packets=100, max_vision_packets=100)
            sample = state["latest_telemetry"]
            if isinstance(sample, TelemetrySample):
                speed = _vector_norm(sample.velocity_mps)
                rates = _vector_norm(sample.angular_rate_radps)
                race_ready = _race_ready(state)
                stable = (
                    not sample.is_armed
                    and speed <= 0.10
                    and rates <= 0.20
                    and state["latest_detection"] is not None
                )
                stable_since = (
                    stable_since
                    if stable and stable_since is not None
                    else time.monotonic() if stable else None
                )
                if (
                    stable_since is not None
                    and time.monotonic() - stable_since >= 0.35
                    and race_ready
                ):
                    break
            time.sleep(0.002)
        else:
            raise RuntimeError("preflight timed out waiting for stable race start")

        sample = state["latest_telemetry"]
        assert isinstance(sample, TelemetrySample)
        initial_position = _vector_tuple(sample.position_m)
        initial_attitude = _attitude_tuple(sample)
        runtime.send_action(RuntimeAction.ARM)
        arm_request_s = time.monotonic()
        next_arm_retry_s = arm_request_s + 0.10
        while time.monotonic() - arm_request_s < 0.35:
            runtime.poll(max_telemetry_packets=50, max_vision_packets=500)
            sample = state["latest_telemetry"]
            if isinstance(sample, TelemetrySample) and sample.is_armed:
                armed_by_runner = True
                break
            if time.monotonic() >= next_arm_retry_s:
                runtime.send_action(RuntimeAction.ARM)
                next_arm_retry_s = time.monotonic() + 0.10
            time.sleep(0.002)

        command_start_s = time.monotonic()
        next_command_s = command_start_s
        next_policy_s = command_start_s
        next_arm_retry_s = command_start_s
        while time.monotonic() - command_start_s < duration_s:
            runtime.poll(max_telemetry_packets=50, max_vision_packets=500)
            now_s = time.monotonic()
            sample = state["latest_telemetry"]
            if not isinstance(sample, TelemetrySample):
                continue
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
                success = True
                break
            if (
                target_gates > 0
                and int(state["max_active_gate_index"]) >= target_gates
            ):
                success = True
                break
            active_gate_index = int(state["active_gate_index"])
            authority_blend = _authority_blend(
                elapsed_s=now_s - command_start_s,
                active_gate_index=active_gate_index,
                uniform_authority=uniform_authority,
                launch_phase_s=launch_phase_s,
                authority_ramp_s=authority_ramp_s,
            )
            active_gate = state["track_gates"].get(active_gate_index)
            if (
                not allow_gate_plane_miss
                and gate_source == "track_pose"
                and active_gate is not None
                and sample.position_m is not None
                and sample.position_m.x < active_gate[0] - 5.0
            ):
                abort_reason = f"gate_{active_gate_index}_plane_miss"
                break
            active_control_rate_hz = _lerp(
                control_rate_hz,
                post_control_rate_hz,
                authority_blend,
            )
            control_period_s = 1.0 / active_control_rate_hz

            if (
                gate_source == "track_pose" or state["new_detection"]
            ) and now_s >= next_policy_s:
                state["new_detection"] = False
                active_gate_index = int(state["active_gate_index"])
                if active_gate_index != policy_gate_index:
                    previous_temporal = None
                    policy_gate_index = active_gate_index
                if gate_source == "track_pose":
                    gate_features = _project_active_gate(
                        sample,
                        state["track_gates"].get(active_gate_index),
                        initial_attitude[1] if initial_attitude else 0.0,
                    )
                    if gate_features is None:
                        time.sleep(0.001)
                        continue
                    gate_center, gate_size, gate_confidence = gate_features
                    gate_age_s = 0.0
                else:
                    detection = state["latest_detection"]
                    assert detection is not None
                    gate_center = (
                        (detection.bbox.center.x - 0.5) * 2.0,
                        -(detection.bbox.center.y - 0.5) * 2.0,
                    )
                    gate_size = (
                        detection.bbox.width,
                        detection.bbox.height,
                    )
                    gate_confidence = detection.confidence
                    gate_age_s = max(
                        0.0,
                        now_s - float(state["latest_detection_time_s"]),
                    )
                body_velocity, gravity_body = _body_features(
                    sample,
                    initial_attitude[1] if initial_attitude else 0.0,
                )
                angular_rate = _vector_tuple(sample.angular_rate_radps) or (
                    0.0,
                    0.0,
                    0.0,
                )
                base = build_temporal_base_observation(
                    TemporalLivePolicyFeatures(
                        body_velocity_mps=body_velocity,
                        gravity_body=gravity_body,
                        angular_rate_radps=(
                            angular_rate[0],
                            -angular_rate[1],
                            angular_rate[2],
                        ),
                        gate_center_normalized=gate_center,
                        gate_size_normalized=gate_size,
                        gate_area_normalized=(
                            gate_size[0] * gate_size[1]
                        ),
                        gate_confidence=gate_confidence,
                        gate_age_s=gate_age_s,
                        previous_action=previous_action,
                    )
                )
                motion = build_motion_observation(
                    base, previous_temporal or base
                )
                previous_temporal = base
                observation = observation_stack.build(motion, previous_action)
                requested_action = policy.act(observation)
                latest_action = apply_action_governor(
                    requested_action,
                    previous_action,
                    motion,
                    _blend_governor(
                        launch_governor,
                        post_gate_governor,
                        authority_blend,
                    ),
                )
                previous_action = latest_action
                policy_steps += 1
                next_policy_s = now_s + control_period_s
                _append_jsonl(
                    policy_log_path,
                    {
                        "monotonic_time_s": now_s,
                        "frame_id": state["latest_frame_id"],
                        "requested_action": dict(
                            zip(ACTION_NAMES, requested_action)
                        ),
                        "governed_action": dict(
                            zip(ACTION_NAMES, latest_action)
                        ),
                        "motion_observation": motion,
                    },
                )

            if latest_action is not None and now_s >= next_command_s:
                actuator_action = (
                    latest_action[0],
                    latest_action[1]
                    * MEASURED_ROLL_ACTION_SIGN
                    * lateral_authority_scale,
                    latest_action[2] * pitch_authority_scale,
                    latest_action[3],
                )
                launch_mapped = launch_calibration.map_action(actuator_action)
                post_mapped = post_gate_calibration.map_action(actuator_action)
                post_mapped["thrust_normalized"] = _clamp(
                    post_gate_calibration.hover_thrust
                    + (
                        post_mapped["thrust_normalized"]
                        - post_gate_calibration.hover_thrust
                    )
                    * post_thrust_gain,
                    0.0,
                    1.0,
                )
                mapped = {
                    key: _lerp(
                        launch_mapped[key],
                        post_mapped[key],
                        authority_blend,
                    )
                    for key in launch_mapped
                }
                command = ControlCommand(
                    monotonic_time_s=now_s,
                    roll_rate_radps=mapped["roll_rate_radps"],
                    pitch_rate_radps=mapped["pitch_rate_radps"],
                    yaw_rate_radps=mapped["yaw_rate_radps"],
                    thrust_normalized=mapped["thrust_normalized"],
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

    summary = SessionEvaluator(recorder.layout).summarize(
        abort_reason=abort_reason
    )
    summary.update(
        {
            "policy_steps": policy_steps,
            "policy_command_count": command_count,
            "gate0_passed": int(state["max_active_gate_index"]) >= 1,
            "target_gates": target_gates,
            "control_rate_hz": control_rate_hz,
            "lateral_authority_scale": lateral_authority_scale,
            "pitch_authority_scale": pitch_authority_scale,
            "governor_slew_scale": governor_slew_scale,
            "gate_source": gate_source,
            "post_control_rate_hz": post_control_rate_hz,
            "uniform_authority": uniform_authority,
            "launch_phase_s": launch_phase_s,
            "authority_ramp_s": authority_ramp_s,
            "thrust_ungoverned_fraction": thrust_ungoverned_fraction,
            "direction_ungoverned_fraction": direction_ungoverned_fraction,
            "post_thrust_gain": post_thrust_gain,
            "allow_gate_plane_miss": allow_gate_plane_miss,
            "target_reached": success,
            "race_finished": bool(state["race_finished"]),
            "race_finish_time_ns": int(state["race_finish_time_ns"]),
            "track_gate_count": state["track_gate_count"],
            "max_active_gate_index": int(state["max_active_gate_index"]),
            "final_active_gate_index": int(state["active_gate_index"]),
            "policy_validation_status": policy.status.validation_status,
        }
    )
    recorder.write_summary(summary)
    return summary


def _body_features(
    sample: TelemetrySample,
    pitch_reference_rad: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    assert sample.velocity_mps is not None
    assert sample.attitude_rad is not None
    return _telemetry_body_features(
        {
            "velocity_mps": asdict(sample.velocity_mps),
            "attitude_rad": {
                "roll_rad": sample.attitude_rad.roll_rad,
                "pitch_rad": -(
                    sample.attitude_rad.pitch_rad - pitch_reference_rad
                ),
                "yaw_rad": sample.attitude_rad.yaw_rad,
            },
        }
    )


def _fast_gate_detections(frame, frame_bytes: bytes | None) -> list[GateObservation]:
    if frame_bytes is None:
        return []
    import cv2
    import numpy as np

    image = cv2.imdecode(
        np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = (
        (saturation >= round(0.45 * 255.0))
        & (value >= round(0.58 * 255.0))
        & ((hue <= round(0.16 * 179.0)) | (hue >= round(0.90 * 179.0)))
    ).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for component_id in range(1, count):
        left, top, width, height, area = (
            int(item) for item in stats[component_id]
        )
        if area < 48 or width < 8 or height < 8:
            continue
        confidence = min(
            1.0,
            float(area) / max(float(width * height), 1.0),
        )
        candidates.append(
            GateObservation(
                frame_id=frame.frame_id,
                monotonic_time_s=frame.monotonic_time_s,
                confidence=confidence,
                bbox=NormalizedBox(
                    center=NormalizedPoint(
                        x=(left + width / 2.0) / frame.width_px,
                        y=(top + height / 2.0) / frame.height_px,
                    ),
                    width=width / frame.width_px,
                    height=height / frame.height_px,
                ),
                gate_label="highlighted_gate_fast",
            )
        )
    candidates.sort(
        key=lambda item: (
            item.bbox.width * item.bbox.height,
            item.confidence,
        ),
        reverse=True,
    )
    return candidates


def _project_active_gate(
    sample: TelemetrySample,
    gate: tuple[float, float, float, float, float] | None,
    pitch_reference_rad: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    if (
        gate is None
        or sample.position_m is None
        or sample.attitude_rad is None
    ):
        return None
    position = sample.position_m
    attitude = sample.attitude_rad
    rotation_body_frd_to_ned = _rotation_matrix(
        attitude.roll_rad,
        -(attitude.pitch_rad - pitch_reference_rad),
        attitude.yaw_rad,
    )
    relative_ned = (
        gate[0] - position.x,
        gate[1] - position.y,
        gate[2] - position.z,
    )
    relative_frd = _transpose_matvec(
        rotation_body_frd_to_ned,
        relative_ned,
    )
    relative_flu = (
        relative_frd[0],
        -relative_frd[1],
        -relative_frd[2],
    )
    depth = relative_flu[0]
    if depth <= 0.05:
        return None
    tan_horizontal_half_fov = math.tan(math.radians(90.0) / 2.0)
    tan_vertical_half_fov = math.tan(math.radians(70.0) / 2.0)
    center_x = -relative_flu[1] / (depth * tan_horizontal_half_fov)
    center_y = -relative_flu[2] / (depth * tan_vertical_half_fov)
    width = min(1.0, (gate[3] * 0.5) / (depth * tan_horizontal_half_fov))
    height = min(1.0, (gate[4] * 0.5) / (depth * tan_vertical_half_fov))
    visible = abs(center_x) <= 1.0 and abs(center_y) <= 1.0
    confidence = (
        math.exp(-0.35 * (center_x * center_x + center_y * center_y))
        if visible
        else 0.0
    )
    return (center_x, center_y), (width, height), confidence


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
    if initial_position is not None and initial_position[2] - position[2] > 4.0:
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


def _vector_tuple(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return (float(value.x), float(value.y), float(value.z))


def _attitude_tuple(
    sample: TelemetrySample,
) -> tuple[float, float, float] | None:
    if sample.attitude_rad is None:
        return None
    return (
        float(sample.attitude_rad.roll_rad),
        float(sample.attitude_rad.pitch_rad),
        float(sample.attitude_rad.yaw_rad),
    )


def _vector_norm(value) -> float:
    vector = _vector_tuple(value)
    if vector is None:
        return math.inf
    return math.sqrt(sum(component * component for component in vector))


def _angle_delta(value: float, reference: float) -> float:
    return (value - reference + math.pi) % (2.0 * math.pi) - math.pi


def _authority_blend(
    *,
    elapsed_s: float,
    active_gate_index: int,
    uniform_authority: bool,
    launch_phase_s: float | None,
    authority_ramp_s: float,
) -> float:
    if uniform_authority or active_gate_index >= 1:
        return 1.0
    if launch_phase_s is None or elapsed_s <= launch_phase_s:
        return 0.0
    if authority_ramp_s <= 0.0:
        return 1.0
    return min(1.0, (elapsed_s - launch_phase_s) / authority_ramp_s)


def _blend_governor(
    launch_governor: dict[str, object] | None,
    post_governor: dict[str, object] | None,
    blend: float,
) -> dict[str, object] | None:
    if blend <= 0.0 or post_governor is None:
        return launch_governor
    if blend >= 1.0 or launch_governor is None:
        return post_governor
    launch_slew = launch_governor.get("slew_limits")
    post_slew = post_governor.get("slew_limits")
    if not isinstance(launch_slew, list) or not isinstance(post_slew, list):
        return post_governor
    return {
        **post_governor,
        "slew_limits": [
            _lerp(float(start), float(end), blend)
            for start, end in zip(launch_slew, post_slew)
        ],
    }


def _lerp(start: float, end: float, blend: float) -> float:
    return start + (end - start) * min(max(blend, 0.0), 1.0)


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
            EXECUTION_ROOT
            / "policy"
            / "models"
            / "ai_gp_017_motion_safety_governed.json"
        ),
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--target-gates",
        type=int,
        default=1,
        help="stop after this many gates; use 0 to continue until race finish",
    )
    parser.add_argument("--control-rate-hz", type=float, default=12.5)
    parser.add_argument(
        "--lateral-authority-scale",
        type=float,
        default=1.0,
        help="scale normalized policy roll before live calibration",
    )
    parser.add_argument(
        "--pitch-authority-scale",
        type=float,
        default=1.0,
        help="scale normalized policy pitch before live calibration",
    )
    parser.add_argument("--governor-slew-scale", type=float, default=1.0)
    parser.add_argument(
        "--gate-source",
        choices=("vision", "track_pose"),
        default="vision",
    )
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--post-control-rate-hz", type=float, default=12.5)
    parser.add_argument("--post-max-roll-rate-radps", type=float, default=0.005)
    parser.add_argument("--post-max-pitch-rate-radps", type=float, default=0.07)
    parser.add_argument("--post-max-yaw-rate-radps", type=float, default=0.01)
    parser.add_argument("--post-thrust-span-up", type=float, default=0.008)
    parser.add_argument("--post-thrust-span-down", type=float, default=0.008)
    parser.add_argument(
        "--uniform-authority",
        action="store_true",
        help="apply post-gate rate, calibration, and governor settings from launch",
    )
    parser.add_argument(
        "--launch-phase-s",
        type=float,
        help="switch permanently to post-gate authority this many seconds after launch",
    )
    parser.add_argument(
        "--authority-ramp-s",
        type=float,
        default=0.0,
        help="interpolate authority over this duration after the launch phase",
    )
    parser.add_argument(
        "--thrust-ungoverned-fraction",
        type=float,
        default=0.0,
        help="0 keeps current thrust mapping; 1 maps policy thrust directly to 0..1",
    )
    parser.add_argument(
        "--direction-ungoverned-fraction",
        type=float,
        default=0.0,
        help="0 keeps current rate limits; 1 uses training limits 3/3/2 rad/s",
    )
    parser.add_argument(
        "--post-thrust-gain",
        type=float,
        default=1.0,
        help="multiply policy-derived post-launch thrust deviation around hover",
    )
    parser.add_argument(
        "--allow-gate-plane-miss",
        action="store_true",
        help="continue after passing an uncompleted gate plane",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_id = args.run_id or (
        "bounded_policy_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    summaries = []
    for attempt_index in range(args.attempts):
        attempt_run_id = (
            run_id
            if args.attempts == 1
            else f"{run_id}_{attempt_index + 1:02d}"
        )
        summaries.append(
            run(
                args.policy,
                attempt_run_id,
                args.duration,
                args.target_gates,
                args.control_rate_hz,
                args.lateral_authority_scale,
                args.pitch_authority_scale,
                args.governor_slew_scale,
                args.gate_source,
                args.post_control_rate_hz,
                args.post_max_roll_rate_radps,
                args.post_max_pitch_rate_radps,
                args.post_max_yaw_rate_radps,
                args.post_thrust_span_up,
                args.post_thrust_span_down,
                args.uniform_authority,
                args.launch_phase_s,
                args.authority_ramp_s,
                args.thrust_ungoverned_fraction,
                args.direction_ungoverned_fraction,
                args.post_thrust_gain,
                args.allow_gate_plane_miss,
            )
        )
    output = (
        summaries[0]
        if len(summaries) == 1
        else {
            "attempt_count": len(summaries),
            "race_finish_count": sum(
                bool(summary["race_finished"]) for summary in summaries
            ),
            "gate0_pass_count": sum(
                bool(summary["gate0_passed"]) for summary in summaries
            ),
            "max_gate_index": max(
                int(summary["max_active_gate_index"])
                for summary in summaries
            ),
            "collision_attempt_count": sum(
                int(summary["collision_count"]) > 0
                for summary in summaries
            ),
            "attempts": summaries,
        }
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
