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


def run(policy_path: Path, run_id: str, duration_s: float) -> dict[str, object]:
    _reset_simulator(1.0)

    policy = load_policy(policy_path)
    policy.verify_test_vectors()
    if policy.observation_contract != "motion_live_v1":
        raise ValueError("bounded runner currently requires motion_live_v1")

    calibration = ActionCalibration(
        hover_thrust=0.295,
        thrust_span_up=0.008,
        thrust_span_down=0.008,
        max_roll_rate_radps=0.005,
        max_pitch_rate_radps=0.07,
        max_yaw_rate_radps=0.01,
    )
    recorder = SessionRecorder(
        STACK_ROOT / "replay" / "sessions",
        run_id,
        {
            "created_at_epoch_s": time.time(),
            "mode": "bounded_learned_policy",
            "policy_path": str(policy_path),
            "policy_validation_status": policy.status.validation_status,
            "policy_observation_contract": policy.observation_contract,
            "calibration": asdict(calibration),
            "duration_s": duration_s,
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
            if int(state["active_gate_index"]) >= 1:
                success = True
                break

            if state["new_detection"]:
                state["new_detection"] = False
                detection = state["latest_detection"]
                assert detection is not None
                body_velocity, gravity_body = _body_features(sample)
                angular_rate = _vector_tuple(sample.angular_rate_radps) or (
                    0.0,
                    0.0,
                    0.0,
                )
                detection_age_s = max(
                    0.0,
                    now_s - float(state["latest_detection_time_s"]),
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
                        gate_center_normalized=(
                            (detection.bbox.center.x - 0.5) * 2.0,
                            (detection.bbox.center.y - 0.5) * 2.0,
                        ),
                        gate_size_normalized=(
                            detection.bbox.width,
                            detection.bbox.height,
                        ),
                        gate_area_normalized=(
                            detection.bbox.width * detection.bbox.height
                        ),
                        gate_confidence=detection.confidence,
                        gate_age_s=detection_age_s,
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
                    policy.action_governor,
                )
                previous_action = latest_action
                policy_steps += 1
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
                    -latest_action[1],
                    latest_action[2],
                    latest_action[3],
                )
                mapped = calibration.map_action(actuator_action)
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
                next_command_s = now_s + 0.021
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
            "gate0_passed": success,
            "final_active_gate_index": int(state["active_gate_index"]),
            "policy_validation_status": policy.status.validation_status,
        }
    )
    recorder.write_summary(summary)
    return summary


def _body_features(
    sample: TelemetrySample,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    assert sample.velocity_mps is not None
    assert sample.attitude_rad is not None
    return _telemetry_body_features(
        {
            "velocity_mps": asdict(sample.velocity_mps),
            "attitude_rad": {
                "roll_rad": sample.attitude_rad.roll_rad,
                "pitch_rad": -sample.attitude_rad.pitch_rad,
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
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates


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
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_id = args.run_id or (
        "bounded_policy_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    print(json.dumps(run(args.policy, run_id, args.duration), indent=2))


if __name__ == "__main__":
    main()
