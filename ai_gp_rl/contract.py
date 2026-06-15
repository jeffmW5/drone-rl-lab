"""Shared observation and action contract for training and live deployment."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


ACTION_NAMES = (
    "collective_offset",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
)
ACTION_DIM = len(ACTION_NAMES)

ACTOR_FEATURE_NAMES = (
    "body_velocity_x",
    "body_velocity_y",
    "body_velocity_z",
    "gravity_body_x",
    "gravity_body_y",
    "gravity_body_z",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
    "gate_center_x",
    "gate_center_y",
    "gate_area",
    "gate_confidence",
    "gate_age",
    "previous_collective_offset",
    "previous_roll_rate",
    "previous_pitch_rate",
    "previous_yaw_rate",
)
ACTOR_OBS_DIM = len(ACTOR_FEATURE_NAMES)

TEMPORAL_BASE_FEATURE_NAMES = (
    "body_velocity_x",
    "body_velocity_y",
    "body_velocity_z",
    "gravity_body_x",
    "gravity_body_y",
    "gravity_body_z",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
    "gate_center_x",
    "gate_center_y",
    "gate_width",
    "gate_height",
    "gate_area",
    "gate_confidence",
    "gate_age",
    "previous_collective_offset",
    "previous_roll_rate",
    "previous_pitch_rate",
    "previous_yaw_rate",
)
TEMPORAL_BASE_OBS_DIM = len(TEMPORAL_BASE_FEATURE_NAMES)
TEMPORAL_HISTORY_LENGTH = 4

CORNER_BASE_FEATURE_NAMES = (
    "body_velocity_x",
    "body_velocity_y",
    "body_velocity_z",
    "gravity_body_x",
    "gravity_body_y",
    "gravity_body_z",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
    "gate_center_x",
    "gate_center_y",
    "gate_width",
    "gate_height",
    "gate_area",
    "gate_top_left_x",
    "gate_top_left_y",
    "gate_top_right_x",
    "gate_top_right_y",
    "gate_bottom_right_x",
    "gate_bottom_right_y",
    "gate_bottom_left_x",
    "gate_bottom_left_y",
    "gate_corner_valid",
    "gate_confidence",
    "gate_age",
    "previous_collective_offset",
    "previous_roll_rate",
    "previous_pitch_rate",
    "previous_yaw_rate",
)
CORNER_BASE_OBS_DIM = len(CORNER_BASE_FEATURE_NAMES)

MOTION_FEATURE_NAMES = TEMPORAL_BASE_FEATURE_NAMES + (
    "gate_center_delta_x",
    "gate_center_delta_y",
    "gate_log_width_delta",
    "gate_log_height_delta",
)
MOTION_OBS_DIM = len(MOTION_FEATURE_NAMES)


def temporal_feature_names(
    history_length: int = TEMPORAL_HISTORY_LENGTH,
) -> tuple[str, ...]:
    return stacked_feature_names(TEMPORAL_BASE_FEATURE_NAMES, history_length)


def corner_temporal_feature_names(
    history_length: int = TEMPORAL_HISTORY_LENGTH,
) -> tuple[str, ...]:
    return stacked_feature_names(CORNER_BASE_FEATURE_NAMES, history_length)


def stacked_feature_names(
    base_feature_names: Sequence[str],
    history_length: int,
) -> tuple[str, ...]:
    if history_length < 1:
        raise ValueError("history_length must be positive")
    return tuple(
        f"t_minus_{history_length - frame_index - 1}.{feature_name}"
        for frame_index in range(history_length)
        for feature_name in base_feature_names
    )


SWIFT_TEACHER_FEATURE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "rotation_00",
    "rotation_01",
    "rotation_02",
    "rotation_10",
    "rotation_11",
    "rotation_12",
    "rotation_20",
    "rotation_21",
    "rotation_22",
    "gate_top_left_x",
    "gate_top_left_y",
    "gate_top_left_z",
    "gate_top_right_x",
    "gate_top_right_y",
    "gate_top_right_z",
    "gate_bottom_right_x",
    "gate_bottom_right_y",
    "gate_bottom_right_z",
    "gate_bottom_left_x",
    "gate_bottom_left_y",
    "gate_bottom_left_z",
    "previous_collective_offset",
    "previous_roll_rate",
    "previous_pitch_rate",
    "previous_yaw_rate",
)
SWIFT_TEACHER_OBS_DIM = len(SWIFT_TEACHER_FEATURE_NAMES)

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
STRUCTURED_TEACHER_OBS_DIM = len(STRUCTURED_TEACHER_FEATURE_NAMES)

VELOCITY_SCALE_MPS = 8.0
RATE_SCALES_RADPS = (3.0, 3.0, 2.0)
DETECTION_AGE_SCALE_S = 0.5


@dataclass(frozen=True)
class LivePolicyFeatures:
    """Deployable policy inputs after telemetry and vision preprocessing."""

    body_velocity_mps: tuple[float, float, float]
    gravity_body: tuple[float, float, float]
    angular_rate_radps: tuple[float, float, float]
    gate_center_normalized: tuple[float, float]
    gate_area_normalized: float
    gate_confidence: float
    gate_age_s: float
    previous_action: tuple[float, float, float, float]


@dataclass(frozen=True)
class TemporalLivePolicyFeatures:
    """One frame of the richer temporal live-policy input contract."""

    body_velocity_mps: tuple[float, float, float]
    gravity_body: tuple[float, float, float]
    angular_rate_radps: tuple[float, float, float]
    gate_center_normalized: tuple[float, float]
    gate_size_normalized: tuple[float, float]
    gate_area_normalized: float
    gate_confidence: float
    gate_age_s: float
    previous_action: tuple[float, float, float, float]


def build_actor_observation(features: LivePolicyFeatures) -> list[float]:
    """Build the exact 18D actor observation used during training."""

    obs = [
        *(value / VELOCITY_SCALE_MPS for value in features.body_velocity_mps),
        *features.gravity_body,
        *(
            value / scale
            for value, scale in zip(features.angular_rate_radps, RATE_SCALES_RADPS)
        ),
        _clamp(features.gate_center_normalized[0], -1.5, 1.5),
        _clamp(features.gate_center_normalized[1], -1.5, 1.5),
        _clamp(features.gate_area_normalized, 0.0, 1.0),
        _clamp(features.gate_confidence, 0.0, 1.0),
        _clamp(features.gate_age_s / DETECTION_AGE_SCALE_S, 0.0, 1.0),
        *(_clamp(value, -1.0, 1.0) for value in features.previous_action),
    ]
    if len(obs) != ACTOR_OBS_DIM or not all(isfinite(value) for value in obs):
        raise ValueError("invalid AI-GP actor observation")
    return obs


def build_temporal_base_observation(
    features: TemporalLivePolicyFeatures,
) -> list[float]:
    """Build one 20D frame for a temporal live policy."""

    obs = [
        *(value / VELOCITY_SCALE_MPS for value in features.body_velocity_mps),
        *features.gravity_body,
        *(
            value / scale
            for value, scale in zip(features.angular_rate_radps, RATE_SCALES_RADPS)
        ),
        _clamp(features.gate_center_normalized[0], -1.5, 1.5),
        _clamp(features.gate_center_normalized[1], -1.5, 1.5),
        _clamp(features.gate_size_normalized[0], 0.0, 1.0),
        _clamp(features.gate_size_normalized[1], 0.0, 1.0),
        _clamp(features.gate_area_normalized, 0.0, 1.0),
        _clamp(features.gate_confidence, 0.0, 1.0),
        _clamp(features.gate_age_s / DETECTION_AGE_SCALE_S, 0.0, 1.0),
        *(_clamp(value, -1.0, 1.0) for value in features.previous_action),
    ]
    if len(obs) != TEMPORAL_BASE_OBS_DIM or not all(
        isfinite(value) for value in obs
    ):
        raise ValueError("invalid AI-GP temporal base observation")
    return obs


def build_corner_base_observation(
    features: TemporalLivePolicyFeatures,
    gate_corners_normalized: Sequence[tuple[float, float]],
    *,
    corner_valid: bool = True,
) -> list[float]:
    """Build one frame with image-measured gate corners and validity."""

    if len(gate_corners_normalized) != 4:
        raise ValueError("gate corner observation requires four ordered corners")
    temporal = build_temporal_base_observation(features)
    obs = [
        *temporal[:14],
        *(
            _clamp(value, -1.5, 1.5)
            for corner in gate_corners_normalized
            for value in corner
        ),
        1.0 if corner_valid else 0.0,
        *temporal[14:],
    ]
    if len(obs) != CORNER_BASE_OBS_DIM or not all(
        isfinite(value) for value in obs
    ):
        raise ValueError("invalid AI-GP corner base observation")
    return obs


def build_motion_observation(
    current_base_observation: Sequence[float],
    previous_base_observation: Sequence[float],
) -> list[float]:
    """Add optical center and expansion deltas to one 20D live frame."""

    if (
        len(current_base_observation) != TEMPORAL_BASE_OBS_DIM
        or len(previous_base_observation) != TEMPORAL_BASE_OBS_DIM
    ):
        raise ValueError("motion observation requires two 20D base frames")
    current = [float(value) for value in current_base_observation]
    previous = [float(value) for value in previous_base_observation]
    center_delta = [
        _clamp((current[index] - previous[index]) / 0.25, -2.0, 2.0)
        for index in (9, 10)
    ]
    size_delta = [
        _clamp(
            _safe_log(current[index]) - _safe_log(previous[index]),
            -2.0,
            2.0,
        )
        for index in (11, 12)
    ]
    obs = [*current, *center_delta, *size_delta]
    if len(obs) != MOTION_OBS_DIM or not all(isfinite(value) for value in obs):
        raise ValueError("invalid AI-GP motion observation")
    return obs


@dataclass(frozen=True)
class ActionCalibration:
    """Maps normalized policy actions onto the live body-rate command contract.

    No default hover thrust is provided because it has not been calibrated in
    the AI-GP simulator. Constructing this object is an explicit promotion gate.
    """

    hover_thrust: float
    thrust_span_up: float
    thrust_span_down: float
    max_roll_rate_radps: float = RATE_SCALES_RADPS[0]
    max_pitch_rate_radps: float = RATE_SCALES_RADPS[1]
    max_yaw_rate_radps: float = RATE_SCALES_RADPS[2]

    def __post_init__(self) -> None:
        values = (
            self.hover_thrust,
            self.thrust_span_up,
            self.thrust_span_down,
            self.max_roll_rate_radps,
            self.max_pitch_rate_radps,
            self.max_yaw_rate_radps,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("action calibration values must be finite")
        if not 0.0 < self.hover_thrust < 1.0:
            raise ValueError("hover_thrust must be calibrated inside (0, 1)")
        if self.thrust_span_up <= 0.0 or self.thrust_span_down <= 0.0:
            raise ValueError("thrust spans must be positive")
        if self.hover_thrust + self.thrust_span_up > 1.0:
            raise ValueError("upward thrust span exceeds normalized command range")
        if self.hover_thrust - self.thrust_span_down < 0.0:
            raise ValueError("downward thrust span exceeds normalized command range")

    def map_action(self, action: Sequence[float]) -> dict[str, float]:
        if len(action) != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM} actions, received {len(action)}")
        collective, roll, pitch, yaw = (_clamp(float(value), -1.0, 1.0) for value in action)
        thrust_span = self.thrust_span_up if collective >= 0.0 else self.thrust_span_down
        return {
            "thrust_normalized": self.hover_thrust + collective * thrust_span,
            "roll_rate_radps": roll * self.max_roll_rate_radps,
            "pitch_rate_radps": pitch * self.max_pitch_rate_radps,
            "yaw_rate_radps": yaw * self.max_yaw_rate_radps,
        }


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _safe_log(value: float) -> float:
    from math import log

    return log(max(value, 1e-4))
