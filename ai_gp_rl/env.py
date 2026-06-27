"""GPU-vectorized surrogate environment for AI Grand Prix gate flight."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, pi
from typing import Any

import torch

from .contract import (
    ACTION_DIM,
    ACTOR_FEATURE_NAMES,
    ACTOR_OBS_DIM,
    CORNER_BASE_FEATURE_NAMES,
    CORNER_BASE_OBS_DIM,
    DETECTION_AGE_SCALE_S,
    MOTION_FEATURE_NAMES,
    MOTION_OBS_DIM,
    RATE_SCALES_RADPS,
    STRUCTURED_TEACHER_FEATURE_NAMES,
    STRUCTURED_TEACHER_OBS_DIM,
    SWIFT_TEACHER_FEATURE_NAMES,
    SWIFT_TEACHER_OBS_DIM,
    TEMPORAL_BASE_FEATURE_NAMES,
    TEMPORAL_BASE_OBS_DIM,
    TEMPORAL_HISTORY_LENGTH,
    VELOCITY_SCALE_MPS,
    corner_temporal_feature_names,
    temporal_feature_names,
)
from .track import (
    AI_GP_GATE_SIZE_M,
    AI_GP_TRACK_GROUND_CLEARANCE_M,
    AI_GP_TRACK_NAME,
    ai_gp_track_altitude_offset_m,
    ai_gp_track_surrogate_positions,
    ned_position_to_surrogate,
)


@dataclass
class AIGPEnvConfig:
    actor_observation_mode: str = "live_features"
    live_observation_mode: str | None = None
    observation_history_length: int = TEMPORAL_HISTORY_LENGTH
    num_envs: int = 4096
    device: str = "cuda"
    seed: int = 42
    dt: float = 0.02
    physics_substeps: int = 2
    max_episode_steps: int = 500
    randomization: bool = True

    track_name: str | None = None
    track_ground_clearance_m: float = AI_GP_TRACK_GROUND_CLEARANCE_M
    start_position_m: tuple[float, float, float] = (-2.0, 0.0, 0.03)
    start_position_ned_m: tuple[float, float, float] | None = None
    start_position_noise_m: tuple[float, float, float] = (0.15, 0.15, 0.01)
    start_velocity_noise_mps: float = 0.15
    align_spawn_heading_to_gate: bool = False
    spawn_yaw_noise_rad: float = 0.20
    spawn_tilt_noise_rad: float = 0.04
    near_gate_spawn_ratio_start: float = 0.70
    near_gate_spawn_ratio_end: float = 0.10
    near_gate_indices: tuple[int, ...] | None = None
    near_gate_index_weights: tuple[float, ...] | None = None
    near_gate_distance_m: tuple[float, float] = (1.5, 3.5)
    near_gate_lateral_offset_m: tuple[float, float] = (-0.45, 0.45)
    near_gate_vertical_offset_m: tuple[float, float] = (-0.35, 0.35)
    near_gate_forward_speed_mps: tuple[float, float] = (0.0, 0.0)
    near_gate_lateral_speed_mps: tuple[float, float] = (-0.3, 0.3)
    near_gate_vertical_speed_mps: tuple[float, float] = (-0.3, 0.3)
    near_gate_roll_rad: tuple[float, float] | None = None
    near_gate_pitch_rad: tuple[float, float] | None = None
    near_gate_yaw_offset_rad: tuple[float, float] | None = None
    near_gate_previous_action: tuple[float, float, float, float] | None = None
    near_gate_previous_action_range: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None = None
    gate_jitter_m: tuple[float, float, float] = (0.0, 0.20, 0.15)

    gate_positions_m: tuple[tuple[float, float, float], ...] = (
        (4.0, 0.0, 1.25),
        (9.0, 1.5, 1.60),
        (14.0, -1.0, 1.20),
        (19.0, 0.5, 1.50),
    )
    gate_half_width_m: float = 0.80
    gate_half_height_m: float = 0.80
    upright_gate_normals: bool = False
    position_observation_scale_m: tuple[float, float, float] = (20.0, 10.0, 5.0)

    max_roll_rate_radps: float = RATE_SCALES_RADPS[0]
    max_pitch_rate_radps: float = RATE_SCALES_RADPS[1]
    max_yaw_rate_radps: float = RATE_SCALES_RADPS[2]
    dynamics_model: str = "legacy_collective"
    collective_range_g: float = 0.80
    thrust_command_center: float = 0.295
    thrust_command_span_up: float = 0.105
    thrust_command_span_down: float = 0.095
    thrust_acceleration_bias_mps2: float = 0.0
    thrust_acceleration_gain_mps2: float = 0.0
    base_pitch_offset_rad: float = 0.0
    linear_drag_xyz: tuple[float, float, float] = (0.15, 0.15, 0.15)
    quadratic_drag_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drag_scale_range: tuple[float, float] = (1.0, 1.0)
    rate_response_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rate_response_gain_scale_range: tuple[float, float] = (1.0, 1.0)
    command_latency_s: float = 0.0
    command_latency_s_range: tuple[float, float] = (0.0, 0.0)
    mass_scale_range: tuple[float, float] = (0.85, 1.15)
    thrust_scale_range: tuple[float, float] = (0.85, 1.15)
    linear_drag_range: tuple[float, float] = (0.05, 0.25)
    rate_time_constant_s: float = 0.10
    rate_time_constant_s_range: tuple[float, float] = (0.06, 0.14)
    action_response: float = 0.70
    action_response_range: tuple[float, float] = (0.45, 0.90)
    wind_accel_mps2: float = 0.35

    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 70.0
    vision_noise_std: float = 0.025
    vision_dropout_probability: float = 0.08
    detection_memory_s: float = DETECTION_AGE_SCALE_S

    max_altitude_m: float = 4.0
    max_lateral_m: float = 8.0
    min_forward_m: float = -6.0
    max_forward_m: float = 28.0
    max_tilt_rad: float = 1.35
    ground_impact_speed_mps: float = 1.5
    vertical_runaway_speed_mps: float = 6.0
    soft_collision_fraction: float = 0.25

    progress_reward: float = 12.0
    gate_reward: float = 20.0
    finish_reward: float = 40.0
    forward_speed_reward: float = 0.10
    visibility_reward: float = 0.02
    camera_alignment_reward: float = 0.0
    camera_alignment_exponent: float = 4.0
    alive_reward: float = 0.002
    angular_rate_penalty: float = 0.002
    action_delta_penalty: float = 0.01
    thrust_saturation_penalty: float = 0.0
    vertical_speed_penalty: float = 0.03
    altitude_penalty: float = 2.0
    low_altitude_penalty: float = 0.0
    soft_floor_altitude_m: float = 0.0
    gate_altitude_error_penalty: float = 0.0
    gate_lateral_error_penalty: float = 0.0
    soft_altitude_limit_m: float | None = None
    collision_penalty: float = 8.0
    out_of_bounds_penalty: float = 12.0
    missed_gate_penalty: float = 0.0
    terminate_on_missed_gate: bool = False

    swift_teacher_speed_base_mps: float = 4.0
    swift_teacher_speed_distance_gain: float = 0.50
    swift_teacher_min_forward_speed_mps: float = 3.5
    swift_teacher_max_forward_speed_mps: float = 13.5
    swift_teacher_cross_track_speed_gain: float = 0.16
    swift_teacher_min_speed_scale: float = 0.45
    swift_teacher_lateral_position_gain: float = 1.8
    swift_teacher_vertical_position_gain: float = 1.6
    swift_teacher_lateral_speed_gain: float = 2.0
    swift_teacher_vertical_speed_gain: float = 2.2
    swift_teacher_target_lateral_speed_limit_mps: float = 4.5
    swift_teacher_target_vertical_speed_limit_mps: float = 4.0
    swift_teacher_forward_accel_limit_mps2: float = 7.0
    swift_teacher_lateral_accel_limit_mps2: float = 8.0
    swift_teacher_vertical_accel_limit_mps2: float = 8.0

    def __post_init__(self) -> None:
        if self.dynamics_model not in {"legacy_collective", "measured_ai_gp_v1"}:
            raise ValueError(
                "dynamics_model must be 'legacy_collective' or "
                "'measured_ai_gp_v1'"
            )
        for name in (
            "swift_teacher_speed_base_mps",
            "swift_teacher_speed_distance_gain",
            "swift_teacher_min_forward_speed_mps",
            "swift_teacher_max_forward_speed_mps",
            "swift_teacher_cross_track_speed_gain",
            "swift_teacher_min_speed_scale",
            "swift_teacher_lateral_position_gain",
            "swift_teacher_vertical_position_gain",
            "swift_teacher_lateral_speed_gain",
            "swift_teacher_vertical_speed_gain",
            "swift_teacher_target_lateral_speed_limit_mps",
            "swift_teacher_target_vertical_speed_limit_mps",
            "swift_teacher_forward_accel_limit_mps2",
            "swift_teacher_lateral_accel_limit_mps2",
            "swift_teacher_vertical_accel_limit_mps2",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.swift_teacher_min_forward_speed_mps > self.swift_teacher_max_forward_speed_mps:
            raise ValueError("swift teacher forward speed limits must be ordered")
        if self.swift_teacher_min_speed_scale > 1.0:
            raise ValueError("swift_teacher_min_speed_scale cannot exceed 1.0")
        if self.physics_substeps < 1:
            raise ValueError("physics_substeps must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.low_altitude_penalty < 0.0:
            raise ValueError("low_altitude_penalty cannot be negative")
        if self.soft_floor_altitude_m < 0.0:
            raise ValueError("soft_floor_altitude_m cannot be negative")
        if self.command_latency_s < 0.0:
            raise ValueError("command_latency_s cannot be negative")
        if self.command_latency_s_range[0] < 0.0:
            raise ValueError("command_latency_s_range cannot be negative")
        if self.command_latency_s_range[0] > self.command_latency_s_range[1]:
            raise ValueError("command_latency_s_range must be ordered")
        for name, limits in (
            ("near_gate_distance_m", self.near_gate_distance_m),
            ("near_gate_lateral_offset_m", self.near_gate_lateral_offset_m),
            ("near_gate_vertical_offset_m", self.near_gate_vertical_offset_m),
            ("near_gate_forward_speed_mps", self.near_gate_forward_speed_mps),
            ("near_gate_lateral_speed_mps", self.near_gate_lateral_speed_mps),
            ("near_gate_vertical_speed_mps", self.near_gate_vertical_speed_mps),
        ):
            if limits[0] > limits[1]:
                raise ValueError(f"{name} must be ordered")
        for name, limits in (
            ("near_gate_roll_rad", self.near_gate_roll_rad),
            ("near_gate_pitch_rad", self.near_gate_pitch_rad),
            ("near_gate_yaw_offset_rad", self.near_gate_yaw_offset_rad),
        ):
            if limits is not None and limits[0] > limits[1]:
                raise ValueError(f"{name} must be ordered")
        if self.near_gate_previous_action is not None and any(
            value < -1.0 or value > 1.0
            for value in self.near_gate_previous_action
        ):
            raise ValueError("near_gate_previous_action must be normalized")
        if (
            self.near_gate_previous_action is not None
            and self.near_gate_previous_action_range is not None
        ):
            raise ValueError(
                "near_gate_previous_action and "
                "near_gate_previous_action_range are mutually exclusive"
            )
        if self.near_gate_previous_action_range is not None:
            if len(self.near_gate_previous_action_range) != ACTION_DIM:
                raise ValueError(
                    "near_gate_previous_action_range must have one range per action"
                )
            for limits in self.near_gate_previous_action_range:
                if limits[0] > limits[1]:
                    raise ValueError("near_gate_previous_action_range must be ordered")
                if limits[0] < -1.0 or limits[1] > 1.0:
                    raise ValueError(
                        "near_gate_previous_action_range must be normalized"
                    )
        if self.near_gate_index_weights is not None:
            if self.near_gate_indices is None:
                raise ValueError(
                    "near_gate_index_weights requires near_gate_indices"
                )
            if len(self.near_gate_index_weights) != len(self.near_gate_indices):
                raise ValueError(
                    "near_gate_index_weights must match near_gate_indices length"
                )
            if any(value < 0.0 for value in self.near_gate_index_weights):
                raise ValueError("near_gate_index_weights cannot be negative")
            if sum(self.near_gate_index_weights) <= 0.0:
                raise ValueError("near_gate_index_weights must have positive sum")
        if self.dynamics_model == "measured_ai_gp_v1":
            if not 0.0 < self.thrust_command_center < 1.0:
                raise ValueError("thrust_command_center must be inside (0, 1)")
            if (
                self.thrust_command_span_up <= 0.0
                or self.thrust_command_span_down <= 0.0
            ):
                raise ValueError("thrust command spans must be positive")
            if self.thrust_command_center + self.thrust_command_span_up > 1.0:
                raise ValueError("upward thrust command span exceeds 1.0")
            if self.thrust_command_center - self.thrust_command_span_down < 0.0:
                raise ValueError("downward thrust command span is below 0.0")
            if self.thrust_acceleration_gain_mps2 <= 0.0:
                raise ValueError(
                    "measured dynamics require positive thrust acceleration gain"
                )
            if any(value <= 0.0 for value in self.rate_response_gain):
                raise ValueError("rate_response_gain values must be positive")

        if self.track_name is None:
            if self.start_position_ned_m is not None:
                raise ValueError(
                    "start_position_ned_m requires a named NED track"
                )
            return
        if self.track_name != AI_GP_TRACK_NAME:
            raise ValueError(f"unsupported AI-GP track: {self.track_name}")

        self.gate_positions_m = ai_gp_track_surrogate_positions(
            self.track_ground_clearance_m
        )
        self.gate_half_width_m = AI_GP_GATE_SIZE_M / 2.0
        self.gate_half_height_m = AI_GP_GATE_SIZE_M / 2.0
        self.upright_gate_normals = True
        if self.start_position_ned_m is not None:
            self.start_position_m = ned_position_to_surrogate(
                self.start_position_ned_m,
                altitude_offset_m=ai_gp_track_altitude_offset_m(
                    self.track_ground_clearance_m
                ),
            )


class AIGPVectorEnv:
    """A batched point-mass/body-rate quadrotor surrogate on one Torch device."""

    actor_observation_dim = ACTOR_OBS_DIM
    observation_dim = 32
    action_dim = ACTION_DIM

    def __init__(self, config: AIGPEnvConfig) -> None:
        self.config = config
        if config.actor_observation_mode == "live_features":
            self.actor_observation_dim = ACTOR_OBS_DIM
            self.actor_feature_names = ACTOR_FEATURE_NAMES
        elif config.actor_observation_mode == "live_features_temporal":
            self.actor_observation_dim = (
                TEMPORAL_BASE_OBS_DIM * config.observation_history_length
            )
            self.actor_feature_names = temporal_feature_names(
                config.observation_history_length
            )
        elif config.actor_observation_mode == "live_features_recurrent":
            self.actor_observation_dim = TEMPORAL_BASE_OBS_DIM
            self.actor_feature_names = TEMPORAL_BASE_FEATURE_NAMES
        elif config.actor_observation_mode == "live_features_corners_temporal":
            self.actor_observation_dim = (
                CORNER_BASE_OBS_DIM * config.observation_history_length
            )
            self.actor_feature_names = corner_temporal_feature_names(
                config.observation_history_length
            )
        elif config.actor_observation_mode == "live_features_corners_recurrent":
            self.actor_observation_dim = CORNER_BASE_OBS_DIM
            self.actor_feature_names = CORNER_BASE_FEATURE_NAMES
        elif config.actor_observation_mode == "live_features_motion":
            self.actor_observation_dim = MOTION_OBS_DIM
            self.actor_feature_names = MOTION_FEATURE_NAMES
        elif config.actor_observation_mode == "swift_teacher":
            self.actor_observation_dim = SWIFT_TEACHER_OBS_DIM
            self.actor_feature_names = SWIFT_TEACHER_FEATURE_NAMES
        elif config.actor_observation_mode == "structured_teacher_v2":
            self.actor_observation_dim = STRUCTURED_TEACHER_OBS_DIM
            self.actor_feature_names = STRUCTURED_TEACHER_FEATURE_NAMES
        else:
            raise ValueError(
                "actor_observation_mode must be 'live_features', "
                "'live_features_temporal', 'live_features_recurrent', "
                "'live_features_corners_temporal', "
                "'live_features_corners_recurrent', 'live_features_motion', "
                "'swift_teacher', or 'structured_teacher_v2'"
            )
        if config.observation_history_length < 1:
            raise ValueError("observation_history_length must be positive")
        self.live_observation_mode = config.live_observation_mode
        if self.live_observation_mode is None:
            self.live_observation_mode = (
                config.actor_observation_mode
                if config.actor_observation_mode
                not in {"swift_teacher", "structured_teacher_v2"}
                else "live_features"
            )
        if self.live_observation_mode not in (
            "live_features",
            "live_features_temporal",
            "live_features_recurrent",
            "live_features_corners_temporal",
            "live_features_corners_recurrent",
            "live_features_motion",
        ):
            raise ValueError(
                "unsupported live_observation_mode"
            )
        if self.live_observation_mode == "live_features_temporal":
            self.live_actor_observation_dim = (
                TEMPORAL_BASE_OBS_DIM * config.observation_history_length
            )
            self.live_actor_feature_names = temporal_feature_names(
                config.observation_history_length
            )
            self.live_base_feature_names = TEMPORAL_BASE_FEATURE_NAMES
            self.live_observation_contract = "temporal_live_v1"
        elif self.live_observation_mode == "live_features_recurrent":
            self.live_actor_observation_dim = TEMPORAL_BASE_OBS_DIM
            self.live_actor_feature_names = TEMPORAL_BASE_FEATURE_NAMES
            self.live_base_feature_names = TEMPORAL_BASE_FEATURE_NAMES
            self.live_observation_contract = "recurrent_live_v1"
        elif self.live_observation_mode == "live_features_corners_temporal":
            self.live_actor_observation_dim = (
                CORNER_BASE_OBS_DIM * config.observation_history_length
            )
            self.live_actor_feature_names = corner_temporal_feature_names(
                config.observation_history_length
            )
            self.live_base_feature_names = CORNER_BASE_FEATURE_NAMES
            self.live_observation_contract = "corner_temporal_live_v1"
        elif self.live_observation_mode == "live_features_corners_recurrent":
            self.live_actor_observation_dim = CORNER_BASE_OBS_DIM
            self.live_actor_feature_names = CORNER_BASE_FEATURE_NAMES
            self.live_base_feature_names = CORNER_BASE_FEATURE_NAMES
            self.live_observation_contract = "corner_recurrent_live_v1"
        elif self.live_observation_mode == "live_features_motion":
            self.live_actor_observation_dim = MOTION_OBS_DIM
            self.live_actor_feature_names = MOTION_FEATURE_NAMES
            self.live_base_feature_names = MOTION_FEATURE_NAMES
            self.live_observation_contract = "motion_live_v1"
        else:
            self.live_actor_observation_dim = ACTOR_OBS_DIM
            self.live_actor_feature_names = ACTOR_FEATURE_NAMES
            self.live_base_feature_names = ACTOR_FEATURE_NAMES
            self.live_observation_contract = "live_features_v1"
        self.observation_dim = self.actor_observation_dim + 14
        self.device = torch.device(config.device)
        self.num_envs = config.num_envs
        self.dtype = torch.float32
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed)
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.command_rate_limits = torch.tensor(
            (
                config.max_roll_rate_radps,
                config.max_pitch_rate_radps,
                config.max_yaw_rate_radps,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        self.rate_limits = self.command_rate_limits
        self.observation_rate_scales = torch.tensor(
            RATE_SCALES_RADPS,
            device=self.device,
            dtype=self.dtype,
        )
        self.gravity_acceleration = torch.tensor(
            (0.0, 0.0, -9.81), device=self.device, dtype=self.dtype
        )
        self.gravity_unit = torch.tensor(
            (0.0, 0.0, -1.0), device=self.device, dtype=self.dtype
        )
        self.position_observation_scale = torch.tensor(
            config.position_observation_scale_m,
            device=self.device,
            dtype=self.dtype,
        )
        self.tan_horizontal_half_fov = torch.tan(
            torch.tensor(config.horizontal_fov_deg * pi / 360.0, device=self.device)
        )
        self.tan_vertical_half_fov = torch.tan(
            torch.tensor(config.vertical_fov_deg * pi / 360.0, device=self.device)
        )

        self.base_gate_positions = torch.tensor(
            config.gate_positions_m, device=self.device, dtype=self.dtype
        )
        if self.base_gate_positions.ndim != 2 or self.base_gate_positions.shape[1] != 3:
            raise ValueError("gate_positions_m must have shape (gate_count, 3)")
        self.gate_count = self.base_gate_positions.shape[0]
        if config.near_gate_indices is not None:
            if not config.near_gate_indices:
                raise ValueError("near_gate_indices cannot be empty")
            if any(
                index < 0 or index >= self.gate_count
                for index in config.near_gate_indices
            ):
                raise ValueError("near_gate_indices contains an invalid gate")
            self.near_gate_indices = torch.tensor(
                config.near_gate_indices,
                device=self.device,
                dtype=torch.long,
            )
            if config.near_gate_index_weights is None:
                self.near_gate_index_weights = None
            else:
                weights = torch.tensor(
                    config.near_gate_index_weights,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.near_gate_index_weights = weights / weights.sum()
        else:
            self.near_gate_indices = None
            self.near_gate_index_weights = None
        self.gate_normals = self._build_gate_normals(
            self.base_gate_positions,
            upright=config.upright_gate_normals,
        )
        self.gate_lateral, self.gate_vertical = self._build_gate_bases(self.gate_normals)

        shape3 = (self.num_envs, 3)
        self.position = torch.zeros(shape3, device=self.device)
        self.velocity = torch.zeros(shape3, device=self.device)
        self.attitude = torch.zeros(shape3, device=self.device)
        self.angular_rate = torch.zeros(shape3, device=self.device)
        self.previous_action = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        self.applied_action = torch.zeros_like(self.previous_action)
        max_command_latency_s = max(
            config.command_latency_s,
            config.command_latency_s_range[1],
        )
        self.action_history_length = ceil(max_command_latency_s / config.dt) + 2
        self.action_history = torch.zeros(
            (self.num_envs, self.action_history_length, ACTION_DIM),
            device=self.device,
        )
        self.last_detection = torch.zeros((self.num_envs, 5), device=self.device)
        self.last_gate_corners = torch.zeros((self.num_envs, 8), device=self.device)
        self.detection_age = torch.full(
            (self.num_envs,), config.detection_memory_s, device=self.device
        )
        self.live_observation_history = torch.zeros(
            (
                self.num_envs,
                config.observation_history_length,
                TEMPORAL_BASE_OBS_DIM,
            ),
            device=self.device,
        )
        self.corner_observation_history = torch.zeros(
            (
                self.num_envs,
                config.observation_history_length,
                CORNER_BASE_OBS_DIM,
            ),
            device=self.device,
        )
        self.live_observation_history_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.has_taken_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gate_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gates_passed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_progress = torch.zeros(self.num_envs, device=self.device)
        self.episode_start_distance = torch.zeros(self.num_envs, device=self.device)

        self.env_gate_positions = self.base_gate_positions.unsqueeze(0).repeat(
            self.num_envs, 1, 1
        )
        self.mass_scale = torch.ones(self.num_envs, device=self.device)
        self.thrust_scale = torch.ones(self.num_envs, device=self.device)
        self.linear_drag = torch.full(shape3, 0.15, device=self.device)
        self.quadratic_drag = torch.zeros(shape3, device=self.device)
        self.rate_response_gain = torch.ones(shape3, device=self.device)
        self.command_latency = torch.zeros(self.num_envs, device=self.device)
        self.thrust_command = torch.full(
            (self.num_envs,),
            config.thrust_command_center,
            device=self.device,
            dtype=self.dtype,
        )
        self.rate_time_constant = torch.full(
            (self.num_envs,),
            config.rate_time_constant_s,
            device=self.device,
            dtype=self.dtype,
        )
        self.action_response = torch.full(
            (self.num_envs,),
            config.action_response,
            device=self.device,
            dtype=self.dtype,
        )
        self.wind_acceleration = torch.zeros(shape3, device=self.device)

        self.curriculum_progress = 0.0
        self.randomization_scale = 0.0
        self.near_gate_spawn_ratio = config.near_gate_spawn_ratio_start
        self.soft_collisions = config.soft_collision_fraction > 0.0
        self.reset()

    def set_curriculum(self, progress: float) -> None:
        progress = min(max(float(progress), 0.0), 1.0)
        self.curriculum_progress = progress
        self.randomization_scale = progress if self.config.randomization else 0.0
        start = self.config.near_gate_spawn_ratio_start
        end = self.config.near_gate_spawn_ratio_end
        self.near_gate_spawn_ratio = start + (end - start) * progress
        self.soft_collisions = progress < self.config.soft_collision_fraction

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        if env_ids is None:
            env_ids = self.all_env_ids
        self._reset_envs(env_ids)
        return self._observe(), {}

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        count = env_ids.numel()
        self._randomize_track(env_ids)
        self._randomize_dynamics(env_ids)

        self.gate_index[env_ids] = 0
        self.gates_passed[env_ids] = 0
        self.steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.episode_progress[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        self.action_history[env_ids] = 0.0
        self.attitude[env_ids] = 0.0
        self.angular_rate[env_ids] = 0.0
        self.last_detection[env_ids] = 0.0
        self.last_gate_corners[env_ids] = 0.0
        self.detection_age[env_ids] = self.config.detection_memory_s
        self.live_observation_history[env_ids] = 0.0
        self.corner_observation_history[env_ids] = 0.0
        self.live_observation_history_initialized[env_ids] = False
        self.has_taken_off[env_ids] = False

        start = torch.tensor(
            self.config.start_position_m, device=self.device, dtype=self.dtype
        ).expand(count, -1)
        noise_scale = torch.tensor(
            self.config.start_position_noise_m, device=self.device, dtype=self.dtype
        )
        start_noise = self._rand_uniform((count, 3), -1.0, 1.0) * noise_scale
        positions = start + start_noise
        velocities = self._rand_uniform(
            (count, 3),
            -self.config.start_velocity_noise_mps,
            self.config.start_velocity_noise_mps,
        )

        near_mask = self._rand_uniform((count,), 0.0, 1.0) < self.near_gate_spawn_ratio
        near_ids = env_ids[:0]
        if near_mask.any():
            near_ids = env_ids[near_mask]
            near_count = near_ids.numel()
            if self.near_gate_indices is None:
                random_gate = torch.randint(
                    0,
                    self.gate_count,
                    (near_count,),
                    generator=self.generator,
                    device=self.device,
                )
            else:
                if self.near_gate_index_weights is None:
                    gate_choice = torch.randint(
                        0,
                        self.near_gate_indices.numel(),
                        (near_count,),
                        generator=self.generator,
                        device=self.device,
                    )
                else:
                    gate_choice = torch.multinomial(
                        self.near_gate_index_weights,
                        near_count,
                        replacement=True,
                        generator=self.generator,
                    )
                random_gate = self.near_gate_indices[gate_choice]
            self.gate_index[near_ids] = random_gate
            self.gates_passed[near_ids] = random_gate
            gate_pos = self.env_gate_positions[near_ids, random_gate]
            normal = self.gate_normals[random_gate]
            lateral = self.gate_lateral[random_gate]
            vertical = self.gate_vertical[random_gate]
            distance = self._rand_uniform(
                (near_count, 1),
                self.config.near_gate_distance_m[0],
                self.config.near_gate_distance_m[1],
            )
            lateral_offset = self._rand_uniform(
                (near_count, 1),
                self.config.near_gate_lateral_offset_m[0],
                self.config.near_gate_lateral_offset_m[1],
            )
            vertical_offset = self._rand_uniform(
                (near_count, 1),
                self.config.near_gate_vertical_offset_m[0],
                self.config.near_gate_vertical_offset_m[1],
            )
            positions[near_mask] = (
                gate_pos
                - normal * distance
                + lateral * lateral_offset
                + vertical * vertical_offset
            )
            forward_speed = self._rand_uniform(
                (near_count, 1),
                self.config.near_gate_forward_speed_mps[0],
                self.config.near_gate_forward_speed_mps[1],
            )
            lateral_speed = self._rand_uniform(
                (near_count, 1),
                self.config.near_gate_lateral_speed_mps[0],
                self.config.near_gate_lateral_speed_mps[1],
            )
            vertical_speed = self._rand_uniform(
                (near_count, 1),
                self.config.near_gate_vertical_speed_mps[0],
                self.config.near_gate_vertical_speed_mps[1],
            )
            velocities[near_mask] = (
                normal * forward_speed
                + lateral * lateral_speed
                + vertical * vertical_speed
            )
            self.has_taken_off[near_ids] = True

        positions[:, 2].clamp_(min=0.03)
        self.position[env_ids] = positions
        self.velocity[env_ids] = velocities
        if self.config.align_spawn_heading_to_gate:
            target = self.env_gate_positions[env_ids, self.gate_index[env_ids]]
            direction = target - positions
            self.attitude[env_ids, 0:2] = self._rand_uniform(
                (count, 2),
                -self.config.spawn_tilt_noise_rad,
                self.config.spawn_tilt_noise_rad,
            )
            self.attitude[env_ids, 2] = (
                torch.atan2(direction[:, 1], direction[:, 0])
                + self._rand_uniform(
                    (count,),
                    -self.config.spawn_yaw_noise_rad,
                    self.config.spawn_yaw_noise_rad,
                )
            )
        if near_ids.numel() > 0:
            if self.config.near_gate_roll_rad is not None:
                self.attitude[near_ids, 0] = self._rand_uniform(
                    (near_ids.numel(),),
                    self.config.near_gate_roll_rad[0],
                    self.config.near_gate_roll_rad[1],
                )
            if self.config.near_gate_pitch_rad is not None:
                self.attitude[near_ids, 1] = self._rand_uniform(
                    (near_ids.numel(),),
                    self.config.near_gate_pitch_rad[0],
                    self.config.near_gate_pitch_rad[1],
                )
            if self.config.near_gate_yaw_offset_rad is not None:
                self.attitude[near_ids, 2] += self._rand_uniform(
                    (near_ids.numel(),),
                    self.config.near_gate_yaw_offset_rad[0],
                    self.config.near_gate_yaw_offset_rad[1],
                )
            if self.config.near_gate_previous_action is not None:
                previous_action = torch.tensor(
                    self.config.near_gate_previous_action,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.previous_action[near_ids] = previous_action
                self.applied_action[near_ids] = previous_action
                self.action_history[near_ids] = previous_action
                self.angular_rate[near_ids] = (
                    previous_action[1:]
                    * self.command_rate_limits
                    * self.rate_response_gain[near_ids]
                )
            elif self.config.near_gate_previous_action_range is not None:
                limits = torch.tensor(
                    self.config.near_gate_previous_action_range,
                    device=self.device,
                    dtype=self.dtype,
                )
                previous_action = self._rand_uniform(
                    (near_ids.numel(), ACTION_DIM),
                    0.0,
                    1.0,
                )
                previous_action = (
                    limits[:, 0]
                    + previous_action * (limits[:, 1] - limits[:, 0])
                )
                self.previous_action[near_ids] = previous_action
                self.applied_action[near_ids] = previous_action
                self.action_history[near_ids] = previous_action.unsqueeze(1)
                self.angular_rate[near_ids] = (
                    previous_action[:, 1:]
                    * self.command_rate_limits
                    * self.rate_response_gain[near_ids]
                )
        self.episode_start_distance[env_ids] = self._distance_to_active_gate(env_ids)

    def step(
        self, raw_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if raw_action.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"expected action shape {(self.num_envs, ACTION_DIM)}, got {tuple(raw_action.shape)}"
            )
        action = torch.tanh(raw_action)
        action_delta = action - self.previous_action
        previous_position = self.position.clone()
        previous_distance = self._distance_to_active_gate()
        active_gate_before = self.gate_index.clone()
        self.action_history[:, :-1] = self.action_history[:, 1:].clone()
        self.action_history[:, -1] = action
        delayed_action = self._delayed_action()

        ground_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        impact_speed = torch.zeros(self.num_envs, device=self.device)
        sub_dt = self.config.dt / self.config.physics_substeps
        for _ in range(self.config.physics_substeps):
            response = self.action_response.unsqueeze(1)
            self.applied_action += response * (
                delayed_action - self.applied_action
            )
            desired_rate = (
                self.applied_action[:, 1:]
                * self.command_rate_limits
                * self.rate_response_gain
            )
            rate_alpha = torch.clamp(
                sub_dt / self.rate_time_constant, min=0.0, max=1.0
            ).unsqueeze(1)
            self.angular_rate += rate_alpha * (desired_rate - self.angular_rate)
            self.attitude += self.angular_rate * sub_dt
            self.attitude[:, 2] = torch.remainder(self.attitude[:, 2] + pi, 2 * pi) - pi

            physics_attitude = self.attitude
            if self.config.dynamics_model == "measured_ai_gp_v1":
                physics_attitude = self.attitude.clone()
                physics_attitude[:, 1] -= self.config.base_pitch_offset_rad
            rotation = self._body_to_world_rotation(physics_attitude)
            thrust_direction = rotation[:, :, 2]
            if self.config.dynamics_model == "measured_ai_gp_v1":
                thrust_span = torch.where(
                    self.applied_action[:, 0] >= 0.0,
                    self.config.thrust_command_span_up,
                    self.config.thrust_command_span_down,
                )
                self.thrust_command = (
                    self.config.thrust_command_center
                    + self.applied_action[:, 0] * thrust_span
                )
                collective_acceleration = (
                    self.config.thrust_acceleration_bias_mps2
                    + self.config.thrust_acceleration_gain_mps2
                    * self.thrust_command
                ).clamp(min=0.0)
                collective_acceleration *= self.thrust_scale / self.mass_scale
            else:
                self.thrust_command = self.applied_action[:, 0]
                collective_acceleration = (
                    9.81
                    * (
                        1.0
                        + self.config.collective_range_g
                        * self.applied_action[:, 0]
                    ).clamp(min=0.15)
                    * self.thrust_scale
                    / self.mass_scale
                )
            drag_acceleration = (
                self.linear_drag * self.velocity
                + self.quadratic_drag
                * self.velocity.abs()
                * self.velocity
            )
            acceleration = (
                thrust_direction * collective_acceleration.unsqueeze(1)
                + self.gravity_acceleration
                - drag_acceleration
                + self.wind_acceleration
            )
            self.velocity += acceleration * sub_dt
            self.position += self.velocity * sub_dt

            contact_now = self.position[:, 2] < 0.0
            impact_speed = torch.maximum(impact_speed, torch.where(
                contact_now, (-self.velocity[:, 2]).clamp(min=0.0), torch.zeros_like(impact_speed)
            ))
            ground_contact |= contact_now
            if contact_now.any():
                self.position[contact_now, 2] = 0.0
                self.velocity[contact_now, 2] = self.velocity[contact_now, 2].clamp(min=0.0)
                self.velocity[contact_now, :2] *= 0.80

        self.steps += 1
        self.has_taken_off |= self.position[:, 2] > 0.15

        gate_position = self.env_gate_positions[
            self.all_env_ids, active_gate_before
        ]
        gate_normal = self.gate_normals[active_gate_before]
        gate_lateral = self.gate_lateral[active_gate_before]
        gate_vertical = self.gate_vertical[active_gate_before]
        previous_plane = ((previous_position - gate_position) * gate_normal).sum(1)
        current_plane = ((self.position - gate_position) * gate_normal).sum(1)
        crossed_gate_plane = (previous_plane < 0.0) & (current_plane >= 0.0)
        plane_delta = current_plane - previous_plane
        crossing_fraction = torch.where(
            plane_delta.abs() > 1e-6,
            -previous_plane / plane_delta,
            torch.zeros_like(plane_delta),
        ).clamp(0.0, 1.0)
        crossing_position = previous_position + crossing_fraction.unsqueeze(1) * (
            self.position - previous_position
        )
        gate_sample_position = torch.where(
            crossed_gate_plane.unsqueeze(1),
            crossing_position,
            self.position,
        )
        gate_offset = gate_sample_position - gate_position
        gate_lateral_offset = (gate_offset * gate_lateral).sum(1)
        gate_vertical_offset = (gate_offset * gate_vertical).sum(1)
        inside_gate = (
            (gate_lateral_offset.abs() <= self.config.gate_half_width_m)
            & (gate_vertical_offset.abs() <= self.config.gate_half_height_m)
        )
        passed_gate = crossed_gate_plane & inside_gate
        missed_gate = crossed_gate_plane & ~inside_gate
        self.gates_passed += passed_gate.long()
        self.gate_index = torch.minimum(
            self.gate_index + passed_gate.long(),
            torch.full_like(self.gate_index, self.gate_count - 1),
        )
        finished = passed_gate & (active_gate_before == self.gate_count - 1)

        current_distance = torch.linalg.vector_norm(self.position - gate_position, dim=1)
        progress = previous_distance - current_distance
        self.episode_progress += progress
        direction_to_gate = gate_position - previous_position
        direction_to_gate /= torch.linalg.vector_norm(
            direction_to_gate, dim=1, keepdim=True
        ).clamp(min=1e-4)
        forward_speed = (self.velocity * direction_to_gate).sum(1).clamp(min=0.0)

        tilt = torch.linalg.vector_norm(self.attitude[:, :2], dim=1)
        hard_ground_crash = ground_contact & (
            self.has_taken_off
            | (impact_speed > self.config.ground_impact_speed_mps)
        ) & (self.steps > 5)
        tilt_crash = (tilt > self.config.max_tilt_rad) & self.has_taken_off
        collision = hard_ground_crash | tilt_crash
        out_of_bounds = (
            (self.position[:, 2] > self.config.max_altitude_m)
            | (self.position[:, 1].abs() > self.config.max_lateral_m)
            | (self.position[:, 0] < self.config.min_forward_m)
            | (self.position[:, 0] > self.config.max_forward_m)
        )

        projected_gate, _, visible, confidence = self._project_gate(
            update_memory=False
        )
        camera_alignment = torch.exp(
            -self.config.camera_alignment_exponent
            * projected_gate[:, :2].square().sum(1)
        ) * visible.float()
        altitude_penalty_start = (
            self.config.max_altitude_m
            if self.config.soft_altitude_limit_m is None
            else self.config.soft_altitude_limit_m
        )
        reward = (
            self.config.progress_reward * progress
            + self.config.gate_reward * passed_gate.float()
            + self.config.finish_reward * finished.float()
            + self.config.forward_speed_reward * forward_speed
            + self.config.visibility_reward * visible.float() * confidence
            + self.config.camera_alignment_reward * camera_alignment
            + self.config.alive_reward
            - self.config.angular_rate_penalty
            * self.angular_rate.square().sum(1)
            - self.config.action_delta_penalty * action_delta.square().sum(1)
            - self.config.thrust_saturation_penalty
            * torch.relu(self.applied_action[:, 0].abs() - 0.90).square()
            - self.config.vertical_speed_penalty
            * torch.relu(self.velocity[:, 2].abs() - 2.0).square()
            - self.config.altitude_penalty
            * torch.relu(self.position[:, 2] - altitude_penalty_start).square()
            - self.config.low_altitude_penalty
            * torch.relu(self.config.soft_floor_altitude_m - self.position[:, 2]).square()
            - self.config.gate_altitude_error_penalty
            * gate_vertical_offset.square()
            - self.config.gate_lateral_error_penalty
            * gate_lateral_offset.square()
            - self.config.collision_penalty * collision.float()
            - self.config.out_of_bounds_penalty * out_of_bounds.float()
            - self.config.missed_gate_penalty * missed_gate.float()
        )
        self.episode_return += reward
        self.previous_action.copy_(action)

        if self.soft_collisions:
            recover = collision & ~out_of_bounds & ~finished
            if self.config.terminate_on_missed_gate:
                recover &= ~missed_gate
            if recover.any():
                self.attitude[recover, :2].clamp_(-0.65, 0.65)
                self.angular_rate[recover] *= 0.20
                self.velocity[recover] *= 0.50
                self.position[recover, 2].clamp_(min=0.05)
            terminated = finished | out_of_bounds
        else:
            terminated = finished | out_of_bounds | collision
        if self.config.terminate_on_missed_gate:
            terminated |= missed_gate
        truncated = self.steps >= self.config.max_episode_steps
        done = terminated | truncated
        info = {
            "done": done.clone(),
            "success": finished.clone(),
            "collision": collision.clone(),
            "out_of_bounds": out_of_bounds.clone(),
            "passed_gate": passed_gate.clone(),
            "missed_gate": missed_gate.clone(),
            "active_gate_index": active_gate_before.clone(),
            "gate_plane_offset": current_plane.clone(),
            "gate_lateral_offset": gate_lateral_offset.clone(),
            "gate_vertical_offset": gate_vertical_offset.clone(),
            "gates_passed": self.gates_passed.clone(),
            "episode_return": self.episode_return.clone(),
            "distance_reduction": self.episode_progress.clone(),
            "position": self.position.clone(),
            "velocity": self.velocity.clone(),
            "attitude": self.attitude.clone(),
            "angular_rate": self.angular_rate.clone(),
            "applied_action": self.applied_action.clone(),
            "thrust_command": self.thrust_command.clone(),
        }
        if done.any():
            self._reset_envs(torch.where(done)[0])
        observation = self._observe()
        return observation, reward, terminated, truncated, info

    def _observe(self) -> torch.Tensor:
        rotation = self._body_to_world_rotation(self.attitude)
        body_velocity = torch.bmm(
            rotation.transpose(1, 2), self.velocity.unsqueeze(2)
        ).squeeze(2)
        gravity_world = self.gravity_unit.expand(
            self.num_envs, -1
        )
        gravity_body = torch.bmm(
            rotation.transpose(1, 2), gravity_world.unsqueeze(2)
        ).squeeze(2)
        detection, gate_corners, _, confidence = self._project_gate(
            update_memory=True
        )
        temporal_base_observation = self._build_temporal_base_observation(
            body_velocity, gravity_body, detection, confidence
        )
        corner_base_observation = self._build_corner_base_observation(
            body_velocity,
            gravity_body,
            detection,
            gate_corners,
            confidence,
        )
        self._update_live_observation_history(
            temporal_base_observation, corner_base_observation
        )

        active_gate_position = self._active_gate_position()
        gate_relative_world = active_gate_position - self.position
        gate_relative_body = torch.bmm(
            rotation.transpose(1, 2), gate_relative_world.unsqueeze(2)
        ).squeeze(2)
        if self.config.actor_observation_mode == "swift_teacher":
            corners_world = self._active_gate_corners_world()
            corners_body = torch.bmm(
                rotation.transpose(1, 2),
                (corners_world - self.position.unsqueeze(1)).transpose(1, 2),
            ).transpose(1, 2)
            actor_obs = torch.cat(
                (
                    self.position / self.position_observation_scale,
                    self.velocity / VELOCITY_SCALE_MPS,
                    rotation.reshape(self.num_envs, 9),
                    (corners_body / 10.0).reshape(self.num_envs, 12),
                    self.previous_action,
                ),
                dim=1,
            )
        elif self.config.actor_observation_mode == "structured_teacher_v2":
            next_gate_index = torch.minimum(
                self.gate_index + 1,
                torch.full_like(self.gate_index, self.gate_count - 1),
            )
            next_gate_position = self.env_gate_positions[
                self.all_env_ids, next_gate_index
            ]
            next_gate_relative_body = torch.bmm(
                rotation.transpose(1, 2),
                (next_gate_position - self.position).unsqueeze(2),
            ).squeeze(2)
            active_gate_normal_body = torch.bmm(
                rotation.transpose(1, 2),
                self.gate_normals[self.gate_index].unsqueeze(2),
            ).squeeze(2)
            next_gate_normal_body = torch.bmm(
                rotation.transpose(1, 2),
                self.gate_normals[next_gate_index].unsqueeze(2),
            ).squeeze(2)
            gate_fraction = (
                self.gate_index.float() / max(self.gate_count - 1, 1)
            ).unsqueeze(1)
            actor_obs = torch.cat(
                (
                    gate_relative_body / 30.0,
                    active_gate_normal_body,
                    next_gate_relative_body / 30.0,
                    next_gate_normal_body,
                    body_velocity / VELOCITY_SCALE_MPS,
                    gravity_body,
                    self.angular_rate / self.observation_rate_scales,
                    self.previous_action,
                    gate_fraction,
                ),
                dim=1,
            )
        elif self.config.actor_observation_mode == "live_features":
            actor_obs = self._build_live_actor_observation(
                body_velocity, gravity_body, detection, confidence
            )
        elif self.config.actor_observation_mode == "live_features_temporal":
            actor_obs = self.live_observation_history.flatten(start_dim=1)
        elif self.config.actor_observation_mode == "live_features_recurrent":
            actor_obs = temporal_base_observation
        elif self.config.actor_observation_mode == "live_features_corners_temporal":
            actor_obs = self.corner_observation_history.flatten(start_dim=1)
        elif self.config.actor_observation_mode == "live_features_corners_recurrent":
            actor_obs = corner_base_observation
        else:
            actor_obs = self._build_motion_observation()

        distance = torch.linalg.vector_norm(gate_relative_world, dim=1, keepdim=True)
        gate_fraction = (
            self.gate_index.float() / max(self.gate_count - 1, 1)
        ).unsqueeze(1)
        privileged = torch.cat(
            (
                gate_relative_body / 10.0,
                distance / 10.0,
                self.position / self.position_observation_scale,
                self.velocity / VELOCITY_SCALE_MPS,
                self.attitude / pi,
                gate_fraction,
            ),
            dim=1,
        )
        observation = torch.cat((actor_obs, privileged), dim=1)
        if observation.shape[1] != self.observation_dim:
            raise RuntimeError(
                f"observation contract mismatch: expected {self.observation_dim}, "
                f"got {observation.shape[1]}"
            )
        return observation

    def live_actor_observation(self) -> torch.Tensor:
        """Return the deployable observation without advancing sensor memory."""

        if self.live_observation_mode == "live_features_temporal":
            return self.live_observation_history.flatten(start_dim=1)
        if self.live_observation_mode == "live_features_recurrent":
            return self.live_observation_history[:, -1]
        if self.live_observation_mode == "live_features_corners_temporal":
            return self.corner_observation_history.flatten(start_dim=1)
        if self.live_observation_mode == "live_features_corners_recurrent":
            return self.corner_observation_history[:, -1]
        if self.live_observation_mode == "live_features_motion":
            return self._build_motion_observation()

        rotation = self._body_to_world_rotation(self.attitude)
        body_velocity = torch.bmm(
            rotation.transpose(1, 2), self.velocity.unsqueeze(2)
        ).squeeze(2)
        gravity_world = self.gravity_unit.expand(self.num_envs, -1)
        gravity_body = torch.bmm(
            rotation.transpose(1, 2), gravity_world.unsqueeze(2)
        ).squeeze(2)
        detection, _, _, confidence = self._project_gate(update_memory=False)
        return self._build_live_actor_observation(
            body_velocity, gravity_body, detection, confidence
        )

    def _build_live_actor_observation(
        self,
        body_velocity: torch.Tensor,
        gravity_body: torch.Tensor,
        detection: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        actor_observation = torch.cat(
            (
                (body_velocity / VELOCITY_SCALE_MPS).clamp(-2.0, 2.0),
                gravity_body,
                self.angular_rate / self.observation_rate_scales,
                detection[:, :2].clamp(-1.5, 1.5),
                detection[:, 4:5].clamp(0.0, 1.0),
                confidence.unsqueeze(1),
                (self.detection_age / self.config.detection_memory_s)
                .clamp(0.0, 1.0)
                .unsqueeze(1),
                self.previous_action,
            ),
            dim=1,
        )
        if actor_observation.shape != (self.num_envs, ACTOR_OBS_DIM):
            raise RuntimeError(
                "live actor observation contract mismatch: "
                f"expected {(self.num_envs, ACTOR_OBS_DIM)}, "
                f"got {tuple(actor_observation.shape)}"
            )
        return actor_observation

    def _build_temporal_base_observation(
        self,
        body_velocity: torch.Tensor,
        gravity_body: torch.Tensor,
        detection: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        base_observation = torch.cat(
            (
                (body_velocity / VELOCITY_SCALE_MPS).clamp(-2.0, 2.0),
                gravity_body,
                self.angular_rate / self.observation_rate_scales,
                detection[:, :2].clamp(-1.5, 1.5),
                detection[:, 2:5].clamp(0.0, 1.0),
                confidence.unsqueeze(1),
                (self.detection_age / self.config.detection_memory_s)
                .clamp(0.0, 1.0)
                .unsqueeze(1),
                self.previous_action,
            ),
            dim=1,
        )
        if base_observation.shape != (self.num_envs, TEMPORAL_BASE_OBS_DIM):
            raise RuntimeError(
                "temporal base observation contract mismatch: "
                f"expected {(self.num_envs, TEMPORAL_BASE_OBS_DIM)}, "
                f"got {tuple(base_observation.shape)}"
            )
        return base_observation

    def _build_corner_base_observation(
        self,
        body_velocity: torch.Tensor,
        gravity_body: torch.Tensor,
        detection: torch.Tensor,
        gate_corners: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        base_observation = torch.cat(
            (
                (body_velocity / VELOCITY_SCALE_MPS).clamp(-2.0, 2.0),
                gravity_body,
                self.angular_rate / self.observation_rate_scales,
                detection[:, :5].clamp(-1.5, 1.5),
                gate_corners.clamp(-1.5, 1.5),
                (confidence > 0.0).float().unsqueeze(1),
                confidence.unsqueeze(1),
                (self.detection_age / self.config.detection_memory_s)
                .clamp(0.0, 1.0)
                .unsqueeze(1),
                self.previous_action,
            ),
            dim=1,
        )
        if base_observation.shape != (self.num_envs, CORNER_BASE_OBS_DIM):
            raise RuntimeError(
                "corner base observation contract mismatch: "
                f"expected {(self.num_envs, CORNER_BASE_OBS_DIM)}, "
                f"got {tuple(base_observation.shape)}"
            )
        return base_observation

    def _build_motion_observation(self) -> torch.Tensor:
        current = self.live_observation_history[:, -1]
        previous = self.live_observation_history[:, -2]
        center_delta = ((current[:, 9:11] - previous[:, 9:11]) / 0.25).clamp(
            -2.0, 2.0
        )
        size_delta = (
            torch.log(current[:, 11:13].clamp(min=1e-4))
            - torch.log(previous[:, 11:13].clamp(min=1e-4))
        ).clamp(-2.0, 2.0)
        observation = torch.cat((current, center_delta, size_delta), dim=1)
        if observation.shape != (self.num_envs, MOTION_OBS_DIM):
            raise RuntimeError("motion observation contract mismatch")
        return observation

    def _update_live_observation_history(
        self,
        current_observation: torch.Tensor,
        current_corner_observation: torch.Tensor,
    ) -> None:
        initialized = self.live_observation_history_initialized
        if initialized.any():
            self.live_observation_history[initialized, :-1] = (
                self.live_observation_history[initialized, 1:].clone()
            )
            self.live_observation_history[initialized, -1] = current_observation[
                initialized
            ]
            self.corner_observation_history[initialized, :-1] = (
                self.corner_observation_history[initialized, 1:].clone()
            )
            self.corner_observation_history[initialized, -1] = (
                current_corner_observation[initialized]
            )
        uninitialized = ~initialized
        if uninitialized.any():
            self.live_observation_history[uninitialized] = current_observation[
                uninitialized
            ].unsqueeze(1)
            self.corner_observation_history[uninitialized] = (
                current_corner_observation[uninitialized].unsqueeze(1)
            )
            self.live_observation_history_initialized[uninitialized] = True

    def _project_gate(
        self, *, update_memory: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rotation = self._body_to_world_rotation(self.attitude)
        relative_world = self._active_gate_position() - self.position
        relative_body = torch.bmm(
            rotation.transpose(1, 2), relative_world.unsqueeze(2)
        ).squeeze(2)
        depth = relative_body[:, 0]
        tan_h = self.tan_horizontal_half_fov
        tan_v = self.tan_vertical_half_fov
        safe_depth = depth.clamp(min=0.05)
        center_x = -relative_body[:, 1] / (safe_depth * tan_h)
        center_y = -relative_body[:, 2] / (safe_depth * tan_v)
        width = (
            self.config.gate_half_width_m / (safe_depth * tan_h)
        ).clamp(0.0, 1.0)
        height = (
            self.config.gate_half_height_m / (safe_depth * tan_v)
        ).clamp(0.0, 1.0)
        area = (width * height).clamp(0.0, 1.0)
        visible = (
            (depth > 0.05)
            & (center_x.abs() <= 1.0)
            & (center_y.abs() <= 1.0)
        )
        confidence = (
            torch.exp(-0.35 * (center_x.square() + center_y.square()))
            * visible.float()
        )
        corners_world = self._active_gate_corners_world()
        corners_body = torch.bmm(
            rotation.transpose(1, 2),
            (corners_world - self.position.unsqueeze(1)).transpose(1, 2),
        ).transpose(1, 2)
        corner_depth = corners_body[:, :, 0].clamp(min=0.05)
        corner_x = -corners_body[:, :, 1] / (corner_depth * tan_h)
        corner_y = -corners_body[:, :, 2] / (corner_depth * tan_v)
        fresh_corners = torch.stack((corner_x, corner_y), dim=2).reshape(
            self.num_envs, 8
        )

        if self.config.randomization and self.randomization_scale > 0.0:
            noise_std = self.config.vision_noise_std * self.randomization_scale
            center_x += self._rand_normal((self.num_envs,)) * noise_std
            center_y += self._rand_normal((self.num_envs,)) * noise_std
            fresh_corners += self._rand_normal(
                (self.num_envs, 8)
            ) * noise_std
            size_noise = (
                1.0 + self._rand_normal((self.num_envs,)) * noise_std
            ).clamp(0.6, 1.4)
            width = (width * size_noise).clamp(0.0, 1.0)
            height = (height * size_noise).clamp(0.0, 1.0)
            area = (width * height).clamp(0.0, 1.0)
            dropout = self._rand_uniform((self.num_envs,), 0.0, 1.0) < (
                self.config.vision_dropout_probability * self.randomization_scale
            )
            visible &= ~dropout
            confidence *= visible.float()

        fresh_detection = torch.stack(
            (center_x, center_y, width, height, area), dim=1
        )
        if update_memory:
            if visible.any():
                self.last_detection[visible] = fresh_detection[visible]
                self.last_gate_corners[visible] = fresh_corners[visible]
            self.detection_age = torch.where(
                visible,
                torch.zeros_like(self.detection_age),
                (self.detection_age + self.config.dt).clamp(
                    max=self.config.detection_memory_s
                ),
            )
        age_decay = torch.exp(
            -self.detection_age / max(self.config.detection_memory_s, 1e-3)
        )
        tracked_confidence = torch.where(visible, confidence, confidence * age_decay)
        detection = torch.where(
            visible.unsqueeze(1), fresh_detection, self.last_detection
        )
        gate_corners = torch.where(
            visible.unsqueeze(1), fresh_corners, self.last_gate_corners
        )
        return detection, gate_corners, visible, tracked_confidence

    def _active_gate_position(self) -> torch.Tensor:
        return self.env_gate_positions[
            self.all_env_ids, self.gate_index
        ]

    def _active_gate_corners_world(self) -> torch.Tensor:
        gate_index = self.gate_index
        center = self._active_gate_position()
        lateral = self.gate_lateral[gate_index] * self.config.gate_half_width_m
        vertical = self.gate_vertical[gate_index] * self.config.gate_half_height_m
        return torch.stack(
            (
                center - lateral + vertical,
                center + lateral + vertical,
                center + lateral - vertical,
                center - lateral - vertical,
            ),
            dim=1,
        )

    def _distance_to_active_gate(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            return torch.linalg.vector_norm(
                self._active_gate_position() - self.position, dim=1
            )
        gate_position = self.env_gate_positions[env_ids, self.gate_index[env_ids]]
        return torch.linalg.vector_norm(gate_position - self.position[env_ids], dim=1)

    def _randomize_track(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        jitter_scale = torch.tensor(
            self.config.gate_jitter_m, device=self.device, dtype=self.dtype
        )
        jitter = self._rand_uniform((count, self.gate_count, 3), -1.0, 1.0)
        jitter *= jitter_scale * self.randomization_scale
        self.env_gate_positions[env_ids] = self.base_gate_positions.unsqueeze(0) + jitter

    def _randomize_dynamics(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        self.mass_scale[env_ids] = self._scaled_random_range(
            count, self.config.mass_scale_range, 1.0
        )
        self.thrust_scale[env_ids] = self._scaled_random_range(
            count, self.config.thrust_scale_range, 1.0
        )
        if self.config.dynamics_model == "measured_ai_gp_v1":
            drag_scale = self._scaled_random_range(
                count, self.config.drag_scale_range, 1.0
            ).unsqueeze(1)
            base_linear_drag = torch.tensor(
                self.config.linear_drag_xyz,
                device=self.device,
                dtype=self.dtype,
            )
            base_quadratic_drag = torch.tensor(
                self.config.quadratic_drag_xyz,
                device=self.device,
                dtype=self.dtype,
            )
            self.linear_drag[env_ids] = drag_scale * base_linear_drag
            self.quadratic_drag[env_ids] = drag_scale * base_quadratic_drag
            rate_gain_scale = self._scaled_random_range(
                count,
                self.config.rate_response_gain_scale_range,
                1.0,
            ).unsqueeze(1)
            base_rate_gain = torch.tensor(
                self.config.rate_response_gain,
                device=self.device,
                dtype=self.dtype,
            )
            self.rate_response_gain[env_ids] = (
                rate_gain_scale * base_rate_gain
            )
            self.command_latency[env_ids] = self._scaled_random_range(
                count,
                self.config.command_latency_s_range,
                self.config.command_latency_s,
            )
        else:
            scalar_drag = self._scaled_random_range(
                count, self.config.linear_drag_range, 0.15
            )
            self.linear_drag[env_ids] = scalar_drag.unsqueeze(1)
            self.quadratic_drag[env_ids] = 0.0
            self.rate_response_gain[env_ids] = 1.0
            self.command_latency[env_ids] = 0.0
        self.rate_time_constant[env_ids] = self._scaled_random_range(
            count,
            self.config.rate_time_constant_s_range,
            self.config.rate_time_constant_s,
        )
        self.action_response[env_ids] = self._scaled_random_range(
            count,
            self.config.action_response_range,
            self.config.action_response,
        )
        wind = self._rand_uniform((count, 3), -1.0, 1.0)
        wind[:, 2] *= 0.35
        self.wind_acceleration[env_ids] = (
            wind * self.config.wind_accel_mps2 * self.randomization_scale
        )

    def _delayed_action(self) -> torch.Tensor:
        delay_steps = self.command_latency / self.config.dt
        recent_steps = torch.floor(delay_steps).long()
        fraction = delay_steps - recent_steps.float()
        recent_index = (
            self.action_history_length - 1 - recent_steps
        ).clamp(min=0)
        older_index = (recent_index - 1).clamp(min=0)
        row_index = self.all_env_ids
        recent = self.action_history[row_index, recent_index]
        older = self.action_history[row_index, older_index]
        return recent + fraction.unsqueeze(1) * (older - recent)

    def _scaled_random_range(
        self, count: int, limits: tuple[float, float], nominal: float
    ) -> torch.Tensor:
        sample = self._rand_uniform((count,), limits[0], limits[1])
        return nominal + (sample - nominal) * self.randomization_scale

    def _rand_uniform(
        self, shape: tuple[int, ...], low: float, high: float
    ) -> torch.Tensor:
        return low + (high - low) * torch.rand(
            shape, device=self.device, generator=self.generator
        )

    def _rand_normal(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, device=self.device, generator=self.generator)

    @staticmethod
    def _body_to_world_rotation(attitude: torch.Tensor) -> torch.Tensor:
        roll = attitude[:, 0]
        pitch = -attitude[:, 1]  # AI-GP convention: negative pitch rate is forward.
        yaw = attitude[:, 2]
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        return torch.stack(
            (
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
                -sp,
                cp * sr,
                cp * cr,
            ),
            dim=1,
        ).reshape(-1, 3, 3)

    @staticmethod
    def _build_gate_normals(
        gates: torch.Tensor,
        *,
        upright: bool = False,
    ) -> torch.Tensor:
        previous = torch.cat(
            (
                gates[:1]
                - torch.tensor((4.0, 0.0, 0.0), device=gates.device),
                gates[:-1],
            )
        )
        normals = gates - previous
        if upright:
            normals[:, 2] = 0.0
        return normals / torch.linalg.vector_norm(normals, dim=1, keepdim=True).clamp(min=1e-6)

    @staticmethod
    def _build_gate_bases(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        up = torch.tensor((0.0, 0.0, 1.0), device=normals.device).expand_as(normals)
        lateral = torch.linalg.cross(up, normals, dim=1)
        fallback = torch.tensor((0.0, 1.0, 0.0), device=normals.device).expand_as(normals)
        lateral_norm = torch.linalg.vector_norm(lateral, dim=1, keepdim=True)
        lateral = torch.where(lateral_norm > 1e-4, lateral / lateral_norm.clamp(min=1e-6), fallback)
        vertical = torch.linalg.cross(normals, lateral, dim=1)
        vertical /= torch.linalg.vector_norm(vertical, dim=1, keepdim=True).clamp(min=1e-6)
        return lateral, vertical
