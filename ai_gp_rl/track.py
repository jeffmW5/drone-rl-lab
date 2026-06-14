"""Measured AI Grand Prix track geometry and coordinate transforms."""

from __future__ import annotations

from collections.abc import Sequence


AI_GP_TRACK_NAME = "ai_gp_six_gate"
AI_GP_GATE_SIZE_M = 2.72
AI_GP_TRACK_GROUND_CLEARANCE_M = 1.5

# Simulator telemetry uses NED: X north, Y east, Z down.
AI_GP_TRACK_GATES_NED: tuple[tuple[float, float, float], ...] = (
    (-23.2979679107666, -0.39990234375, -0.03195800632238388),
    (-46.89374923706055, -2.499990224838257, 5.068041801452637),
    (-74.59375, 1.2000097036361694, 13.668041229248047),
    (-111.49374389648438, -5.099989891052246, 24.56804084777832),
    (-135.49374389648438, -0.7999902367591858, 25.355653762817383),
    (-159.19374084472656, -4.399990081787109, 25.968040466308594),
)


def ned_vector_to_surrogate(
    vector_ned: Sequence[float],
) -> tuple[float, float, float]:
    """Map a NED vector into the surrogate's forward-left-up world frame.

    The AI-GP course advances toward decreasing north, so surrogate +X is
    north-negative. East maps directly to +Y to preserve the measured roll
    command convention, and NED down maps to surrogate -Z.
    """

    north, east, down = (float(value) for value in vector_ned)
    return -north, east, -down


def ned_position_to_surrogate(
    position_ned_m: Sequence[float],
    *,
    altitude_offset_m: float,
) -> tuple[float, float, float]:
    """Map a NED position into the surrogate world with an explicit Z offset."""

    x, y, z = ned_vector_to_surrogate(position_ned_m)
    return x, y, z + float(altitude_offset_m)


def ai_gp_track_altitude_offset_m(
    ground_clearance_m: float = AI_GP_TRACK_GROUND_CLEARANCE_M,
) -> float:
    """Return the offset that places the lowest gate above the ground plane."""

    if ground_clearance_m <= 0.0:
        raise ValueError("ground_clearance_m must be positive")
    lowest_gate_down_m = max(position[2] for position in AI_GP_TRACK_GATES_NED)
    return lowest_gate_down_m + float(ground_clearance_m)


def ai_gp_track_surrogate_positions(
    ground_clearance_m: float = AI_GP_TRACK_GROUND_CLEARANCE_M,
) -> tuple[tuple[float, float, float], ...]:
    """Return the six measured gate centers in surrogate coordinates."""

    altitude_offset_m = ai_gp_track_altitude_offset_m(ground_clearance_m)
    return tuple(
        ned_position_to_surrogate(
            position_ned_m,
            altitude_offset_m=altitude_offset_m,
        )
        for position_ned_m in AI_GP_TRACK_GATES_NED
    )


def ai_gp_track_gates_ned_with_dimensions(
) -> dict[int, tuple[float, float, float, float, float]]:
    """Return the live-runner gate payload from the canonical track definition."""

    return {
        gate_index: (*position_ned_m, AI_GP_GATE_SIZE_M, AI_GP_GATE_SIZE_M)
        for gate_index, position_ned_m in enumerate(AI_GP_TRACK_GATES_NED)
    }
