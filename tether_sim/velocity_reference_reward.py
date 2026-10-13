"""Reward17: keep reward16 and score horizontal motion toward the hover center.

The reference is used only to calculate reward. It never supplies motor actions
or changes physics. A gain of 1 per second maps carrier displacement in metres
into an inward reference velocity, capped at 0.5 metres per second. Vertical
speed retains its reward16 cost. Spatial distance, ground, upright, boundary and
failure terms retain reward16 definitions; angular rates remain unpenalized.
"""
import torch
from .fine_zone_reward import soft_absolute


def horizontal_velocity_reference(displacement):
    """Inward XY velocity in m/s, with a zero-safe 0.5 m/s vector-norm cap."""
    reference = -displacement * 1.  # Metres times a gain of 1 per second.
    norm = torch.linalg.vector_norm(reference, dim=-1, keepdim=True)
    return reference * (.5 / norm.clamp_min(.5))


def velocity_reference_reward(positions, velocity, orientation, center, failed):
    """Return reward, true square-boundary escape and negative-band masks."""
    displacement = positions[:, 0, :2] - center[:, :2]
    extent = displacement.abs().amax(1)
    escaped = extent >= 2.
    band = (extent > 1.) & ~escaped
    height = positions[:, 0, 2]
    reference = horizontal_velocity_reference(displacement)
    horizontal_error = velocity[:, 0, :2] - reference
    speed_cost = 2 * (horizontal_error.square().sum(1)
                      + velocity[:, 0, 2].square())
    cost = (4 * soft_absolute(height - .8)
            + soft_absolute(displacement[:, 0])
            + soft_absolute(displacement[:, 1])
            + speed_cost)
    airborne = ((height - .05) / .25).clamp(0, 1)
    upright = 1 - 2 * orientation[:, 1:3].square().sum(1)
    core = .95 * airborne / (1 + cost) + .05 * upright
    ground_cost = .8 * (height < .15)
    inside_reward = (1 - extent) * core - ground_cost
    band_reward = -(extent - 1) - ground_cost
    reward = torch.where(band, band_reward, inside_reward)
    reward = torch.where(escaped | (failed != 0), -torch.ones_like(reward), reward)
    return reward.float(), escaped, band
