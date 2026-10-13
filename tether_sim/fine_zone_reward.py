"""Reward16: carrier hover with metre distances resolved below a centimetre.

Pure reward calculation; no controller or physics update. The square XY limits
remain 1 m for the hover zone and 2 m for terminal failure. The 1 cm soft absolute
keeps height and horizontal distance costs continuous at the target. Airborne
rewards are continuous across the 1 m and 2 m boundaries. Ground rewards may still
clip below -1 in the band; this version removes the near-hover upper-clip plateau,
not every clipping effect. No angular-rate penalty is applied.
"""
import torch


def soft_absolute(distance):
    """Smooth absolute distance in metres, with a 0.01 m transition scale."""
    return torch.sqrt(distance.square() + .01**2) - .01


def fine_zone_reward(positions, velocity, orientation, center, failed):
    """Return reward, true square-boundary escape and negative-band masks."""
    displacement = positions[:, 0, :2] - center[:, :2]
    extent = displacement.abs().amax(1)
    escaped = extent >= 2.
    band = (extent > 1.) & ~escaped
    height = positions[:, 0, 2]
    cost = (4 * soft_absolute(height - .8)
            + soft_absolute(displacement[:, 0])
            + soft_absolute(displacement[:, 1])
            + 2 * velocity[:, 0].square().sum(1))
    airborne = ((height - .05) / .25).clamp(0, 1)
    upright = 1 - 2 * orientation[:, 1:3].square().sum(1)
    core = .95 * airborne / (1 + cost) + .05 * upright
    ground_cost = .8 * (height < .15)
    inside_reward = (1 - extent) * core - ground_cost
    band_reward = -(extent - 1) - ground_cost
    reward = torch.where(band, band_reward, inside_reward)
    reward = torch.where(escaped | (failed != 0), -torch.ones_like(reward), reward)
    return reward.float(), escaped, band
