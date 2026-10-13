"""Versioned reset orientations for a pure learned recovery curriculum.

Reset 6 changes only warm-lane orientation relative to reset 5: each odd lane
gets one signed 2-degree roll or pitch tilt. No feedback control, rates, motor
compensation or random draws are introduced.
"""
import math
import torch


def initial_orientations(start, version):
    """Return N x 7 FP32 quaternions/body rates on start.device.

    Versions 5 and 6 share identity ground lanes. Version 5 also keeps all warm
    lanes at identity; version 6 cycles +roll, -roll, +pitch, -pitch across odd
    lanes. Geometry and motor initialization are owned by the task, unchanged.
    """
    if version not in (5, 6):
        raise ValueError('Recovery orientations require reset version 5 or 6')
    if start.ndim != 3 or tuple(start.shape[1:]) != (14, 3) or start.shape[0] < 1:
        raise ValueError('start must have shape (N, 14, 3), with N positive')
    orientation = torch.zeros(start.shape[0], 7, device=start.device, dtype=torch.float32)
    orientation[:, 0] = 1
    if version == 5:
        return orientation
    warm_lanes = torch.arange(1, start.shape[0], 2, device=start.device)
    cases = torch.arange(len(warm_lanes), device=start.device) % 4
    axes = 1 + (cases >= 2).to(torch.long)
    half_angle = math.radians(2) / 2
    sine = math.sin(half_angle)
    signs = torch.where(cases % 2 == 0, sine, -sine)
    orientation[warm_lanes, 0] = math.cos(half_angle)
    orientation[warm_lanes, axes] = signs
    return orientation
