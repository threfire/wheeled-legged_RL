"""Fast, batched lookup for the Dogleg closed-chain inverse kinematics."""

from pathlib import Path

import numpy as np
import torch

from agent_world import AssetPath


_LUT_PATH = Path(AssetPath) / "usd_files/dogleg_floating/configuration/dogleg_ik_lut.npz"
_LUT = None


def _load_lut(device: torch.device):
    global _LUT
    if _LUT is None:
        with np.load(_LUT_PATH) as data:
            _LUT = (
                torch.as_tensor(data["lengths"], dtype=torch.float32),
                torch.as_tensor(data["angles"], dtype=torch.float32),
                torch.as_tensor(data["joint_pos"], dtype=torch.float32),
            )
    return tuple(value.to(device=device) for value in _LUT)


def dogleg_inverse_kinematics(leg_length: torch.Tensor, leg_angle: torch.Tensor) -> torch.Tensor:
    """Map per-side targets to reset joints in V14 order.

    Inputs have shape (num_envs, 2), with sides ordered left/right. The output
    shape is (num_envs, 2, 6): front1, rear1, front2, front3, front4, rear2.
    """
    if leg_length.shape != leg_angle.shape or leg_length.ndim != 2 or leg_length.shape[1] != 2:
        raise ValueError(
            f"Expected matching (num_envs, 2) inputs; got {tuple(leg_length.shape)} "
            f"and {tuple(leg_angle.shape)}"
        )
    lengths, angles, table = _load_lut(leg_length.device)
    if (
        torch.any(leg_length < lengths[0] - 1.0e-6)
        or torch.any(leg_length > lengths[-1] + 1.0e-6)
        or torch.any(leg_angle < angles[0] - 1.0e-6)
        or torch.any(leg_angle > angles[-1] + 1.0e-6)
    ):
        raise ValueError(
            "Dogleg IK target is outside its validated table: "
            f"length=[{lengths[0].item():.3f}, {lengths[-1].item():.3f}] m, "
            f"angle=[{torch.rad2deg(angles[0]).item():.1f}, "
            f"{torch.rad2deg(angles[-1]).item():.1f}] deg"
        )

    length = leg_length.clamp(lengths[0], lengths[-1])
    angle = leg_angle.clamp(angles[0], angles[-1])
    i1 = torch.searchsorted(lengths, length.contiguous(), right=True).clamp(1, len(lengths) - 1)
    j1 = torch.searchsorted(angles, angle.contiguous(), right=True).clamp(1, len(angles) - 1)
    i0, j0 = i1 - 1, j1 - 1
    wl = (length - lengths[i0]) / (lengths[i1] - lengths[i0])
    wa = (angle - angles[j0]) / (angles[j1] - angles[j0])

    side_values = []
    for side in range(2):
        q00 = table[side, i0[:, side], j0[:, side]]
        q10 = table[side, i1[:, side], j0[:, side]]
        q01 = table[side, i0[:, side], j1[:, side]]
        q11 = table[side, i1[:, side], j1[:, side]]
        side_values.append(
            (1.0 - wl[:, side, None]) * (1.0 - wa[:, side, None]) * q00
            + wl[:, side, None] * (1.0 - wa[:, side, None]) * q10
            + (1.0 - wl[:, side, None]) * wa[:, side, None] * q01
            + wl[:, side, None] * wa[:, side, None] * q11
        )
    return torch.stack(side_values, dim=1)
