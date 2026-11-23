#!/usr/bin/env python3
"""Run a Dogleg RSL-RL Actor checkpoint in MuJoCo.

This is a sim-to-sim smoke test.  It reproduces the 35-D policy observation
layout used by the current Dogleg run and the 6-D action decoding, while using
MuJoCo PD torques in place of Isaac Lab actuator models.  It is intended to
check the calibrated neutral pose and basic standing/locomotion transfer; it
does not claim bit-identical behavior to PhysX (contact and delay models differ).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT.parent / "wheeled-legged_RL_printk" / "狗腿子总装_MuJoCo_模型更新版" / "狗腿子总装_description" / "dogleg_floating.xml"
DEFAULT_CHECKPOINT = ROOT / "logs/rsl_rl/dogleg_flat_direct/2026-11-23_23-14-11_dogleg_neutral_resume/model_1500.pt"

# Solved from the Dogleg XML closed-chain geometry to match the available
# V14.2 neutral wheel vector (leg length about 0.125 m, angle about 3.03 deg).
NEUTRAL_DEG = {
    "left_hip_thigh": 25.8492016830,
    "right_hip_thigh": -25.8492016830,
    "left_hip_upper": -36.9696432378,
    "right_hip_upper": 36.9696432379,
    "left_knee": -39.7462030698,
    "right_knee": 39.7462030698,
    "left_thigh_ternary": -39.7462030699,
    "right_thigh_ternary": 39.7462030700,
    "left_ternary_coupler": 39.7462030699,
    "right_ternary_coupler": -39.7462030699,
    "left_upper_rocker": 39.7462030699,
    "right_upper_rocker": -39.7462030700,
}

# This is the deployment/training contract order:
# left_front1, left_rear1, right_front1, right_rear1.
ACTIVE_LEG = ["left_hip_thigh", "left_hip_upper", "right_hip_thigh", "right_hip_upper"]
PASSIVE_LEG = [
    "left_knee", "right_knee", "left_thigh_ternary", "right_thigh_ternary",
    "left_ternary_coupler", "right_ternary_coupler", "left_upper_rocker", "right_upper_rocker",
]
WHEELS = ["left_wheel", "right_wheel"]


class Actor(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dict["actor.0.weight"].shape[1], state_dict["actor.0.weight"].shape[0]),
            nn.ELU(),
            nn.Linear(state_dict["actor.2.weight"].shape[1], state_dict["actor.2.weight"].shape[0]),
            nn.ELU(),
            nn.Linear(state_dict["actor.4.weight"].shape[1], state_dict["actor.4.weight"].shape[0]),
            nn.ELU(),
            nn.Linear(state_dict["actor.6.weight"].shape[1], state_dict["actor.6.weight"].shape[0]),
        )
        self.net.load_state_dict({
            "0.weight": state_dict["actor.0.weight"], "0.bias": state_dict["actor.0.bias"],
            "2.weight": state_dict["actor.2.weight"], "2.bias": state_dict["actor.2.bias"],
            "4.weight": state_dict["actor.4.weight"], "4.bias": state_dict["actor.4.bias"],
            "6.weight": state_dict["actor.6.weight"], "6.bias": state_dict["actor.6.bias"],
        })

    @torch.inference_mode()
    def forward(self, obs: np.ndarray) -> np.ndarray:
        return self.net(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy()


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a world vector into a body frame for a wxyz quaternion."""
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    return v * (2.0 * w * w - 1.0) + 2.0 * qv * np.dot(qv, v) - 2.0 * w * np.cross(qv, v)


def load_actor(path: Path) -> Actor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    actor_keys = [key for key in state if key.startswith("actor.")]
    if tuple(state["actor.0.weight"].shape) != (256, 35) or tuple(state["actor.6.weight"].shape) != (6, 64):
        raise ValueError("This adapter expects the current Dogleg actor shape [35 -> 256 -> 128 -> 64 -> 6].")
    print(f"Loaded actor checkpoint: {path} (iteration={checkpoint.get('iter', 'unknown')}, tensors={len(actor_keys)})")
    return Actor(state).eval()


def joint_maps(model):
    qpos = {}
    qvel = {}
    for name in ACTIVE_LEG + PASSIVE_LEG + WHEELS:
        jid = model.joint(name).id
        qpos[name] = int(model.jnt_qposadr[jid])
        qvel[name] = int(model.jnt_dofadr[jid])
    return qpos, qvel


def set_neutral(model, data, qpos: dict[str, int]) -> None:
    for name, degrees in NEUTRAL_DEG.items():
        data.qpos[qpos[name]] = math.radians(degrees)
    for name in WHEELS:
        data.qpos[qpos[name]] = 0.0
    # Preserve XML root pose/freejoint and reset all velocities.
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0


def make_observation(model, data, qpos, qvel, previous_action, vx, wz, height_cmd=0.22):
    root_id = model.body("base_link").id
    root_quat = np.asarray(data.xquat[root_id], dtype=np.float32)
    root_ang_vel_world = np.asarray(data.qvel[3:6], dtype=np.float32)
    root_ang_vel_b = quat_rotate_inverse(root_quat, root_ang_vel_world)
    gravity_b = quat_rotate_inverse(root_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    # The policy was trained with command, height, angular velocity, projected
    # gravity, 6 actuated joint positions/velocities, previous action, and the
    # 7-D normal ctrl-mode one-hot/phase block.
    command = np.array([vx, 0.0, wz], dtype=np.float32)
    height = np.array([float(data.qpos[2]) * 5.0], dtype=np.float32)  # absolute-height mode
    joint_pos = np.array([data.qpos[qpos[n]] - math.radians(NEUTRAL_DEG.get(n, 0.0)) for n in ACTIVE_LEG + WHEELS], dtype=np.float32)
    joint_vel = np.array([data.qvel[qvel[n]] for n in ACTIVE_LEG + WHEELS], dtype=np.float32)
    # Isaac scales these observation blocks according to env.yaml.
    obs = np.concatenate([
        command,
        height,
        root_ang_vel_b * 0.5,
        gravity_b,
        joint_pos,
        joint_vel[:4] * 0.1,
        joint_vel[4:] * 0.1,
        previous_action,
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ])
    if obs.shape != (35,):
        raise RuntimeError(f"Constructed observation has shape {obs.shape}, expected (35,)")
    return obs


def apply_control(model, data, qpos, qvel, action, kp_leg, kd_leg, kp_wheel, kd_wheel):
    # Isaac leg actions are position residuals around the calibrated neutral
    # pose; wheel actions are target angular velocities with scale 10 rad/s.
    for i, name in enumerate(ACTIVE_LEG):
        target = math.radians(NEUTRAL_DEG[name]) + 0.5 * float(np.clip(action[i], -3.14, 3.14))
        err = target - data.qpos[qpos[name]]
        data.qfrc_applied[qvel[name]] = kp_leg * err - kd_leg * data.qvel[qvel[name]]
    for i, name in enumerate(WHEELS):
        target_vel = 10.0 * float(np.clip(action[4 + i], -100.0, 100.0))
        data.qfrc_applied[qvel[name]] = kp_wheel * (target_vel - data.qvel[qvel[name]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_XML)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vx", type=float, default=0.0, help="Forward command (m/s).")
    parser.add_argument("--wz", type=float, default=0.0, help="Yaw-rate command (rad/s).")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--kp-leg", type=float, default=60.0)
    parser.add_argument("--kd-leg", type=float, default=2.0)
    parser.add_argument("--kp-wheel", type=float, default=1.0)
    parser.add_argument("--kd-wheel", type=float, default=0.2)
    args = parser.parse_args()

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise SystemExit("MuJoCo is not installed in this Python environment. Install it with: pip install mujoco") from exc

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    qpos, qvel = joint_maps(model)
    set_neutral(model, data, qpos)
    mujoco.mj_forward(model, data)
    actor = load_actor(args.checkpoint)
    previous_action = np.zeros(6, dtype=np.float32)
    policy_dt = 0.02  # Isaac Lab: dt=0.005, decimation=4
    next_report = 0.0

    viewer = None
    if not args.no_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as exc:
            raise SystemExit(f"Could not open MuJoCo viewer ({exc}); retry with --no-viewer.") from exc

    steps = int(args.duration / policy_dt)
    for step in range(steps):
        obs = make_observation(model, data, qpos, qvel, previous_action, args.vx, args.wz)
        action = actor(obs)
        apply_control(model, data, qpos, qvel, action, args.kp_leg, args.kd_leg, args.kp_wheel, args.kd_wheel)
        previous_action = action.astype(np.float32)
        for _ in range(4):
            mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if data.time >= next_report:
            root_id = model.body("base_link").id
            wheel_id = model.body("left_wheel_link").id
            print(f"t={data.time:6.2f}s base_z={data.xpos[root_id, 2]:+.3f} wheel_z={data.xpos[wheel_id, 2]:+.3f} action={np.round(action, 3)}")
            next_report += 1.0

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
