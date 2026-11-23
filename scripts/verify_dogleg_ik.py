#!/usr/bin/env python3
"""Check whether the inherited V14 inverse kinematics closes Dogleg's loops.

The script writes IK joint positions into one Dogleg articulation, performs a
kinematic forward update without advancing physics, then measures the distance
between both anchors of every spherical closure joint in the USD.
"""

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Robotics-Dogleg-Stand-Flat-v0")
parser.add_argument("--leg-lengths", default="0.20,0.25,0.30", help="meters, comma-separated")
parser.add_argument("--leg-angles-deg", default="0", help="degrees, comma-separated")
parser.add_argument("--right-sign", type=float, choices=(-1.0, 1.0), default=1.0,
                    help="Multiply all right-side IK joint angles by this sign (diagnostic only).")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply
import isaaclab_tasks  # noqa: F401
import agent_world  # noqa: F401
import agent_tasks  # noqa: F401


def parse_csv_floats(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def get_closure_joints(stage):
    closures = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsSphericalJoint":
            continue
        body0_rel = prim.GetRelationship("physics:body0")
        body1_rel = prim.GetRelationship("physics:body1")
        body0_targets = body0_rel.GetTargets() if body0_rel else []
        body1_targets = body1_rel.GetTargets() if body1_rel else []
        if not body0_targets or not body1_targets:
            continue
        p0 = prim.GetAttribute("physics:localPos0").Get()
        p1 = prim.GetAttribute("physics:localPos1").Get()
        closures.append((str(prim.GetPath()), str(body0_targets[0]).split("/")[-1],
                         str(body1_targets[0]).split("/")[-1], tuple(p0), tuple(p1)))
    return closures


def main():
    lengths = parse_csv_floats(args.leg_lengths)
    angles_deg = parse_csv_floats(args.leg_angles_deg)
    if not lengths or not angles_deg:
        raise ValueError("Provide at least one leg length and one angle.")

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.robot
    stage = unwrapped.sim.stage
    closures = get_closure_joints(stage)
    if not closures:
        raise RuntimeError("No PhysicsSphericalJoint closure joints found in the stage.")

    print(f"Dogleg bodies: {len(robot.body_names)}; closure joints found: {len(closures)}")
    print("Residual is the 3D distance between corresponding closure anchors.")
    limits = robot.data.soft_joint_pos_limits[0]
    limit_labels = []
    for index in unwrapped.reorder_reset_joint_idx:
        low, high = torch.rad2deg(limits[index]).tolist()
        limit_labels.append(f"{robot.joint_names[index]}=[{low:.1f}, {high:.1f}] deg")
    print("Runtime soft joint limits: " + ", ".join(limit_labels))
    try:
        for length in lengths:
            for angle_deg in angles_deg:
                env.reset()
                # V14 IK's second dimension is the left/right leg pair; test both sides identically.
                leg_length = torch.full((args.num_envs, 2), length, device=unwrapped.device)
                leg_angle = torch.full((args.num_envs, 2), angle_deg * torch.pi / 180.0,
                                       device=unwrapped.device)
                joint_pos = unwrapped._inverse_kinematics(leg_length, leg_angle)
                joint_pos = joint_pos.transpose(-2, -1).reshape(args.num_envs, -1)
                ordered_names = [robot.joint_names[index] for index in unwrapped.reorder_reset_joint_idx]
                if args.right_sign < 0.0:
                    right_mask = torch.tensor([name.startswith("right_") for name in ordered_names],
                                              dtype=torch.bool, device=unwrapped.device)
                    joint_pos[:, right_mask] *= -1.0
                joint_ids = torch.tensor(unwrapped.reorder_reset_joint_idx,
                                         dtype=torch.long, device=unwrapped.device)
                env_ids = torch.arange(args.num_envs, device=unwrapped.device)
                robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), joint_ids, env_ids)
                robot.set_joint_position_target(joint_pos, joint_ids, env_ids)
                unwrapped.sim.forward()
                unwrapped.scene.update(1.0e-6)

                body_to_idx = {name: idx for idx, name in enumerate(robot.body_names)}
                max_error = 0.0
                errors = []
                for path, name0, name1, local0, local1 in closures:
                    if name0 not in body_to_idx or name1 not in body_to_idx:
                        continue
                    i0, i1 = body_to_idx[name0], body_to_idx[name1]
                    p0 = torch.tensor(local0, dtype=torch.float, device=unwrapped.device).expand(args.num_envs, 3)
                    p1 = torch.tensor(local1, dtype=torch.float, device=unwrapped.device).expand(args.num_envs, 3)
                    world0 = robot.data.body_pos_w[0:1, i0] + quat_apply(robot.data.body_quat_w[0:1, i0], p0[0:1])
                    world1 = robot.data.body_pos_w[0:1, i1] + quat_apply(robot.data.body_quat_w[0:1, i1], p1[0:1])
                    error = torch.linalg.vector_norm(world0 - world1, dim=-1).item()
                    errors.append((error, path, name0, name1))
                    max_error = max(max_error, error)

                print(f"\nleg_length={length:.3f} m, leg_angle={angle_deg:.1f} deg")
                names = ordered_names
                print("  IK joints (deg): " + ", ".join(
                    f"{name}={value:.2f}" for name, value in
                    zip(names, torch.rad2deg(joint_pos[0]).tolist())
                ))
                print(f"  right-side sign multiplier: {args.right_sign:+.0f}")
                if not errors:
                    print("  No closure endpoints matched articulation body names; cannot compute residuals.")
                    continue
                for error, path, name0, name1 in sorted(errors, reverse=True):
                    print(f"  {error * 1000:8.3f} mm  {path.rsplit('/', 1)[-1]}: {name0} <-> {name1}")
                print(f"  MAX residual: {max_error * 1000:.3f} mm")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
