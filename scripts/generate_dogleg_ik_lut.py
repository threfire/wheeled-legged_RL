#!/usr/bin/env python3
"""Generate a Dogleg closed-chain IK lookup table from its deployment MJCF.

Requires mujoco, numpy, and scipy in the Python environment. Example:
  python scripts/generate_dogleg_ik_lut.py \
    --mjcf ../wheelbipe_ros2_sim2sim/src/resources/robot_descriptions/dogleg/mjcf/dogleg_floating_source.xml
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


MJCF_ORDER = ("hip_thigh", "knee", "thigh_ternary", "ternary_coupler", "hip_upper", "upper_rocker")
TASK_ORDER = ("hip_thigh", "hip_upper", "knee", "thigh_ternary", "ternary_coupler", "upper_rocker")
LENGTH_RANGE = (0.13, 0.40)
ANGLE_RANGE_DEG = (-120.0, 135.0)


def side_solver(model, side):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joint_names = [f"{side}_{name}" for name in MJCF_ORDER]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    if min(joint_ids) < 0:
        raise RuntimeError(f"MJCF is missing leg joints for {side}: {joint_names}")
    qpos_ids = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    equality_names = (
        f"{side}_calf_coupler_closure_minus",
        f"{side}_calf_coupler_closure_plus",
        f"{side}_rocker_ternary_closure_minus",
        f"{side}_rocker_ternary_closure_plus",
    )
    site_pairs = []
    for name in equality_names:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id < 0:
            raise RuntimeError(f"MJCF is missing closure equality {name}")
        site_pairs.append((int(model.eq_obj1id[eq_id]), int(model.eq_obj2id[eq_id])))

    hip_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_thigh"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_upper"),
    ]
    origin = np.mean([data.xanchor[joint_id] for joint_id in hip_ids], axis=0)
    wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wheel_link")
    if wheel_id < 0:
        raise RuntimeError(f"MJCF is missing {side}_wheel_link")

    seed_deg = np.asarray((25.84920168, -39.74620307, -39.74620307, 39.74620307, -36.96964324, 39.74620307))
    if side == "right":
        seed_deg *= -1.0
    seed = np.deg2rad(seed_deg)

    def residual(q, length, angle):
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        values = []
        for site_a, site_b in site_pairs:
            values.extend((data.site_xpos[site_a] - data.site_xpos[site_b]) * 1000.0)
        target_xz = np.asarray((length * np.sin(angle), -length * np.cos(angle)))
        actual_xz = data.xpos[wheel_id, [0, 2]] - origin[[0, 2]]
        values.extend((actual_xz - target_xz) * 1000.0)
        return np.asarray(values)

    def solve(length, angle, initial):
        result = least_squares(
            residual,
            initial,
            args=(float(length), float(angle)),
            max_nfev=100,
            diff_step=1.0e-5,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        errors = residual(result.x, length, angle)
        closure_mm = np.max(np.linalg.norm(errors[:12].reshape(4, 3), axis=1))
        target_mm = np.linalg.norm(errors[12:14])
        return result.x, float(closure_mm), float(target_mm)

    return seed, solve


def solve_grid(model, side, lengths, angles):
    seed, solve = side_solver(model, side)
    table = np.empty((len(lengths), len(angles), 6), dtype=np.float64)
    closure_errors = np.empty((len(lengths), len(angles)), dtype=np.float64)
    target_errors = np.empty_like(closure_errors)
    previous = seed
    for length_index, length in enumerate(lengths):
        angle_indices = range(len(angles)) if length_index % 2 == 0 else range(len(angles) - 1, -1, -1)
        for angle_index in angle_indices:
            q, closure_mm, target_mm = solve(length, angles[angle_index], previous)
            table[length_index, angle_index] = q
            closure_errors[length_index, angle_index] = closure_mm
            target_errors[length_index, angle_index] = target_mm
            previous = q
        if length_index % 8 == 0 or length_index == len(lengths) - 1:
            print(f"{side}: solved length {length:.3f} m ({length_index + 1}/{len(lengths)})", flush=True)

    table = table[..., [MJCF_ORDER.index(name) for name in TASK_ORDER]]
    return table, closure_errors, target_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", required=True, type=Path, help="Dogleg floating source MJCF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("source/agent_world/agent_world/assets/usd_files/dogleg_floating/configuration/dogleg_ik_lut.npz"),
    )
    parser.add_argument("--length-steps", type=int, default=47)
    parser.add_argument("--angle-steps", type=int, default=101)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    lengths = np.linspace(*LENGTH_RANGE, args.length_steps, dtype=np.float64)
    angles = np.deg2rad(np.linspace(*ANGLE_RANGE_DEG, args.angle_steps, dtype=np.float64))
    sides, closure_errors, target_errors = [], [], []
    for side in ("left", "right"):
        q, closure, target = solve_grid(model, side, lengths, angles)
        sides.append(q)
        closure_errors.append(closure)
        target_errors.append(target)

    table = np.asarray(sides, dtype=np.float32)
    closure_errors = np.asarray(closure_errors)
    target_errors = np.asarray(target_errors)
    max_closure = float(np.max(closure_errors))
    max_target = float(np.max(target_errors))
    if max_closure > 0.1 or max_target > 0.1:
        raise RuntimeError(
            f"Grid solve did not close tightly: max closure={max_closure:.4f} mm, "
            f"max target error={max_target:.4f} mm"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        lengths=lengths.astype(np.float32),
        angles=angles.astype(np.float32),
        joint_pos=table,
        max_closure_error_mm=np.asarray(max_closure, dtype=np.float32),
        max_target_error_mm=np.asarray(max_target, dtype=np.float32),
        mjcf_path=np.asarray(str(args.mjcf)),
    )
    print(f"Saved {table.shape} joint table to {args.output}")
    print(f"Grid maximum closure residual: {max_closure:.6f} mm")
    print(f"Grid maximum target-position error: {max_target:.6f} mm")
    for side, side_table in zip(("left", "right"), table):
        q_range = np.rad2deg(side_table.reshape(-1, 6))
        print(f"{side} joint range (deg), task order {TASK_ORDER}:")
        print(f"  {np.round(q_range.min(axis=0), 2)} .. {np.round(q_range.max(axis=0), 2)}")


if __name__ == "__main__":
    main()
