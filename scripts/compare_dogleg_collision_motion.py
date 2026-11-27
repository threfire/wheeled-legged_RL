#!/usr/bin/env python3
"""Run a repeatable open-loop wheel-motion/contact test on either Dogleg USD.

Run once with --asset legacy and once with --asset current. The same task
configuration, reset pose, wheel-speed profile and randomization-disabled
physics settings are used in both runs. A CSV is written for per-step review.
"""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", choices=("legacy", "current", "contactfix", "compound"), required=True)
parser.add_argument("--steps", type=int, default=220, help="Policy steps; default is 4.4 simulated seconds.")
parser.add_argument("--wheel-speed", type=float, default=4.0, help="Peak target wheel speed in rad/s.")
parser.add_argument("--output", type=str, default=None, help="CSV output path.")
parser.add_argument("--hold-open", action="store_true", help="Keep the GUI responsive after the motion test until its window is closed.")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import agent_world  # noqa: F401
import agent_tasks  # noqa: F401


TASKS = {
    "legacy": "Robotics-Dogleg-Unified-Legacy-Asset-v0",
    "current": "Robotics-Dogleg-Unified-Flat-v0",
    "contactfix": "Robotics-Dogleg-Unified-Flat-v0",
    "compound": "Robotics-Dogleg-Unified-Flat-v0",
}
EVENT_TERMS_TO_DISABLE = (
    "add_base_mass", "base_inertia", "add_leg_mass", "add_wheel_mass", "wheels_inertia",
    "base_com", "base_material", "guide_material", "physics_material",
    "leg_front_joint_frictions", "leg_rear_joint_frictions", "wheel_joint_frictions",
    "legs_inact_joint_frictions", "gimbal_joint_frictions", "robot_joint_stiffness_and_damping",
    "spring_damping", "leg_effort_noise", "wheel_effort_noise", "push_robot",
    "base_external_force_torque_xyz", "reset_base",
)
CSV_FIELDS = (
    "step", "time_s", "phase", "wheel_target_rad_s",
    "left_wheel_vel_rad_s", "right_wheel_vel_rad_s",
    "left_wheel_contact_peak_N", "right_wheel_contact_peak_N",
    "max_undesired_contact_N", "top_undesired_contact_body",
    "left_wheel_slip_speed_m_s", "right_wheel_slip_speed_m_s",
    "base_vx_m_s", "base_vy_m_s", "base_yaw_rate_rad_s", "base_z_m",
    "base_roll_rad", "base_pitch_rad", "leg_target_error_max_rad",
    "closure_residual_max_mm", "reward_track_height_tight",
    "reward_wheel_air_spin", "reward_undesired_contact",
    "reward_track_lin_vel_xy", "reward_track_ang_vel_z", "terminated", "time_out",
)


def phase_and_target(t: float, wheel_speed: float) -> tuple[str, float]:
    if t < 0.5:
        return "settle", 0.0
    if t < 0.9:
        return "ramp", wheel_speed * (t - 0.5) / 0.4
    if t < 3.0:
        return "drive", wheel_speed
    return "coast", 0.0


def collect_closure_anchors(stage, body_names):
    body_to_idx = {name: i for i, name in enumerate(body_names)}
    closures = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsSphericalJoint":
            continue
        path = str(prim.GetPath())
        if "/env_0/Robot/" not in path:
            continue
        rel0 = prim.GetRelationship("physics:body0")
        rel1 = prim.GetRelationship("physics:body1")
        targets0 = rel0.GetTargets() if rel0 else []
        targets1 = rel1.GetTargets() if rel1 else []
        if not targets0 or not targets1:
            continue
        name0 = str(targets0[0]).split("/")[-1]
        name1 = str(targets1[0]).split("/")[-1]
        if name0 not in body_to_idx or name1 not in body_to_idx:
            continue
        p0 = prim.GetAttribute("physics:localPos0").Get()
        p1 = prim.GetAttribute("physics:localPos1").Get()
        closures.append((
            str(prim.GetName()), body_to_idx[name0], body_to_idx[name1],
            torch.tensor(tuple(p0), dtype=torch.float32),
            torch.tensor(tuple(p1), dtype=torch.float32),
        ))
    return closures


def max_closure_residual(robot, closures) -> float:
    if not closures:
        return float("nan")
    device = robot.device
    residuals = []
    for _, i0, i1, local0_cpu, local1_cpu in closures:
        local0 = local0_cpu.to(device).unsqueeze(0)
        local1 = local1_cpu.to(device).unsqueeze(0)
        pos0 = robot.data.body_pos_w[0, i0]
        pos1 = robot.data.body_pos_w[0, i1]
        quat0 = robot.data.body_quat_w[0, i0].unsqueeze(0)
        quat1 = robot.data.body_quat_w[0, i1].unsqueeze(0)
        world0 = pos0 + quat_apply(quat0, local0)[0]
        world1 = pos1 + quat_apply(quat1, local1)[0]
        residuals.append(torch.linalg.vector_norm(world0 - world1))
    return float(torch.stack(residuals).max().item() * 1000.0)


def quaternion_roll_pitch(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll and pitch from an Isaac (w, x, y, z) quaternion."""
    w, x, y, z = quat_wxyz.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    sin_pitch = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    return roll, pitch


def scalar_reward(env, key: str) -> float:
    values = getattr(env, "_last_reward_terms", {})
    value = values.get(key)
    if value is None:
        return float("nan")
    return float(value[0].item())


def main():
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")

    device = args.device or "cuda:0"
    cfg = parse_env_cfg(TASKS[args.asset], device=device, num_envs=1)
    if args.asset == "contactfix":
        cfg.robot_cfg.spawn.usd_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../source/agent_world/agent_world/assets/usd_files/dogleg_floating/configuration/"
                "dogleg_floating_physics_contactfix.usda",
            )
        )
    elif args.asset == "compound":
        cfg.robot_cfg.spawn.usd_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../source/agent_world/agent_world/assets/usd_files/dogleg_floating/configuration/"
                "dogleg_floating_physics_compound_candidate.usda",
            )
        )
    cfg.seed = 4217
    cfg.use_leg_random_start = False
    cfg.use_predefined_leg_random_start = False
    cfg.use_joint_vel_random_start = False
    cfg.default_height_cmd = 0.224
    cfg.height_range = [0.224, 0.224]
    cfg.commands.rel_standing_envs = 1.0
    cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.ranges.heading = (0.0, 0.0)
    # Remove random changes to mass, friction, actuator gains, reset velocity,
    # and effort. Ground and robot materials then use the same authored values.
    for name in EVENT_TERMS_TO_DISABLE:
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)

    env = gym.make(TASKS[args.asset], cfg=cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.robot
    contact_sensor = unwrapped.contact_sensor
    wheel_joint_ids = list(unwrapped._wheel_idx)
    leg_joint_ids = list(unwrapped._legs_act_idx)
    if len(wheel_joint_ids) != 2 or len(leg_joint_ids) != 4:
        raise RuntimeError(f"Expected 2 wheel and 4 leg actuators; got {wheel_joint_ids}, {leg_joint_ids}")
    wheel_body_ids, wheel_body_names = robot.find_bodies(".*_wheel_link")
    if len(wheel_body_ids) != 2:
        raise RuntimeError(f"Expected 2 wheel links; found {wheel_body_names}")
    sensor_names = list(contact_sensor.body_names)
    wheel_sensor_ids = [sensor_names.index(name) for name in wheel_body_names if name in sensor_names]
    if len(wheel_sensor_ids) != 2:
        raise RuntimeError(f"Wheel links not found in contact sensor: {wheel_body_names}, {sensor_names}")

    env.reset()
    # Explicitly restore the same model-defined neutral pose and zero velocity.
    default_q = robot.data.default_joint_pos.clone()
    zero_qd = torch.zeros_like(default_q)
    robot.write_joint_state_to_sim(default_q, zero_qd)
    robot.set_joint_position_target(default_q)
    robot.set_joint_velocity_target(zero_qd)
    root_state = robot.data.root_state_w.clone()
    root_state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root_state)
    unwrapped.sim.forward()
    unwrapped.scene.update(unwrapped.physics_dt)

    closures = collect_closure_anchors(unwrapped.sim.stage, robot.body_names)
    print(f"Asset={args.asset}; task={TASKS[args.asset]}; device={device}")
    print(f"USD={cfg.robot_cfg.spawn.usd_path}")
    print(f"Envs=1; seed=4217; zero command; fixed height=0.224 m; randomization disabled")
    print(f"Wheel links={wheel_body_names}; closure joints={len(closures)}")
    print(f"Motion: 0-0.5s settle, 0.5-0.9s ramp, 0.9-3.0s drive, then coast; peak={args.wheel_speed:.2f} rad/s")

    output_path = args.output or f"logs/collision_diagnostic/dogleg_{args.asset}.csv"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows = []
    dt = float(unwrapped.step_dt)
    wheel_scale = float(unwrapped.wheel_action_scale)
    wheel_radius = float(getattr(unwrapped.cfg, "height_reward_airborne_wheel_radius", 0.06))
    actions = torch.zeros((1, 6), dtype=torch.float32, device=unwrapped.device)

    try:
        for step in range(args.steps):
            t = step * dt
            phase, target_wheel = phase_and_target(t, args.wheel_speed)
            actions.zero_()
            # Four zero leg actions hold the configured neutral targets;
            # the last two channels are normalized wheel-velocity targets.
            actions[0, 4] = target_wheel / wheel_scale
            actions[0, 5] = target_wheel / wheel_scale
            _, _, terminated, time_out, _ = env.step(actions)

            wheel_vel = robot.data.joint_vel[0, wheel_joint_ids]
            contact_hist = getattr(contact_sensor.data, "net_forces_w_history", None)
            if contact_hist is not None:
                per_body_peaks = torch.amax(torch.linalg.vector_norm(contact_hist[0], dim=-1), dim=0)
            else:
                force_now = contact_sensor.data.net_forces_w[0, wheel_sensor_ids]
                per_body_peaks = torch.zeros(len(sensor_names), dtype=torch.float32, device=unwrapped.device)
                per_body_peaks[wheel_sensor_ids] = torch.linalg.vector_norm(force_now, dim=-1)
            force_peaks = per_body_peaks[wheel_sensor_ids]
            undesired_ids = list(getattr(unwrapped, "_undesired_contact_link_idx", []))
            if undesired_ids:
                undesired_peaks = per_body_peaks[undesired_ids]
                top_local = int(torch.argmax(undesired_peaks).item())
                max_undesired_force = float(undesired_peaks[top_local].item())
                top_undesired_body = sensor_names[undesired_ids[top_local]]
            else:
                max_undesired_force = float("nan")
                top_undesired_body = ""

            # Same wheel-link slip estimate used by the task's play diagnostics.
            wheel_lin = robot.data.body_lin_vel_w[0, wheel_body_ids]
            wheel_ang = robot.data.body_ang_vel_w[0, wheel_body_ids]
            bottom_offset = torch.zeros_like(wheel_lin)
            bottom_offset[:, 2] = -wheel_radius
            contact_vel = wheel_lin + torch.cross(wheel_ang, bottom_offset, dim=-1)
            slip = torch.linalg.vector_norm(contact_vel[:, :2], dim=-1)

            root_lin = robot.data.root_lin_vel_b[0]
            root_ang = robot.data.root_ang_vel_b[0]
            root_z = float(robot.data.root_pos_w[0, 2].item())
            roll, pitch = quaternion_roll_pitch(robot.data.root_quat_w[0])
            q_error = torch.max(torch.abs(robot.data.joint_pos[0, leg_joint_ids] - default_q[0, leg_joint_ids]))
            row = {
                "step": step,
                "time_s": t,
                "phase": phase,
                "wheel_target_rad_s": target_wheel,
                "left_wheel_vel_rad_s": float(wheel_vel[0].item()),
                "right_wheel_vel_rad_s": float(wheel_vel[1].item()),
                "left_wheel_contact_peak_N": float(force_peaks[0].item()),
                "right_wheel_contact_peak_N": float(force_peaks[1].item()),
                "max_undesired_contact_N": max_undesired_force,
                "top_undesired_contact_body": top_undesired_body,
                "left_wheel_slip_speed_m_s": float(slip[0].item()),
                "right_wheel_slip_speed_m_s": float(slip[1].item()),
                "base_vx_m_s": float(root_lin[0].item()),
                "base_vy_m_s": float(root_lin[1].item()),
                "base_yaw_rate_rad_s": float(root_ang[2].item()),
                "base_z_m": root_z,
                "base_roll_rad": float(roll.item()),
                "base_pitch_rad": float(pitch.item()),
                "leg_target_error_max_rad": float(q_error.item()),
                "closure_residual_max_mm": max_closure_residual(robot, closures),
                "reward_track_height_tight": scalar_reward(unwrapped, "track_height_exp_tight"),
                "reward_wheel_air_spin": scalar_reward(unwrapped, "wheel_air_spin"),
                "reward_undesired_contact": scalar_reward(unwrapped, "undesired_contact"),
                "reward_track_lin_vel_xy": scalar_reward(unwrapped, "track_lin_vel_xy"),
                "reward_track_ang_vel_z": scalar_reward(unwrapped, "track_ang_vel_z"),
                "terminated": bool(terminated[0].item()),
                "time_out": bool(time_out[0].item()),
            }
            rows.append(row)
            if step % 10 == 0 or step == args.steps - 1:
                print(
                    f"[{step:04d} {t:5.2f}s {phase:6s}] cmd_w={target_wheel:5.2f} "
                    f"wheel_w=({row['left_wheel_vel_rad_s']:6.2f},{row['right_wheel_vel_rad_s']:6.2f}) "
                    f"Fz=({row['left_wheel_contact_peak_N']:7.2f},{row['right_wheel_contact_peak_N']:7.2f})N "
                    f"other_contact={max_undesired_force:7.2f}N:{top_undesired_body} "
                    f"slip=({row['left_wheel_slip_speed_m_s']:.3f},{row['right_wheel_slip_speed_m_s']:.3f})m/s "
                    f"base_vx={row['base_vx_m_s']:.3f} z={root_z:.3f} "
                    f"closure_max={row['closure_residual_max_mm']:.2f}mm "
                    f"r_height={row['reward_track_height_tight']:.4f} "
                    f"r_air={row['reward_wheel_air_spin']:.4f} "
                    f"r_contact={row['reward_undesired_contact']:.4f}"
                )
            if bool(terminated[0].item()) or bool(time_out[0].item()):
                print(f"[STOP] environment reset at step {step}: terminated={bool(terminated[0])}, timeout={bool(time_out[0])}")
                break

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written: {os.path.abspath(output_path)}")
        if rows:
            for phase in ("settle", "ramp", "drive", "coast"):
                subset = [r for r in rows if r["phase"] == phase]
                if not subset:
                    continue
                avg_l = sum(r["left_wheel_contact_peak_N"] for r in subset) / len(subset)
                avg_r = sum(r["right_wheel_contact_peak_N"] for r in subset) / len(subset)
                avg_slip = sum(r["left_wheel_slip_speed_m_s"] + r["right_wheel_slip_speed_m_s"] for r in subset) / (2 * len(subset))
                max_closure = max(r["closure_residual_max_mm"] for r in subset)
                print(f"SUMMARY {phase}: avg wheel contact=({avg_l:.2f},{avg_r:.2f}) N; avg slip={avg_slip:.4f} m/s; max closure residual={max_closure:.3f} mm")
        if args.hold_open:
            print("Test complete. Holding default leg targets and zero wheel speed; close the window to exit.")
            # Continue stepping the environment so Isaac Lab refreshes the
            # scene/Fabric transforms and GUI rendering after the diagnostic
            # trajectory. A Kit-only update can leave the physics viewport
            # stale or blank after the final env.step().
            hold_actions = torch.zeros_like(actions)
            while simulation_app.is_running():
                env.step(hold_actions)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
