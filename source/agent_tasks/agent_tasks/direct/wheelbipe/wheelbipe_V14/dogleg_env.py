"""Environment wrapper for the converted dogleg asset."""

from collections.abc import Mapping

import torch

from .env import WheelbipeV14Env
from .dogleg_ik import dogleg_inverse_kinematics


class DoglegEnv(WheelbipeV14Env):
    """Reuse the wheelbipe locomotion environment with dogleg bookkeeping."""

    def _setup_scene(self):
        preview = getattr(self.cfg, "single_env_preview", None)
        if preview is not None:
            from scripts.utils.single_env_preview import hide_other_environments

            self._preview_hidden_env_count = hide_other_environments(
                self.sim.get_initial_stage(), self.cfg.scene.num_envs, preview["env_index"]
            )
            print(
                f"[Preview] Hidden {self._preview_hidden_env_count} environments before robot spawning/cloning.",
                flush=True,
            )
        super()._setup_scene()

    def _inverse_kinematics(self, leg_length: torch.Tensor, leg_angle: torch.Tensor) -> torch.Tensor:
        return dogleg_inverse_kinematics(leg_length, leg_angle)

    def _ensure_recovery_buffers(self) -> None:
        if hasattr(self, "_recovery_active"):
            return
        self._recovery_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._recovery_stable_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._recovery_attempted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._recovery_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._recovery_time_to_success = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _recovery_cfg(self) -> dict:
        cfg = getattr(self.cfg, "recovery_training_cfg", {})
        return cfg if isinstance(cfg, Mapping) else {}

    def _custom_reset_random(self, env_ids) -> None:
        self._ensure_recovery_buffers()
        self._recovery_active[env_ids] = False
        self._recovery_stable_time[env_ids] = 0.0
        self._recovery_attempted[env_ids] = False
        self._recovery_success[env_ids] = False
        self._recovery_time_to_success[env_ids] = 0.0
        super()._custom_reset_random(env_ids)

        cfg = self._recovery_cfg()
        if not bool(cfg.get("enabled", False)):
            return
        mode_ids = getattr(self, "predefined_reset_air_command_limit_mode_id", None)
        if mode_ids is None:
            return
        active_modes = self._get_active_predefined_reset_air_modes()
        recovery_mode_indices = {
            idx for idx, mode in enumerate(active_modes) if bool(mode.get("recover", False))
        }
        if not recovery_mode_indices:
            return
        selected = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        for mode_idx in recovery_mode_indices:
            selected |= mode_ids[env_ids] == mode_idx
        recovery_env_ids = env_ids[selected]
        self._recovery_active[recovery_env_ids] = True
        self._recovery_stable_time[recovery_env_ids] = 0.0
        self._recovery_attempted[recovery_env_ids] = True
        self.command[recovery_env_ids, :3] = 0.0
        self.height_cmd[recovery_env_ids] = float(cfg.get("height_target", 0.30))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self._ensure_recovery_buffers()
        cfg = self._recovery_cfg()
        if not bool(cfg.get("enabled", False)):
            return
        projected_gravity = self.robot.data.projected_gravity_b
        root_height = self.robot.data.root_pos_w[:, 2]
        root_lin_speed = torch.linalg.norm(self.robot.data.root_lin_vel_b, dim=-1)
        root_ang_speed = torch.linalg.norm(self.robot.data.root_ang_vel_b, dim=-1)
        iteration = self._get_training_iteration()
        fallen = (
            (torch.abs(projected_gravity[:, 0])
             >= float(cfg.get("fallen_pitch_gravity_x_min", 0.75)))
            | (projected_gravity[:, 2]
               >= float(cfg.get("fallen_inverted_gravity_z_min", 0.55)))
        )
        newly_fallen = (~self._recovery_active) & fallen & (
            iteration >= int(cfg.get("iteration_start", 1000))
        )
        self._recovery_active |= newly_fallen
        self._recovery_attempted |= newly_fallen
        self._recovery_stable_time[newly_fallen] = 0.0

        active = self._recovery_active
        upright = (
            (torch.linalg.norm(projected_gravity[:, :2], dim=-1)
             <= float(cfg.get("upright_projected_gravity_xy_max", 0.35)))
            & (projected_gravity[:, 2] <= float(cfg.get("upright_projected_gravity_z_max", -0.80)))
            & (root_height >= float(cfg.get("minimum_root_height", 0.20)))
            & (root_lin_speed <= float(cfg.get("maximum_root_linear_speed", 0.35)))
            & (root_ang_speed <= float(cfg.get("maximum_root_angular_speed", 0.60)))
        )
        self._recovery_stable_time[active & upright] += self.step_dt
        self._recovery_stable_time[active & ~upright] = 0.0
        recovered = active & (
            self._recovery_stable_time >= float(cfg.get("stable_duration_s", 0.50))
        )
        if torch.any(recovered):
            recovered_ids = recovered.nonzero(as_tuple=False).flatten()
            self._recovery_active[recovered_ids] = False
            self._recovery_stable_time[recovered_ids] = 0.0
            self._recovery_success[recovered_ids] = True
            self._recovery_time_to_success[recovered_ids] = (
                self.episode_length_buf[recovered_ids].to(dtype=torch.float) * self.step_dt
            )
            self._clear_predefined_reset_air_command_limits(recovered_ids)
            self.command_generator._resample_command(recovered_ids)
            self.command_generator._update_command()
            self.height_cmd[recovered_ids] = self._sample_height_command(
                recovered_ids, use_env_origins=True
            )

        active = self._recovery_active
        self.command[active, :3] = 0.0
        self.height_cmd[active] = float(cfg.get("height_target", 0.30))

    def _reset_idx(self, env_ids) -> None:
        self._ensure_recovery_buffers()
        if env_ids is None:
            log_ids = self.robot._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            log_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            log_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        attempted = self._recovery_attempted[log_ids].clone()
        succeeded = self._recovery_success[log_ids].clone()
        recovery_times = self._recovery_time_to_success[log_ids].clone()
        super()._reset_idx(env_ids)
        if torch.any(attempted):
            attempted_success = succeeded[attempted]
            successful_times = recovery_times[attempted & succeeded]
            logs = self.extras.setdefault("log", {})
            logs["Episode/Recovery/success_rate"] = attempted_success.float().mean().item()
            logs["Episode/Recovery/attempts"] = float(attempted.sum().item())
            if successful_times.numel() > 0:
                logs["Episode/Recovery/time_to_recover_s"] = successful_times.mean().item()

    def _get_ctrl_mode_obs_raw(self) -> torch.Tensor:
        obs = super()._get_ctrl_mode_obs_raw()
        self._ensure_recovery_buffers()
        if obs.shape[-1] < 4 or not bool(self._recovery_cfg().get("enabled", False)):
            return obs
        obs = obs.clone()
        active = self._recovery_active
        obs[:, :5] = torch.where(active.unsqueeze(-1), torch.zeros_like(obs[:, :5]), obs[:, :5])
        obs[:, 3] = torch.where(active, torch.ones_like(obs[:, 3]), obs[:, 3])
        return obs

    def _postprocess_reward_terms(self, reward_terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        reward_terms = super()._postprocess_reward_terms(reward_terms)
        self._ensure_recovery_buffers()
        active = self._recovery_active
        if not bool(self._recovery_cfg().get("enabled", False)):
            reward_terms["recovery_upright"] = torch.zeros(self.num_envs, device=self.device)
            reward_terms["recovery_height"] = torch.zeros(self.num_envs, device=self.device)
            return reward_terms

        # During recovery, locomotion and nominal height targets are irrelevant;
        # body contact is expected while starting face-down/up.
        for name in (
            "track_lin_vel_xy", "track_lin_vel_xy_tight", "track_lin_vel_xy_square",
            "track_ang_vel_z", "track_ang_vel_z_square", "stand_still_lin_vel",
            "stand_still", "track_height_exp", "track_height_exp_soft",
            "track_height_exp_tight", "track_height_exp_both_wheels_contact",
            "track_height_square", "undesired_contact",
        ):
            if name in reward_terms:
                reward_terms[name] = reward_terms[name] * (~active).float()

        gravity = self.robot.data.projected_gravity_b
        upright_score = -gravity[:, 2] - 0.25 * torch.sum(torch.square(gravity[:, :2]), dim=-1)
        reward_terms["recovery_upright"] = upright_score * active.float()
        target_height = float(self._recovery_cfg().get("height_target", 0.30))
        height_error = torch.abs(self.robot.data.root_pos_w[:, 2] - target_height)
        reward_terms["recovery_height"] = torch.exp(-height_error / 0.07) * active.float()
        return reward_terms

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, timed_out = super()._get_dones()
        self._ensure_recovery_buffers()
        # The configured body contacts that initiate recovery are expected until
        # the robot rights itself. Numerical safety and episode timeouts remain active.
        safety_terminated = getattr(
            self, "_numerical_safety_reset_buf", torch.zeros_like(self._recovery_active)
        )
        terminated = (terminated & ~self._recovery_active) | (
            safety_terminated & self._recovery_active
        )
        return terminated, timed_out

    def _validate_v14_bookkeeping(self) -> None:
        checks = {
            "legs_act": (len(self._legs_act_idx), 4),
            "wheel": (len(self._wheel_idx), 2),
            "ordered_leg_joint_idx": (len(self._ordered_leg_joint_idx), 12),
            "ordered_leg_body_idx": (len(self._ordered_leg_body_idx), 12),
        }
        if self._use_gimbal:
            checks["gimbal_yaw"] = (len(self._gimbal_yaw_idx), 1)
            checks["gimbal_pitch"] = (len(self._gimbal_pitch_idx), 1)
        bad_checks = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in checks.items()
            if actual != expected
        ]
        if bad_checks:
            raise RuntimeError("Dogleg asset discovery mismatch: " + ", ".join(bad_checks))

