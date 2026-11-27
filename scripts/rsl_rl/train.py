# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import inspect
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--preview_env_index", type=int, default=None, help="Show only this environment in a light GUI.")
parser.add_argument("--preview_render_interval", type=int, default=20, help="Physics steps between preview frames.")
parser.add_argument("--preview_width", type=int, default=960)
parser.add_argument("--preview_height", type=int, default=540)
parser.add_argument(
    "--zero_command_env_fraction",
    type=float,
    default=None,
    help="Override the fraction of environments assigned zero-velocity commands.",
)
parser.add_argument(
    "--stand_still_lin_vel_weight",
    type=float,
    default=None,
    help="Override the stand_still_lin_vel reward coefficient (normally negative).",
)
parser.add_argument(
    "--checkpoint_tag",
    type=str,
    default=None,
    help="One short label appended to periodic checkpoint filenames (e.g. model_500_flat.pt).",
)
parser.add_argument(
    "--resume_training",
    action="store_true",
    default=False,
    help="Resume optimizer state and iteration counter from --checkpoint, then continue from the next iteration.",
)
parser.add_argument(
    "--resume_iteration_only",
    action="store_true",
    default=False,
    help="Load policy weights and continue from checkpoint iteration, but initialize a fresh optimizer.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.resume_training and args_cli.resume_iteration_only:
    parser.error("--resume_training and --resume_iteration_only are mutually exclusive")
if (args_cli.resume_training or args_cli.resume_iteration_only) and args_cli.checkpoint is None:
    parser.error("A checkpoint is required when restoring training progress")
if args_cli.preview_env_index is not None:
    if args_cli.headless or args_cli.video or args_cli.distributed:
        parser.error("Single environment preview requires GUI mode, without --video or --distributed")
    # AppLauncher forwards these SimulationApp options, rather than Hydra.
    args_cli.width = args_cli.preview_width
    args_cli.height = args_cli.preview_height
    args_cli.window_width = args_cli.preview_width
    args_cli.window_height = args_cli.preview_height

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import warnings
import logging
# 抑制quat_rotate弃用警告，避免卡顿
# 这个警告来自isaaclab.utils.math，通过warnings和logging都可能输出
warnings.filterwarnings("ignore", message=".*quat_rotate.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="isaaclab.utils.math")
# 设置math模块的logging级别为ERROR，抑制WARNING
logging.getLogger("isaaclab.utils.math").setLevel(logging.ERROR)
# 添加自定义过滤器，过滤包含quat_rotate的警告消息
class QuatRotateFilter(logging.Filter):
    def filter(self, record):
        return "quat_rotate" not in str(record.getMessage())
logging.getLogger().addFilter(QuatRotateFilter())

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl.runners import *  # noqa: F401
from agent_rl.rsl_rl.runners import *  # noqa: F401
import agent_world  # noqa: F401
import agent_tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# 清理任务名称，去除可能的前导 '=' 符号
if args_cli.task and args_cli.task.startswith('='):
    args_cli.task = args_cli.task[1:]
    print(f"[INFO] 已清理任务名称，去除前导 '=' 符号: {args_cli.task}")


def _load_policy_for_finetune(runner, checkpoint_path: str) -> None:
    """Load matching policy tensors only, without optimizer or iteration state."""
    loaded_dict = torch.load(checkpoint_path, weights_only=False, map_location=runner.device)
    source_state = loaded_dict["model_state_dict"]
    target_state = runner.alg.policy.state_dict()
    compatible_state = {}
    expanded_state = {}
    skipped = []
    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is not None and tuple(target_value.shape) == tuple(value.shape):
            compatible_state[key] = value
        elif (
            target_value is not None
            and value.ndim == 2
            and target_value.ndim == 2
            and target_value.shape[0] == value.shape[0]
            and target_value.shape[1] >= value.shape[1]
        ):
            expanded_value = target_value.clone()
            expanded_value[:, : value.shape[1]] = value
            expanded_state[key] = expanded_value
        else:
            skipped.append((key, tuple(value.shape), tuple(target_value.shape) if target_value is not None else None))
    target_state.update(compatible_state)
    target_state.update(expanded_state)
    runner.alg.policy.load_state_dict(target_state, strict=True)
    runner.current_learning_iteration = 0
    print(
        f"[INFO]: Finetune checkpoint load: loaded {len(compatible_state)} policy tensors, "
        f"expanded {len(expanded_state)} input tensors, "
        f"skipped {len(skipped)} incompatible tensors."
    )
    for key, value in expanded_state.items():
        source_shape = tuple(source_state[key].shape)
        target_shape = tuple(value.shape)
        print(f"[INFO]:   expanded {key}: checkpoint{source_shape} -> model{target_shape}")
    if skipped:
        for key, source_shape, target_shape in skipped[:20]:
            print(f"[INFO]:   skipped {key}: checkpoint{source_shape} -> model{target_shape}")
        if len(skipped) > 20:
            print(f"[INFO]:   ... {len(skipped) - 20} more skipped tensors")


def _notify_env_training_progress(env, iteration: int) -> None:
    """Seed env-side step curricula with the runner iteration before learn()."""

    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "set_training_progress"):
        unwrapped.set_training_progress(iteration=iteration)


def _attach_checkpoint_metadata(env_cfg, agent_cfg, checkpoint_path: str | None) -> None:
    """Record CLI checkpoint provenance in dumped training YAML files."""

    if checkpoint_path is None:
        return

    checkpoint_path = os.path.abspath(checkpoint_path)
    checkpoint_metadata = {
        "path": checkpoint_path,
        "dir": os.path.dirname(checkpoint_path),
        "file": os.path.basename(checkpoint_path),
        "resume_training": bool(args_cli.resume_training),
    }
    setattr(env_cfg, "launch_checkpoint", checkpoint_metadata)
    setattr(agent_cfg, "launch_checkpoint", checkpoint_metadata)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.zero_command_env_fraction is not None:
        fraction = args_cli.zero_command_env_fraction
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("--zero_command_env_fraction must be between 0 and 1")
        env_cfg.commands.rel_standing_envs = fraction
    if args_cli.stand_still_lin_vel_weight is not None:
        if args_cli.stand_still_lin_vel_weight > 0.0:
            raise ValueError("--stand_still_lin_vel_weight must be zero or negative")
        env_cfg.rewards["stand_still_lin_vel"] = args_cli.stand_still_lin_vel_weight

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    if args_cli.preview_env_index is not None:
        from scripts.utils.single_env_preview import configure_preview

        configure_preview(
            env_cfg, args_cli.preview_env_index, args_cli.preview_render_interval,
            args_cli.preview_width, args_cli.preview_height,
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    if args_cli.preview_env_index is not None:
        from scripts.utils.single_env_preview import isolate_preview, preview_initialization

        with preview_initialization():
            env = gym.make(args_cli.task, cfg=env_cfg)
            isolate_preview(env.unwrapped, args_cli.preview_env_index, args_cli.preview_width, args_cli.preview_height)
    else:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    runner_class = eval(getattr(agent_cfg, "runner_class", "OnPolicyRunner"))
    # wrap around environment for rsl-rl
    if runner_class is OnPolicyRunner:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    else:
        from agent_rl.rsl_rl.env import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.checkpoint_tag:
        # RSL-RL hard-codes periodic names as model_<iteration>.pt. Wrap only
        # this runner instance so tagged Dogleg runs remain easy to distinguish
        # without changing the shared RSL-RL package or default filenames.
        import re

        checkpoint_tag = args_cli.checkpoint_tag
        if not re.fullmatch(r"[A-Za-z0-9-]+", checkpoint_tag):
            raise ValueError("--checkpoint_tag must be one short label using only letters, digits, or hyphens")
        save_checkpoint = runner.save

        def save_tagged_checkpoint(path, *save_args, **save_kwargs):
            directory, filename = os.path.split(path)
            match = re.fullmatch(r"model_(\d+)\.pt", filename)
            if match:
                path = os.path.join(directory, f"model_{match.group(1)}_{checkpoint_tag}.pt")
            return save_checkpoint(path, *save_args, **save_kwargs)

        runner.save = save_tagged_checkpoint
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.resume_training:
            # True continuation: restore optimizer/iteration and continue from the next iteration index.
            load_signature = inspect.signature(runner.load)
            if "load_iteration" in load_signature.parameters:
                runner.load(resume_path, load_optimizer=True, load_iteration=True)
            else:
                runner.load(resume_path, load_optimizer=True)
            runner.current_learning_iteration += 1
            print(
                f"[INFO]: Resuming training from next iteration: {runner.current_learning_iteration}"
            )
        elif args_cli.resume_iteration_only:
            checkpoint_data = torch.load(resume_path, weights_only=False, map_location="cpu")
            if "iter" not in checkpoint_data:
                raise KeyError(f"Checkpoint has no iteration counter: {resume_path}")
            checkpoint_iteration = int(checkpoint_data["iter"])
            _load_policy_for_finetune(runner, resume_path)
            runner.current_learning_iteration = checkpoint_iteration + 1
            print(
                "[INFO]: Loaded policy weights with a fresh optimizer; "
                f"continuing at iteration {runner.current_learning_iteration} "
                f"(checkpoint iteration {checkpoint_iteration})."
            )
        else:
            # Weight initialization / finetune-from-checkpoint workflow.
            _load_policy_for_finetune(runner, resume_path)

    _notify_env_training_progress(env, int(getattr(runner, "current_learning_iteration", 0)))
    _attach_checkpoint_metadata(env_cfg, agent_cfg, args_cli.checkpoint)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
