#!/usr/bin/env bash
set -euo pipefail

# Fine-tune from the contact-fixed Dogleg 1500 checkpoint's policy weights only.
# Optimizer state and iteration count start fresh with Dogleg random-reset IK.
TASK="Robotics-Dogleg-Unified-Flat-v0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_SH="/workspace/isaaclab/isaaclab.sh"
ASSET_USD="${REPO_ROOT}/source/agent_world/agent_world/assets/usd_files/dogleg_floating/configuration/dogleg_floating_physics_contactfix.usda"
CHECKPOINT="${REPO_ROOT}/logs/rsl_rl/dogleg_flat_direct/2026-11-26_01-54-48_dogleg_contactfix_unified_scratch_20000_highspeed6000_zero10/model_1500_contactfix.pt"
CHECKPOINT_TAG="ik"
RUN_NAME="dogleg_contactfix_from1500_weights_20000_zero10_ik"
NUM_ENVS=20000
MAX_ITERATIONS=6000
ZERO_COMMAND_ENV_FRACTION=0.1
STAND_STILL_LIN_VEL_WEIGHT=-1.0
DEVICE="cuda:0"
# Run without opening the Isaac Sim graphical window.
HEADLESS=true

if [[ ! -x "${ISAACLAB_SH}" ]]; then
  echo "Isaac Lab launcher not found or not executable: ${ISAACLAB_SH}" >&2
  exit 1
fi
if [[ ! -f "${ASSET_USD}" ]]; then
  echo "Verified contact-fixed USD not found: ${ASSET_USD}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if ! grep -Fq 'dogleg_floating_physics_contactfix.usda' \
  "${REPO_ROOT}/source/agent_world/agent_world/assets/dogleg.py"; then
  echo "Dogleg_CFG is not configured to use the contact-fixed USD." >&2
  exit 1
fi
if ! grep -Eq '^[[:space:]]*use_leg_random_start = True' \
  "${REPO_ROOT}/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/dogleg_env_cfg.py"; then
  echo "Dogleg-specific random-reset IK is not enabled in the task config." >&2
  exit 1
fi
cd "${REPO_ROOT}"
export TERM="xterm-256color"
export PYTHONPATH="${REPO_ROOT}/source/agent_world:${REPO_ROOT}/source/agent_tasks:${REPO_ROOT}/source/agent_rl:${REPO_ROOT}"

echo "Task: ${TASK}"
echo "Asset: ${ASSET_USD}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Environments: ${NUM_ENVS}; fresh iterations: ${MAX_ITERATIONS}; zero-command fraction: ${ZERO_COMMAND_ENV_FRACTION}"
echo "Loading policy weights only; optimizer/iteration reset; dash starts at new iteration 2000."

train_args=(
  --task="${TASK}"
  --num_envs="${NUM_ENVS}"
  --max_iterations="${MAX_ITERATIONS}"
  --zero_command_env_fraction="${ZERO_COMMAND_ENV_FRACTION}"
  --stand_still_lin_vel_weight="${STAND_STILL_LIN_VEL_WEIGHT}"
  --checkpoint="${CHECKPOINT}"
  --run_name="${RUN_NAME}"
  --checkpoint_tag="${CHECKPOINT_TAG}"
  --device="${DEVICE}"
)
if [[ "${HEADLESS}" == true ]]; then
  train_args+=(--headless)
fi

exec "${ISAACLAB_SH}" -p scripts/rsl_rl/train.py \
  "${train_args[@]}"
