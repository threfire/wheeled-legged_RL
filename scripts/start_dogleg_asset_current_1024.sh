#!/usr/bin/env bash
set -euo pipefail

# Scratch baseline: current physics USD (simplified collision shapes and 10
# PhysX spherical closure joints). No checkpoint is loaded. The only intended
# experiment change versus the legacy script is the selected Dogleg USD/task.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_SH="/workspace/isaaclab/isaaclab.sh"
TASK="Robotics-Dogleg-Unified-Flat-v0"
RUN_NAME="dogleg_current_asset_scratch_1024_zero10"
NUM_ENVS=1024
MAX_ITERATIONS=2000
SEED=42
ZERO_COMMAND_ENV_FRACTION=0.1
STAND_STILL_LIN_VEL_WEIGHT=-1.0
DEVICE="cuda:0"
HEADLESS="${HEADLESS:-true}"

if [[ ! -x "${ISAACLAB_SH}" ]]; then
  echo "Isaac Lab launcher not found or not executable: ${ISAACLAB_SH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export TERM="xterm-256color"
export PYTHONPATH="${REPO_ROOT}/source/agent_world:${REPO_ROOT}/source/agent_tasks:${REPO_ROOT}/source/agent_rl:${REPO_ROOT}"

args=(
  --task="${TASK}"
  --num_envs="${NUM_ENVS}"
  --max_iterations="${MAX_ITERATIONS}"
  --seed="${SEED}"
  --zero_command_env_fraction="${ZERO_COMMAND_ENV_FRACTION}"
  --stand_still_lin_vel_weight="${STAND_STILL_LIN_VEL_WEIGHT}"
  --run_name="${RUN_NAME}"
  --checkpoint_tag="current"
  --device="${DEVICE}"
)
if [[ "${HEADLESS}" == "true" ]]; then
  args+=(--headless)
fi

exec "${ISAACLAB_SH}" -p scripts/rsl_rl/train.py "${args[@]}"
