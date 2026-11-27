#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"
TASK="Robotics-Dogleg-Unified-Flat-v1"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/dogleg_flat_direct/2026-11-26_06-44-42_dogleg_contactfix_from1500_weights_20000_zero10_ik/model_5000_ik.pt}"
NUM_ENVS="${NUM_ENVS:-16384}"
# Fresh training iterations after loading only the checkpoint policy weights.
MAX_ITERATIONS="${MAX_ITERATIONS:-6000}"
RUN_NAME="${RUN_NAME:-dogleg_unified_v1_weights5000_boxrec10}"
CHECKPOINT_TAG="rec10"
HEADLESS="${HEADLESS:-true}"
DEVICE="${DEVICE:-cuda:0}"
ASSET_USD="${REPO_ROOT}/source/agent_world/agent_world/assets/usd_files/dogleg_floating/configuration/dogleg_floating_physics_compound_candidate.usda"

[[ -x "${ISAACLAB_SH}" ]] || { echo "Isaac Lab launcher not found: ${ISAACLAB_SH}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${ASSET_USD}" ]] || { echo "Compound-box candidate Dogleg USD not found: ${ASSET_USD}" >&2; exit 1; }
grep -Fq 'dogleg_floating_physics_compound_candidate.usda' \
  "${REPO_ROOT}/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/dogleg_env_cfg.py" || {
  echo "Dogleg Unified Flat-v1 is not configured for the compound-box candidate USD." >&2
  exit 1
}
[[ "${HEADLESS}" == true || "${HEADLESS}" == false ]] || { echo "HEADLESS must be true or false" >&2; exit 1; }

cd "${REPO_ROOT}"
export TERM="xterm-256color"
export PYTHONPATH="${REPO_ROOT}/source/agent_world:${REPO_ROOT}/source/agent_tasks:${REPO_ROOT}/source/agent_rl:${REPO_ROOT}"

echo "Task: ${TASK} (ordinary lin_vel_x range restored to [-2.7, 2.7] m/s)"
echo "Asset: ${ASSET_USD}"
echo "Yaw command range and spin/dash yaw modes are inherited unchanged from v0."
echo "Recovery curriculum: 10% forward/backward fallen-pose resets from new iteration 1000; lateral-roll starts excluded."
echo "Checkpoint: ${CHECKPOINT}"
echo "Loading policy weights only; optimizer and iteration start fresh at 0."
echo "Training ${MAX_ITERATIONS} new iterations; envs=${NUM_ENVS}; headless=${HEADLESS}."
echo "New checkpoints use the single suffix: _${CHECKPOINT_TAG}.pt"

train_args=(
  --task="${TASK}"
  --num_envs="${NUM_ENVS}"
  --max_iterations="${MAX_ITERATIONS}"
  --checkpoint="${CHECKPOINT}"
  --run_name="${RUN_NAME}"
  --checkpoint_tag="${CHECKPOINT_TAG}"
  --device="${DEVICE}"
)
if [[ "${HEADLESS}" == true ]]; then
  train_args+=(--headless)
fi

exec "${ISAACLAB_SH}" -p scripts/rsl_rl/train.py "${train_args[@]}"
