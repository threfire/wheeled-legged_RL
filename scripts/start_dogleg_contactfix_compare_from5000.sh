#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"
TASK="Robotics-Dogleg-Unified-Contactfix-Compare-v0"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/dogleg_flat_direct/2026-11-26_06-44-42_dogleg_contactfix_from1500_weights_20000_zero10_ik/model_5000_ik.pt}"
NUM_ENVS="${NUM_ENVS:-16384}"
# Same starting policy and total training budget as the compound-box run.
MAX_ITERATIONS="${MAX_ITERATIONS:-6000}"
RUN_NAME="${RUN_NAME:-dogleg_contactfix_compare_from5000}"
CHECKPOINT_TAG="cfcmp"
HEADLESS_MODE="${HEADLESS_MODE:-${HEADLESS:-true}}"
DEVICE="${DEVICE:-cuda:0}"
ASSET_USD="${REPO_ROOT}/source/agent_world/agent_world/assets/usd_files/dogleg_floating/configuration/dogleg_floating_physics_contactfix.usda"
CFG_FILE="${REPO_ROOT}/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/dogleg_contactfix_compare_env_cfg.py"

[[ -x "${ISAACLAB_SH}" ]] || { echo "Isaac Lab launcher not found: ${ISAACLAB_SH}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${ASSET_USD}" ]] || { echo "Contactfix USD not found: ${ASSET_USD}" >&2; exit 1; }
grep -Fq 'dogleg_floating_physics_contactfix.usda' "${CFG_FILE}" || {
  echo "Comparison config is not using contactfix.usda." >&2
  exit 1
}
[[ "${HEADLESS_MODE}" == true || "${HEADLESS_MODE}" == false ]] || { echo "HEADLESS_MODE must be true or false" >&2; exit 1; }
if [[ "${HEADLESS_MODE}" == true ]]; then
  export HEADLESS=1
else
  export HEADLESS=0
fi

cd "${REPO_ROOT}"
export TERM="xterm-256color"
export PYTHONPATH="${REPO_ROOT}/source/agent_world:${REPO_ROOT}/source/agent_tasks:${REPO_ROOT}/source/agent_rl:${REPO_ROOT}"

echo "Task: ${TASK}"
echo "Checkpoint: ${CHECKPOINT} (same starting policy as compound-box run; policy weights only)"
echo "Asset: ${ASSET_USD}"
echo "Other settings: inherited unchanged from current Unified Flat-v1"
echo "Training: ${MAX_ITERATIONS} fresh iterations; envs=${NUM_ENVS}; headless=${HEADLESS_MODE}."
echo "Control variable: same model_5000_ik.pt start; only collision USD differs."
echo "New checkpoints use suffix: _${CHECKPOINT_TAG}.pt"

train_args=(
  --task="${TASK}"
  --num_envs="${NUM_ENVS}"
  --max_iterations="${MAX_ITERATIONS}"
  --checkpoint="${CHECKPOINT}"
  --run_name="${RUN_NAME}"
  --checkpoint_tag="${CHECKPOINT_TAG}"
  --device="${DEVICE}"
)
if [[ "${HEADLESS_MODE}" == true ]]; then
  train_args+=(--headless)
fi

exec "${ISAACLAB_SH}" -p scripts/rsl_rl/train.py "${train_args[@]}"
