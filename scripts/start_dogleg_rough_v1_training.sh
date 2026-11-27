#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_SH="/workspace/isaaclab/isaaclab.sh"
TASK="Robotics-Dogleg-Rough-v1"

# Edit these defaults, or override with e.g. NUM_ENVS=1024 HEADLESS=false ./scripts/...
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/dogleg_flat_direct/2026-11-26_06-44-42_dogleg_contactfix_from1500_weights_20000_zero10_ik/model_5000_ik.pt}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERATIONS="${MAX_ITERATIONS:-6000}"
# Default: 16384 environments without a graphical window.
HEADLESS="${HEADLESS:-true}"
PREVIEW_ENV_INDEX=0
PREVIEW_WIDTH=960
PREVIEW_HEIGHT=540
# 20 * 0.005 s = 0.1 simulation seconds between frames (10 Hz simulation time).
PREVIEW_RENDER_INTERVAL=20
DEVICE="${DEVICE:-cuda:0}"
RUN_NAME="${RUN_NAME:-dogleg_rough_v1_from5000_weights_${NUM_ENVS}}"
CHECKPOINT_TAG="rough"

[[ -x "${ISAACLAB_SH}" ]] || { echo "Isaac Lab launcher missing: ${ISAACLAB_SH}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint missing: ${CHECKPOINT}" >&2; exit 1; }
[[ "${HEADLESS}" == true || "${HEADLESS}" == false ]] || { echo "HEADLESS must be true or false" >&2; exit 1; }

cd "${REPO_ROOT}"
export TERM=xterm-256color
export PYTHONPATH="${REPO_ROOT}/source/agent_world:${REPO_ROOT}/source/agent_tasks:${REPO_ROOT}/source/agent_rl:${REPO_ROOT}"

echo "Task: ${TASK}; environments: ${NUM_ENVS}; new iterations: ${MAX_ITERATIONS}; headless: ${HEADLESS}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Weights only: fresh optimizer and iteration 0. Logs: logs/rsl_rl/dogleg_rough_v1"
echo "V14 Rough-v1 commands/terrain/state machines; Dogleg asset, IK and actuators; spring disabled."
echo "PhysX collision stack: 256 MiB; rigid patch capacity: 4194304."

train_args=(
  --task="${TASK}"
  --num_envs="${NUM_ENVS}"
  --max_iterations="${MAX_ITERATIONS}"
  --checkpoint="${CHECKPOINT}"
  --checkpoint_tag="${CHECKPOINT_TAG}"
  --run_name="${RUN_NAME}"
  --device="${DEVICE}"
)
if [[ "${HEADLESS}" == true ]]; then
  isaac_headless=1
  train_args+=(--headless)
else
  isaac_headless=0
  train_args+=(
    --rendering_mode=performance
    --preview_env_index="${PREVIEW_ENV_INDEX}"
    --preview_width="${PREVIEW_WIDTH}"
    --preview_height="${PREVIEW_HEIGHT}"
    --preview_render_interval="${PREVIEW_RENDER_INTERVAL}"
  )
fi

# Additional train.py / Hydra arguments can be supplied after the script name.
# Isaac Lab expects its HEADLESS environment variable to be integer 0/1,
# while this script accepts the human-readable true/false setting above.
exec env HEADLESS="${isaac_headless}" "${ISAACLAB_SH}" -p scripts/rsl_rl/train.py "${train_args[@]}" "$@"
