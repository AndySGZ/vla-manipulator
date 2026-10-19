#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/root/autodl-tmp/workspace/openpi}"
EXP_NAME="${EXP_NAME:-so101_state_true_25k_v1}"
TRAIN_STEPS="${TRAIN_STEPS:-25000}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/datasets/lerobot_v1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/root/autodl-tmp/openpi-data}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export UV_NO_SYNC=1
export WANDB_MODE=offline
export PATH="/root/.local/bin:${PATH}"

"${OPENPI_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/openpi_ext/install_state_conditioned_config.py" --openpi-root "${OPENPI_ROOT}"
"${OPENPI_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/openpi_ext/check_state_experiment.py"
MIN_FREE_GB="${MIN_FREE_GB:-15}" bash "${PROJECT_ROOT}/scripts/check_server_storage.sh"
if [[ -e "/root/autodl-tmp/checkpoints/pi05_so101_lora_state/${EXP_NAME}" ]]; then
  echo "Refusing to overwrite existing state experiment: ${EXP_NAME}" >&2
  exit 1
fi
cd "${OPENPI_ROOT}"
uv run scripts/train.py pi05_so101_lora_state \
  --exp-name="${EXP_NAME}" \
  --num-train-steps="${TRAIN_STEPS}" \
  "$@"
