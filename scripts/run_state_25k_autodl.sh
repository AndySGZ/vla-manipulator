#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_FILE="/root/autodl-tmp/so101_state_true_25k_v1.status"
LOG_FILE="/root/autodl-tmp/so101_state_true_25k_v1.log"

printf 'TRAINING_STARTED %s\n' "$(date -Is)" > "${STATUS_FILE}"
if EXP_NAME=so101_state_true_25k_v1 TRAIN_STEPS=25000 \
    bash "${PROJECT_ROOT}/scripts/train_pi05_so101_lora_state.sh" > "${LOG_FILE}" 2>&1; then
  printf 'TRAIN_25000_COMPLETE %s\n' "$(date -Is)" > "${STATUS_FILE}"
else
  result=$?
  printf 'TRAINING_FAILED exit=%s %s\n' "${result}" "$(date -Is)" > "${STATUS_FILE}"
  exit "${result}"
fi
