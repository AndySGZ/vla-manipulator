#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint=$base/checkpoints/pi05_so101_lora_state/so101_state_true_25k_v1/24999
service_log=$base/so101_state_25000_replan5_max600_serve.log
run_log=$base/so101_state_25000_replan5_max600_eval.log
status_log=$base/so101_state_25000_replan5_max600.status

test -f "$checkpoint/_CHECKPOINT_METADATA"
test -d "$checkpoint/params"
test -d "$checkpoint/assets"
test ! -e "$status_log"

export OPENPI_DATA_HOME=$base/openpi-data
export UV_NO_SYNC=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "EVAL_START $(date -Is)" > "$status_log"
cd "$base/workspace/openpi"
/root/.local/bin/uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_lora_state \
  --policy.dir="$checkpoint" > "$service_log" 2>&1 &
service_pid=$!
cleanup() {
  kill "$service_pid" 2>/dev/null || true
  wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for attempt in $(seq 1 180); do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo SERVICE_EXITED_EARLY | tee -a "$status_log"
    exit 1
  fi
  if (echo > /dev/tcp/127.0.0.1/8000) 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo SERVICE_NOT_READY | tee -a "$status_log"
  exit 1
fi
echo "SERVICE_READY $(date -Is)" | tee -a "$status_log"

cd "$base/workspace/vla-manipulator/simulation"
: > "$run_log"
for target in rod_red rod_yellow sphere_blue sphere_green cube_red cube_blue dumbbell_purple dumbbell_orange; do
  echo "===== ${target} =====" | tee -a "$run_log"
  /root/.local/bin/uv run python "$project/simulation/mj_env/scripts/eval_no_ping.py" \
    --host localhost --port 8000 --target "$target" \
    --seed 1000 --episodes 20 --replan-steps 5 --max-steps 600 \
    2>&1 | tee -a "$run_log"
done
echo "EVAL_COMPLETE $(date -Is)" | tee -a "$status_log"
