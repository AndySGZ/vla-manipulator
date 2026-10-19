#!/usr/bin/env bash
set -euo pipefail

# Small, fixed diagnostic sample from the previous 25k / max600 failures.
# Keep model, data and prior evaluation outputs untouched.
root=/root/autodl-tmp
project="$root/workspace/vla-manipulator"
checkpoint="$root/checkpoint_backups/pi05_so101_lora_so101_all_v1_step24999"
run_id="${RUN_ID:-so101_old25k_diagnosis}"
log="$root/$run_id.log"
serve_log="$root/${run_id}_serve.log"
status="$root/$run_id.status"
result="$root/$run_id.jsonl"

test -d "$checkpoint/params"
test -d "$checkpoint/assets"
test -d "$project/simulation/.venv"
test -d "$root/workspace/openpi/.venv"
if [ -e "$status" ] || [ -e "$result" ]; then
  echo "diagnostic run already exists: $run_id" >&2
  exit 1
fi
printf 'STARTED %s\n' "$(date -Is)" > "$status"

export OPENPI_DATA_HOME="$root/openpi-data"
export HF_HOME="$root/huggingface"
export UV_NO_SYNC=1
export PATH="/root/.local/bin:$PATH"
cd "$project"
bash scripts/serve_pi05_so101.sh "$checkpoint" > "$serve_log" 2>&1 &
serve_pid=$!
cleanup() {
  rc=$?
  kill "$serve_pid" 2>/dev/null || true
  wait "$serve_pid" 2>/dev/null || true
  if [ "$rc" -ne 0 ]; then
    printf 'FAILED exit=%s %s\n' "$rc" "$(date -Is)" > "$status"
  fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$serve_pid" 2>/dev/null; then
    printf 'SERVICE_FAILED %s\n' "$(date -Is)" > "$status"
    exit 1
  fi
  if (echo > /dev/tcp/127.0.0.1/8000) >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  printf 'SERVICE_TIMEOUT %s\n' "$(date -Is)" > "$status"
  exit 1
fi

cd "$project/simulation"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
"$project/simulation/.venv/bin/python" -m mj_env.scripts.diagnose_openpi \
  --host localhost --port 8000 --replan-steps 5 --max-steps 600 \
  --cases \
    rod_red:1003 rod_yellow:1000 sphere_blue:1003 sphere_green:1002 \
    cube_red:1000 cube_blue:1000 dumbbell_orange:1000 dumbbell_purple:1000 \
  --output "$result" | tee "$log"
printf 'DIAG_COMPLETE %s\n' "$(date -Is)" > "$status"
