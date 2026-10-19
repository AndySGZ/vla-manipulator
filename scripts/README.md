# Operational scripts

Run these on the AutoDL Linux server from the repository root.

```bash
# Set this to your real Hugging Face namespace. Use the same value every time.
export HF_DATASET_REPO=your-hf-username/so101-mujoco-pick

# 1. Install the guarded SO101 extension into the official OpenPI checkout.
bash scripts/prepare_openpi_so101.sh

# 2. Convert five local smoke episodes to LeRobot format.
bash scripts/convert_dataset.sh

# Optional: authenticate with `hf auth login`, then convert and upload privately.
bash scripts/convert_dataset.sh --push-to-hub

# 3. Compute q01/q99 normalization statistics using the official script.
bash scripts/compute_norm_stats.sh

# 4. Run a 10-step smoke train (default).
bash scripts/train_pi05_so101_lora.sh

# Longer run only after smoke validation.
EXP_NAME=so101_v1 TRAIN_STEPS=30000 bash scripts/train_pi05_so101_lora.sh

# 5. Serve a selected checkpoint.
bash scripts/serve_pi05_so101.sh \
  /root/autodl-tmp/checkpoints/pi05_so101_lora/so101_smoke/9
```

`check_server_storage.sh` refuses training below 20 GiB free space by default.
`archive_checkpoint.sh` creates a `.tar.zst` and uploads it to a Hugging Face
model repository; it never deletes the local checkpoint automatically.

Large datasets, model weights, logs and checkpoints are intentionally excluded
from Git.

## Completed comparison and diagnosis scripts

The 2026-10-18 state-input comparison used
`train_pi05_so101_lora_state.sh` through `run_state_25k_autodl.sh`, then
`eval_state_25k_replan5_max600.sh`. This is a separate experiment, not a
continuation of the old state-free checkpoint. The evaluation helper now lives
in `simulation/mj_env/scripts/eval_no_ping.py` rather than a temporary server
path. See `docs/so101_state_input_experiment_2026-10-18.md` for the actual
result and limitations.

The 2026-10-19 old-checkpoint diagnosis used
`diagnose_pi05_old25k_autodl.sh`. It serves the **old** 25k checkpoint with
`pi05_so101_lora` (`discrete_state_input=False`) and runs eight fixed cases.
The per-episode metrics come from `simulation/mj_env/scripts/diagnose_openpi.py`.
Supply a fresh `RUN_ID` on every run: the launcher refuses to overwrite an
existing status or result file. The dataset and motor adapter roundtrip checks
are `openpi_ext/check_so101_roundtrip.py` and
`simulation/mj_env/scripts/check_adapter_roundtrip.py`. See
`docs/so101_old25k_diagnosis_2026-10-19.md` for what was actually verified.

Raw diagnostic JSONL and evaluation logs stay outside Git; they were downloaded
and hash-checked in the user's local `SO101_25k_诊断记录` folder.
