"""Validate the separate SO101 state-input experiment before training."""

from __future__ import annotations

import dataclasses

from openpi.training import config as training_config


def main() -> None:
    old = training_config.get_config("pi05_so101_lora")
    new = training_config.get_config("pi05_so101_lora_state")
    assert old.model.discrete_state_input is False
    assert new.model.discrete_state_input is True
    assert old.model.action_horizon == new.model.action_horizon == 10
    assert old.batch_size == new.batch_size == 8
    assert old.data.repo_id == new.data.repo_id == "HF2147552431/so101-mujoco-pick"
    old = dataclasses.replace(old, exp_name="so101_all_v1")
    new = dataclasses.replace(new, exp_name="so101_state_true_25k_v1")
    assert old.checkpoint_dir != new.checkpoint_dir
    data = new.data.create(new.assets_dirs, new.model)
    assert data.norm_stats is not None, "normalization statistics were not loaded"
    assert "state" in data.norm_stats and "actions" in data.norm_stats
    print(f"config={new.name} state_input={new.model.discrete_state_input}")
    print(f"dataset={data.repo_id} norm_keys={sorted(data.norm_stats)}")
    print(f"checkpoint_dir={new.checkpoint_dir}")


if __name__ == "__main__":
    main()
