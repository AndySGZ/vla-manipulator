"""Check one real LeRobot frame through the SO101 OpenPI action transforms.

Run from the official OpenPI venv after installing this project's SO101 config.
This tests plumbing, not policy prediction or closed-loop success.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi import transforms
from openpi.training import checkpoints, config


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_so101_lora")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    train_config = config.get_config(args.config)
    assert train_config.model.pi05 and not train_config.model.discrete_state_input
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    stats = checkpoints.load_norm_stats(args.checkpoint / "assets", data_config.asset_id)
    dataset = LeRobotDataset(train_config.data.repo_id, root=args.dataset_root)
    frame = dataset[args.index]
    state = _array(frame["state"])
    action = _array(frame["actions"])
    assert state.shape == action.shape == (6,), (state.shape, action.shape)
    raw = {
        "image": _array(frame["image"]),
        "wrist_image": _array(frame["wrist_image"]),
        "state": state.copy(),
        "actions": action[None, :].copy(),
        "prompt": "pick up the target object",
    }
    # Exact repack + data-side ordering used by the SO101 policy pipeline.
    for transform in data_config.repack_transforms.inputs:
        raw = transform(raw)
    for transform in data_config.data_transforms.inputs:
        raw = transform(raw)
    delta = raw["actions"].copy()
    normalized = transforms.Normalize(stats, use_quantiles=data_config.use_quantile_norm)(raw)
    restored = transforms.Unnormalize(stats, use_quantiles=data_config.use_quantile_norm)(normalized)
    for transform in data_config.data_transforms.outputs:
        restored = transform(restored)
    recovered = np.asarray(restored["actions"], dtype=np.float32)[0]
    error = float(np.max(np.abs(recovered - action)))
    print(f"index={args.index} state={state.tolist()} absolute={action.tolist()}")
    print(f"delta={delta[0].tolist()} recovered={recovered.tolist()} max_abs_error={error:.8g}")
    if error > args.atol:
        raise AssertionError(f"action roundtrip failed: {error} > {args.atol}")
    if not np.allclose(delta[0, :5], action[:5] - state[:5], atol=args.atol):
        raise AssertionError("joint delta does not equal action minus observed state")
    if not np.isclose(delta[0, 5], action[5], atol=args.atol):
        raise AssertionError("gripper should remain absolute")
    print("ROUNDTRIP_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
