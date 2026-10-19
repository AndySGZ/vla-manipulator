"""Add a separate state-conditioned SO101 LoRA config to an OpenPI checkout.

The original state-free experiment is left untouched. The new config reuses
its dataset and normalization assets, but writes checkpoints under a new name.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_BEGIN = "    # BEGIN VLA-MANIPULATOR SO101 TRAIN CONFIG"
OLD_END = "    # END VLA-MANIPULATOR SO101 TRAIN CONFIG"
NEW_BEGIN = "    # BEGIN VLA-MANIPULATOR SO101 STATE TRAIN CONFIG"
NEW_END = "    # END VLA-MANIPULATOR SO101 STATE TRAIN CONFIG"


def render_config(source: str) -> str:
    if NEW_BEGIN in source or NEW_END in source:
        if NEW_BEGIN in source and NEW_END in source:
            return source
        raise RuntimeError("partial state-conditioned config found")
    if source.count(OLD_BEGIN) != 1 or source.count(OLD_END) != 1:
        raise RuntimeError("expected exactly one installed SO101 training config")

    old_block = source.split(OLD_BEGIN, 1)[1].split(OLD_END, 1)[0]
    if old_block.count('name="pi05_so101_lora"') != 1:
        raise RuntimeError("unexpected original SO101 config name")
    if old_block.count("discrete_state_input=False") != 2:
        raise RuntimeError("unexpected original SO101 state settings")
    if old_block.count("base_config=DataConfig(prompt_from_task=True),") != 1:
        raise RuntimeError("unexpected original SO101 data config")

    new_block = old_block.replace('name="pi05_so101_lora"', 'name="pi05_so101_lora_state"')
    new_block = new_block.replace("discrete_state_input=False", "discrete_state_input=True")
    new_block = new_block.replace(
        "base_config=DataConfig(prompt_from_task=True),",
        'assets=AssetsConfig(assets_dir="/root/autodl-tmp/assets/pi05_so101_lora"),\n'
        "            base_config=DataConfig(prompt_from_task=True),",
    )
    new_block = new_block.replace("save_interval=5_000,", "save_interval=25_000,")
    addition = NEW_BEGIN + new_block + NEW_END
    return source.replace(OLD_END, OLD_END + "\n" + addition, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = args.openpi_root / "src/openpi/training/config.py"
    original = config.read_text(encoding="utf-8")
    updated = render_config(original)
    if args.check:
        print("state config valid" if updated == original else "state config ready to install")
    elif updated != original:
        config.write_text(updated, encoding="utf-8")
        print("installed pi05_so101_lora_state")
    else:
        print("pi05_so101_lora_state already installed")


if __name__ == "__main__":
    main()
