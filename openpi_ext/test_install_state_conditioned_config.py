from __future__ import annotations

import unittest

from openpi_ext.install_state_conditioned_config import render_config


SOURCE = '''_CONFIGS = [
    # BEGIN VLA-MANIPULATOR SO101 TRAIN CONFIG
    TrainConfig(
        name="pi05_so101_lora",
        model=Pi0Config(discrete_state_input=False),
        data=LeRobotSO101DataConfig(
            base_config=DataConfig(prompt_from_task=True),
        ),
        freeze_filter=Pi0Config(discrete_state_input=False).get_freeze_filter(),
        save_interval=5_000,
    ),
    # END VLA-MANIPULATOR SO101 TRAIN CONFIG
]
'''


class StateConfigInstallerTest(unittest.TestCase):
    def test_installs_separate_config_idempotently(self) -> None:
        updated = render_config(SOURCE)
        self.assertIn('name="pi05_so101_lora"', updated)
        self.assertIn('name="pi05_so101_lora_state"', updated)
        self.assertEqual(updated.count("discrete_state_input=False"), 2)
        self.assertEqual(updated.count("discrete_state_input=True"), 2)
        self.assertIn('assets_dir="/root/autodl-tmp/assets/pi05_so101_lora"', updated)
        self.assertIn("save_interval=25_000,", updated)
        self.assertEqual(render_config(updated), updated)

    def test_refuses_incomplete_original_config(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected exactly one"):
            render_config("_CONFIGS = []\n")


if __name__ == "__main__":
    unittest.main()
