"""Unit checks for the conservative symptom labels in diagnose_openpi."""

from __future__ import annotations

import unittest

from mj_env.scripts.diagnose_openpi import diagnose


def record(**overrides):
    result = {
        "success": False,
        "min_tcp_target_distance_m": 0.02,
        "ever_grasped": False,
        "max_lift_height_m": 0.0,
        "wrong_object_two_pad_contact_steps": 0,
        "clipped_action_steps": 0,
        "mean_joint_tracking_error_rad": 0.01,
    }
    result.update(overrides)
    return result


class DiagnoseLabelsTest(unittest.TestCase):
    def test_close_without_grasp(self):
        self.assertEqual(diagnose(record()), ["approached_but_no_two_pad_grasp"])

    def test_brief_grasp_without_lift_and_clipping(self):
        labels = diagnose(record(ever_grasped=True, max_lift_height_m=0.02, clipped_action_steps=7))
        self.assertEqual(labels, ["grasped_but_not_lifted_6cm", "action_outside_motor_range"])

    def test_success_is_not_mislabeled(self):
        self.assertEqual(diagnose(record(success=True, clipped_action_steps=7)), ["success"])

    def test_center_distance_is_only_proxy(self):
        self.assertEqual(
            diagnose(record(min_tcp_target_distance_m=0.08)),
            ["tcp_not_within_5cm_of_object_center"],
        )


if __name__ == "__main__":
    unittest.main()
