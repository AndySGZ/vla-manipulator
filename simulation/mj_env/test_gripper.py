"""The jaw must grip on flat pads, and the calibration must describe those pads.

Two invariants keep the gripper honest:

1. Only the flat plates collide.  The curved finger meshes have a convex hull
   far wider than the visible mesh, and colliding them makes a grasp rest on a
   line, which lets an object sink 5-10 mm into the jaw.
2. ``calibrate`` measures the same plates, so "opening 35 mm" means the pads are
   35 mm apart to the controller, the collision engine and the renderer alike.
"""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from mj_env import gripper
from mj_env.scene import (
    FIXED_FINGER_GEOM,
    FIXED_PAD_GEOM,
    MOVING_FINGER_GEOM,
    MOVING_PAD_GEOM,
    PAD_THICKNESS,
    build_model,
)


def _geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


class GripperPadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_model()
        cls.calibration = gripper.calibrate(cls.model)

    def test_finger_meshes_do_not_collide(self) -> None:
        for name in (FIXED_FINGER_GEOM, MOVING_FINGER_GEOM):
            geom = _geom_id(self.model, name)
            self.assertEqual(self.model.geom_contype[geom], 0, name)
            self.assertEqual(self.model.geom_conaffinity[geom], 0, name)

    def test_pads_exist_and_collide(self) -> None:
        for name in (FIXED_PAD_GEOM, MOVING_PAD_GEOM):
            geom = _geom_id(self.model, name)
            self.assertGreaterEqual(geom, 0, name)
            self.assertEqual(self.model.geom_contype[geom], 1, name)
            self.assertEqual(self.model.geom_conaffinity[geom], 1, name)

    def test_opening_is_monotonic_and_spans_the_catalogue(self) -> None:
        openings = self.calibration.openings
        self.assertTrue(np.all(np.diff(openings) > 0), "opening must grow")
        # Narrowest catalogue grasp is the 16 mm dumbbell handle; widest is the
        # 40 mm ball.  Both have to land strictly inside the measured span.
        self.assertLess(self.calibration.min_opening, 0.016)
        self.assertGreater(self.calibration.max_opening, 0.040)

    def test_calibrated_opening_is_the_physical_pad_gap(self) -> None:
        """Sweep back from an opening to its angle and measure the pads again."""
        for width in (0.020, 0.035):
            angle = self.calibration.angle_for(width)
            data = mujoco.MjData(self.model)
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
            data.qpos[:] = self.model.qpos0
            data.qpos[self.model.jnt_qposadr[joint]] = angle
            mujoco.mj_forward(self.model, data)

            faces = [
                gripper._pad_face_points(self.model, data, name)
                for name in (FIXED_PAD_GEOM, MOVING_PAD_GEOM)
            ]
            from scipy.spatial import cKDTree

            gap = cKDTree(faces[0]).query(faces[1])[0].min()
            self.assertAlmostEqual(gap, width, delta=2.0 * PAD_THICKNESS, msg=f"{width}")


if __name__ == "__main__":
    unittest.main()
