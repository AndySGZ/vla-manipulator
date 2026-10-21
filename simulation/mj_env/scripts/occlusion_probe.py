"""Size O3's occluding wall from measurements, not from a drawing.

The question is not "how tall can a wall be" -- the arm clears a 45 mm wall at
55 mm from the object with room to spare -- it is "where does a wall have to
stand, and how tall, for the wrist camera to actually lose sight of the grasp
point".  So this probe renders the wrist camera at the grasp pose three ways:

* object present, wall hidden  -> the object's own pixels (its mask)
* object hidden,  wall hidden  -> the reference for the mask difference
* object present, wall placed  -> what the camera actually sees

and reports, per candidate, the fraction of the object's mask that the wall
changes, alongside the smallest distance between the wall and any robot geom at
that pose.  A candidate is only usable if it occludes *and* stays clear.

Wall local azimuth matters because the wrist camera is mounted 60 mm toward the
gripper's local -y: which way that points in the table plane depends on the
wrist roll, which is itself a function of the object's heading.  That is why the
wall can be a rigid offset in the object's frame at all -- and why the azimuth has
to be found by measurement.

    PYTHONPATH=<WYSD> uv run python -m mj_env.scripts.occlusion_probe
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from mj_env.scene import WRIST_CAMERA
from mj_env.scripts.scripted_pick import (
    PAD_CLEARANCE,
    GraspSolver,
    PickTuning,
    _glide,
    grasp_axis_yaw,
)
from mj_env.task_env import TaskEnv

AZIMUTHS_DEG = (0, 10)
OFFSETS = (0.068, 0.076, 0.084, 0.092)
HEIGHTS = (0.034, 0.040, 0.046)
LENGTHS = (0.040, 0.060, 0.080, 0.100)
#: A *narrow* plate that extends radially away from the target, i.e. a partition
#: seen edge-on from the wrist camera.  The orientation is not a free choice:
#: the pads are arranged along the target's local +x, so anything long and
#: tangential within ~50 mm of the target runs straight through the fingers --
#: a 192-candidate sweep of tangential walls occluded 0% because every size that
#: would have occluded also collided.  A partition whose long axis points away
#: from the target clears the fingers and still crosses the line of sight.
LENGTHS = (0.100,)
WALL_THICKNESS = 0.006
#: How much of the object's mask the wall has to change to count as occlusion.
#: Below ~8% the wall is a corner in the frame; above ~70% the level becomes
#: "the object is gone", which is a different (and unfair) task.
MIN_OCCLUSION = 0.08
MAX_OCCLUSION = 0.70
#: Robot-to-wall clearance a candidate must keep at the grasp pose, in metres.
MIN_CLEARANCE = 0.004
FAR_AWAY = -10.0


def robot_geoms(env: TaskEnv) -> list[int]:
    """Every geom that is not the world, a free prop or scenery."""
    scenery = set(env._static_ids.values()) | {
        env._objects[name].body_id for name in env.spec.prop_bodies
    }
    geoms = []
    for geom in range(env.model.ngeom):
        body = int(env.model.geom_bodyid[geom])
        if body == 0 or body in scenery:
            continue
        geoms.append(geom)
    return geoms


def move_body(env: TaskEnv, body: int, position) -> None:
    """Move a *static* body (no joint): its pose lives in ``body_pos``."""
    env.model.body_pos[body] = position
    mujoco.mj_forward(env.model, env.data)


def move_free(env: TaskEnv, name: str, position) -> None:
    """Move a *free* body: its pose lives in ``qpos``, not ``body_pos``.

    Writing ``body_pos`` for a body with a freejoint does nothing at all, which
    is how an attempt to hide the object for a mask reference produced a clean
    empty mask: both renders were identical because neither had moved it.
    """
    joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
    if joint < 0:
        raise ValueError(f"{name} has no freejoint")
    address = int(env.model.jnt_qposadr[joint])
    env.data.qpos[address : address + 3] = position
    mujoco.mj_forward(env.model, env.data)


def render(env: TaskEnv, size: int) -> np.ndarray:
    renderer = mujoco.Renderer(env.model, size, size)
    try:
        renderer.update_scene(env.data, camera=WRIST_CAMERA)
        return np.asarray(renderer.render(), dtype=np.int16)
    finally:
        renderer.close()


def clearance(env: TaskEnv, wall_geom: int) -> float:
    """Smallest distance between the wall and any robot geom, in metres."""
    worst = float("inf")
    for geom in robot_geoms(env):
        fromto = np.zeros(6)
        distance = mujoco.mj_geomDistance(env.model, env.data, wall_geom, geom, 1.0, fromto)
        worst = min(worst, float(distance))
    return worst


def object_clearance(env: TaskEnv, wall_geom: int) -> float:
    """Distance between the wall and the target, in metres.

    Needed because a plate whose *inner end* reaches inside the target's
    rotational footprint looks fine at one heading and swallows the cube at
    another: the target's yaw is randomised, so its corners sweep a circle of
    ``max_radius`` -- 24.7 mm for the 35 mm cube, not the 17.5 mm its face
    suggests.
    """
    worst = float("inf")
    for name in env.spec.prop_bodies:
        body = env._objects[name].body_id
        for geom in range(env.model.ngeom):
            if int(env.model.geom_bodyid[geom]) != body:
                continue
            fromto = np.zeros(6)
            distance = mujoco.mj_geomDistance(
                env.model, env.data, wall_geom, geom, 1.0, fromto
            )
            worst = min(worst, float(distance))
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="o3_wrist_occlusion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--out", type=Path, default=Path(".tmp/occlusion_probe.json"))
    args = parser.parse_args()

    env = TaskEnv(args.task, max_episode_steps=10_000)
    solver = GraspSolver(env.model)
    wall_body = env._static_ids["occluder"]
    wall_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "occluder_geom")
    object_body = env._objects["object"].body_id
    try:
        env.reset(seed=args.seed)
        tuning = PickTuning()
        width = env._objects["object"].grip_width
        offset = solver.calibration.offset_for(width + 2.0 * PAD_CLEARANCE)
        aim = env.prop_pos("object").copy()
        solutions = solver.solve_waypoints(
            aim, grasp_axis_yaw(env, "object"), offset, tuning
        )
        if solutions is None:
            raise SystemExit("could not solve the grasp pose; nothing to measure")
        # Descend exactly the way the expert does, so the measured occlusion is
        # the occlusion the policy would actually experience.
        jaw = solver.calibration.angle_for(width + 2.0 * PAD_CLEARANCE)
        for waypoint, steps in (
            ("hover", tuning.steps_hover),
            ("reach", tuning.steps_reach),
            ("grasp", tuning.steps_descend),
        ):
            _glide(env, np.array([*solutions[waypoint], jaw]), steps)

        object_xy = env.prop_pos("object")[:2].copy()
        yaw = math.atan2(
            env.data.xmat[object_body].reshape(3, 3)[1, 0],
            env.data.xmat[object_body].reshape(3, 3)[0, 0],
        )
        home = np.asarray(env.model.body_pos[wall_body], dtype=float).copy()

        # The object's own pixels, from a wall-free pair.
        move_body(env, wall_body, [home[0], home[1], FAR_AWAY])
        with_object = render(env, args.size)
        rest = env.prop_pos("object").copy()
        move_free(env, "object", [rest[0], rest[1], FAR_AWAY])
        without_object = render(env, args.size)
        mask = np.abs(with_object - without_object).max(axis=2) > 12
        move_free(env, "object", rest)
        print(f"object mask: {int(mask.sum())} px of {args.size ** 2}")

        rows = []
        for azimuth_deg in AZIMUTHS_DEG:
            for wall_offset in OFFSETS:
                for height in HEIGHTS:
                    for length in LENGTHS:
                        angle = yaw + math.radians(azimuth_deg)
                        local = np.array([math.cos(angle), math.sin(angle)]) * wall_offset
                        # Long *radially* (away from the target), thin
                        # tangentially: a partition, not a fence.
                        env.model.geom_size[wall_geom] = [
                            length / 2.0,
                            WALL_THICKNESS / 2.0,
                            height / 2.0,
                        ]
                        env.model.geom_pos[wall_geom] = [0.0, 0.0, height / 2.0]
                        env.model.body_quat[wall_body] = [
                            math.cos(angle / 2.0),
                            0.0,
                            0.0,
                            math.sin(angle / 2.0),
                        ]
                        move_body(
                            env,
                            wall_body,
                            [object_xy[0] + local[0], object_xy[1] + local[1], 0.0],
                        )
                        with_wall = render(env, args.size)
                        changed = (np.abs(with_wall - with_object).max(axis=2) > 12) & mask
                        occlusion = float(changed.sum()) / max(int(mask.sum()), 1)
                        gap = clearance(env, wall_geom)
                        object_gap = object_clearance(env, wall_geom)
                        rows.append(
                            {
                                "azimuth_deg": azimuth_deg,
                                "offset_mm": wall_offset * 1000.0,
                                "height_mm": height * 1000.0,
                                "length_mm": length * 1000.0,
                                "occlusion": occlusion,
                                "clearance_mm": gap * 1000.0,
                                "object_clearance_mm": object_gap * 1000.0,
                            }
                        )

        for row in rows:
            usable = (
                MIN_OCCLUSION <= row["occlusion"] <= MAX_OCCLUSION
                and row["clearance_mm"] >= MIN_CLEARANCE * 1000.0
                and row.get("object_clearance_mm", 1e3) >= MIN_CLEARANCE * 1000.0
            )
            row["usable"] = usable
        usable = [row for row in rows if row["usable"]]
        usable.sort(key=lambda row: -row["occlusion"])
        print(f"{len(usable)}/{len(rows)} candidates occlude and stay clear")
        print(
            f"  {'azim':>5} {'offset':>7} {'height':>7} {'len':>6}"
            f" {'occlusion':>10} {'robot':>8} {'object':>8}"
        )
        for row in usable[:24]:
            print(
                f"  {row['azimuth_deg']:>5} {row['offset_mm']:>7.0f} {row['height_mm']:>7.0f}"
                f" {row['length_mm']:>6.0f} {row['occlusion'] * 100:>9.1f}%"
                f" {row['clearance_mm']:>7.1f}mm {row['object_clearance_mm']:>7.1f}mm"
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        import json

        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"report: {args.out}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())