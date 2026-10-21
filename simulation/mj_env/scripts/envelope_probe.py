"""M0: SO101 桌面操作包线标定（envelope probe）。

把后面所有场景（P / O 族）的几何尺寸从"猜"变成"量"。只读环境，不改任何现有文件。

量八件事：

1. 可达带（r × 方位角）——IK 成功率；
2. 夹爪外廓（闭合方向 / 铰链方向）——决定 O4 缝宽、P3 容器内腔；
3. **竖直占用剖面** ``H(A)``：以目标正上方为中心、水平半径 ``A`` 以外，机械臂的**最低高度**。
   一条曲线同时回答三件事：容器壁高上限、O4 导墙高上限、O2 遮挡板能被抬到多高而不撞臂。
   （手指的顶点都在半径 ``A`` 以内，被条件自动排除，所以不需要手工挑 body。）
4. 倾斜接近可行性（O3 低梁：把腕部"压下去"之后还在不在 IK 可达范围内）；
5. 俯视相机下的遮挡几何——挡住目标一半需要多高的立板，以及这个高度会不会撞臂；
6. 专家轨迹扫掠包络 + 候选遮挡板的干涉检查；
7. 布局不变量复核（间距 / 桌面边界 / 底座 keep-out / 兜底触发率）；
8. 静置稳定性。

运行（仓库 ``simulation/`` 目录下）::

    uv run python -m mj_env.scripts.envelope_probe --quick
    uv run python -m mj_env.scripts.envelope_probe

输出 ``.tmp/envelope/envelope_report.json`` 与终端表。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from spatialmath import SE3

from so101 import SO101Arm

from mj_env import gripper, layout
from mj_env.env import HOME_QPOS, SO101TabletopEnv
from mj_env.scene import (
    FIXED_FINGER_GEOM,
    GLOBAL_CAMERA_FOVY,
    GLOBAL_CAMERA_POS,
    GRASP_ZONE_HALF_ANGLE,
    MOVING_FINGER_GEOM,
    TABLE_HEIGHT,
)
from mj_env.scripts.scripted_pick import (
    JAW_SQUEEZE,
    PAD_CLEARANCE,
    GraspSolver,
    PickTuning,
    grasp_axis_yaw,
    ik_seed,
)

ROBOT_ROOT_BODY = "base"

#: A 35 mm cube is the canonical object for every probe.
CUBE_WIDTH = 0.035
CUBE_HALF = 0.0175

_POSITION_TOLERANCE = 0.003
#: Radius at which the profile describes the hand/wrist shell rather than the fingers.
WRIST_RADIUS = 0.04


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _sphere_directions(count: int) -> np.ndarray:
    """Roughly uniform directions on the unit sphere (Fibonacci lattice)."""
    index = np.arange(count) + 0.5
    phi = np.arccos(1.0 - 2.0 * index / count)
    theta = np.pi * (1.0 + 5.0**0.5) * index
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1
    )


def geom_local_vertices(model: mujoco.MjModel, geom_id: int, *, ring: int = 24) -> np.ndarray:
    """Geom surface samples in the geom frame.  Meshes are exact, primitives sampled."""
    kind = model.geom_type[geom_id]
    size = np.asarray(model.geom_size[geom_id], dtype=float)

    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        return np.asarray(model.mesh_vert[start : start + count], dtype=float)

    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        return signs * size[:3]

    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return _sphere_directions(64) * size[0]

    if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        angles = np.linspace(0.0, 2.0 * np.pi, ring, endpoint=False)
        circle = np.stack([np.cos(angles), np.sin(angles), np.zeros(ring)], axis=1) * size[0]
        top = circle + np.array([0.0, 0.0, size[1]])
        bottom = circle - np.array([0.0, 0.0, size[1]])
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            caps = _sphere_directions(32) * size[0]
            return np.vstack([top, bottom, caps + np.array([0.0, 0.0, size[1]]),
                              caps - np.array([0.0, 0.0, size[1]])])
        return np.vstack([top, bottom])

    # Plane (infinite) or unsupported: fall back to the bounding sphere.
    return _sphere_directions(16) * float(model.geom_rbound[geom_id])


def _reduce_vertices(verts: np.ndarray, raw_budget: int) -> np.ndarray:
    """Convex-hull vertices plus a stride sample, to bound the point count.

    The hull preserves the part's outer envelope exactly (its vertices are a
    subset of the input), and the stride sample keeps concave regions -- a jaw
    opening, for instance -- represented.  Anything the reduction drops would
    have to be *inside* the hull, so the resulting envelope never understates
    how much room the arm takes.
    """
    if len(verts) <= raw_budget:
        return verts
    keep = np.unique(np.linspace(0, len(verts) - 1, raw_budget).astype(int))
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(verts)
        keep = np.union1d(keep, hull.vertices.astype(int))
    except Exception:  # degenerate or collinear geometry: the sample is enough
        pass
    return verts[keep]


def _robot_bodies(model: mujoco.MjModel, root_name: str) -> set[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root < 0:
        raise ValueError(f"scene has no body named {root_name!r}")
    keep: set[int] = set()
    for body in range(model.nbody):
        ancestor = body
        while ancestor != 0 and ancestor != root:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == root:
            keep.add(body)
    return keep


class RobotGeometry:
    """Cached local vertices per robot geom, transformed per frame."""

    #: Raw mesh vertices kept per geom, on top of the convex hull.  The STL
    #: parts carry ~360k vertices in total, which is far more than a
    #: millimetre-scale clearance question needs; the hull keeps the outer
    #: envelope exact and the stride samples keep concave regions represented.
    RAW_SAMPLES_PER_GEOM = 1500

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        bodies = _robot_bodies(model, ROBOT_ROOT_BODY)
        self.geom_ids = np.array(
            [
                geom
                for geom in range(model.ngeom)
                if int(model.geom_bodyid[geom]) in bodies
                and model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_PLANE
            ],
            dtype=int,
        )
        local, owner_geom = [], []
        mesh_names: dict[int, str] = {}
        raw_total = hull_total = 0
        for geom in self.geom_ids:
            raw = geom_local_vertices(model, int(geom))
            raw_total += len(raw)
            verts = _reduce_vertices(raw, self.RAW_SAMPLES_PER_GEOM)
            hull_total += len(verts)
            local.append(verts)
            owner_geom.append(np.full(len(verts), int(geom)))
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = int(model.geom_dataid[geom])
                mesh_names[int(geom)] = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
                )
        self.raw_vertex_total = raw_total
        self.used_vertex_total = hull_total
        self.local = np.vstack(local)
        self.geom_of_vertex = np.concatenate(owner_geom)
        self.mesh_name_of_geom = mesh_names
        self._slot = {int(geom): index for index, geom in enumerate(self.geom_ids)}

    @property
    def vertex_count(self) -> int:
        return len(self.local)

    def world(self, data: mujoco.MjData) -> np.ndarray:
        slots = np.array([self._slot[int(g)] for g in self.geom_of_vertex], dtype=int)
        rotation = data.geom_xmat[self.geom_ids].reshape(-1, 3, 3)[slots]
        position = data.geom_xpos[self.geom_ids][slots]
        return np.einsum("nij,nj->ni", rotation, self.local) + position

    def mesh_mask(self, *mesh_names: str) -> np.ndarray:
        """Vertices belonging to the named meshes."""
        wanted = {
            geom
            for geom, name in self.mesh_name_of_geom.items()
            if name in set(mesh_names)
        }
        return np.isin(self.geom_of_vertex, list(wanted))


def finger_mask(robot: RobotGeometry) -> np.ndarray:
    """Vertices of the two finger meshes (the parts that go *inside* a container)."""
    return robot.mesh_mask("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1")


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def apply_config(env: SO101TabletopEnv, q_arm: np.ndarray, jaw_angle: float) -> None:
    env.data.qpos[env._qpos_ids] = np.append(np.asarray(q_arm, dtype=float), jaw_angle)
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def aperture_midpoint(env: SO101TabletopEnv, offset: np.ndarray) -> np.ndarray:
    rotation = env.data.site_xmat[env._tcp_site_id].reshape(3, 3)
    return env.data.site_xpos[env._tcp_site_id] + rotation @ offset


def jaw_closing_axis(env: SO101TabletopEnv) -> np.ndarray:
    """World horizontal unit vector along which the two pads separate."""
    joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    low, high = env.model.jnt_range[joint]
    home = np.asarray(HOME_QPOS[:5], dtype=float)
    centroids = []
    for angle in (float(low), float(high)):
        apply_config(env, home, angle)
        fixed = gripper._pad_vertices(env.model, env.data, FIXED_FINGER_GEOM, joint).mean(axis=0)
        moving = gripper._pad_vertices(env.model, env.data, MOVING_FINGER_GEOM, joint).mean(axis=0)
        centroids.append(moving - fixed)
    axis = centroids[1] - centroids[0]
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("could not determine the jaw closing axis")
    return axis / norm


def solve_tilted(
    arm: SO101Arm, point: np.ndarray, psi: float, tilt: float, q0: np.ndarray
) -> tuple[np.ndarray | None, float]:
    """Vertical-grasp IK with the approach tilted by ``tilt`` about the tool +y."""
    pose = SE3(*point) * SE3.Ry(np.pi + tilt) * SE3.Rz(psi)
    limits = arm.joint_limits
    epsilon = 1e-8
    seed = np.clip(np.asarray(q0, dtype=float), limits[:, 0] + epsilon, limits[:, 1] - epsilon)
    best: tuple[np.ndarray | None, float] = (None, float("inf"))
    for mode in ("soft_orientation", "auto_soft_axis"):
        result = arm.inverse_pose(pose, q0=seed, mode=mode, return_result=True)
        error = float(result.position_error_norm)
        if result.success and error < best[1]:
            best = (np.asarray(result.q, dtype=float), error)
        if result.success and error < _POSITION_TOLERANCE:
            return np.asarray(result.q, dtype=float), error
    return best


def sector_profile(
    vertices: np.ndarray,
    center_xy: np.ndarray,
    radius: float,
    *,
    band: float = 0.010,
) -> dict:
    """``H(A)`` split by side: "near" is toward the robot base, "far" is away.

    A low bar for O3 sits on the near side (between the base and the object),
    so what matters is whether a given approach posture clears *that* side --
    a ring-wide minimum cannot see the difference.
    """
    offset = np.asarray(center_xy, dtype=float)
    norm = float(np.linalg.norm(offset))
    if norm < 1e-9:
        return {"near": None, "far": None}
    toward_base = -offset / norm
    radial = np.hypot(vertices[:, 0] - offset[0], vertices[:, 1] - offset[1])
    projection = (vertices[:, :2] - offset) @ toward_base
    inside = (radial >= radius) & (radial <= radius + band)
    heights = vertices[:, 2]
    near = inside & (projection > 0.0)
    far = inside & (projection < 0.0)
    return {
        "near": float(heights[near].min()) if near.any() else None,
        "far": float(heights[far].min()) if far.any() else None,
    }


def occupancy_profile(
    vertices: np.ndarray,
    center_xy: np.ndarray,
    radius_bins: np.ndarray,
    *,
    band: float = 0.010,
) -> np.ndarray:
    """``H(A)``: lowest robot height **in the annulus** ``A <= r <= A + band``.

    A wall standing at horizontal distance ``A`` from the object can be at most
    ``H(A)`` tall without the arm touching it.  An annulus rather than "r >= A"
    matters: the container wall is a thin ring at radius ``A``, so material at
    much larger radius (the forearm, the base) must not be counted.
    """
    radial = np.hypot(vertices[:, 0] - center_xy[0], vertices[:, 1] - center_xy[1])
    heights = vertices[:, 2]
    profile = np.empty(len(radius_bins), dtype=float)
    for index, radius in enumerate(radius_bins):
        inside = (radial >= radius) & (radial <= radius + band)
        profile[index] = float(heights[inside].min()) if inside.any() else float("nan")
    return profile


def lowest_in_annulus(
    vertices: np.ndarray, center_xy: np.ndarray, radius: float, *, band: float = 0.010
) -> dict:
    """Where the lowest material in an annulus sits, in world coordinates."""
    radial = np.hypot(vertices[:, 0] - center_xy[0], vertices[:, 1] - center_xy[1])
    inside = vertices[(radial >= radius) & (radial <= radius + band)]
    if not len(inside):
        return {"z_min": None, "x_min": None, "x_max": None, "y_min": None, "y_max": None}
    z_min = float(inside[:, 2].min())
    low = inside[inside[:, 2] <= z_min + 0.03]
    return {
        "z_min": z_min,
        "x_min": float(low[:, 0].min()),
        "x_max": float(low[:, 0].max()),
        "y_min": float(low[:, 1].min()),
        "y_max": float(low[:, 1].max()),
    }


#: World grid used for the swept-occupancy height map.  Covers the whole work
#: area with margin, at 2.5 mm resolution.
HEIGHTMAP_X = (-0.06, 0.36)
HEIGHTMAP_Y = (-0.32, 0.32)
HEIGHTMAP_Z_FLOOR = -0.05
HEIGHTMAP_Z_CEIL = 0.40
HEIGHTMAP_STEP = 0.0025


class HeightMap:
    """``Zmin[y, x]``: the lowest height the robot ever occupies at each (x, y).

    One scatter-min pass per swept frame, then every obstacle question -- a
    vertical board, a suspended plate, a ring-shaped container wall -- is a
    cheap lookup.  This replaces re-scanning the 100k-vertex cloud against
    every candidate obstacle.
    """

    def __init__(self) -> None:
        self.nx = int(round((HEIGHTMAP_X[1] - HEIGHTMAP_X[0]) / HEIGHTMAP_STEP))
        self.ny = int(round((HEIGHTMAP_Y[1] - HEIGHTMAP_Y[0]) / HEIGHTMAP_STEP))
        self.grid = np.full((self.ny, self.nx), HEIGHTMAP_Z_CEIL, dtype=float)
        self.frames = 0

    def add(self, vertices: np.ndarray) -> None:
        ix = np.floor((vertices[:, 0] - HEIGHTMAP_X[0]) / HEIGHTMAP_STEP).astype(int)
        iy = np.floor((vertices[:, 1] - HEIGHTMAP_Y[0]) / HEIGHTMAP_STEP).astype(int)
        keep = (
            (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            & (vertices[:, 2] >= HEIGHTMAP_Z_FLOOR)
        )
        if not keep.any():
            return
        flat = iy[keep] * self.nx + ix[keep]
        heights = vertices[keep, 2]
        order = np.argsort(flat, kind="stable")
        flat, heights = flat[order], heights[order]
        starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
        cell_min = np.minimum.reduceat(heights, starts)
        cells = flat[starts]
        self.grid.reshape(-1)[cells] = np.minimum(self.grid.reshape(-1)[cells], cell_min)
        self.frames += 1

    def _cell_slice(self, x_range, y_range) -> tuple[slice, slice]:
        x0 = max(0, int(np.floor((x_range[0] - HEIGHTMAP_X[0]) / HEIGHTMAP_STEP)))
        x1 = min(self.nx, int(np.ceil((x_range[1] - HEIGHTMAP_X[0]) / HEIGHTMAP_STEP)) + 1)
        y0 = max(0, int(np.floor((y_range[0] - HEIGHTMAP_Y[0]) / HEIGHTMAP_STEP)))
        y1 = min(self.ny, int(np.ceil((y_range[1] - HEIGHTMAP_Y[0]) / HEIGHTMAP_STEP)) + 1)
        return slice(y0, y1), slice(x0, x1)

    def lowest_in(self, x_range, y_range) -> float:
        ys, xs = self._cell_slice(x_range, y_range)
        block = self.grid[ys, xs]
        return float(block.min()) if block.size else float("inf")

    def lowest_in_ring(self, center_xy, radius: float, *, band: float = 0.010) -> float:
        """Lowest robot height in the annulus around ``center_xy`` (container wall)."""
        ys, xs = self._cell_slice(
            (center_xy[0] - radius - band, center_xy[0] + radius + band),
            (center_xy[1] - radius - band, center_xy[1] + radius + band),
        )
        block = self.grid[ys, xs]
        if not block.size:
            return float("inf")
        gy = HEIGHTMAP_Y[0] + (np.arange(ys.start, ys.stop) + 0.5) * HEIGHTMAP_STEP
        gx = HEIGHTMAP_X[0] + (np.arange(xs.start, xs.stop) + 0.5) * HEIGHTMAP_STEP
        radial = np.hypot(gx[None, :] - center_xy[0], gy[:, None] - center_xy[1])
        selected = block[(radial >= radius) & (radial <= radius + band)]
        return float(selected.min()) if selected.size else float("inf")

    def to_list(self) -> dict:
        return {
            "x_range": list(HEIGHTMAP_X),
            "y_range": list(HEIGHTMAP_Y),
            "step": HEIGHTMAP_STEP,
            "ny": self.ny,
            "nx": self.nx,
            "grid_mm": np.round(self.grid * 1000.0).astype(int).tolist(),
        }


# ---------------------------------------------------------------------------
# probe 1: reachable band
# ---------------------------------------------------------------------------
def classify_solve(arm: SO101Arm, point: np.ndarray, psi: float, q0: np.ndarray) -> str:
    """``exact`` = full vertical approach; ``position_only`` = the fallback mode.

    The distinction matters: ``auto_soft_axis`` gives up the hardest orientation
    component to keep the position, so it can place the fingertip but not hold
    the vertical approach.  Counting it as a success overstates the envelope.
    """
    pose = SE3(*point) * SE3.Ry(np.pi) * SE3.Rz(psi)
    limits = arm.joint_limits
    epsilon = 1e-8
    seed = np.clip(np.asarray(q0, dtype=float), limits[:, 0] + epsilon, limits[:, 1] - epsilon)
    for mode, label in (("soft_orientation", "exact"), ("auto_soft_axis", "position_only")):
        result = arm.inverse_pose(pose, q0=seed, mode=mode, return_result=True)
        if result.success and float(result.position_error_norm) < _POSITION_TOLERANCE:
            return label
    return "fail"


def probe_reach(arm: SO101Arm, *, quick: bool) -> dict:
    radii = np.arange(0.15, 0.335, 0.02 if quick else 0.01)
    azimuths = np.deg2rad(np.linspace(-60.0, 60.0, 5))
    rolls = np.deg2rad(np.linspace(-90.0, 67.5, 4 if quick else 8))
    heights = (CUBE_HALF, 0.10) if quick else (CUBE_HALF, 0.04, 0.08, 0.12, 0.18)

    rows = []
    for radius in radii:
        per_azimuth = []
        exact_total = position_total = fail_total = 0
        for azimuth in azimuths:
            counts = {"exact": 0, "position_only": 0, "fail": 0}
            for roll in rolls:
                for height in heights:
                    point = np.array(
                        [radius * math.cos(azimuth), radius * math.sin(azimuth), height]
                    )
                    counts[classify_solve(arm, point, float(roll), ik_seed(point, float(roll)))] += 1
            total = sum(counts.values())
            exact_total += counts["exact"]
            position_total += counts["position_only"]
            fail_total += counts["fail"]
            per_azimuth.append(counts["exact"] / total if total else 0.0)
        total = exact_total + position_total + fail_total
        rows.append(
            {
                "radius": float(radius),
                "exact_rate": exact_total / total if total else 0.0,
                "position_only_rate": position_total / total if total else 0.0,
                "fail_rate": fail_total / total if total else 0.0,
                "exact_rate_by_azimuth": per_azimuth,
            }
        )
    return {"radii": rows, "azimuths_deg": np.rad2deg(azimuths).tolist()}


# ---------------------------------------------------------------------------
# probe 2b: radius out to which a full cube grasp stays solvable
# ---------------------------------------------------------------------------
def probe_grasp_reach(env: SO101TabletopEnv, solver: GraspSolver, *, quick: bool) -> dict:
    """The number scene design actually needs: can the *expert* grasp there?

    A cube is placed at (r, azimuth) and the whole scripted grasp is solved --
    aperture correction included, which pushes the IK target up to ~2 cm off the
    object's centre and is what shrinks the usable band relative to a bare TCP
    reachability sweep.
    """
    tuning = PickTuning()
    radii = np.arange(0.16, 0.335, 0.02 if quick else 0.01)
    azimuths = np.deg2rad(np.linspace(-60.0, 60.0, 3 if quick else 5))
    metrics = env._objects["cube_red"]
    offset = solver.calibration.offset_for(CUBE_WIDTH + 2.0 * PAD_CLEARANCE)
    rows = []
    for radius in radii:
        solved = total = 0
        for azimuth in azimuths:
            total += 1
            x, y = radius * math.cos(azimuth), radius * math.sin(azimuth)
            env.data.qpos[metrics.qpos_adr : metrics.qpos_adr + 3] = (x, y, CUBE_HALF)
            env.data.qpos[metrics.qpos_adr + 3 : metrics.qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
            mujoco.mj_forward(env.model, env.data)
            try:
                solutions = solver.solve_waypoints(
                    np.array([x, y, CUBE_HALF]), 0.0, offset, tuning
                )
            except (ValueError, FloatingPointError):
                solutions = None
            solved += int(solutions is not None)
        rows.append({"radius": float(radius), "success_rate": solved / total if total else 0.0})
    return {"rows": rows, "azimuths_deg": np.rad2deg(azimuths).tolist()}


# ---------------------------------------------------------------------------
# probe 2/3: hand outline + vertical occupancy at the grasp pose
# ---------------------------------------------------------------------------
def probe_hand_and_occupancy(
    env: SO101TabletopEnv, robot: RobotGeometry, solver: GraspSolver, *, seeds: int
) -> dict:
    model = env.model
    tuning = PickTuning()
    radius_bins = np.arange(0.02, 0.121, 0.01 if seeds <= 2 else 0.005)
    fingers = finger_mask(robot)

    widths = {
        "closed": 0.002,
        "dumbbell_handle": 0.016,
        "cube_grip": CUBE_WIDTH,
        "cube_descent": CUBE_WIDTH + 2.0 * PAD_CLEARANCE,
        # What the scripted expert actually descends with: JAW_CLEARANCE is
        # 22 mm on top of the object's width, not the 8 mm the waypoint solver
        # uses.  A slot sized from `cube_descent` is too narrow by ~15 mm, and
        # the fixed finger grinds along the wall (measured: -4.6 mm penetration).
        "expert_descent": CUBE_WIDTH + 2.0 * 0.022,
        "dumbbell_disc": 0.044,
        "max_open": solver.calibration.max_opening,
    }

    outlines: dict[str, list[dict]] = {name: [] for name in widths}
    profiles: list[np.ndarray] = []
    deepest: list[dict] = []
    solved = 0

    for seed in range(seeds):
        _, info = env.reset(seed=seed, options={"target_object": "cube_red"})
        width = env._objects["cube_red"].grip_width
        aim = np.asarray(info["target_position"], dtype=float)
        offset = solver.calibration.offset_for(width + 2.0 * PAD_CLEARANCE)
        try:
            solutions = solver.solve_waypoints(
                aim, grasp_axis_yaw(env, "cube_red"), offset, tuning
            )
        except (ValueError, FloatingPointError):
            solutions = None
        if solutions is None:
            continue
        solved += 1
        axis = jaw_closing_axis(env)

        for name, open_width in widths.items():
            apply_config(env, solutions["grasp"], solver.calibration.angle_for(open_width))
            point = aperture_midpoint(env, solver.calibration.offset_for(open_width))
            fingers_only = robot.world(env.data)[fingers]
            delta = fingers_only - point
            along = delta @ axis
            across = delta @ np.array([-axis[1], axis[0], 0.0])
            outlines[name].append(
                {
                    "outer_half_span_closing": float(max(abs(along.min()), abs(along.max()))),
                    "outer_half_span_hinge": float(max(abs(across.min()), abs(across.max()))),
                    "span_closing": float(along.max() - along.min()),
                    "span_hinge": float(across.max() - across.min()),
                }
            )

        apply_config(
            env, solutions["grasp"], solver.calibration.angle_for(width + 2.0 * PAD_CLEARANCE)
        )
        vertices = robot.world(env.data)
        profiles.append(occupancy_profile(vertices, aim[:2], radius_bins))
        deepest.append(
            {
                "seed": seed,
                "object_centre_z": float(aim[2]),
                "object_top_z": TABLE_HEIGHT + 2.0 * CUBE_HALF,
                **lowest_in_annulus(vertices, aim[:2], WRIST_RADIUS),
            }
        )

    profile = (
        np.nanmin(np.vstack(profiles), axis=0) if profiles else np.full_like(radius_bins, np.nan)
    )
    table = [
        {"radius": float(radius), "wall_height_limit": float(height - TABLE_HEIGHT)}
        for radius, height in zip(radius_bins, profile)
    ]
    summary = {
        name: {
            "outer_half_span_closing": max(row["outer_half_span_closing"] for row in rows),
            "outer_half_span_hinge": max(row["outer_half_span_hinge"] for row in rows),
            "span_closing": max(row["span_closing"] for row in rows),
            "span_hinge": max(row["span_hinge"] for row in rows),
        }
        for name, rows in outlines.items()
        if rows
    }

    grasp_vertices = np.asarray(
        [[row["z_min"] if row["z_min"] is not None else np.nan] for row in deepest]
    )
    return {
        "seeds_solved": solved,
        "seeds_total": seeds,
        "jaw_opening": {
            "min_mm": solver.calibration.min_opening * 1000.0,
            "max_mm": solver.calibration.max_opening * 1000.0,
        },
        "outline": summary,
        "wall_height_limit_by_inner_half_width": table,
        "wrist_radius": WRIST_RADIUS,
        "grasp_pose": {
            "wrist_lowest_z": float(np.nanmin(grasp_vertices)) if len(grasp_vertices) else None,
            "per_seed": deepest,
        },
    }


# ---------------------------------------------------------------------------
# probe 4: tilted approach feasibility (O3)
# ---------------------------------------------------------------------------
def probe_tilt(env: SO101TabletopEnv, arm: SO101Arm, robot: RobotGeometry, *, quick: bool) -> dict:
    """Does tilting the approach lift the wrist out of the way?  (O3 low bar.)

    Reports, per requested tilt, whether the IK still holds the position, the
    tilt actually achieved (the TCP site's +x axis is the approach direction),
    and ``H(A)`` at the radii a beam would sit at.  A beam at radius ``A`` is
    passable exactly when the arm's lowest material there is above the beam.
    """
    tilts_deg = [0.0, 10.0, 20.0, 30.0, 40.0, -10.0, -20.0, -30.0, -40.0]
    if quick:
        tilts_deg = [0.0, 20.0, -20.0]
    rolls_deg = [-90.0, -45.0, 0.0, 45.0] if not quick else [-45.0, 0.0]
    probe_radii = (0.050, 0.060, 0.070)
    point = np.array([0.22, 0.0, CUBE_HALF])
    rows = []
    for tilt_deg in tilts_deg:
        tilt = math.radians(tilt_deg)
        feasible = total = 0
        errors: list[float] = []
        achieved: list[float] = []
        profiles: list[np.ndarray] = []
        x_low: list[float] = []
        x_high: list[float] = []
        sector_values = {
            int(radius * 1000): {"near": [], "far": []} for radius in probe_radii
        }
        for roll_deg in rolls_deg:
            roll = math.radians(roll_deg)
            total += 1
            q, error = solve_tilted(arm, point, roll, tilt, ik_seed(point, roll))
            if q is None or error >= _POSITION_TOLERANCE:
                continue
            feasible += 1
            errors.append(error)
            apply_config(env, q, 0.02)
            rotation = env.data.site_xmat[env._tcp_site_id].reshape(3, 3)
            approach = rotation[:, 0]  # site +x points out through the fingers
            achieved.append(
                math.degrees(
                    math.acos(float(np.clip(abs(approach @ np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))
                )
            )
            vertices = robot.world(env.data)
            profiles.append(occupancy_profile(vertices, point[:2], np.array(probe_radii)))
            for radius in probe_radii:
                measured_sector = sector_profile(vertices, point[:2], radius)
                sector_values[int(radius * 1000)]["near"].append(measured_sector["near"])
                sector_values[int(radius * 1000)]["far"].append(measured_sector["far"])
            measured = lowest_in_annulus(vertices, point[:2], probe_radii[0])
            if measured["x_min"] is not None:
                x_low.append(measured["x_min"] - point[0])
                x_high.append(measured["x_max"] - point[0])
        row: dict[str, Any] = {
            "tilt_deg": tilt_deg,
            "feasible": feasible,
            "total": total,
            "position_error_max_mm": max(errors) * 1000.0 if errors else None,
            "achieved_tilt_deg": float(np.mean(achieved)) if achieved else None,
            "lowest_x_min_rel": min(x_low) if x_low else None,
            "lowest_x_max_rel": max(x_high) if x_high else None,
        }
        for radius in probe_radii:
            key = int(radius * 1000)
            near = [value for value in sector_values[key]["near"] if value is not None]
            far = [value for value in sector_values[key]["far"] if value is not None]
            row[f"near_wall_height_at_{key}mm"] = min(near) - TABLE_HEIGHT if near else None
            row[f"far_wall_height_at_{key}mm"] = min(far) - TABLE_HEIGHT if far else None
        if profiles:
            stacked = np.vstack(profiles)
            for index, radius in enumerate(probe_radii):
                row[f"wall_height_limit_at_{int(radius * 1000)}mm"] = float(
                    np.nanmin(stacked[:, index]) - TABLE_HEIGHT
                )
        else:
            for radius in probe_radii:
                row[f"wall_height_limit_at_{int(radius * 1000)}mm"] = None
        rows.append(row)
    return {"point": point.tolist(), "probe_radii": list(probe_radii), "rows": rows}


# ---------------------------------------------------------------------------
# probe 5: occluder geometry under the top-down camera (O2)
# ---------------------------------------------------------------------------
def _ray_plane_crossing(
    camera: np.ndarray, points: np.ndarray, axis: int, value: float
) -> tuple[np.ndarray, np.ndarray]:
    denominator = points[:, axis] - camera[axis]
    usable = np.abs(denominator) > 1e-12
    t = np.full(len(points), np.nan)
    t[usable] = (value - camera[axis]) / denominator[usable]
    inside = usable & (t > 0.0) & (t < 1.0)
    crossings = camera + t[inside, None] * (points[inside] - camera)
    return crossings, np.flatnonzero(inside)


def probe_occluder(
    env: SO101TabletopEnv,
    *,
    seeds: int,
    distances: tuple[float, ...] = (0.030, 0.045, 0.060, 0.080, 0.100),
) -> dict:
    """Board height needed to block the object from the top-down camera.

    Swept over the board's distance from the object, because shorter boards need
    to sit closer, and "closer" is exactly where the jaws are.  Whether a given
    (distance, height) also clears the arm is checked later against the swept
    height map.
    """
    camera = np.asarray(GLOBAL_CAMERA_POS, dtype=float)
    grid = np.linspace(-CUBE_HALF * 0.9, CUBE_HALF * 0.9, 9)
    local_top = np.array([[x, y, CUBE_HALF] for x in grid for y in grid], dtype=float)
    rows = []
    for seed in range(seeds):
        _, info = env.reset(seed=seed, options={"target_object": "cube_red"})
        centre = np.asarray(info["target_position"], dtype=float)
        body = env._objects["cube_red"].body_id
        rotation = env.data.xmat[body].reshape(3, 3)
        world_top = local_top @ rotation.T + centre

        for axis, label in ((1, "y_side"), (0, "x_side")):
            sign = 1.0 if centre[axis] > camera[axis] else -1.0
            for distance in distances:
                value = centre[axis] - sign * distance
                base = {
                    "seed": seed,
                    "side": label,
                    "axis": axis,
                    "distance": distance,
                    "object_centre": centre.tolist(),
                    "board_plane_value": float(value),
                }
                if (value - camera[axis]) * sign < 0.005:
                    rows.append({**base, "feasible": False, "reason": "板落在相机轴之外，挡不住"})
                    continue
                crossings, _ = _ray_plane_crossing(camera, world_top, axis, value)
                if len(crossings) < 4:
                    rows.append({**base, "feasible": False, "reason": "没有射线穿过板平面"})
                    continue
                heights = crossings[:, 2]
                spread = crossings[:, 1 - axis]
                entry = {
                    **base,
                    "feasible": True,
                    "board_span_needed": [float(spread.min()), float(spread.max())],
                }
                for target in (0.25, 0.5, 0.65):
                    low, high = float(heights.min()) - 0.02, float(heights.max()) + 0.02
                    for _ in range(40):
                        mid = 0.5 * (low + high)
                        if float(np.mean(heights <= mid)) < target:
                            low = mid
                        else:
                            high = mid
                    entry[f"board_height_for_{int(target * 100)}pct"] = float(high - TABLE_HEIGHT)
                    entry[f"board_top_for_{int(target * 100)}pct"] = float(high)
                rows.append(entry)
    return {
        "camera_pos": camera.tolist(),
        "fovy_deg": GLOBAL_CAMERA_FOVY,
        "distances": list(distances),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# probe 6: expert sweep envelope + board interference check
# ---------------------------------------------------------------------------
def probe_sweep(
    env: SO101TabletopEnv,
    robot: RobotGeometry,
    solver: GraspSolver,
    *,
    seeds: int,
    radius_bins: np.ndarray,
    boards: list[dict],
) -> dict:
    """Sweep the scripted expert and record everywhere the arm may ever be.

    Produces (a) the overall AABB, (b) the ring profile ``H(A)``, and (c) a
    per-frame ``Zmin(y, x)`` height map that every candidate obstacle is then
    checked against.
    """
    tuning = PickTuning()
    frames = (
        tuning.steps_hover,
        tuning.steps_reach,
        tuning.steps_descend,
        tuning.steps_close,
        tuning.steps_lift,
        tuning.steps_hold,
    )
    profiles: list[np.ndarray] = []
    near_profiles: list[np.ndarray] = []
    far_profiles: list[np.ndarray] = []
    aabbs: list[dict] = []
    solved = 0
    board_results: list[dict] = []
    first_heightmap: dict | None = None

    for seed in range(seeds):
        _, info = env.reset(seed=seed, options={"target_object": "cube_red"})
        width = env._objects["cube_red"].grip_width
        aim = np.asarray(info["target_position"], dtype=float)
        offset = solver.calibration.offset_for(width + 2.0 * PAD_CLEARANCE)
        try:
            solutions = solver.solve_waypoints(
                aim, grasp_axis_yaw(env, "cube_red"), offset, tuning
            )
        except (ValueError, FloatingPointError):
            solutions = None
        if solutions is None:
            continue
        solved += 1
        heightmap = HeightMap()

        jaw_open = solver.calibration.angle_for(width + 2.0 * 0.004)
        jaw_shut = solver.calibration.angle_for(max(width - JAW_SQUEEZE, 0.0))
        sequence = [
            (np.array([*solutions["hover"], jaw_open]), frames[0]),
            (np.array([*solutions["reach"], jaw_open]), frames[1]),
            (np.array([*solutions["grasp"], jaw_open]), frames[2]),
            (np.array([*solutions["grasp"], jaw_shut]), frames[3]),
            (np.array([*solutions["lift"], jaw_shut]), frames[4]),
            (np.array([*solutions["lift"], jaw_shut]), frames[5]),
        ]
        start = env.data.ctrl[env._actuator_ids].copy()
        low = np.array([np.inf, np.inf, np.inf])
        high = -low.copy()
        for target, steps in sequence:
            for index in range(steps):
                alpha = (index + 1) / steps
                action = start + alpha * (np.asarray(target, dtype=float) - start)
                env.step(np.asarray(action, dtype=np.float32), capture_observation=False)
                start = np.asarray(action, dtype=float)
                vertices = robot.world(env.data)
                low = np.minimum(low, vertices.min(axis=0))
                high = np.maximum(high, vertices.max(axis=0))
                profiles.append(occupancy_profile(vertices, aim[:2], radius_bins))
                near_row, far_row = [], []
                for radius in radius_bins:
                    measured_sector = sector_profile(vertices, aim[:2], float(radius))
                    near_row.append(
                        measured_sector["near"]
                        if measured_sector["near"] is not None
                        else np.nan
                    )
                    far_row.append(
                        measured_sector["far"] if measured_sector["far"] is not None else np.nan
                    )
                near_profiles.append(np.asarray(near_row))
                far_profiles.append(np.asarray(far_row))
                heightmap.add(vertices)
        aabbs.append({"seed": seed, "min": low.tolist(), "max": high.tolist()})

        for board in boards:
            if int(board["seed"]) != seed:
                continue
            lowest = heightmap.lowest_in(board["x_range"], board["y_range"])
            board_results.append(
                {
                    "key": board["key"],
                    "side": board["side"],
                    "distance": board["distance"],
                    "percent": board["percent"],
                    "height": board["height"],
                    "robot_lowest_z_there": lowest,
                    "margin": board["height"] - lowest,
                    "interference": bool(lowest < board["height"]),
                }
            )
        if first_heightmap is None:
            first_heightmap = heightmap.to_list()

    profile = (
        np.nanmin(np.vstack(profiles), axis=0) if profiles else np.full_like(radius_bins, np.nan)
    )
    return {
        "seeds_solved": solved,
        "seeds_total": seeds,
        "frames": len(profiles),
        "aabb_over_seeds": {
            "min": np.min([row["min"] for row in aabbs], axis=0).tolist() if aabbs else None,
            "max": np.max([row["max"] for row in aabbs], axis=0).tolist() if aabbs else None,
        },
        "wall_height_limit_by_inner_half_width": [
            {"radius": float(radius), "wall_height_limit": float(height - TABLE_HEIGHT)}
            for radius, height in zip(radius_bins, profile)
        ],
        "near_wall_height_limit_by_inner_half_width": [
            {"radius": float(radius), "wall_height_limit": float(height - TABLE_HEIGHT)}
            for radius, height in zip(
                radius_bins,
                np.nanmin(np.vstack(near_profiles), axis=0)
                if near_profiles
                else np.full_like(radius_bins, np.nan),
            )
        ],
        "far_wall_height_limit_by_inner_half_width": [
            {"radius": float(radius), "wall_height_limit": float(height - TABLE_HEIGHT)}
            for radius, height in zip(
                radius_bins,
                np.nanmin(np.vstack(far_profiles), axis=0)
                if far_profiles
                else np.full_like(radius_bins, np.nan),
            )
        ],
        "board_results": board_results,
        "feasible_boards": [row["key"] for row in board_results if not row["interference"]],
        "heightmap_example": first_heightmap,
    }


# ---------------------------------------------------------------------------
# probe 7: layout invariants
# ---------------------------------------------------------------------------
def probe_layout(env: SO101TabletopEnv, *, samples: int) -> dict:
    rng = np.random.default_rng(0)
    footprints = env._footprints
    fallbacks = 0
    min_clearance = float("inf")
    min_base = float("inf")
    out_of_bounds = 0
    for _ in range(samples):
        positions, yaws, fallback = layout.sample_layout(
            rng,
            footprints,
            radius_range=(0.17, 0.28),
            half_angle=math.radians(60.0),
            base_keepout=0.12,
            margin=0.012,
            bounds_x=(-0.02, 0.82),
            bounds_y=(-0.39, 0.39),
        )
        fallbacks += int(fallback)
        for i in range(len(footprints)):
            min_base = min(
                min_base,
                layout.point_clearance(footprints[i], positions[i], yaws[i], (0.0, 0.0)),
            )
            for j in range(i + 1, len(footprints)):
                min_clearance = min(
                    min_clearance,
                    layout.clearance(
                        footprints[i], positions[i], yaws[i],
                        footprints[j], positions[j], yaws[j],
                    ),
                )
            a0, a1 = layout.spine(footprints[i], positions[i], yaws[i])
            span_low = np.minimum(a0, a1) - footprints[i].radius
            span_high = np.maximum(a0, a1) + footprints[i].radius
            if span_low[0] < -0.02 or span_high[0] > 0.82 or span_low[1] < -0.39 or span_high[1] > 0.39:
                out_of_bounds += 1
    return {
        "samples": samples,
        "objects": list(env.objects_subset),
        "fallback_rate": fallbacks / samples,
        "min_pairwise_clearance": min_clearance,
        "min_base_keepout_clearance": min_base,
        "bounds_violations": out_of_bounds,
    }


# ---------------------------------------------------------------------------
# probe 8: settle stability
# ---------------------------------------------------------------------------
def probe_settle(env: SO101TabletopEnv, *, seeds: int, steps: int = 250) -> dict:
    drifts: list[float] = []
    tilts: list[float] = []
    per_object: dict[str, float] = {}
    for seed in range(seeds):
        env.reset(seed=seed, options={"target_object": "cube_red"})
        before = {
            name: (
                np.asarray(env.data.xpos[m.body_id], dtype=float).copy(),
                env.data.xmat[m.body_id].reshape(3, 3).copy(),
            )
            for name, m in env._objects.items()
        }
        for _ in range(steps):
            mujoco.mj_step(env.model, env.data)
        for name, m in env._objects.items():
            after = np.asarray(env.data.xpos[m.body_id], dtype=float)
            drift = float(np.linalg.norm(after - before[name][0]))
            drifts.append(drift)
            per_object[name] = max(per_object.get(name, 0.0), drift)
            rotation = env.data.xmat[m.body_id].reshape(3, 3)
            cosine = float(np.clip((rotation @ before[name][1].T)[2, 2], -1.0, 1.0))
            tilts.append(math.degrees(math.acos(cosine)))
    return {
        "seeds": seeds,
        "steps": steps,
        "seconds": steps * float(env.model.opt.timestep),
        "max_drift_m": max(drifts),
        "mean_drift_m": float(np.mean(drifts)),
        "max_tilt_deg": max(tilts),
        "max_drift_by_object": per_object,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _print_occupancy(rows: list[dict], indent: str = "    ") -> None:
    print(f"{indent}A(mm)   H_max(mm)")
    for row in rows:
        radius = row["radius"]
        if radius > 0.1001 and radius < 0.1199:
            continue
        print(f"{indent}{radius * 1000:>5.0f}   {row['wall_height_limit'] * 1000:>8.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="coarser grids")
    parser.add_argument("--seeds", type=int, default=0, help="layouts per trajectory probe")
    parser.add_argument("--out", type=Path, default=Path(".tmp/envelope/envelope_report.json"))
    parser.add_argument(
        "--only",
        default="all",
        help="逗号分隔子集：reach,hand,tilt,occluder,sweep,layout,settle",
    )
    args = parser.parse_args()

    selected = (
        {"reach", "hand", "tilt", "occluder", "sweep", "layout", "settle"}
        if args.only == "all"
        else {piece.strip() for piece in args.only.split(",") if piece.strip()}
    )
    if {"occluder", "sweep"} & selected:
        selected.add("hand")

    seeds = args.seeds or (2 if args.quick else 5)
    started = time.perf_counter()

    env = SO101TabletopEnv(max_episode_steps=10_000)
    arm = SO101Arm()
    solver = GraspSolver(env.model)
    robot = RobotGeometry(env.model)
    radius_bins = np.arange(0.02, 0.121, 0.01 if args.quick else 0.005)

    print(
        f"mujoco {mujoco.__version__}; robot vertices {robot.raw_vertex_total}"
        f" → {robot.used_vertex_total} (hull + sample)"
    )
    print(
        f"jaw opening {solver.calibration.min_opening * 1000:.1f}"
        f"-{solver.calibration.max_opening * 1000:.1f} mm"
    )

    report: dict[str, Any] = {
        "mujoco_version": mujoco.__version__,
        "quick": bool(args.quick),
        "grasp_zone_radius": [0.17, 0.28],
        "grasp_zone_half_angle_deg": math.degrees(GRASP_ZONE_HALF_ANGLE),
    }

    if "reach" in selected:
        t0 = time.perf_counter()
        report["reach"] = probe_reach(arm, quick=args.quick)
        print(f"  [1] reach {time.perf_counter() - t0:.1f}s", flush=True)

        t0 = time.perf_counter()
        report["grasp_reach"] = probe_grasp_reach(env, solver, quick=args.quick)
        print(f"  [1b] grasp reach {time.perf_counter() - t0:.1f}s", flush=True)

    if "hand" in selected:
        t0 = time.perf_counter()
        report["hand"] = probe_hand_and_occupancy(env, robot, solver, seeds=seeds)
        print(f"  [2/3] hand+occupancy {time.perf_counter() - t0:.1f}s", flush=True)

    if "tilt" in selected:
        t0 = time.perf_counter()
        report["tilt"] = probe_tilt(env, arm, robot, quick=args.quick)
        print(f"  [4] tilt {time.perf_counter() - t0:.1f}s", flush=True)

    if "occluder" in selected:
        jaw_span = max(
            report["hand"]["outline"]["cube_descent"]["outer_half_span_closing"],
            report["hand"]["outline"]["cube_descent"]["outer_half_span_hinge"],
        )
        report["jaw_half_span_cube_descent"] = jaw_span
        t0 = time.perf_counter()
        report["occluder"] = probe_occluder(env, seeds=seeds)
        print(f"  [5] occluder {time.perf_counter() - t0:.1f}s", flush=True)

    boards: list[dict] = []
    thickness = 0.008
    for row in report.get("occluder", {}).get("rows", []):
        if not row.get("feasible"):
            continue
        axis = int(row["axis"])
        plane = float(row["board_plane_value"])
        low_span, high_span = row["board_span_needed"]
        for percent in (25, 50, 65):
            x_range = (
                (low_span, high_span)
                if axis == 1
                else (plane - thickness / 2.0, plane + thickness / 2.0)
            )
            y_range = (
                (plane - thickness / 2.0, plane + thickness / 2.0)
                if axis == 1
                else (low_span, high_span)
            )
            boards.append(
                {
                    "key": f"{row['side']}_d{int(row['distance'] * 1000)}_"
                    f"{percent}pct_seed{row['seed']}",
                    "side": row["side"],
                    "axis": axis,
                    "seed": row["seed"],
                    "distance": row["distance"],
                    "percent": percent,
                    "top": float(row[f"board_top_for_{percent}pct"]),
                    "height": float(row[f"board_height_for_{percent}pct"]),
                    "x_range": x_range,
                    "y_range": y_range,
                }
            )
    report["boards"] = boards

    if "sweep" in selected:
        t0 = time.perf_counter()
        report["sweep"] = probe_sweep(
            env, robot, solver, seeds=seeds, radius_bins=radius_bins, boards=boards
        )
        print(f"  [6] sweep {time.perf_counter() - t0:.1f}s", flush=True)

    if "layout" in selected or "settle" in selected:
        t0 = time.perf_counter()
        if "layout" in selected:
            report["layout"] = probe_layout(env, samples=50 if args.quick else 200)
        if "settle" in selected:
            report["settle"] = probe_settle(env, seeds=5 if args.quick else 20)
        print(f"  [7/8] layout+settle {time.perf_counter() - t0:.1f}s", flush=True)

    report["elapsed_s"] = time.perf_counter() - started
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_plain(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[1] 可达带（行 = r；exact = 完整竖直抓取，position_only = 只保住位置）")
    print("   r     exact   pos-only  fail")
    for row in report.get("reach", {}).get("radii", []):
        print(
            f"  {row['radius']:.3f}   {row['exact_rate'] * 100:>5.0f}%"
            f"   {row['position_only_rate'] * 100:>7.0f}%   {row['fail_rate'] * 100:>4.0f}%"
        )

    print("\n[1b] 真正能抓的半径（含齿面中点补偿，整个专家轨迹都要解出来）")
    for row in report.get("grasp_reach", {}).get("rows", []):
        print(f"    r={row['radius']:.3f}   {row['success_rate'] * 100:>5.0f}%")

    print("\n[2] 夹爪外廓（相对齿面中点，世界水平面）")
    for name, row in report.get("hand", {}).get("outline", {}).items():
        print(
            f"    {name:>16}  闭合方向 ±{row['outer_half_span_closing'] * 1000:>5.1f}"
            f"  全宽 {row['span_closing'] * 1000:>6.1f} mm"
            f"   铰链方向 ±{row['outer_half_span_hinge'] * 1000:>5.1f}"
            f"  全宽 {row['span_hinge'] * 1000:>6.1f} mm"
        )

    print("\n[3] 竖直占用剖面 H(A)（抓取位；A 以外的最低高度，相对台面）")
    _print_occupancy(report.get("hand", {}).get("wall_height_limit_by_inner_half_width") or [])
    wrist = report.get("hand", {}).get("grasp_pose", {}).get("wrist_lowest_z")
    if wrist is not None:
        print(f"    → 半径 {WRIST_RADIUS * 1000:.0f}mm 以外的腕部最低 z = {wrist * 1000:.1f} mm")

    print("\n[4] 倾斜接近（O3 低梁用）")
    for row in report.get("tilt", {}).get("rows", []):
        limits = "  ".join(
            f"H({int(radius * 1000)})="
            + (
                f"{row[f'wall_height_limit_at_{int(radius * 1000)}mm'] * 1000:.0f}mm"
                if row[f"wall_height_limit_at_{int(radius * 1000)}mm"] is not None
                else "n/a"
            )
            for radius in report.get("tilt", {}).get("probe_radii", [])
        )
        if row["achieved_tilt_deg"] is None:
            print(
                f"    tilt {row['tilt_deg']:>6.0f}°   不可解 "
                f"{row['total'] - row['feasible']}/{row['total']}"
            )
            continue
        near = row["near_wall_height_at_50mm"]
        far = row["far_wall_height_at_50mm"]
        fmt = lambda value: f"{value * 1000:>6.1f}" if value is not None else "   n/a"
        print(
            f"    tilt {row['tilt_deg']:>6.0f}°   可解 {row['feasible']}/{row['total']}"
            f"   实际达成 {row['achieved_tilt_deg']:>5.1f}°"
            f"   位置误差 ≤{row['position_error_max_mm']:>5.1f} mm"
            f"   整环H {limits}"
            f"   近侧(朝底座) {fmt(near)} / 远侧 {fmt(far)} mm @50mm"
        )

    print("\n[5] 俯视相机下的遮挡板（高度相对台面；'撞臂' 来自下一步的扫掠检查）")
    for row in report.get("occluder", {}).get("rows", []):
        if not row.get("feasible"):
            print(
                f"    seed {row['seed']} {row['side']:>6} d={row['distance'] * 1000:>3.0f}mm"
                f"  不可行：{row.get('reason')}"
            )
            continue
        print(
            f"    seed {row['seed']} {row['side']:>6} d={row['distance'] * 1000:>3.0f}mm"
            f"   挡 25% {row['board_height_for_25pct'] * 1000:>6.1f}"
            f"   挡一半 {row['board_height_for_50pct'] * 1000:>6.1f}"
            f"   挡 65% {row['board_height_for_65pct'] * 1000:>6.1f} mm"
            f"   板需覆盖 [{row['board_span_needed'][0] * 1000:.0f},"
            f" {row['board_span_needed'][1] * 1000:.0f}] mm"
        )

    if "sweep" in report:
        print("\n[6] 专家轨迹扫掠包络")
        sweep = report["sweep"]
        aabb = sweep["aabb_over_seeds"]
        if aabb["min"] is not None:
            print(
                f"    AABB  x [{aabb['min'][0] * 1000:.0f}, {aabb['max'][0] * 1000:.0f}]"
                f"   y [{aabb['min'][1] * 1000:.0f}, {aabb['max'][1] * 1000:.0f}]"
                f"   z [{aabb['min'][2] * 1000:.0f}, {aabb['max'][2] * 1000:.0f}] mm"
            )
        print("    全轨迹占用剖面 H(A)（整环 / 近侧朝底座 / 远侧背离）：")
        print("      A(mm)   整环    近侧    远侧")
        near_rows = sweep["near_wall_height_limit_by_inner_half_width"]
        far_rows = sweep["far_wall_height_limit_by_inner_half_width"]
        for index, row in enumerate(sweep["wall_height_limit_by_inner_half_width"]):
            if row["radius"] > 0.1001 and row["radius"] < 0.1199:
                continue
            near = near_rows[index]["wall_height_limit"] * 1000 if near_rows else float("nan")
            far = far_rows[index]["wall_height_limit"] * 1000 if far_rows else float("nan")
            print(
                f"      {row['radius'] * 1000:>5.0f}  {row['wall_height_limit'] * 1000:>7.1f}"
                f"  {near:>7.1f}  {far:>7.1f}"
            )
        print("    候选遮挡板 vs 机械臂扫掠（margin = 板顶 − 该处机械臂最低高度）：")
        for board in sweep["board_results"]:
            print(
                f"      {board['key']:<28} 板高 {board['height'] * 1000:>6.1f} mm"
                f"   该处机械臂最低 {board['robot_lowest_z_there'] * 1000:>6.1f} mm"
                f"   margin {board['margin'] * 1000:>+7.1f} mm"
                f"   {'干涉' if board['interference'] else '通过'}"
            )
        feasible = sweep["feasible_boards"]
        print(
            f"    → 可行的 (距离, 高度) 组合：{len(feasible)}/{len(sweep['board_results'])}"
            + (f"  {feasible}" if feasible else "  —— 俯视相机下的竖直遮挡板在本场景不可行")
        )

    if "layout" in report:
        print("\n[7] 布局不变量")
        row = report["layout"]
        print(
            f"    {row['samples']} 组：兜底率 {row['fallback_rate'] * 100:.1f}%"
            f"   最小两两间距 {row['min_pairwise_clearance'] * 1000:.1f} mm"
            f"   底座最小让开 {row['min_base_keepout_clearance'] * 1000:.1f} mm"
            f"   越界 {row['bounds_violations']}"
        )

    if "settle" in report:
        print("\n[8] 静置稳定性")
        row = report["settle"]
        print(
            f"    {row['seeds']} seed × {row['steps']} 步（{row['seconds']:.1f}s）："
            f"最大漂移 {row['max_drift_m'] * 1000:.2f} mm，"
            f"平均 {row['mean_drift_m'] * 1000:.2f} mm，"
            f"最大倾角 {row['max_tilt_deg']:.2f}°"
        )

    print(f"\n报告写入 {out}（总计 {report['elapsed_s']:.1f}s）")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())