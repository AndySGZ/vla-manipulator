"""Scene self-test: the gate a level has to pass before any data is collected.

A level whose layout is unreachable, whose obstacle intersects the arm, or whose
criterion the *starting layout* already satisfies would produce a dataset of
failures, and a policy trained on that learns to give up.  Collecting first and
discovering it later costs a server run; this costs seconds.

The nine check classes, following ``WYSD/underwater_sampling/selftest.py``:

1. ``check_load``          -- the scene compiles and declares what the spec says
2. ``check_randomization`` -- layout invariants hold over many seeds, and every
   obstacle keeps a *constant* offset from the target (otherwise the difficulty
   drifts with the randomisation and the ladder is not comparable)
3. ``check_settle``        -- props actually come to rest, with no penetration
4. ``check_reach``         -- the scripted expert can solve the level's layout
5. ``check_clearance``     -- and can execute it without touching a static prop
6. ``check_no_free_success`` -- reset never satisfies the criterion
7. ``check_schema``        -- a real episode records 20 Hz, 6-D, 224^2, one task
8. ``check_vfx``           -- both imaging sides render; params round-trip
9. ``check_pairing``       -- same seed -> identical layout on both sides

Run it after touching any scene, spec or criterion::

    uv run python -m mj_env.scripts.selftest_scenes
    uv run python -m mj_env.scripts.selftest_scenes --task o4_narrow_slot --quick

Exit code 0 means every level passed; 1 means at least one failed, and the report
says which.  No rendering is needed for checks 1-6 beyond the schema check.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from mj_env.adapters import LeRobotSO101Adapter
from mj_env.demonstrations import StagingDataset, SynchronizedEpisodeRecorder
from mj_env.specs import LEVEL_ORDER, get
from mj_env.task_env import TaskEnv
from mj_env.vfx import CLEAN_RUNG, load_rungs

DEFAULT_REPORT = Path(".tmp/selftest/selftest_report.json")
STAGING_ROOT = Path(".tmp/selftest/staging")

#: A contact may be a fraction of a millimetre deep even with no interference,
#: because the solver allows a small soft-contact overlap.  Anything beyond this
#: is the arm actually in the prop.
PENETRATION_TOLERANCE_MM = 0.5
#: A settled prop may creep this far over the settle window.
SETTLE_DRIFT_TOLERANCE_MM = 0.5
#: Two props whose vertical extents are within this of merely touching count as
#: stacked rather than overlapping, which is what lets a cover cube sit on a
#: target without the overlap check firing.
STACK_TOUCH_TOLERANCE_M = 0.0005
#: An obstacle's offset from the target, measured in the target's frame, may
#: wobble by this much between seeds.  It is exact in the layout; the noise comes
#: from measuring it against the *settled* object.
STATIC_OFFSET_TOLERANCE_M = 1e-5
#: Table bounds the layout must respect, from mj_env/env.py.
TABLE_BOUNDS_X = (-0.02, 0.82)
TABLE_BOUNDS_Y = (-0.39, 0.39)


@dataclass
class Report:
    """Accumulates checks so the summary can be printed and written as JSON."""

    verbose: bool = True
    results: list[dict] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append({"check": name, "ok": bool(ok), "detail": detail})
        if self.verbose:
            print(f"      {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
        return bool(ok)

    def measure(self, text: str) -> None:
        self.measurements.append(text)
        if self.verbose:
            print(f"      ----  {text}")

    @property
    def failed(self) -> list[dict]:
        return [row for row in self.results if not row["ok"]]

    def to_json(self) -> dict:
        return {
            "checks": len(self.results),
            "passed": len(self.results) - len(self.failed),
            "failed": len(self.failed),
            "results": self.results,
            "measurements": self.measurements,
        }


# ---------------------------------------------------------------------------
# 1. load
# ---------------------------------------------------------------------------
def check_load(env: TaskEnv, report: Report) -> None:
    spec = env.spec
    report.check(
        "load/action_space_is_six",
        env.action_space.shape == (6,),
        f"shape={env.action_space.shape}",
    )
    keys = set(env.observation_space.spaces)
    report.check(
        "load/observation_has_two_cameras_and_state",
        {"observation.state", "observation.image", "observation.wrist_image"} <= keys,
        f"keys={sorted(keys)}",
    )
    missing = [name for name in spec.prop_bodies if name not in env._objects]
    report.check("load/free_props_present", not missing, f"missing={missing}")
    missing = [name for name in spec.statics if name not in env._static_ids]
    report.check("load/statics_present", not missing, f"missing={missing}")
    report.check(
        "load/goal_body_present",
        spec.goal_body is None or spec.goal_body in env._static_ids,
        f"goal={spec.goal_body}",
    )
    report.check(
        "load/scene_file_matches_the_spec",
        (Path("mj_env/assets/scenes") / spec.scene).is_file(),
        spec.scene,
    )
    report.measure(
        f"scene {spec.scene}: bodies={env.model.nbody} geoms={env.model.ngeom}"
        f" actuators={env.model.nu}"
    )


# ---------------------------------------------------------------------------
# 2. randomization
# ---------------------------------------------------------------------------
def check_randomization(env: TaskEnv, report: Report, seeds: list[int]) -> None:
    spec = env.spec
    offsets: dict[str, list[np.ndarray]] = {name: [] for name in spec.statics}
    inside = True
    overlapping = 0
    for seed in seeds:
        env.reset(seed=seed)
        for name in spec.prop_bodies:
            position = env.prop_pos(name)
            if not (
                TABLE_BOUNDS_X[0] <= position[0] <= TABLE_BOUNDS_X[1]
                and TABLE_BOUNDS_Y[0] <= position[1] <= TABLE_BOUNDS_Y[1]
            ):
                inside = False
        # Pairwise separation between free props.  **Three-dimensional**: a
        # cover cube resting on the target shares its footprint on purpose, and
        # a flat 2D check would call a legitimate stack an overlap.  Two props
        # only collide if their vertical extents also meet.
        footprints = {name: env._objects[name].footprint for name in spec.prop_bodies}
        for index, first in enumerate(spec.prop_bodies):
            for second in spec.prop_bodies[index + 1 :]:
                vertical = abs(
                    float(env.prop_pos(first)[2]) - float(env.prop_pos(second)[2])
                )
                # Touching counts as separated: two stacked cubes have centres
                # exactly (half_a + half_b) apart, and settling eats a few
                # micrometres of that, so an exact comparison would call every
                # stack an overlap.
                if vertical >= (
                    footprints[first].rest_clearance
                    + footprints[second].rest_clearance
                    - STACK_TOUCH_TOLERANCE_M
                ):
                    continue
                gap = float(
                    np.linalg.norm(env.prop_pos(first)[:2] - env.prop_pos(second)[:2])
                ) - footprints[first].max_radius - footprints[second].max_radius
                overlapping += int(gap < 0.0)
        # Obstacles must keep a constant offset in the target's own frame, or
        # "the wall is 53 mm from the object" stops being a property of the level.
        body = env._objects[spec.target].body_id
        rotation = env.data.xmat[body].reshape(3, 3)
        for name in spec.statics:
            if name == spec.goal_body or name in spec.independent_statics:
                # `independent_statics` are placed on purpose relative to the
                # *level*, not to the target: O5's parking spot has to dodge the
                # goal, which is itself random, so holding it rigid to the
                # target is impossible -- and would be wrong anyway, since the
                # whole point of a parking spot is that it is somewhere else.
                continue
            world = np.asarray(env.model.body_pos[env._static_ids[name]], dtype=float)
            offsets[name].append(rotation.T @ (world - env.prop_pos(spec.target)))

    report.check("randomization/props_inside_table", inside)
    report.check("randomization/props_do_not_overlap", overlapping == 0, f"overlaps={overlapping}")
    for name, rows in offsets.items():
        if len(rows) < 2:
            continue
        # The offset is exact in the layout, but it is measured from the
        # *settled* object, so allow sub-0.01 mm noise.  Anything larger means
        # the obstacle really does move relative to the target, and the level's
        # difficulty drifts with the randomisation.
        spread = float(np.max(np.std(np.vstack(rows), axis=0)))
        report.check(
            f"randomization/{name}_moves_with_target",
            spread < STATIC_OFFSET_TOLERANCE_M,
            f"offset spread={spread * 1000:.5f} mm",
        )
    if offsets:
        first = next((rows for rows in offsets.values() if rows), None)
        if first:
            report.measure(f"static offset in target frame: {np.round(first[0] * 1000, 1)} mm")


# ---------------------------------------------------------------------------
# 3. settle
# ---------------------------------------------------------------------------
def check_settle(env: TaskEnv, report: Report, seeds: list[int]) -> None:
    spec = env.spec
    worst_drift = 0.0
    worst_penetration = 0.0
    for seed in seeds:
        env.reset(seed=seed)
        before = {name: env.prop_pos(name) for name in spec.prop_bodies}
        for _ in range(250):
            mujoco.mj_step(env.model, env.data)
        for name in spec.prop_bodies:
            worst_drift = max(worst_drift, float(np.linalg.norm(env.prop_pos(name) - before[name])))
        static = set(env._static_ids.values())
        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            first = int(env.model.geom_bodyid[contact.geom1])
            second = int(env.model.geom_bodyid[contact.geom2])
            if (first in static) != (second in static):
                worst_penetration = min(worst_penetration, float(contact.dist))
    report.measure(
        f"settle: worst drift {worst_drift * 1000:.3f} mm, worst static penetration"
        f" {worst_penetration * 1000:+.3f} mm"
    )
    report.check(
        "settle/props_come_to_rest",
        worst_drift * 1000.0 <= SETTLE_DRIFT_TOLERANCE_MM,
        f"{worst_drift * 1000:.3f} mm",
    )
    report.check(
        "settle/no_penetration_at_reset",
        worst_penetration * 1000.0 >= -PENETRATION_TOLERANCE_MM,
        f"{worst_penetration * 1000:+.3f} mm",
    )


# ---------------------------------------------------------------------------
# 6. no free success
# ---------------------------------------------------------------------------
def check_no_free_success(env: TaskEnv, report: Report, seeds: list[int]) -> None:
    free = 0
    for seed in seeds:
        env.reset(seed=seed)
        free += int(env.criterion_ok())
    report.check(
        "no_free_success/reset_never_satisfies_the_criterion",
        free == 0,
        f"satisfied on {free}/{len(seeds)} resets",
    )


# ---------------------------------------------------------------------------
# 4/5. reach and clearance, via the scripted expert
# ---------------------------------------------------------------------------
def check_expert(env: TaskEnv, report: Report, seeds: list[int]) -> None:
    from mj_env.scripts.scripted_pick import GraspSolver
    from mj_env.scripts.scripted_pick_place import expert_for

    solver = GraspSolver(env.model)
    expert = expert_for(env)
    solved = 0
    succeeded = 0
    worst_penetration = 0.0
    outcomes: list[str] = []
    for seed in seeds:
        result = expert(env, solver, seed=seed)
        outcomes.append(result["outcome"])
        solved += int(result["outcome"] not in ("ik_fail", "transport_fail", "retreat_fail"))
        succeeded += int(result["success"])
        worst_penetration = min(worst_penetration, float(result.get("static_penetration_mm", 0.0)))

    report.check(
        "reach/expert_solves_every_layout",
        solved == len(seeds),
        f"{solved}/{len(seeds)} solved",
    )
    report.check(
        "reach/expert_succeeds_on_every_layout",
        succeeded == len(seeds),
        f"{succeeded}/{len(seeds)} succeeded; outcomes={sorted(set(outcomes))}",
    )
    report.check(
        "clearance/expert_never_penetrates_a_static_prop",
        worst_penetration >= -PENETRATION_TOLERANCE_MM,
        f"worst {worst_penetration:+.3f} mm",
    )


# ---------------------------------------------------------------------------
# 7. schema
# ---------------------------------------------------------------------------
def check_schema(env: TaskEnv, report: Report) -> None:
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    STAGING_ROOT.parent.mkdir(parents=True, exist_ok=True)
    # Reset *first*: the instruction is sampled at reset, so a recorder built
    # before it would label the episode with the pre-reset default prompt.  The
    # real collector builds it per episode for the same reason.
    observation, info = env.reset(seed=0)
    fps = round(env.control_hz)
    dataset = StagingDataset(root=STAGING_ROOT, fps=fps)
    adapter = LeRobotSO101Adapter(env)
    recorder = SynchronizedEpisodeRecorder(
        dataset=dataset, adapter=adapter, fps=fps, task=env.task
    )
    try:
        for _ in range(5):
            action = np.asarray(env.data.ctrl[env._actuator_ids], dtype=np.float32)
            recorder.on_frame(action.copy(), observation, info)
            observation, _, _, _, info = env.step(action)
        dataset.save_episode(metadata={"source": "selftest"})
        episode = np.load(STAGING_ROOT / "episodes" / "episode_000000.npz", allow_pickle=False)
        frames = int(episode["state"].shape[0])
        report.check("schema/recorded_five_frames", frames == 5, f"frames={frames}")
        report.check("schema/state_is_six_dimensional", episode["state"].shape[1:] == (6,))
        report.check("schema/actions_are_six_dimensional", episode["actions"].shape[1:] == (6,))
        report.check(
            "schema/images_are_uint8_224",
            episode["image"].dtype == np.uint8 and episode["image"].shape[1:] == (224, 224, 3),
            f"image={episode['image'].shape} {episode['image'].dtype}",
        )
        timestamps = episode["timestamp"]
        expected = np.arange(frames) / round(env.control_hz)
        report.check(
            "schema/timestamps_are_uniform_20hz",
            bool(np.allclose(timestamps, expected, rtol=0.0, atol=1e-6)),
        )
        report.check("schema/task_text_is_recorded", str(episode["task"].item()) == env.task)
    finally:
        shutil.rmtree(STAGING_ROOT, ignore_errors=True)


# ---------------------------------------------------------------------------
# 8/9. imaging and pairing
# ---------------------------------------------------------------------------
def check_vfx(report: Report, task_id: str) -> None:
    rungs = load_rungs()
    if not rungs:
        report.check("vfx/config_is_calibrated", False, "run mj_env.scripts.water_probe")
        return
    report.check("vfx/config_is_calibrated", True, f"rungs={sorted(rungs)}")
    water = next((name for name in rungs if not rungs[name].clean), None)
    if water is None:
        report.check("vfx/has_a_water_rung", False)
        return
    try:
        params = rungs[water].params
        report.check(
            "vfx/params_round_trip",
            type(params).from_dict(params.to_dict()).hash() == rungs[water].hash,
        )
    except Exception as exc:  # pragma: no cover - defensive
        report.check("vfx/params_round_trip", False, repr(exc))

    clean_env = TaskEnv(task_id, vfx=CLEAN_RUNG)
    water_env = TaskEnv(task_id, vfx=water, vfx_seed=0)
    try:
        clean_observation, clean_info = clean_env.reset(seed=0)
        water_observation, water_info = water_env.reset(seed=0)
        same_layout = clean_info["layout"] == water_info["layout"]
        identical_props = all(
            np.array_equal(clean_info["init_props"][name], water_info["init_props"][name])
            for name in clean_info["init_props"]
        )
        report.check(
            "pairing/same_seed_gives_the_same_layout",
            same_layout and identical_props,
            f"layout_equal={same_layout} props_equal={identical_props}",
        )
        report.check(
            "pairing/imaging_identity_differs",
            clean_info["vfx_hash"] != water_info["vfx_hash"],
            f"clean={clean_info['vfx_hash']!r} water={water_info['vfx_hash'][:12]!r}",
        )
        difference = float(
            np.abs(
                clean_observation["observation.image"].astype(int)
                - water_observation["observation.image"].astype(int)
            ).mean()
        )
        report.check(
            "pairing/pixels_differ",
            difference > 2.0,
            f"mean |clean-water| = {difference:.2f}",
        )
        report.check(
            "pairing/schema_is_identical_on_both_sides",
            clean_observation["observation.state"].shape
            == water_observation["observation.state"].shape
            == (6,),
        )
    finally:
        clean_env.close()
        water_env.close()


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def check_registry(report: Report) -> None:
    for task_id in LEVEL_ORDER:
        spec = get(task_id)
        report.check(
            f"registry/{task_id}_has_instructions",
            len(spec.instructions) >= 3 and len(spec.holdout) >= 1,
            f"{len(spec.instructions)} train + {len(spec.holdout)} hold-out",
        )
        report.check(
            f"registry/{task_id}_language_pools_are_disjoint",
            set(spec.instructions).isdisjoint(spec.holdout),
        )
        report.check(f"registry/{task_id}_has_a_hash", len(spec.hash()) == 12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="level to test (repeatable)")
    parser.add_argument("--quick", action="store_true", help="fewer seeds, skip the scene sweep")
    parser.add_argument("--seeds", type=int, default=0, help="layouts per level")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-expert", action="store_true", help="checks 1-3 and 6-9 only")
    args = parser.parse_args()

    tasks = args.task or LEVEL_ORDER
    seeds = args.seeds or (2 if args.quick else 4)
    layout_seeds = list(range(seeds if args.quick else 6))

    report = Report()
    payload: dict[str, dict] = {"levels": {}}
    print(f"selftest: {len(tasks)} level(s), {seeds} episode seeds, {len(layout_seeds)} layout seeds")

    check_registry(report)
    check_vfx(report, tasks[0])

    for task_id in tasks:
        spec = get(task_id)
        print(f"\n  [{task_id}] {spec.title}")
        env = TaskEnv(task_id, max_episode_steps=10_000)
        before = len(report.results)
        try:
            check_load(env, report)
            check_randomization(env, report, layout_seeds)
            check_settle(env, report, layout_seeds[:3])
            check_no_free_success(env, report, layout_seeds)
            if not args.skip_expert:
                check_expert(env, report, list(range(seeds)))
            check_schema(env, report)
        finally:
            env.close()
        payload["levels"][task_id] = report.to_json()["results"][before:]

    summary = report.to_json()
    payload["summary"] = {key: summary[key] for key in ("checks", "passed", "failed")}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\nselftest: {summary['passed']}/{summary['checks']} checks passed"
        + (f", {summary['failed']} FAILED" if summary["failed"] else "")
    )
    for row in report.failed:
        print(f"  FAIL {row['check']}  {row['detail']}")
    print(f"report: {args.out}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())