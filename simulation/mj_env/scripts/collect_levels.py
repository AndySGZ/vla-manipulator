"""Collect demonstrations for the TaskSpec levels, in clean/underwater pairs.

Two things separate this from ``collect_demonstrations.py``, which drives the
legacy 8-object pick task:

* it works for **every registered level**, taking the instruction from the
  level's own pool and the expert from ``expert_for`` -- so a new level needs no
  edit here, which is the point of the registry;
* it collects **pairs**.  For each layout seed it runs the same seed on both
  imaging rungs, so the two episodes share a layout and differ in exactly one
  thing.  That is what makes "the clean/underwater gap" a measurement instead of
  a confound: an unpaired comparison also differs in object placement.

The instruction is sampled here rather than by the environment, because the
recorder has to be constructed with the task text *before* the episode runs.
Letting the environment sample it and reading ``env.task`` afterwards is how the
scene self-test caught a whole batch of episodes being labelled with the default
prompt.  Sampling from the declared pool also keeps hold-out sentences out of the
training set by construction.

    PYTHONPATH=<WYSD> uv run python -m mj_env.scripts.collect_levels \
        --all --episodes 1 --root .tmp/staging/so101_levels_smoke
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mj_env.adapters import LeRobotSO101Adapter
from mj_env.demonstrations import StagingDataset, SynchronizedEpisodeRecorder
from mj_env.scripts.scripted_pick import GraspSolver
from mj_env.scripts.scripted_pick_place import expert_for
from mj_env.specs import LEVEL_ORDER, get
from mj_env.task_env import TaskEnv
from mj_env.vfx import LADDER

#: The two sides of a pair.  v2_medium is the calibrated middle rung (1.20 m
#: visibility, 0.488 relative object signal on the top-down camera); using the
#: heaviest rung for a smoke run would stack two difficulties before we know the
#: easy side works.
DEFAULT_RUNGS = ("v0_clean", "v2_medium")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", action="append", choices=LEVEL_ORDER)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--episodes", type=int, default=1, help="layout seeds to attempt per level")
    parser.add_argument(
        "--successes",
        type=int,
        help="stop each level once this many successful episodes are saved; defaults to --episodes",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="stop each level after this many attempts; defaults to 3x --successes",
    )
    parser.add_argument("--vfx", action="append", choices=LADDER, help="imaging rung (repeatable)")
    parser.add_argument(
        "--instruction-pool",
        choices=("train", "holdout"),
        default="train",
        help="holdout sentences are for evaluation only and never enter the dataset",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    successes = args.successes if args.successes is not None else args.episodes
    if successes <= 0:
        parser.error("--successes must be positive")
    max_attempts = args.max_attempts if args.max_attempts is not None else 3 * successes
    if max_attempts < successes:
        parser.error("--max-attempts must be at least --successes")
    if args.root.exists():
        parser.error(
            f"dataset root already exists: {args.root}. Choose a new directory "
            "so an existing dataset is never overwritten."
        )
    tasks = LEVEL_ORDER if args.all else tuple(args.task)
    rungs = tuple(args.vfx) if args.vfx else DEFAULT_RUNGS

    results_path = args.root / "collection_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[tuple[str, str], set[int]] = defaultdict(set)
    attempted: dict[tuple[str, str], int] = defaultdict(int)

    for task_id in tasks:
        spec = get(task_id)
        pool = spec.instruction_pool(args.instruction_pool)
        if not pool:
            raise SystemExit(f"{task_id}: empty {args.instruction_pool} instruction pool")
        # One environment per rung, both alive: the *seed sequence* has to be
        # identical on both sides, so the loop order is seed-outer, rung-inner.
        # A level that fails a seed on one side simply has one fewer pair, which
        # the summary reports as the intersection of the two saved seed sets
        # rather than as two independent success counts.
        envs: dict[str, TaskEnv] = {}
        try:
            for rung in rungs:
                envs[rung] = TaskEnv(
                    task_id,
                    render_mode=None if args.headless else "human",
                    image_size=(args.height, args.width),
                    max_episode_steps=10_000,
                    vfx=rung,
                )
            fps = round(envs[rungs[0]].control_hz)
            if abs(envs[rungs[0]].control_hz - fps) > 1e-9:
                raise ValueError(f"LeRobot requires integer fps, got {envs[rungs[0]].control_hz}")
            datasets = {
                rung: StagingDataset(root=args.root / task_id / rung, fps=fps)
                for rung in rungs
            }
            adapters = {rung: LeRobotSO101Adapter(envs[rung]) for rung in rungs}
            experts = {rung: expert_for(envs[rung]) for rung in rungs}
            solvers = {rung: GraspSolver(envs[rung].model) for rung in rungs}

            for offset in range(max_attempts):
                if all(len(saved[(task_id, rung)]) >= successes for rung in rungs):
                    break
                seed = args.seed + offset
                # Deterministic per (seed, level): the same layout seed picks the
                # same sentence on both rungs, so a pair differs only in its
                # imaging.
                instruction = str(np.random.default_rng(seed).choice(pool))
                for rung in rungs:
                    if len(saved[(task_id, rung)]) >= successes:
                        continue
                    env = envs[rung]
                    recorder = SynchronizedEpisodeRecorder(
                        dataset=datasets[rung],
                        adapter=adapters[rung],
                        fps=fps,
                        task=instruction,
                    )
                    try:
                        result = experts[rung](
                            env,
                            solvers[rung],
                            seed=seed,
                            instruction=instruction,
                            on_frame=recorder.on_frame,
                        )
                    except Exception as error:  # noqa: BLE001 - recorded, not raised
                        result = {"outcome": f"error:{type(error).__name__}", "success": False}
                    attempted[(task_id, rung)] += 1
                    record = {
                        **result,
                        "task_id": task_id,
                        "family": spec.family,
                        "level": spec.level,
                        "layout_seed": seed,
                        "vfx_rung": rung,
                        "vfx_hash": env.vfx_hash,
                        "instruction": instruction,
                        "instruction_kind": args.instruction_pool,
                        "spec_hash": spec.hash(),
                        "frames": recorder.frame_index,
                        "fps": fps,
                        "saved": bool(result.get("success")),
                    }
                    if record["saved"]:
                        datasets[rung].save_episode(metadata=record)
                        saved[(task_id, rung)].add(seed)
                    else:
                        datasets[rung].clear_episode_buffer()
                    with results_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(
                        f"{task_id:<20} {rung:<10} seed={seed} "
                        f"outcome={record['outcome']:<12} frames={recorder.frame_index:>4} "
                        f"saved={record['saved']}",
                        flush=True,
                    )
        finally:
            for env in envs.values():
                env.close()

    print()
    header = " ".join(f"{rung:>16}" for rung in rungs)
    print(f"  {'level':<20} {header}   paired")
    paired_total = 0
    for task_id in tasks:
        cells = " ".join(
            f"{len(saved[(task_id, rung)]):>6}/{attempted[(task_id, rung)]:<9}" for rung in rungs
        )
        paired = set.intersection(*(saved[(task_id, rung)] for rung in rungs)) if rungs else set()
        paired_total += len(paired)
        print(f"  {task_id:<20} {cells} {len(paired):>7}")
    total_saved = sum(len(seeds) for seeds in saved.values())
    total_attempted = sum(attempted.values())
    print(f"saved {total_saved}/{total_attempted} episodes to {args.root}")
    if len(rungs) >= 2:
        print(f"complete pairs (same layout seed on every rung): {paired_total}")
        short = [
            task_id
            for task_id in tasks
            if len(set.intersection(*(saved[(task_id, rung)] for rung in rungs))) < successes
        ]
        if short:
            print(f"levels short of {successes} pairs: " + ", ".join(short))
    return 0 if total_saved else 1


if __name__ == "__main__":
    raise SystemExit(main())