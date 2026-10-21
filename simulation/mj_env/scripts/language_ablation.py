"""Language ablation: same scene, same seed, only the instruction changes.

This is the experiment that answers "is the language branch being used at all".
A policy that ignores the text and always grabs the same cube can look fine on a
single-object level; on a two-cube level it can still score 50%, which is exactly
the number that hides the problem.  So the measurement is not a success rate --
it is **whether the object that ends up on the goal changes when the sentence
does**.

The scripted expert is *supposed* to be perfect here: it reads the target from
the instruction through privileged information, so 100% means the harness can
express the experiment, not that a policy has learned it.  Run the same table
against a trained checkpoint later and the same columns give the real answer.

    PYTHONPATH=<WYSD> uv run python -m mj_env.scripts.language_ablation \
        --task p2_language_ref --episodes 8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mj_env.scripts.scripted_pick import GraspSolver
from mj_env.scripts.scripted_pick_place import run_pick_place
from mj_env.specs import LEVEL_ORDER, get
from mj_env.task_env import TaskEnv


def sentences(spec) -> tuple[str, str]:
    """Two training sentences that name two different objects."""
    by_target: dict[str, str] = {}
    for text in spec.instructions:
        target = spec.target_for_instruction(text)
        by_target.setdefault(target, text)
    if len(by_target) < 2:
        raise SystemExit(
            f"{spec.task_id} does not reference two objects; nothing to ablate"
        )
    first, second = sorted(by_target)
    return by_target[first], by_target[second]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=LEVEL_ORDER, default="p2_language_ref")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(".tmp/language_ablation.json"))
    args = parser.parse_args()

    spec = get(args.task)
    sentence_a, sentence_b = sentences(spec)
    env = TaskEnv(args.task, max_episode_steps=10_000)
    solver = GraspSolver(env.model)

    print(f"{spec.task_id}: {sentence_a!r}  vs  {sentence_b!r}")
    print(f"  {'seed':>5} {'sentence':<46} {'target':<8} {'on_goal':<8} {'err(mm)':>8} {'ok':>4}")
    rows = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            for sentence in (sentence_a, sentence_b):
                result = run_pick_place(env, solver, seed=seed, instruction=sentence)
                target = result["target"]
                errors = {
                    name: float(np.linalg.norm(env.prop_pos(name)[:2] - env.goal_xy()))
                    for name in spec.prop_bodies
                }
                on_goal = min(errors, key=errors.get)
                rows.append(
                    {
                        "seed": seed,
                        "sentence": sentence,
                        "target": target,
                        "on_goal": on_goal,
                        "on_goal_error_mm": errors[on_goal] * 1000.0,
                        "success": result["success"],
                        "followed_instruction": on_goal == target,
                    }
                )
                print(
                    f"  {seed:>5} {sentence:<46} {target:<8} {on_goal:<8}"
                    f" {errors[on_goal] * 1000:>8.1f} {result['success']!s:>4}"
                )
    finally:
        env.close()

    followed = sum(row["followed_instruction"] for row in rows)
    swapped = 0
    for episode in range(args.episodes):
        seed = args.seed + episode
        pair = [row for row in rows if row["seed"] == seed]
        if len(pair) == 2 and pair[0]["on_goal"] != pair[1]["on_goal"]:
            swapped += 1

    print()
    print(f"  followed the instruction : {followed}/{len(rows)}")
    print(f"  the goal object changed  : {swapped}/{args.episodes} seeds")
    print(f"  outcomes                 : {dict(Counter(row['success'] for row in rows))}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "task_id": spec.task_id,
                "spec_hash": spec.hash(),
                "sentences": [sentence_a, sentence_b],
                "episodes": args.episodes,
                "rows": rows,
                "followed_instruction": followed,
                "goal_object_swapped": swapped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"report: {args.out}")
    # The harness is only usable if the sentence really selects the object.
    return 0 if followed == len(rows) and swapped == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())