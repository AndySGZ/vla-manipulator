"""Check that a staged dataset's clean/underwater episodes are real pairs.

A pair is only a comparison if the two episodes differ in **exactly one** thing.
The scene self-test already asserts that at the environment level; this checks it
on what was actually written to disk, which is the thing a training run consumes:

* the same layout seed, and therefore the same ``state`` and ``actions`` --
  the scripted expert is imaging-blind by design, so any difference in the
  action stream means the pair is confounded;
* different images, with the underwater side measurably degraded;
* the same task text, and a schema that is still 6-dim state / 6-dim actions /
  two 224x224 images;
* timestamps at a strict 20 Hz.

    PYTHONPATH=<WYSD> uv run python -m mj_env.scripts.verify_pairs \
        --root .tmp/staging/so101_levels_smoke
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mj_env.vfx import CLEAN_RUNG, LADDER

FPS = 20.0
FPS_TOLERANCE = 1e-6


def load_manifest(rung_root: Path) -> dict[int, dict]:
    path = rung_root / "manifest.jsonl"
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            rows[int(record["layout_seed"])] = record
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--water", default="v2_medium", choices=LADDER)
    args = parser.parse_args()

    clean_root = args.root / CLEAN_RUNG
    water_root = args.root / args.water
    if not clean_root.is_dir() or not water_root.is_dir():
        raise SystemExit(f"expected {clean_root} and {water_root} to exist")

    clean = load_manifest(clean_root)
    water = load_manifest(water_root)
    shared = sorted(set(clean) & set(water))
    print(f"clean episodes: {len(clean)}   water episodes: {len(water)}   paired: {len(shared)}")
    if not shared:
        print("no paired episodes: the two sides share no layout seed")
        return 1

    failures: list[str] = []
    schema_seen = Counter()
    clean_bytes = water_bytes = 0
    for seed in shared:
        left = np.load(clean_root / clean[seed]["path"])
        right = np.load(water_root / water[seed]["path"])
        for key, expected in (("state", 6), ("actions", 6)):
            if left[key].shape[1] != expected:
                failures.append(f"seed {seed}: {key} is {left[key].shape[1]}-dim, not {expected}")
        for key in ("image", "wrist_image"):
            for side, payload in (("clean", left), ("water", right)):
                frames = payload[key].shape
                schema_seen[f"{key}={frames[1]}x{frames[2]}"] += 1
                if frames[1] != 224 or frames[2] != 224:
                    failures.append(f"seed {seed}: {side} {key} is {frames[1]}x{frames[2]}")
        if not np.array_equal(left["actions"], right["actions"]):
            delta = float(np.abs(left["actions"] - right["actions"]).max())
            failures.append(
                f"seed {seed}: action streams differ by {delta:.6f}; the expert is "
                "imaging-blind, so this pair is confounded"
            )
        if not np.array_equal(left["state"], right["state"]):
            failures.append(f"seed {seed}: state streams differ")
        if str(left["task"]) != str(right["task"]):
            failures.append(
                f"seed {seed}: task text differs ({left['task']!r} vs {right['task']!r})"
            )
        for side, payload in (("clean", left), ("water", right)):
            steps = np.diff(payload["timestamp"])
            if steps.size and np.abs(steps - 1.0 / FPS).max() > FPS_TOLERANCE:
                failures.append(f"seed {seed}: {side} timestamps are not a strict {FPS} Hz")
        for key in ("image", "wrist_image"):
            if np.array_equal(left[key], right[key]):
                failures.append(
                    f"seed {seed}: {key} is identical on both sides -- the water rung "
                    "was not applied"
                )
        clean_bytes += (clean_root / clean[seed]["path"]).stat().st_size
        water_bytes += (water_root / water[seed]["path"]).stat().st_size
        difference = np.abs(
            left["image"].astype(np.int16) - right["image"].astype(np.int16)
        ).mean()
        print(f"  seed {seed}: frames={left['image'].shape[0]} mean |clean-water| = {difference:5.2f}")

    contrast = clean_bytes / max(water_bytes, 1)
    print()
    print(f"  schema: {dict(schema_seen)}")
    print(
        f"  staged size: clean {clean_bytes / 1e6:.1f} MB, water {water_bytes / 1e6:.1f} MB"
        f" (water is {water_bytes / max(clean_bytes, 1):.1f}x the clean size;"
        f" the imaging noise does not compress)"
    )
    if failures:
        print()
        for line in failures:
            print(f"  FAIL {line}")
        return 1
    print(f"\nall {len(shared)} pairs satisfy the invariants (clean/water size ratio {contrast:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())