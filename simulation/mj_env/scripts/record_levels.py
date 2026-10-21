"""Record one GIF per registered level, using that level's own expert.

Each segment is a complete episode of the level -- the expert picks, carries
and places -- rendered from a free 3/4 camera next to the wrist camera.  A
level that fails its first layout is retried on the next seed, because the
clip is meant to show the task being solved, not the harness failing.

    python -m mj_env.scripts.record_levels --out ../../levels
    python -m mj_env.scripts.record_levels --tasks p2_language_ref p4_stack
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mj_env.scripts.scripted_pick import GraspSolver
from mj_env.scripts.scripted_pick_place import expert_for
from mj_env.specs import LEVEL_ORDER, get
from mj_env.task_env import TaskEnv

#: Free camera: 3/4 view of the workspace.  The policy cameras are top-down and
#: hide the object under the arm, which is unreadable in a clip.
CAMERA_LOOKAT = (0.22, 0.0, 0.05)
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -34.0
CAMERA_DISTANCE = 0.72

#: Layouts tried before giving up.  The experts solve roughly two layouts in
#: three, so a level is retried until one succeeds; a level that cannot be
#: solved in this many layouts produces no file at all rather than a clip of
#: the harness failing.
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)


def _font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def record_level(
    out: Path, task_id: str, *, panel: int, fps: int
) -> dict:
    spec = get(task_id)
    pool = spec.instruction_pool("train")
    instruction = pool[0] if pool else spec.title

    env = TaskEnv(task_id)
    solver = GraspSolver(env.model)
    expert = expert_for(env)
    renderer = mujoco.Renderer(env.model, panel, panel)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = list(CAMERA_LOOKAT)
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION
    camera.distance = CAMERA_DISTANCE

    title_font, body_font, small = _font(24), _font(19), _font(17)
    width, height = panel * 2, panel + 54
    frames: list[Image.Image] = []
    result: dict = {"outcome": "not_run", "success": False}

    for seed in SEEDS:
        frames.clear()
        step = [0]

        def on_frame(action, observation, info, step=step):
            renderer.update_scene(env.data, camera=camera)
            left = Image.fromarray(np.asarray(renderer.render()).copy())
            renderer.update_scene(env.data, camera="wrist_camera")
            right = Image.fromarray(np.asarray(renderer.render()).copy())
            canvas = Image.new("RGB", (width, height), (16, 18, 22))
            canvas.paste(left, (0, 54))
            canvas.paste(right, (panel, 54))
            draw = ImageDraw.Draw(canvas)
            step[0] += 1
            draw.text((12, 5), instruction, font=body_font, fill=(240, 240, 240))
            draw.text((12, 30), f"{task_id}   seed {seed}   t={env.data.time:5.2f}s",
                      font=small, fill=(140, 195, 255))
            draw.text((panel + 12, 30), "wrist camera", font=small,
                      fill=(140, 195, 255))
            frames.append(canvas)

        result = expert(env, solver, seed=seed, instruction=instruction,
                        on_frame=on_frame)
        if result["success"]:
            break

    env.close()
    if not result["success"] or not frames:
        # Never ship a clip of the expert failing: the point of the file is
        # that this level can be solved, and a red banner proves it cannot.
        return result

    banner = frames[-1].copy()
    draw = ImageDraw.Draw(banner)
    colour = (120, 235, 140) if result["success"] else (240, 120, 120)
    draw.rectangle([0, height - 24, width, height], fill=colour)
    draw.text((12, height - 22), f"outcome: {result['outcome']}",
              font=small, fill=(20, 20, 20))
    frames.extend([banner] * int(fps * 0.8))

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(LEVEL_ORDER),
                        choices=LEVEL_ORDER)
    parser.add_argument("--out", type=Path, default=Path("levels"))
    parser.add_argument("--panel", type=int, default=360)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    failures = 0
    for task_id in args.tasks:
        path = args.out / f"{task_id}.gif"
        if path.exists():
            path.unlink()  # never leave a stale clip next to a failed retry
        result = record_level(path, task_id, panel=args.panel, fps=args.fps)
        size = path.stat().st_size / 1e6 if path.exists() else 0.0
        print(f"{task_id:22s} {result['outcome']:10s} {size:5.1f} MB", flush=True)
        failures += 0 if result["success"] else 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
