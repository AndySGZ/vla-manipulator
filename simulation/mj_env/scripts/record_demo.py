"""Record complete expert demonstrations as an animated GIF.

Each segment is one whole episode -- transit, reach, descend, close, lift -- and
reports the phase and the simulated clock as it goes.  The left panel is a free
3/4 camera: the policy's ``global_camera`` is a true top-down view, which is
right for the network but hides the object underneath the arm.  The right panel
is the wrist camera, so the clip shows both the motion and the grip.

    python -m mj_env.scripts.record_demo
    python -m mj_env.scripts.record_demo --objects rod_red sphere_blue \\
        --out ../../docs/expert_demo.gif

Requires no video encoder: the frames are written as a GIF, which every browser
and slide deck plays.  Convert to MP4 with ffmpeg if you need smaller files::

    ffmpeg -i expert_demo.gif -pix_fmt yuv420p -movflags faststart expert_demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mj_env import objects
from mj_env.env import SO101TabletopEnv
from mj_env.scripts.scripted_pick import GraspSolver, PickTuning, run_episode

#: Control steps per phase, in the order ``run_episode`` glides them.
_PHASES = (("transit", 30), ("reach", 20), ("descend", 25),
           ("close", 20), ("lift", 25), ("hold", 10))

#: Free camera: 3/4 view of the whole workspace, looking down the table.
CAMERA_LOOKAT = (0.22, 0.0, 0.05)
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -34.0
CAMERA_DISTANCE = 0.72


def _font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _phase(step: int) -> str:
    total = 0
    for name, count in _PHASES:
        total += count
        if step < total:
            return name
    return "hold"


def record(
    out: Path, targets: tuple[str, ...], *, seed: int, panel: int, fps: int
) -> None:
    env = SO101TabletopEnv(max_episode_steps=10_000, render_mode=None)
    solver = GraspSolver(env.model)
    # One renderer for both panels.  ``env.render_camera`` would build a second
    # one and the two fight over the GL context: the second frame comes back
    # black as soon as the first renderer is closed.
    renderer = mujoco.Renderer(env.model, panel, panel)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = list(CAMERA_LOOKAT)
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION
    camera.distance = CAMERA_DISTANCE

    title_font, body_font, small = _font(30), _font(21), _font(20)
    width, height = panel * 2, panel + 60
    frames: list[Image.Image] = []

    for index, target in enumerate(targets, start=1):
        prompt = objects.BY_NAME[target].prompt
        card = Image.new("RGB", (width, height), (16, 18, 22))
        draw = ImageDraw.Draw(card)
        draw.text((40, panel // 2 - 90), f"demonstration {index}/{len(targets)}",
                  font=small, fill=(140, 195, 255))
        draw.text((40, panel // 2 - 50), prompt, font=title_font,
                  fill=(240, 240, 240))
        draw.text((40, panel // 2 + 10), "IK expert, 20 Hz, 5-DoF arm + jaw",
                  font=small, fill=(170, 170, 170))
        # A filling bar keeps each held frame unique: identical frames get
        # merged by the GIF encoder and the card would flash for one frame.
        hold = int(fps * 0.9)
        for tick in range(hold):
            frame = card.copy()
            bar = ImageDraw.Draw(frame)
            progress = int((width - 80) * (tick + 1) / hold)
            bar.rectangle([40, height - 22, 40 + progress, height - 16],
                          fill=(90, 140, 200))
            frames.append(frame)

        step = [0]

        def on_frame(action, observation, info, step=step, prompt=prompt):
            renderer.update_scene(env.data, camera=camera)
            left = Image.fromarray(np.asarray(renderer.render()).copy())
            renderer.update_scene(env.data, camera="wrist_camera")
            right = Image.fromarray(np.asarray(renderer.render()).copy())
            canvas = Image.new("RGB", (width, height), (16, 18, 22))
            canvas.paste(left, (0, 60))
            canvas.paste(right, (panel, 60))
            draw = ImageDraw.Draw(canvas)
            phase = _phase(step[0])
            step[0] += 1
            draw.text((12, 6), prompt, font=body_font, fill=(240, 240, 240))
            draw.text((12, 33), f"phase: {phase}    t={env.data.time:5.2f}s",
                      font=small, fill=(140, 195, 255))
            draw.text((panel + 12, 33), "wrist camera", font=small,
                      fill=(140, 195, 255))
            frames.append(canvas)

        result = run_episode(env, solver, seed=seed, target=target,
                             tuning=PickTuning(), on_frame=on_frame)
        print(f"{target}: {result['outcome']}")

        banner = frames[-1].copy()
        draw = ImageDraw.Draw(banner)
        colour = (120, 235, 140) if result["success"] else (240, 120, 120)
        draw.rectangle([0, height - 26, width, height], fill=colour)
        draw.text((12, height - 24), f"outcome: {result['outcome']}",
                  font=small, fill=(20, 20, 20))
        frames.extend([banner] * int(fps * 0.7))

    env.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0, optimize=True,
    )
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", default=["cube_red",
                                                         "dumbbell_purple"],
                        choices=objects.NAMES, help="one segment per object")
    parser.add_argument("--out", type=Path,
                        default=Path("expert_demo.gif"),
                        help="output GIF path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--panel", type=int, default=440,
                        help="pixel size of each camera panel")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    record(args.out, tuple(args.objects), seed=args.seed,
           panel=args.panel, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
