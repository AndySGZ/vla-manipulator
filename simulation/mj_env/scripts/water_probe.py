"""M0.5: calibrate the water-imaging ladder for the SO101 tabletop scene.

Every level must exist in two imaging conditions -- clean and underwater -- with
the *same* seed and layout, so a pair differs in nothing but the imaging.  That
makes the imaging layer a prerequisite for collection, and it means the water
parameters have to be calibrated for *this* scene.

``WYSD/underwater_vision`` was tuned for the Uranus arm, whose camera sits
1.2-2.0 m from the work area.  The SO101's global camera is 0.66 m up, so the
shipped presets barely register: measured on a real layout, ``coastal`` moved the
object's contrast by 5%.  This probe re-calibrates by *visibility in metres*
(``WaterParams.from_visibility``, ``beta ~ 3 / visibility``), which is a physical
label that can be printed on a slide.

Three measurement traps this probe exists to avoid:

* **RMS contrast is the wrong metric.**  Backscatter and particles *add*
  high-frequency structure, so a heavily degraded frame can score a *higher* RMS
  contrast than a clean one -- measured: the background edge energy rose from 59
  to 73 as the water got worse.  The primary metric here is the **object-vs-
  background luminance difference**, i.e. "can you still tell the object from
  what is behind it", which noise averages out of rather than into.
* **One arm pose does not work for both cameras.**  At the grasp pose the hand
  hides the object from the top-down camera (object mask: 0 px); at HOME the
  wrist camera looks at empty table.  Each camera is therefore measured at the
  pose where it can actually see the object, and the pose is recorded.
* **A snowstorm is not a difficulty axis.**  ``from_visibility`` scales the whole
  parameter set, and below a metre it puts so many bright specks in the frame
  that they dominate the picture.  Attenuation and backscatter are the physical
  part; the particle layer is capped, and the resulting rung is re-measured so
  the gradient claim is still checked.

    PYTHONPATH=<WYSD> uv run python -m mj_env.scripts.water_probe --images

Outputs ``configs/vfx/so101_tabletop.json`` (consumed by ``mj_env.vfx``) and
side-by-side PNGs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from underwater_vision import WaterParams, WaterRenderer, save_png

from mj_env.env import HOME_QPOS
from mj_env.scene import SCENE_CAMERA, WRIST_CAMERA
from mj_env.task_env import TaskEnv
from mj_env.vfx import CLEAN_RUNG, CONFIG_PATH, LADDER, write_config

#: Visibilities to sweep, in metres.  Camera-to-target is about 0.66 m, so the
#: interesting range is sub-metre: at 2 m visibility the water is nearly
#: transparent over that path.
VISIBILITY_SWEEP: tuple[float, ...] = (5.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.45, 0.3)

#: Target relative object-vs-background signal for the three water rungs.  A rung
#: indistinguishable from its neighbour carries no information; a rung where the
#: signal is gone is a blind task, not a hard one.
TARGETS: dict[str, float] = {"v1_light": 0.70, "v2_medium": 0.45, "v3_heavy": 0.25}

#: Which arm pose each camera is measured at -- see the module docstring.
POSE_FOR_CAMERA: dict[str, str] = {SCENE_CAMERA: "home", WRIST_CAMERA: "grasp"}

#: Caps on the cosmetic particle layer (see the module docstring).
PARTICLE_DENSITY_CAP = 0.55
PARTICLE_BRIGHTNESS_CAP = 0.30

CAMERAS = (SCENE_CAMERA, WRIST_CAMERA)
RING_WIDTH = 6


def rung_params(visibility: float) -> WaterParams:
    """Water parameters for one visibility, with the particle layer capped."""
    params = WaterParams.from_visibility(visibility)
    return params.replace(
        particle_density=min(float(params.particle_density), PARTICLE_DENSITY_CAP),
        particle_brightness=min(float(params.particle_brightness), PARTICLE_BRIGHTNESS_CAP),
    )


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0].astype(float)
        + 0.7152 * rgb[..., 1].astype(float)
        + 0.0722 * rgb[..., 2].astype(float)
    )


def rms_contrast(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    luma = _luminance(image)
    if mask is not None and mask.any():
        luma = luma[mask]
    if luma.size == 0 or float(luma.mean()) <= 0.0:
        return 0.0
    return float(luma.std() / luma.mean())


def edge_energy(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean gradient magnitude.  Noise-sensitive: use it as a *noise* indicator."""
    from scipy import ndimage

    magnitude = np.hypot(ndimage.sobel(_luminance(image), 0), ndimage.sobel(_luminance(image), 1))
    if mask is not None and mask.any():
        magnitude = magnitude[mask]
    return float(magnitude.mean())


def background_ring(mask: np.ndarray, width: int = RING_WIDTH) -> np.ndarray:
    """A ring just outside the object: the "what is it sitting against" sample."""
    from scipy import ndimage

    grown = ndimage.binary_dilation(mask, iterations=width)
    return grown & ~mask


def object_signal(image: np.ndarray, mask: np.ndarray, ring: np.ndarray) -> float:
    """Object-vs-background luminance difference -- the primary degradation metric."""
    if not mask.any() or not ring.any():
        return 0.0
    luma = _luminance(image)
    return float(luma[mask].mean() - luma[ring].mean())


def set_arm_pose(env: TaskEnv, kind: str) -> None:
    """Park the arm where the camera under test can actually see the object."""
    if kind == "home":
        env.data.qpos[env._qpos_ids] = np.asarray(HOME_QPOS, dtype=float)
    elif kind == "grasp":
        from mj_env.scripts.scripted_pick import (
            PAD_CLEARANCE,
            GraspSolver,
            PickTuning,
            grasp_axis_yaw,
        )

        solver = GraspSolver(env.model)
        width = env._objects[env.spec.target].grip_width
        offset = solver.calibration.offset_for(width + 2.0 * PAD_CLEARANCE)
        solutions = solver.solve_waypoints(
            env.prop_pos(env.spec.target),
            grasp_axis_yaw(env, env.spec.target),
            offset,
            PickTuning(),
        )
        if solutions is None:
            raise RuntimeError("the grasp pose could not be solved; cannot image the object")
        jaw = solver.calibration.angle_for(width + 2.0 * PAD_CLEARANCE)
        env.data.qpos[env._qpos_ids] = np.append(solutions["grasp"], jaw)
    else:
        raise ValueError(f"unknown arm pose {kind!r}")
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def object_mask(env: TaskEnv, renderer: WaterRenderer, camera: str) -> np.ndarray:
    """Pixels the manipulated object occupies, by differencing it in and out.

    The trick ``WYSD/obstacle_grasp`` used for occlusion: move the object out of
    the world, render, put it back, render, threshold the difference.  No
    segmentation model, and it is exactly the region the policy has to look at.
    """
    address = env._objects[env.spec.target].qpos_adr
    saved = env.data.qpos[address : address + 7].copy()

    with_object = renderer.render(env.data, camera=camera, apply=False)
    env.data.qpos[address : address + 3] = (0.0, 0.0, -1.0)
    mujoco.mj_forward(env.model, env.data)
    without_object = renderer.render(env.data, camera=camera, apply=False)
    env.data.qpos[address : address + 7] = saved
    mujoco.mj_forward(env.model, env.data)

    difference = np.abs(with_object.astype(int) - without_object.astype(int)).max(axis=2)
    return difference > 6


def measure(
    env: TaskEnv,
    renderer: WaterRenderer,
    camera: str,
    mask: np.ndarray,
    ring: np.ndarray,
    params: WaterParams | None,
) -> dict:
    if params is None:
        image = renderer.render(env.data, camera=camera, apply=False)
    else:
        renderer.set_params(params)
        image = renderer.render(env.data, camera=camera)
    return {
        "signal_object_vs_background": object_signal(image, mask, ring),
        "rms_contrast": rms_contrast(image),
        "rms_contrast_object": rms_contrast(image, mask),
        "background_edge_energy": edge_energy(image, ring),
        "object_edge_energy": edge_energy(image, mask),
        "image": image,
    }


def _numbers(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if key != "image"}


def choose_rungs(sweep: list[tuple[float, float]]) -> dict[str, float]:
    """Pick, per target, the swept visibility landing closest to it."""
    chosen: dict[str, float] = {
        name: min(sweep, key=lambda item: abs(item[1] - target))[0]
        for name, target in TARGETS.items()
    }
    if len(set(chosen.values())) != len(chosen):
        # Two rungs collapsing onto one visibility means the ladder is not a
        # ladder; make it distinct so any "difference" is a real difference.
        ladder = sorted({value for value, _ in sweep})
        mid = len(ladder) // 2
        chosen = {
            "v1_light": ladder[max(0, mid - 1)],
            "v2_medium": ladder[mid],
            "v3_heavy": ladder[min(len(ladder) - 1, mid + 1)],
        }
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="p1_place_disc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path(".tmp/vfx"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()

    env = TaskEnv(args.task, image_size=(args.size, args.size), max_episode_steps=10_000)
    env.reset(seed=args.seed)
    renderer = WaterRenderer(env.model, args.size, args.size, params=None, seed=args.seed)

    cameras: dict[str, dict] = {}
    sweeps: dict[str, list[tuple[float, float]]] = {}
    try:
        for camera in CAMERAS:
            pose = POSE_FOR_CAMERA[camera]
            set_arm_pose(env, pose)
            mask = object_mask(env, renderer, camera)
            if int(mask.sum()) == 0:
                raise RuntimeError(
                    f"{camera} sees none of the object at pose {pose!r}; the measurement "
                    "would be a whole-image statistic wearing an object-region label"
                )
            ring = background_ring(mask)
            clean = measure(env, renderer, camera, mask, ring, None)
            base = max(abs(clean["signal_object_vs_background"]), 1e-9)

            print(f"\n{camera} @ {pose}: object {int(mask.sum())} px, ring {int(ring.sum())} px")
            print(f"  {'visibility':>10} {'obj-bg':>9} {'rel':>7} {'bg noise':>9} {'RMS':>7}")
            print(
                f"  {'clean':>10} {clean['signal_object_vs_background']:>9.1f}"
                f" {1.0:>7.3f} {clean['background_edge_energy']:>9.2f}"
                f" {clean['rms_contrast']:>7.3f}"
            )
            sweep: list[tuple[float, float]] = []
            rungs: dict[str, dict] = {}
            for visibility in VISIBILITY_SWEEP:
                params = rung_params(visibility)
                result = measure(env, renderer, camera, mask, ring, params)
                relative = abs(result["signal_object_vs_background"]) / base
                sweep.append((visibility, relative))
                rungs[f"v_{visibility}"] = {
                    "visibility_m": visibility,
                    "hash": params.hash(),
                    "params": params.to_dict(),
                    **_numbers(result),
                    "relative_signal": relative,
                }
                print(
                    f"  {visibility:>9.2f}m {result['signal_object_vs_background']:>9.1f}"
                    f" {relative:>7.3f} {result['background_edge_energy']:>9.2f}"
                    f" {result['rms_contrast']:>7.3f}"
                )
            cameras[camera] = {
                "arm_pose": pose,
                "mask_pixels": int(mask.sum()),
                "clean": _numbers(clean),
                "sweep": rungs,
            }
            sweeps[camera] = sweep

        chosen = choose_rungs(sweeps[SCENE_CAMERA])
        print("\nselected rungs (by global_camera object-vs-background signal):")
        rung_payload: dict[str, dict] = {
            CLEAN_RUNG: {
                "mode": "clean",
                "measured": {camera: cameras[camera]["clean"] for camera in CAMERAS},
            }
        }
        for name, visibility in chosen.items():
            params = rung_params(visibility)
            measured = {
                camera: {
                    key: value
                    for key, value in cameras[camera]["sweep"][f"v_{visibility}"].items()
                    if key != "params"
                }
                for camera in CAMERAS
            }
            rung_payload[name] = {
                "mode": "water",
                "visibility_m": visibility,
                "hash": params.hash(),
                "params": params.to_dict(),
                "measured": measured,
            }
            print(
                f"  {name:<10} visibility {visibility:>5.2f} m"
                f"  rel_signal global={measured[SCENE_CAMERA]['relative_signal']:.3f}"
                f"  wrist={measured[WRIST_CAMERA]['relative_signal']:.3f}"
                f"  bg_noise global={measured[SCENE_CAMERA]['background_edge_energy']:.1f}"
            )

        payload = {
            "scene": args.task,
            "image_size": args.size,
            "seed": args.seed,
            "ladder": list(LADDER),
            "calibration": (
                "visibility chosen by matching the object-vs-background signal of "
                "global_camera to the targets " + json.dumps(TARGETS, ensure_ascii=False)
            ),
            "targets": TARGETS,
            "rungs": rung_payload,
            "cameras": cameras,
        }
        print(f"\nwrote {write_config(payload, args.config)}")

        if args.images:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for camera in CAMERAS:
                set_arm_pose(env, POSE_FOR_CAMERA[camera])
                save_png(
                    args.out_dir / f"{args.seed}_{camera}_{CLEAN_RUNG}.png",
                    renderer.render(env.data, camera=camera, apply=False),
                )
                for name in chosen:
                    renderer.set_params(rung_params(chosen[name]))
                    save_png(
                        args.out_dir / f"{args.seed}_{camera}_{name}.png",
                        renderer.render(env.data, camera=camera),
                    )
            print(f"wrote paired PNGs to {args.out_dir}")
    finally:
        renderer.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())