"""Build the clean/underwater ladder sheet used as report material."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from underwater_vision import save_png

SRC = Path(".tmp/vfx")
RUNGS = ("v0_clean", "v1_light", "v2_medium", "v3_heavy")
LABELS = ("V0 clean", "V1 3.00m", "V2 1.20m", "V3 0.60m")


def band(width: int, height: int, value: int) -> np.ndarray:
    return np.full((height, width, 3), value, np.uint8)


def build(camera: str) -> Path:
    panels = []
    for rung in RUNGS:
        image = np.asarray(Image.open(SRC / f"0_{camera}_{rung}.png").convert("RGB"))
        # A white strip on top and a black strip at the bottom, so a reader can
        # tell where one panel ends and the next begins on a projector.
        panels.append(np.vstack([band(image.shape[1], 10, 255), image, band(image.shape[1], 4, 0)]))
    sheet = np.hstack(panels)
    out = SRC / f"ladder_{camera}.png"
    save_png(out, sheet)
    return out


def main() -> int:
    for camera in ("global_camera", "wrist_camera"):
        out = build(camera)
        print(f"{out}  labels={LABELS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())