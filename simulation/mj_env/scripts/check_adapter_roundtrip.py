"""Verify SO101 motor-unit <-> MuJoCo-radian conversion and clipping."""

from __future__ import annotations

import numpy as np

from mj_env.adapters import LeRobotSO101Adapter, NORMALIZED_RANGES
from mj_env.env import SO101TabletopEnv


def main() -> int:
    env = SO101TabletopEnv(render_mode=None)
    try:
        adapter = LeRobotSO101Adapter(env)
        low = np.asarray([a for a, _ in NORMALIZED_RANGES], dtype=np.float32)
        high = np.asarray([b for _, b in NORMALIZED_RANGES], dtype=np.float32)
        for fraction in (0.0, 0.1, 0.5, 0.9, 1.0):
            normalized = low + fraction * (high - low)
            restored = adapter.to_lerobot(adapter.from_lerobot(normalized))
            if not np.allclose(restored, normalized, atol=1e-4):
                raise AssertionError((fraction, normalized, restored))
        beyond = high + 10.0
        clipped = adapter.to_lerobot(adapter.from_lerobot(beyond))
        if not np.allclose(clipped, high, atol=1e-4):
            raise AssertionError(("upper clipping", clipped, high))
        print("ADAPTER_ROUNDTRIP_PASS; out-of-range motor targets are clipped")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
