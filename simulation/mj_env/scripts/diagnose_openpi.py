"""Diagnose observable failure stages of an OpenPI SO101 pick policy.

Use the same policy request and action execution as evaluate_openpi.py. Output
contains measured geometry, contacts, tracking and timing; labels are symptoms,
not proof that vision, language or the model weights caused a failure.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mj_env import objects
from mj_env.adapters import LeRobotSO101Adapter, NORMALIZED_RANGES
from mj_env.env import SO101TabletopEnv


def diagnose(record: dict) -> list[str]:
    if record["success"]:
        return ["success"]
    labels = []
    # Center distance is only a coarse proxy: a long rod can be grasped far
    # from its center. Do not interpret this label as proven target misdetection.
    if record["min_tcp_target_distance_m"] > 0.05:
        labels.append("tcp_not_within_5cm_of_object_center")
    elif not record["ever_grasped"]:
        labels.append("approached_but_no_two_pad_grasp")
    elif record["max_lift_height_m"] < 0.06:
        labels.append("grasped_but_not_lifted_6cm")
    else:
        labels.append("grasp_or_lift_not_sustained_together")
    if record["wrong_object_two_pad_contact_steps"]:
        labels.append("other_object_contact")
    if record["clipped_action_steps"]:
        labels.append("action_outside_motor_range")
    if record["mean_joint_tracking_error_rad"] > 0.15:
        labels.append("large_joint_tracking_error")
    return labels


def run_episode(client, target: str, seed: int, max_steps: int, replan_steps: int) -> dict:
    env = SO101TabletopEnv(target_object=target, render_mode=None, max_episode_steps=max_steps)
    adapter = LeRobotSO101Adapter(env)
    lo = np.asarray([a for a, _ in NORMALIZED_RANGES], dtype=np.float32)
    hi = np.asarray([b for _, b in NORMALIZED_RANGES], dtype=np.float32)
    start = time.monotonic()
    try:
        observation, info = env.reset(seed=seed, options={"target_object": target})
        min_distance = float(info["tcp_target_distance"])
        max_lift = float(info["lift_height"])
        ever_grasped = bool(info["grasped"])
        first_grasp_step = None
        closest_step = 0
        wrong_contact_steps = 0
        clipped_steps = 0
        clipped_by_motor = np.zeros(6, dtype=np.int64)
        max_motor_range_excess = 0.0
        tracking_errors = []
        grasp_contact_steps = 0
        grasp_run = 0
        longest_grasp_run = 0
        inference_seconds = []
        executed = 0
        while executed < max_steps and not info["success"]:
            request = {
                "observation/image": observation["observation.image"],
                "observation/wrist_image": observation["observation.wrist_image"],
                "observation/state": adapter.to_lerobot(observation["observation.state"]),
                "prompt": env.task,
            }
            infer_start = time.monotonic()
            chunk = np.asarray(client.infer(request)["actions"], dtype=np.float32)
            inference_seconds.append(time.monotonic() - infer_start)
            if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != 6 or not np.isfinite(chunk).all():
                raise ValueError(f"invalid policy action chunk: {chunk.shape}")
            for normalized_action in chunk[:replan_steps]:
                excess = np.maximum(np.maximum(lo - normalized_action, normalized_action - hi), 0.0)
                clipped = excess > 0.5  # ignore tiny float round-off in motor units
                clipped_steps += int(bool(np.any(clipped)))
                clipped_by_motor += clipped
                max_motor_range_excess = max(max_motor_range_excess, float(np.max(excess)))
                motor_target = adapter.from_lerobot(normalized_action)
                observation, _, terminated, truncated, info = env.step(motor_target)
                executed += 1
                tracking_errors.append(float(np.mean(np.abs(observation["observation.state"] - motor_target))))
                distance = float(info["tcp_target_distance"])
                if distance < min_distance:
                    min_distance, closest_step = distance, executed
                max_lift = max(max_lift, float(info["lift_height"]))
                if info["grasped"] and first_grasp_step is None:
                    first_grasp_step = executed
                ever_grasped |= bool(info["grasped"])
                grasp_contact_steps += int(bool(info["grasped"]))
                grasp_run = grasp_run + 1 if info["grasped"] else 0
                longest_grasp_run = max(longest_grasp_run, grasp_run)
                wrong_contact_steps += int(any(
                    env._pads_engaged(metrics.body_id)
                    for name, metrics in env._objects.items()
                    if name != target and name in env.objects_subset
                ))
                if terminated or truncated or executed >= max_steps:
                    break
            if terminated or truncated:
                break
        record = {
            "target": target,
            "seed": seed,
            "success": bool(info["success"]),
            "steps": executed,
            "max_steps": max_steps,
            "min_tcp_target_distance_m": round(min_distance, 5),
            "closest_step": closest_step,
            "ever_grasped": ever_grasped,
            "first_grasp_step": first_grasp_step,
            "grasp_contact_steps": grasp_contact_steps,
            "longest_grasp_run_steps": longest_grasp_run,
            "max_lift_height_m": round(max_lift, 5),
            "wrong_object_two_pad_contact_steps": wrong_contact_steps,
            "clipped_action_steps": clipped_steps,
            "clipped_by_motor": clipped_by_motor.tolist(),
            "max_motor_range_excess": round(max_motor_range_excess, 4),
            "mean_joint_tracking_error_rad": round(float(np.mean(tracking_errors)), 5),
            "mean_inference_seconds": round(float(np.mean(inference_seconds)), 4),
            "inference_calls": len(inference_seconds),
            "wall_seconds": round(time.monotonic() - start, 2),
            "layout_fallback": bool(info["layout_fallback"]),
        }
        record["symptoms"] = diagnose(record)
        return record
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--targets", nargs="+", choices=objects.NAMES, default=list(objects.NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1000])
    parser.add_argument("--cases", nargs="*", help="optional target:seed pairs, overriding targets/seeds")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.replan_steps, args.max_steps) <= 0:
        parser.error("replan-steps and max-steps must be positive")
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    cases = []
    if args.cases:
        for case in args.cases:
            target, separator, seed_text = case.partition(":")
            if not separator or target not in objects.NAMES:
                parser.error(f"invalid case {case!r}; expected target:seed")
            try:
                cases.append((target, int(seed_text)))
            except ValueError:
                parser.error(f"invalid seed in {case!r}")
    else:
        cases = [(target, seed) for target in args.targets for seed in args.seeds]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for target, seed in cases:
            record = run_episode(client, target, seed, args.max_steps, args.replan_steps)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
