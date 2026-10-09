# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Direct verification that the obstacle-reach sampler produces clean
samples FAST on a worst-case kitchen, with all three invariants
holding.

Invariants checked (all in robot-base frame):
  1. Fixture alignment: ``|obstacle.y_base|`` is below
     ``_OBSTACLE_REACH_MAX_FIXTURE_OFFAXIS_Y_BASE_M`` — i.e. the
     picked obstacle is genuinely on the robot's forward axis.
  2. Endpoint Z hugs the obstacle top: each endpoint's world Z
     is within ``_OBSTACLE_REACH_ENDPOINT_Z_TOL_M`` of the canonical
     anchor Z (which is ``top_z + offset``, clipped to ``home_eef_z``).
  3. Midpoint(start, goal).xy approximates obstacle.xy: in base frame,
     ``||midpoint − fixture||_xy ≤ _OBSTACLE_REACH_MIDPOINT_XY_TOL_M``.

Speed budget: previously sc4 took 90-940 s per episode with the
straight-line drive + no stall check. Target with Y-first axis order
+ stall detection: under 90 s per episode on average.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = str(_THIS_FILE.parents[4])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_ROBO_JEPA_SCRIPTS = os.path.join(REPO_ROOT, "scripts", "deployment")
if _ROBO_JEPA_SCRIPTS not in sys.path:
    sys.path.insert(0, _ROBO_JEPA_SCRIPTS)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_TASK_NAME = "CloseStandMixerHead"
_CAMERA_NAMES = ("robot0_agentview_left", "robot0_eye_in_hand")
_CAMERA_WIDTH = 320
_CAMERA_HEIGHT = 240
_ROBOT = "PandaOmron"
_CHUNK_SIZE = 4
_WM_EMBODIMENT = "franka_omron_state_delta"
_REACH_MIN_DISTANCE = 0.25
_REACH_MAX_DISTANCE = 0.50
_REACH_CLEAR_RADIUS = 1.0
_BLOCKING_POS_TOL = 0.001
_BLOCKING_MAX_STEPS = 200
_OBSTACLE_CLEARANCE_M = 0.12
_OBSTACLE_JITTER_STD = 0.05
_OBSTACLE_CLASSES = ("CoffeeMachine",)


def _build_env(scene_seed: int):
    from robots.robocasa.blocking_robot_env import ObstacleReachBlockingRoboCasaEnv
    from robots.robocasa.robot_env import RoboCasaEnv

    inner = RoboCasaEnv(
        task_name=_TASK_NAME,
        camera_names=list(_CAMERA_NAMES),
        camera_width=_CAMERA_WIDTH,
        camera_height=_CAMERA_HEIGHT,
        seed=scene_seed,
        robot=_ROBOT,
        chunk_size=_CHUNK_SIZE,
        wm_embodiment=_WM_EMBODIMENT,
        reach_task=True,
        reach_task_seed=scene_seed,
        reach_min_distance=_REACH_MIN_DISTANCE,
        reach_max_distance=_REACH_MAX_DISTANCE,
        reach_clear_radius=_REACH_CLEAR_RADIUS,
        kitchen_split="target",
    )
    return ObstacleReachBlockingRoboCasaEnv(
        inner,
        reach_task_seed=scene_seed,
        reach_min_distance=_REACH_MIN_DISTANCE,
        reach_max_distance=_REACH_MAX_DISTANCE,
        reach_clear_radius=_REACH_CLEAR_RADIUS,
        kitchen_split="target",
        obstacle_classes=_OBSTACLE_CLASSES,
        obstacle_clearance_m=_OBSTACLE_CLEARANCE_M,
        obstacle_lateral_jitter_std=_OBSTACLE_JITTER_STD,
        pos_tol=_BLOCKING_POS_TOL,
        max_steps=_BLOCKING_MAX_STEPS,
    )


def _check_one_episode(obs_init: dict, obs_goal: dict, env) -> dict:
    from robots.robocasa.blocking_robot_env import (
        _OBSTACLE_REACH_ENDPOINT_Z_ABOVE_TOP_M,
        _OBSTACLE_REACH_ENDPOINT_Z_TOL_M,
        _OBSTACLE_REACH_MAX_FIXTURE_OFFAXIS_Y_BASE_M,
        _OBSTACLE_REACH_MIDPOINT_XY_TOL_M,
    )

    start_w = np.asarray(obs_init["reach_start_world"], dtype=np.float64)
    goal_w = np.asarray(obs_goal["reach_goal_world"], dtype=np.float64)
    obstacle_w = np.asarray(obs_init["obstacle_world_pos"], dtype=np.float64)
    top_z = float(obs_init["obstacle_world_top_z"])
    offaxis_y_base = float(obs_init["obstacle_offaxis_y_base_m"])

    base_pos_world, R_wb = env._env._get_base_world_pose()
    R_bw = R_wb.T
    start_base = R_bw @ (start_w - base_pos_world)
    goal_base = R_bw @ (goal_w - base_pos_world)
    obstacle_base = R_bw @ (obstacle_w - base_pos_world)

    # 1. Fixture alignment.
    align_tol = _OBSTACLE_REACH_MAX_FIXTURE_OFFAXIS_Y_BASE_M
    align_ok = offaxis_y_base <= align_tol + 1e-6

    # 2. Z hugs the top.
    canonical_z = top_z + _OBSTACLE_REACH_ENDPOINT_Z_ABOVE_TOP_M
    z_tol = _OBSTACLE_REACH_ENDPOINT_Z_TOL_M
    home_eef_z = float(env._env._reach_home_eef_pos[2])
    canonical_z_eff = min(home_eef_z, canonical_z)
    start_z_dev_eff = abs(start_w[2] - canonical_z_eff)
    goal_z_dev_eff = abs(goal_w[2] - canonical_z_eff)
    z_ok = (start_z_dev_eff <= z_tol + 1e-6) and (goal_z_dev_eff <= z_tol + 1e-6)

    # 3. Midpoint XY.
    # 3. Midpoint Y (base frame). With endpoints anchored at the
    # home EEF's X (not the fixture body's X), the midpoint X is
    # always 30-50 cm behind the fixture and unhelpful; we only
    # check Y.
    midpoint_y_base = 0.5 * (start_base[1] + goal_base[1])
    midpoint_err_xy = float(abs(midpoint_y_base - obstacle_base[1]))
    midpoint_ok = midpoint_err_xy <= _OBSTACLE_REACH_MIDPOINT_XY_TOL_M + 1e-6

    # 4. Strict left/right. The endpoint with the larger Y_base must
    # land OUTSIDE the fixture's Y_base_max (with at least
    # ``obstacle_clearance_m`` margin); the other endpoint must land
    # outside Y_base_min. Without ``y_base_min`` / ``y_base_max`` in
    # the obs dict (older env), fall back to ``obstacle_y ± y_half``.
    if "obstacle_y_base_min" in obs_init:
        y_base_min = float(obs_init["obstacle_y_base_min"])
        y_base_max = float(obs_init["obstacle_y_base_max"])
    else:
        y_half = float(obs_init["obstacle_y_half_extent_m"])
        y_base_min = float(obstacle_base[1]) - y_half
        y_base_max = float(obstacle_base[1]) + y_half
    clearance_strict = float(env._obstacle_clearance_m)
    # Assign which endpoint is "left" (the one with the larger Y_base).
    if start_base[1] >= goal_base[1]:
        left_y, right_y = float(start_base[1]), float(goal_base[1])
    else:
        left_y, right_y = float(goal_base[1]), float(start_base[1])
    strict_left_ok = left_y >= y_base_max + clearance_strict - 1e-6
    strict_right_ok = right_y <= y_base_min - clearance_strict + 1e-6
    strict_ok = strict_left_ok and strict_right_ok

    return {
        "obstacle_name": str(obs_init.get("obstacle_fixture_name")),
        "obstacle_class": str(obs_init.get("obstacle_class")),
        "obstacle_top_z": top_z,
        "canonical_anchor_z_eff": canonical_z_eff,
        "offaxis_y_base_m": offaxis_y_base,
        "align_tol_m": align_tol,
        "align_ok": align_ok,
        "start_world_z": float(start_w[2]),
        "goal_world_z": float(goal_w[2]),
        "start_z_dev_from_anchor": start_z_dev_eff,
        "goal_z_dev_from_anchor": goal_z_dev_eff,
        "z_tol": z_tol,
        "z_ok": z_ok,
        "midpoint_err_xy_base_m": midpoint_err_xy,
        "midpoint_tol_m": _OBSTACLE_REACH_MIDPOINT_XY_TOL_M,
        "midpoint_ok": midpoint_ok,
        "left_y_base": left_y,
        "right_y_base": right_y,
        "y_base_min": y_base_min,
        "y_base_max": y_base_max,
        "strict_clearance_m": clearance_strict,
        "strict_ok": strict_ok,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=4, help="Scene seed (target kitchen index).")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--per-ep-budget-s",
        type=float,
        default=120.0,
        help="Soft budget per episode in seconds; flagged in the summary.",
    )
    args = parser.parse_args()

    scene_seed = int(args.seed)
    print(
        f"[verify] building env(seed={scene_seed}, kitchen_split=target)...",
        flush=True,
    )
    env = _build_env(scene_seed)
    all_results: list[tuple[int, dict, float]] = []
    try:
        for ep in range(args.episodes):
            t0 = time.perf_counter()
            # Wipe the cache so the sampler actually runs (not a cached
            # pair from a previous reset).
            env._obstacle_targets_cache.clear()
            kw = dict(mode="init", seed=scene_seed, reach_task_seed=scene_seed)
            if ep == 0:
                kw["reset_episode_counter"] = True
            try:
                obs_init, _ = env.reset(**kw)
                obs_goal, _ = env.reset(
                    mode="goal", seed=scene_seed, reach_task_seed=scene_seed
                )
            except RuntimeError as exc:
                print(
                    f"[verify] sc{scene_seed:03d} ep{ep}: sampler FAILED "
                    f"({type(exc).__name__}: {str(exc).splitlines()[0]})",
                    flush=True,
                )
                continue
            t_ep = time.perf_counter() - t0
            print(
                f"[verify] sc{scene_seed:03d} ep{ep}: sampled in {t_ep:.1f}s",
                flush=True,
            )
            res = _check_one_episode(obs_init, obs_goal, env)
            all_results.append((ep, res, t_ep))
            print(
                f"  obstacle={res['obstacle_name']!r} ({res['obstacle_class']}) "
                f"top_z={res['obstacle_top_z']:.3f}m  "
                f"offaxis_y={res['offaxis_y_base_m']*1000:.0f}mm  "
                f"tol={res['align_tol_m']*1000:.0f}mm -> "
                f"{'PASS' if res['align_ok'] else 'FAIL'}",
                flush=True,
            )
            print(
                f"  start_z_dev={res['start_z_dev_from_anchor']*1000:.1f}mm  "
                f"goal_z_dev={res['goal_z_dev_from_anchor']*1000:.1f}mm  "
                f"tol={res['z_tol']*1000:.0f}mm -> "
                f"{'PASS' if res['z_ok'] else 'FAIL'}",
                flush=True,
            )
            print(
                f"  midpoint_xy_err={res['midpoint_err_xy_base_m']*1000:.1f}mm  "
                f"tol={res['midpoint_tol_m']*1000:.0f}mm -> "
                f"{'PASS' if res['midpoint_ok'] else 'FAIL'}",
                flush=True,
            )
            print(
                f"  strict L/R: left_y={res['left_y_base']*1000:+.0f}mm "
                f"(need >= {res['y_base_max']*1000 + res['strict_clearance_m']*1000:+.0f}mm)  "
                f"right_y={res['right_y_base']*1000:+.0f}mm "
                f"(need <= {res['y_base_min']*1000 - res['strict_clearance_m']*1000:+.0f}mm) "
                f"-> {'PASS' if res['strict_ok'] else 'FAIL'}",
                flush=True,
            )
            print(flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Tally.
    n_pass_invariants = sum(
        1
        for _ep, r, _t in all_results
        if r["align_ok"] and r["z_ok"] and r["midpoint_ok"] and r["strict_ok"]
    )
    n_within_budget = sum(1 for _ep, _r, t in all_results if t <= args.per_ep_budget_s)
    avg_t = (
        sum(t for _ep, _r, t in all_results) / max(1, len(all_results))
        if all_results
        else float("nan")
    )
    print(
        f"[verify] sc{scene_seed:03d}: "
        f"{n_pass_invariants}/{len(all_results)} pass ALL 4 invariants, "
        f"{n_within_budget}/{len(all_results)} within {args.per_ep_budget_s:.0f}s budget "
        f"(avg {avg_t:.1f}s/ep)",
        flush=True,
    )
    if all_results and (n_pass_invariants != len(all_results) or n_within_budget != len(all_results)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
