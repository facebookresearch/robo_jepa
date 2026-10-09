# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Quickly enumerate which target kitchens have a CoffeeMachine that
passes the picker's in-front + on-axis filter.
"""

from __future__ import annotations

import os
import sys
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


def _build_env(seed: int):
    from robots.robocasa.blocking_robot_env import ObstacleReachBlockingRoboCasaEnv
    from robots.robocasa.robot_env import RoboCasaEnv

    inner = RoboCasaEnv(
        task_name="CloseStandMixerHead",
        camera_names=["robot0_agentview_left"],
        camera_width=160,
        camera_height=120,
        seed=seed,
        robot="PandaOmron",
        chunk_size=4,
        wm_embodiment="franka_omron_state_delta",
        reach_task=True,
        reach_task_seed=seed,
        reach_min_distance=0.25,
        reach_max_distance=0.50,
        reach_clear_radius=1.0,
        kitchen_split="target",
    )
    return ObstacleReachBlockingRoboCasaEnv(
        inner,
        reach_task_seed=seed,
        reach_min_distance=0.25,
        reach_max_distance=0.50,
        reach_clear_radius=1.0,
        kitchen_split="target",
        obstacle_classes=("CoffeeMachine",),
        obstacle_clearance_m=0.05,
        obstacle_lateral_jitter_std=0.05,
    )


def main():
    summary: list[tuple[int, str]] = []
    for seed in range(10):
        env = _build_env(seed)
        try:
            # Force the reset machinery so _reach_home_eef_pos is set
            # and we can call _pick_obstacle_fixture.
            try:
                env.reset(mode="init", seed=seed, reach_task_seed=seed, reset_episode_counter=True)
                meta = env._obstacle_meta
                msg = (
                    f"OK   {meta['name']!r} "
                    f"offaxis_y={meta['offaxis_y_base_m']*1000:.0f}mm "
                    f"y_half={meta['y_half_extent_m']*1000:.0f}mm "
                    f"top_z={meta['world_top_z']:.3f}m "
                    f"yspan=[{meta['y_base_min']*1000:.0f}, {meta['y_base_max']*1000:.0f}]mm"
                )
            except RuntimeError as exc:
                first = str(exc).splitlines()[0]
                msg = f"FAIL ({first[:160]})"
        finally:
            try:
                env.close()
            except Exception:
                pass
        print(f"sc{seed:03d}: {msg}", flush=True)
        summary.append((seed, msg))
    print(flush=True)
    print("=== usable kitchens ===", flush=True)
    for s, m in summary:
        if m.startswith("OK"):
            print(f"  sc{s:03d}: {m}", flush=True)


if __name__ == "__main__":
    main()
