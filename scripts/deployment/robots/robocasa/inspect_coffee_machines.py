# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Print the world / base-frame position of every CoffeeMachine in
every target kitchen, regardless of the picker's filters. Lets us see
why no kitchen passes the in-front + on-axis cap.
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


def main():
    from robots.robocasa.robot_env import RoboCasaEnv

    for seed in range(10):
        env = RoboCasaEnv(
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
        try:
            # Drive a normal reach reset so _reach_home_eef_pos is set.
            env.reset(mode="init", seed=seed, reach_task_seed=seed, reset_episode_counter=True)
            base_pos, R_wb = env._get_base_world_pose()
            R_bw = R_wb.T
            inner = env._env
            print(f"\n=== sc{seed:03d} ===")
            for name, fixture in inner.fixtures.items():
                cls = type(fixture).__name__
                if cls != "CoffeeMachine":
                    continue
                root_body = getattr(fixture, "root_body", None)
                if root_body is None:
                    print(f"  {name}: no root_body")
                    continue
                try:
                    bid = int(inner.sim.model.body_name2id(root_body))
                except Exception:
                    print(f"  {name}: body lookup failed")
                    continue
                w = np.asarray(inner.sim.data.body_xpos[bid], dtype=np.float64)
                b = R_bw @ (w - base_pos)
                size = getattr(fixture, "size", [None, None, None])
                print(
                    f"  {name!r}: world={w.tolist()}  base=[{b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f}]  "
                    f"size={list(size)}"
                )
        finally:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
