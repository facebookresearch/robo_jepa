# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal example: create and run the RoboCasa object-push env (no planner).

Constructs ObjectPushBlockingRoboCasaEnv, samples a start/goal pair (with the
forward-cone direction filter), prints the geometry, and rolls out a few steps
pushing the closed-gripper EEF toward the goal while printing the dense reward.

Run (headless, needs a GPU for EGL rendering):

    cd <repo_root>
    MUJOCO_GL=egl PYTHONPATH="$PWD:$PWD/scripts/deployment:$PYTHONPATH" \
      python scripts/deployment/robots/robocasa/minimal_push_env.py

Env vars: KITCHEN=4, OBJECT=flourbag001, TASK_SEED=0, STEPS=18.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Importing the module registers "ObjectPushKitchen" in the robosuite registry.
import robots.robocasa.object_push_kitchen  # noqa: F401,E402
from robots.robocasa.blocking_robot_env import ObjectPushBlockingRoboCasaEnv  # noqa: E402
from robots.robocasa.robot_env import RoboCasaEnv  # noqa: E402

KITCHEN = int(os.environ.get("KITCHEN", "4"))
OBJECT = os.environ.get("OBJECT", "flourbag001")
SEED = int(os.environ.get("TASK_SEED", "0"))
STEPS = int(os.environ.get("STEPS", "18"))


def make_env(kitchen_id: int) -> ObjectPushBlockingRoboCasaEnv:
    inner = RoboCasaEnv(
        task_name="ObjectPushKitchen",
        camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
        camera_width=320,
        camera_height=240,
        seed=0,
        robot="PandaOmron",
        init_robot_to_home=True,
        lock_base_in_arm_mode=True,
        wm_embodiment="franka_omron_state_delta",
        layout_and_style_ids=[(kitchen_id, kitchen_id)],  # pin one kitchen
    )
    return ObjectPushBlockingRoboCasaEnv(
        inner,
        object_push_object_name=OBJECT,
        object_push_task_seed=SEED,
        object_push_min_distance=0.10,
        object_push_max_distance=0.15,
        object_push_forward_cone_deg=60.0,  # forward-cone direction filter (+/-30 deg)
        object_push_clear_radius=1.0,
        pos_tol=0.001,
        max_steps=200,
    )


def main() -> None:
    env = make_env(KITCHEN)

    # reset(mode="goal") caches the goal object pose AND computes d0;
    # reset(mode="init") then places the object at the start. Same task seed
    # => the same (start, goal) pair for both.
    goal_obs, _ = env.reset(mode="goal", object_push_task_seed=SEED)
    obs, _ = env.reset(mode="init", object_push_task_seed=SEED)

    start = np.asarray(goal_obs["object_push_start_world"], dtype=np.float64)
    goal = np.asarray(goal_obs["object_push_goal_world"], dtype=np.float64)
    d0 = float(goal_obs["object_push_d0"])
    print(f"kitchen={KITCHEN} object={OBJECT} seed={SEED}")
    print(f"  start={start.round(3)}  goal={goal.round(3)}  d0={d0:.3f} m")

    # Roll out a few steps pushing the (closed-gripper) EEF toward the goal.
    # Action = [dx, dy, dz, drx, dry, drz, gripper]; gripper +1 = closed pusher.
    unit = (goal - start) / (np.linalg.norm(goal - start) + 1e-9)
    for t in range(STEPS):
        action = np.zeros(7, dtype=np.float64)
        action[:3] = 0.03 * unit  # small forward delta each step
        action[6] = +1.0
        obs, reward, done, info = env.step(action)
        print(
            f"  t={t:02d} reward={reward:+.3f} "
            f"dist_to_goal={info.get('object_push_distance_to_goal', float('nan')):.3f}"
        )
        if done:
            break


if __name__ == "__main__":
    main()
