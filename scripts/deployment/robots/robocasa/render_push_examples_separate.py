# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render object-push start/goal examples as 4 SEPARATE images per variant.

Same setup as the planner runs: brown ``flourbag001``, forward-cone (+/-30 deg)
placement, kitchens {4,5,6,8}. For each (kitchen, variant) it drives
``reset(mode='goal')`` then ``reset(mode='init')`` and writes FOUR separate PNGs
at 480x640 (HxW):

    sc<KID>_ep<E>_init_side.png    sc<KID>_ep<E>_init_wrist.png
    sc<KID>_ep<E>_goal_side.png    sc<KID>_ep<E>_goal_wrist.png

Run (headless, needs a GPU):

    cd <repo_root>
    MUJOCO_GL=egl PYTHONPATH="$PWD:$PWD/scripts/deployment:$PYTHONPATH" \
      python scripts/deployment/robots/robocasa/render_push_examples_separate.py

Env vars: KITCHENS="4 5 6 8", EPISODES=5, OBJECT_NAME=flourbag001,
          FORWARD_CONE_DEG=60, CAM_H=480, CAM_W=640.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import robots.robocasa.object_push_kitchen  # noqa: F401,E402  (registers ObjectPushKitchen)
from robots.robocasa.blocking_robot_env import ObjectPushBlockingRoboCasaEnv  # noqa: E402
from robots.robocasa.robot_env import RoboCasaEnv  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
KITCHENS = [int(x) for x in os.environ.get("KITCHENS", "4 5 6 8").split()]
N_EPISODES = int(os.environ.get("EPISODES", "5"))
OBJECT_NAME = os.environ.get("OBJECT_NAME", "flourbag001")
FORWARD_CONE_DEG = float(os.environ.get("FORWARD_CONE_DEG", "60"))
CAM_H = int(os.environ.get("CAM_H", "480"))
CAM_W = int(os.environ.get("CAM_W", "640"))
CAMERAS = ["robot0_agentview_left", "robot0_eye_in_hand"]  # img_0=side, img_1=wrist
OUT_DIR = os.path.join(_THIS_DIR, "object_push_examples_separate")


def make_env(kid: int) -> ObjectPushBlockingRoboCasaEnv:
    inner = RoboCasaEnv(
        task_name="ObjectPushKitchen",
        camera_names=list(CAMERAS),
        camera_width=CAM_W,
        camera_height=CAM_H,
        seed=0,
        robot="PandaOmron",
        init_robot_to_home=True,
        reach_task=False,
        layout_and_style_ids=[(kid, kid)],
    )
    return ObjectPushBlockingRoboCasaEnv(
        inner,
        object_push_object_name=OBJECT_NAME,
        object_push_task_seed=0,
        object_push_forward_cone_deg=FORWARD_CONE_DEG,
        object_push_clear_radius=1.0,
    )


def main() -> None:
    import cv2

    os.makedirs(OUT_DIR, exist_ok=True)
    print(
        f"object-push separate images | object={OBJECT_NAME} | kitchens={KITCHENS} | "
        f"episodes={N_EPISODES} | cone={FORWARD_CONE_DEG} deg | {CAM_H}x{CAM_W}\n"
    )
    n_written = 0
    for kid in KITCHENS:
        env = make_env(kid)
        try:
            for ep in range(N_EPISODES):
                # same seed => same (start, goal) for both resets
                obs_g, _ = env.reset(mode="goal", object_push_task_seed=ep)
                obs_i, _ = env.reset(mode="init", object_push_task_seed=ep)
                d0 = float(obs_g["object_push_d0"])
                views = {
                    "init_side": obs_i["img_0"],
                    "init_wrist": obs_i["img_1"],
                    "goal_side": obs_g["img_0"],
                    "goal_wrist": obs_g["img_1"],
                }
                for name, img in views.items():
                    arr = np.asarray(img)
                    path = os.path.join(OUT_DIR, f"sc{kid:03d}_ep{ep}_{name}.png")
                    cv2.imwrite(path, arr[:, :, ::-1])  # RGB -> BGR for cv2
                    n_written += 1
                print(f"  sc{kid:03d} ep{ep}: wrote 4 files ({CAM_H}x{CAM_W}), d0={d0:.3f} m")
        finally:
            try:
                env.close()
            except Exception:
                pass
    print(f"\nwrote {n_written} PNGs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
