# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate a few start/goal example pairs for the object-push task across all
9 target kitchens, with BOTH the side (``robot0_agentview_left``) and wrist
(``robot0_eye_in_hand``) cameras, all rendered at 240x320 (HxW).

For each kitchen x episode this drives ``reset(mode='init')`` then
``reset(mode='goal')``, prints the sampled start/goal object world positions
(+ push distance d0), and composites the pair into a highlighted 2x2 quad:

    +-----------------------------------+   <- colored border highlights the
    | INIT side    | INIT wrist         |      init/goal PAIR as one block
    | GOAL side    | GOAL wrist         |
    +-----------------------------------+

Outputs (in object_push_examples/):
  * per-(kitchen, episode) quad PNGs   sc<KID>_ep<E>_quad.png
  * one annotated grid                 grid.png
    (rows = kitchens, columns = episodes; each cell = the highlighted quad)

Run (headless):

    cd <repo_root>
    MUJOCO_GL=egl PYTHONPATH="$PWD:$PWD/scripts/deployment:$PYTHONPATH" \
      python \
      scripts/deployment/robots/robocasa/generate_object_push_examples.py

Env vars: KITCHENS="1 4 9", EPISODES=3, OBJECT_NAME=banana, NO_VIZ=1.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from robots.robocasa.blocking_robot_env import (  # noqa: E402
    _OBJECT_PUSH_INNER_TASK_NAME,
    ObjectPushBlockingRoboCasaEnv,
)
from robots.robocasa.robot_env import RoboCasaEnv  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_KITCHENS = tuple(
    int(x) for x in os.environ.get("KITCHENS", "1 2 3 4 5 6 8 9 10").split()
)
N_EPISODES = int(os.environ.get("EPISODES", "3"))
OBJECT_NAME = os.environ.get("OBJECT_NAME", "flour_bag")
# Forward-cone direction filter (deg full width). Empty = uniform 360 (default).
# The planner runs used FORWARD_CONE_DEG=60 (i.e. +/-30 deg forward).
FORWARD_CONE_DEG = os.environ.get("FORWARD_CONE_DEG", "")
SCENE_SEED = 0
# img_0 = side view, img_1 = wrist view.
CAMERAS = ["robot0_agentview_left", "robot0_eye_in_hand"]
CAM_H, CAM_W = 240, 320  # 240x320 (HxW) as requested
NO_VIZ = bool(int(os.environ.get("NO_VIZ", "0")))
OUT_DIR = os.path.join(_THIS_DIR, "object_push_examples")

# BGR is what cv2.imwrite expects after the [:, :, ::-1] flip; we build images
# in RGB and flip on write, so border colors below are RGB.
_BORDER_RGB = (40, 200, 60)  # green highlight around each init/goal pair
_GAP_RGB = (255, 255, 255)  # white gaps between quads / rows
_BORDER = 6
_GAP = 14


def _label(img, text):
    import cv2

    out = np.ascontiguousarray(img.copy())
    H, W = out.shape[:2]
    bar_h = max(20, H // 12)
    cv2.rectangle(out, (0, 0), (W, bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (6, max(15, bar_h - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, bar_h / 40.0),
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return out


def _title_bar(width, text, height=26):
    import cv2

    bar = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (6, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 200, 60),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return bar


def _quad(obs_i, obs_g, title):
    """2x2 highlighted block: [init side | init wrist] over [goal side | goal
    wrist], with a title bar and a colored border around the whole pair."""
    import cv2

    top = np.concatenate(
        [_label(obs_i["img_0"], "INIT side"), _label(obs_i["img_1"], "INIT wrist")],
        axis=1,
    )
    bot = np.concatenate(
        [_label(obs_g["img_0"], "GOAL side"), _label(obs_g["img_1"], "GOAL wrist")],
        axis=1,
    )
    block = np.concatenate([_title_bar(top.shape[1], title), top, bot], axis=0)
    return cv2.copyMakeBorder(
        block,
        _BORDER,
        _BORDER,
        _BORDER,
        _BORDER,
        cv2.BORDER_CONSTANT,
        value=_BORDER_RGB,
    )


def _hstack_gap(imgs, gap, color):
    if len(imgs) == 1:
        return imgs[0]
    H = imgs[0].shape[0]
    sep = np.full((H, gap, 3), color, dtype=np.uint8)
    out = [imgs[0]]
    for im in imgs[1:]:
        out.extend([sep, im])
    return np.concatenate(out, axis=1)


def _vstack_gap(imgs, gap, color):
    if len(imgs) == 1:
        return imgs[0]
    W = imgs[0].shape[1]
    sep = np.full((gap, W, 3), color, dtype=np.uint8)
    out = [imgs[0]]
    for im in imgs[1:]:
        out.extend([sep, im])
    return np.concatenate(out, axis=0)


def _make_env(kid):
    inner = RoboCasaEnv(
        task_name=_OBJECT_PUSH_INNER_TASK_NAME,
        camera_names=list(CAMERAS),
        camera_width=CAM_W,
        camera_height=CAM_H,
        seed=SCENE_SEED,
        robot="PandaOmron",
        init_robot_to_home=True,
        reach_task=False,
        layout_and_style_ids=[(kid, kid)],
    )
    return ObjectPushBlockingRoboCasaEnv(
        inner,
        object_push_object_name=OBJECT_NAME,
        object_push_task_seed=0,
        object_push_forward_cone_deg=(float(FORWARD_CONE_DEG) if FORWARD_CONE_DEG else None),
        object_push_clear_radius=1.0,
    )


def main():
    if not NO_VIZ:
        os.makedirs(OUT_DIR, exist_ok=True)

    print(
        f"object-push examples | object={OBJECT_NAME} | kitchens={TARGET_KITCHENS} | "
        f"episodes={N_EPISODES} | cameras={CAMERAS} | {CAM_H}x{CAM_W}\n"
    )
    header = (
        f"{'kitchen':>7} {'ep':>3} | "
        f"{'start_world (x,y,z)':>28} | {'goal_world (x,y,z)':>28} | {'d0(m)':>6}"
    )
    print(header)
    print("-" * len(header))

    grid_rows = []
    for kid in TARGET_KITCHENS:
        env = _make_env(kid)
        quads = []
        try:
            for ep in range(N_EPISODES):
                obs_i, _ = env.reset(mode="init", object_push_task_seed=ep)
                obs_g, _ = env.reset(mode="goal", object_push_task_seed=ep)
                s = np.asarray(obs_g["object_push_start_world"], dtype=np.float64)
                g = np.asarray(obs_g["object_push_goal_world"], dtype=np.float64)
                d0 = float(obs_g["object_push_d0"])
                print(
                    f"{kid:>7} {ep:>3} | "
                    f"({s[0]:+.3f},{s[1]:+.3f},{s[2]:+.3f}) | "
                    f"({g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f}) | {d0:>6.3f}"
                )
                if NO_VIZ:
                    continue
                import cv2

                quad = _quad(obs_i, obs_g, f"k{kid} ep{ep}  d0={d0:.3f} m")
                cv2.imwrite(
                    os.path.join(OUT_DIR, f"sc{kid:03d}_ep{ep}_quad.png"),
                    quad[:, :, ::-1],
                )
                quads.append(quad)
        finally:
            try:
                env.close()
            except Exception:
                pass
        if not NO_VIZ and quads:
            grid_rows.append(_hstack_gap(quads, _GAP, _GAP_RGB))

    if not NO_VIZ and grid_rows:
        import cv2

        width = max(r.shape[1] for r in grid_rows)
        padded = [
            np.pad(
                r,
                ((0, 0), (0, width - r.shape[1]), (0, 0)),
                mode="constant",
                constant_values=255,
            )
            for r in grid_rows
        ]
        grid = _vstack_gap(padded, _GAP, _GAP_RGB)
        grid_path = os.path.join(OUT_DIR, "grid.png")
        cv2.imwrite(grid_path, grid[:, :, ::-1])
        print(f"\nwrote per-pair quad PNGs + grid to {OUT_DIR}/ (grid: {grid_path})")


if __name__ == "__main__":
    main()
