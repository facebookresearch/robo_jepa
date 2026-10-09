# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate N object-reach-task JSON files + per-task start/goal preview images.

Each task file contains a (scene_seed, object_reach_task_seed) pair that is
forwarded to ``ObjectReachBlockingRoboCasaEnv.reset()`` via
``task_info["init"|"goal"]`` by the planner client.

The accompanying preview PNG shows the rendered start (left) and goal
(right) image side-by-side at the requested cameras (default
``robot0_agentview_left`` + ``robot0_eye_in_hand``) so a reviewer can
eyeball whether each task is reasonable before kicking off a multi-hour
planner run. Tasks whose programmatic top-down grasp fails on either
the init or the goal reset are skipped (logged to stdout) and produce
neither a JSON nor a preview.

Usage
-----
    cd scripts/deployment/robots/robocasa
    conda activate robo_jepa
    python generate_object_reach_tasks.py \
        --n_tasks 50 \
        --object_name banana \
        --tasks_dir <path>/tasks_robocasa_object_reach \
        --previews_dir <path>/tasks_robocasa_object_reach_previews
"""
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser

# Headless rendering must be set before importing mujoco/robosuite.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Importing ``blocking_robot_env`` side-effect-registers ``ObjectReachKitchen``
# with robosuite (via ``object_reach_kitchen``), so the inner env can be
# built with ``task_name="ObjectReachKitchen"`` below.
from robots.robocasa.blocking_robot_env import (  # noqa: E402
    _OBJECT_REACH_INNER_TASK_NAME,
    ObjectReachBlockingRoboCasaEnv,
)
from robots.robocasa.robot_env import RoboCasaEnv  # noqa: E402


def _label_image(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = img.copy()
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


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("Usage")[0])
    parser.add_argument(
        "--task_name",
        type=str,
        default=_OBJECT_REACH_INNER_TASK_NAME,
        help=(
            "Underlying robocasa task. Default is the generic "
            "ObjectReachKitchen empty-kitchen task; override only if a "
            "custom kitchen subclass is registered."
        ),
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default="banana",
        help="Object key in OBJECT_REACH_REGISTRY (default: banana).",
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=50,
        help="Number of (scene, object-reach) tasks to generate.",
    )
    parser.add_argument(
        "--scene_seed_offset",
        type=int,
        default=0,
        help="First scene seed (subsequent ones are scene_seed_offset + i).",
    )
    parser.add_argument(
        "--object_reach_seed_offset",
        type=int,
        default=0,
        help=(
            "First object-reach-task seed (subsequent ones are "
            "object_reach_seed_offset + i)."
        ),
    )
    parser.add_argument(
        "--fix_scene_seed",
        action="store_true",
        default=False,
        help=(
            "If set, scene_seed stays constant at scene_seed_offset for all "
            "episodes (only object_reach_task_seed varies as "
            "object_reach_seed_offset + i). Output filenames use the "
            "_ep{i:03d} suffix instead of _rs{seed:03d} to indicate "
            "per-episode variation."
        ),
    )
    parser.add_argument("--object_reach_max_distance", type=float, default=0.15)
    parser.add_argument("--object_reach_clear_radius", type=float, default=1.0)
    parser.add_argument("--object_reach_max_retries", type=int, default=3)
    parser.add_argument(
        "--target_kitchen_id",
        type=int,
        default=None,
        help=(
            "Optional: pin one of robocasa's 10 target kitchens. When set, "
            "layout_and_style_ids=[(ID, ID)] is passed to robosuite.make and "
            "output filenames use the object_reach_{object_name}_t{ID}_rs{NNN}.json "
            "pattern (object_name is the ``--object_name`` arg)."
        ),
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["robot0_agentview_left", "robot0_eye_in_hand"],
        help="Cameras for the preview render (and for the planner env).",
    )
    parser.add_argument("--camera_width", type=int, default=320)
    parser.add_argument("--camera_height", type=int, default=240)
    parser.add_argument(
        "--tasks_dir",
        type=str,
        required=True,
        help="Output directory for the .json task files.",
    )
    parser.add_argument(
        "--previews_dir",
        type=str,
        required=True,
        help="Output directory for the per-task start|goal preview PNGs.",
    )
    parser.add_argument(
        "--summary_grid",
        type=str,
        default=None,
        help=(
            "Optional path to save a single grid PNG containing all "
            "tasks (one row per task, columns = init left, init right, "
            "goal left, goal right). Useful as a one-shot reviewer view."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.tasks_dir, exist_ok=True)
    os.makedirs(args.previews_dir, exist_ok=True)
    if args.summary_grid:
        os.makedirs(os.path.dirname(args.summary_grid) or ".", exist_ok=True)

    print(
        f"[generate_object_reach_tasks] generating {args.n_tasks} tasks "
        f"(scene_seed={args.scene_seed_offset}+i, "
        f"object_reach_task_seed={args.object_reach_seed_offset}+i) "
        f"task={args.task_name} object={args.object_name} "
        f"max_dist={args.object_reach_max_distance} m "
        f"clear_radius={args.object_reach_clear_radius} m "
        f"cameras={args.camera_names}"
    )

    rows_for_grid: list[np.ndarray] = []
    n_emitted = 0
    n_skipped = 0

    for i in range(args.n_tasks):
        if args.fix_scene_seed:
            scene_seed = args.scene_seed_offset
        else:
            scene_seed = args.scene_seed_offset + i
        object_reach_seed = args.object_reach_seed_offset + i
        if args.fix_scene_seed:
            suffix = f"_ep{i:03d}"
        else:
            suffix = f"_rs{object_reach_seed:03d}"
        if args.target_kitchen_id is not None:
            task_name_str = (
                f"object_reach_{args.object_name}_t{args.target_kitchen_id}{suffix}"
            )
        else:
            task_name_str = (
                f"object_reach_{args.object_name}_sc{scene_seed:03d}{suffix}"
            )
        print(
            f"[generate_object_reach_tasks] === [{i+1}/{args.n_tasks}] "
            f"{task_name_str} ==="
        )

        # 1) Construct a fresh inner env per task so the kitchen is
        # rebuilt with the target scene_seed; wrap with the
        # ObjectReach blocking wrapper so reset() runs the
        # clear/spawn/grasp/carry sequence.
        layout_and_style_ids = (
            [(args.target_kitchen_id, args.target_kitchen_id)]
            if args.target_kitchen_id is not None
            else None
        )
        inner_env = RoboCasaEnv(
            task_name=args.task_name,
            camera_names=list(args.camera_names),
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            seed=scene_seed,
            robot="PandaOmron",
            init_robot_to_home=True,
            reach_task=False,
            layout_and_style_ids=layout_and_style_ids,
        )
        env = ObjectReachBlockingRoboCasaEnv(
            inner_env,
            object_reach_object_name=args.object_name,
            object_reach_task_seed=object_reach_seed,
            object_reach_max_distance=args.object_reach_max_distance,
            object_reach_clear_radius=args.object_reach_clear_radius,
            object_reach_max_retries=args.object_reach_max_retries,
        )
        try:
            obs_init, _ = env.reset(mode="init")
            if obs_init.get("object_reach_grasp_failed", False):
                print(
                    f"  [skip] grasp failed on init for {task_name_str}"
                )
                n_skipped += 1
                continue

            obs_goal, _ = env.reset(mode="goal")
            if obs_goal.get("object_reach_grasp_failed", False):
                print(
                    f"  [skip] grasp failed on goal for {task_name_str}"
                )
                n_skipped += 1
                continue

            d0 = float(
                np.linalg.norm(
                    obs_goal["object_reach_obj_pos_world"]
                    - obs_init["object_reach_obj_pos_world"]
                )
            )
            r_init = float(obs_init["object_reach_residual"])
            r_goal = float(obs_goal["object_reach_residual"])
            print(
                f"  d_0 (start\u2194goal) = {d0:.4f} m  "
                f"residual init = {r_init:.4f} m  goal = {r_goal:.4f} m"
            )

            # 2) Save preview PNG (init|goal across all cameras).
            import cv2

            init_strip = []
            goal_strip = []
            for col_idx, cam_name in enumerate(args.camera_names):
                cam_short = cam_name.split("_")[-1]
                init_strip.append(
                    _label_image(
                        obs_init[f"img_{col_idx}"],
                        f"{task_name_str} init {cam_short}  d0={d0:.2f}m",
                    )
                )
                goal_strip.append(
                    _label_image(
                        obs_goal[f"img_{col_idx}"],
                        f"{task_name_str} goal {cam_short}  d0={d0:.2f}m",
                    )
                )
            preview = np.concatenate([
                np.concatenate(init_strip, axis=1),
                np.concatenate(goal_strip, axis=1),
            ], axis=0)
            preview_path = os.path.join(
                args.previews_dir, f"{task_name_str}.png"
            )
            cv2.imwrite(preview_path, preview[:, :, ::-1])

            if args.summary_grid:
                # 1 row per task, cameras laid out as init+goal alternating.
                row = np.concatenate(init_strip + goal_strip, axis=1)
                rows_for_grid.append(row)
        finally:
            env.close()

        # 3) Write task JSON. Both init and goal carry the wrapper-side
        # kwargs so the planner-side ``env.reset(**task_info[mode])``
        # forwards them verbatim to ``ObjectReachBlockingRoboCasaEnv.reset``
        # (changing scene_seed triggers an inner-env rebuild; changing
        # object_reach_task_seed re-rolls the (start_eef, goal_eef) pair).
        task_info = {
            "scene_seed": scene_seed,
            "object_reach_task_seed": object_reach_seed,
            "object_name": args.object_name,
            "object_reach_max_distance": args.object_reach_max_distance,
            "expected_d0_m": d0,
            "init": {
                "mode": "init",
                "seed": scene_seed,
                "object_reach_task_seed": object_reach_seed,
                "object_reach_object_name": args.object_name,
            },
            "goal": {
                "mode": "goal",
                "seed": scene_seed,
                "object_reach_task_seed": object_reach_seed,
                "object_reach_object_name": args.object_name,
            },
        }
        task_path = os.path.join(args.tasks_dir, f"{task_name_str}.json")
        with open(task_path, "w") as f:
            json.dump(task_info, f, indent=2)
        n_emitted += 1

    if args.summary_grid and rows_for_grid:
        import cv2

        grid = np.concatenate(rows_for_grid, axis=0)
        cv2.imwrite(args.summary_grid, grid[:, :, ::-1])
        print(
            f"[generate_object_reach_tasks] wrote summary grid "
            f"{args.summary_grid} ({grid.shape[1]}x{grid.shape[0]}, "
            f"{len(rows_for_grid)} rows)"
        )

    print(
        f"[generate_object_reach_tasks] done: {n_emitted} task files in "
        f"{args.tasks_dir} ({n_skipped} skipped due to grasp failure)"
    )
    print(
        f"[generate_object_reach_tasks] previews in {args.previews_dir}"
    )


if __name__ == "__main__":
    main()
