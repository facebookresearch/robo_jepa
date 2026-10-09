# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render start/goal example PNGs for the obstacle-reach task.

For each requested target kitchen seed, samples episodes one at a time
(fresh — does NOT consult the old-sampler golden), classifies each as
L2R or R2L by the robot-base Y axis, and captures the first 2 L2R + 1
R2L. Each example renders 4 PNGs (init / goal × side / wrist camera).

Designed to be driven in parallel from the shell (one process per
kitchen seed):

    seq 5 9 | xargs -P 5 -I{} bash -c "\\
        MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
        PYTHONPATH=... \\
        python -u scripts/deployment/robots/robocasa/render_obstacle_reach_examples.py \\
            --seeds {}"

then merge the per-kitchen rows into one grid via ``--merge``.
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

# Eval-pipeline knobs mirror generate_obstacle_reach_golden.py.
_TASK_NAME = "CloseStandMixerHead"
_CAMERA_NAMES = ("robot0_agentview_left", "robot0_eye_in_hand")
_CAMERA_WIDTH = 320
_CAMERA_HEIGHT = 240
_ROBOT = "PandaOmron"
_CHUNK_SIZE = 4
_WM_EMBODIMENT = "franka_omron_state_delta"
_REACH_MIN_DISTANCE = 0.40
_REACH_MAX_DISTANCE = 0.80
_REACH_CLEAR_RADIUS = 1.0
_BLOCKING_POS_TOL = 0.001
_BLOCKING_MAX_STEPS = 200
_OBSTACLE_CLEARANCE_M = 0.12
_OBSTACLE_JITTER_STD = 0.05
_OBSTACLE_CLASSES = ("CoffeeMachine",)

_OUT_DIR = _THIS_FILE.parent / "obstacle_reach_examples"

# Default kitchens to render (caller can override via --seeds). Picked
# from the 9 working target seeds, excluding sc6 (its episodes often exhaust
# the obstacle sampler).
_DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5, 7, 8, 9)

# How many L2R / R2L examples we want per kitchen. Walking the per-seed
# episode counter is the bottleneck (each ``init+goal`` pair costs one
# fresh sampler invocation ≈ 90 s), so cap the search at a generous
# upper bound that we shouldn't normally hit.
_TARGET_L2R = 2
_TARGET_R2L = 1
_MAX_EPISODES_TO_SEARCH = 10


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


def _classify_l2r_or_r2l(obs_init: dict, obs_goal: dict, env) -> str:
    """Return ``"L2R"`` if start sits on +Y_base of the obstacle and goal on
    -Y_base, ``"R2L"`` for the reverse. (``L`` = robot's left =
    +Y_base.) Used to bucket episodes into 2 L2R + 1 R2L per kitchen.
    """
    base_pos_world, R_wb = env._env._get_base_world_pose()
    R_bw = R_wb.T
    obstacle_w = np.asarray(obs_init["obstacle_world_pos"], dtype=np.float64)
    obstacle_y = float((R_bw @ (obstacle_w - base_pos_world))[1])
    start_y = float((R_bw @ (np.asarray(obs_init["reach_start_world"], dtype=np.float64) - base_pos_world))[1])
    goal_y = float((R_bw @ (np.asarray(obs_goal["reach_goal_world"], dtype=np.float64) - base_pos_world))[1])
    if start_y > obstacle_y and goal_y < obstacle_y:
        return "L2R"
    if start_y < obstacle_y and goal_y > obstacle_y:
        return "R2L"
    return "?"


def _save_pngs(obs: dict, out_dir: Path, prefix: str) -> list[tuple[Path, str]]:
    """Save every ``img_<k>`` in ``obs``. Returns ``[(path, camera_short_name), ...]``."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[Path, str]] = []
    for cam_idx, cam_name in enumerate(_CAMERA_NAMES):
        img_key = f"img_{cam_idx}"
        if img_key not in obs:
            continue
        img = np.asarray(obs[img_key], dtype=np.uint8)
        out_path = out_dir / f"{prefix}_cam{cam_idx}_{cam_name}.png"
        Image.fromarray(img).save(out_path, optimize=False)
        cam_short = "side" if "agentview" in cam_name else "wrist"
        paths.append((out_path, cam_short))
    return paths


def _render_kitchen(scene_seed: int) -> None:
    """Sample episodes until we have 2 L2R + 1 R2L (or run out of budget),
    saving 4 PNGs per kept example. Writes an index file
    ``out_dir/index.txt`` listing the rows in the final grid order so
    the merge step can re-assemble.
    """
    out_dir = _OUT_DIR / f"sc{scene_seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _build_env(scene_seed)
    found: dict[str, list[tuple[int, list[tuple[Path, str]], list[tuple[Path, str]]]]] = {
        "L2R": [],
        "R2L": [],
    }
    try:
        for ep in range(_MAX_EPISODES_TO_SEARCH):
            if len(found["L2R"]) >= _TARGET_L2R and len(found["R2L"]) >= _TARGET_R2L:
                break
            t0 = time.perf_counter()
            kw = dict(mode="init", seed=scene_seed, reach_task_seed=scene_seed)
            if ep == 0:
                kw["reset_episode_counter"] = True
            try:
                obs_init, _ = env.reset(**kw)
                obs_goal, _ = env.reset(mode="goal", seed=scene_seed, reach_task_seed=scene_seed)
            except RuntimeError as exc:
                msg = str(exc).splitlines()[0]
                print(
                    f"[sc{scene_seed:03d}] ep{ep}: sampler failed ({msg}); "
                    f"skipping ({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
                continue
            cat = _classify_l2r_or_r2l(obs_init, obs_goal, env)
            wanted = (cat == "L2R" and len(found["L2R"]) < _TARGET_L2R) or (
                cat == "R2L" and len(found["R2L"]) < _TARGET_R2L
            )
            if not wanted:
                print(
                    f"[sc{scene_seed:03d}] ep{ep}: cat={cat} (have "
                    f"L2R={len(found['L2R'])}/{_TARGET_L2R} "
                    f"R2L={len(found['R2L'])}/{_TARGET_R2L}); skipping "
                    f"({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
                continue
            init_paths = _save_pngs(obs_init, out_dir, f"ep{ep:02d}_{cat}_init")
            goal_paths = _save_pngs(obs_goal, out_dir, f"ep{ep:02d}_{cat}_goal")
            found[cat].append((ep, init_paths, goal_paths))
            print(
                f"[sc{scene_seed:03d}] ep{ep}: cat={cat} captured "
                f"(L2R={len(found['L2R'])}/{_TARGET_L2R} "
                f"R2L={len(found['R2L'])}/{_TARGET_R2L}, "
                f"{time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
    # Persist row order so --merge can find the per-kitchen rows
    # without re-running the env.
    index_path = out_dir / "index.txt"
    with index_path.open("w") as f:
        f.write("# scene_seed ep cat init_cam0 init_cam1 goal_cam0 goal_cam1\n")
        for cat_label in ("L2R", "R2L"):
            for ep, init_paths, goal_paths in found[cat_label]:
                tokens = [str(scene_seed), str(ep), cat_label]
                for p, _ in init_paths + goal_paths:
                    tokens.append(p.name)
                f.write(" ".join(tokens) + "\n")
    if len(found["L2R"]) < _TARGET_L2R or len(found["R2L"]) < _TARGET_R2L:
        print(
            f"[sc{scene_seed:03d}] WARN: under target after "
            f"{_MAX_EPISODES_TO_SEARCH} episodes "
            f"(L2R={len(found['L2R'])}/{_TARGET_L2R}, "
            f"R2L={len(found['R2L'])}/{_TARGET_R2L})",
            flush=True,
        )


def _merge_grid(seeds: list[int]) -> None:
    """Compose the per-kitchen index files into one annotated grid."""
    from PIL import Image, ImageDraw, ImageFont

    rows: list[list[tuple[Path, str]]] = []
    for sd in seeds:
        idx = _OUT_DIR / f"sc{sd:03d}" / "index.txt"
        if not idx.exists():
            print(f"[merge] sc{sd:03d}: no index — skipping", flush=True)
            continue
        for line in idx.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            tok = line.split()
            scene = int(tok[0])
            ep = int(tok[1])
            cat = tok[2]
            files = tok[3:]
            row: list[tuple[Path, str]] = []
            cell_labels = (
                ("init", "side"),
                ("init", "wrist"),
                ("goal", "side"),
                ("goal", "wrist"),
            )
            for fname, (which, cam_short) in zip(files, cell_labels):
                p = _OUT_DIR / f"sc{scene:03d}" / fname
                row.append((p, f"sc{scene:03d} ep{ep:02d} {cat} {which} {cam_short}"))
            rows.append(row)
    if not rows:
        print("[merge] no rows to merge", flush=True)
        return

    tile_w, tile_h = _CAMERA_WIDTH, _CAMERA_HEIGHT
    label_h = 24
    cell_w, cell_h = tile_w, tile_h + label_h
    cols = max(len(r) for r in rows)
    canvas = Image.new(
        "RGB", (cols * cell_w, len(rows) * cell_h), color=(20, 20, 20)
    )
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for r, row in enumerate(rows):
        for c, (path, label) in enumerate(row):
            img = Image.open(path).convert("RGB").resize((tile_w, tile_h))
            canvas.paste(img, (c * cell_w, r * cell_h))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle(
                [c * cell_w, r * cell_h + tile_h, (c + 1) * cell_w, (r + 1) * cell_h],
                fill=(40, 40, 40),
            )
            draw.text(
                (c * cell_w + 8, r * cell_h + tile_h + 4),
                label,
                fill=(220, 220, 220),
                font=font,
            )
    grid_path = _OUT_DIR / "grid.png"
    canvas.save(grid_path, optimize=False)
    print(f"[merge] wrote {grid_path} ({canvas.size[0]}x{canvas.size[1]})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Scene seeds for this worker (default: 3 4 7 8 9 — the 5 working kitchens).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Skip rendering; scan each sc<NNN>/index.txt and emit the composite grid.",
    )
    args = parser.parse_args()
    seeds = list(args.seeds) if args.seeds is not None else list(_DEFAULT_SEEDS)
    if args.merge:
        _merge_grid(seeds)
        return
    t0 = time.perf_counter()
    for sd in seeds:
        _render_kitchen(sd)
    print(
        f"[done] seeds={seeds} rendered in "
        f"{(time.perf_counter() - t0) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
