# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the per-(scene, episode) (start, goal) + obstacle golden file
for the obstacle-reach invariants tests.

Mirrors :mod:`scripts.robo_jepa.robots.robocasa.generate_reach_invariants_golden`
but writes a target-only golden at
``scripts/deployment/robots/robocasa/obstacle_reach_golden_target.json``.
Per-episode payload additionally captures ``obstacle_fixture_name`` and
``obstacle_world_pos`` so the test suite can pin both the spatial and
the identity invariants.

Two modes:

1. **Per-seed worker** (default): for each ``scene_seed`` passed via
   ``--seeds``, run all 10 episodes and write a per-seed partial JSON
   to ``--partials-dir/sc<XXX>.json``. Also, when ``--viz-dir`` is set,
   saves the two camera views (side + wrist) for the start and goal
   frames as PNGs so the operator can sanity-check the layouts.

2. **Merge** (``--merge``): scan ``--partials-dir`` for every
   ``sc<XXX>.json`` partial and merge them into the final golden file
   at ``--output`` (``scripts/deployment/robots/robocasa/obstacle_reach_golden_target.json``).

Usage
-----
Worker (1 seed):
    python -u scripts/deployment/robots/robocasa/generate_obstacle_reach_golden.py \\
        --seeds 0 \\
        --partials-dir scripts/deployment/robots/robocasa/obstacle_reach_golden_partials_target \\
        --viz-dir scripts/deployment/robots/robocasa/obstacle_reach_golden_viz_target

Merge:
    python -u scripts/deployment/robots/robocasa/generate_obstacle_reach_golden.py --merge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Auto-detect the repo root from this file's location so the script
# works from any checkout. The script lives at
# ``<repo_root>/scripts/deployment/robots/robocasa/generate_obstacle_reach_golden.py``.
_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = str(_THIS_FILE.parents[4])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
ROBO_JEPA_SCRIPTS = os.path.join(REPO_ROOT, "scripts", "deployment")
if ROBO_JEPA_SCRIPTS not in sys.path:
    sys.path.insert(0, ROBO_JEPA_SCRIPTS)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# ── Production eval defaults (mirror generate_reach_invariants_golden.py)
DEFAULT_TASK_NAME = "CloseStandMixerHead"
DEFAULT_CAMERA_NAMES = ("robot0_agentview_left", "robot0_eye_in_hand")
DEFAULT_CAMERA_WIDTH = 320
DEFAULT_CAMERA_HEIGHT = 240
DEFAULT_ROBOT = "PandaOmron"
DEFAULT_CHUNK_SIZE = 4
DEFAULT_WM_EMBODIMENT = "franka_omron_state_delta"
DEFAULT_REACH_MIN_DISTANCE = 0.25
DEFAULT_REACH_MAX_DISTANCE = 0.50
DEFAULT_REACH_CLEAR_RADIUS = 1.0
DEFAULT_BLOCKING_POS_TOL = 0.001
DEFAULT_BLOCKING_MAX_STEPS = 200
DEFAULT_OBSTACLE_CLEARANCE_M = 0.05
DEFAULT_OBSTACLE_JITTER_STD = 0.05
DEFAULT_OBSTACLE_CLASSES = ("StandMixer", "Blender", "CoffeeMachine", "Toaster")

# Obstacle-reach is a target-only task: each ``seed in 0..9`` maps to a
# distinct (layout, style) pair via ``seed % 10`` inside
# ``RoboCasaEnv._create_env``.
DEFAULT_TARGET_SCENE_SEEDS = tuple(range(10))
DEFAULT_EPISODES = 10
SCHEMA_VERSION = 1

_THIS_DIR = _THIS_FILE.parent


def _default_partials_dir() -> Path:
    return _THIS_DIR / "obstacle_reach_golden_partials_target"


def _default_viz_dir() -> Path:
    return _THIS_DIR / "obstacle_reach_golden_viz_target"


def _default_output_path() -> Path:
    return _THIS_DIR / "obstacle_reach_golden_target.json"


def _build_env(scene_seed: int):
    """Construct an :class:`ObstacleReachBlockingRoboCasaEnv` matching the eval setup."""
    from robots.robocasa.blocking_robot_env import ObstacleReachBlockingRoboCasaEnv
    from robots.robocasa.robot_env import RoboCasaEnv

    inner = RoboCasaEnv(
        task_name=DEFAULT_TASK_NAME,
        camera_names=list(DEFAULT_CAMERA_NAMES),
        camera_width=DEFAULT_CAMERA_WIDTH,
        camera_height=DEFAULT_CAMERA_HEIGHT,
        seed=scene_seed,
        robot=DEFAULT_ROBOT,
        chunk_size=DEFAULT_CHUNK_SIZE,
        wm_embodiment=DEFAULT_WM_EMBODIMENT,
        reach_task=True,
        reach_task_seed=scene_seed,
        reach_min_distance=DEFAULT_REACH_MIN_DISTANCE,
        reach_max_distance=DEFAULT_REACH_MAX_DISTANCE,
        reach_clear_radius=DEFAULT_REACH_CLEAR_RADIUS,
        kitchen_split="target",
    )
    return ObstacleReachBlockingRoboCasaEnv(
        inner,
        reach_task_seed=scene_seed,
        reach_min_distance=DEFAULT_REACH_MIN_DISTANCE,
        reach_max_distance=DEFAULT_REACH_MAX_DISTANCE,
        reach_clear_radius=DEFAULT_REACH_CLEAR_RADIUS,
        kitchen_split="target",
        obstacle_classes=DEFAULT_OBSTACLE_CLASSES,
        obstacle_clearance_m=DEFAULT_OBSTACLE_CLEARANCE_M,
        obstacle_lateral_jitter_std=DEFAULT_OBSTACLE_JITTER_STD,
        pos_tol=DEFAULT_BLOCKING_POS_TOL,
        max_steps=DEFAULT_BLOCKING_MAX_STEPS,
    )


def _save_obs_images(obs: dict, scene_seed: int, episode_idx: int, mode: str, viz_dir: Path) -> None:
    from PIL import Image  # lazy import — only needed when viz is enabled

    out_dir = viz_dir / f"sc{scene_seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam_idx, cam_name in enumerate(DEFAULT_CAMERA_NAMES):
        img_key = f"img_{cam_idx}"
        if img_key not in obs:
            continue
        img = np.asarray(obs[img_key], dtype=np.uint8)
        out_path = out_dir / f"ep{episode_idx:02d}_{mode}_cam{cam_idx}_{cam_name}.png"
        Image.fromarray(img).save(out_path, optimize=False)


def _run_one_scene(scene_seed: int, episodes: int, viz_dir: Path | None) -> tuple[dict, dict, list[tuple[int, str]]]:
    """Return ``(episodes_dict, base_pose_dict, failed_episodes)``.

    ``episodes_dict`` keys are episode ints stringified; each value is::

        {
            "init": [x, y, z],
            "goal": [x, y, z],
            "obstacle_fixture_name": str,
            "obstacle_class": str,
            "obstacle_world_pos": [x, y, z],
            "obstacle_y_half_extent_m": float,
        }

    Episodes whose sampling fails (rotation-validator rejection or
    sampler-budget exhaustion) are recorded in ``failed_episodes`` as
    ``(episode_idx, error_message)`` tuples and OMITTED from
    ``episodes_dict``. The partial JSON still gets written so the rest
    of the scene's data is preserved.
    """
    env = _build_env(scene_seed)
    scene_out: dict = {}
    failed_episodes: list[tuple[int, str]] = []
    try:
        for ep in range(episodes):
            t0 = time.perf_counter()
            init_kwargs = dict(mode="init", seed=scene_seed, reach_task_seed=scene_seed)
            if ep == 0:
                init_kwargs["reset_episode_counter"] = True
            try:
                obs_init, _ = env.reset(**init_kwargs)
                init_world = np.asarray(obs_init["reach_start_world"], dtype=np.float64)
                if viz_dir is not None:
                    _save_obs_images(obs_init, scene_seed, ep, "init", viz_dir)

                obs_goal, _ = env.reset(mode="goal", seed=scene_seed, reach_task_seed=scene_seed)
                goal_world = np.asarray(obs_goal["reach_goal_world"], dtype=np.float64)
                if viz_dir is not None:
                    _save_obs_images(obs_goal, scene_seed, ep, "goal", viz_dir)
            except RuntimeError as exc:
                # Per-episode sampler/validator failure: log and move on
                # to the next episode. The partial JSON still gets all
                # the successful episodes for this scene.
                msg = str(exc).splitlines()[0]
                failed_episodes.append((ep, msg))
                print(
                    f"  sc{scene_seed:03d} ep{ep}: SKIPPED ({type(exc).__name__}: {msg})",
                    flush=True,
                )
                continue

            scene_out[str(ep)] = {
                "init": init_world.tolist(),
                "goal": goal_world.tolist(),
                "obstacle_fixture_name": str(obs_init["obstacle_fixture_name"]),
                "obstacle_class": str(obs_init["obstacle_class"]),
                "obstacle_world_pos": np.asarray(
                    obs_init["obstacle_world_pos"], dtype=np.float64
                ).tolist(),
                "obstacle_y_half_extent_m": float(obs_init["obstacle_y_half_extent_m"]),
            }
            print(
                f"  sc{scene_seed:03d} ep{ep}: init={init_world.tolist()} "
                f"goal={goal_world.tolist()} obstacle={scene_out[str(ep)]['obstacle_fixture_name']!r} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
        cached = env._env._reach_base_pose_cache.get(scene_seed)
        if cached is None:
            raise RuntimeError(
                f"sc{scene_seed:03d}: env did not populate _reach_base_pose_cache "
                f"— the test harness will not be able to reproduce this scene's pose."
            )
        base_pose = {
            "init_robot_base_pos": list(cached[0]),
            "init_robot_base_ori": list(cached[1]),
        }
    finally:
        try:
            env.close()
        except Exception:
            pass
    return scene_out, base_pose, failed_episodes


def _write_partial(
    scene_seed: int,
    scene_data: dict,
    base_pose: dict,
    partials_dir: Path,
    failed_episodes: list[tuple[int, str]] | None = None,
) -> Path:
    partials_dir.mkdir(parents=True, exist_ok=True)
    out_path = partials_dir / f"sc{scene_seed:03d}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene_seed": scene_seed,
        "kitchen_split": "target",
        "reach_min_distance": DEFAULT_REACH_MIN_DISTANCE,
        "reach_max_distance": DEFAULT_REACH_MAX_DISTANCE,
        "reach_clear_radius": DEFAULT_REACH_CLEAR_RADIUS,
        "obstacle_clearance_m": DEFAULT_OBSTACLE_CLEARANCE_M,
        "obstacle_lateral_jitter_std": DEFAULT_OBSTACLE_JITTER_STD,
        "obstacle_classes": list(DEFAULT_OBSTACLE_CLASSES),
        "episodes": scene_data,
        "base_pose": base_pose,
    }
    if failed_episodes:
        payload["failed_episodes"] = [
            {"episode_idx": int(ep), "error": str(msg)} for ep, msg in failed_episodes
        ]
    tmp = out_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(out_path)
    return out_path


def _merge_partials(partials_dir: Path, output_path: Path) -> None:
    if not partials_dir.exists():
        raise SystemExit(f"--partials-dir not found: {partials_dir}")

    scenes: dict = {}
    knob_values: dict = {}
    files = sorted(partials_dir.glob("sc*.json"))
    if not files:
        raise SystemExit(f"no sc*.json partials in {partials_dir}")

    for f in files:
        with f.open("r") as fh:
            payload = json.load(fh)
        scene_seed = payload.get("scene_seed")
        if scene_seed is None:
            print(f"[merge] WARN: {f} missing scene_seed; skipping")
            continue
        for key in (
            "kitchen_split",
            "reach_min_distance",
            "reach_max_distance",
            "reach_clear_radius",
            "obstacle_clearance_m",
            "obstacle_lateral_jitter_std",
            "obstacle_classes",
        ):
            v = payload.get(key)
            if v is None:
                continue
            if key not in knob_values:
                knob_values[key] = v
            else:
                assert knob_values[key] == v, f"{f} {key} mismatch: {v} vs {knob_values[key]}"
        scenes[str(scene_seed)] = {
            "episodes": payload["episodes"],
            "base_pose": payload.get("base_pose"),
        }

    out = {"schema_version": SCHEMA_VERSION}
    out.update(knob_values)
    out["scenes"] = scenes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    tmp.replace(output_path)
    print(f"[merge] wrote {output_path} ({len(scenes)} scenes from {len(files)} partials)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Final merged golden file path (used by --merge).",
    )
    parser.add_argument("--partials-dir", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    partials_dir = Path(args.partials_dir) if args.partials_dir else _default_partials_dir()
    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    output_path = Path(args.output) if args.output else _default_output_path()

    if args.merge:
        _merge_partials(partials_dir, output_path)
        return

    seeds = tuple(args.seeds) if args.seeds is not None else DEFAULT_TARGET_SCENE_SEEDS
    episodes = int(args.episodes)

    print(
        f"[worker pid={os.getpid()}] obstacle-reach golden: target seeds={list(seeds)} "
        f"(episodes={episodes}, viz={'on' if viz_dir else 'off'})",
        flush=True,
    )

    t_total = time.perf_counter()
    failures: list[tuple[int, str]] = []
    for scene_seed in seeds:
        t_scene = time.perf_counter()
        try:
            scene_data, base_pose, failed_eps = _run_one_scene(scene_seed, episodes, viz_dir)
            out_path = _write_partial(scene_seed, scene_data, base_pose, partials_dir, failed_eps)
            print(
                f"[worker pid={os.getpid()}] wrote {out_path} "
                f"({(time.perf_counter() - t_scene) / 60:.2f} min for this scene; "
                f"{len(scene_data)}/{episodes} eps, {len(failed_eps)} per-ep failures)",
                flush=True,
            )
            if failed_eps:
                for ep, msg in failed_eps:
                    print(f"    ep{ep}: {msg}", flush=True)
        except Exception as exc:
            failures.append((scene_seed, f"{type(exc).__name__}: {exc}"))
            print(
                f"[worker pid={os.getpid()}] FAILED sc{scene_seed:03d}: {failures[-1][1]}",
                flush=True,
            )
    print(
        f"[worker pid={os.getpid()}] DONE seeds={list(seeds)} "
        f"({(time.perf_counter() - t_total) / 60:.2f} min total, "
        f"{len(failures)} failures)",
        flush=True,
    )
    if failures:
        for seed, msg in failures:
            print(f"  - sc{seed:03d}: {msg}", flush=True)


if __name__ == "__main__":
    main()
