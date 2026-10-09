# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the (start, goal) golden file for the reach-task invariants tests.

Two modes:

1. **Per-seed worker** (default): for each ``scene_seed`` passed via
   ``--seeds``, run all 10 episodes and write a per-seed partial JSON
   to ``--partials-dir/sc<XXX>.json``. Also, when ``--viz-dir`` is set,
   saves the two camera views (side + wrist) for the start and goal
   frames as PNGs so the operator can sanity-check the layouts.

2. **Merge** (``--merge``): scan ``--partials-dir`` for every
   ``sc<XXX>.json`` partial and merge them into the final golden file
   at ``--output`` (``scripts/deployment/robots/robocasa/reach_invariants_golden.json``).

A separate driver script (``launch_reach_invariants_golden.sh``) fans
the per-seed work out over 20 concurrent ``xargs`` workers on a single
H200 node so the full sweep finishes in ~5-10 min instead of ~80 min.

Usage
-----
Worker (1 seed):
    python -u scripts/deployment/robots/robocasa/generate_reach_invariants_golden.py \\
        --seeds 41 \\
        --partials-dir scripts/deployment/robots/robocasa/reach_invariants_golden_partials \\
        --viz-dir scripts/deployment/robots/robocasa/reach_invariants_golden_viz

Merge:
    python -u scripts/deployment/robots/robocasa/generate_reach_invariants_golden.py --merge \\
        --partials-dir scripts/deployment/robots/robocasa/reach_invariants_golden_partials \\
        --output scripts/deployment/robots/robocasa/reach_invariants_golden.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[4])  # <repo_root>/scripts/deployment/robots/robocasa/<this>
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
ROBO_JEPA_SCRIPTS = os.path.join(REPO_ROOT, "scripts", "deployment")
if ROBO_JEPA_SCRIPTS not in sys.path:
    sys.path.insert(0, ROBO_JEPA_SCRIPTS)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# ── Production eval defaults (the env-server flags of the reach planning sweeps)
DEFAULT_TASK_NAME = "CloseStandMixerHead"
DEFAULT_CAMERA_NAMES = ("robot0_agentview_left", "robot0_eye_in_hand")
DEFAULT_CAMERA_WIDTH = 320
DEFAULT_CAMERA_HEIGHT = 240
DEFAULT_ROBOT = "PandaOmron"
DEFAULT_CHUNK_SIZE = 4
DEFAULT_WM_EMBODIMENT = "franka_omron_state_delta"
DEFAULT_REACH_MIN_DISTANCE = 0.25
DEFAULT_REACH_MAX_DISTANCE = 0.35
DEFAULT_REACH_CLEAR_RADIUS = 1.0
DEFAULT_BLOCKING_POS_TOL = 0.001
DEFAULT_BLOCKING_MAX_STEPS = 200

# 49-scene pretraining set: {0..49} \ {6} (matches INCLUDED_SEEDS in
# the pretrain planner script). The target set is exactly the 10
# held-out target kitchens, indexed by ``seed=0..9`` (each maps to a
# distinct ``(layout, style)`` pair via ``seed % 10`` inside
# ``RoboCasaEnv._create_env``).
DEFAULT_PRETRAIN_SCENE_SEEDS = tuple(s for s in range(50) if s != 6)
DEFAULT_TARGET_SCENE_SEEDS = tuple(range(10))
DEFAULT_EPISODES = 10
SCHEMA_VERSION = 1


def _default_seeds_for_split(split: str) -> tuple:
    if split == "pretrain":
        return DEFAULT_PRETRAIN_SCENE_SEEDS
    if split == "target":
        return DEFAULT_TARGET_SCENE_SEEDS
    raise ValueError(f"unknown kitchen split: {split!r}")


def _default_golden_path(split: str) -> Path:
    # Pretrain keeps the legacy ``reach_invariants_golden.json`` filename
    # (already-committed). Target writes alongside as ``...golden_target.json``.
    if split == "pretrain":
        return Path(REPO_ROOT) / "tests" / "data" / "reach_invariants_golden.json"
    return Path(REPO_ROOT) / "tests" / "data" / f"reach_invariants_golden_{split}.json"


def _default_partials_dir(split: str) -> Path:
    if split == "pretrain":
        return Path(REPO_ROOT) / "tests" / "data" / "reach_invariants_golden_partials"
    return Path(REPO_ROOT) / "tests" / "data" / f"reach_invariants_golden_partials_{split}"


def _default_viz_dir(split: str) -> Path:
    if split == "pretrain":
        return Path(REPO_ROOT) / "tests" / "data" / "reach_invariants_golden_viz"
    return Path(REPO_ROOT) / "tests" / "data" / f"reach_invariants_golden_viz_{split}"


def _build_env(scene_seed: int, kitchen_split: str):
    """Construct a :class:`ReachBlockingRoboCasaEnv` matching the eval setup."""
    from robots.robocasa.blocking_robot_env import ReachBlockingRoboCasaEnv
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
        kitchen_split=kitchen_split,
    )
    return ReachBlockingRoboCasaEnv(
        inner,
        reach_task_seed=scene_seed,
        reach_min_distance=DEFAULT_REACH_MIN_DISTANCE,
        reach_max_distance=DEFAULT_REACH_MAX_DISTANCE,
        reach_clear_radius=DEFAULT_REACH_CLEAR_RADIUS,
        kitchen_split=kitchen_split,
        pos_tol=DEFAULT_BLOCKING_POS_TOL,
        max_steps=DEFAULT_BLOCKING_MAX_STEPS,
    )


def _save_obs_images(obs: dict, scene_seed: int, episode_idx: int, mode: str, viz_dir: Path) -> None:
    """Save every ``img_<k>`` in ``obs`` as a PNG under ``viz_dir/sc<seed>/``."""
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


def _run_one_scene(scene_seed: int, episodes: int, viz_dir: Path | None, kitchen_split: str) -> tuple[dict, dict]:
    """Return ``(episodes_dict, base_pose_dict)``.

    ``episodes_dict`` is ``{ "0": {"init": [x,y,z], "goal": [x,y,z]}, ..., "{N-1}": ... }``.
    ``base_pose_dict`` is ``{"init_robot_base_pos": [...], "init_robot_base_ori": [...]}``
    (the natural first-reset pose for this seed, captured into the
    env's ``_reach_base_pose_cache``). The test pre-injects this back
    so it uses the same base pose as the generator (the natural pose
    is not bit-deterministic across processes due to upstream Python
    hash-randomization affecting fixture-dict iteration order, but it
    IS stable within a process — our wrapper's cache pins it after
    the first reset).
    """
    env = _build_env(scene_seed, kitchen_split=kitchen_split)
    scene_out: dict = {}
    try:
        for ep in range(episodes):
            t0 = time.perf_counter()
            init_kwargs = dict(mode="init", seed=scene_seed, reach_task_seed=scene_seed)
            if ep == 0:
                # Mimic a fresh process: zero the per-seed counter so the
                # first init lands on episode 0 deterministically.
                init_kwargs["reset_episode_counter"] = True
            obs_init, _ = env.reset(**init_kwargs)
            init_world = np.asarray(obs_init["reach_start_world"], dtype=np.float64)
            if viz_dir is not None:
                _save_obs_images(obs_init, scene_seed, ep, "init", viz_dir)

            obs_goal, _ = env.reset(mode="goal", seed=scene_seed, reach_task_seed=scene_seed)
            goal_world = np.asarray(obs_goal["reach_goal_world"], dtype=np.float64)
            if viz_dir is not None:
                _save_obs_images(obs_goal, scene_seed, ep, "goal", viz_dir)

            scene_out[str(ep)] = {
                "init": init_world.tolist(),
                "goal": goal_world.tolist(),
            }
            print(
                f"  sc{scene_seed:03d} ep{ep}: init={init_world.tolist()} "
                f"goal={goal_world.tolist()} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
        # Snapshot the cached base pose AFTER all episodes so we capture
        # the same value all resets used.
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
    return scene_out, base_pose


def _write_partial(
    scene_seed: int,
    scene_data: dict,
    base_pose: dict,
    partials_dir: Path,
    kitchen_split: str,
) -> Path:
    """Write the per-seed partial JSON (one file per worker invocation)."""
    partials_dir.mkdir(parents=True, exist_ok=True)
    out_path = partials_dir / f"sc{scene_seed:03d}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene_seed": scene_seed,
        "kitchen_split": kitchen_split,
        "reach_min_distance": DEFAULT_REACH_MIN_DISTANCE,
        "reach_max_distance": DEFAULT_REACH_MAX_DISTANCE,
        "reach_clear_radius": DEFAULT_REACH_CLEAR_RADIUS,
        "episodes": scene_data,
        "base_pose": base_pose,
    }
    tmp = out_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(out_path)  # atomic publish
    return out_path


def _merge_partials(partials_dir: Path, output_path: Path) -> None:
    """Read every ``sc*.json`` partial under ``partials_dir`` and write the golden."""
    if not partials_dir.exists():
        raise SystemExit(f"--partials-dir not found: {partials_dir}")

    scenes: dict = {}
    kitchen_split: str | None = None
    reach_min_distance: float | None = None
    reach_max_distance: float | None = None
    reach_clear_radius: float | None = None
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
        # Cross-check eval knobs are consistent across partials.
        if kitchen_split is None:
            kitchen_split = payload.get("kitchen_split", "pretrain")
        if reach_min_distance is None:
            reach_min_distance = payload.get("reach_min_distance", 0.0)
        if reach_max_distance is None:
            reach_max_distance = payload["reach_max_distance"]
        if reach_clear_radius is None:
            reach_clear_radius = payload["reach_clear_radius"]
        assert payload.get("kitchen_split", "pretrain") == kitchen_split, (
            f"{f} kitchen_split mismatch: {payload.get('kitchen_split', 'pretrain')} vs {kitchen_split}"
        )
        assert payload.get("reach_min_distance", 0.0) == reach_min_distance, (
            f"{f} reach_min_distance mismatch: {payload.get('reach_min_distance', 0.0)} vs {reach_min_distance}"
        )
        assert payload["reach_max_distance"] == reach_max_distance, (
            f"{f} reach_max_distance mismatch: {payload['reach_max_distance']} vs {reach_max_distance}"
        )
        assert payload["reach_clear_radius"] == reach_clear_radius, (
            f"{f} reach_clear_radius mismatch: {payload['reach_clear_radius']} vs {reach_clear_radius}"
        )
        scenes[str(scene_seed)] = {
            "episodes": payload["episodes"],
            "base_pose": payload.get("base_pose"),
        }

    out = {
        "schema_version": SCHEMA_VERSION,
        "kitchen_split": kitchen_split,
        "reach_min_distance": reach_min_distance,
        "reach_max_distance": reach_max_distance,
        "reach_clear_radius": reach_clear_radius,
        "scenes": scenes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    tmp.replace(output_path)
    print(f"[merge] wrote {output_path} ({len(scenes)} scenes from {len(files)} partials)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=("pretrain", "target"),
        help=(
            "Which robocasa kitchen pool to draw from. ``pretrain`` "
            "(default) is the ~2,500-scene pretraining pool; ``target`` "
            "is the 10 held-out target kitchens (1,1)..(10,10) indexed "
            "by ``seed % 10``. Default seed set, output, partials, and "
            "viz dirs all derive from this flag unless overridden."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Final merged golden file path (used by --merge). Default: "
            "``scripts/deployment/robots/robocasa/reach_invariants_golden.json`` for pretrain, "
            "``scripts/deployment/robots/robocasa/reach_invariants_golden_<split>.json`` otherwise."
        ),
    )
    parser.add_argument(
        "--partials-dir",
        type=str,
        default=None,
        help=("Per-seed partial JSON directory (read by --merge; written by workers). Default derives from --split."),
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help=(
            "If set, save the two camera views (side + wrist) for the start "
            "and goal frames of every (seed, episode) as PNGs under "
            "<viz-dir>/sc<XXX>/."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Scene seeds for this worker to process (writes one partial "
            "JSON per seed). Default depends on --split: pretrain uses "
            "{0..49}\\{6} (49 seeds); target uses 0..9 (one per target "
            "kitchen)."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Episodes per scene (default: 10).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "Skip env construction; instead merge every sc*.json under "
            "--partials-dir into a single --output golden file."
        ),
    )
    args = parser.parse_args()

    split = args.split
    partials_dir = Path(args.partials_dir) if args.partials_dir else _default_partials_dir(split)
    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    output_path = Path(args.output) if args.output else _default_golden_path(split)

    if args.merge:
        _merge_partials(partials_dir, output_path)
        return

    seeds = tuple(args.seeds) if args.seeds is not None else _default_seeds_for_split(split)
    episodes = int(args.episodes)

    print(
        f"[worker pid={os.getpid()}] split={split} processing seeds={list(seeds)} "
        f"(episodes={episodes}, viz={'on' if viz_dir else 'off'})",
        flush=True,
    )

    t_total = time.perf_counter()
    for scene_seed in seeds:
        t_scene = time.perf_counter()
        scene_data, base_pose = _run_one_scene(scene_seed, episodes, viz_dir, kitchen_split=split)
        out_path = _write_partial(scene_seed, scene_data, base_pose, partials_dir, kitchen_split=split)
        print(
            f"[worker pid={os.getpid()}] wrote {out_path} "
            f"({(time.perf_counter() - t_scene) / 60:.2f} min for this scene)",
            flush=True,
        )
    print(
        f"[worker pid={os.getpid()}] DONE split={split} seeds={list(seeds)} "
        f"({(time.perf_counter() - t_total) / 60:.2f} min total)",
        flush=True,
    )


if __name__ == "__main__":
    main()
