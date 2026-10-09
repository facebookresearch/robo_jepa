# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate the OBJECT_REACH golden capture files.

Single-shot replacement for the earlier per-kitchen-worker + consolidator
scripts. Loops over every object in
:data:`DEFAULT_OBJECTS` × the 9 working target kitchens, runs 3 episodes
× 2 modes per (object, kitchen), captures the exact 7 fields
``test_golden_regression`` reads, and writes one
``object_reach_golden_<object>.json`` per object.

When to regenerate:
* Env semantics changed deliberately (grasp sequence, per-object
  heuristics, rotation feedback in carry, step-time orientation hold).
* New per-target-kitchen heuristic added (registry-side, per object).
* Adding a new object to :data:`OBJECT_REACH_REGISTRY`.
* Otherwise: do NOT regenerate — let the existing ``test_golden_regression``
  catch unintended drift.

Usage::

    cd <repo_root>
    MUJOCO_GL=egl PYTHONPATH="$PWD:$PWD/scripts/deployment:$PYTHONPATH" \\
        python \\
        scripts/deployment/robots/robocasa/generate_object_reach_golden.py

Optional kwargs (override via env var):
    OBJECT_REACH_GOLDEN_KITCHENS="1 2 3"
        restrict to a subset of kitchens (default = all 9 working target
        kitchens). When set with len(kitchens) != 9, the regenerator
        merges new captures into the existing per-object golden so the
        rest of the file stays consolidated.
    OBJECT_REACH_GOLDEN_OBJECTS="can"
        restrict to a subset of objects (default = "banana can"). Useful
        when adding a new object and you only want to regenerate that
        one.
    OBJECT_REACH_GOLDEN_OUT="/tmp/golden.json"
        override the output path. Only honoured when exactly one object
        is being regenerated (multi-object regen always derives one
        ``object_reach_golden_<name>.json`` per object).

Wall clock: ~9 kitchens × ~9 min/kitchen ≈ 1 h sequential per object on
a single GPU node. With the default 2 objects, full
regeneration is ~2 h.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Headless rendering must be set before importing mujoco/robosuite.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from robots.robocasa.blocking_robot_env import (  # noqa: E402
    _OBJECT_REACH_BODY_NAME,
    _OBJECT_REACH_INNER_TASK_NAME,
    ObjectReachBlockingRoboCasaEnv,
)
from robots.robocasa.robot_env import RoboCasaEnv  # noqa: E402


_THIS_DIR = Path(__file__).resolve().parent

# Match the constants in test_object_reach_env.py exactly. Any drift
# here vs. there breaks the test's per-(kitchen, episode, mode)
# expectations.
TARGET_KITCHENS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 9, 10)
N_EPISODES: int = 3
SCENE_SEED: int = 0
ROBOT: str = "PandaOmron"

# Default objects to regenerate when invoked with no overrides. Each
# object emits its own golden file (``object_reach_golden_<name>.json``).
# Override via ``OBJECT_REACH_GOLDEN_OBJECTS="banana"`` (or ``"banana can"``,
# etc.) for partial / single-object regen.
DEFAULT_OBJECTS: tuple[str, ...] = ("banana", "can")


def _golden_path_for(object_name: str) -> Path:
    """Per-object golden file path. Mirrors the loader in
    :func:`test_object_reach_env.test_golden_regression`."""
    return _THIS_DIR / f"object_reach_golden_{object_name}.json"


def _xyzw_to_wxyz(q) -> list[float]:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _eef_quat_wxyz(inner) -> list[float]:
    if hasattr(inner, "_get_observations"):
        obs_dict = inner._get_observations(force_update=True)
    else:
        obs_dict = inner._get_observation()
    return _xyzw_to_wxyz(obs_dict["robot0_eef_quat"])


def _object_quat_wxyz(inner) -> list[float]:
    body_id_map = getattr(inner, "obj_body_id", None) or {}
    if _OBJECT_REACH_BODY_NAME not in body_id_map:
        return [float("nan")] * 4
    bid = body_id_map[_OBJECT_REACH_BODY_NAME]
    # robosuite returns body_xquat as wxyz already (mujoco convention).
    return list(map(float, inner.sim.data.body_xquat[bid]))


def _base_pose(sim) -> tuple[list[float], list[float]]:
    """Return ``(pos, wxyz)`` for the canonical robot base body."""
    for body_name in ("mobilebase0_base", "robot0_base", "base"):
        try:
            bid = sim.model.body_name2id(body_name)
        except Exception:
            continue
        pos = list(map(float, sim.data.body_xpos[bid]))
        quat = list(map(float, sim.data.body_xquat[bid]))
        return pos, quat
    raise RuntimeError("could not resolve robot base body")


def _capture_after_reset(env, obs: dict) -> dict:
    """Capture the 7 golden fields after a single reset call."""
    inner = env._env._env  # robosuite Kitchen env
    eef_pos = list(map(float, env._get_eef_pos()))
    eef_quat_wxyz = _eef_quat_wxyz(inner)
    if "object_reach_obj_pos_world" in obs:
        obj_pos = list(map(float, obs["object_reach_obj_pos_world"]))
    else:
        try:
            obj_pos = list(map(float, env._get_object_world_pos()))
        except Exception:
            obj_pos = [float("nan")] * 3
    obj_quat_wxyz = _object_quat_wxyz(inner)
    base_pos, base_quat_wxyz = _base_pose(inner.sim)
    return {
        "init_eef_pos": eef_pos,
        "init_eef_quat_wxyz": eef_quat_wxyz,
        "object_pos_world": obj_pos,
        "object_quat_wxyz": obj_quat_wxyz,
        "robot_base_pos": base_pos,
        "robot_base_quat_wxyz": base_quat_wxyz,
        "grasp_failed": bool(obs.get("object_reach_grasp_failed", True)),
    }


def _run_three_episodes(kitchen_id: int, object_name: str) -> list[dict]:
    """Build env, run 3 (init + goal) episodes, close env. Returns a
    list of ``{"init": ..., "goal": ...}`` dicts."""
    inner_env = RoboCasaEnv(
        task_name=_OBJECT_REACH_INNER_TASK_NAME,
        camera_names=["robot0_agentview_left"],
        camera_width=128,
        camera_height=128,
        seed=SCENE_SEED,
        robot=ROBOT,
        init_robot_to_home=True,
        reach_task=False,
        layout_and_style_ids=[(kitchen_id, kitchen_id)],
    )
    env = ObjectReachBlockingRoboCasaEnv(
        inner_env,
        object_reach_object_name=object_name,
        object_reach_task_seed=0,
        object_reach_max_distance=0.15,
        object_reach_clear_radius=1.0,
        object_reach_max_retries=3,
    )
    episodes: list[dict] = []
    try:
        for ep in range(N_EPISODES):
            obs_init, _ = env.reset(mode="init", object_reach_task_seed=ep)
            init_cap = _capture_after_reset(env, obs_init)
            obs_goal, _ = env.reset(mode="goal", object_reach_task_seed=ep)
            goal_cap = _capture_after_reset(env, obs_goal)
            episodes.append({"init": init_cap, "goal": goal_cap})
    finally:
        try:
            env.close()
        except Exception:
            pass
    return episodes


def _regenerate_one_object(
    object_name: str,
    kitchens: tuple[int, ...],
    out_path: Path,
) -> None:
    print(
        f"[generate_object_reach_golden] === object={object_name} === "
        f"writing -> {out_path}",
        flush=True,
    )
    print(
        f"[generate_object_reach_golden] kitchens = {list(kitchens)} | "
        f"scene_seed={SCENE_SEED} | task_seeds={list(range(N_EPISODES))} | "
        f"object={object_name} robot={ROBOT}",
        flush=True,
    )

    payload = {
        "_meta": {
            "scene_seed": SCENE_SEED,
            "task_seeds": list(range(N_EPISODES)),
            "kitchen_ids": list(TARGET_KITCHENS),
            "object_name": object_name,
            "robot": ROBOT,
            "capture_fields": [
                "init_eef_pos",
                "init_eef_quat_wxyz",
                "object_pos_world",
                "object_quat_wxyz",
                "robot_base_pos",
                "robot_base_quat_wxyz",
                "grasp_failed",
            ],
        },
        "kitchens": {},
    }

    # If we're regenerating a subset, preserve the rest from the
    # existing file so the resulting JSON stays consolidated.
    if out_path.exists() and len(kitchens) != len(TARGET_KITCHENS):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            payload["kitchens"] = existing.get("kitchens", {})
        except Exception as e:
            print(
                f"[generate_object_reach_golden] WARNING: failed to read "
                f"existing golden {out_path} for partial-update merge: {e}; "
                f"starting from empty kitchens.",
                flush=True,
            )

    for kitchen_id in kitchens:
        t0 = time.perf_counter()
        print(
            f"[generate_object_reach_golden] === object={object_name} "
            f"kitchen {kitchen_id} === starting",
            flush=True,
        )
        try:
            episodes = _run_three_episodes(kitchen_id, object_name)
            payload["kitchens"][str(kitchen_id)] = {"episodes": episodes}
            grasp_oks = sum(
                not ep["init"]["grasp_failed"] and not ep["goal"]["grasp_failed"]
                for ep in episodes
            )
            print(
                f"[generate_object_reach_golden] object={object_name} "
                f"kitchen {kitchen_id} done in "
                f"{time.perf_counter() - t0:.1f}s — {grasp_oks}/{len(episodes)} "
                f"episodes with successful grasp in both modes",
                flush=True,
            )
        except Exception as e:
            print(
                f"[generate_object_reach_golden] object={object_name} "
                f"kitchen {kitchen_id} FAILED "
                f"after {time.perf_counter() - t0:.1f}s: {e!r}",
                flush=True,
            )
            raise

    # Atomic write so a partial failure doesn't leave a corrupt file.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(out_path)
    print(
        f"[generate_object_reach_golden] wrote {out_path} "
        f"({sum(len(v['episodes']) for v in payload['kitchens'].values())} "
        f"total episodes across {len(payload['kitchens'])} kitchens)",
        flush=True,
    )


def main() -> int:
    kitchens_env = os.environ.get("OBJECT_REACH_GOLDEN_KITCHENS", "")
    if kitchens_env.strip():
        kitchens = tuple(int(k) for k in kitchens_env.split())
    else:
        kitchens = TARGET_KITCHENS

    objects_env = os.environ.get("OBJECT_REACH_GOLDEN_OBJECTS", "")
    if objects_env.strip():
        objects = tuple(objects_env.split())
    else:
        objects = DEFAULT_OBJECTS

    # ``OBJECT_REACH_GOLDEN_OUT`` overrides the output path. Only honoured
    # when exactly one object is being regenerated (otherwise we'd be
    # writing multiple objects to the same file). For multi-object regen
    # we always derive ``object_reach_golden_<name>.json``.
    out_override = os.environ.get("OBJECT_REACH_GOLDEN_OUT", "").strip() or None
    if out_override is not None and len(objects) != 1:
        print(
            f"[generate_object_reach_golden] ERROR: OBJECT_REACH_GOLDEN_OUT "
            f"is set but OBJECT_REACH_GOLDEN_OBJECTS has {len(objects)} "
            f"entries ({list(objects)}); the override only makes sense for "
            f"a single object. Either unset OBJECT_REACH_GOLDEN_OUT or set "
            f"OBJECT_REACH_GOLDEN_OBJECTS to one name.",
            flush=True,
        )
        return 2

    for object_name in objects:
        out_path = (
            Path(out_override) if out_override is not None
            else _golden_path_for(object_name)
        )
        _regenerate_one_object(object_name, kitchens, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
