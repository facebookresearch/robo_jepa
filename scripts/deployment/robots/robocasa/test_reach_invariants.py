# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Invariants for the RoboCasa reach task.

Covers four properties of the reach-task fix plus two end-effector
orientation invariants:

1. The mixer (and every other non-structural fixture within
   ``reach_clear_radius``) is hidden in **every** episode of every scene.
2. The Euclidean ``||goal_eef_world − init_eef_world||`` is bounded by
   ``reach_max_distance`` in every episode of every scene.
3. The mobile-base body of the robot is at exactly the same world pose
   across all episodes of a scene and identical between
   ``mode='init'`` and ``mode='goal'`` resets within an episode.
4. The per-(scene, episode) ``(init_eef_world, goal_eef_world)`` matches
   the pre-recorded golden file at
   ``scripts/deployment/robots/robocasa/reach_invariants_golden.json``.
5. The end-effector world-frame orientation (Euler XYZ angles) at the
   end of ``mode='init'`` is (approximately) identical to the one at
   the end of ``mode='goal'`` for every episode of every scene. The
   reach driver (:meth:`RoboCasaEnv._drive_eef_to_world_target`)
   actively feeds back the EE rotation error in the OSC closed loop
   AND the sampler (:meth:`RoboCasaEnv._sample_reach_targets`) does
   rotation-aware rejection sampling — any ``(start, goal)`` candidate
   whose actual driven EE rotations would differ by more than
   :data:`_REACH_ROT_INVARIANCE_TOL` is rejected and a new candidate is
   drawn from the same distance-annulus.
6. **Per-step orientation hold**: after every :meth:`RoboCasaEnv.step`
   in reach-task mode, the EE world rotation stays within
   :data:`_EEF_STEP_ROT_TOL_RAD` of the home rotation snapshotted at
   reset. Enforced by
   :meth:`RoboCasaEnv._reach_apply_orientation_hold`, which (a)
   sphere-clamps the commanded position delta to
   :data:`_REACH_STEP_MAX_POS_NORM` per OSC sub-step and
   (b) overrides the planner's rotation channel with a
   quaternion-based corrective toward the home EE quaternion.

Each test spins up its own :class:`ReachBlockingRoboCasaEnv` (no shared
fixtures) so the suite is xdist-safe. The recommended invocation is

    pytest -n 4 scripts/deployment/robots/robocasa/test_reach_invariants.py -v

— 4 workers × ~3 GiB host RAM ≈ 12 GiB peak, well within the per-node
budget; higher worker counts invite mujoco / EGL state contention and
risk OOM under the slurm cgroup.

The scene set is intentionally small (``_SMOKE_SEEDS = {0, 1, 35, 41}``)
so the full file runs in ~3-5 min on a single H200. ``41`` and ``35``
were the previously-broken scenes (their layout drifted on episodes
1..9 before Fix A); ``0`` is the canonical known-good scene; ``1`` is a
spot-check on a structurally different kitchen.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root + robocasa script path are on sys.path so the
# inner-env imports (``from action_components import ...``,
# ``from robots.robocasa.object_reach_kitchen import ...``) resolve.
REPO_ROOT = str(Path(__file__).resolve().parents[4])  # <repo_root>/scripts/deployment/robots/robocasa/<this>
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_ROBO_JEPA_SCRIPTS = os.path.join(REPO_ROOT, "scripts", "deployment")
if _ROBO_JEPA_SCRIPTS not in sys.path:
    sys.path.insert(0, _ROBO_JEPA_SCRIPTS)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# ── Eval-pipeline knobs (the env-server flags of the reach planning sweeps) ──
_TASK_NAME = "CloseStandMixerHead"
_CAMERA_NAMES = ("robot0_agentview_left", "robot0_eye_in_hand")
_CAMERA_WIDTH = 320
_CAMERA_HEIGHT = 240
_ROBOT = "PandaOmron"
_CHUNK_SIZE = 4
_WM_EMBODIMENT = "franka_omron_state_delta"
_REACH_MIN_DISTANCE = 0.25
_REACH_MAX_DISTANCE = 0.35
_REACH_CLEAR_RADIUS = 1.0
_BLOCKING_POS_TOL = 0.001
_BLOCKING_MAX_STEPS = 200

_EPISODES = 10
# Pretrain smoke set: 0 (canonical), 1 (spot-check), 35 / 41 (the
# scenes that exposed the layout-drift bug originally).
_PRETRAIN_SMOKE_SEEDS = (0, 1, 35, 41)
# Target smoke set: scenes 0 / 4 / 9 cover three structurally different
# target kitchens (one-wall, L-shape, wraparound) at minimal cost.
_TARGET_SMOKE_SEEDS = (0, 4, 9)
_TEST_CASES = [("pretrain", s) for s in _PRETRAIN_SMOKE_SEEDS] + [("target", s) for s in _TARGET_SMOKE_SEEDS]
_TEST_IDS = [f"{split}-sc{s:03d}" for split, s in _TEST_CASES]

_GOLDEN_DIR = Path(__file__).resolve().parent
_GOLDEN_PATH_BY_SPLIT = {
    # Goldens live next to this test file inside the robocasa folder
    # (moved here from tests/data/ so the goldens travel with the env
    # implementation that depends on them).
    "pretrain": _GOLDEN_DIR / "reach_invariants_golden.json",
    "target": _GOLDEN_DIR / "reach_invariants_golden_target.json",
}
# Robocasa fixture key has a varying ``_group_<N>`` suffix (e.g.
# ``stand_mixer_main_group_1`` for sc0, ``stand_mixer_main_group_2`` for
# another scene). We look the fixture up by its Python class name instead
# so the lookup is layout-stable.
_MIXER_FIXTURE_CLASS = "StandMixer"
# Tolerance for the golden-match assertion (m). The (start, goal) values
# come from a deterministic numpy RandomState seed, so they should match
# bit-exactly across runs of the same git revision. 1e-6 leaves headroom
# for float64 ↔ JSON ↔ float64 round-tripping.
_GOLDEN_TOL_M = 1e-6
# Tolerance for the base-pose constancy assertion (m). The cached
# `_ep_meta["init_robot_base_pos"]` injection pins the base target to
# the same value across all resets of a seed, but the subsequent
# settling-step + forward-kinematics inside `_reset_internal` can
# nudge `body_xpos` by ~1 nm. 1e-6 m (= 1 µm) is conservative — well
# above the observed noise floor (~1.3 nm) and well below any layout
# drift we care about (drift was ~1-2 m before the fix).
_BASE_POSE_TOL = 1e-6
# Tolerance for the EE-rotation-invariance assertion (rad). Both
# ``mode='init'`` and ``mode='goal'`` resets snapshot the same home arm
# pose (Fix A rebuilds the env and the wrapper resets the arm to
# ``init_qpos`` on every reset), then drive the EE position to the
# target through OSC. The reach driver actively feeds back the world-
# frame rotation error each sub-step (mirroring the ``return_home``
# loop in :meth:`RoboCasaEnv._run_expert_to_goal`) so the EE rotation
# at exit is within ``_GOAL_RETURN_HOME_ROT_TOL = 0.02 rad`` of the
# home snapshot for both modes. The init-vs-goal delta is therefore
# bounded above by ``2 * _GOAL_RETURN_HOME_ROT_TOL = 0.04 rad`` plus
# a small post-step settling margin (~0.02 rad in practice). 0.08 rad
# (~4.6°) is the smallest value that fits all observed cases with
# headroom and still catches the original no-feedback driver (which
# allowed ~0.6 rad / 35° drift) by an order of magnitude.
_EEF_ROT_TOL_RAD = 0.08
# Tolerance for the per-step EE-rotation-invariance assertion (rad).
# Used by ``test_eef_rotation_held_across_steps``: after each
# ``env.step(...)`` in reach-task mode, the EE world rotation must be
# within this tolerance of the home rotation snapshotted at reset.
# The active orientation-hold inside
# :meth:`RoboCasaEnv._reach_apply_orientation_hold` keeps the
# per-step drift well under 0.05 rad on most scenes; on a small
# fraction (e.g. target sc000 ep1, near a wrist near-singularity)
# the OSC controller cannot fully satisfy both pos + rot in the
# same sub-step and the residual creeps up to ~0.06 rad. 0.08 rad
# (~4.6°, matches the reset-time
# ``test_eef_rotation_matches_between_init_and_goal`` tolerance) is
# comfortable headroom over the worst observed step-time drift and
# would loudly catch any regression where the per-sub-step
# rotation feedback is bypassed (drift then ramps to ≥ 0.5 rad).
_EEF_STEP_ROT_TOL_RAD = 0.08


def _build_env(scene_seed: int, kitchen_split: str = "pretrain"):
    """Construct a :class:`ReachBlockingRoboCasaEnv` matching the eval setup."""
    from robots.robocasa.blocking_robot_env import ReachBlockingRoboCasaEnv
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
        kitchen_split=kitchen_split,
    )
    return ReachBlockingRoboCasaEnv(
        inner,
        reach_task_seed=scene_seed,
        reach_min_distance=_REACH_MIN_DISTANCE,
        reach_max_distance=_REACH_MAX_DISTANCE,
        reach_clear_radius=_REACH_CLEAR_RADIUS,
        kitchen_split=kitchen_split,
        pos_tol=_BLOCKING_POS_TOL,
        max_steps=_BLOCKING_MAX_STEPS,
    )


def _close_env(env) -> None:
    try:
        env.close()
    except Exception:
        pass


def _reset_init(env, scene_seed: int, *, fresh: bool = False):
    """Call ``env.reset(mode='init', seed=N, reach_task_seed=N, ...)``.

    ``fresh=True`` adds ``reset_episode_counter=True`` so the per-seed
    counter restarts at episode 0 — used for the first reset of a scene
    to mimic a fresh process state.
    """
    kwargs = dict(mode="init", seed=scene_seed, reach_task_seed=scene_seed)
    if fresh:
        kwargs["reset_episode_counter"] = True
    obs, _ = env.reset(**kwargs)
    return obs


def _reset_goal(env, scene_seed: int):
    obs, _ = env.reset(mode="goal", seed=scene_seed, reach_task_seed=scene_seed)
    return obs


def _mixer_geom_groups(env) -> list[tuple[str, list[int], float]]:
    """Return ``[(fixture_name, geom_ids, dist_to_eef_m), ...]`` for every
    StandMixer fixture currently in the model.

    A kitchen scene can contain multiple StandMixer fixtures (the
    structural one near the workspace plus decorative ones far away).
    ``_hide_nearby_appliances`` only hides instances within
    ``reach_clear_radius`` of the captured home EEF; instances outside
    that radius are intentionally left visible. We return all of them
    annotated with the distance from the **current EEF** so the caller
    can filter on the same radius.
    """
    inner = env._env._env
    fixtures = getattr(inner, "fixtures", None) or {}
    model = inner.sim.model
    # Use the current EEF position as the radius anchor — matches the
    # value :meth:`RoboCasaEnv._hide_nearby_appliances` was called with
    # (the env captures ``_reach_home_eef_pos = self._get_eef_pos()`` on
    # every reset and immediately calls ``_hide_nearby_appliances`` with
    # that value).
    eef_pos = env._env._get_eef_pos()
    out: list[tuple[str, list[int], float]] = []
    for fname, v in fixtures.items():
        if type(v).__name__ != _MIXER_FIXTURE_CLASS:
            continue
        root_body_name = getattr(v, "root_body", None)
        if root_body_name is None:
            continue
        try:
            root_bid = int(model.body_name2id(root_body_name))
        except Exception:
            continue
        root_pos = np.asarray(inner.sim.data.body_xpos[root_bid], dtype=np.float64)
        dist = float(np.linalg.norm(root_pos - eef_pos))
        geom_ids: list[int] = []
        for bid in env._env._body_subtree_ids(root_bid):
            n = int(model.body_geomnum[bid])
            start = int(model.body_geomadr[bid])
            geom_ids.extend(range(start, start + n))
        out.append((fname, geom_ids, dist))
    return out


def _base_pose_world(env) -> np.ndarray:
    """Return the mobile-base root body's world XYZ (3-vector, float64)."""
    inner = env._env._env
    for body_name in ("mobilebase0_base", "robot0_base", "base"):
        try:
            bid = int(inner.sim.model.body_name2id(body_name))
            return np.asarray(inner.sim.data.body_xpos[bid], dtype=np.float64).copy()
        except Exception:
            continue
    raise RuntimeError("Could not find mobile-base body on the inner env.")


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kitchen_split,scene_seed", _TEST_CASES, ids=_TEST_IDS)
def test_mixer_hidden_across_episodes(kitchen_split, scene_seed):
    """Every StandMixer inside ``reach_clear_radius`` of the EEF must be
    invisible + non-colliding in every episode.

    Scenes can spawn multiple StandMixer instances (workspace mixer +
    decorative far-away ones). Only the ones within ``reach_clear_radius``
    of the captured home EEF are hidden by ``_hide_nearby_appliances``;
    distant ones are intentionally left in the scene. This test asserts
    the in-radius ones are hidden on every reset.
    """
    env = _build_env(scene_seed, kitchen_split=kitchen_split)
    try:
        for ep in range(_EPISODES):
            for mode in ("init", "goal"):
                kw = dict(mode=mode, seed=scene_seed, reach_task_seed=scene_seed)
                if ep == 0 and mode == "init":
                    kw["reset_episode_counter"] = True
                env.reset(**kw)

                groups = _mixer_geom_groups(env)
                # Note: a scene may have ZERO StandMixer fixtures at all
                # (some target kitchens like (8,8) Rustic don't spawn one).
                # We only assert hiddenness for the in-radius set; absence
                # of any mixer is silently accepted because the reach
                # workspace is already clear of mixers in that case.
                in_radius = [(fname, gids, dist) for (fname, gids, dist) in groups if dist <= _REACH_CLEAR_RADIUS]

                model = env._env._env.sim.model
                for fname, gids, dist in in_radius:
                    alphas = np.asarray([model.geom_rgba[g, 3] for g in gids])
                    contype = np.asarray([model.geom_contype[g] for g in gids])
                    conaff = np.asarray([model.geom_conaffinity[g] for g in gids])
                    assert np.all(alphas == 0.0), (
                        f"{kitchen_split} sc{scene_seed:03d} ep{ep} mode={mode}: "
                        f"mixer {fname!r} ({dist:.3f} m from EEF) still visible "
                        f"(alphas={alphas.tolist()})"
                    )
                    assert np.all(contype == 0) and np.all(conaff == 0), (
                        f"{kitchen_split} sc{scene_seed:03d} ep{ep} mode={mode}: "
                        f"mixer {fname!r} ({dist:.3f} m from EEF) still collides "
                        f"(contype={contype.tolist()}, conaffinity={conaff.tolist()})"
                    )
    finally:
        _close_env(env)


@pytest.mark.parametrize("kitchen_split,scene_seed", _TEST_CASES, ids=_TEST_IDS)
def test_reach_distance_bounded(kitchen_split, scene_seed):
    """``reach_min_distance ≤ ||goal - start|| ≤ reach_max_distance`` for every episode.

    The env's ``_sample_reach_targets`` does rejection sampling to
    enforce this annulus directly, so the bound here matches the
    sampler's exact contract.
    """
    env = _build_env(scene_seed, kitchen_split=kitchen_split)
    try:
        for ep in range(_EPISODES):
            obs_init = _reset_init(env, scene_seed, fresh=(ep == 0))
            start_world = np.asarray(obs_init["reach_start_world"], dtype=np.float64)

            obs_goal = _reset_goal(env, scene_seed)
            goal_world = np.asarray(obs_goal["reach_goal_world"], dtype=np.float64)

            d = float(np.linalg.norm(goal_world - start_world))
            lo = _REACH_MIN_DISTANCE - 1e-6
            hi = _REACH_MAX_DISTANCE + 1e-6
            assert lo <= d <= hi, (
                f"{kitchen_split} sc{scene_seed:03d} ep{ep}: |goal - start| = {d:.4f} m "
                f"outside [{_REACH_MIN_DISTANCE:.4f}, {_REACH_MAX_DISTANCE:.4f}] m"
            )
    finally:
        _close_env(env)


@pytest.mark.parametrize("kitchen_split,scene_seed", _TEST_CASES, ids=_TEST_IDS)
def test_robot_base_constant_across_episodes_and_modes(kitchen_split, scene_seed):
    """Mobile-base world pose must be bit-identical across all resets."""
    env = _build_env(scene_seed, kitchen_split=kitchen_split)
    try:
        poses: list[np.ndarray] = []
        for ep in range(_EPISODES):
            _reset_init(env, scene_seed, fresh=(ep == 0))
            poses.append(_base_pose_world(env))
            _reset_goal(env, scene_seed)
            poses.append(_base_pose_world(env))
        assert len(poses) == 2 * _EPISODES
        ref = poses[0]
        for i, p in enumerate(poses[1:], start=1):
            delta = float(np.linalg.norm(p - ref))
            assert delta <= _BASE_POSE_TOL, (
                f"{kitchen_split} sc{scene_seed:03d}: base pose drifted by "
                f"{delta:.3e} m at reset index {i} "
                f"({'init' if i % 2 == 0 else 'goal'} of ep{i // 2}): "
                f"ref={ref.tolist()}, this={p.tolist()}"
            )
    finally:
        _close_env(env)


def _golden_for(kitchen_split: str) -> Path:
    return _GOLDEN_PATH_BY_SPLIT[kitchen_split]


@pytest.mark.parametrize("kitchen_split,scene_seed", _TEST_CASES, ids=_TEST_IDS)
def test_reach_targets_match_golden(kitchen_split, scene_seed):
    """For each (scene, episode), captured (start, goal) must match the golden file.

    Pre-injects the golden file's cached base pose into the env's
    ``_reach_base_pose_cache`` BEFORE the first reset so the test uses
    the **same** base pose as the generator. The natural first-reset
    pose isn't bit-deterministic across processes (PYTHONHASHSEED
    randomizes some fixture-dict iteration orders inside robocasa)
    but our wrapper's per-seed cache pins it for the rest of the
    process — so reading the generator's cached value and injecting
    it before the first reset makes the test reproducible.
    """
    golden_path = _golden_for(kitchen_split)
    if not golden_path.exists():
        pytest.skip(
            f"golden file not present for split={kitchen_split}: {golden_path}. "
            f"Generate it with `sbatch --export=ALL,KITCHEN_SPLIT={kitchen_split} "
            f"scripts/deployment/robots/robocasa/launch_reach_invariants_golden.sh`"
        )
    with golden_path.open("r") as f:
        data = json.load(f)
    scenes = data["scenes"]
    scene_key = str(scene_seed)
    if scene_key not in scenes:
        pytest.skip(
            f"{kitchen_split} sc{scene_seed:03d} not in golden file (available scenes: {sorted(scenes.keys())})"
        )
    scene_payload = scenes[scene_key]
    # Back-compat: older partials stored episodes at the top level; the
    # new schema nests them under ``episodes`` alongside ``base_pose``.
    if isinstance(scene_payload, dict) and "episodes" in scene_payload:
        golden_scene = scene_payload["episodes"]
        golden_base_pose = scene_payload.get("base_pose")
    else:
        golden_scene = scene_payload
        golden_base_pose = None
    assert data.get("reach_min_distance", 0.0) == pytest.approx(_REACH_MIN_DISTANCE)
    assert data.get("reach_max_distance") == pytest.approx(_REACH_MAX_DISTANCE)
    assert data.get("reach_clear_radius") == pytest.approx(_REACH_CLEAR_RADIUS)
    # Some legacy golden files predate the ``kitchen_split`` field; in
    # that case treat the absence as the legacy default ("pretrain")
    # so the test still cross-checks consistency post-upgrade.
    assert data.get("kitchen_split", "pretrain") == kitchen_split, (
        f"golden file {golden_path} declares kitchen_split="
        f"{data.get('kitchen_split', 'pretrain')!r} but the test was parameterised "
        f"with {kitchen_split!r}"
    )

    env = _build_env(scene_seed, kitchen_split=kitchen_split)
    try:
        # Pre-inject the golden's cached base pose so our first reset
        # uses the same pose as the generator. Without this, the
        # cross-process non-determinism in the natural first-reset
        # pose can shift our EEF positions by ~1.5 m relative to the
        # golden on layouts like sc035 / sc041.
        if golden_base_pose is not None:
            env._env._reach_base_pose_cache[scene_seed] = (
                list(golden_base_pose["init_robot_base_pos"]),
                list(golden_base_pose["init_robot_base_ori"]),
            )

        for ep in range(_EPISODES):
            ep_key = str(ep)
            if ep_key not in golden_scene:
                continue
            golden = golden_scene[ep_key]
            golden_init = np.asarray(golden["init"], dtype=np.float64)
            golden_goal = np.asarray(golden["goal"], dtype=np.float64)

            obs_init = _reset_init(env, scene_seed, fresh=(ep == 0))
            init_world = np.asarray(obs_init["reach_start_world"], dtype=np.float64)
            obs_goal = _reset_goal(env, scene_seed)
            goal_world = np.asarray(obs_goal["reach_goal_world"], dtype=np.float64)

            d_init = float(np.linalg.norm(init_world - golden_init))
            d_goal = float(np.linalg.norm(goal_world - golden_goal))
            assert d_init <= _GOLDEN_TOL_M, (
                f"{kitchen_split} sc{scene_seed:03d} ep{ep}: init drifted from golden by "
                f"{d_init:.3e} m: got {init_world.tolist()}, "
                f"expected {golden_init.tolist()}"
            )
            assert d_goal <= _GOLDEN_TOL_M, (
                f"{kitchen_split} sc{scene_seed:03d} ep{ep}: goal drifted from golden by "
                f"{d_goal:.3e} m: got {goal_world.tolist()}, "
                f"expected {golden_goal.tolist()}"
            )
    finally:
        _close_env(env)


def _eef_euler_world(obs) -> np.ndarray:
    """Return the EE world-frame Euler XYZ (rad) from the reset obs dict.

    ``RoboCasaEnv._build_obs`` packs the proprio vector as
    ``[eef_pos(3), eef_euler_xyz_world(3), gripper(1)]`` so slots
    ``[3:6]`` are the EE world-frame Euler angles we want.
    """
    pose = np.asarray(obs["pose"], dtype=np.float64)
    assert pose.shape == (7,), f"unexpected pose shape: {pose.shape}"
    return pose[3:6].copy()


@pytest.mark.parametrize("kitchen_split,scene_seed", _TEST_CASES, ids=_TEST_IDS)
def test_eef_rotation_matches_between_init_and_goal(kitchen_split, scene_seed):
    """EE world rotation must be ~identical at the end of mode='init' and mode='goal'.

    Two complementary mechanisms enforce this in
    :mod:`robots.robocasa.robot_env`:

    1. The reach driver
       (:meth:`RoboCasaEnv._drive_eef_to_world_target`) snapshots the
       home arm rotation at entry and feeds back the world-frame
       rotation error each OSC sub-step, mirroring the active
       rotation control used by the ``return_home`` loop in
       :meth:`RoboCasaEnv._run_expert_to_goal`. Combined with Fix A
       (the env is fully rebuilt each reset, so both ``init`` and
       ``goal`` resets start from the same arm ``init_qpos``), the
       EE rotation at exit is within
       ``_GOAL_RETURN_HOME_ROT_TOL = 0.02`` rad of the home pose for
       most scenes.
    2. On a small fraction of kitchen / episode combinations the OSC
       controller cannot simultaneously satisfy position and rotation
       constraints (kinematic singularities / joint-limit
       configurations push the wrist by 0.2–0.3 rad). To eliminate
       those, :meth:`RoboCasaEnv._sample_reach_targets` performs
       rotation-aware rejection sampling: each ``(start, goal)``
       candidate that passes the distance band is also verified by
       :meth:`RoboCasaEnv._validate_rotation_for_targets`, which
       actually drives the EE to both points and rejects the pair if
       the geodesic SO(3) drift exceeds
       :data:`_REACH_ROT_INVARIANCE_TOL = 0.05` rad. Rejected pairs
       are silently replaced with a new draw from the same annulus.

    Test tolerance (``_EEF_ROT_TOL_RAD = 0.08``) gives a ~0.03 rad
    margin over the sampler-side threshold to absorb numerical noise
    in the per-call settling step.
    """
    from scipy.spatial.transform import Rotation

    env = _build_env(scene_seed, kitchen_split=kitchen_split)
    try:
        for ep in range(_EPISODES):
            obs_init = _reset_init(env, scene_seed, fresh=(ep == 0))
            init_euler = _eef_euler_world(obs_init)

            obs_goal = _reset_goal(env, scene_seed)
            goal_euler = _eef_euler_world(obs_goal)

            r_init = Rotation.from_euler("xyz", init_euler, degrees=False)
            r_goal = Rotation.from_euler("xyz", goal_euler, degrees=False)
            # Geodesic distance on SO(3): angle of (R_init * R_goal^{-1}).
            delta = float(np.linalg.norm((r_init * r_goal.inv()).as_rotvec()))
            assert delta <= _EEF_ROT_TOL_RAD, (
                f"{kitchen_split} sc{scene_seed:03d} ep{ep}: EE world rotation "
                f"between init and goal differs by {delta:.4f} rad "
                f"(> {_EEF_ROT_TOL_RAD:.4f} rad tolerance). "
                f"init_euler_xyz={init_euler.tolist()}, "
                f"goal_euler_xyz={goal_euler.tolist()}"
            )
    finally:
        _close_env(env)


def _eef_quat_world(env) -> np.ndarray:
    """Return the EE world-frame rotation as a quaternion (xyzw, shape ``(4,)``).

    Reads ``robot0_eef_quat`` directly from the inner robosuite env
    so the test does not depend on the Euler-from-quat round-trip
    (which is degenerate near pitch = ±π/2 and is the principal source
    of gimbal-lock-driven wrist flips).
    """
    inner_env = env._env._env  # ReachBlockingRoboCasaEnv → RoboCasaEnv → robosuite env
    if hasattr(inner_env, "_get_observations"):
        obs_dict = inner_env._get_observations(force_update=True)
    else:
        obs_dict = inner_env._get_observation()
    return np.asarray(obs_dict["robot0_eef_quat"], dtype=np.float64).copy()


# Step-time test parametrisation: one (scene, seed) per kitchen split
# combo (matches the existing `_TEST_CASES`). To keep the test cheap
# (it spends most of its time in OSC sub-steps), we run only a few
# episodes per case and a few WM ticks per episode.
_STEP_TEST_EPISODES = 3
_STEP_TEST_TICKS = 6
# Per-tick position-delta sample magnitude (normalised OSC units in
# [-1, +1]). 0.6 is comfortably above the per-sub-step clamp (0.4)
# defined by `_REACH_STEP_MAX_POS_NORM` in the env, so the clamp gets
# exercised; direction is random so we hit every octant of the
# action space across the 6 ticks.
_STEP_TEST_POS_DELTA_NORM = 0.6


@pytest.mark.parametrize("kitchen_split,scene_seed", _TEST_CASES, ids=_TEST_IDS)
def test_eef_rotation_held_across_steps(kitchen_split, scene_seed):
    """EE world rotation must stay locked to the home pose across every
    :meth:`RoboCasaEnv.step` call in reach-task mode.

    After ``env.reset(mode='init', ...)`` snapshots the home EE
    rotation, the per-sub-step orientation hold inside
    :meth:`RoboCasaEnv._reach_apply_orientation_hold` overrides the
    planner's commanded rotation channel with a quaternion-based
    corrective toward the home rotation AND sphere-clamps the
    commanded position-delta magnitude (so the OSC stays in a regime
    where the wrist can hold orientation). Combined, these guarantee
    every post-step EE rotation lies within
    :data:`_EEF_STEP_ROT_TOL_RAD` of the home rotation.

    The test feeds the env a stream of deliberately-aggressive random
    position-only WM-tick actions (magnitude 0.6 in normalised OSC
    units, which is above the per-sub-step clamp of 0.4) so the
    clamping path is exercised. The rotation channel is zeroed in the
    test action — the env's override is the only thing that can put a
    rotation corrective into the OSC.
    """
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng((scene_seed * 100003) & 0xFFFFFFFF)
    env = _build_env(scene_seed, kitchen_split=kitchen_split)
    try:
        inner_env = env._env._env
        chunk_size = env._env._chunk_size
        for ep in range(_STEP_TEST_EPISODES):
            _reset_init(env, scene_seed, fresh=(ep == 0))
            home_quat = _eef_quat_world(env)
            R_home = Rotation.from_quat(home_quat)

            for tick in range(_STEP_TEST_TICKS):
                # Random direction × fixed magnitude, repeated across
                # all chunk-size OSC sub-steps (matches the
                # broadcast-action convention used by the planner).
                direction = rng.normal(size=3)
                direction = direction / (np.linalg.norm(direction) + 1e-12)
                pos_delta = direction * _STEP_TEST_POS_DELTA_NORM
                action12 = np.zeros((chunk_size, 12), dtype=np.float64)
                action12[:, 5:8] = pos_delta  # ee_pos
                # ee_rot stays zero — the env's override is what
                # actually drives the wrist back to home.
                inner_env  # silence linter; the env wrapper handles the
                # actual sub-stepping
                env._env.step(action12)

                cur_quat = _eef_quat_world(env)
                # Sign-continuity correction so the geodesic distance
                # never crosses the q ↔ -q hemisphere boundary.
                if float(np.dot(cur_quat, home_quat)) < 0.0:
                    cur_quat = -cur_quat
                R_now = Rotation.from_quat(cur_quat)
                delta = float(np.linalg.norm((R_home * R_now.inv()).as_rotvec()))
                assert delta <= _EEF_STEP_ROT_TOL_RAD, (
                    f"{kitchen_split} sc{scene_seed:03d} ep{ep} tick{tick}: "
                    f"EE world rotation drifted {delta:.4f} rad from home "
                    f"(> {_EEF_STEP_ROT_TOL_RAD:.4f} rad tolerance). "
                    f"home_quat={home_quat.tolist()}, "
                    f"cur_quat={cur_quat.tolist()}, "
                    f"pos_delta_norm={float(np.linalg.norm(pos_delta)):.3f}"
                )
    finally:
        _close_env(env)


def _arm_qpos_and_limits(env):
    """Read the 7-DOF Panda arm joint positions + (lower, upper) limits.

    Mirrors :meth:`RoboCasaEnv._get_arm_joint_pos_and_limits` but lives
    in the test module so we don't depend on test-internal helpers
    leaking back into the env API.
    """
    inner_env = env._env._env  # ReachBlockingRoboCasaEnv → RoboCasaEnv → robosuite env
    robot = inner_env.robots[0]
    sim = inner_env.sim
    addrs = [sim.model.jnt_qposadr[sim.model.joint_name2id(j)] for j in robot.robot_joints]
    qpos = np.array([sim.data.qpos[a] for a in addrs], dtype=np.float64)
    lower = np.array(
        [sim.model.jnt_range[sim.model.joint_name2id(j), 0] for j in robot.robot_joints],
        dtype=np.float64,
    )
    upper = np.array(
        [sim.model.jnt_range[sim.model.joint_name2id(j), 1] for j in robot.robot_joints],
        dtype=np.float64,
    )
    return qpos, lower, upper


def _eef_pos_world(env) -> np.ndarray:
    """Return the EE world position (3-vector) from the inner sim."""
    inner_env = env._env._env
    if hasattr(inner_env, "_get_observations"):
        obs_dict = inner_env._get_observations(force_update=True)
    else:
        obs_dict = inner_env._get_observation()
    return np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float64).copy()


# Cases for the drive-forever singularity test. Small scene set to
# keep runtime down; we already know from the v3/v4/v5 sweeps that
# sc049 / sc035 / sc022 / sc016 expose elbow-straightening on the
# planner side under aggressive position commands, so we include
# them here to lock the regression in.
_DRIVE_FOREVER_CASES = [
    ("pretrain", 0),
    ("pretrain", 16),
    ("pretrain", 22),
    ("pretrain", 35),
    ("pretrain", 49),
    ("target", 0),
]
_DRIVE_FOREVER_IDS = [f"{split}-sc{s:03d}" for split, s in _DRIVE_FOREVER_CASES]

# How long to drive forever (in WM ticks; each tick = chunk_size=4 OSC sub-steps).
_DRIVE_FOREVER_TICKS = 10
# Per-joint clearance from a physical limit that we require throughout
# the drive. 0.10 rad (~5.7°) is a hair tighter than the env's reset-
# time validator margin (0.15 rad), so even brief excursions inside
# the env's safety band are caught — but loose enough to absorb
# numerical noise from a single sub-step settling.
_DRIVE_FOREVER_JOINT_MARGIN_RAD = 0.10
# Per-tick EE rotation drift tolerance — same idea as for
# test_eef_rotation_held_across_steps (0.08 rad ~ 4.6°).
_DRIVE_FOREVER_ROT_TOL_RAD = 0.08
# Cartesian "stay near home" tolerance after clipping. The env's
# internal workspace cap is :data:`_REACH_WORKSPACE_R` (set in
# robot_env.py); the test asserts the EE never gets pushed more
# than that + small numerical slack from
# ``_reach_home_eef_pos``. Catches the case where the env stops
# clipping and the planner pushes the EE off into the kitchen.
_DRIVE_FOREVER_EEF_BUDGET_M = 0.50


@pytest.mark.parametrize("kitchen_split,scene_seed", _DRIVE_FOREVER_CASES, ids=_DRIVE_FOREVER_IDS)
@pytest.mark.parametrize("direction_name,direction", [("+x", (1.0, 0.0, 0.0)), ("+z", (0.0, 0.0, 1.0))])
def test_drive_forever_no_singularity(kitchen_split, scene_seed, direction_name, direction):
    """Aggressive constant-direction EE commands must not trigger any
    of the failure modes seen in the v3/v4/v5 planner sweeps:

      * No arm joint may reach within
        :data:`_DRIVE_FOREVER_JOINT_MARGIN_RAD` of its physical limit
        (the "elbow fully straight" → IK degeneracy → wrist locks
         pattern visible in sc022/sc035/sc049/sc016 etc.).
      * The EE world rotation must stay within
        :data:`_DRIVE_FOREVER_ROT_TOL_RAD` of the home rotation
        snapshotted at reset.
      * The EE world position must stay within
        :data:`_DRIVE_FOREVER_EEF_BUDGET_M` of ``_reach_home_eef_pos``
        (proxy: confirms the env's workspace bound is actually
        clipping the commanded delta).

    The test sends a constant max-magnitude position-only WM action
    for :data:`_DRIVE_FOREVER_TICKS` consecutive WM ticks
    (≈ 40 OSC sub-steps). Without env-side clipping the planner
    would push the EE meters away from home; with the workspace bound
    + joint-singularity guard active, the EE saturates at the
    boundary, joints relax, and rotation snaps back to home.

    The two directions parametrised here (+x = forward in base frame,
    +z = up) are chosen because they are the two directions most
    likely to elicit elbow extension on a forward-facing Panda mount
    (forward reach over the counter; vertical reach away from the
    base).
    """
    from scipy.spatial.transform import Rotation

    env = _build_env(scene_seed, kitchen_split=kitchen_split)
    try:
        _reset_init(env, scene_seed, fresh=True)
        home_quat = _eef_quat_world(env)
        R_home = Rotation.from_quat(home_quat)
        home_eef = _eef_pos_world(env)

        chunk_size = env._env._chunk_size
        dir_vec = np.array(direction, dtype=np.float64)

        for tick in range(_DRIVE_FOREVER_TICKS):
            action12 = np.zeros((chunk_size, 12), dtype=np.float64)
            action12[:, 5:8] = dir_vec  # constant max ee_pos delta in +direction
            env._env.step(action12)

            # 1) Joint singularity check — must stay clear of limits.
            qpos, lower, upper = _arm_qpos_and_limits(env)
            margin = float(np.min(np.minimum(qpos - lower, upper - qpos)))
            assert margin >= _DRIVE_FOREVER_JOINT_MARGIN_RAD, (
                f"{kitchen_split} sc{scene_seed:03d} dir={direction_name} tick{tick}: "
                f"arm joint within {margin:.4f} rad of physical limit "
                f"(threshold {_DRIVE_FOREVER_JOINT_MARGIN_RAD:.4f}). "
                f"qpos={qpos.tolist()}, lower={lower.tolist()}, upper={upper.tolist()}"
            )

            # 2) EE rotation must stay near home.
            cur_quat = _eef_quat_world(env)
            if float(np.dot(cur_quat, home_quat)) < 0.0:
                cur_quat = -cur_quat
            R_now = Rotation.from_quat(cur_quat)
            rot_drift = float(np.linalg.norm((R_home * R_now.inv()).as_rotvec()))
            assert rot_drift <= _DRIVE_FOREVER_ROT_TOL_RAD, (
                f"{kitchen_split} sc{scene_seed:03d} dir={direction_name} tick{tick}: "
                f"EE rotation drifted {rot_drift:.4f} rad from home "
                f"(threshold {_DRIVE_FOREVER_ROT_TOL_RAD:.4f}). "
                f"home_quat={home_quat.tolist()}, cur_quat={cur_quat.tolist()}"
            )

            # 3) EE position must stay near home (workspace clipping active).
            eef_now = _eef_pos_world(env)
            eef_dist = float(np.linalg.norm(eef_now - home_eef))
            assert eef_dist <= _DRIVE_FOREVER_EEF_BUDGET_M, (
                f"{kitchen_split} sc{scene_seed:03d} dir={direction_name} tick{tick}: "
                f"EE position drifted {eef_dist:.4f} m from home "
                f"(threshold {_DRIVE_FOREVER_EEF_BUDGET_M:.4f}). "
                f"home_eef={home_eef.tolist()}, cur_eef={eef_now.tolist()}"
            )
    finally:
        _close_env(env)
