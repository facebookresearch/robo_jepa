# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Invariant tests for the **object-push** RoboCasa task.

Sibling of ``test_object_reach_env.py``. The object-push task places a single
free object (default ``"banana"``) on a counter at a per-episode ``start``
position in ``mode='init'`` and at a small-displacement ``goal`` position in
``mode='goal'``; the EEF is driven to a standoff just behind the object so a
non-trivial push remains for the planner. These tests lock in the env-side
invariants the planner / world-model rely on:

  1. ``test_determinism``                — same seed → bit-identical
     start/goal/base across rebuilds; distinct across seeds.
  2. ``test_no_mixer``                   — kitchen is cleared to a backdrop.
  3. ``test_robot_body_static``          — mobile base pose is constant.
  4. ``test_base_spawn_static``          — the kitchen's deterministic base
     object spawn (pre-reposition) is identical across episodes.
  5. ``test_push_distance_bounded``      — ``d_0 = ||goal - start||`` lies in
     the configured ``[min, max]`` band, and matches the obs object positions.
  6. ``test_eef_standoff_dist_bounded``  — the init vs goal standoff EEF poses
     are within a sane budget.
  7. ``test_eef_rotation_matches_between_init_and_goal``.
  8. ``test_eef_rotation_held_across_steps``.
  9. ``test_drive_forever_no_singularity``.

Run (headless, single kitchen smoke ~1-2 min):

    cd <repo_root>
    MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
      PYTHONPATH="$PWD:$PWD/scripts/deployment:$PYTHONPATH" \
      python -m pytest -xvs \
      scripts/deployment/robots/robocasa/test_object_push_env.py \
      -k "kitchen4" -m slow

Full parallel suite (recommend -n 3, each worker ~3-5 GiB):

    MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
      PYTHONPATH="$PWD:$PWD/scripts/deployment:$PYTHONPATH" \
      python -m pytest \
      scripts/deployment/robots/robocasa/test_object_push_env.py \
      -m slow -n 3 --dist loadgroup -v
"""

from __future__ import annotations

import os
import sys

# Headless rendering must be configured before importing mujoco / robosuite.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

# Tests live under scripts/deployment/robots/robocasa/. The wrapper imports
# itself as ``robots.robocasa.*`` so we mirror that layout by prepending
# ``scripts/deployment`` to sys.path (matches test_object_reach_env.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from robots.robocasa.blocking_robot_env import (  # noqa: E402
    _OBJECT_PUSH_INNER_TASK_NAME,
    _OBJECT_PUSH_MAX_DISTANCE_M,
    _OBJECT_PUSH_MIN_DISTANCE_M,
    _OBJECT_PUSH_WRIST_CAMERA,
    _OBJECT_REACH_BODY_NAME,
    ObjectPushBlockingRoboCasaEnv,
)
from robots.robocasa.object_push_kitchen import ObjectPushKitchen  # noqa: E402
from robots.robocasa.robot_env import RoboCasaEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Base object spawn capture (monkey-patched onto ObjectPushKitchen)
# ---------------------------------------------------------------------------
#
# To support :func:`test_base_spawn_static` we capture the object's world xyz
# at the END of ``ObjectPushKitchen._reset_internal`` — i.e. after the fixed
# yaw + per-kitchen/per-seed base override, but BEFORE the wrapper repositions
# it to the per-episode start / goal xy. That base spawn must be identical
# across episodes of the same scene (deterministic kitchen).
_OBJECT_SPAWN_STASH: dict = {}


@pytest.fixture(scope="session", autouse=True)
def _patch_object_push_kitchen_reset_for_object_capture():
    orig = ObjectPushKitchen._reset_internal

    def _patched(self):
        orig(self)
        try:
            objs = getattr(self, "objects", None) or {}
            if "obj" in objs:
                obj = objs["obj"]
                joint_name = obj.joints[0]
                qpos = np.asarray(self.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
                _OBJECT_SPAWN_STASH[id(self)] = qpos[:3].copy()
        except Exception:
            pass

    ObjectPushKitchen._reset_internal = _patched
    try:
        yield
    finally:
        ObjectPushKitchen._reset_internal = orig


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

# Robocasa ships 10 canonical kitchens; layout id 7 is excluded (mirrors the
# object_reach suite). Parametrized via the ``kitchen_runs`` fixture below.
TARGET_KITCHENS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 9, 10)

N_EPISODES: int = 3  # object_push_task_seed = 0, 1, 2
N_RUNS: int = 2  # build → run → close → rebuild → run
SCENE_SEED: int = 0  # fixed scene seed across both runs

TEST_OBJECTS: tuple[str, ...] = ("flour_bag",)

# (object, kitchen_id) tuples to SKIP. Empty for now.
OBJECT_KITCHEN_SKIPS: set[tuple[str, int]] = set()

# Push distance band (imported from the wrapper so the test stays in sync).
PUSH_MIN_M: float = _OBJECT_PUSH_MIN_DISTANCE_M
PUSH_MAX_M: float = _OBJECT_PUSH_MAX_DISTANCE_M

# Tolerances.
DETERMINISM_ATOL: float = 1e-6
BASE_STATIC_ATOL: float = 1e-5
_EEF_ROT_TOL_RAD: float = 0.08
_EEF_STEP_ROT_TOL_RAD: float = 0.08


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_body_pose(sim) -> np.ndarray:
    """Concatenated ``[xpos (3,), xquat (4, wxyz)]`` for the robot base body."""
    for body_name in ("mobilebase0_base", "robot0_base", "base"):
        try:
            bid = sim.model.body_name2id(body_name)
        except Exception:
            continue
        pos = np.asarray(sim.data.body_xpos[bid], dtype=np.float64).copy()
        quat = np.asarray(sim.data.body_xquat[bid], dtype=np.float64).copy()
        return np.concatenate([pos, quat])
    raise RuntimeError("could not resolve robot base body (tried mobilebase0_base, robot0_base, base)")


def _xyzw_to_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _eef_quat_wxyz(inner) -> np.ndarray:
    """Current EEF orientation (wxyz) from the inner env."""
    if hasattr(inner, "_get_observations"):
        obs_dict = inner._get_observations(force_update=True)
    else:
        obs_dict = inner._get_observation()
    return _xyzw_to_wxyz(obs_dict["robot0_eef_quat"])


def _snapshot_reset(env, task_seed: int, mode: str) -> dict:
    """Reset ``env`` with ``object_push_task_seed`` + ``mode`` and capture the
    per-mode state needed by all tests."""
    obs, _ = env.reset(mode=mode, object_push_task_seed=int(task_seed))
    inner = env._env._env  # robosuite Kitchen env
    eef_pos = np.asarray(env._get_eef_pos(), dtype=np.float64).copy()
    eef_quat_wxyz = _eef_quat_wxyz(inner)
    if "object_push_obj_pos_world" in obs:
        obj_pos = np.asarray(obs["object_push_obj_pos_world"], dtype=np.float64).copy()
    else:
        try:
            obj_pos = env._get_object_world_pos().copy()
        except Exception:
            obj_pos = np.full(3, np.nan, dtype=np.float64)
    base_pose = _base_body_pose(inner.sim)
    obj_spawn_xyz = np.asarray(
        _OBJECT_SPAWN_STASH.get(id(inner), np.full(3, np.nan, dtype=np.float64)),
        dtype=np.float64,
    ).copy()
    start_world = np.asarray(
        obs.get("object_push_start_world", np.full(3, np.nan)), dtype=np.float64
    ).copy()
    goal_world = np.asarray(
        obs.get("object_push_goal_world", np.full(3, np.nan)), dtype=np.float64
    ).copy()
    return {
        "eef_pos": eef_pos,
        "eef_quat_wxyz": eef_quat_wxyz,
        "obj_pos": obj_pos,
        "base_pose": base_pose,
        "start_world": start_world,
        "goal_world": goal_world,
        "d0": float(obs.get("object_push_d0", float("nan"))),
        "residual": float(obs.get("object_push_residual", float("nan"))),
        "obj_spawn_xyz": obj_spawn_xyz,
    }


def _make_env(kitchen_id: int, object_name: str) -> ObjectPushBlockingRoboCasaEnv:
    inner_env = RoboCasaEnv(
        task_name=_OBJECT_PUSH_INNER_TASK_NAME,
        camera_names=["robot0_agentview_left", _OBJECT_PUSH_WRIST_CAMERA],
        camera_width=320,
        camera_height=240,
        seed=SCENE_SEED,
        robot="PandaOmron",
        init_robot_to_home=True,
        reach_task=False,
        layout_and_style_ids=[(kitchen_id, kitchen_id)],
    )
    return ObjectPushBlockingRoboCasaEnv(
        inner_env,
        object_push_object_name=object_name,
        object_push_task_seed=0,
        object_push_clear_radius=1.0,
    )


def _run_three_episodes(kitchen_id: int, object_name: str) -> dict:
    """Build env, run 3 (init + goal) episodes, close. Returns ``episodes``
    (list[{init, goal}]) and ``body_names``."""
    env = _make_env(kitchen_id, object_name)
    try:
        body_names = [str(n) for n in env._env._env.sim.model.body_names if n is not None]
    except Exception:
        body_names = []

    episodes: list[dict] = []
    try:
        for ep in range(N_EPISODES):
            init = _snapshot_reset(env, task_seed=ep, mode="init")
            goal = _snapshot_reset(env, task_seed=ep, mode="goal")
            episodes.append({"init": init, "goal": goal})
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "kitchen_id": kitchen_id,
        "object_name": object_name,
        "episodes": episodes,
        "body_names": body_names,
    }


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            (obj, kid),
            marks=[pytest.mark.xdist_group(name=f"push-{obj}-k{kid}")],
            id=f"{obj}-kitchen{kid}",
        )
        for obj in TEST_OBJECTS
        for kid in TARGET_KITCHENS
        if (obj, kid) not in OBJECT_KITCHEN_SKIPS
    ],
)
def kitchen_runs(request) -> dict:
    """Per-(object, kitchen) fixture: build env, run 3 eps, close, rebuild,
    run 3 eps again."""
    object_name, kitchen_id = request.param
    runs = [_run_three_episodes(int(kitchen_id), str(object_name)) for _ in range(N_RUNS)]
    return {
        "kitchen_id": int(kitchen_id),
        "object_name": str(object_name),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_determinism(kitchen_runs):
    """run_a and run_b must produce identical EEF + object + base-pose
    sequences at every (episode, mode); the init object position must also
    differ between task_seed 0 and 1 (so the seed knob is wired through)."""
    runs = kitchen_runs["runs"]
    assert len(runs) == N_RUNS, f"expected {N_RUNS} runs, got {len(runs)}"
    run_a, run_b = runs[0]["episodes"], runs[1]["episodes"]
    assert len(run_a) == len(run_b) == N_EPISODES

    for ep_idx, (ea, eb) in enumerate(zip(run_a, run_b)):
        for mode in ("init", "goal"):
            for key in ("eef_pos", "obj_pos", "base_pose"):
                a = ea[mode][key]
                b = eb[mode][key]
                np.testing.assert_allclose(
                    a,
                    b,
                    atol=DETERMINISM_ATOL,
                    rtol=DETERMINISM_ATOL,
                    err_msg=(
                        f"determinism failed: kitchen="
                        f"{kitchen_runs['kitchen_id']} ep={ep_idx} "
                        f"mode={mode} key={key}\n  run_a={a}\n  run_b={b}"
                    ),
                )

    # The start (init mode) object position for episode 0 and 1 use different
    # ``object_push_task_seed`` values, so must land on distinct samples.
    x0 = run_a[0]["init"]["obj_pos"]
    x1 = run_a[1]["init"]["obj_pos"]
    assert not np.allclose(x0, x1, atol=1e-3), (
        f"kitchen={kitchen_runs['kitchen_id']}: start object position unchanged "
        f"across episodes 0 and 1 (task_seed re-roll appears broken): "
        f"x_0={x0} x_1={x1}"
    )


@pytest.mark.slow
def test_no_mixer(kitchen_runs):
    """No body name may contain ``"mixer"`` — the kitchen is cleared to a
    backdrop with only the pushable object visible / collidable."""
    body_names = kitchen_runs["runs"][0]["body_names"]
    offenders = [n for n in body_names if "mixer" in n.lower()]
    assert not offenders, (
        f"kitchen={kitchen_runs['kitchen_id']}: found body names containing "
        f"'mixer' (expected empty kitchen): {offenders}"
    )


@pytest.mark.slow
def test_robot_body_static(kitchen_runs):
    """The robot base body pose must be identical at every reset point across
    both runs (3 ep × 2 modes × 2 runs = 12 captures)."""
    poses: list[np.ndarray] = []
    labels: list[str] = []
    for run_idx, run in enumerate(kitchen_runs["runs"]):
        for ep_idx, ep in enumerate(run["episodes"]):
            for mode in ("init", "goal"):
                poses.append(ep[mode]["base_pose"])
                labels.append(f"run={run_idx} ep={ep_idx} mode={mode}")
    assert poses, "no base poses collected"
    ref, ref_label = poses[0], labels[0]
    for label, pose in zip(labels[1:], poses[1:]):
        np.testing.assert_allclose(
            pose,
            ref,
            atol=BASE_STATIC_ATOL,
            rtol=BASE_STATIC_ATOL,
            err_msg=(
                f"kitchen={kitchen_runs['kitchen_id']}: robot base pose "
                f"drifted at {label}\n  ref ({ref_label}) = {ref}\n  got = {pose}"
            ),
        )


@pytest.mark.slow
def test_base_spawn_static(kitchen_runs):
    """The kitchen's deterministic BASE object spawn (captured at the end of
    ``ObjectPushKitchen._reset_internal``, before the wrapper repositions the
    object to start/goal) must be identical across all episodes / modes of a
    scene — it is the anchor the per-episode sampler jitters around."""
    spawns: list[np.ndarray] = []
    for run in kitchen_runs["runs"]:
        for ep in run["episodes"]:
            for mode in ("init", "goal"):
                spawns.append(ep[mode]["obj_spawn_xyz"])
    assert spawns, "no base spawns collected"
    for s in spawns:
        assert np.all(np.isfinite(s)), (
            f"kitchen={kitchen_runs['kitchen_id']}: base spawn capture missing (got {s}) — monkeypatch did not fire."
        )
    ref = spawns[0]
    for idx, s in enumerate(spawns[1:], start=1):
        np.testing.assert_allclose(
            s[:2],
            ref[:2],
            atol=DETERMINISM_ATOL,
            rtol=DETERMINISM_ATOL,
            err_msg=(
                f"kitchen={kitchen_runs['kitchen_id']}: base object spawn xy "
                f"drifted at capture {idx}\n  ref={ref}\n  got={s}"
            ),
        )


@pytest.mark.slow
def test_push_distance_bounded(kitchen_runs):
    """The sampled (intended) push displacement ``d_0 = ||goal - start||``
    must lie in the configured band ``[PUSH_MIN, PUSH_MAX]`` for every
    episode, and the obs ``object_push_d0`` must agree with the exposed
    intended ``start_world`` / ``goal_world``. The actual placed object may
    roll a little from the intended spot (rounded banana), so the band is
    asserted on the intended sample — that is the task contract."""
    for run_idx, run in enumerate(kitchen_runs["runs"]):
        for ep_idx, ep in enumerate(run["episodes"]):
            start_w = ep["goal"]["start_world"]
            goal_w = ep["goal"]["goal_world"]
            disp = float(np.linalg.norm(goal_w - start_w))
            d0 = ep["goal"]["d0"]
            assert PUSH_MIN_M - 1e-6 <= disp <= PUSH_MAX_M + 1e-6, (
                f"kitchen={kitchen_runs['kitchen_id']} run={run_idx} ep={ep_idx}: "
                f"intended push displacement {disp:.4f} m outside band "
                f"[{PUSH_MIN_M:.3f}, {PUSH_MAX_M:.3f}]\n"
                f"  start_world={start_w}\n  goal_world={goal_w}"
            )
            assert abs(disp - d0) <= 1e-6, (
                f"kitchen={kitchen_runs['kitchen_id']} run={run_idx} ep={ep_idx}: "
                f"obs d_0={d0:.6f} disagrees with ||goal-start||={disp:.6f}"
            )
            # The init/goal intended samples must be consistent across modes
            # of the same episode (same cached pair).
            np.testing.assert_allclose(
                ep["init"]["start_world"],
                start_w,
                atol=DETERMINISM_ATOL,
                err_msg=(
                    f"kitchen={kitchen_runs['kitchen_id']} ep={ep_idx}: init vs "
                    f"goal disagree on intended start_world"
                ),
            )


@pytest.mark.slow
def test_init_goal_eef_pose_identical(kitchen_runs):
    """REQUIREMENT (2): the robot EEF pose at the end of ``mode='init'`` must
    be EXACTLY the same as at the end of ``mode='goal'`` (the parked overhead
    pose is mode-independent), so the only visual difference between the init
    and goal observations is the object's position."""
    for run_idx, run in enumerate(kitchen_runs["runs"]):
        for ep_idx, ep in enumerate(run["episodes"]):
            init_eef = ep["init"]["eef_pos"]
            goal_eef = ep["goal"]["eef_pos"]
            np.testing.assert_allclose(
                init_eef,
                goal_eef,
                atol=1e-3,
                err_msg=(
                    f"kitchen={kitchen_runs['kitchen_id']} run={run_idx} "
                    f"ep={ep_idx}: init vs goal EEF position differ "
                    f"(must be identical)\n  init={init_eef}\n  goal={goal_eef}"
                ),
            )


@pytest.mark.slow
def test_eef_pose_fixed_across_kitchens_height(kitchen_runs):
    """REQUIREMENT (2): the parked EEF sits a fixed height above the object
    plane (overhead view) — the EEF z minus the object z is ~constant per
    episode and matches the configured view height."""
    from robots.robocasa.blocking_robot_env import _OBJECT_PUSH_VIEW_HEIGHT_M

    for run_idx, run in enumerate(kitchen_runs["runs"]):
        for ep_idx, ep in enumerate(run["episodes"]):
            eef_z = float(ep["init"]["eef_pos"][2])
            obj_z = float(ep["init"]["obj_pos"][2])
            dz = eef_z - obj_z
            assert abs(dz - _OBJECT_PUSH_VIEW_HEIGHT_M) <= 0.05, (
                f"kitchen={kitchen_runs['kitchen_id']} run={run_idx} ep={ep_idx}: "
                f"EEF height above object {dz:.3f} m deviates from configured "
                f"view height {_OBJECT_PUSH_VIEW_HEIGHT_M:.3f} m"
            )


@pytest.mark.slow
def test_object_visible_in_wrist(kitchen_runs):
    """REQUIREMENT (2): the object (and both push endpoints) must project
    inside the wrist (eye-in-hand) camera frame at the parked overhead pose,
    for both init and goal."""
    env = _make_env(kitchen_runs["kitchen_id"], kitchen_runs["object_name"])
    try:
        for ep in range(N_EPISODES):
            for mode in ("init", "goal"):
                obs, _ = env.reset(mode=mode, object_push_task_seed=ep)
                for key, label in (
                    ("object_push_obj_pos_world", "object"),
                    ("object_push_start_world", "start"),
                    ("object_push_goal_world", "goal"),
                ):
                    row, col, H, W = env._project_to_wrist(np.asarray(obs[key]))
                    assert (0 <= row < H) and (0 <= col < W), (
                        f"kitchen={kitchen_runs['kitchen_id']} ep={ep} mode={mode}: "
                        f"{label} not visible in wrist camera — pixel "
                        f"(row={row:.1f}, col={col:.1f}) outside {H}x{W}"
                    )
    finally:
        try:
            env.close()
        except Exception:
            pass


@pytest.mark.slow
def test_eef_rotation_matches_between_init_and_goal(kitchen_runs):
    """The standoff EE world rotation at the end of ``mode='init'`` must match
    the one at the end of ``mode='goal'`` within :data:`_EEF_ROT_TOL_RAD`
    (both standoffs drive to the same top-down target quat)."""
    from scipy.spatial.transform import Rotation

    for run_idx, run in enumerate(kitchen_runs["runs"]):
        for ep_idx, ep in enumerate(run["episodes"]):
            q_init = ep["init"]["eef_quat_wxyz"]
            q_goal = ep["goal"]["eef_quat_wxyz"]
            R_init = Rotation.from_quat(np.array([q_init[1], q_init[2], q_init[3], q_init[0]], dtype=np.float64))
            R_goal = Rotation.from_quat(np.array([q_goal[1], q_goal[2], q_goal[3], q_goal[0]], dtype=np.float64))
            delta = float(np.linalg.norm((R_init * R_goal.inv()).as_rotvec()))
            assert delta <= _EEF_ROT_TOL_RAD, (
                f"kitchen={kitchen_runs['kitchen_id']} run={run_idx} ep={ep_idx}: "
                f"init vs goal standoff EE rotation differ {delta:.4f} rad "
                f"(> {_EEF_ROT_TOL_RAD:.4f} rad)\n"
                f"  q_init_wxyz={q_init.tolist()}\n  q_goal_wxyz={q_goal.tolist()}"
            )


# ---------------------------------------------------------------------------
# Step-time invariants
# ---------------------------------------------------------------------------

_STEP_TEST_KITCHENS: tuple[int, ...] = (1, 4, 9)
_STEP_TEST_TICKS: int = 6
_STEP_TEST_POS_DELTA_M: float = 0.05

_DRIVE_FOREVER_TICKS: int = 6
_DRIVE_FOREVER_POS_DELTA_M: float = 0.05
_DRIVE_FOREVER_JOINT_MARGIN_RAD: float = 0.02
_DRIVE_FOREVER_ROT_TOL_RAD: float = 0.10
_DRIVE_FOREVER_EEF_BUDGET_M: float = 0.50


def _arm_qpos_and_limits(env):
    """7-DOF arm joint positions + (lower, upper) limits."""
    inner_env = env._env._env
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
    inner_env = env._env._env
    if hasattr(inner_env, "_get_observations"):
        obs_dict = inner_env._get_observations(force_update=True)
    else:
        obs_dict = inner_env._get_observation()
    return np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float64).copy()


def _eef_quat_wxyz_env(env) -> np.ndarray:
    return _eef_quat_wxyz(env._env._env)


@pytest.mark.slow
@pytest.mark.parametrize(
    "object_name,kitchen_id",
    [
        pytest.param(
            obj,
            kid,
            marks=[pytest.mark.xdist_group(name=f"push-{obj}-k{kid}-step")],
            id=f"{obj}-kitchen{kid}",
        )
        for obj in TEST_OBJECTS
        for kid in _STEP_TEST_KITCHENS
    ],
)
def test_eef_rotation_held_across_steps(object_name, kitchen_id):
    """EE world rotation must stay locked to the post-reset standoff quat
    across every ``step`` call, regardless of the planner's commanded delta."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng((kitchen_id * 100003 + 1_000_003 * {"banana": 0}.get(object_name, 7)) & 0xFFFFFFFF)
    env = _make_env(kitchen_id, object_name)
    try:
        env.reset(mode="init", object_push_task_seed=0)
        home_quat_wxyz = _eef_quat_wxyz_env(env)
        home_quat_xyzw = np.array(
            [home_quat_wxyz[1], home_quat_wxyz[2], home_quat_wxyz[3], home_quat_wxyz[0]],
            dtype=np.float64,
        )
        R_home = Rotation.from_quat(home_quat_xyzw)

        for tick in range(_STEP_TEST_TICKS):
            direction = rng.normal(size=3)
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            action7 = np.zeros(7, dtype=np.float64)
            action7[:3] = direction * _STEP_TEST_POS_DELTA_M
            action7[6] = +1.0  # gripper closed (solid pusher)
            env.step(action7)

            cur_quat_wxyz = _eef_quat_wxyz_env(env)
            cur_quat_xyzw = np.array(
                [cur_quat_wxyz[1], cur_quat_wxyz[2], cur_quat_wxyz[3], cur_quat_wxyz[0]],
                dtype=np.float64,
            )
            if float(np.dot(cur_quat_xyzw, home_quat_xyzw)) < 0.0:
                cur_quat_xyzw = -cur_quat_xyzw
            R_now = Rotation.from_quat(cur_quat_xyzw)
            delta = float(np.linalg.norm((R_home * R_now.inv()).as_rotvec()))
            assert delta <= _EEF_STEP_ROT_TOL_RAD, (
                f"object={object_name} kitchen={kitchen_id} tick{tick}: EE world "
                f"rotation drifted {delta:.4f} rad from standoff home "
                f"(> {_EEF_STEP_ROT_TOL_RAD:.4f} rad). "
                f"home_quat_wxyz={home_quat_wxyz.tolist()}, "
                f"cur_quat_wxyz={cur_quat_wxyz.tolist()}"
            )
    finally:
        try:
            env.close()
        except Exception:
            pass


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction_name,direction",
    [("+x", (1.0, 0.0, 0.0))],
)
@pytest.mark.parametrize(
    "object_name,kitchen_id",
    [
        pytest.param(
            obj,
            kid,
            marks=[pytest.mark.xdist_group(name=f"push-{obj}-k{kid}-step")],
            id=f"{obj}-kitchen{kid}",
        )
        for obj in TEST_OBJECTS
        for kid in _STEP_TEST_KITCHENS
    ],
)
def test_drive_forever_no_singularity(object_name, kitchen_id, direction_name, direction):
    """Aggressive constant-direction commands must not push any arm joint to
    within :data:`_DRIVE_FOREVER_JOINT_MARGIN_RAD` of a limit, drift the wrist
    rotation more than :data:`_DRIVE_FOREVER_ROT_TOL_RAD`, or push the EE more
    than :data:`_DRIVE_FOREVER_EEF_BUDGET_M` from the standoff home — the
    workspace-bound + joint-margin guard inside the inherited step must hold."""
    from scipy.spatial.transform import Rotation

    env = _make_env(kitchen_id, object_name)
    try:
        env.reset(mode="init", object_push_task_seed=0)
        home_quat_wxyz = _eef_quat_wxyz_env(env)
        home_quat_xyzw = np.array(
            [home_quat_wxyz[1], home_quat_wxyz[2], home_quat_wxyz[3], home_quat_wxyz[0]],
            dtype=np.float64,
        )
        R_home = Rotation.from_quat(home_quat_xyzw)
        home_eef = _eef_pos_world(env)

        dir_vec = np.array(direction, dtype=np.float64) * _DRIVE_FOREVER_POS_DELTA_M
        for tick in range(_DRIVE_FOREVER_TICKS):
            action7 = np.zeros(7, dtype=np.float64)
            action7[:3] = dir_vec
            action7[6] = +1.0
            env.step(action7)

            qpos, lower, upper = _arm_qpos_and_limits(env)
            margin = float(np.min(np.minimum(qpos - lower, upper - qpos)))
            assert margin >= _DRIVE_FOREVER_JOINT_MARGIN_RAD, (
                f"object={object_name} kitchen={kitchen_id} dir={direction_name} "
                f"tick{tick}: arm joint within {margin:.4f} rad of limit "
                f"(threshold {_DRIVE_FOREVER_JOINT_MARGIN_RAD:.4f}). qpos={qpos.tolist()}"
            )

            cur_quat_wxyz = _eef_quat_wxyz_env(env)
            cur_quat_xyzw = np.array(
                [cur_quat_wxyz[1], cur_quat_wxyz[2], cur_quat_wxyz[3], cur_quat_wxyz[0]],
                dtype=np.float64,
            )
            if float(np.dot(cur_quat_xyzw, home_quat_xyzw)) < 0.0:
                cur_quat_xyzw = -cur_quat_xyzw
            R_now = Rotation.from_quat(cur_quat_xyzw)
            rot_drift = float(np.linalg.norm((R_home * R_now.inv()).as_rotvec()))
            assert rot_drift <= _DRIVE_FOREVER_ROT_TOL_RAD, (
                f"object={object_name} kitchen={kitchen_id} dir={direction_name} "
                f"tick{tick}: EE rotation drifted {rot_drift:.4f} rad from standoff "
                f"home (threshold {_DRIVE_FOREVER_ROT_TOL_RAD:.4f})."
            )

            eef_now = _eef_pos_world(env)
            eef_dist = float(np.linalg.norm(eef_now - home_eef))
            assert eef_dist <= _DRIVE_FOREVER_EEF_BUDGET_M, (
                f"object={object_name} kitchen={kitchen_id} dir={direction_name} "
                f"tick{tick}: EE position drifted {eef_dist:.4f} m from standoff "
                f"home (threshold {_DRIVE_FOREVER_EEF_BUDGET_M:.4f})."
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
