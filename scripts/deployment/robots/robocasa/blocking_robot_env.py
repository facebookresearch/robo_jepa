# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Blocking-control wrapper around :class:`RoboCasaEnv`.

Each :meth:`step` interprets the caller's per-WM-step EE delta as a
**target pose** rather than a single OSC sub-step input. Internally the
wrapper runs as many robosuite OSC sub-steps as needed (capped) to
drive the EE to the commanded target — converting world-frame
observations to base-frame controller inputs and clipping each
sub-step's correction to the controller's per-step capacity
(``output_max=0.05 m``).

This is exactly the behaviour the goal generator already uses
(:meth:`RoboCasaEnv._step_blocking_expert`) — generalised to any
caller-supplied 7-D / components-dict / canonical 12-D action.

Why this exists:
    The legacy ``RoboCasaEnv.step`` runs exactly ``chunk_size`` (= 4)
    OSC sub-steps and returns immediately, even if the commanded EE
    delta exceeds the controller's per-step capacity. For env-vs-WM
    comparisons and for closed-loop CEM where the planner expresses
    actions in metric m of *intended* motion (rather than per-OSC
    normalised inputs), this means the env physically realises only a
    fraction of the commanded delta — a 15 cm intent might land as 2 cm
    of actual motion. The blocking wrapper closes the loop in the env
    so the realised motion matches the intent (up to ``pos_tol`` and
    ``max_steps``).

API: drop-in replacement for ``RoboCasaEnv``. ``reset``, ``step``,
``step_chunk``, observation format, and all attributes are identical;
only ``info["sim_steps"]`` reflects the actual (variable) sub-step
count rather than a fixed ``chunk_size``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from action_components import is_components_dict
from robots.robocasa.object_reach_kitchen import (
    OBJECT_REACH_REGISTRY,
    ObjectReachKitchen,
)
from robots.robocasa.object_push_kitchen import (
    OBJECT_PUSH_REGISTRY,
    ObjectPushKitchen,
)
from robots.robocasa.robot_env import (
    _GOAL_BLOCKING_MAX_STEPS,
    _GOAL_BLOCKING_TOL,
    _GOAL_OSC_OUTPUT_MAX,
    _GOAL_RETURN_HOME_POS_TOL,
    _GOAL_RETURN_HOME_ROT_TOL,
    _REACH_DEFAULT_CLEAR_RADIUS,
    _REACH_DEFAULT_MAX_DISTANCE,
    _REACH_DEFAULT_MIN_DISTANCE,
    _REACH_DRIVE_MAX_STEPS,
    _REACH_DRIVE_TOL,
    _REACH_JOINT_LIMIT_MARGIN_RAD,
    _REACH_STEP_MAX_POS_NORM,
    _REACH_WORKSPACE_R,
    RoboCasaEnv,
    ACTION_PER_STEP_DIM,
    _goal_prepare_action,
    _goal_prepare_axisangle,
    _layoutA_to_robosuite12,
)
from scipy.spatial.transform import Rotation

try:  # robocasa is required at runtime for ObjectReach; import is cheap.
    from robocasa.utils import object_utils as OU
except Exception:  # pragma: no cover — keep blocking_robot_env importable
    OU = None  # type: ignore[assignment]


_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _components_to_planner7(components: dict) -> np.ndarray:
    """Pull a planner-style 7-D ``[dx, dy, dz, droll, dpitch, dyaw,
    dgrip]`` (metric m / rad / continuous) from a components dict.

    Mirrors :func:`scripts.robo_jepa.action_components.split_7d_action`
    inverse: ``eef_translation`` (m) → xyz, ``eef_rotation`` (rad
    axis-angle / Euler convention used elsewhere) → euler slots,
    ``gripper`` → last slot. Missing keys default to zero.
    """
    out = np.zeros(7, dtype=np.float64)
    pos = components.get("eef_translation")
    if pos is not None:
        out[:3] = np.asarray(pos, dtype=np.float64).reshape(-1)[:3]
    rot = components.get("eef_rotation")
    if rot is not None:
        out[3:6] = np.asarray(rot, dtype=np.float64).reshape(-1)[:3]
    grip = components.get("gripper")
    if grip is not None:
        out[6] = float(np.asarray(grip).reshape(-1)[0])
    return out


def _action_to_planner7_base(action) -> np.ndarray:
    """Normalise any of the supported ``step()`` action forms to a
    single planner-style 7-D base-frame action vector.

    Supported inputs (mirrors
    :meth:`RoboCasaEnv._coerce_action_to_chunk`):

    * components dict with at least ``eef_translation`` (m, base frame)
    * 7-D numpy array (planner DROID-style, base frame)
    * canonical layout-A 2-D array of shape ``(K, 12)`` or 1-D
      ``(K * 12,)``: we collapse it into a single 7-D delta by
      multiplying the (averaged) ee_pos / ee_rot slots by the OSC
      ``output_max`` (since the canonical-array path encodes one
      per-OSC-step normalised input). For homogeneous chunks (the
      common case where all rows are identical, as produced by
      ``_components_to_chunk``) this exactly recovers the intended
      per-WM-step metric delta.
    """
    if is_components_dict(action):
        return _components_to_planner7(action)

    a = np.asarray(action, dtype=np.float64)
    if a.ndim == 1 and a.size == 7:
        return a.copy()

    if a.ndim == 1:
        # canonical layout-A flat vector: shape (K * 12,).
        assert a.size % 12 == 0, f"flat canonical action must have size K*12, got {a.size}"
        a = a.reshape(-1, 12)

    if a.ndim == 2:
        # canonical layout-A (K, 12). Slots: [0:4]=base_motion,
        # [4]=control_mode, [5:8]=ee_pos (normalised), [8:11]=ee_rot
        # (normalised), [11]=gripper.
        from robots.robocasa.robot_env import (  # local to avoid cycles
            _GOAL_OSC_OUTPUT_MAX,
        )

        # Sum over the chunk so the intended per-WM-step motion is the
        # composition of the per-OSC-step canonical inputs (each input
        # ≈ 0.05 m × value, applied K times).
        ee_pos_norm = a[:, 5:8].sum(axis=0)  # (3,)
        ee_rot_norm = a[:, 8:11].sum(axis=0)  # (3,)
        gripper_norm = a[:, 11].mean()  # scalar
        out = np.zeros(7, dtype=np.float64)
        out[:3] = ee_pos_norm * _GOAL_OSC_OUTPUT_MAX[0]  # 0.05 m / unit
        out[3:6] = ee_rot_norm * _GOAL_OSC_OUTPUT_MAX[3]  # 0.5 rad / unit
        out[6] = float(gripper_norm)
        return out

    raise ValueError(f"action must be dict, 1-D, or 2-D; got ndim={a.ndim}")


# ---------------------------------------------------------------------------
# Reach-task reward helper (shared between the back-compat default in
# :class:`BlockingRoboCasaEnv._compute_task_reward` and the explicit
# override in :class:`ReachBlockingRoboCasaEnv`).
# ---------------------------------------------------------------------------


def _apply_reach_reward(
    env: RoboCasaEnv,
    reward: float,
    out_obs: dict,
    info: dict,
) -> tuple[float, dict, dict]:
    """Recompute the reach-task reward from inner-env state.

    Mirrors :meth:`RoboCasaEnv.step` lines 1126-1149: produces
    ``r = max(0, 1 - dist/d_0)`` (or ``1.0`` for the degenerate
    ``d_0 ~ 0`` case), and writes the standard ``reach_*`` keys onto
    ``out_obs`` and ``info``. Caller is responsible for guarding on
    ``env._reach_task`` / ``env._reach_goal_world is not None``.
    """
    eef_now = env._get_eef_pos()
    dist_to_goal = float(np.linalg.norm(eef_now - env._reach_goal_world))
    d0 = env._reach_d0
    if d0 > 1e-9:
        reward = max(0.0, 1.0 - dist_to_goal / d0)
    else:
        reward = 1.0
    out_obs["reach_distance_to_goal"] = dist_to_goal
    out_obs["reach_d0"] = d0
    out_obs["reach_goal_world"] = env._reach_goal_world.copy()
    info["reach_distance_to_goal"] = dist_to_goal
    info["reach_d0"] = d0
    info["reach_reward"] = float(reward)
    return float(reward), out_obs, info


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class BlockingRoboCasaEnv:
    """Blocking-control wrapper around :class:`RoboCasaEnv`.

    Drop-in: forwards every attribute / method to the wrapped env; only
    overrides :meth:`step` (and :meth:`step_chunk`, which iterates).

    Parameters
    ----------
    env : RoboCasaEnv
        The underlying robocasa env (already constructed). The wrapper
        re-uses its ``_get_base_world_pose`` / ``_get_eef_pos`` /
        ``_restore_base_joints`` / ``_lock_base_in_arm_mode`` helpers.
    pos_tol : float, default :data:`_GOAL_BLOCKING_TOL`
        World-frame Euclidean tolerance (m) for declaring the EE
        converged to the commanded target.
    max_steps : int, default :data:`_GOAL_BLOCKING_MAX_STEPS`
        Hard cap on OSC sub-steps per :meth:`step` call (so a
        physically-unreachable target — e.g. an action that would
        collide with the table — eventually returns instead of
        spinning forever).
    apply_rotation_in_first_substep : bool, default True
        Mirror :meth:`RoboCasaEnv._step_blocking_expert`: send the
        commanded ``ee_rot`` axis-angle delta on iteration 0 only;
        subsequent iterations send rot=0 corrective. (The current goal
        generator does the same.)
    """

    # Set of attribute names that live on the wrapper itself rather than
    # being forwarded to ``self._env`` via ``__setattr__``. Subclasses
    # extend this by overriding the class attribute (``cls._WRAPPER_PRIVATE_ATTRS
    # = BlockingRoboCasaEnv._WRAPPER_PRIVATE_ATTRS + ("_extra", ...)``).
    _WRAPPER_PRIVATE_ATTRS: tuple[str, ...] = (
        "_env",
        "_pos_tol",
        "_max_steps",
        "_apply_rotation_first",
    )

    def __init__(
        self,
        env: RoboCasaEnv,
        pos_tol: float = _GOAL_BLOCKING_TOL,
        max_steps: int = _GOAL_BLOCKING_MAX_STEPS,
        apply_rotation_in_first_substep: bool = True,
    ):
        self._env = env
        self._pos_tol = float(pos_tol)
        self._max_steps = int(max_steps)
        self._apply_rotation_first = bool(apply_rotation_in_first_substep)

    # ------------------------------------------------------------------
    # Attribute / method delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Delegate any attribute we don't define ourselves to the
        # wrapped env (so e.g. ``env.action_space`` / ``env._build_obs``
        # / ``env._chunk_size`` keep working transparently). NOTE:
        # Python only calls ``__getattr__`` when normal attribute lookup
        # fails — so our own attributes (``_env``, ``_pos_tol`` …) take
        # precedence.
        return getattr(self._env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Only our own attrs are stored on self; everything else goes
        # to the underlying env. The whitelist is defined on the
        # ``type(self)`` class so subclasses can add their own private
        # slots without forwarding them to ``self._env``.
        if name in type(self)._WRAPPER_PRIVATE_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)

    # ------------------------------------------------------------------
    # Core blocking step
    # ------------------------------------------------------------------

    def step(self, action):
        """Execute one *blocking* WM tick.

        The action's xyz delta is interpreted as a **target offset in
        the base frame** to reach (in metres). The wrapper runs OSC
        sub-steps until the EE is within ``pos_tol`` of the world-frame
        target (``current + R_wb @ Δxyz_base``), or ``max_steps`` is
        reached. Each sub-step's correction is computed in the world
        frame and rotated back into the base frame (the OSC controller
        consumes base-frame inputs).

        Returns
        -------
        obs : dict           — observation after the last sub-step
        reward : float       — robosuite reward of the last sub-step
        done : bool          — robosuite done flag of the last sub-step
        info : dict          — ``{"success", "sim_steps", "control_modes",
            "blocking_converged", "blocking_residual_m"}``.
        """
        a7_base = _action_to_planner7_base(action)

        # Snapshot base orientation + position (the base is locked while
        # we're in arm mode, so this is fixed across the blocking loop).
        base_pos_world, R_wb = self._env._get_base_world_pose()
        R_bw = R_wb.T

        # Compute the world-frame target from the base-frame Δxyz.
        eef_before_world = self._env._get_eef_pos()
        target_world = eef_before_world + R_wb @ a7_base[:3]

        modes: list[int] = []
        reward = 0.0
        done = False
        residual_m = float("nan")
        converged = False

        # Per-sub-step EE orientation hold (REACH invariant #6) setup. The
        # active hold lives in ``RoboCasaEnv.step`` via
        # ``_reach_apply_orientation_hold``, but this blocking drive bypasses
        # that path (it steps ``self._env._env`` directly), so without
        # re-applying it the wrist drifts freely (observed up to ~180° in
        # obstacle-reach). We (a) push a quaternion corrective toward the home
        # EE quat every sub-step, and (b) require orientation to converge —
        # not just position — before exiting, so every completed env.step ends
        # within ``_ORIENT_TOL`` of the home rotation (the invariant the test
        # ``test_eef_rotation_held_across_steps`` asserts at 0.08 rad).
        inner = self._env
        _reach_hold = bool(getattr(inner, "_reach_task", False)) and (
            getattr(inner, "_reach_home_eef_quat", None) is not None
        )

        # Workspace bound (REACH fix #6, step 0): clamp the world target to a
        # sphere of radius ``_REACH_WORKSPACE_R`` around the home EE position.
        # Both reach start & goal are sampled within this sphere, so it never
        # blocks a legitimate reach — it only stops the planner driving the arm
        # past it into the elbow-straight singularity, the config where the
        # OSC can no longer hold orientation.
        if _reach_hold and getattr(inner, "_reach_home_eef_pos", None) is not None:
            home_pos = np.asarray(inner._reach_home_eef_pos, dtype=np.float64)
            off = target_world - home_pos
            off_n = float(np.linalg.norm(off))
            if off_n > _REACH_WORKSPACE_R:
                target_world = home_pos + off * (_REACH_WORKSPACE_R / off_n)

        for i in range(self._max_steps):
            eef_now_world = self._env._get_eef_pos()
            err_world = target_world - eef_now_world
            residual_m = float(np.linalg.norm(err_world))
            if residual_m < self._pos_tol:
                converged = True
                break

            if _reach_hold:
                # Reuse the VALIDATED ``RoboCasaEnv._reach_apply_orientation_hold``
                # (the exact workspace-bound + joint-margin guard + home-quat
                # rotation corrective that ``test_eef_rotation_held_across_steps``
                # asserts) by building the layout-A action it expects and stepping
                # via the same layout-A path RoboCasaEnv.step uses. This blocking
                # drive otherwise bypasses that hold (it steps self._env._env
                # directly), which let the wrist tumble (~180°) in obstacle-reach.
                # sub[5:8] = normalised base-frame position delta toward target;
                # sub[8:11] (rotation) is overwritten by the hold; sub[4]=0 keeps
                # arm mode; sub[11]=gripper.
                sub = np.zeros(ACTION_PER_STEP_DIM, dtype=np.float64)
                sub[5:8] = np.clip((R_bw @ err_world) / _GOAL_OSC_OUTPUT_MAX[:3], -1.0, 1.0)
                sub[11] = a7_base[6]
                sub = inner._reach_apply_orientation_hold(sub)
                full = _layoutA_to_robosuite12(sub)
            else:
                # Non-reach tasks: original goal-generator path.
                corrective = np.zeros(7, dtype=np.float64)
                corrective[:3] = R_bw @ err_world
                if self._apply_rotation_first and i == 0:
                    corrective[3:6] = a7_base[3:6]
                corrective[6] = a7_base[6]
                full = _goal_prepare_action(corrective)

            # Each call advances the simulator by exactly one OSC sub-step.
            _, reward, done, _ = self._env._env.step(full)
            modes.append(-1)

            if self._env._lock_base_in_arm_mode:
                self._env._restore_base_joints()

        success = self._env._env._check_success()
        info = {
            "success": bool(success),
            "sim_steps": len(modes),
            "control_modes": modes,
            "blocking_converged": bool(converged),
            "blocking_residual_m": residual_m,
        }
        out_obs = self._env._build_obs()

        reward, out_obs, info = self._compute_task_reward(float(reward), out_obs, info)

        return out_obs, float(reward), bool(done), info

    # ------------------------------------------------------------------
    # Task-reward hook
    # ------------------------------------------------------------------

    def _compute_task_reward(
        self,
        reward: float,
        out_obs: dict,
        info: dict,
    ) -> tuple[float, dict, dict]:
        """Hook: subclasses override to inject a task-specific reward
        (and matching obs/info keys).

        Default behaviour returns the upstream robosuite reward of the
        last OSC sub-step unchanged. As a back-compat shim it ALSO
        reproduces the legacy reach-task reward block when the inner
        env was constructed with ``reach_task=True`` — so existing
        callers that still wrap a ``RoboCasaEnv(reach_task=True, …)``
        with the bare ``BlockingRoboCasaEnv`` keep observing the same
        ``r = max(0, 1 - dist/d_0)`` reward and the same
        ``reach_*`` obs/info keys as before the refactor. New code
        should construct :class:`ReachBlockingRoboCasaEnv` explicitly.
        """
        if getattr(self._env, "_reach_task", False) and getattr(self._env, "_reach_goal_world", None) is not None:
            return _apply_reach_reward(self._env, reward, out_obs, info)
        return reward, out_obs, info

    # ------------------------------------------------------------------
    # step_chunk: just iterate :meth:`step` to keep the planner contract.
    # ------------------------------------------------------------------

    def step_chunk(self, actions, dt=None):
        chunk_obs: list[dict] = []
        chunk_rewards: list[float] = []
        last_obs = None
        last_reward = 0.0
        last_done = False
        last_info: dict = {}
        for k, a in enumerate(actions):
            t0 = time.perf_counter()
            obs, reward, done, info = self.step(a)
            chunk_obs.append(obs)
            chunk_rewards.append(float(reward))
            last_obs, last_reward, last_done, last_info = obs, reward, done, info
            if dt is not None and k < len(actions) - 1:
                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        last_info = dict(last_info or {})
        last_info["chunk_obs"] = chunk_obs
        # Per-tick rewards — see :meth:`RoboCasaEnv.step_chunk` for the
        # rationale. Mirrors the inner env's contract so the planner
        # gets the same shape no matter which wrapper layer is on top.
        last_info["chunk_rewards"] = chunk_rewards
        return last_obs, float(last_reward), bool(last_done), last_info


# ---------------------------------------------------------------------------
# Reach-task wrapper subclass
# ---------------------------------------------------------------------------


class ReachBlockingRoboCasaEnv(BlockingRoboCasaEnv):
    """Blocking wrapper specialised for the legacy reach task.

    Drop-in replacement for :class:`BlockingRoboCasaEnv` when the planner
    wants the dense reach reward: it owns the ``reach_task_seed`` /
    ``reach_max_distance`` / ``reach_clear_radius`` knobs that mirror
    :class:`RoboCasaEnv`'s ``reach_task=True`` constructor kwargs and
    forces the inner env into reach-task mode in ``__init__`` (so the
    inner env's ``reset`` branch — ``_sample_reach_targets`` /
    ``_drive_eef_to_world_target`` / ``_hide_nearby_appliances`` and
    friends — keeps working unchanged). On every :meth:`step` the
    overridden :meth:`_compute_task_reward` recomputes
    ``r = max(0, 1 - dist_to_goal / d_0)`` from inner-env state and
    flags ``info['success']`` once the EEF is within
    :data:`_GOAL_RETURN_HOME_POS_TOL` of the goal.

    Parameters
    ----------
    env : RoboCasaEnv
        The underlying robocasa env. May have been constructed with
        either ``reach_task=False`` (this wrapper will flip it on) or
        ``reach_task=True`` (this wrapper will simply reuse it).
    reach_task_seed : int, default 0
        Dedicated RNG seed for sampling the per-episode start/goal eef
        poses. Independent of the scene seed, identical semantics to
        :class:`RoboCasaEnv`'s ``reach_task_seed`` ctor kwarg.
    reach_max_distance : float
        Radius (m) of the 3-D ball around the captured home eef inside
        which start/goal eef poses are sampled. Identical semantics to
        :class:`RoboCasaEnv`'s ``reach_max_distance``.
    reach_clear_radius : float
        Radius (m) inside which non-structural appliances are hidden at
        reset time. Identical semantics to :class:`RoboCasaEnv`'s
        ``reach_clear_radius``.
    pos_tol, max_steps, apply_rotation_in_first_substep
        Forwarded to :class:`BlockingRoboCasaEnv`.
    """

    # No new wrapper-private attrs: every reach-task knob is forwarded
    # to the inner env (where ``RoboCasaEnv.__init__`` already owns the
    # canonical ``self._reach_*`` slots).
    _WRAPPER_PRIVATE_ATTRS = BlockingRoboCasaEnv._WRAPPER_PRIVATE_ATTRS

    def __init__(
        self,
        env: RoboCasaEnv,
        *,
        reach_task_seed: int = 0,
        reach_min_distance: float = _REACH_DEFAULT_MIN_DISTANCE,
        reach_max_distance: float = _REACH_DEFAULT_MAX_DISTANCE,
        reach_clear_radius: float = _REACH_DEFAULT_CLEAR_RADIUS,
        kitchen_split: str | None = None,
        pos_tol: float = _GOAL_BLOCKING_TOL,
        max_steps: int = _GOAL_BLOCKING_MAX_STEPS,
        apply_rotation_in_first_substep: bool = True,
    ):
        super().__init__(
            env,
            pos_tol=pos_tol,
            max_steps=max_steps,
            apply_rotation_in_first_substep=apply_rotation_in_first_substep,
        )
        # Force reach-task mode on the inner env. The wrapper's
        # ``__setattr__`` forwards unknown attribute writes to
        # ``self._env`` already, but we go through the inner env
        # explicitly here to make the side-effect obvious. ``RoboCasaEnv.reset``
        # consults these attrs every reset, so flipping them post-construction
        # is enough to enable the reach-task branch on the next reset.
        env._reach_task = True
        env._reach_task_seed = int(reach_task_seed)
        env._reach_min_distance = float(reach_min_distance)
        env._reach_max_distance = float(reach_max_distance)
        env._reach_clear_radius = float(reach_clear_radius)
        assert env._reach_min_distance >= 0.0
        assert env._reach_min_distance <= env._reach_max_distance
        # Optional override of the inner env's kitchen pool (e.g. switch
        # an already-constructed env from the pretrain pool to the
        # 10-kitchen target pool without re-instantiating it). The
        # override only takes effect on the next ``_create_env`` call
        # (i.e. the next reset, since reach-task mode force-rebuilds
        # the inner env every reset \u2014 see Fix A).
        if kitchen_split is not None:
            from robots.robocasa.robot_env import _KITCHEN_SPLITS

            assert kitchen_split in _KITCHEN_SPLITS, (
                f"kitchen_split must be one of {_KITCHEN_SPLITS}, got {kitchen_split!r}"
            )
            env._kitchen_split = kitchen_split

    def _compute_task_reward(
        self,
        reward: float,
        out_obs: dict,
        info: dict,
    ) -> tuple[float, dict, dict]:
        """Override: compute the dense reach reward from inner-env state.

        Formula: ``r = max(0, 1 - dist_to_goal / d_0)`` (with the
        ``d_0 ~ 0`` degenerate case clamped to ``r = 1``). Sets
        ``info['success'] = True`` once the EEF is within
        :data:`_GOAL_RETURN_HOME_POS_TOL` of the sampled goal so the
        planner can short-circuit a successful episode without consulting
        the underlying robocasa task's ``_check_success`` (which has no
        relation to the reach geometry).
        """
        if not self._env._reach_task or self._env._reach_goal_world is None:
            return reward, out_obs, info
        reward, out_obs, info = _apply_reach_reward(self._env, reward, out_obs, info)
        if info["reach_distance_to_goal"] < _GOAL_RETURN_HOME_POS_TOL:
            info["success"] = True
        return reward, out_obs, info


# ---------------------------------------------------------------------------
# Obstacle-reach wrapper subclass
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Obstacle-reach wrapper subclass
# ---------------------------------------------------------------------------
#
# All obstacle-reach state, constants, and helpers live in this section
# so that :mod:`robot_env` stays generic (it only carries the
# ``_reach_task_hook`` extension point — see
# :meth:`RoboCasaEnv.reset`). The wrapper itself implements the hook
# protocol (``before_hide`` / ``sample_targets`` / ``decorate_obs`` /
# ``drive_axis_order``) so the inner env's reach branch delegates
# task-specific behaviour back here.

# Allowlist of robocasa Fixture class names eligible to be the obstacle.
# Single-class default keeps the strict-left/right placement bbox
# consistent across episodes.
_OBSTACLE_REACH_DEFAULT_CLASSES: tuple[str, ...] = ("CoffeeMachine",)
# Extra clearance (m) on top of the obstacle's Y_base half-extent when
# computing the canonical left / right endpoint midpoints. 5 cm was
# too tight for the gripper geometry (~7 cm half-width) to swing past
# the fixture corner without clipping.
_OBSTACLE_REACH_DEFAULT_CLEARANCE_M: float = 0.12
# Gaussian std (m, base frame) used to jitter the canonical left /
# right midpoints when sampling start / goal.
_OBSTACLE_REACH_DEFAULT_JITTER_STD_M: float = 0.05
# Sanity bounds (m) on a candidate fixture's Y_base footprint half-
# extent. Filters malformed assets.
_OBSTACLE_REACH_MIN_Y_HALF_M: float = 0.025
_OBSTACLE_REACH_MAX_Y_HALF_M: float = 0.40
# Hard caps on rejection-sampling attempts inside the sampler.
_OBSTACLE_REACH_SAMPLE_MAX_TRIES: int = 10_000
_OBSTACLE_REACH_VALIDATION_MAX_TRIES: int = 200
# Per-episode workspace cap (m, base frame, per axis) measured from
# the captured home EEF. Endpoints outside this box are rejected.
_OBSTACLE_REACH_WORKSPACE_HALF_EXTENT_M: float = 1.00
# How far an endpoint's Z component may drift from the canonical
# fixture-center anchor Z (rejection sampling clips Gaussian Z jitter
# to ± this).
_OBSTACLE_REACH_ENDPOINT_Z_TOL_M: float = 0.05
# Max ||(midpoint.y − fixture.y)|| in base frame — the obstacle must
# sit between the two endpoints along the lateral (Y_base) axis.
_OBSTACLE_REACH_MIDPOINT_XY_TOL_M: float = 0.08
# Verify-only knob: canonical endpoint Z offset above the fixture top
# (used by :mod:`verify_obstacle_reach_invariants` to derive the
# expected anchor Z from the obs dict's ``obstacle_world_top_z``).
# The runtime sampler anchors on the fixture body Z, not on top+offset.
_OBSTACLE_REACH_ENDPOINT_Z_ABOVE_TOP_M: float = 0.04
# Verify-only knob: max allowed off-axis Y_base offset of the picked
# obstacle. With the placer pinning the fixture to base Y = 0, this is
# effectively a sanity check that placement actually ran.
_OBSTACLE_REACH_MAX_FIXTURE_OFFAXIS_Y_BASE_M: float = 1.00
# Distance (m, robot base frame +X) between the home EEF and the
# obstacle fixture's NEAR FACE. 0.0 m places the fixture's near
# face right at the home EEF's base_X (visually: the CoffeeMachine
# sits at the closest counter edge).
_OBSTACLE_REACH_PLACEMENT_FORWARD_M: float = 0.00
# Cap the X anchor of the straddle endpoints at this distance ahead
# of the home EEF (base frame). For very deep CoffeeMachines (depth
# ≥ 0.40 m) the fixture body centre sits past the Panda's converged
# reach, so we anchor endpoints 15 cm forward of home instead —
# still inside the fixture's X span, but inside reach.
_OBSTACLE_REACH_MAX_X_ANCHOR_FROM_EEF_M: float = 0.15


class ObstacleReachBlockingRoboCasaEnv(ReachBlockingRoboCasaEnv):
    """Blocking wrapper specialised for the obstacle-reach task.

    Plugs into :class:`RoboCasaEnv`'s generic reach machinery via the
    ``_reach_task_hook`` extension point. The wrapper itself implements
    the hook protocol:

    * :meth:`before_hide` picks an allow-listed Fixture (default:
      ``CoffeeMachine``) and physically moves it to a canonical
      in-front-of-robot position via :meth:`_place_obstacle_in_front_of_robot`.
      Returns the picked fixture's name so the inner env's
      ``_hide_nearby_appliances`` clearer skips it.
    * :meth:`sample_targets` overrides the default reach sampler with
      :meth:`_sample_obstacle_reach_targets`, which produces
      strict-left / strict-right straddle pairs anchored on the
      placed fixture's body center.
    * :attr:`drive_axis_order` = ``"yxz"`` so the reset-time drive
      swings the EEF laterally past the fixture before approaching
      it forward — without this the OSC plows the wrist into the
      fixture's corner.
    * :meth:`decorate_obs` adds the obstacle's name / class / world
      pose / footprint extents to the reset obs.

    Reuses :meth:`ReachBlockingRoboCasaEnv._compute_task_reward` for
    the dense ``r = max(0, 1 − dist/d_0)`` reward — the obstacle
    branch populates the same ``_reach_goal_world`` / ``_reach_d0``
    slots the plain reach path does.

    Parameters
    ----------
    env : RoboCasaEnv
        The underlying robocasa env. ``_reach_task`` is forced on (via
        the super-class ctor) and the wrapper installs itself as
        ``env._reach_task_hook``.
    obstacle_classes : tuple[str, ...], optional
        Fixture class names eligible to be the obstacle. Defaults to
        :data:`_OBSTACLE_REACH_DEFAULT_CLASSES`.
    obstacle_clearance_m : float, optional
        See :data:`_OBSTACLE_REACH_DEFAULT_CLEARANCE_M`.
    obstacle_lateral_jitter_std : float, optional
        See :data:`_OBSTACLE_REACH_DEFAULT_JITTER_STD_M`.
    reach_task_seed, reach_min_distance, reach_max_distance,
    reach_clear_radius, kitchen_split, pos_tol, max_steps,
    apply_rotation_in_first_substep
        Forwarded to :class:`ReachBlockingRoboCasaEnv`.
    """

    # All obstacle-reach state lives on the wrapper (not on the inner
    # env). The wrapper's ``__setattr__`` would otherwise forward
    # writes to ``self._env``.
    _WRAPPER_PRIVATE_ATTRS = ReachBlockingRoboCasaEnv._WRAPPER_PRIVATE_ATTRS + (
        "_obstacle_classes",
        "_obstacle_clearance_m",
        "_obstacle_lateral_jitter_std",
        "_obstacle_meta",
        "_obstacle_targets_cache",
        "drive_axis_order",
    )

    def __init__(
        self,
        env: RoboCasaEnv,
        *,
        reach_task_seed: int = 0,
        reach_min_distance: float = _REACH_DEFAULT_MIN_DISTANCE,
        reach_max_distance: float = _REACH_DEFAULT_MAX_DISTANCE,
        reach_clear_radius: float = _REACH_DEFAULT_CLEAR_RADIUS,
        kitchen_split: str | None = None,
        obstacle_classes: tuple[str, ...] | None = None,
        obstacle_clearance_m: float | None = None,
        obstacle_lateral_jitter_std: float | None = None,
        pos_tol: float = _GOAL_BLOCKING_TOL,
        max_steps: int = _GOAL_BLOCKING_MAX_STEPS,
        apply_rotation_in_first_substep: bool = True,
    ):
        super().__init__(
            env,
            reach_task_seed=reach_task_seed,
            reach_min_distance=reach_min_distance,
            reach_max_distance=reach_max_distance,
            reach_clear_radius=reach_clear_radius,
            kitchen_split=kitchen_split,
            pos_tol=pos_tol,
            max_steps=max_steps,
            apply_rotation_in_first_substep=apply_rotation_in_first_substep,
        )
        # Wrapper-side obstacle config + state.
        self._obstacle_classes = tuple(str(c) for c in (obstacle_classes or _OBSTACLE_REACH_DEFAULT_CLASSES))
        self._obstacle_clearance_m = float(
            _OBSTACLE_REACH_DEFAULT_CLEARANCE_M if obstacle_clearance_m is None else obstacle_clearance_m
        )
        self._obstacle_lateral_jitter_std = float(
            _OBSTACLE_REACH_DEFAULT_JITTER_STD_M
            if obstacle_lateral_jitter_std is None
            else obstacle_lateral_jitter_std
        )
        assert self._obstacle_clearance_m >= 0.0
        assert self._obstacle_lateral_jitter_std >= 0.0
        # Per-(effective_reach_seed) cache of validated (start, goal)
        # pairs. Independent of the inner env's ``_reach_targets_cache``
        # so a single env instance can switch between obstacle and
        # plain reach modes without cache contamination.
        self._obstacle_targets_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Metadata about the obstacle chosen on the current reset.
        # Populated by :meth:`before_hide` (hook protocol).
        self._obstacle_meta: dict | None = None
        # Hook protocol: drive axis order for the reset-time
        # ``_drive_eef_to_world_target`` call.
        self.drive_axis_order = "yxz"
        # Install this wrapper as the inner env's reach-task hook so
        # ``RoboCasaEnv.reset`` delegates to us at the right moments.
        env._reach_task_hook = self

    # ------------------------------------------------------------------
    # Reach-task hook protocol (consumed by RoboCasaEnv.reset)
    # ------------------------------------------------------------------

    def before_hide(self, env: RoboCasaEnv) -> str | None:
        """Hook: pick + place the obstacle BEFORE the workspace clearer
        runs. Returns the picked fixture's name so the clearer skips
        it.
        """
        meta = self._pick_obstacle_fixture()
        meta = self._place_obstacle_in_front_of_robot(meta)
        self._obstacle_meta = meta
        return meta["name"]

    def sample_targets(self, env: RoboCasaEnv, effective_reach_seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Hook: return (start, goal) for this episode. Uses the
        obstacle-aware sampler, with rotation-validation wrapping the
        candidate drives in Y-first + stall-check mode so the OSC
        doesn't plow the wrist into the obstacle.
        """
        cached = self._obstacle_targets_cache.get(effective_reach_seed)
        if cached is not None:
            return cached
        assert self._obstacle_meta is not None, "before_hide must run first"

        def _validator(s_w, g_w):
            return self._env._validate_rotation_for_targets(
                s_w,
                g_w,
                axis_order="yxz",
                stall_check=True,
                max_steps=200,
            )

        start_pos, goal_pos = self._sample_obstacle_reach_targets(
            effective_reach_seed,
            obstacle_meta=self._obstacle_meta,
            validator=_validator,
        )
        self._obstacle_targets_cache[effective_reach_seed] = (
            np.asarray(start_pos, dtype=np.float64).copy(),
            np.asarray(goal_pos, dtype=np.float64).copy(),
        )
        return start_pos, goal_pos

    def decorate_obs(self, env: RoboCasaEnv, obs: dict) -> None:
        """Hook: add obstacle-specific keys to the reset obs."""
        meta = self._obstacle_meta
        if meta is None:
            return
        obs["obstacle_fixture_name"] = meta["name"]
        obs["obstacle_class"] = meta["class"]
        obs["obstacle_world_pos"] = np.asarray(meta["world_pos"], dtype=np.float64).copy()
        obs["obstacle_y_half_extent_m"] = float(meta["y_half_extent_m"])
        obs["obstacle_world_top_z"] = float(meta.get("world_top_z", meta["world_pos"][2]))
        obs["obstacle_offaxis_y_base_m"] = float(meta.get("offaxis_y_base_m", 0.0))
        if "y_base_min" in meta:
            obs["obstacle_y_base_min"] = float(meta["y_base_min"])
            obs["obstacle_y_base_max"] = float(meta["y_base_max"])

    # ------------------------------------------------------------------
    # Obstacle picker / placement / sampler (wrapper-side, formerly on
    # RoboCasaEnv prior to the refactor)
    # ------------------------------------------------------------------

    def _compute_obstacle_meta(self, name: str, fixture) -> dict | None:
        """Build the obstacle metadata dict for a fixture at its CURRENT
        world position. Returns ``None`` on lookup failure or
        implausible size. Reads ``sim.data.body_xpos[root_bid]`` for
        the world position, so it always reflects the up-to-date body
        pose after ``sim.forward()``.
        """
        env = self._env
        sim = env._env.sim
        model = sim.model
        cls_name = type(fixture).__name__
        root_body_name = getattr(fixture, "root_body", None)
        if root_body_name is None:
            return None
        try:
            root_bid = int(model.body_name2id(root_body_name))
        except Exception:
            return None
        fixture_size = getattr(fixture, "size", None)
        fixture_rot_z = float(getattr(fixture, "rot", 0.0) or 0.0)
        if fixture_size is None or len(fixture_size) < 2:
            return None
        w_local = float(fixture_size[0])
        d_local = float(fixture_size[1])
        if not np.isfinite(w_local) or not np.isfinite(d_local):
            return None

        fixture_world = np.asarray(sim.data.body_xpos[root_bid], dtype=np.float64).copy()
        base_pos_world, R_wb = env._get_base_world_pose()
        R_bw = R_wb.T
        base_to_fixture_base = R_bw @ (fixture_world - base_pos_world)

        # Body-local axis-aligned XY footprint corners rotated through
        # the fixture's own world yaw, then re-expressed in the robot
        # base XY frame.
        hw, hd = 0.5 * w_local, 0.5 * d_local
        corners_local = np.array(
            [[+hw, +hd], [+hw, -hd], [-hw, +hd], [-hw, -hd]],
            dtype=np.float64,
        )
        c, s = np.cos(fixture_rot_z), np.sin(fixture_rot_z)
        R_wf = np.array([[c, -s], [s, c]], dtype=np.float64)
        corners_world_xy = corners_local @ R_wf.T
        R_bw_xy = R_bw[:2, :2]
        corners_base_xy = corners_world_xy @ R_bw_xy.T
        y_half = float(np.max(np.abs(corners_base_xy[:, 1])))
        x_half = float(np.max(np.abs(corners_base_xy[:, 0])))

        # Absolute base-frame Y bounds (corners translated by the
        # fixture's body position).
        footprint_corners_base_xy = corners_base_xy + base_to_fixture_base[:2]
        y_base_min = float(np.min(footprint_corners_base_xy[:, 1]))
        y_base_max = float(np.max(footprint_corners_base_xy[:, 1]))

        try:
            bbox_pts = fixture.get_bbox_points()
            world_top_z = float(max(p[2] for p in bbox_pts))
        except Exception:
            world_top_z = float(fixture_world[2] + 0.5 * float(fixture_size[2]))

        eef_pos = np.asarray(env._reach_home_eef_pos, dtype=np.float64)
        return {
            "name": str(name),
            "class": cls_name,
            "world_pos": fixture_world.copy(),
            "y_half_extent_m": y_half,
            "x_half_extent_m": x_half,
            "world_top_z": world_top_z,
            "offaxis_y_base_m": abs(float(base_to_fixture_base[1])),
            "x_base_m": float(base_to_fixture_base[0]),
            "y_base_min": y_base_min,
            "y_base_max": y_base_max,
            "root_body_id": root_bid,
            "dist_to_eef_m": float(np.linalg.norm(fixture_world - eef_pos)),
        }

    def _pick_obstacle_fixture(self) -> dict:
        """Pick an allow-listed Fixture for the obstacle-reach task.

        Returns the EEF-closest fixture whose class is in
        ``self._obstacle_classes``. Does NOT filter on the fixture's
        CURRENT position — placement immediately follows.
        """
        env = self._env
        fixtures = getattr(env._env, "fixtures", None) or {}
        if not fixtures:
            raise RuntimeError(
                f"_pick_obstacle_fixture: env.fixtures is empty; obstacle_classes={list(self._obstacle_classes)}"
            )
        candidates: list[tuple[float, dict]] = []
        for name, fixture in fixtures.items():
            cls_name = type(fixture).__name__
            if cls_name not in self._obstacle_classes:
                continue
            if env._is_structural_fixture(fixture):
                continue
            meta = self._compute_obstacle_meta(name, fixture)
            if meta is None:
                continue
            if not (_OBSTACLE_REACH_MIN_Y_HALF_M <= meta["y_half_extent_m"] <= _OBSTACLE_REACH_MAX_Y_HALF_M):
                continue
            candidates.append((meta["dist_to_eef_m"], meta))
        if not candidates:
            available_classes = sorted({type(f).__name__ for f in fixtures.values()})
            raise RuntimeError(
                f"_pick_obstacle_fixture: no allow-listed fixture in scene for "
                f"seed={env._config.get('seed')!r}, "
                f"kitchen_split={env._kitchen_split!r}. "
                f"obstacle_classes={list(self._obstacle_classes)}, "
                f"available_classes={available_classes}"
            )
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def _place_obstacle_in_front_of_robot(
        self,
        obstacle_meta: dict,
        forward_clearance_from_eef_m: float = _OBSTACLE_REACH_PLACEMENT_FORWARD_M,
    ) -> dict:
        """Physically move the obstacle's static body so its NEAR FACE
        sits ``forward_clearance_from_eef_m`` ahead of the captured
        home EEF on the robot's Y=0 axis. Returns refreshed metadata.

        Robocasa fixtures are merged into the arena as static (no free
        joint) world-frame bodies; the runtime position lives in
        ``sim.model.body_pos[root_bid]``. We write the new world
        position there, call ``sim.forward()`` to propagate to
        ``sim.data.body_xpos`` and all derived geom transforms, then
        re-compute the metadata.

        Two-pass placement so the fixture's own X-extent (deep
        CoffeeMachines have body depth ~0.50 m) is accounted for:
        Trial 1 places the body centre at ``home_eef + clearance``,
        then Trial 2 reads the resulting x_half_extent and shifts
        forward so the NEAR face lands at the requested clearance.
        """
        env = self._env
        sim = env._env.sim
        model = sim.model
        root_bid = int(obstacle_meta["root_body_id"])
        base_pos_world, R_wb = env._get_base_world_pose()
        R_bw = R_wb.T

        home_eef_x_base = float((R_bw @ (np.asarray(env._reach_home_eef_pos, dtype=np.float64) - base_pos_world))[0])
        target_x_base = home_eef_x_base + forward_clearance_from_eef_m
        cur_world = np.asarray(sim.data.body_xpos[root_bid], dtype=np.float64)

        def _write_body(target_base_x: float) -> None:
            target_base_xyz = np.array([target_base_x, 0.0, 0.0], dtype=np.float64)
            target_world_xy = base_pos_world + R_wb @ target_base_xyz
            new_world = np.array(
                [target_world_xy[0], target_world_xy[1], cur_world[2]],
                dtype=np.float64,
            )
            model.body_pos[root_bid] = new_world
            sim.forward()

        _write_body(target_x_base)
        fixture = env._env.fixtures[obstacle_meta["name"]]
        meta = self._compute_obstacle_meta(obstacle_meta["name"], fixture)
        if meta is None:
            raise RuntimeError(
                f"_place_obstacle_in_front_of_robot: lost metadata for {obstacle_meta['name']!r} after placement"
            )

        x_half = float(meta.get("x_half_extent_m", 0.0))
        if x_half > 1e-3:
            target_x_base_2 = home_eef_x_base + forward_clearance_from_eef_m + x_half
            if abs(target_x_base_2 - target_x_base) > 1e-3:
                _write_body(target_x_base_2)
                meta = self._compute_obstacle_meta(obstacle_meta["name"], fixture)
                if meta is None:
                    raise RuntimeError(
                        f"_place_obstacle_in_front_of_robot: lost metadata for "
                        f"{obstacle_meta['name']!r} after second placement"
                    )
        return meta

    def _sample_obstacle_reach_targets(
        self,
        seed,
        obstacle_meta: dict,
        validator=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``(start_pos_world, goal_pos_world)`` so the two
        endpoints sit on OPPOSITE Y_base sides of ``obstacle_meta`` in
        a strict L/R band derived from the fixture's bbox.

        X and Z are anchored at the fixture body centre (with an X cap
        at ``home_eef + 0.15 m`` so very deep fixtures don't push
        endpoints past the Panda's converged reach). Y is the only
        axis that differs between left / right.
        """
        env = self._env
        rng = np.random.RandomState(int(seed))
        base_pos_world, R_wb = env._get_base_world_pose()
        R_bw = R_wb.T
        fixture_world = np.asarray(obstacle_meta["world_pos"], dtype=np.float64)
        fixture_base = R_bw @ (fixture_world - base_pos_world)
        home_eef_base = R_bw @ (np.asarray(env._reach_home_eef_pos, dtype=np.float64) - base_pos_world)

        y_half = float(obstacle_meta["y_half_extent_m"])
        if "y_base_min" in obstacle_meta and "y_base_max" in obstacle_meta:
            y_base_min = float(obstacle_meta["y_base_min"])
            y_base_max = float(obstacle_meta["y_base_max"])
        else:
            y_base_min = float(fixture_base[1]) - y_half
            y_base_max = float(fixture_base[1]) + y_half
        clearance = self._obstacle_clearance_m
        anchor_world_z = float(fixture_world[2])
        anchor_world = np.array([fixture_world[0], fixture_world[1], anchor_world_z], dtype=np.float64)
        anchor_base = R_bw @ (anchor_world - base_pos_world)
        sigma = self._obstacle_lateral_jitter_std
        z_tol = _OBSTACLE_REACH_ENDPOINT_Z_TOL_M
        midpoint_xy_tol = _OBSTACLE_REACH_MIDPOINT_XY_TOL_M

        # Cap the X anchor at the Panda's reachable forward distance
        # from home. Deep fixtures' body centres can sit past this.
        max_anchor_x = float(home_eef_base[0]) + _OBSTACLE_REACH_MAX_X_ANCHOR_FROM_EEF_M
        fixture_anchor_x_base = min(float(anchor_base[0]), max_anchor_x)
        left_mid_base = np.array(
            [fixture_anchor_x_base, y_base_max + clearance, anchor_base[2]],
            dtype=np.float64,
        )
        right_mid_base = np.array(
            [fixture_anchor_x_base, y_base_min - clearance, anchor_base[2]],
            dtype=np.float64,
        )

        d_min = env._reach_min_distance
        d_max = env._reach_max_distance
        tol = 1e-9
        ws_half = _OBSTACLE_REACH_WORKSPACE_HALF_EXTENT_M

        def _within_workspace(endpoint_base: np.ndarray) -> bool:
            return bool(np.all(np.abs(endpoint_base - home_eef_base) <= ws_half))

        validation_attempts = 0
        for _attempt in range(_OBSTACLE_REACH_SAMPLE_MAX_TRIES):
            start_is_left = rng.uniform() < 0.5
            start_mid = left_mid_base if start_is_left else right_mid_base
            goal_mid = right_mid_base if start_is_left else left_mid_base
            start_base = start_mid + rng.normal(scale=sigma, size=3)
            goal_base = goal_mid + rng.normal(scale=sigma, size=3)

            if abs(start_base[2] - anchor_base[2]) > z_tol or abs(goal_base[2] - anchor_base[2]) > z_tol:
                continue

            left_endpoint, right_endpoint = (start_base, goal_base) if start_is_left else (goal_base, start_base)
            if left_endpoint[1] < y_base_max + clearance:
                continue
            if right_endpoint[1] > y_base_min - clearance:
                continue

            midpoint_y_base = 0.5 * (start_base[1] + goal_base[1])
            if abs(midpoint_y_base - fixture_base[1]) > midpoint_xy_tol:
                continue

            if not _within_workspace(start_base) or not _within_workspace(goal_base):
                continue

            start_world = base_pos_world + R_wb @ start_base
            goal_world = base_pos_world + R_wb @ goal_base

            d0 = float(np.linalg.norm(goal_world - start_world))
            if not (d_min - tol <= d0 <= d_max + tol):
                continue

            if validator is None or validator(start_world, goal_world):
                return start_world, goal_world
            validation_attempts += 1
            if validation_attempts >= _OBSTACLE_REACH_VALIDATION_MAX_TRIES:
                raise RuntimeError(
                    f"_sample_obstacle_reach_targets: validator rejected "
                    f"{_OBSTACLE_REACH_VALIDATION_MAX_TRIES} consecutive "
                    f"candidates (seed={seed}, obstacle={obstacle_meta['name']}, "
                    f"clearance={clearance:.3f} m, sigma={sigma:.3f} m). "
                    f"The scene is structurally infeasible for the rotation invariant."
                )

        raise RuntimeError(
            f"_sample_obstacle_reach_targets: no accepted draw after "
            f"{_OBSTACLE_REACH_SAMPLE_MAX_TRIES} tries "
            f"(seed={seed}, obstacle={obstacle_meta['name']}, "
            f"y_half={y_half:.3f} m, clearance={clearance:.3f} m, "
            f"anchor_world_z={anchor_world_z:.3f}, "
            f"d_min={d_min:.3f}, d_max={d_max:.3f}). Either widen "
            f"[reach_min_distance, reach_max_distance], shrink "
            f"obstacle_clearance_m, shrink obstacle_lateral_jitter_std, "
            f"or loosen the Z / midpoint tolerances."
        )


# ---------------------------------------------------------------------------
# Object-reach wrapper subclass
# ---------------------------------------------------------------------------


# Body name used by :class:`ObjectReachKitchen._get_obj_cfgs` for the single
# free object it spawns. Consumers index ``env.obj_body_id`` and
# ``env.objects`` by this fixed key, regardless of which registry entry
# (banana / mug / can / ...) is currently active.
_OBJECT_REACH_BODY_NAME = "obj"

# Fixed ``task_name`` for the inner :class:`RoboCasaEnv` whenever an
# :class:`ObjectReachBlockingRoboCasaEnv` wraps it.
_OBJECT_REACH_INNER_TASK_NAME = "ObjectReachKitchen"

# World-frame Euclidean tolerance (m) at which the carried object is
# considered to have "reached" the target.
_OBJECT_REACH_SUCCESS_TOL_M: float = 0.02

# Grasp-sequence tunables (all metres / rad / OSC sub-steps).
_GRASP_APPROACH_HEIGHT_M: float = 0.10
_GRASP_DESCEND_DEPTH_M: float = 0.10  # below object center
_GRASP_LIFT_HEIGHT_M: float = 0.10
_GRASP_PRE_OPEN_SUBSTEPS: int = 10
_GRASP_CLOSE_MIN_STEPS: int = 20
_GRASP_VERIFY_THRESHOLD_M: float = 0.035

# Per-phase ``_drive_eef_with_gripper`` tunables (the drive helper accepts
# these as kwargs). The CLOSE phase intentionally has no ``max_steps`` so
# callers can derive it from the per-instance grasp-substeps knob.
_GRASP_DRIVE_APPROACH: dict = dict(max_steps=300, tol=0.005, rot_tol=0.05, max_step_m=0.02, max_step_rad=0.10)
_GRASP_DRIVE_DESCEND: dict = dict(max_steps=300, tol=0.002, rot_tol=0.05, max_step_m=0.003, max_step_rad=0.05)
_GRASP_DRIVE_CLOSE: dict = dict(tol=0.0005, rot_tol=0.05, max_step_m=0.001, max_step_rad=0.02)
_GRASP_DRIVE_LIFT: dict = dict(max_steps=200, tol=0.005, rot_tol=0.05, max_step_m=0.02, max_step_rad=0.10)

# Per-substep XYZ correction cap (m) used by the carry phase.
_CARRY_MAX_STEP_M: float = 0.02

# Object-reach rotation invariance & step-time tolerances (mirror the
# corresponding REACH-side constants in ``robot_env.py``):
#
# * ``_OBJECT_REACH_ROT_INVARIANCE_TOL`` — max geodesic SO(3) drift
#   (rad) tolerated by the wrapper's rotation-aware target validator
#   (:meth:`ObjectReachBlockingRoboCasaEnv._validate_rotation_for_object_reach_targets`).
#   Same value as :data:`robot_env._REACH_ROT_INVARIANCE_TOL` (= 0.05
#   rad); a (start, goal) pair that drives the wrist beyond this from
#   the post-grasp home quat is silently re-sampled.
# * ``_OBJECT_REACH_VALIDATION_MAX_TRIES`` — hard cap on rotation-
#   rejection resampling attempts before the wrapper raises
#   ``RuntimeError``.
# * ``_OBJECT_REACH_EEF_ROT_TOL_RAD`` — tolerance used by
#   ``test_eef_rotation_matches_between_init_and_goal``.
# * ``_OBJECT_REACH_STEP_ROT_TOL_RAD`` — tolerance used by
#   ``test_eef_rotation_held_across_steps``.
_OBJECT_REACH_ROT_INVARIANCE_TOL: float = 0.05
_OBJECT_REACH_VALIDATION_MAX_TRIES: int = 100
_OBJECT_REACH_EEF_ROT_TOL_RAD: float = 0.08
_OBJECT_REACH_STEP_ROT_TOL_RAD: float = 0.08

# Distance (m) between the object center and the EEF site above which we
# consider the object to have FALLEN from the gripper. The grasp verifier
# uses :data:`_GRASP_VERIFY_THRESHOLD_M = 0.035 m` as the "still held"
# bound right after lift; we use a more permissive 0.10 m for the runtime
# drop detector so brief tracking gaps during fast motion don't trigger
# false positives. If you observe spurious drops during normal carry,
# loosen toward 0.15 m; if you want a stricter "the object is anywhere
# other than between the fingers" detector, tighten to ~0.05 m.
_OBJECT_REACH_DROP_THRESHOLD_M: float = 0.10

# Max acceptable z drop (m) of the OBJECT between its spawn pose and
# the end of the grasp+lift sequence. A normal grasp+lift produces
# ``post_grasp_z >= spawn_z`` (the object was picked up, so it never
# moves DOWN). A "fallen-pose recovery" (the previous attempt's
# descend knocked the object off the counter into a fixture and the
# retry re-grasped it from inside the fixture) produces
# ``post_grasp_z << spawn_z`` because the object fell before being
# re-grasped. If the drop exceeds this threshold the wrapper rejects
# the grasp and returns ``grasp_failed=True`` instead of carrying a
# fallen object. Use object-relative drop rather than EEF-relative-
# to-home so the heuristic doesn't false-trigger on kitchens with a
# low counter (where the post-lift EEF naturally sits well below the
# robot's home pose).
_GRASP_FROM_FALLEN_REJECT_DROP_M: float = 0.10

# Max per-WM-step change in the gripper "closedness" target. The wrapper
# interprets the planner-side gripper signal in [-1, +1] as a DELTA in
# closedness (negative = open more, positive = close more) and integrates
# it into a persistent absolute target ``self._gripper_target`` clipped to
# ``[-1, +1]`` that is forwarded to the OSC every sub-step:
#
#     target ← clip(target + planner_cmd * _OBJECT_REACH_GRIPPER_DELTA_PER_STEP, -1, +1)
#
# With ``_gripper_target = +1.0`` after :meth:`reset` (grasp-held), any
# positive planner command saturates at the clip ceiling (stays closed)
# while any negative command opens proportionally.
#
# Value chosen so that ``planner_cmd = -1.0`` traverses the full
# closedness range ``[+1 → -1]`` in a single WM step (and conversely
# ``+1.0`` traverses ``[-1 → +1]``). This matches the literal
# "delta in closedness, magnitude is corresponding" semantic the
# wrapper exposes to the planner: -0.5 opens halfway, -0.25 by a
# quarter, -1.0 fully opens. Episodes are short (default
# ``--total_steps 5``) so a smaller per-step rate (e.g. 0.1) would
# make the gripper effectively un-controllable by the planner within
# one rollout.
_OBJECT_REACH_GRIPPER_DELTA_PER_STEP: float = 2.0


class ObjectReachBlockingRoboCasaEnv(ReachBlockingRoboCasaEnv):
    """Blocking wrapper for the **object-reach** task.

    Variant of :class:`ReachBlockingRoboCasaEnv` where the robot starts
    already grasping a single graspable object (default: ``"banana"``)
    spawned on a reachable counter by :class:`ObjectReachKitchen`. The
    reward is driven by the world-frame distance from the carried object
    to a randomly sampled target world position (rather than by EEF
    distance to a sampled EEF target as in plain reach).

    ``reset(mode="init")`` caches the object's start pose;
    ``reset(mode="goal")`` caches the object's target pose AND computes
    ``d_0 = ||obj_target - obj_start||``. The dense reward emitted from
    :meth:`step` is then ``r = max(0, 1 - dist/d_0)`` and
    ``info['success']`` is flagged once the carried object is within
    :data:`_OBJECT_REACH_SUCCESS_TOL` of the target.

    The inner :class:`RoboCasaEnv` is held in ``reach_task=False`` mode;
    we inherit from :class:`ReachBlockingRoboCasaEnv` for the inheritance
    hierarchy only — the inner env's reach reset branch is explicitly
    disabled and all clear / spawn / grasp / carry orchestration happens
    here on the wrapper.
    """

    _WRAPPER_PRIVATE_ATTRS = ReachBlockingRoboCasaEnv._WRAPPER_PRIVATE_ATTRS + (
        "_object_reach_object_name",
        "_object_reach_object_spec",
        "_object_reach_task_seed",
        "_object_reach_max_distance",
        "_object_reach_min_distance",
        "_object_reach_max_pair_distance",
        "_object_reach_clear_radius",
        "_object_reach_grasp_substeps",
        "_object_reach_max_retries",
        "_object_reach_kinematic_carry",
        "_object_reach_rotation_validation",
        "_object_reach_home_eef_pos",
        "_object_reach_home_eef_quat",
        "_object_reach_home_eef_quat_site_wxyz",
        "_object_reach_start_world",
        "_object_reach_target_world",
        "_object_reach_d0",
        "_object_reach_d0_warned",
        "_object_reach_targets_cache",
        "_object_reach_base_pose_cache",
        "_last_drive_substeps",
        "_gripper_target",
    )

    # World-frame Euclidean tolerance (m) at which the carried object is
    # considered to have "reached" the target. Class alias of the module-
    # level :data:`_OBJECT_REACH_SUCCESS_TOL_M` for back-compat.
    _OBJECT_REACH_SUCCESS_TOL: float = _OBJECT_REACH_SUCCESS_TOL_M

    # NOTE: per-object approach biases (``approach_offset_base_xyz``,
    # ``per_kitchen_approach_offset_delta``, ``per_seed_approach_offset_delta``)
    # live in :data:`OBJECT_REACH_REGISTRY` per object — see
    # ``object_reach_kitchen.py``. The grasp pipeline reads them via
    # :meth:`_resolve_object_spec`. Defaults are
    # ``approach_offset_base_xyz=(0.0, 0.0, 0.0)`` (no bias) and the two
    # delta dicts default to ``{}`` (no kitchen/seed-specific tuning).

    # Strict top-down EEF orientation in WORLD frame, encoded as a wxyz
    # quaternion. ``[0, 1, 0, 0]`` is a 180° rotation about world +X,
    # which puts the PandaOmron tool z-axis along ``-world_z`` (fingers
    # pointing straight down) regardless of the home pose's residual tilt.
    # The rotation about world z (gripper yaw) is NOT pinned by this quat
    # — the OSC controller will track yaw to whatever value satisfies the
    # 180°-about-X axis alignment, which is the only constraint we need
    # for a top-down grasp where the object is symmetric in plan view
    # (banana, mug, can, ...).
    _OBJECT_REACH_TOPDOWN_QUAT_WXYZ: tuple[float, ...] = (0.0, 1.0, 0.0, 0.0)

    def __init__(
        self,
        env: RoboCasaEnv,
        *,
        object_reach_object_name: str = "banana",
        object_reach_task_seed: int | None = None,
        object_reach_max_distance: float = 0.15,
        object_reach_min_distance: float = 0.0,
        object_reach_max_pair_distance: float | None = None,
        object_reach_clear_radius: float = 1.0,
        object_reach_grasp_substeps: int = 10,
        object_reach_max_retries: int = 3,
        object_reach_kinematic_carry: bool = False,
        object_reach_rotation_validation: bool = False,
        pos_tol: float = _GOAL_BLOCKING_TOL,
        max_steps: int = _GOAL_BLOCKING_MAX_STEPS,
        apply_rotation_in_first_substep: bool = True,
    ):
        # Skip ``ReachBlockingRoboCasaEnv.__init__`` (which would force
        # the inner env's ``_reach_task = True`` and activate the inner
        # reach reset branch we explicitly do NOT want here). Initialise
        # the base wrapper directly.
        BlockingRoboCasaEnv.__init__(
            self,
            env,
            pos_tol=pos_tol,
            max_steps=max_steps,
            apply_rotation_in_first_substep=apply_rotation_in_first_substep,
        )
        # Belt-and-braces: the inner env must NOT run its own reach-task
        # reset branch; the wrapper drives clear/spawn/grasp/carry.
        env._reach_task = False

        self._object_reach_object_name = str(object_reach_object_name)
        self._object_reach_task_seed = int(object_reach_task_seed) if object_reach_task_seed is not None else None
        self._object_reach_max_distance = float(object_reach_max_distance)
        # Optional start->goal distance band, enforced at RESET via rejection
        # sampling. Default OFF (min=0.0, no upper cap) == legacy behaviour:
        # the sampled pair is accepted as-is and the RNG stream is byte-
        # identical, so in-flight jobs are unaffected if they restart. New
        # submissions opt in via the CLI / env-var passthrough.
        self._object_reach_min_distance = float(object_reach_min_distance)
        self._object_reach_max_pair_distance = (
            float(object_reach_max_pair_distance) if object_reach_max_pair_distance is not None else None
        )
        assert self._object_reach_min_distance >= 0.0
        if self._object_reach_max_pair_distance is not None:
            assert self._object_reach_max_pair_distance >= self._object_reach_min_distance, (
                "object_reach_max_pair_distance must be >= object_reach_min_distance"
            )
        assert self._object_reach_min_distance <= 2.0 * self._object_reach_max_distance, (
            f"object_reach_min_distance={self._object_reach_min_distance} exceeds "
            f"2*radius={2.0 * self._object_reach_max_distance}; band unsatisfiable"
        )
        self._object_reach_clear_radius = float(object_reach_clear_radius)
        self._object_reach_grasp_substeps = int(object_reach_grasp_substeps)
        self._object_reach_max_retries = int(object_reach_max_retries)
        self._object_reach_kinematic_carry = bool(object_reach_kinematic_carry)
        # Whether to run the per-candidate rotation-aware rejection
        # sampler (REACH fix #5 ported). OFF by default for OBJECT_REACH
        # because:
        #   * the sampling ball is small (max_distance=0.15 m, vs.
        #     REACH's 0.35 m), so wrist drift over the carry is small;
        #   * the carry-time rotation hold (REACH fixes #4+#7) and the
        #     per-step orientation hold inside :meth:`step` (REACH fix
        #     #6) together already keep the wrist locked to the
        #     post-grasp home quat — the validator is overkill except
        #     for pathological kitchen / object combinations.
        # Set ``object_reach_rotation_validation=True`` if a regression
        # surfaces a hard kitchen that drifts the wrist mid-carry.
        self._object_reach_rotation_validation = bool(object_reach_rotation_validation)

        self._object_reach_object_spec = None
        self._object_reach_home_eef_pos = None
        self._object_reach_home_eef_quat = None
        self._object_reach_home_eef_quat_site_wxyz = None
        self._object_reach_start_world = None
        self._object_reach_target_world = None
        self._object_reach_d0 = 0.0
        self._object_reach_d0_warned = False

        # Validated ``(start_eef, goal_eef)`` cache keyed by
        # ``(scene_seed, object_reach_task_seed)``. Mirrors the per-
        # ``effective_reach_seed`` cache in :class:`RoboCasaEnv` so the
        # rotation-invariance validator only runs on the first reset of
        # each episode (the second reset hits the cache and pays zero
        # validation cost).
        self._object_reach_targets_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

        # Robot-base pose cache (parallels :attr:`RoboCasaEnv._reach_base_pose_cache`).
        # Keyed by scene seed; the first reset captures the post-rebuild base
        # pose, every subsequent reset re-injects it into
        # ``self._env._env._ep_meta`` BEFORE the inner reset so the
        # robocasa ``set_robot_base()`` collision-retry loop doesn't
        # micro-shift the base across episodes.
        self._object_reach_base_pose_cache: dict[int, tuple[list, list]] = {}

        # Drive-helper accounting: ``_drive_eef_with_gripper`` writes the
        # number of OSC sub-steps it consumed into this attr so the
        # wrapper's :meth:`step` override can report a faithful
        # ``info["sim_steps"]``.
        self._last_drive_substeps = 0

        # Persistent absolute gripper target ``∈ [-1, +1]`` used by the
        # delta-accumulator semantic in :meth:`step` (planner-side
        # gripper signal is interpreted as a DELTA, integrated into
        # this state, and forwarded to the OSC every sub-step). Init
        # to ``+1.0`` (closed) so the first step after :meth:`reset`
        # holds the carry-grasp until the planner explicitly opens.
        # See :data:`_OBJECT_REACH_GRIPPER_DELTA_PER_STEP`.
        self._gripper_target = +1.0

        # Pin the inner kitchen task to ``ObjectReachKitchen`` and tell
        # that class which object to spawn via the class-level default.
        # The next inner-env reset triggers a rebuild via
        # ``_maybe_recreate_env`` if ``task_name`` actually changed.
        ObjectReachKitchen.DEFAULT_OBJECT_NAME = self._object_reach_object_name
        env._config["task_name"] = _OBJECT_REACH_INNER_TASK_NAME

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _resolve_object_spec(self) -> dict:
        """Look up the active object's registry entry (cached).

        Raises ``KeyError`` (with the available keys listed) if
        ``self._object_reach_object_name`` is not present in
        :data:`OBJECT_REACH_REGISTRY`.
        """
        if self._object_reach_object_spec is not None:
            return self._object_reach_object_spec
        name = self._object_reach_object_name
        if name not in OBJECT_REACH_REGISTRY:
            raise KeyError(
                f"object_reach_object_name={name!r} not in OBJECT_REACH_REGISTRY; "
                f"available: {sorted(OBJECT_REACH_REGISTRY)}"
            )
        self._object_reach_object_spec = OBJECT_REACH_REGISTRY[name]
        return self._object_reach_object_spec

    def _get_object_world_pos(self) -> np.ndarray:
        """Return the world-frame XYZ of the kitchen task's spawned object.

        Looks up :data:`_OBJECT_REACH_BODY_NAME` (= ``"obj"``) on the
        inner robosuite env. Raises ``KeyError`` (with the available
        ``obj_body_id`` keys listed) on miss.
        """
        inner = self._env._env
        body_id_map = getattr(inner, "obj_body_id", None) or {}
        if _OBJECT_REACH_BODY_NAME not in body_id_map:
            raise KeyError(
                f"ObjectReach kitchen task did not spawn body "
                f"{_OBJECT_REACH_BODY_NAME!r}; available obj_body_id "
                f"keys: {sorted(body_id_map)}"
            )
        bid = body_id_map[_OBJECT_REACH_BODY_NAME]
        return np.asarray(inner.sim.data.body_xpos[bid], dtype=np.float64).copy()

    def _snapshot_site_quat_wxyz(self) -> np.ndarray:
        """Read the EE grip-site world rotation from ``sim.data.site_xmat``
        and return it as a wxyz quaternion.

        This is the rotation the drive helper internally compares against
        (via :attr:`sim.data.site_xmat[eef_site_id]`) when it has a
        ``target_quat_wxyz`` to drive toward. The grip-site rotation
        has a fixed offset (~90° about world Z on PandaOmron) from the
        gripper body rotation returned by ``robot0_eef_quat`` /
        :meth:`RoboCasaEnv._get_eef_quat_world`, so the wrapper must
        snapshot BOTH if it wants to (a) preserve the body-frame quat
        for tests / fixture / golden capture AND (b) drive the wrist
        to a consistent rotation in the OSC loop.

        Mirrors the :func:`_eef_site_id` resolution dance from
        :meth:`_drive_eef_with_gripper` (one-arm SingleArm or
        per-arm-keyed dict).
        """
        inner_env = self._env._env
        robot0 = inner_env.robots[0]
        eef_site_id_attr = robot0.eef_site_id
        if isinstance(eef_site_id_attr, dict):
            arm_key = next(iter(eef_site_id_attr))
            eef_site_id = int(eef_site_id_attr[arm_key])
        else:
            eef_site_id = int(eef_site_id_attr)
        mat = np.asarray(inner_env.sim.data.site_xmat[eef_site_id], dtype=np.float64).reshape(3, 3)
        quat_xyzw = Rotation.from_matrix(mat).as_quat()
        # xyzw → wxyz
        return np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )

    def _sample_object_reach_targets(
        self,
        validator=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``(start_eef_world, goal_eef_world)`` deterministically.

        Same uniform-in-ball algorithm as
        :meth:`RoboCasaEnv._sample_reach_targets` (``r = R * u**(1/3)``)
        but anchored on the wrapper-owned home EEF and seeded with
        ``self._object_reach_task_seed`` (or the scene seed if that
        knob is ``None``). Sampling order matches the reach convention:
        GOAL is consumed first, START second — so the same task-seed
        produces the same (start, goal) pair across mode='init' and
        mode='goal' resets.

        :param validator: optional callable ``(start, goal) -> bool``.
            If provided, the helper resamples until the validator
            accepts the pair (or :data:`_OBJECT_REACH_VALIDATION_MAX_TRIES`
            attempts are exhausted, in which case ``RuntimeError`` is
            raised). The validator is used by reset() to inject
            rotation-aware rejection so the pair doesn't induce a
            wrist-twist mid-carry — mirrors REACH fix #5.
        """
        seed = self._object_reach_task_seed
        if seed is None:
            seed = int(self._env._config.get("seed", 0) or 0)
        rng = np.random.RandomState(int(seed))
        anchor = self._object_reach_home_eef_pos
        radius = self._object_reach_max_distance
        d_min = self._object_reach_min_distance
        d_max = self._object_reach_max_pair_distance

        def _sample_in_ball():
            d = rng.normal(size=3)
            d = d / (np.linalg.norm(d) + 1e-12)
            r = radius * (rng.uniform(0.0, 1.0) ** (1.0 / 3.0))
            return d * r

        def _distance_ok(start_w, goal_w):
            # Start->goal distance band. Default (d_min=0.0, d_max=None)
            # accepts everything, so legacy runs are unaffected.
            dd = float(np.linalg.norm(goal_w - start_w))
            if dd < d_min:
                return False
            if d_max is not None and dd > d_max:
                return False
            return True

        # Loop (consuming extra RNG draws) only when some rejection is
        # active. With the band OFF and no rotation validator this runs
        # exactly once, preserving the legacy (start, goal) byte-for-byte.
        band_active = (d_min > 0.0) or (d_max is not None)
        needs_loop = band_active or (validator is not None)
        max_tries = _OBJECT_REACH_VALIDATION_MAX_TRIES if needs_loop else 1
        last_pair: tuple[np.ndarray, np.ndarray] | None = None
        for _attempt in range(max_tries):
            goal_offset = _sample_in_ball()
            start_offset = _sample_in_ball()
            start_world = (anchor + start_offset).astype(np.float64)
            goal_world = (anchor + goal_offset).astype(np.float64)
            last_pair = (start_world, goal_world)
            if _distance_ok(start_world, goal_world) and (validator is None or validator(start_world, goal_world)):
                return start_world, goal_world
        raise RuntimeError(
            f"object_reach target sampler exhausted {max_tries} attempts for "
            f"seed={seed}, home_eef={anchor.tolist()}, max_distance={radius}, "
            f"min_distance={d_min}, max_pair_distance={d_max}; widen the radius, "
            f"loosen the distance band, or loosen _OBJECT_REACH_ROT_INVARIANCE_TOL."
        )

    def _validate_rotation_for_object_reach_targets(
        self,
        start_world: np.ndarray,
        goal_world: np.ndarray,
        tol: float = _OBJECT_REACH_ROT_INVARIANCE_TOL,
    ) -> bool:
        """Per-candidate rotation-invariance check for a sampled
        ``(start_eef, goal_eef)`` pair.

        Mirrors :meth:`RoboCasaEnv._validate_rotation_for_targets` but
        runs **with the object already grasped** (``gripper_command=+1``,
        ``kinematic_carry=self._object_reach_kinematic_carry``,
        ``target_quat_wxyz=self._object_reach_home_eef_quat``) so the
        validator measures the wrist drift the planner would actually
        experience during a carry.

        Algorithm: snapshot ``sim.get_state()``, drive to ``start`` with
        gripper closed + topdown quat target, capture R_start via the
        quat-native helper, restore state; drive to ``goal``, capture
        R_goal, restore state; accept iff the geodesic SO(3) drift
        between R_home and each of (R_start, R_goal) is ≤ ``tol``.

        Returns False on any drive failure (the helper returns
        ``residual >= drive_tol`` if the OSC fails to converge) — that's
        a strong signal the candidate pair is unreachable with the
        object held and should be re-sampled.
        """
        if self._object_reach_home_eef_quat is None:
            # Validator must be invoked AFTER the grasp+lift snapshot;
            # if not, we have no reference and must conservatively
            # accept (lets reset() fall back to position-only sampling).
            return True
        inner = self._env
        home_quat_xyzw = np.asarray(self._object_reach_home_eef_quat, dtype=np.float64)
        # Convert xyzw → wxyz for the drive helper's API.
        home_quat_wxyz = np.array(
            [home_quat_xyzw[3], home_quat_xyzw[0], home_quat_xyzw[1], home_quat_xyzw[2]],
            dtype=np.float64,
        )
        R_home = Rotation.from_quat(home_quat_xyzw)

        sim = inner._env.sim
        snap = sim.get_state()

        def _drive_and_measure(target_world: np.ndarray) -> float:
            residual = self._drive_eef_with_gripper(
                target_world_pos=target_world,
                gripper_command=+1.0,
                max_steps=_REACH_DRIVE_MAX_STEPS,
                tol=_REACH_DRIVE_TOL,
                max_step_m=_CARRY_MAX_STEP_M,
                target_quat_wxyz=home_quat_wxyz,
                rot_tol=_GOAL_RETURN_HOME_ROT_TOL,
                kinematic_carry=self._object_reach_kinematic_carry,
            )
            if residual > 10 * _REACH_DRIVE_TOL:
                # Drive failed to converge — unreachable.
                return float("inf")
            cur_quat_xyzw = inner._get_eef_quat_world(q_ref=home_quat_xyzw)
            R_now = Rotation.from_quat(cur_quat_xyzw)
            return float(np.linalg.norm((R_home * R_now.inv()).as_rotvec()))

        try:
            drift_start = _drive_and_measure(np.asarray(start_world, dtype=np.float64))
            sim.set_state(snap)
            sim.forward()
            drift_goal = _drive_and_measure(np.asarray(goal_world, dtype=np.float64))
        finally:
            sim.set_state(snap)
            sim.forward()

        return drift_start <= tol and drift_goal <= tol

    # ------------------------------------------------------------------
    # OSC drive primitives
    # ------------------------------------------------------------------

    def _drive_eef_with_gripper(
        self,
        target_world_pos: np.ndarray,
        gripper_command: float,
        max_steps: int = _REACH_DRIVE_MAX_STEPS,
        tol: float = _REACH_DRIVE_TOL,
        max_step_m: float | None = None,
        kinematic_carry: bool = False,
        target_quat_wxyz: np.ndarray | None = None,
        rot_tol: float = 0.05,
        max_step_rad: float | None = None,
        force_run: bool = False,
        workspace_center: np.ndarray | None = None,
        workspace_radius: float | None = None,
        joint_margin_check: bool = False,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> float:
        """Closed-loop blocking drive of the EEF to a world-frame target,
        holding the gripper command at ``gripper_command`` every sub-step.

        Mirrors :meth:`RoboCasaEnv._drive_eef_to_world_target` (which
        always sends ``corrective[6] = 0``) but exposes the gripper slot
        so the carry phase can keep the fingers closed (+1) while the
        approach phase keeps them open (-1). ``max_step_m`` optionally
        clamps the per-substep XYZ correction magnitude so a held
        object isn't knocked loose by an aggressive 5 cm OSC step. If
        ``kinematic_carry`` is True, also snaps the held object's free
        joint qpos to the EEF after every sub-step.

        When ``target_quat_wxyz`` is provided, the helper also drives the
        EEF orientation to that target quaternion every sub-step using
        the **quaternion-native** EE rotation read
        (:meth:`RoboCasaEnv._get_eef_quat_world` with sign-continuity
        against ``target_quat_wxyz``) — mirrors REACH fix #7, prevents
        the gimbal-lock-driven wrist flip that the matrix→Rotation path
        would silently allow at pitch ≈ ±π/2. The rotation error is
        computed in the world frame as
        ``R_err_world = R_target * R_current.T`` and re-expressed in the
        controller's input frame (base) by ``axangle_base = R_bw @
        axangle_world`` — the same convention used by
        :meth:`RoboCasaEnv._run_expert_to_goal` for the return-home
        tracker. ``max_step_rad`` clamps the per-substep axis-angle
        magnitude (rad). ``rot_tol`` is the convergence threshold (rad,
        axis-angle magnitude). When ``target_quat_wxyz`` is None the
        rotation slots are zero-filled — preserving the historical
        position-only behaviour for callers that don't pass the new
        kwargs (back-compat).

        ``workspace_center`` + ``workspace_radius`` (both world-frame,
        meters) optionally project the per-sub-step XYZ command so the
        commanded NEXT EE position stays inside a sphere of that
        radius around the centre. Mirrors :meth:`RoboCasaEnv._reach_apply_orientation_hold`'s
        workspace-bound branch (REACH fix #6 step 0) — it is the only
        mechanism that reliably prevents the elbow-fully-straight
        singularity the planner used to drive the arm into. Defaults
        to no projection.

        ``joint_margin_check`` (bool) optionally zeroes the per-sub-step
        XYZ command when any arm joint is within
        :data:`_REACH_JOINT_LIMIT_MARGIN_RAD` of its physical limit;
        the rotation corrective still fires so the wrist gets pulled
        back to ``target_quat_wxyz`` and the offending joint mechanically
        relaxes. Mirrors REACH fix #6 step 1b. Off by default.

        ``force_run=True`` disables the position/orientation convergence
        early-exit so the loop runs the full ``max_steps`` no matter
        what — needed by the close phase, which anchors the drive at
        the *current* EEF pose (so position residual is ~0 on iteration
        zero) and relies on the loop dispatching the +1 gripper command
        every sub-step.

        Returns the final XYZ residual (m). Also writes the realised
        sub-step count into ``self._last_drive_substeps`` so callers
        (e.g. :meth:`step`) can report a faithful ``info["sim_steps"]``.
        """
        target_world_pos = np.asarray(target_world_pos, dtype=np.float64)
        inner = self._env
        R_wb = inner._get_base_world_R()
        R_bw = R_wb.T

        # Cache direct sim refs once so the inner loop bypasses
        # ``_get_observations(force_update=True)`` (the recompute-full-obs
        # path is per-substep expensive). The grip-site xpos / xmat live
        # in ``sim.data.site_xpos`` / ``sim.data.site_xmat`` and update
        # automatically as ``inner._env.step(...)`` advances physics.
        inner_env = inner._env
        sim = inner_env.sim
        robot0 = inner_env.robots[0]
        eef_site_id_attr = robot0.eef_site_id
        if isinstance(eef_site_id_attr, dict):
            # robosuite >= 1.5: per-arm-keyed dict (e.g. ``{"right": id}``
            # for PandaOmron). The first key is the only arm in our
            # single-arm setup.
            arm_key = next(iter(eef_site_id_attr))
            eef_site_id = int(eef_site_id_attr[arm_key])
        else:
            eef_site_id = int(eef_site_id_attr)

        target_R_world = None
        if target_quat_wxyz is not None:
            tq = np.asarray(target_quat_wxyz, dtype=np.float64).reshape(4)
            # scipy expects xyzw; we accept wxyz for explicitness with
            # the conventional top-down quat ``[0, 1, 0, 0]``.
            target_R_world = Rotation.from_quat(np.array([tq[1], tq[2], tq[3], tq[0]], dtype=np.float64))

        residual = float("inf")
        n_substeps = 0
        for _ in range(max_steps):
            n_substeps += 1
            eef_now_world = np.asarray(sim.data.site_xpos[eef_site_id], dtype=np.float64)
            err_world = target_world_pos - eef_now_world
            residual = float(np.linalg.norm(err_world))

            rot_err_world = np.zeros(3, dtype=np.float64)
            rot_err_norm = 0.0
            if target_R_world is not None:
                # Rotation feedback: read EE rotation MATRIX directly from
                # ``sim.data.site_xmat`` (NOT via ``_get_eef_quat_world``).
                # Matrices have no q ↔ -q hemisphere ambiguity so
                # sign-continuity correction is unnecessary here. The
                # quaternion-with-sign-continuity helper is reserved for
                # the per-step orientation hold (where the rotation
                # corrective is computed from raw quat comparisons that
                # CAN cross the hemisphere boundary). Using the matrix
                # read also avoids the expensive
                # ``_get_observations(force_update=True)`` call inside
                # the hot OSC loop — that recompute has measurable
                # per-sub-step side effects on robocasa's internal
                # state cache.
                cur_R_world = Rotation.from_matrix(
                    np.asarray(sim.data.site_xmat[eef_site_id], dtype=np.float64).reshape(3, 3)
                )
                rot_err_world = (target_R_world * cur_R_world.inv()).as_rotvec()
                rot_err_norm = float(np.linalg.norm(rot_err_world))

            if not force_run:
                pos_converged = residual < tol
                rot_converged = target_R_world is None or rot_err_norm < rot_tol
                if pos_converged and rot_converged:
                    break

            if max_step_m is not None and residual > max_step_m:
                err_world = err_world * (max_step_m / residual)
            if target_R_world is not None and max_step_rad is not None and rot_err_norm > max_step_rad:
                rot_err_world = rot_err_world * (max_step_rad / rot_err_norm)

            # Workspace bound (REACH fix #6 step 0): project the
            # proposed world-frame displacement so the NEXT EE position
            # stays inside the sphere of radius ``workspace_radius``
            # around ``workspace_center``. Same quadratic solver as
            # ``RoboCasaEnv._reach_apply_orientation_hold``.
            if workspace_center is not None and workspace_radius is not None:
                a = eef_now_world - np.asarray(workspace_center, dtype=np.float64)
                d = err_world
                R = float(workspace_radius)
                a_norm = float(np.linalg.norm(a))
                a_dot_d = float(np.dot(a, d))
                d_norm_sq = float(np.dot(d, d))
                if d_norm_sq < 1e-18:
                    scale = 1.0
                elif a_norm >= R - 1e-9:
                    scale = 0.0 if a_dot_d > 0 else 1.0
                else:
                    disc = a_dot_d * a_dot_d - d_norm_sq * (a_norm * a_norm - R * R)
                    if disc < 0:
                        scale = 1.0
                    else:
                        t_pos = (-a_dot_d + float(np.sqrt(disc))) / d_norm_sq
                        scale = max(0.0, min(1.0, t_pos))
                if scale < 1.0:
                    err_world = err_world * scale

            # Vertical (z) band clamp: keep the NEXT EE world z within
            # [z_min, z_max]. Used by the push task's per-step drive so the
            # wrist can descend to contact-push height but cannot fly up off
            # the object or drive down into the counter and jam (issue: "robot
            # goes too much up or too much down and gets stuck"). Adjusts only
            # the z component of the correction.
            if z_min is not None or z_max is not None:
                next_z = float(eef_now_world[2] + err_world[2])
                clamped_z = next_z
                if z_max is not None:
                    clamped_z = min(clamped_z, float(z_max))
                if z_min is not None:
                    clamped_z = max(clamped_z, float(z_min))
                err_world = np.array(err_world, dtype=np.float64)
                err_world[2] += clamped_z - next_z

            # Joint-singularity guard (REACH fix #6 step 1b): zero the
            # position command if any arm joint is within
            # ``_REACH_JOINT_LIMIT_MARGIN_RAD`` of its limit. The
            # rotation corrective still fires so the wrist gets pulled
            # back toward target_quat_wxyz, mechanically relaxing the
            # offending joint.
            if joint_margin_check:
                qpos, j_lower, j_upper = inner._get_arm_joint_pos_and_limits()
                joint_margin = float(np.min(np.minimum(qpos - j_lower, j_upper - qpos)))
                if joint_margin < _REACH_JOINT_LIMIT_MARGIN_RAD:
                    err_world = np.zeros(3, dtype=np.float64)

            pos_err_base = R_bw @ err_world
            if target_R_world is not None:
                # Axis-angle vector transforms world→base by left-mul
                # with R_bw (axis lives in space, angle is invariant).
                rot_err_base = R_bw @ rot_err_world
                full = _goal_prepare_axisangle(pos_err_base, rot_err_base, gripper=float(gripper_command))
            else:
                corrective = np.zeros(7, dtype=np.float64)
                corrective[:3] = pos_err_base
                corrective[6] = float(gripper_command)
                full = _goal_prepare_action(corrective)
            inner._env.step(full)
            if inner._lock_base_in_arm_mode:
                inner._restore_base_joints()
            if kinematic_carry:
                self._snap_object_to_eef()
        self._last_drive_substeps = n_substeps
        return residual

    def _hold_gripper(self, gripper_command: float, substeps: int) -> None:
        """Send ``substeps`` zero-motion OSC steps with a fixed gripper command.

        Used for the dwell-and-close phase of the grasp. Restores the
        mobile-base joints after every step (camera-stability lock).
        """
        inner = self._env
        for _ in range(int(substeps)):
            corrective = np.zeros(7, dtype=np.float64)
            corrective[6] = float(gripper_command)
            inner._env.step(_goal_prepare_action(corrective))
            if inner._lock_base_in_arm_mode:
                inner._restore_base_joints()

    def _snap_object_to_eef(self) -> None:
        """Kinematic fallback: pin the object joint XYZ qpos to the EEF.

        Best-effort: if the object has no free joint or the lookup
        fails, this is a no-op. Used only when
        ``object_reach_kinematic_carry=True``.
        """
        inner = self._env._env
        objects = getattr(inner, "objects", None) or {}
        obj = objects.get(_OBJECT_REACH_BODY_NAME)
        if obj is None:
            return
        joint_names = getattr(obj, "joints", None)
        if not joint_names:
            return
        sim = inner.sim
        eef_pos = self._env._get_eef_pos()
        for jname in joint_names:
            try:
                addr = sim.model.get_joint_qpos_addr(jname)
            except Exception:
                continue
            start = addr[0] if isinstance(addr, tuple) else addr
            qpos = sim.data.qpos.copy()
            # Free-joint qpos layout: [x, y, z, qw, qx, qy, qz].
            # Preserve the current quat (slot 3:7) and only overwrite XYZ.
            qpos[start : start + 3] = eef_pos
            sim.data.qpos[:] = qpos
            sim.forward()

    # ------------------------------------------------------------------
    # Grasp / carry / release orchestration
    # ------------------------------------------------------------------

    def _grasp_object_topdown(self) -> bool:
        """Programmatic top-down grasp of the spawned object.

        Five-phase strict-vertical sequence:

        * **Phase 0 — pre-approach open dwell:** force the gripper fully
          open (command ``-1``) for :data:`_GRASP_PRE_OPEN_SUBSTEPS` so a
          partially-closed gripper carried over from a prior reset cannot
          block the descent.
        * **Phase 1 — approach from above:** drive the EEF to
          ``(target_xy, obj_z + _GRASP_APPROACH_HEIGHT_M)`` with strict-
          vertical quat (:data:`_OBJECT_REACH_TOPDOWN_QUAT_WXYZ`, fingers
          along ``-world_z``); gripper command ``-1`` (open).
        * **Phase 2 — descend until physics blocks:** drive to a
          deliberately low target (``obj_z - _GRASP_DESCEND_DEPTH_M``)
          with the strict-vertical quat held; gripper command still
          ``-1``. The loop runs the full ``max_steps`` and physics
          stops the EEF when the fingertips contact the object / counter.
        * **Phase 3 — close gripper in place:** anchor the drive at the
          post-descent EEF pose, dispatch a ``+1`` gripper command for
          ``max(_GRASP_CLOSE_MIN_STEPS, self._object_reach_grasp_substeps)``
          sub-steps. ``force_run=True`` is critical (target == current
          EEF pos so position convergence would trip on iteration 0).
        * **Phase 4 — lift up:** drive straight up by
          :data:`_GRASP_LIFT_HEIGHT_M` with gripper command ``+1``
          (closed) so a held object is carried up and a missed grasp
          visibly leaves the object on the counter.
        * **Phase 5 — verify:** :func:`OU.check_obj_grasped` with
          ``threshold=_GRASP_VERIFY_THRESHOLD_M`` AFTER the lift, so a
          failed grasp is detected by the object being left behind.

        ``target_xy`` includes the registry's ``grasp_offset_xyz[:2]``
        so any lateral fingertip-vs-EEF-site bias is compensated, plus
        :data:`_OBJECT_REACH_APPROACH_OFFSET_BASE_XYZ` projected from
        base into world via ``R_wb`` so the approach + descend XY are
        biased along the robot's base +X (default +4 cm forward).

        Returns ``True`` if the post-lift grasp verification passes
        (or if ``robocasa.utils.object_utils`` is unavailable, treated
        as success with a warning), ``False`` otherwise.
        """
        spec = self._resolve_object_spec()
        grasp_offset = np.asarray(spec.get("grasp_offset_xyz", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(-1)
        if grasp_offset.size < 3:
            grasp_offset = np.pad(grasp_offset, (0, 3 - grasp_offset.size))

        # Per-object finger lateral offset (world frame XYZ, added to
        # target_xy after grasp_offset + approach_offset). Used to
        # position the wrist such that ONE finger lands inside an
        # opening (e.g. mug cup) and the OTHER outside. Defaults to
        # (0, 0, 0). Only the XY components are used.
        finger_offset = np.asarray(
            spec.get("grasp_finger_offset_xyz", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ).reshape(-1)
        if finger_offset.size < 3:
            finger_offset = np.pad(finger_offset, (0, 3 - finger_offset.size))

        # Per-object descend depth (m). Default ``_GRASP_DESCEND_DEPTH_M
        # = 0.10`` works for flat objects (banana) and most upright
        # cylinders, but a too-large value drives the wrist into the
        # counter and physics stops it WITH lateral push that can
        # destabilise tall/thin objects. Tune down per-object if the
        # gripper "pushes too hard" during the descend phase.
        #
        # ``per_kitchen_descend_depth_m`` is a per-layout override
        # that takes precedence over the baseline. Used for kitchens
        # where the baseline descend pushes the object off the
        # counter (e.g. (mug, k3): the descend's lateral component
        # combined with k3's spawn-near-sink-edge knocks the mug into
        # the sink → grasp pipeline "recovers" the fallen mug from
        # the sink and the long vertical carry slips it out → 100%
        # fall in deployment). Setting a per-kitchen value of -0.01
        # makes the wrist hover ABOVE the object so the close phase
        # wraps around it without pushing.
        layout_id_for_descend = getattr(getattr(self._env, "_env", None), "layout_id", None)
        per_kitchen_descend = spec.get("per_kitchen_descend_depth_m", {}).get(layout_id_for_descend)
        descend_depth_m = float(
            per_kitchen_descend
            if per_kitchen_descend is not None
            else spec.get("descend_depth_m", _GRASP_DESCEND_DEPTH_M)
        )

        obj_pos = self._get_object_world_pos()
        # Project the base-frame approach offset into world via R_wb so
        # the +4 cm shift goes along the robot's base +X (forward),
        # not world +X. ``_get_base_world_R`` returns world-from-base.
        R_wb = self._env._get_base_world_R()
        approach_offset_base = np.asarray(
            spec.get("approach_offset_base_xyz", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ).reshape(3)
        # Per-seed / per-target-kitchen additive delta on top of the
        # default approach offset. Active scene seed lives on the
        # wrapper's inner-env config dict. The kitchen layout_id is set
        # by upstream Kitchen on the inner-inner env after model load.
        # Per-target-kitchen takes precedence over per-seed when both
        # are set; layouts/seeds not in their dicts fall back to the
        # default offset unchanged. Both dicts are PER-OBJECT and read
        # from the registry entry (so a new object can ship its own
        # per-kitchen tuning without touching this class).
        scene_seed = self._env._config.get("seed")
        layout_id = getattr(getattr(self._env, "_env", None), "layout_id", None)
        per_target_delta = spec.get("per_kitchen_approach_offset_delta", {}).get(layout_id)
        per_seed_delta = spec.get("per_seed_approach_offset_delta", {}).get(scene_seed)
        override_delta = per_target_delta if per_target_delta is not None else per_seed_delta
        if override_delta is not None:
            approach_offset_base = approach_offset_base + np.asarray(override_delta, dtype=np.float64).reshape(3)
        approach_offset_world = R_wb @ approach_offset_base
        target_xy = obj_pos[:2] + grasp_offset[:2] + approach_offset_world[:2] + finger_offset[:2]
        obj_z = float(obj_pos[2])

        approach_xyz = np.array(
            [target_xy[0], target_xy[1], obj_z + _GRASP_APPROACH_HEIGHT_M],
            dtype=np.float64,
        )
        descend_xyz = np.array(
            [target_xy[0], target_xy[1], obj_z - descend_depth_m],
            dtype=np.float64,
        )
        topdown_quat = np.asarray(type(self)._OBJECT_REACH_TOPDOWN_QUAT_WXYZ, dtype=np.float64)

        # Phase 0 — pre-approach open dwell (gripper command ``-1`` is
        # ACTIVELY open in robosuite OSC; ``0`` is neutral / no change).
        self._hold_gripper(-1.0, substeps=_GRASP_PRE_OPEN_SUBSTEPS)

        # Phase 1 — approach from above (gripper held fully open).
        self._drive_eef_with_gripper(
            target_world_pos=approach_xyz,
            gripper_command=-1.0,
            target_quat_wxyz=topdown_quat,
            **_GRASP_DRIVE_APPROACH,
        )

        # Phase 2 — descend until physics blocks (gripper still open).
        self._drive_eef_with_gripper(
            target_world_pos=descend_xyz,
            gripper_command=-1.0,
            target_quat_wxyz=topdown_quat,
            **_GRASP_DRIVE_DESCEND,
        )

        # Phase 3 — close gripper in place at the post-descent EEF xyz.
        post_descent_xyz = self._env._get_eef_pos().copy()
        self._drive_eef_with_gripper(
            target_world_pos=post_descent_xyz,
            gripper_command=+1.0,
            target_quat_wxyz=topdown_quat,
            max_steps=max(_GRASP_CLOSE_MIN_STEPS, int(self._object_reach_grasp_substeps)),
            force_run=True,
            **_GRASP_DRIVE_CLOSE,
        )

        # Phase 4 — lift straight up with gripper held closed.
        post_close_xyz = self._env._get_eef_pos().copy()
        lift_xyz = np.array(
            [post_close_xyz[0], post_close_xyz[1], post_close_xyz[2] + _GRASP_LIFT_HEIGHT_M],
            dtype=np.float64,
        )
        self._drive_eef_with_gripper(
            target_world_pos=lift_xyz,
            gripper_command=+1.0,
            target_quat_wxyz=topdown_quat,
            **_GRASP_DRIVE_LIFT,
        )

        # Phase 5 — verify the grasp AFTER the lift — a failed grasp
        # leaves the object behind on the counter so check_obj_grasped
        # will return False.
        if OU is None:
            _LOGGER.warning(
                "robocasa.utils.object_utils unavailable; skipping grasp verification — assuming grasp succeeded."
            )
            return True
        try:
            return bool(
                OU.check_obj_grasped(
                    self._env._env,
                    _OBJECT_REACH_BODY_NAME,
                    threshold=_GRASP_VERIFY_THRESHOLD_M,
                )
            )
        except Exception as exc:
            _LOGGER.warning("check_obj_grasped raised %r — treating as failed grasp", exc)
            return False

    def _drive_carrying_object_to_target(self, target_eef_world: np.ndarray) -> float:
        """Carry the held object to ``target_eef_world`` (gripper closed).

        Uses :meth:`_drive_eef_with_gripper` with ``gripper_command=+1``
        and a :data:`_CARRY_MAX_STEP_M` per-substep cap so a fast OSC
        step can't shake the object loose. If
        ``object_reach_kinematic_carry`` is set, also snaps the object's
        free-joint qpos to the EEF every sub-step.

        REACH fixes #4 + #7 ported into the carry path: the drive
        always locks the wrist to ``self._object_reach_home_eef_quat_site_wxyz``
        (the SITE-frame quat snapshotted right after grasp+lift). This
        bounds the wrist drift over the carry to ``rot_tol``, so the
        init-mode and goal-mode end-of-reset orientations match
        (``test_eef_rotation_matches_between_init_and_goal``).

        The site-frame quat is used (not the body-frame
        ``_object_reach_home_eef_quat``) because the drive helper
        internally reads ``sim.data.site_xmat`` for rotation feedback,
        and the two frames have a fixed ~90° offset on PandaOmron.
        See :meth:`_snapshot_site_quat_wxyz` for the rationale.
        """
        return self._drive_eef_with_gripper(
            target_world_pos=np.asarray(target_eef_world, dtype=np.float64),
            gripper_command=+1.0,
            max_step_m=_CARRY_MAX_STEP_M,
            kinematic_carry=self._object_reach_kinematic_carry,
            target_quat_wxyz=self._object_reach_home_eef_quat_site_wxyz,
            rot_tol=_GOAL_RETURN_HOME_ROT_TOL,
        )

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, mode: str = "init", **kwargs):
        """Drive the scene to the START or GOAL of the object-reach task.

        Both modes share: scene rebuild → optional appliance clearing →
        sample (start, goal) EEF poses → grasp the spawned object →
        carry the object to the mode-specific EEF target. They differ
        only in which sampled EEF pose is used as the carry target and
        in which side of ``d_0`` the resulting object pose anchors:

        * ``mode='init'`` caches ``obj_start_world``;
        * ``mode='goal'`` caches ``obj_target_world`` AND, if a prior
          ``mode='init'`` reset already cached ``obj_start_world``,
          computes ``d_0 = ||target - start||`` for the dense reward.

        The planner contract guarantees ``reset(mode='init')`` is called
        before ``reset(mode='goal')`` and that ``step()`` is invoked
        only after the goal reset. If a caller violates that contract,
        :meth:`_compute_task_reward` falls back to ``d_0 = 1.0`` and
        logs a one-shot warning.

        REACH-port note: every reset force-rebuilds the inner env via
        ``_maybe_recreate_env(..., force_rebuild=True)`` (REACH fix #1),
        which transitively re-seeds the global ``random`` / ``np.random``
        RNGs (REACH fix #2). The robot base pose is cached per scene
        seed and re-injected before every subsequent reset (REACH fix
        #3) so the base is bit-identical across episodes. After the
        grasp + lift completes the wrapper snapshots
        ``_object_reach_home_eef_quat`` and validates the sampled
        ``(start, goal)`` against a rotation-invariance check
        (REACH fix #5, with the wrapper-side analog
        :meth:`_validate_rotation_for_object_reach_targets`). The
        validated pair is cached per ``(scene_seed, object_reach_task_seed)``
        so the second reset of an episode pays zero validation cost.
        """
        if mode not in ("init", "goal"):
            raise ValueError(f"Unknown reset mode for object_reach: {mode!r} (expected 'init' or 'goal')")

        # ---- pull wrapper-only kwargs out of the call --------------------
        if "object_reach_task_seed" in kwargs:
            self._object_reach_task_seed = int(kwargs.pop("object_reach_task_seed"))
        if "object_reach_object_name" in kwargs:
            new_name = str(kwargs.pop("object_reach_object_name"))
            if new_name != self._object_reach_object_name:
                self._object_reach_object_name = new_name
                self._object_reach_object_spec = None  # invalidate cache
                ObjectReachKitchen.DEFAULT_OBJECT_NAME = new_name
                # Object change requires a full inner-env rebuild so the
                # kitchen's placement_initializer re-runs with the new
                # ``_get_obj_cfgs``. Force it via ``_maybe_recreate_env``.
                self._env._maybe_recreate_env({"task_name": _OBJECT_REACH_INNER_TASK_NAME})
        kwargs.pop("mode", None)
        # Always force the inner kitchen task — defensive against reset()
        # callers that pass an unrelated task_name.
        kwargs["task_name"] = _OBJECT_REACH_INNER_TASK_NAME

        # ---- REACH fixes #1+#2: force full inner-env rebuild per reset --
        # Mirrors :meth:`RoboCasaEnv.reset`'s ``force_rebuild=self._reach_task``
        # branch. The rebuild reseeds every internal robocasa RNG
        # (per-fixture placement initializers, ``random.seed``, etc.)
        # so the kitchen layout + object spawn + base pose are
        # deterministic per ``scene_seed``. Without this, repeated
        # ``reset(seed=N)`` calls produced micro-shifts in base pose and
        # object spawn xy across episodes (BASE_STATIC_ATOL was loosened
        # to 1e-5 to absorb this — with the rebuild it tightens to ≤1e-6,
        # parity with REACH).
        scene_seed_for_cache = int(self._env._config.get("seed") or 0)
        self._env._maybe_recreate_env(
            {k: v for k, v in kwargs.items() if k in self._env._config},
            force_rebuild=True,
        )
        # Pull anything else (e.g. wrapper-only kwargs not handled
        # above) and let the inner reset see them too.

        # ---- REACH fix #3: base-pose cache injection --------------------
        # Re-inject the first-reset base pose into ``_ep_meta`` so the
        # robocasa ``set_robot_base()`` collision-retry loop doesn't
        # micro-shift the base between resets of the same scene.
        cached_base = self._object_reach_base_pose_cache.get(scene_seed_for_cache)
        if cached_base is not None:
            inner_env = self._env._env
            if not hasattr(inner_env, "_ep_meta") or inner_env._ep_meta is None:
                inner_env._ep_meta = {}
            pos, ori = cached_base
            inner_env._ep_meta["init_robot_base_pos"] = list(pos)
            inner_env._ep_meta["init_robot_base_ori"] = list(ori)

        # ---- inner env reset (vanilla — reach_task is False) -------------
        self._env.reset(**kwargs)

        # ---- REACH fix #3: capture base pose on first reset per seed ----
        if scene_seed_for_cache not in self._object_reach_base_pose_cache:
            inner_env = self._env._env
            pos = list(map(float, getattr(inner_env, "init_robot_base_pos", [0.0, 0.0, 0.0])))
            ori = list(map(float, getattr(inner_env, "init_robot_base_ori", [0.0, 0.0, 0.0])))
            self._object_reach_base_pose_cache[scene_seed_for_cache] = (pos, ori)

        # ---- snapshot home EEF on first reset ----------------------------
        if self._object_reach_home_eef_pos is None:
            self._object_reach_home_eef_pos = self._env._get_eef_pos().copy()

        # ---- clear non-structural appliances around the home EEF ---------
        if self._object_reach_clear_radius > 0:
            self._env._hide_nearby_appliances(
                self._object_reach_home_eef_pos,
                self._object_reach_clear_radius,
            )

        # ---- grasp the object (with retries) -----------------------------
        # The grasp must complete BEFORE we sample/validate (start, goal)
        # because the validator measures wrist drift WITH the object held
        # — see :meth:`_validate_rotation_for_object_reach_targets`.
        #
        # Fix #A — reject a "successful" grasp that's actually picking
        # up an object the previous attempt knocked off the counter.
        # The robocasa ``check_obj_grasped`` verifier only checks
        # whether the object is close to the gripper after lift; it
        # doesn't know whether the object is in its proper spawn pose
        # or fell into a fixture (sink, cabinet, …). On some
        # (mug, k3) spawns the first descend knocks the mug off the
        # counter into the sink, the wrapper retries, and the retry
        # "grasps" the fallen mug from inside the sink. The subsequent
        # carry to the planner's start_eef target (at the normal lifted
        # z) has to lift the held object 30+ cm vertically and reliably
        # loses contact during the lift.
        #
        # Discriminator: the object's z BEFORE the very first grasp
        # attempt (its spawn z = counter + half-height) compared to
        # its z AFTER each attempted grasp+lift. A NORMAL grasp+lift
        # leaves obj_z near spawn_z + 0 to +lift_height (object is
        # picked up and lifted), so the post-grasp z is GREATER OR
        # EQUAL to spawn z. A FALLEN-POSE recovery has post-grasp z
        # WELL BELOW spawn z because the object fell off the counter
        # before being re-grasped. Rejecting attempts where
        # ``spawn_z - post_grasp_z > _GRASP_FROM_FALLEN_REJECT_DROP_M``
        # forces the wrapper to return grasp_failed=True instead of
        # silently carrying a fallen object.
        try:
            spawn_obj_z = float(self._get_object_world_pos()[2])
        except KeyError:
            spawn_obj_z = None

        grasp_ok = False
        for _attempt in range(max(1, self._object_reach_max_retries)):
            grasp_ok = self._grasp_object_topdown()
            if grasp_ok and spawn_obj_z is not None:
                try:
                    post_grasp_obj_z = float(self._get_object_world_pos()[2])
                except KeyError:
                    post_grasp_obj_z = spawn_obj_z
                drop_m = spawn_obj_z - post_grasp_obj_z
                if drop_m > _GRASP_FROM_FALLEN_REJECT_DROP_M:
                    _LOGGER.warning(
                        "grasp verifier returned True but obj_z dropped %.3f m vs spawn "
                        "(spawn_z=%.3f, post_grasp_z=%.3f, threshold=%.3f) — treating as a "
                        "grasp from a fallen pose (object likely knocked off the counter on a "
                        "prior attempt and re-grasped from a fixture). Rejecting.",
                        drop_m,
                        spawn_obj_z,
                        post_grasp_obj_z,
                        _GRASP_FROM_FALLEN_REJECT_DROP_M,
                    )
                    grasp_ok = False
                    continue
            if grasp_ok:
                break

        if not grasp_ok:
            obs = self._env._build_obs()
            obs["object_reach_grasp_failed"] = True
            try:
                obs["object_reach_obj_pos_world"] = self._get_object_world_pos()
            except KeyError:
                pass
            return obs, []

        # ---- REACH fix #4/#7: snapshot post-LIFT EE rotation BEFORE carry
        # The grasp+lift sequence just completed inside
        # ``_grasp_object_topdown``; the wrist is now at its
        # "lifted-with-object" pose. Snapshot it NOW so the carry
        # below can lock the wrist there every sub-step (REACH fixes
        # #4 + #7 ported to the carry path), and so :meth:`step` (the
        # post-reset planner-time path) can also lock the wrist
        # there. Two quats are stored because the grip-site rotation
        # (``sim.data.site_xmat``) and the gripper body rotation
        # (``robot0_eef_quat`` via :meth:`RoboCasaEnv._get_eef_quat_world`)
        # have a fixed ~90° offset on PandaOmron:
        #
        # * ``_object_reach_home_eef_quat`` (xyzw, body-frame) — what
        #   the test fixture and golden capture sees. Preserved for
        #   backwards-compat with ``test_golden_regression`` /
        #   observation surface.
        # * ``_object_reach_home_eef_quat_site_wxyz`` (wxyz, site-frame)
        #   — what the drive helper compares against internally (it
        #   reads ``sim.data.site_xmat`` for performance). Without
        #   the distinction the drive helper would lock the site to
        #   the body's rotation and pull the actual wrist ~90° away
        #   from the intended target. See :meth:`_snapshot_site_quat_wxyz`.
        self._object_reach_home_eef_quat = self._env._get_eef_quat_world()
        self._object_reach_home_eef_quat_site_wxyz = self._snapshot_site_quat_wxyz()

        # ---- REACH fix #5: rotation-aware (start, goal) sampling --------
        # Look up cached pair (parallels ``RoboCasaEnv._reach_targets_cache``).
        # Sampler key intentionally also includes the scene seed because the
        # validator depends on the post-grasp pose which varies per kitchen.
        task_seed_for_cache = int(
            self._object_reach_task_seed if self._object_reach_task_seed is not None else scene_seed_for_cache
        )
        cache_key = (scene_seed_for_cache, task_seed_for_cache)
        cached_targets = self._object_reach_targets_cache.get(cache_key)
        if cached_targets is None:
            validator = (
                self._validate_rotation_for_object_reach_targets if self._object_reach_rotation_validation else None
            )
            start_eef, goal_eef = self._sample_object_reach_targets(
                validator=validator,
            )
            self._object_reach_targets_cache[cache_key] = (
                start_eef.copy(),
                goal_eef.copy(),
            )
        else:
            start_eef, goal_eef = cached_targets[0].copy(), cached_targets[1].copy()
        target_eef = goal_eef if mode == "goal" else start_eef

        # ---- physical carry to the mode-specific target EEF pose ---------
        # Carry locks the wrist to the post-lift site-frame quat
        # (REACH fixes #4 + #7 ported into the carry path). After the
        # carry, the body-frame quat read from ``robot0_eef_quat`` is
        # within ``rot_tol`` of the snapshotted post-lift body-frame
        # quat, so the init/goal end-of-reset rotations match
        # (``test_eef_rotation_matches_between_init_and_goal``).
        residual = self._drive_carrying_object_to_target(target_eef)

        # ---- reset gripper delta-accumulator to "closed" ---------------
        # The wrapper's grasp + carry phases drive the gripper closed
        # to hold the banana; the planner takes over after reset() and
        # its delta-semantic gripper signal accumulates on top of this
        # starting state. Without this reset, target would carry over
        # from the previous episode (potentially open) and the planner
        # would have to spend ~10 ticks re-closing before the carry
        # state matches reality.
        self._gripper_target = +1.0

        # ---- cache obj_start / obj_target / d_0 --------------------------
        obj_now = self._get_object_world_pos()
        if mode == "init":
            self._object_reach_start_world = obj_now.copy()
        else:  # mode == "goal"
            self._object_reach_target_world = obj_now.copy()

        if self._object_reach_start_world is not None and self._object_reach_target_world is not None:
            self._object_reach_d0 = float(
                np.linalg.norm(self._object_reach_target_world - self._object_reach_start_world)
            )
        # Reset the d0-warning latch so a fresh goal cycle can re-warn
        # if ever reached out of order.
        self._object_reach_d0_warned = False

        # ---- render obs --------------------------------------------------
        obs = self._env._build_obs()
        obs["object_reach_grasp_failed"] = False
        obs["object_reach_obj_pos_world"] = obj_now
        obs["object_reach_d0"] = float(self._object_reach_d0)
        obs["object_reach_residual"] = float(residual)
        obs["object_reach_home_eef_quat_xyzw"] = self._object_reach_home_eef_quat.copy()
        obs["success"] = bool(
            self._object_reach_target_world is not None
            and np.linalg.norm(obj_now - self._object_reach_target_world) < _OBJECT_REACH_SUCCESS_TOL_M
        )
        return obs, []

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_task_reward(
        self,
        reward: float,
        out_obs: dict,
        info: dict,
    ) -> tuple[float, dict, dict]:
        """Override: compute the dense object-reach reward from wrapper state.

        Formula: ``r = max(0, 1 - dist_to_target / d_0)`` where
        ``dist_to_target = ||obj_pos_world - obj_target_world||``. Sets
        ``info['success'] = True`` once the carried object is within
        :data:`_OBJECT_REACH_SUCCESS_TOL` (= 0.02 m) of the target.

        If ``step()`` is called before any ``reset(mode='goal')`` (so
        ``obj_target_world`` is unset), the raw robosuite reward is
        passed through unchanged. If ``obj_target_world`` is set but
        ``d_0`` is missing or ~ 0 (e.g. ``reset(mode='goal')`` was
        called without a prior ``reset(mode='init')``), falls back to
        ``d_0 = 1.0`` and emits a one-shot warning.
        """
        if self._object_reach_target_world is None:
            return reward, out_obs, info

        try:
            obj_pos = self._get_object_world_pos()
        except KeyError:
            # Inner env has no spawned object — defensive pass-through.
            return reward, out_obs, info

        d_t = float(np.linalg.norm(obj_pos - self._object_reach_target_world))
        d0 = float(self._object_reach_d0)
        if d0 < 1e-6:
            if not self._object_reach_d0_warned:
                _LOGGER.warning(
                    "object_reach d_0 is unset / ~0 (was reset(mode='init') "
                    "called before reset(mode='goal')?); falling back to "
                    "d_0 = 1.0 for the dense reward."
                )
                self._object_reach_d0_warned = True
            d0 = 1.0
        r = max(0.0, 1.0 - d_t / d0)
        success = d_t < _OBJECT_REACH_SUCCESS_TOL_M

        info["object_reach_distance_to_goal"] = d_t
        info["object_reach_d0"] = float(self._object_reach_d0)
        info["object_reach_reward"] = float(r)
        info["success"] = bool(success)
        out_obs["object_reach_obj_pos_world"] = obj_pos
        out_obs["object_reach_distance_to_goal"] = d_t
        out_obs["object_reach_d0"] = float(self._object_reach_d0)
        return float(r), out_obs, info

    # ------------------------------------------------------------------
    # Step override: per-sub-step orientation hold + workspace bound
    # ------------------------------------------------------------------

    def step(self, action):
        """Per-step orientation-hold override for the object-reach task.

        REACH ports #4 + #6 + #7 onto the planner-time path. The base
        :meth:`BlockingRoboCasaEnv.step` runs a closed-loop OSC drive
        to the commanded base-frame Δxyz, but only sends the
        planner's commanded rotation on the FIRST sub-step (the
        remaining sub-steps drive position only). For OBJECT_REACH this
        lets the wrist drift across a multi-sub-step blocking call,
        breaking the per-step orientation invariant the planner needs
        to preserve while carrying the object.

        This override re-uses :meth:`_drive_eef_with_gripper` — the
        same primitive the carry phase of :meth:`reset` already uses —
        so every sub-step gets:

        * **Continuous rotation feedback** toward
          ``self._object_reach_home_eef_quat`` (the post-grasp snapshot,
          REACH fix #4 applied to step()).
        * **Quaternion-native EE rotation reads** with sign-continuity
          (REACH fix #7) — no Euler round-trip, no gimbal-lock-driven
          wrist flips.
        * **Workspace sphere bound** around ``self._object_reach_home_eef_pos``
          (radius :data:`_REACH_WORKSPACE_R`) so an aggressive planner
          delta can't push the elbow into a fully-straight singularity
          (REACH fix #6 step 0).
        * **Joint-margin guard** that zeroes the position channel when
          any arm joint is within :data:`_REACH_JOINT_LIMIT_MARGIN_RAD`
          of its limit, letting the rotation corrective mechanically
          relax the offending joint (REACH fix #6 step 1b).

        The gripper command from the planner action is honoured —
        an action that commands an open gripper WILL release the
        object. The orientation hold guards rotation only.

        Behaviour falls back to the parent's plain blocking step when
        the wrapper hasn't yet snapshotted ``_object_reach_home_eef_quat``
        (e.g. before the first successful reset, or after a grasp
        failure where the reset short-circuits).
        """
        if self._object_reach_home_eef_quat_site_wxyz is None or self._object_reach_home_eef_pos is None:
            return super().step(action)

        a7_base = _action_to_planner7_base(action)
        base_pos_world, R_wb = self._env._get_base_world_pose()
        eef_before_world = self._env._get_eef_pos()
        target_world = eef_before_world + R_wb @ a7_base[:3]

        # ---- Gripper delta-accumulator semantic --------------------------
        # Planner-side gripper signal in ``[-1, +1]`` is interpreted as
        # a DELTA in closedness, integrated into ``self._gripper_target``
        # and clipped to ``[-1, +1]``. The clipped target is then
        # forwarded to the OSC every sub-step as an ABSOLUTE gripper
        # command. With ``_gripper_target = +1.0`` after :meth:`reset`,
        # positive planner commands saturate (stay closed) and negative
        # commands open proportionally. See
        # :data:`_OBJECT_REACH_GRIPPER_DELTA_PER_STEP`.
        planner_gripper_delta = float(a7_base[6])
        self._gripper_target = float(
            np.clip(
                self._gripper_target + planner_gripper_delta * _OBJECT_REACH_GRIPPER_DELTA_PER_STEP,
                -1.0,
                +1.0,
            )
        )

        # Use the SITE-frame quat (not the body-frame quat) — the drive
        # helper reads ``sim.data.site_xmat`` for rotation feedback, so
        # its target must also be a site-frame quat. The body-frame
        # quat (``self._object_reach_home_eef_quat``) is preserved for
        # tests / fixture / golden but not used as the OSC's lock
        # target. See :meth:`_snapshot_site_quat_wxyz` for why.
        home_quat_wxyz = np.asarray(self._object_reach_home_eef_quat_site_wxyz, dtype=np.float64)

        residual_m = self._drive_eef_with_gripper(
            target_world_pos=target_world,
            gripper_command=self._gripper_target,
            max_steps=self._max_steps,
            tol=self._pos_tol,
            max_step_m=_CARRY_MAX_STEP_M,
            target_quat_wxyz=home_quat_wxyz,
            rot_tol=_GOAL_RETURN_HOME_ROT_TOL,
            kinematic_carry=self._object_reach_kinematic_carry,
            workspace_center=self._object_reach_home_eef_pos,
            workspace_radius=_REACH_WORKSPACE_R,
            joint_margin_check=True,
            z_min=getattr(self, "_push_step_z_min", None),
            z_max=getattr(self, "_push_step_z_max", None),
        )
        converged = residual_m < self._pos_tol
        n_substeps = int(self._last_drive_substeps)

        success_inner = self._env._env._check_success()

        # ---- Object-dropped detector ------------------------------------
        # Compare object world XYZ to EEF site world XYZ. If they're
        # more than :data:`_OBJECT_REACH_DROP_THRESHOLD_M` apart, the
        # banana is no longer between the fingers (either dropped to
        # the counter / floor, or rolled out of reach). Surfaced both
        # in ``info`` and ``out_obs`` so downstream consumers (logger,
        # reward shaping, planner) can read it from either side.
        # ``_get_object_world_pos`` may raise ``KeyError`` if the inner
        # env hasn't spawned the obj body yet (defensive path: e.g.
        # ``step()`` called before any successful ``reset()`` — return
        # ``object_fell = False, obj_eef_distance = NaN``).
        try:
            obj_pos = self._get_object_world_pos()
            eef_pos = self._env._get_eef_pos()
            obj_eef_distance = float(np.linalg.norm(obj_pos - eef_pos))
            object_fell = bool(obj_eef_distance > _OBJECT_REACH_DROP_THRESHOLD_M)
        except KeyError:
            obj_eef_distance = float("nan")
            object_fell = False

        info = {
            "success": bool(success_inner),
            "sim_steps": n_substeps,
            "control_modes": [-1] * n_substeps,
            "blocking_converged": bool(converged),
            "blocking_residual_m": float(residual_m),
            "object_reach_gripper_target": float(self._gripper_target),
            "object_reach_gripper_delta_cmd": float(planner_gripper_delta),
            "object_reach_obj_eef_distance_m": obj_eef_distance,
            "object_reach_object_fell": object_fell,
        }
        out_obs = self._env._build_obs()
        # Inject the post-step EE quat (xyzw) so callers / tests can
        # measure rotation drift across steps without re-reading the
        # sim externally.
        out_obs["object_reach_eef_quat_xyzw"] = self._env._get_eef_quat_world(q_ref=self._object_reach_home_eef_quat)
        out_obs["object_reach_obj_eef_distance_m"] = obj_eef_distance
        out_obs["object_reach_object_fell"] = object_fell
        # Reward (object_reach formula) — the upstream robosuite reward
        # is meaningless because the inner task is just a kitchen
        # backdrop; the wrapper's :meth:`_compute_task_reward` overrides
        # it with the object-to-target distance formula.
        reward, out_obs, info = self._compute_task_reward(0.0, out_obs, info)
        return out_obs, float(reward), False, info


# ---------------------------------------------------------------------------
# Object-push wrapper subclass
# ---------------------------------------------------------------------------


# Fixed ``task_name`` for the inner :class:`RoboCasaEnv` whenever an
# :class:`ObjectPushBlockingRoboCasaEnv` wraps it.
_OBJECT_PUSH_INNER_TASK_NAME = "ObjectPushKitchen"

# Push distance band (m): the goal object position is the start position plus
# an in-plane vector whose norm is drawn uniformly from [min, max]. "Slightly
# different position" — small enough to remain a slight change, large enough
# that the push is non-trivial.
_OBJECT_PUSH_MIN_DISTANCE_M: float = 0.10
_OBJECT_PUSH_MAX_DISTANCE_M: float = 0.15

# Half-width (m) of the uniform square the START object xy is jittered within,
# around whatever the kitchen spawned (so successive episodes at the same
# scene get distinct start poses).
_OBJECT_PUSH_START_JITTER_M: float = 0.03

# Forward offset (m, robot base-frame +x) applied to the object SAMPLING
# ANCHOR relative to the overhead EEF park centre, so the object rests a bit
# in FRONT of the EEF instead of straight underneath it. The EEF park stays
# above the solid-counter centre; only the object start/goal sampling shifts
# forward by this much.
_OBJECT_PUSH_FORWARD_OFFSET_M: float = 0.07

# Per-step vertical (z) band for the push drive, relative to the overhead park
# pose. z_max = park_z + RISE (barely rise above the park); z_min = park_z -
# view_height + FLOOR_CLEARANCE (descend to ~FLOOR_CLEARANCE above the object
# plane to make contact, but never into the counter). Bounds the wrist so it
# can push at contact height without flying up or jamming down.
_OBJECT_PUSH_STEP_Z_RISE_M: float = 0.03
_OBJECT_PUSH_STEP_Z_FLOOR_CLEARANCE_M: float = 0.06

# Max distance (m) the goal object xy may sit from the kitchen's spawned base
# centre. Both start (jittered) and goal (start + push) are kept inside this
# disk so they stay on the reachable counter region.
_OBJECT_PUSH_REACH_RADIUS_M: float = 0.18

# Height (m) the EEF is parked ABOVE the object plane for the canonical
# OVERHEAD viewpoint. The EEF holds a strict top-down orientation directly
# above the start/goal midpoint, so (a) init and goal robot poses are
# identical, (b) the viewpoint is consistent across kitchens, and (c) the
# downward wrist camera sees the object.
_OBJECT_PUSH_VIEW_HEIGHT_M: float = 0.22

# Wrist (eye-in-hand) camera used to enforce object visibility.
_OBJECT_PUSH_WRIST_CAMERA: str = "robot0_eye_in_hand"

# Per-kitchen robot base shift (metres) along the robot's base-frame Y axis,
# applied so the robot spawns beside a sink instead of in front of it. The
# robot base frame is x-forward / y-LEFT / z-up, so +Y = left, -Y = right.
# Only the kitchens where the robot otherwise faces the sink are listed; all
# others default to 0 (no shift). Tuned from the placement grid:
#   k1: 40 cm right, k2: 40 cm left, k3: 60 cm right.
_OBJECT_PUSH_BASE_SHIFT_BASE_Y: dict[int, float] = {
    1: -0.40,
    2: +0.40,
    3: -0.60,
}

# Hard cap on push-target rejection-sampling attempts before raising.
_OBJECT_PUSH_SAMPLE_MAX_TRIES: int = 200


class ObjectPushBlockingRoboCasaEnv(ObjectReachBlockingRoboCasaEnv):
    """Blocking wrapper for the **object-push** task.

    Variant of :class:`ObjectReachBlockingRoboCasaEnv` where the robot does
    NOT grasp the object. A single free object (default ``"flour_bag"``) sits
    on the counter at a per-episode ``start`` position in ``mode='init'`` and
    at a ``goal`` position (a small in-plane displacement away) in
    ``mode='goal'``.

    Reset semantics (key invariants enforced here):

    * **Identical, fixed overhead robot pose**: the EEF is parked at a strict
      top-down orientation directly above the start/goal *midpoint* at a fixed
      height :data:`_OBJECT_PUSH_VIEW_HEIGHT_M`. The target is
      mode-independent, so the ``init`` and ``goal`` robot poses are EXACTLY
      the same; the height + orientation are constant across kitchens, so the
      viewpoint is consistent everywhere. The only visual difference between
      the init and goal observations is the object's position.
    * **Object visible from the wrist camera**: after parking, the object is
      projected into the ``robot0_eye_in_hand`` camera and visibility is
      *verified* — see :meth:`_assert_object_visible_in_wrist`.

    The dense reward + per-step orientation-hold machinery is inherited from
    :class:`ObjectReachBlockingRoboCasaEnv`: ``r = max(0, 1 - ||obj - goal|| /
    d_0)``. The inner :class:`RoboCasaEnv` is held in ``reach_task=False`` and
    its task is pinned to :data:`_OBJECT_PUSH_INNER_TASK_NAME`.
    """

    _WRAPPER_PRIVATE_ATTRS = ObjectReachBlockingRoboCasaEnv._WRAPPER_PRIVATE_ATTRS + (
        "_object_push_min_distance",
        "_object_push_max_distance",
        "_object_push_start_jitter_m",
        "_object_push_reach_radius_m",
        "_object_push_forward_cone_rad",
        "_object_push_view_height_m",
        "_object_push_robot_home_eef_pos",
        "_object_push_home_eef_quat_xyzw",
        "_object_push_home_eef_quat_site_wxyz",
        "_push_step_z_min",
        "_push_step_z_max",
    )

    def __init__(
        self,
        env: RoboCasaEnv,
        *,
        object_push_object_name: str = "flour_bag",
        object_push_task_seed: int | None = None,
        object_push_min_distance: float = _OBJECT_PUSH_MIN_DISTANCE_M,
        object_push_max_distance: float = _OBJECT_PUSH_MAX_DISTANCE_M,
        object_push_clear_radius: float = 1.0,
        object_push_view_height_m: float = _OBJECT_PUSH_VIEW_HEIGHT_M,
        object_push_start_jitter_m: float = _OBJECT_PUSH_START_JITTER_M,
        object_push_reach_radius_m: float = _OBJECT_PUSH_REACH_RADIUS_M,
        object_push_forward_cone_deg: float | None = None,
        pos_tol: float = _GOAL_BLOCKING_TOL,
        max_steps: int = _GOAL_BLOCKING_MAX_STEPS,
        apply_rotation_in_first_substep: bool = True,
    ):
        # Reuse the object_reach ctor to wire up the shared drive / step /
        # reward state (caches, gripper target, home-eef snapshots, …). We
        # pass the push object name through as the object_reach object name so
        # the inherited helpers that key on ``_object_reach_object_name``
        # resolve correctly; ``_resolve_object_spec`` is overridden below to
        # read OBJECT_PUSH_REGISTRY.
        super().__init__(
            env,
            object_reach_object_name=object_push_object_name,
            object_reach_task_seed=object_push_task_seed,
            object_reach_max_distance=object_push_max_distance,
            object_reach_clear_radius=object_push_clear_radius,
            pos_tol=pos_tol,
            max_steps=max_steps,
            apply_rotation_in_first_substep=apply_rotation_in_first_substep,
        )

        # Push-specific knobs.
        self._object_push_min_distance = float(object_push_min_distance)
        self._object_push_max_distance = float(object_push_max_distance)
        self._object_push_start_jitter_m = float(object_push_start_jitter_m)
        self._object_push_reach_radius_m = float(object_push_reach_radius_m)
        # Optional FORWARD-only goal filter: when set, the push direction is
        # sampled in the robot BASE frame within +/-(cone/2) of +x (forward,
        # away from the base) instead of a uniform 360deg. Forward pushes were
        # ~2x more successful in the confounder analysis. None = uniform (old
        # behaviour, unchanged).
        self._object_push_forward_cone_rad = (
            None if object_push_forward_cone_deg is None else float(np.radians(object_push_forward_cone_deg)) / 2.0
        )
        self._object_push_view_height_m = float(object_push_view_height_m)

        # Robot home EEF pos (snapshotted on the first reset, before any
        # drive) — used as the centre for appliance clearing.
        self._object_push_robot_home_eef_pos = None
        # Overhead-view orientation snapshots (body xyzw + site wxyz) so obs /
        # tests can read the post-park EE rotation.
        self._object_push_home_eef_quat_xyzw = None
        self._object_push_home_eef_quat_site_wxyz = None

        # Per-step vertical (z) clamp band (world frame), set each reset once
        # the overhead park pose is known. None until the first reset; the
        # inherited object_reach step() reads these via getattr and passes them
        # to the drive helper (object_reach itself leaves them unset -> no clamp).
        self._push_step_z_min = None
        self._push_step_z_max = None

        # Re-pin the inner kitchen task to ObjectPushKitchen (the super ctor
        # pinned it to ObjectReachKitchen). The next inner reset rebuilds via
        # ``_maybe_recreate_env`` because ``task_name`` changed.
        ObjectPushKitchen.DEFAULT_OBJECT_NAME = self._object_reach_object_name
        env._config["task_name"] = _OBJECT_PUSH_INNER_TASK_NAME

    # ------------------------------------------------------------------
    # Lookup overrides
    # ------------------------------------------------------------------

    def _resolve_object_spec(self) -> dict:
        """Look up the active object's entry in :data:`OBJECT_PUSH_REGISTRY`."""
        if self._object_reach_object_spec is not None:
            return self._object_reach_object_spec
        name = self._object_reach_object_name
        if name not in OBJECT_PUSH_REGISTRY:
            raise KeyError(
                f"object_push_object_name={name!r} not in OBJECT_PUSH_REGISTRY; "
                f"available: {sorted(OBJECT_PUSH_REGISTRY)}"
            )
        self._object_reach_object_spec = OBJECT_PUSH_REGISTRY[name]
        return self._object_reach_object_spec

    def _base_shift_base_y(self) -> float:
        """Per-kitchen robot base shift (m) along the base-frame Y axis, so the
        robot spawns beside the sink instead of in front of it. 0 if the
        current kitchen layout is not listed in
        :data:`_OBJECT_PUSH_BASE_SHIFT_BASE_Y`."""
        lid = getattr(self._env._env, "layout_id", None)
        if lid is None:
            return 0.0
        return float(_OBJECT_PUSH_BASE_SHIFT_BASE_Y.get(int(lid), 0.0))

    # ------------------------------------------------------------------
    # Push target sampler
    # ------------------------------------------------------------------

    def _sample_push_targets(self, obj_base_world: np.ndarray, validator=None) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``(start_world, goal_world)`` object positions deterministically.

        Anchored on ``obj_base_world`` (the validated solid-counter centre). The
        START is jittered within a small square (half-width
        :data:`_OBJECT_PUSH_START_JITTER_M`); the GOAL is START plus an
        in-plane vector whose norm is drawn uniformly from
        ``[min_distance, max_distance]`` in a uniformly-random direction. Both
        points are kept inside a disk of radius
        :data:`_OBJECT_PUSH_REACH_RADIUS_M` and, if ``validator`` is given,
        each must pass ``validator(xy) -> bool`` (the ground-truth settle check
        that the object rests on solid counter). Seeded by
        ``self._object_reach_task_seed`` (or the scene seed) so the same
        task-seed produces the same pair across ``mode='init'`` / ``'goal'``.
        """
        seed = self._object_reach_task_seed
        if seed is None:
            seed = int(self._env._config.get("seed", 0) or 0)
        rng = np.random.RandomState(int(seed))

        base_xy = np.asarray(obj_base_world[:2], dtype=np.float64)
        z = float(obj_base_world[2])
        r_reach = self._object_push_reach_radius_m

        # Forward-cone goal filter: rotate a base-frame +x-centred direction
        # into the world XY plane. R_wb[:2,:2] maps base-xy -> world-xy.
        cone = self._object_push_forward_cone_rad
        R_wb_xy = None
        if cone is not None:
            R_wb_xy = np.asarray(self._env._get_base_world_R(), dtype=np.float64)[:2, :2]

        for _attempt in range(_OBJECT_PUSH_SAMPLE_MAX_TRIES):
            start_xy = base_xy + rng.uniform(
                -self._object_push_start_jitter_m,
                +self._object_push_start_jitter_m,
                size=2,
            )
            if cone is None:
                theta = rng.uniform(0.0, 2.0 * np.pi)
                push_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
            else:
                # phi=0 -> base +x (forward, away from robot); +/-cone half-width.
                phi = rng.uniform(-cone, +cone)
                push_dir = R_wb_xy @ np.array([np.cos(phi), np.sin(phi)], dtype=np.float64)
                nrm = float(np.linalg.norm(push_dir))
                if nrm > 1e-9:
                    push_dir = push_dir / nrm
            dist = rng.uniform(self._object_push_min_distance, self._object_push_max_distance)
            push = dist * push_dir
            goal_xy = start_xy + push
            if (
                np.linalg.norm(start_xy - base_xy) <= r_reach
                and np.linalg.norm(goal_xy - base_xy) <= r_reach
                and (validator is None or (validator(start_xy) and validator(goal_xy)))
            ):
                start_world = np.array([start_xy[0], start_xy[1], z], dtype=np.float64)
                goal_world = np.array([goal_xy[0], goal_xy[1], z], dtype=np.float64)
                return start_world, goal_world

        raise RuntimeError(
            f"object_push target sampler exhausted {_OBJECT_PUSH_SAMPLE_MAX_TRIES} "
            f"attempts for seed={seed}, base_xy={base_xy.tolist()}, "
            f"distance band=[{self._object_push_min_distance}, "
            f"{self._object_push_max_distance}], reach_radius={r_reach}; the "
            f"solid-counter region around the centre may be too small."
        )

    # ------------------------------------------------------------------
    # Solid-counter validation (ground truth: settle + reject falls)
    # ------------------------------------------------------------------

    def _settle_object_rests(
        self,
        xy_world,
        z_ref: float,
        settle_steps: int = 15,
        fall_thresh: float = 0.05,
        roll_thresh: float = 0.08,
    ) -> bool:
        """Place the object at ``xy_world`` and step physics briefly (arm held
        at home) to check it RESTS on solid counter — i.e. it neither drops
        more than ``fall_thresh`` below ``z_ref`` (fell through the removed-sink
        cutout / off an edge) nor rolls more than ``roll_thresh`` in xy. The
        full sim state is snapshotted and restored, so this is side-effect free.
        """
        sim = self._env._env.sim
        snap = sim.get_state()
        try:
            self._set_object_xy(np.asarray(xy_world, dtype=np.float64))
            self._hold_gripper(+1.0, substeps=settle_steps)
            p = self._get_object_world_pos()
            rests = (z_ref - float(p[2])) < fall_thresh and float(
                np.linalg.norm(p[:2] - np.asarray(xy_world)[:2])
            ) < roll_thresh
        except Exception:
            rests = False
        finally:
            sim.set_state(snap)
            sim.forward()
        return bool(rests)

    def _find_solid_center(self, home_xy, base_xy, z_ref: float):
        """Pick an object centre that rests on solid counter and is reachable.

        Starts at the home nadir and steps TOWARD the robot base (the counter
        lies between the robot and the sink, so the base direction moves off
        the sink onto solid counter) until the object rests. Returns the first
        resting xy; falls back to the home nadir if none is found (the caller's
        per-sample validator will still reject bad draws)."""
        home_xy = np.asarray(home_xy, dtype=np.float64)
        base_xy = np.asarray(base_xy, dtype=np.float64)
        d = base_xy - home_xy
        n = float(np.linalg.norm(d))
        direction = d / n if n > 1e-6 else np.array([-1.0, 0.0])
        for step in np.arange(0.0, 0.281, 0.04):
            cand = home_xy + direction * float(step)
            if self._settle_object_rests(cand, z_ref):
                return cand
        return home_xy

    def _set_object_xy(self, xy_world: np.ndarray) -> None:
        """Force the spawned object's free-joint xy to ``xy_world`` (world).

        Preserves the object's current z and quaternion (slots 2 and 3:7),
        overwriting only the planar position, then settles physics so the
        object rests on the counter at the new xy.
        """
        inner = self._env._env
        objects = getattr(inner, "objects", None) or {}
        obj = objects.get(_OBJECT_REACH_BODY_NAME)
        if obj is None:
            raise KeyError(f"ObjectPush kitchen did not spawn body {_OBJECT_REACH_BODY_NAME!r}")
        joint_name = obj.joints[0]
        sim = inner.sim
        qpos = np.array(sim.data.get_joint_qpos(joint_name)).copy()
        qpos[0:2] = np.asarray(xy_world, dtype=np.float64).reshape(2)
        sim.data.set_joint_qpos(joint_name, qpos)
        sim.forward()

    # ------------------------------------------------------------------
    # Wrist-camera visibility (+ verification)
    # ------------------------------------------------------------------

    def _wrist_camera_hw(self) -> tuple[int, int]:
        cfg = self._env._config
        h = int(cfg.get("camera_height", 240) or 240)
        w = int(cfg.get("camera_width", 320) or 320)
        return h, w

    def _project_to_wrist(self, world_xyz: np.ndarray):
        """Project a world point into the wrist camera; return (row, col, H, W)."""
        import robosuite.utils.camera_utils as cu

        sim = self._env._env.sim
        H, W = self._wrist_camera_hw()
        K = cu.get_camera_transform_matrix(sim, _OBJECT_PUSH_WRIST_CAMERA, H, W)
        px = cu.project_points_from_world_to_camera(np.asarray(world_xyz, dtype=np.float64).reshape(1, 3), K, H, W)[0]
        return float(px[0]), float(px[1]), H, W

    def _object_visible_in_wrist(self, world_xyz: np.ndarray, margin_px: int = 2) -> bool:
        row, col, H, W = self._project_to_wrist(world_xyz)
        return (margin_px <= row < H - margin_px) and (margin_px <= col < W - margin_px)

    def _assert_object_visible_in_wrist(self, world_xyz: np.ndarray, label: str) -> None:
        row, col, H, W = self._project_to_wrist(world_xyz)
        if not ((0 <= row < H) and (0 <= col < W)):
            raise AssertionError(
                f"object_push: object NOT visible in wrist camera "
                f"({_OBJECT_PUSH_WRIST_CAMERA}) at {label}: projected pixel "
                f"(row={row:.1f}, col={col:.1f}) is outside the {H}x{W} frame. "
                f"obj_world={np.asarray(world_xyz).tolist()}."
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, mode: str = "init", **kwargs):
        """Drive the scene to the START or GOAL of the object-push task.

        Both modes: scene rebuild -> appliance clearing -> sample (start, goal)
        object positions on solid counter -> place the object at the
        mode-specific position -> park the EEF at the fixed, mode-independent
        overhead pose (top-down, above the object centre at
        _OBJECT_PUSH_VIEW_HEIGHT_M) -> verify the object is visible in the wrist
        camera. Because the parked pose does not depend on mode, the init and
        goal robot poses are EXACTLY the same; the only visual difference
        between the two observations is the object's position. mode='init'
        caches the intended start; mode='goal' caches the intended goal and
        computes d_0.
        """
        if mode not in ("init", "goal"):
            raise ValueError(f"Unknown reset mode for object_push: {mode!r} (expected 'init' or 'goal')")

        # ---- pull wrapper-only kwargs out of the call --------------------
        if "object_push_task_seed" in kwargs:
            self._object_reach_task_seed = int(kwargs.pop("object_push_task_seed"))
        if "object_push_object_name" in kwargs:
            new_name = str(kwargs.pop("object_push_object_name"))
            if new_name != self._object_reach_object_name:
                self._object_reach_object_name = new_name
                self._object_reach_object_spec = None
                ObjectPushKitchen.DEFAULT_OBJECT_NAME = new_name
                self._env._maybe_recreate_env({"task_name": _OBJECT_PUSH_INNER_TASK_NAME})
        kwargs.pop("mode", None)
        kwargs["task_name"] = _OBJECT_PUSH_INNER_TASK_NAME

        # ---- force full inner-env rebuild per reset ----------------------
        scene_seed_for_cache = int(self._env._config.get("seed") or 0)
        self._env._maybe_recreate_env(
            {k: v for k, v in kwargs.items() if k in self._env._config},
            force_rebuild=True,
        )

        # ---- base-pose cache injection ----------------------------------
        cached_base = self._object_reach_base_pose_cache.get(scene_seed_for_cache)
        if cached_base is not None:
            inner_env = self._env._env
            if not hasattr(inner_env, "_ep_meta") or inner_env._ep_meta is None:
                inner_env._ep_meta = {}
            pos, ori = cached_base
            inner_env._ep_meta["init_robot_base_pos"] = list(pos)
            inner_env._ep_meta["init_robot_base_ori"] = list(ori)

        # ---- inner env reset --------------------------------------------
        self._env.reset(**kwargs)

        # ---- capture base pose on first reset per seed ------------------
        # On the first reset of a scene, capture the default base pose, then
        # apply the per-kitchen base shift (move the robot beside the sink
        # instead of in front of it) and re-place the base so THIS reset
        # already reflects the shift. The shifted pose is cached so every
        # subsequent reset re-injects it (deterministic).
        if scene_seed_for_cache not in self._object_reach_base_pose_cache:
            inner_env = self._env._env
            pos = list(map(float, getattr(inner_env, "init_robot_base_pos", [0.0, 0.0, 0.0])))
            ori = list(map(float, getattr(inner_env, "init_robot_base_ori", [0.0, 0.0, 0.0])))

            y_off = self._base_shift_base_y()
            if abs(y_off) > 1e-9:
                R_wb = self._env._get_base_world_R()
                world_shift = R_wb @ np.array([0.0, y_off, 0.0], dtype=np.float64)
                pos = list(np.asarray(pos, dtype=np.float64) + world_shift)
                if not hasattr(inner_env, "_ep_meta") or inner_env._ep_meta is None:
                    inner_env._ep_meta = {}
                inner_env._ep_meta["init_robot_base_pos"] = list(pos)
                inner_env._ep_meta["init_robot_base_ori"] = list(ori)
                self._env.reset(**kwargs)

            self._object_reach_base_pose_cache[scene_seed_for_cache] = (pos, ori)

        # ---- snapshot robot home EEF on first reset ---------------------
        if self._object_push_robot_home_eef_pos is None:
            self._object_push_robot_home_eef_pos = self._env._get_eef_pos().copy()

        # ---- clear non-structural appliances around the robot home EEF --
        if self._object_reach_clear_radius > 0:
            self._env._hide_nearby_appliances(
                self._object_push_robot_home_eef_pos,
                self._object_reach_clear_radius,
            )

        # ---- sample (start, goal) object positions (cached per episode) --
        obj_spawn = self._get_object_world_pos()
        z_ref = float(obj_spawn[2])
        home_xy = np.asarray(self._object_push_robot_home_eef_pos[:2], dtype=np.float64)
        base_xy = np.asarray(self._env._get_base_world_pose()[0][:2], dtype=np.float64)
        center_xy = self._find_solid_center(home_xy, base_xy, z_ref)
        obj_center = np.array([center_xy[0], center_xy[1], z_ref], dtype=np.float64)
        # Sampling anchor sits a bit in FRONT of the overhead EEF park (base
        # frame +x), so the object does not rest straight under the EEF. The
        # EEF still parks above ``obj_center``; only the object start/goal
        # sampling is shifted forward.
        R_wb = self._env._get_base_world_R()
        obj_anchor = obj_center + R_wb @ np.array([_OBJECT_PUSH_FORWARD_OFFSET_M, 0.0, 0.0], dtype=np.float64)
        task_seed_for_cache = int(
            self._object_reach_task_seed if self._object_reach_task_seed is not None else scene_seed_for_cache
        )
        cache_key = (scene_seed_for_cache, task_seed_for_cache)
        cached = self._object_reach_targets_cache.get(cache_key)
        if cached is None:
            start_world, goal_world = self._sample_push_targets(
                obj_anchor,
                validator=lambda xy: self._settle_object_rests(xy, z_ref),
            )
            self._object_reach_targets_cache[cache_key] = (
                start_world.copy(),
                goal_world.copy(),
            )
        else:
            start_world, goal_world = cached[0].copy(), cached[1].copy()

        # ---- place the object at the mode-specific position -------------
        target_obj_world = goal_world if mode == "goal" else start_world
        self._set_object_xy(target_obj_world[:2])
        obj_now = self._get_object_world_pos()

        # ---- identical, fixed OVERHEAD robot pose -----------------------
        overhead = np.array(
            [obj_center[0], obj_center[1], obj_center[2] + self._object_push_view_height_m],
            dtype=np.float64,
        )
        topdown_site = np.asarray(type(self)._OBJECT_REACH_TOPDOWN_QUAT_WXYZ, dtype=np.float64)
        residual = self._drive_eef_with_gripper(
            target_world_pos=overhead,
            gripper_command=+1.0,
            max_steps=_REACH_DRIVE_MAX_STEPS,
            tol=_REACH_DRIVE_TOL,
            max_step_m=_CARRY_MAX_STEP_M,
            target_quat_wxyz=topdown_site,
            rot_tol=_GOAL_RETURN_HOME_ROT_TOL,
            max_step_rad=0.10,
        )
        self._object_reach_home_eef_pos = self._env._get_eef_pos().copy()
        self._object_reach_home_eef_quat = self._env._get_eef_quat_world()
        self._object_reach_home_eef_quat_site_wxyz = topdown_site.copy()
        self._gripper_target = +1.0

        # Per-step vertical band (world z): allow the wrist to descend from the
        # overhead park to ~FLOOR_CLEARANCE above the object plane and rise at
        # most RISE above the park, so push steps can make contact without the
        # arm flying up or jamming into the counter.
        park_z = float(self._object_reach_home_eef_pos[2])
        self._push_step_z_max = park_z + _OBJECT_PUSH_STEP_Z_RISE_M
        self._push_step_z_min = park_z - self._object_push_view_height_m + _OBJECT_PUSH_STEP_Z_FLOOR_CLEARANCE_M

        # ---- verify the object is visible in the wrist camera ----------
        self._assert_object_visible_in_wrist(obj_now, f"mode={mode} (placed obj)")
        self._assert_object_visible_in_wrist(start_world, "start_world")
        self._assert_object_visible_in_wrist(goal_world, "goal_world")

        # ---- cache obj_start / obj_target / d_0 ------------------------
        obj_actual = self._get_object_world_pos()
        if mode == "init":
            self._object_reach_start_world = start_world.copy()
        else:  # mode == "goal"
            self._object_reach_target_world = goal_world.copy()
        if self._object_reach_start_world is not None and self._object_reach_target_world is not None:
            self._object_reach_d0 = float(
                np.linalg.norm(self._object_reach_target_world - self._object_reach_start_world)
            )
        self._object_reach_d0_warned = False

        # ---- render obs (object_push_* keys) ---------------------------
        obs = self._env._build_obs()
        obs["object_push_obj_pos_world"] = obj_actual
        obs["object_push_start_world"] = start_world.copy()
        obs["object_push_goal_world"] = goal_world.copy()
        obs["object_push_eef_pos_world"] = self._object_reach_home_eef_pos.copy()
        obs["object_push_d0"] = float(self._object_reach_d0)
        obs["object_push_residual"] = float(residual)
        obs["object_push_home_eef_quat_xyzw"] = self._object_reach_home_eef_quat.copy()
        obs["success"] = bool(
            self._object_reach_target_world is not None
            and np.linalg.norm(obj_actual - self._object_reach_target_world) < _OBJECT_REACH_SUCCESS_TOL_M
        )
        return obs, []

    # ------------------------------------------------------------------
    # Step (thin alias over the inherited object_reach step)
    # ------------------------------------------------------------------

    def step(self, action):
        """Reuse the object_reach per-step orientation-hold drive, then expose
        ``object_push_*`` aliases for the ``object_reach_*`` obs / info keys so
        push consumers get task-consistent naming. The inherited step already
        holds the wrist at the post-reset standoff orientation, bounds the EEF
        to a workspace sphere, and computes the dense object-distance reward —
        all correct for pushing, where the object moves by contact as the EEF
        is driven along the planner's commanded delta.
        """
        out_obs, reward, done, info = super().step(action)

        if "object_reach_eef_quat_xyzw" in out_obs:
            out_obs["object_push_eef_quat_xyzw"] = out_obs["object_reach_eef_quat_xyzw"]
        if "object_reach_obj_pos_world" in out_obs:
            out_obs["object_push_obj_pos_world"] = out_obs["object_reach_obj_pos_world"]
        if "object_reach_distance_to_goal" in out_obs:
            out_obs["object_push_distance_to_goal"] = out_obs["object_reach_distance_to_goal"]
        if "object_reach_d0" in out_obs:
            out_obs["object_push_d0"] = out_obs["object_reach_d0"]
        if "object_reach_distance_to_goal" in info:
            info["object_push_distance_to_goal"] = info["object_reach_distance_to_goal"]
        if "object_reach_reward" in info:
            info["object_push_reward"] = info["object_reach_reward"]
        if "object_reach_d0" in info:
            info["object_push_d0"] = info["object_reach_d0"]

        # --- Persist object START / CURRENT / GOAL positions to the npz ---
        # policy_client only records SCALAR info values (vectors are dropped),
        # so we emit per-axis components. World frame (raw) plus the base-frame
        # push vector (goal - start) so push DIRECTION is trivially extractable
        # from logs: dx_base>0 = forward/away from robot base, dy_base>0 = left.
        obj_w = out_obs.get("object_push_obj_pos_world")
        goal_w = self._object_reach_target_world
        start_w = self._object_reach_start_world
        if obj_w is not None:
            obj_w = np.asarray(obj_w, dtype=np.float64)
            info["object_push_obj_x"] = float(obj_w[0])
            info["object_push_obj_y"] = float(obj_w[1])
            info["object_push_obj_z"] = float(obj_w[2])
        if goal_w is not None:
            goal_w = np.asarray(goal_w, dtype=np.float64)
            info["object_push_goal_x"] = float(goal_w[0])
            info["object_push_goal_y"] = float(goal_w[1])
            info["object_push_goal_z"] = float(goal_w[2])
        if start_w is not None:
            start_w = np.asarray(start_w, dtype=np.float64)
            info["object_push_start_x"] = float(start_w[0])
            info["object_push_start_y"] = float(start_w[1])
            info["object_push_start_z"] = float(start_w[2])
        if start_w is not None and goal_w is not None:
            R_wb = np.asarray(self._env._get_base_world_R(), dtype=np.float64)
            vec_base = R_wb.T @ (goal_w - start_w)
            info["object_push_goal_dx_base"] = float(vec_base[0])  # + = forward (away from base)
            info["object_push_goal_dy_base"] = float(vec_base[1])  # + = left
            info["object_push_dir_base_deg"] = float(np.degrees(np.arctan2(vec_base[1], vec_base[0])))

        return out_obs, reward, done, info
