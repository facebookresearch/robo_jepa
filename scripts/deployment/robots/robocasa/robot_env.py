# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
RoboCasa simulation environment wrapper for the robo_jepa world model.

Accepts actions in either of two layouts:

1. **Components dict** — the unified ``GymEnvServer`` action format
   shared by every robo_jepa env wrapper::

       {
           "eef_translation": (3,) float32 — base-frame Δ position (m)
           "eef_rotation":    (3,) float32 — axis-angle Δ rotation (rad)
           "gripper":         (1,) float32 — binary command (binarised
                                              at >=0.5 → +1)
           # optional, RoboCasa-specific:
           "base_motion":     (4,) float32 — forward, side, yaw, torso
                                              (default: zeros)
           "control_mode":    (1,) float32 — ±1 (default: -1 = arm)
       }

   Per ``env.step`` (or per chunk row of ``env.step_chunk``) the components
   dict is packed into the canonical 12-D PandaOmron layout A
   (``app.robo_jepa.embodiment_preprocessing``) and broadcast across the
   ``chunk_size`` robosuite OSC sub-steps.

2. **Canonical 12-D layout A** as a numpy array of shape
   ``(chunk_size, 12)`` or flat ``(chunk_size * 12,)`` — bit-for-bit the
   training-data convention. Used by ``env_vs_wm_eef*.py`` test scripts
   and any caller already speaking layout A.

   Layout A::

       [0:4]   base_motion (base_x, base_y, base_yaw, torso)
       [4]     control_mode  (binary, ±1)
       [5:8]   ee_pos delta  (BASE frame)
       [8:11]  ee_rot delta  (axis-angle, BASE frame)
       [11]    gripper       (binary, ±1)

   Values are normalised to [-1, 1] exactly as in training data. The
   wrapper does NOT renormalise, NOT convert Euler→axis-angle, and NOT
   override the robocasa default controller config — the env is created
   with ``load_composite_controller_config(controller=None)``, which is
   bit-for-bit what the dataset was recorded under.

3. (Legacy back-compat) **Flat 7-D numpy array** — DROID-style
   ``[dx, dy, dz, droll, dpitch, dyaw, d_grip]``. Routed through
   :func:`scripts.robo_jepa.action_components.split_7d_action` and then
   handled identically to the components-dict path.

reset() supports a ``mode`` kwarg:
* ``mode="init"`` (default): standard reset to the initial scene state.
* ``mode="goal"``: reset, then run a per-task hard-coded **expert action
  heuristic** (kept in :mod:`robots.robocasa.goal_expert_actions`) inside
  a temporary robosuite env to render the task-completed frame. The
  returned obs is the goal frame; the main env's state is unchanged.
  Used by minimal task JSONs like ``tasks_robocasa/close_mixer_42.json``.

Obs: dict with keys
    ``pose``   (7,) float64 — [x, y, z, roll, pitch, yaw, grip_closure]
    ``joints`` (7,) float64 — robot0_joint_pos
    ``img_0..N`` (H,W,3) uint8 — one entry per requested camera
    ``time_step_begin``, ``time_step_end``, ``time_send_obs`` — ms timestamps

Niceties retained:
    * ``init_robot_to_home``: reset arm to its home qpos after scene setup.
    * Mobile-base locking that is **gated by control_mode**: after each
      arm-mode (cm = -1) sub-step we restore the base qpos so cameras
      don't drift; after a base-mode (cm = +1) sub-step we let the base
      actually move and re-snapshot the new positions.
    * Multi-camera obs, runtime reconfig via ``reset(...)`` kwargs.

Usage
-----
    cd scripts/deployment/robots/robocasa
    python robot_env_server.py --port 4754
"""

import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from action_components import is_components_dict, merge_components, split_7d_action

# Headless rendering on GPU nodes
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# Canonical PandaOmron action layout (modality.json "layout A"):
#   [0:4]=base_motion(x,y,yaw,torso), [4]=control_mode,
#   [5:8]=ee_pos, [8:11]=ee_rot, [11]=gripper
ACTION_PER_STEP_DIM = 12

# Default sub-step chunk size: matches `raw_fps / fps = 20 / 5` used in training.
DEFAULT_CHUNK_SIZE = 4

# Mobile-base joints we lock for camera stability in arm-mode sub-steps.
_BASE_JOINT_NAMES = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
    "mobilebase0_joint_torso_height",
)


# ── Expert-actions goal generation (TEMPORARY) ────────────────────────────
#
# Hard-coded per-task waypoint sequences used by ``reset(mode="goal")`` to
# drive the EE through a successful task completion (e.g. close the mixer
# head). Each value is a list of ``(label, [dx, dy, dz])`` total EEF
# displacements in **metres** in the **base** frame:
#   +x = forward (toward counter), -x = back (away from counter)
#   +y = left, -y = right
#   +z = up, -z = down
#
# ``return_home`` is auto-computed at runtime as the negated cumulative
# xyz so the EE ends up back at start.
#
# This whole block is intended to be **deleted** once we replace the
# scripted goal generator with recorded `.npz` goal frames.
EXPERT_ACTIONS = {
    ("CloseStandMixerHead", 42): [
        ("back_10cm", [-0.097164, 0.000169, -0.001525]),
        ("up_5cm", [0.000389, -0.000565, 0.048379]),
        ("right_50cm", [-0.000861, -0.480324, -0.000435]),
        ("forward_15cm", [0.146005, 0.001436, -0.001470]),
        ("left_up_40cm_3cm", [-0.006008, 0.342873, 0.058440]),
        ("back_15cm", [-0.143988, 0.000374, -0.001192]),
        ("return_home", None),
    ],
}

# RoboCasa OSC output_max: position (m), axis-angle (rad), gripper.
_GOAL_OSC_OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0])

# Per-step displacement (m) used to chunk waypoints; closed-loop tunables.
# The default robocasa controller has ``ramp_ratio=0.2`` (vs the OLD
# env's 1.0), so each sim step ramps only ~20% toward the requested
# delta. The blocking-max-steps below is sized to give the slow ramp
# enough room to converge per blocking sub-action (≈ 0.8^N ⇒ residual
# err < tol after a few hundred steps in the worst case).
_GOAL_EXPERT_STEP_SIZE = 0.02
_GOAL_BLOCKING_MAX_STEPS = 200
_GOAL_BLOCKING_TOL = 0.001
_GOAL_RETURN_HOME_MAX_STEPS = 600
_GOAL_RETURN_HOME_POS_TOL = 0.002
_GOAL_RETURN_HOME_ROT_TOL = 0.02

# Robocasa supports two disjoint kitchen pools: the ~2,500 pretraining
# scenes (default), and 10 held-out "target" kitchens used for eval.
# See ``robocasa/utils/env_utils.py`` and the docs at
# ``docs/assets/scenes.md`` / ``docs/benchmarking/benchmarking_overview.md``
# for the canonical split. The pool is selected by passing
# ``layout_and_style_ids=...`` to ``robosuite.make()`` (see
# ``robocasa/environments/kitchen/kitchen.py:595``).
_KITCHEN_SPLITS = ("pretrain", "target")
# 10 held-out target kitchens (1,1) … (10,10) — see
# ``docs/assets/scenes.md`` lines 272-285 and
# ``robocasa/utils/env_utils.py`` line 90.
_TARGET_LAYOUT_STYLE_PAIRS = tuple((i, i) for i in range(1, 11))


# Reach-task defaults: a free-form custom "task" implemented entirely in
# this wrapper (no underlying robocasa task), in which the EE is driven
# to a random start / goal pose within a 3D ball around the home EEF
# position. Two purposes:
#  * fast smoke-test for closed-loop CEM planning that doesn't depend on
#    a recorded expert sequence,
#  * deterministic, scene-agnostic start/goal generator (the underlying
#    robocasa task is just used as a kitchen backdrop).
_REACH_DEFAULT_MIN_DISTANCE = 0.0  # m, min 3D EEF distance ||goal - start||
_REACH_DEFAULT_MAX_DISTANCE = 0.20  # m, max 3D EEF distance from home
_REACH_DRIVE_MAX_STEPS = 400  # OSC sub-steps per drive call
_REACH_DRIVE_TOL = 0.002  # world-frame Euclidean (m)
# Hard cap on rejection-sampling draws inside ``_sample_reach_targets``.
# With the default sampler (two independent uniform-in-ball draws around
# home), the empirical reach-distance distribution peaks near ~0.6R and
# tails out to 2R, so the worst-case acceptance rate inside a thin
# annulus like ``[0.25 R, 0.35 R]`` is still > 1 %. 10k draws gives us
# >>99.99 % probability of succeeding even on the tightest reasonable
# band; a failure here would mean the requested ``[min, max]`` interval
# is empty under the current sampler and should be raised loudly so the
# caller can widen it.
_REACH_SAMPLE_MAX_TRIES = 10_000

# When set (positive), reach-task sampling additionally rejects any
# candidate ``(start, goal)`` pair whose actual driven EE world-frame
# rotation drifts by more than ``_REACH_ROT_INVARIANCE_TOL`` rad
# between the start drive and the goal drive. This enforces the
# end-effector rotation invariance asserted by
# ``scripts/deployment/robots/robocasa/test_reach_invariants.py::test_eef_rotation_matches_between_init_and_goal``
# at the sampler level: even on kitchen scenes where the OSC controller
# cannot fully hold rotation through a particular position-only drive
# (e.g. near a kinematic singularity or joint limit), we simply pick a
# different ``(start, goal)`` pair within the same annulus until the
# pair is rotation-invariant. Cost: each validation costs 2 extra
# :meth:`_drive_eef_to_world_target` calls (~2 × _REACH_DRIVE_MAX_STEPS
# OSC sub-steps) per attempt; results are cached per
# ``effective_reach_seed`` so subsequent ``reset(mode=...)`` calls for
# the same (scene, episode) skip the validation entirely.
_REACH_ROT_INVARIANCE_TOL = 0.05  # rad
# Hard cap on candidates considered by the rotation-invariance
# rejection loop. The first candidate usually passes (>= 95 % rate
# under :data:`_REACH_ROT_INVARIANCE_TOL = 0.05`), so a cap of 100
# leaves >> 10⁶× margin against pathological scenes.
_REACH_VALIDATION_MAX_TRIES = 100

# Joint-level singularity rejection used by
# :meth:`RoboCasaEnv._validate_rotation_for_targets`. After driving
# the EE to a candidate ``start`` or ``goal``, the validator reads
# the 7-DOF Panda arm qpos and rejects the candidate if either:
#
#  (a) any joint is within :data:`_REACH_JOINT_LIMIT_MARGIN_RAD` of
#      its lower or upper limit (catches the "fully extended" /
#      "fully bent" cases where the arm is at a physical
#      singularity — visually the corresponding joint stops
#      moving and the wrist locks);
#  (b) any joint deviates from its home (init_qpos) value by more
#      than :data:`_REACH_JOINT_DEVIATION_MAX_RAD` (catches the case
#      where the IK switched to a different solution branch — e.g.
#      elbow-down vs elbow-up — even when no individual joint is
#      at its physical limit).
#
# Tuning: 0.15 rad (~8.6°) of margin from each joint limit, and
# 1.6 rad (~92°) of allowed deviation from home. Both are
# deliberately loose — they only fire on configurations the IK
# would consider visibly "weird" relative to the home pose.
_REACH_JOINT_LIMIT_MARGIN_RAD = 0.15
_REACH_JOINT_DEVIATION_MAX_RAD = 1.6

# Workspace cap used by the per-sub-step EE-pose projection in
# :meth:`RoboCasaEnv._reach_apply_orientation_hold`. The planner's
# commanded EE delta is projected so the new EE world position stays
# within a sphere of radius :data:`_REACH_WORKSPACE_R` centred at
# ``_reach_home_eef_pos``. This is the principal mechanism that
# prevents elbow-fully-straight / IK-degenerate configurations during
# the planner's rollout:
#
#   * The reach annulus is ``[reach_min_distance, reach_max_distance]``
#     = ``[0.25, 0.35]`` m around home in the production setup, so a
#     workspace radius of 0.40 m gives 5 cm of slack for the EE to
#     overshoot the goal mid-trajectory without ever leaving the
#     safe zone.
#   * Empirically on the v3/v4/v5 sweeps the elbow only fully
#     straightens when the planner's commanded EE position exceeds
#     ~0.5 m from home, so 0.40 m sits comfortably below that
#     boundary.
#   * The clip is a SOFT projection: when the commanded delta would
#     leave the sphere, we scale it down to the largest fraction
#     that stays inside. The direction is preserved; only magnitude
#     is clipped. So legitimate reaches toward goals inside
#     ``reach_max_distance`` go through unmodified, but pathological
#     "drive forever" commands saturate at the sphere boundary.
_REACH_WORKSPACE_R = 0.40

# Per-OSC-sub-step normalised position-delta clip used by the
# orientation-hold branch in :meth:`RoboCasaEnv.step` (reach-task only).
# The 12-D OSC action lives in [-1, +1] (normalised against
# :data:`_GOAL_OSC_OUTPUT_MAX`). A norm bound of 0.4 ↔ 0.4 × 0.05 m =
# 0.02 m commanded displacement per OSC sub-step → 0.08 m per WM tick
# (chunk_size=4). The OSC has ramp_ratio=0.2, so the actual per-step
# motion is at most ~0.004 m; this is tight enough to keep the IK
# inside the regime where the wrist can hold the home orientation while
# still letting the planner make meaningful progress toward the goal
# over the 5-step rollouts the production planner uses.
_REACH_STEP_MAX_POS_NORM = 0.4
# Step-time rotation-invariance tolerance (rad). Used by the new
# ``test_eef_rotation_held_across_steps`` smoke test. The active
# rotation feedback in :meth:`RoboCasaEnv.step` drives the wrist back
# toward ``_reach_home_eef_quat`` every sub-step; combined with the
# position clamp above, the per-step rotation drift is typically
# < 0.02 rad. 0.05 rad (~2.9°) is comfortable headroom over the
# worst observed step-time drift.
_EEF_STEP_ROT_TOL_RAD = 0.05

# Reach-task workspace clearing: hide the non-structural fixtures (small
# kitchen appliances such as StandMixer, Toaster, Blender, CoffeeMachine,
# DishRack, ElectricKettle, ToasterOven, Accessory, ...) within this
# radius of the home EEF so the reach task has a clean workspace.
# Structural fixtures (cabinets, drawers, counters, walls, stove, hood,
# sink, microwave, oven, fridge, dishwasher, stool, window, floor, box)
# are always kept. Set <= 0 to disable.
_REACH_DEFAULT_CLEAR_RADIUS = 1.0  # m

# Allow-list of robocasa fixture base-class names that are considered
# part of the kitchen architecture and never hidden by the reach-task
# workspace clearing. The check uses ``__mro__`` name membership so any
# subclass (e.g. ``Drawer`` < ``Cabinet``, ``Stovetop`` < ``Stove``,
# ``FridgeFrenchDoor`` < ``Fridge``, ``Window``/``Stool`` < ``WallAccessory``)
# inherits the structural treatment automatically.
_REACH_STRUCTURAL_BASE_CLASSES = frozenset(
    {
        "Cabinet",  # incl. Drawer, SingleCabinet, HingeCabinet, OpenCabinet,
        #       PanelCabinet, HousingCabinet
        "Counter",
        "Wall",
        "Floor",
        "Stove",  # incl. Stovetop
        "Hood",
        "Sink",
        "Microwave",
        "Oven",
        "Dishwasher",
        "Fridge",  # incl. FridgeBottomFreezer, FridgeFrenchDoor, FridgeSideBySide
        "Stool",
        "Window",
        "WallAccessory",
        "Box",  # structural primitive cubes used as fillers / bases
        "BoxObject",
    }
)


# ── Obstacle-reach task: see ObstacleReachBlockingRoboCasaEnv in
# scripts/deployment/robots/robocasa/blocking_robot_env.py ─────────────
# The obstacle-reach task (CoffeeMachine straddle) is implemented as a
# wrapper subclass — it plugs into the generic reach-task path here
# via the ``_reach_task_hook`` attribute (see :meth:`reset`). All
# obstacle-specific constants / methods / state live in the wrapper
# file so this module stays generic.

# Stall detection inside :meth:`_drive_eef_to_world_target`. If the EEF
# moves less than :data:`_REACH_DRIVE_STALL_TOL_M` (m, world-frame
# Euclidean) over the last :data:`_REACH_DRIVE_STALL_WINDOW` sub-steps,
# the wrist is almost certainly pushing against geometry (a fixture
# right between EEF and target). Bailing out lets the rotation
# validator silently reject the offending (start, goal) pair and the
# sampler re-draws within a few seconds, instead of wedging the
# controller for hundreds of sub-steps × dozens of candidates.
# Window sized for a fast bail: 15 OSC sub-steps ≈ 0.75 s of sim time;
# a healthy drive moves the wrist by ~2 mm per sub-step under the OSC's
# default ramp, so anything below 2 mm of cumulative motion over 15
# sub-steps is the wrist genuinely stuck.
_REACH_DRIVE_STALL_WINDOW: int = 15
_REACH_DRIVE_STALL_TOL_M: float = 0.002


def _goal_chunk_waypoints(task_name, seed, step_size=_GOAL_EXPERT_STEP_SIZE):
    """Convert :data:`EXPERT_ACTIONS` waypoints into per-step 7-D actions
    ``[dx, dy, dz, 0, 0, 0, 0]`` (only EE position moves). Auto-computes
    ``return_home`` as the negated cumulative xyz.
    """
    key = (task_name, seed)
    if key not in EXPERT_ACTIONS:
        raise KeyError(f"No expert actions for {key}. Available: {list(EXPERT_ACTIONS.keys())}")
    waypoints = EXPERT_ACTIONS[key]

    cumulative = np.zeros(3)
    resolved = []
    for label, delta in waypoints:
        if delta is None:
            delta = (-cumulative).tolist()
        cumulative = cumulative + np.array(delta)
        resolved.append((label, np.array(delta)))

    out = []
    for label, delta in resolved:
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            continue
        n = max(1, int(np.ceil(dist / step_size)))
        actions = np.zeros((n, 7))
        actions[:, :3] = delta / n
        out.append((label, actions))
    return out


def _goal_prepare_action(action7):
    """Goal-only: 7-D ``[dxyz, deuler, dgripper]`` (m / rad) → 12-D
    normalised layout B. Mirrors the OLD ``_prepare_action``.
    """
    a = np.asarray(action7, dtype=np.float64).copy()
    a[3:6] = Rotation.from_euler("xyz", a[3:6], degrees=False).as_rotvec()
    n = np.clip(a / _GOAL_OSC_OUTPUT_MAX, -1.0, 1.0)
    full = np.zeros(ACTION_PER_STEP_DIM, dtype=np.float64)
    full[:7] = n
    return full


def _goal_prepare_axisangle(pos_delta, axisangle_delta, gripper=0.0):
    """Same as :func:`_goal_prepare_action` but takes an axis-angle
    rotation delta directly (used by the return_home tracker).
    """
    a = np.zeros(7, dtype=np.float64)
    a[:3] = pos_delta
    a[3:6] = axisangle_delta
    a[6] = gripper
    n = np.clip(a / _GOAL_OSC_OUTPUT_MAX, -1.0, 1.0)
    full = np.zeros(ACTION_PER_STEP_DIM, dtype=np.float64)
    full[:7] = n
    return full


def _layoutA_to_robosuite12(a12):
    """Translate one canonical 12-D action (modality.json layout A) into
    the 12-D vector robosuite's PandaOmron HybridMobileBase composite
    controller actually consumes (layout B):

        layout A: [base_xy_yaw(3), torso(1), control_mode(1),
                   ee_pos(3), ee_rot(3), gripper(1)]
        layout B: [ee_pos(3), ee_rot(3), gripper(1),
                   base_xy_yaw(3), torso(1), base_mode(1)]

    Binary channels (control_mode, gripper) are thresholded at 0.5 to ±1.
    """
    a = np.asarray(a12, dtype=np.float64)
    assert a.shape == (ACTION_PER_STEP_DIM,), f"expected (12,), got {a.shape}"
    out = np.zeros(ACTION_PER_STEP_DIM, dtype=np.float64)
    out[0:3] = a[5:8]
    out[3:6] = a[8:11]
    out[6] = +1.0 if a[11] >= 0.5 else -1.0
    out[7:10] = a[0:3]
    out[10] = a[3]
    out[11] = +1.0 if a[4] >= 0.5 else -1.0
    return out


class RoboCasaEnv:
    """Thin RoboCasa wrapper that consumes WM-aligned action chunks."""

    def __init__(
        self,
        task_name="CloseStandMixerHead",
        camera_names=None,
        camera_width=256,
        camera_height=256,
        seed=None,
        robot="PandaOmron",
        init_robot_to_home=True,
        chunk_size=DEFAULT_CHUNK_SIZE,
        lock_base_in_arm_mode=True,
        wm_embodiment="franka_omron",
        reach_task=False,
        reach_task_seed=0,
        reach_min_distance=_REACH_DEFAULT_MIN_DISTANCE,
        reach_max_distance=_REACH_DEFAULT_MAX_DISTANCE,
        reach_clear_radius=_REACH_DEFAULT_CLEAR_RADIUS,
        kitchen_split="pretrain",
        layout_and_style_ids=None,
    ):
        if camera_names is None:
            camera_names = ["robot0_agentview_left"]

        self._config = {
            "task_name": task_name,
            "camera_names": list(camera_names),
            "camera_width": camera_width,
            "camera_height": camera_height,
            "seed": seed,
            "robot": robot,
            "layout_and_style_ids": layout_and_style_ids,
        }
        self._init_robot_to_home = init_robot_to_home
        self._chunk_size = int(chunk_size)
        self._lock_base_in_arm_mode = bool(lock_base_in_arm_mode)
        # Selects the merge_components branch used by ``_components_to_chunk``.
        # Both ``franka_omron`` and ``franka_omron_state_delta`` produce the
        # same canonical layout-A 12-D vector (env-side execution is shared);
        # the state-delta branch additionally asserts the planner sent zero
        # rotation and zeros the rotation slots before the OSC controller.
        assert wm_embodiment in ("franka_omron", "franka_omron_state_delta"), (
            f"unsupported wm_embodiment={wm_embodiment!r}"
        )
        self._wm_embodiment = wm_embodiment

        # Reach-task config (see :meth:`reset` and :meth:`_sample_reach_targets`).
        # The reach-task RNG is intentionally separate from ``self._config['seed']``
        # (which controls the underlying robocasa scene/layout/objects) so that
        # the start/goal poses can be regenerated for the same kitchen layout
        # by changing only ``reach_task_seed``, and vice versa.
        self._reach_task = bool(reach_task)
        self._reach_task_seed = int(reach_task_seed)
        self._reach_min_distance = float(reach_min_distance)
        self._reach_max_distance = float(reach_max_distance)
        self._reach_clear_radius = float(reach_clear_radius)
        assert self._reach_min_distance >= 0.0, f"reach_min_distance must be >= 0, got {self._reach_min_distance}"
        assert self._reach_min_distance <= self._reach_max_distance, (
            f"reach_min_distance ({self._reach_min_distance}) must be <= "
            f"reach_max_distance ({self._reach_max_distance})"
        )
        # Kitchen pool selector (see :data:`_KITCHEN_SPLITS`). ``"pretrain"``
        # leaves the default ~2,500-scene pretraining pool (back-compat
        # with every prior caller); ``"target"`` restricts the kitchen
        # selection to one of the 10 held-out target kitchens. The
        # specific target kitchen used per env instance is derived
        # deterministically from ``seed`` in :meth:`_create_env`
        # (``seed % 10`` indexes into :data:`_TARGET_LAYOUT_STYLE_PAIRS`),
        # so ``seed=0…9`` cleanly covers all 10 target kitchens.
        assert kitchen_split in _KITCHEN_SPLITS, (
            f"kitchen_split must be one of {_KITCHEN_SPLITS}, got {kitchen_split!r}"
        )
        self._kitchen_split = kitchen_split
        self._reach_home_eef_pos = None  # captured on first reset
        # Per-episode reach state, populated in reset() when reach_task=True.
        # The goal is needed every step (for the reward); the start is
        # needed at reset(mode='init') to anchor d_0.
        self._reach_start_world = None
        self._reach_goal_world = None
        self._reach_d0 = 0.0
        # Per-seed episode counter (reach-task only). Maps
        # reach_task_seed -> episode_idx_within_seed. Incremented only on
        # reset(mode='init'); reset(mode='goal') reads the most recent
        # value without incrementing so init_i and goal_i share the same
        # effective reach seed. ``reset(reset_episode_counter=True)``
        # zeros the counter for the active seed (used by the golden-data
        # generator and the test harness to mimic a fresh process).
        self._reach_episode_idx_by_seed: dict[int, int] = {}
        # Per-(reach_task_seed) last-mode tracker. Used by the
        # per-episode counter state machine: a (init, goal) or
        # (goal, init) pair shares an episode_idx (the SECOND call
        # reuses the FIRST call's counter); a same-mode-twice call or
        # a call after the other-mode was already consumed starts a
        # new episode (increment counter). This makes the counter
        # order-agnostic, so both the generator (init-then-goal) and
        # the planner (goal-then-init) bind the SAME episode_idx to
        # the corresponding ``_reach_goal_world`` used by the
        # per-step reward in :meth:`step`. Without this, the planner
        # rendered episode (i-1)'s goal frame while computing
        # rewards against episode i's goal, producing visually-
        # successful rollouts with zero reward.
        self._reach_last_mode_by_seed: dict[int, str] = {}
        # Per-(scene-seed) base-pose cache. ``robocasa.set_robot_base()``
        # uses a collision-retry loop whose iteration count subtly
        # alternates across ``robosuite.make()`` calls in the same
        # process even at fixed seed (we observed A,B,A,B pose patterns
        # on sc035 / sc041 even though layout_id/style_id stay constant).
        # To pin the base pose deterministically, we cache the first
        # post-reset value and inject it into ``self._env._ep_meta``
        # on every subsequent reset so the env takes the
        # ``init_robot_base_pos in self._ep_meta`` fast path
        # (kitchen.py:1118) and bypasses ``set_robot_base`` entirely.
        self._reach_base_pose_cache: dict[int, tuple[list, list]] = {}
        # Pre-populate the base-pose cache from the per-split golden
        # JSON if it exists on disk. Robocasa's ``set_robot_base()``
        # collision-retry loop is NOT cross-Python-process deterministic
        # (we verified empirically: identical seed + identical code
        # produce ~1.5–2.4 m base_pos drift across separate processes
        # on scenes sc012, sc016, sc035, sc049, target sc001, sc006).
        # The drift is purely translational on the fixture grid; yaw
        # always matches. Without pre-population, the planner /
        # generator / reachability-check / test harness each capture
        # whatever base pose set_robot_base happens to land on first
        # in their own process, and the resulting ``_reach_home_eef_pos``
        # offset shifts every (start, goal) sample by the same vector.
        # Loading the golden's base_pose makes ``set_robot_base`` skip
        # the collision-retry loop entirely (the env takes the
        # ``init_robot_base_pos in self._ep_meta`` fast path at
        # kitchen.py:1118) and bypasses the non-determinism. Goldens
        # are committed and travel with the code, so this is a
        # zero-network-dependency, zero-CI-flake fix.
        if self._reach_task:
            self._preload_reach_base_pose_cache_from_golden()
        # Per-(effective_reach_seed) validated (start, goal) cache. The
        # rotation-invariance rejection loop in :meth:`reset` populates
        # this on the first reset for a given (scene, episode). Both
        # the subsequent ``reset(mode='goal')`` and any further
        # ``reset(mode='init')`` calls for the same (scene, episode)
        # reuse the cached pair so we pay the validation cost
        # (~2 extra drive calls per accepted candidate) only once.
        self._reach_targets_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Home EE world-frame rotation snapshotted at the end of every
        # reach-task reset (after :meth:`_drive_eef_to_world_target`
        # has settled the arm at ``start`` or ``goal``). Stored as a
        # unit quaternion in scipy (xyzw) layout. Used by the
        # per-sub-step orientation-hold branch in :meth:`step` to keep
        # the wrist locked to this rotation throughout the WM rollout.
        # Cleared (set to ``None``) outside reach-task mode.
        self._reach_home_eef_quat = None

        # Reach-task task-specific hook. Default ``None`` keeps the
        # plain reach behaviour intact. Subclasses / wrappers that
        # specialise the reach task (e.g.
        # :class:`ObstacleReachBlockingRoboCasaEnv` in
        # ``blocking_robot_env.py``) install a hook object here whose
        # methods customise (a) which fixture survives
        # ``_hide_nearby_appliances``, (b) the (start, goal) sampler,
        # (c) the per-base-axis drive order, and (d) extra obs keys.
        # The hook protocol is duck-typed; see the wrapper for the
        # expected interface.
        self._reach_task_hook = None

        self._base_joint_addrs = []  # populated after reset
        self._base_joint_qpos = None  # snapshot updated after each sub-step

        self._env = None
        self._create_env()

        # Framework convention (matches MockRobotEnv)
        self.action_space = {}
        self.observation_space = {}

    # ------------------------------------------------------------------
    # Action handling helpers
    # ------------------------------------------------------------------

    def _components_to_chunk(self, components: dict) -> np.ndarray:
        """Pack one components dict into ``(chunk_size, 12)`` layout A.

        The 12-D vector is broadcast across all ``chunk_size`` sub-steps,
        matching the training-time convention (the dataset stored a single
        12-D action per WM tick which the simulator replays for
        ``chunk_size`` controller cycles).
        """
        a12 = merge_components(components, self._wm_embodiment)
        chunk = np.broadcast_to(a12, (self._chunk_size, ACTION_PER_STEP_DIM))
        return np.ascontiguousarray(chunk, dtype=np.float64)

    def _coerce_action_to_chunk(self, action) -> np.ndarray:
        """Normalise any accepted ``step`` input to a ``(chunk_size, 12)``
        canonical-layout-A chunk.

        Accepts:
        * components dict
        * 7-D legacy numpy array (split via :func:`split_7d_action`)
        * canonical layout A numpy array of shape ``(chunk_size, 12)`` or
          ``(chunk_size * 12,)``
        """
        if is_components_dict(action):
            return self._components_to_chunk(action)

        a = np.asarray(action, dtype=np.float64)
        if a.ndim == 1 and a.size == 7:
            return self._components_to_chunk(split_7d_action(a))
        if a.ndim == 1:
            assert a.size == self._chunk_size * ACTION_PER_STEP_DIM, (
                f"flat action must be size 7 (legacy) or chunk_size * 12 = "
                f"{self._chunk_size * ACTION_PER_STEP_DIM}, got {a.size}"
            )
            return a.reshape(self._chunk_size, ACTION_PER_STEP_DIM)
        if a.ndim == 2:
            assert a.shape == (self._chunk_size, ACTION_PER_STEP_DIM), (
                f"2-D action must be {(self._chunk_size, ACTION_PER_STEP_DIM)}, got {a.shape}"
            )
            return a
        raise ValueError(f"action must be 1-D, 2-D, or dict; got ndim={a.ndim}")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _preload_reach_base_pose_cache_from_golden(self) -> None:
        """Pre-populate ``self._reach_base_pose_cache`` from the per-split
        reach-invariant golden JSON file (if present).

        Goldens live at
        ``scripts/deployment/robots/robocasa/reach_invariants_golden{,_target}.json`` and store
        the canonical (``init_robot_base_pos``, ``init_robot_base_ori``)
        per scene seed. Pre-loading them makes the env's ``reset()``
        cache-injection path fire on the FIRST reset for every seed
        (no matter which Python process the env runs in), which in
        turn makes ``set_robot_base()``'s collision-retry loop a
        no-op — kitchen.py takes the ``init_robot_base_pos in
        self._ep_meta`` fast path at line 1118.

        Without this, the cache is empty on process start and
        ``set_robot_base()`` lands on whatever fixture-grid mount cell
        its non-deterministic retry-count happens to choose first;
        that picked cell then gets cached for the lifetime of the
        process but differs across processes (we observed ~1.5–2.4 m
        translational drift on a handful of scenes).

        Quietly no-ops if the golden file is missing or malformed —
        the env continues to work with first-reset auto-capture as
        before.
        """
        import json
        import os

        # Locate the golden file. The reach-invariant goldens now live
        # next to this env file (in the robocasa folder); same dir, no
        # "tests/data/" indirection.
        here = os.path.dirname(os.path.abspath(__file__))
        fname = (
            "reach_invariants_golden.json"
            if self._kitchen_split == "pretrain"
            else "reach_invariants_golden_target.json"
        )
        golden_path = os.path.join(here, fname)
        if not os.path.isfile(golden_path):
            return  # silent no-op; behaviour falls back to first-reset capture
        try:
            with open(golden_path) as f:
                golden = json.load(f)
            scenes = golden.get("scenes", {})
            loaded = 0
            for sid_str, sc in scenes.items():
                base = sc.get("base_pose") if isinstance(sc, dict) else None
                if not isinstance(base, dict):
                    continue
                pos = base.get("init_robot_base_pos")
                ori = base.get("init_robot_base_ori")
                if pos is None or ori is None:
                    continue
                try:
                    seed = int(sid_str)
                except (TypeError, ValueError):
                    continue
                self._reach_base_pose_cache[seed] = (
                    [float(x) for x in pos],
                    [float(x) for x in ori],
                )
                loaded += 1
            print(
                f"[RoboCasaEnv] preloaded {loaded} base-pose cache entries "
                f"from {golden_path} (kitchen_split={self._kitchen_split})",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[RoboCasaEnv] WARN: failed to preload base-pose cache "
                f"from {golden_path}: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def _create_env(self):
        """Create the underlying robosuite env with the **default** robocasa
        composite controller — no overrides. Matches data-collection /
        training settings so our actions land in-distribution.

        Note: ``robosuite.make(seed=N, ...)`` only reseeds robosuite's
        own RNG (``env.rng``). Robocasa's kitchen layout / cell pick
        / fixture placement use Python's stdlib ``random`` and the
        global ``np.random`` module, which are NOT reset by the
        ``seed`` kwarg. Without explicitly reseeding both globals here
        the kitchen layout drifts across successive ``_create_env``
        calls (this manifested as different mobile-base spawn poses
        between ``mode='init'`` and ``mode='goal'`` resets on scenes
        like sc035 / sc041 before this line was added).
        """
        import random

        import robocasa  # noqa: F401  — registers RoboCasa envs with robosuite
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        cfg = self._config
        controller_config = load_composite_controller_config(controller=None, robot=cfg["robot"])
        # Intentionally NO overrides on input_ref_frame / ramp_ratio / etc.

        seed = cfg["seed"]
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed) & 0xFFFFFFFF)

        # Restrict the kitchen pool to the 10 held-out target kitchens
        # when ``kitchen_split == "target"``. We pick one specific
        # ``(layout, style)`` pair per env instance via ``seed % 10``
        # so a caller iterating ``seed=0..9`` cleanly covers all 10
        # target kitchens with no random reuse. Pretrain mode leaves
        # the kwarg unset so robocasa's default pool of ~2,500
        # pretraining scenes is used (back-compat with every prior
        # caller of this class).
        extra_make_kwargs: dict = {}
        if self._kitchen_split == "target":
            target_idx = (int(seed) if seed is not None else 0) % len(_TARGET_LAYOUT_STYLE_PAIRS)
            extra_make_kwargs["layout_and_style_ids"] = [_TARGET_LAYOUT_STYLE_PAIRS[target_idx]]

        make_kwargs = dict(
            env_name=cfg["task_name"],
            robots=cfg["robot"],
            controller_configs=controller_config,
            camera_names=cfg["camera_names"],
            camera_widths=cfg["camera_width"],
            camera_heights=cfg["camera_height"],
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,
            translucent_robot=False,
            seed=cfg["seed"],
            **extra_make_kwargs,
        )
        # Optional: pin one of robocasa's 10 target kitchens (layout, style).
        # When None, robocasa samples its default layout/style distribution
        # — preserving back-compat with existing seeds.
        if cfg.get("layout_and_style_ids") is not None:
            make_kwargs["layout_and_style_ids"] = cfg["layout_and_style_ids"]

        self._env = robosuite.make(**make_kwargs)

    def _maybe_recreate_env(self, config_kwargs, force_rebuild: bool = False):
        """Recreate the env if structural config changed or ``force_rebuild`` is True.

        ``force_rebuild=True`` is the reach-task path (Fix A): the only
        proven way to get layout determinism per seed is to re-call
        ``robosuite.make(seed=N, ...)`` every reset — that single call
        reseeds every internal RNG robocasa uses (per-fixture placement
        initializers, ``random.seed``, robosuite's composite controller
        RNG). Without this, repeated ``reset(seed=N)`` calls produce
        different kitchen layouts / mobile-base spawn poses / fixture
        positions, which in turn break ``_hide_nearby_appliances`` and
        the ``_sample_reach_targets`` anchor.
        """
        # Detect changes BEFORE writing them into self._config so we can
        # decide on recreation. The kitchen layout / object placements
        # are baked in at robosuite.make() time from the seed, so a seed
        # change *does* require a full env rebuild (not just an rng
        # reseed) to roll a new layout.
        changed = bool(force_rebuild)
        for key in (
            "task_name",
            "camera_names",
            "camera_width",
            "camera_height",
            "robot",
            "seed",
            "layout_and_style_ids",
        ):
            if key in config_kwargs and config_kwargs[key] != self._config[key]:
                changed = True
                break

        if "seed" in config_kwargs:
            self._config["seed"] = config_kwargs["seed"]

        if not changed:
            return

        for key, val in config_kwargs.items():
            if key in self._config:
                self._config[key] = val

        if self._env is not None:
            self._env.close()
        self._create_env()
        # The home EEF pos was captured for the OLD kitchen; re-roll
        # so the next reset re-captures it for the new layout. With
        # ``force_rebuild=True`` (the reach-task path) this also
        # guarantees the re-capture happens against a freshly-seeded
        # layout — so ``_hide_nearby_appliances`` operates on the
        # right model + anchor and ``_sample_reach_targets`` is
        # anchored to a deterministic point per scene seed.
        self._reach_home_eef_pos = None

    # ------------------------------------------------------------------
    # Mobile-base locking helpers (camera stability)
    # ------------------------------------------------------------------

    def _snapshot_base_joints(self):
        """Capture current mobile-base joint addresses + qpos."""
        sim = self._env.sim
        self._base_joint_addrs = []
        for name in _BASE_JOINT_NAMES:
            try:
                jid = sim.model.joint_name2id(name)
                self._base_joint_addrs.append(sim.model.jnt_qposadr[jid])
            except Exception:
                pass
        if self._base_joint_addrs:
            self._base_joint_qpos = np.array(
                [sim.data.qpos[a] for a in self._base_joint_addrs],
                dtype=np.float64,
            )
        else:
            self._base_joint_qpos = None

    def _restore_base_joints(self):
        """Write the last snapshot back into the simulation qpos."""
        if self._base_joint_qpos is None or not self._base_joint_addrs:
            return
        sim = self._env.sim
        for addr, val in zip(self._base_joint_addrs, self._base_joint_qpos):
            sim.data.qpos[addr] = val
        sim.forward()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs(self):
        """Build a unified observation dict with pose, joints, images, timestamps."""
        time_begin = time.perf_counter() * 1000

        if hasattr(self._env, "_get_observations"):
            obs_dict = self._env._get_observations(force_update=True)
        elif hasattr(self._env, "_get_observation"):
            obs_dict = self._env._get_observation()
        else:
            obs_dict = {}

        eef_pos = obs_dict["robot0_eef_pos"]
        eef_quat = obs_dict["robot0_eef_quat"]
        gripper_qpos = obs_dict["robot0_gripper_qpos"]
        eef_euler = Rotation.from_quat(eef_quat).as_euler("xyz", degrees=False)
        gripper_closure = gripper_qpos[0:1] - gripper_qpos[1:2]
        proprio = np.concatenate([eef_pos, eef_euler, gripper_closure])

        # 14-D RoboCasa proprio (the same layout as the training data):
        #   [0:3]   base_pos          (m, world)
        #   [3:7]   base_quat (xyzw)
        #   [7:10]  ee_pos_rel        (m, base frame) = R_bw @ (eef_world - base_pos)
        #   [10:13] ee_rot_rel quat[xyz]              = first 3 of quat(R_bw @ R_we)
        #   [13]    gripper_qpos[0]   (sim m, raw)
        # Used by the franka_omron_state_delta WM variant.
        base_pos_world, R_wb = self._get_base_world_pose()  # R_wb := R_world_from_base
        R_bw = R_wb.T  # R_base_from_world
        # mujoco wxyz → scipy xyzw for the base quaternion.
        try:
            bid = self._env.sim.model.body_name2id("mobilebase0_base")
            base_quat_wxyz = self._env.sim.data.body_xquat[bid]
        except Exception:
            base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        base_quat_xyzw = np.array(
            [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]],
            dtype=np.float64,
        )
        ee_pos_rel = R_bw @ (np.asarray(eef_pos, dtype=np.float64) - base_pos_world)
        # Compose R_be = R_bw @ R_we, take first 3 components (xyz) of its quat.
        R_we = Rotation.from_quat(eef_quat).as_matrix()
        ee_rot_rel_quat = Rotation.from_matrix(R_bw @ R_we).as_quat()  # xyzw
        ee_rot_rel_quat3 = ee_rot_rel_quat[:3]
        proprio_robocasa = np.concatenate(
            [
                base_pos_world.astype(np.float64),
                base_quat_xyzw,
                ee_pos_rel.astype(np.float64),
                ee_rot_rel_quat3.astype(np.float64),
                np.asarray(gripper_qpos[0:1], dtype=np.float64),
            ]
        )

        joints = np.array(obs_dict.get("robot0_joint_pos", np.zeros(7)))

        H = self._config["camera_height"]
        W = self._config["camera_width"]
        sim = self._env.sim
        sim.forward()  # ensure kinematics consistent with current qpos
        img_dict = {}
        for idx, cam_name in enumerate(self._config["camera_names"]):
            img = sim.render(height=H, width=W, camera_name=cam_name)[::-1].copy()
            img_dict[f"img_{idx}"] = img

        time_end = time.perf_counter() * 1000
        return {
            "pose": proprio.astype(np.float64),
            "proprio_robocasa": proprio_robocasa.astype(np.float64),
            "joints": joints.astype(np.float64),
            **img_dict,
            "time_step_begin": time_begin,
            "time_step_end": time_end,
            "time_send_obs": time.perf_counter() * 1000,
        }

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        """Reset the environment and return ``(obs, init_obs_list)``.

        Supports runtime reconfiguration via kwargs:
        ``task_name, camera_names, camera_width, camera_height, seed, robot``.
        The environment is only recreated if structural config actually
        changed.

        Mode kwargs (compat with task JSONs):

        * ``mode="init"`` (default) — standard reset: returns the
          initial-scene observation from the main env.
        * ``mode="goal"`` — standard reset, then runs the per-task
          hard-coded expert action sequence (see :data:`EXPERT_ACTIONS`)
          on **the same env** in closed-loop blocking mode (matches the
          OLD ``robot_env_old.py`` behaviour). The simulator ends up at
          the task-completed state and the returned obs is rendered
          from there. The returned obs also carries a boolean
          ``"success"`` field populated from
          ``self._env._check_success()``.

        ``init_obs_list`` is empty for both modes (mirrors the
        Franka / DROID convention).
        """
        config_kwargs = {}
        for key in ("task_name", "camera_names", "camera_width", "camera_height", "seed", "robot"):
            if key in kwargs:
                config_kwargs[key] = kwargs[key]

        # ``reach_task_seed`` is a reach-task-only knob — does not require
        # env rebuild. Update in-place; the per-episode counter logic
        # below uses this fresh value when deriving ``effective_reach_seed``.
        if "reach_task_seed" in kwargs:
            self._reach_task_seed = int(kwargs["reach_task_seed"])

        mode = kwargs.get("mode", "init")
        # ``reset_episode_counter=True`` zeros the per-seed reach-task
        # episode counter (so the next ``mode='init'`` reset for the
        # current ``reach_task_seed`` corresponds to episode 0). Used
        # by the golden-data generator and by the test harness to
        # mimic a fresh process when iterating over scenes.
        reset_episode_counter = bool(kwargs.get("reset_episode_counter", False))

        # ``reach_episode_idx=K`` (opt-in, default None) PINS this reset
        # to a specific reach episode index instead of the auto-
        # incrementing per-seed counter. Both the ``mode='goal'`` and
        # ``mode='init'`` calls of the pair pass the same K, so a fresh
        # single-episode process reproduces the BIT-IDENTICAL (start,
        # goal) draw of sequential episode K (effective_reach_seed =
        # reach_task_seed*100003 + K). Enables one-SLURM-job-per-episode
        # collection with no drift vs the sequential loop. Default None
        # preserves the auto-counter behaviour exactly.
        forced_reach_episode_idx = kwargs.get("reach_episode_idx", None)
        if forced_reach_episode_idx is not None:
            forced_reach_episode_idx = int(forced_reach_episode_idx)

        # Fix A: when the env is in reach-task mode, force a full
        # robosuite.make() rebuild for every reset() call. This is the
        # only proven way to get bit-identical kitchen layout / mobile-
        # base spawn / fixture placements across repeated resets at
        # the same ``seed``. The non-reach-task branch (e.g. the legacy
        # ``CloseStandMixerHead`` training-mode resets) keeps the
        # cheap rebuild-only-when-config-changes path.
        if config_kwargs or self._reach_task:
            self._maybe_recreate_env(config_kwargs, force_rebuild=self._reach_task)

        seed = self._config.get("seed")
        if seed is not None:
            self._env.rng = np.random.default_rng(seed)

        # Inject the cached (deterministic) base pose into ``_ep_meta``
        # so robocasa's ``_reset_internal`` takes the fast path at
        # kitchen.py:1118 and uses the cached pose instead of
        # re-running ``set_robot_base`` (whose collision-retry loop
        # gives non-deterministic poses across rebuilds on some scenes
        # — see :attr:`_reach_base_pose_cache` for the rationale).
        if self._reach_task and seed is not None:
            cached = self._reach_base_pose_cache.get(seed)
            if cached is not None:
                pos, ori = cached
                if not hasattr(self._env, "_ep_meta") or self._env._ep_meta is None:
                    self._env._ep_meta = {}
                self._env._ep_meta["init_robot_base_pos"] = list(pos)
                self._env._ep_meta["init_robot_base_ori"] = list(ori)

        self._env.reset()

        # Capture the natural base pose on the FIRST reset for this
        # seed (cache empty path); on subsequent resets the cache
        # injection above already pinned the pose, so this read is a
        # no-op for cache correctness.
        if self._reach_task and seed is not None and seed not in self._reach_base_pose_cache:
            pos = list(map(float, getattr(self._env, "init_robot_base_pos", [0.0, 0.0, 0.0])))
            ori = list(map(float, getattr(self._env, "init_robot_base_ori", [0.0, 0.0, 0.0])))
            self._reach_base_pose_cache[seed] = (pos, ori)

        # Optionally restore the arm to its home qpos.
        if self._init_robot_to_home:
            robot = self._env.robots[0]
            if robot is not None and hasattr(robot, "init_qpos"):
                robot_qpos_addr = self._env.sim.model.get_joint_qpos_addr(robot.robot_joints[0])
                start = robot_qpos_addr[0] if isinstance(robot_qpos_addr, tuple) else robot_qpos_addr
                qpos = self._env.sim.data.qpos.copy()
                qpos[start : start + len(robot.init_qpos)] = robot.init_qpos
                self._env.sim.data.qpos[:] = qpos
                self._env.sim.forward()

        # Snapshot base joints so we can lock cameras during arm-mode sub-steps.
        self._snapshot_base_joints()

        obs_list = []

        # Reach task: scene-agnostic random start/goal driving. The
        # underlying robocasa task is used only as a kitchen backdrop;
        # success / waypoints from EXPERT_ACTIONS are bypassed entirely.
        if self._reach_task:
            # Per-seed episode counter state machine (order-agnostic
            # over (init, goal) call pairs). The planner calls
            # ``reset(mode='goal')`` FIRST and ``reset(mode='init')``
            # SECOND per episode; the generator / test harness do the
            # reverse. Both must yield the SAME ``effective_reach_seed``
            # for the two calls that belong to the SAME episode, so the
            # rendered goal frame matches the per-step reward's goal.
            #
            # Logic:
            #   * ``last_mode == None`` (fresh seed OR previous pair
            #     was completed): this call STARTS a new episode →
            #     increment counter, record ``mode`` as last_mode.
            #   * ``last_mode == mode`` (same mode twice in a row,
            #     e.g. two inits): this is also START of a new episode
            #     (the previous one was abandoned) → increment, record.
            #   * ``last_mode != mode`` (the OTHER mode was the most
            #     recent call): this is the SECOND call of the pair →
            #     reuse counter, clear last_mode so the next call
            #     starts a fresh pair.
            if reset_episode_counter:
                self._reach_episode_idx_by_seed.pop(self._reach_task_seed, None)
                self._reach_last_mode_by_seed.pop(self._reach_task_seed, None)

            if forced_reach_episode_idx is not None:
                # Opt-in: caller pinned the episode index (same K for the
                # goal + init calls of the pair) → bypass the auto-counter
                # state machine entirely so a single-episode process lands
                # on episode K's exact draw.
                episode_idx = forced_reach_episode_idx
            else:
                last_mode = self._reach_last_mode_by_seed.get(self._reach_task_seed)
                if last_mode is None or last_mode == mode:
                    episode_idx = self._reach_episode_idx_by_seed.get(self._reach_task_seed, -1) + 1
                    self._reach_episode_idx_by_seed[self._reach_task_seed] = episode_idx
                    self._reach_last_mode_by_seed[self._reach_task_seed] = mode
                else:
                    episode_idx = self._reach_episode_idx_by_seed.get(self._reach_task_seed, 0)
                    self._reach_last_mode_by_seed.pop(self._reach_task_seed, None)

            # ``effective_reach_seed = (reach_task_seed * P + episode_idx)``
            # masked to uint32 so np.random.RandomState accepts it. The
            # multiplier is a large prime so a 49-scene × 10-episode set
            # maps to a well-spread region of the seed space (no
            # collisions and no obvious periodicity at small N).
            effective_reach_seed = (self._reach_task_seed * 100003 + episode_idx) & 0xFFFFFFFF

            # Capture the home EEF pos from the freshly-rebuilt env.
            # With Fix A in place, ``_create_env(seed=N)`` produces a
            # bit-identical layout per seed so this re-capture lands
            # on the same value across resets — no need to freeze the
            # first value any more (the freeze was a workaround for
            # the pre-Fix-A layout drift). Reset to None on every
            # call so changing the scene seed re-rolls the anchor.
            self._reach_home_eef_pos = self._get_eef_pos().copy()

            # ``_reach_task_hook`` is the extension point for reach-task
            # specialisations. ``None`` (default) preserves the plain
            # reach behaviour; subclasses / wrappers install a hook to
            # customise (a) which fixture survives
            # ``_hide_nearby_appliances`` (and any pre-hide side
            # effects, e.g. moving the obstacle to a canonical
            # position), (b) the (start, goal) sampler, (c) the per-
            # base-axis drive order for the reset-time drive, and (d)
            # any extra obs keys. See
            # :class:`ObstacleReachBlockingRoboCasaEnv` in
            # ``blocking_robot_env.py`` for the concrete obstacle-
            # reach implementation.
            hook = self._reach_task_hook
            except_fixture_name = None
            if hook is not None and hasattr(hook, "before_hide"):
                except_fixture_name = hook.before_hide(self)

            # Clear non-structural appliances within radius of home so
            # the reach task has a clean workspace. No-op for ``reach_task=False``
            # (the standard CloseStandMixerHead path is therefore
            # unaffected). With Fix A this runs against a freshly-built
            # model on every reset, so the mixer (and every other
            # non-structural fixture within radius) is hidden in
            # every episode of every scene. ``except_fixture_name`` is
            # populated by the reach-task hook (when set) so one chosen
            # fixture survives the clearer.
            if self._reach_clear_radius > 0:
                self._hide_nearby_appliances(
                    self._reach_home_eef_pos,
                    self._reach_clear_radius,
                    except_fixture_name=except_fixture_name,
                )

            if hook is not None and hasattr(hook, "sample_targets"):
                start_pos, goal_pos = hook.sample_targets(self, effective_reach_seed)
            else:
                start_pos, goal_pos = self._reach_targets_cache.get(effective_reach_seed, (None, None))
                if start_pos is None:
                    start_pos, goal_pos = self._sample_reach_targets(
                        effective_reach_seed,
                        validator=self._validate_rotation_for_targets,
                    )
                    self._reach_targets_cache[effective_reach_seed] = (
                        np.asarray(start_pos, dtype=np.float64).copy(),
                        np.asarray(goal_pos, dtype=np.float64).copy(),
                    )
            # Persist BOTH so :meth:`step` can compute the per-step
            # reach reward (always uses the goal) and so a subsequent
            # reset(mode='init') can anchor ``d_0`` (the start↔goal
            # distance the planner has to bridge). Sampling is
            # deterministic from ``reach_task_seed``, so this is a no-op
            # when called twice.
            self._reach_start_world = start_pos.astype(np.float64).copy()
            self._reach_goal_world = goal_pos.astype(np.float64).copy()
            if mode == "goal":
                target_pos = goal_pos
            elif mode == "init":
                target_pos = start_pos
            else:
                raise ValueError(f"Unknown reset mode for reach_task: {mode!r}")

            drive_axis_order = getattr(hook, "drive_axis_order", "xyz") if hook is not None else "xyz"
            residual = self._drive_eef_to_world_target(
                target_pos, axis_order=drive_axis_order
            )
            # Snapshot the home EE rotation AFTER the drive has
            # settled. The per-sub-step orientation-hold branch in
            # :meth:`step` uses this as the reference quaternion the
            # wrist is locked to throughout the WM rollout.
            self._reach_home_eef_quat = self._get_eef_quat_world()
            obs = self._build_obs()
            obs["reach_target_world"] = target_pos.astype(np.float64).copy()
            obs["reach_home_world"] = self._reach_home_eef_pos.astype(np.float64).copy()
            obs["reach_start_world"] = self._reach_start_world.copy()
            obs["reach_goal_world"] = self._reach_goal_world.copy()
            obs["reach_residual"] = float(residual)
            obs["success"] = bool(residual < _GOAL_RETURN_HOME_POS_TOL)

            # ``d_0`` = start ↔ goal world-frame Euclidean distance. The
            # reach reward in :meth:`step` is
            # ``max(0, 1 - ||eef_t - goal|| / d_0)``, so anchor it once
            # at reset(mode='init') (the canonical start of an
            # episode). Re-anchoring on reset(mode='goal') is also fine
            # since the (start, goal) pair is deterministic from the
            # seed and identical across the two reset calls.
            self._reach_d0 = float(np.linalg.norm(self._reach_goal_world - self._reach_start_world))
            obs["reach_d0"] = self._reach_d0
            obs["reach_distance_to_goal"] = float(np.linalg.norm(self._get_eef_pos() - self._reach_goal_world))
            if hook is not None and hasattr(hook, "decorate_obs"):
                hook.decorate_obs(self, obs)
            return obs, obs_list

        if mode == "goal":
            self._run_expert_to_goal()
            obs = self._build_obs()
            obs["success"] = bool(self._env._check_success())
            return obs, obs_list
        elif mode != "init":
            raise ValueError(f"Unknown reset mode: {mode!r}")

        return self._build_obs(), obs_list

    # ------------------------------------------------------------------
    # Expert-actions goal generation (used by reset(mode="goal"); TEMPORARY)
    # ------------------------------------------------------------------

    def _get_eef_pos(self):
        """Return the current EE position (3,) from the simulator."""
        if hasattr(self._env, "_get_observations"):
            obs_dict = self._env._get_observations(force_update=True)
        else:
            obs_dict = self._env._get_observation()
        return np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float64)

    def _get_eef_pose7(self):
        """Return ``[x, y, z, roll, pitch, yaw, grip_closure]``."""
        if hasattr(self._env, "_get_observations"):
            obs_dict = self._env._get_observations(force_update=True)
        else:
            obs_dict = self._env._get_observation()
        pos = np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float64)
        quat = obs_dict["robot0_eef_quat"]
        euler = Rotation.from_quat(quat).as_euler("xyz", degrees=False)
        grip = obs_dict["robot0_gripper_qpos"][0:1] - obs_dict["robot0_gripper_qpos"][1:2]
        return np.concatenate([pos, euler, grip])

    def _get_eef_quat_world(self, q_ref=None) -> np.ndarray:
        """Return the EE world-frame rotation as a quaternion (xyzw).

        Reads ``robot0_eef_quat`` directly from the simulator and
        applies **sign-continuity correction** against an optional
        reference quaternion ``q_ref`` (xyzw). Unit quaternions ``q``
        and ``-q`` describe the same rotation, but the simulator
        sometimes returns them in different hemispheres across
        successive reads; without the flip, an operation like
        ``(R_ref * R_now.inv()).as_rotvec()`` can jump by 2π between
        sub-steps even though the underlying rotation changed by an
        infinitesimal amount, sending a spurious large rotation
        corrective into the OSC controller and physically flipping
        the wrist by π. The fix is a single dot-product check.

        Unlike :meth:`_get_eef_pose7` this method NEVER touches the
        Euler chart: the round-trip ``quat → euler → matrix`` is
        degenerate around pitch = ±π/2 and is the principal source
        of the gimbal-lock-driven wrist flips observed during reach
        rollouts.

        :param q_ref: optional reference quaternion (xyzw, shape (4,)).
            When provided, the returned quaternion is flipped to
            ``-q`` whenever ``dot(q, q_ref) < 0``.
        :returns: ``np.ndarray`` of shape ``(4,)`` in xyzw layout.
        """
        if hasattr(self._env, "_get_observations"):
            obs_dict = self._env._get_observations(force_update=True)
        else:
            obs_dict = self._env._get_observation()
        q = np.asarray(obs_dict["robot0_eef_quat"], dtype=np.float64).copy()
        if q_ref is not None:
            q_ref_arr = np.asarray(q_ref, dtype=np.float64)
            if float(np.dot(q, q_ref_arr)) < 0.0:
                q = -q
        return q

    def _get_arm_joint_pos_and_limits(self) -> tuple:
        """Return ``(qpos, lower, upper)`` for the active arm's 7 joints.

        ``qpos`` reads the current joint angles from the simulator
        (length = number of arm joints, typically 7 for PandaOmron);
        ``lower``/``upper`` are the corresponding ``sim.model.jnt_range``
        entries. Used by
        :meth:`_validate_rotation_for_targets` to detect the
        joint-level "fully straight" / "at limit" singularity cases
        that the EE-rotation-only check cannot catch on its own.

        :returns: tuple ``(qpos, lower, upper)`` of three
            ``np.ndarray`` of identical length.
        """
        robot = self._env.robots[0]
        sim = self._env.sim
        addrs = []
        for jname in robot.robot_joints:
            jid = sim.model.joint_name2id(jname)
            addrs.append(sim.model.jnt_qposadr[jid])
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

    def _get_base_world_pose(self):
        """Return ``(base_pos_world, R_world_from_base)`` for the mobile base.

        The default robocasa controller takes EE deltas in the **base**
        frame; observations come back in the **world** frame. We use
        this rotation to convert between them inside
        :meth:`_run_expert_to_goal` (the base is locked during goal
        generation so this matrix is sampled once at the start) AND to
        build the 14-D RoboCasa proprio in :meth:`_build_obs` for the
        ``franka_omron_state_delta`` WM variant.

        :returns: ``(pos, R)`` where ``pos`` is the base body's world
            position (3-vector, m) and ``R`` is its world-from-base
            rotation matrix (3×3). If the body lookup fails, falls back
            to ``(zeros(3), eye(3))``.
        """
        for body_name in ("mobilebase0_base", "robot0_base", "base"):
            try:
                bid = self._env.sim.model.body_name2id(body_name)
                pos = np.asarray(self._env.sim.data.body_xpos[bid], dtype=np.float64).copy()
                quat_wxyz = self._env.sim.data.body_xquat[bid]
                # mujoco stores (w, x, y, z); scipy expects (x, y, z, w).
                q = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                R = Rotation.from_quat(q).as_matrix()
                return pos, R
            except Exception:
                continue
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    def _get_base_world_R(self):
        """Return ``R_world_from_base`` (3x3) for the mobile base body.

        Thin compatibility wrapper around :meth:`_get_base_world_pose` so
        the historical call sites in :meth:`_run_expert_to_goal` keep
        working unchanged.
        """
        _, R = self._get_base_world_pose()
        return R

    def _sample_reach_targets(self, seed=None, validator=None):
        """Deterministically sample ``(start_pos_world, goal_pos_world)``
        for the reach task with **rejection sampling** to enforce
        ``reach_min_distance ≤ ||goal - start|| ≤ reach_max_distance``
        and (optionally) end-effector rotation invariance.

        Inner draw: ``start`` and ``goal`` are each sampled uniformly
        inside a ball of radius ``self._reach_max_distance`` around
        ``self._reach_home_eef_pos`` (world frame, m) — same as the
        legacy sampler so each individual point stays close to home
        (the EEF drive needs to reach it from home in one
        :meth:`_drive_eef_to_world_target` call). Without rejection,
        the resulting ``||goal - start||`` distribution covers
        ``[0, 2R]`` and peaks near ``0.6R``; rejection trims this to
        the requested annulus.

        Both draws use a single ``np.random.RandomState(seed)``
        INSTANTIATED FRESH every call, so:

        * the same effective seed always produces the same
          ``(start, goal)`` pair (regardless of how many env resets
          happened in between);
        * the underlying robocasa scene RNG (``self._env.rng`` from
          ``self._config['seed']``) is left completely untouched, so
          the kitchen layout / objects are decoupled from the
          start/goal sampling;
        * a different (scene_seed, episode_idx) combination produces
          a deterministically-distinct pair (Fix B).

        :param seed: explicit RNG seed. If ``None``, falls back to
            ``self._reach_task_seed`` (back-compat for callers that
            haven't been migrated to the per-episode
            ``effective_reach_seed`` introduced by Fix B). Callers
            inside :meth:`reset` pass ``effective_reach_seed`` so each
            episode within a scene gets a distinct ``(start, goal)``
            pair.
        :param validator: optional callable
            ``(start_world: np.ndarray, goal_world: np.ndarray) -> bool``.
            For every candidate that passes the
            ``[reach_min_distance, reach_max_distance]`` annulus
            check, the validator is invoked and the candidate is
            accepted only when the validator returns ``True``. The
            primary use is the rotation-invariance rejection loop:
            see :meth:`_validate_rotation_for_targets`. The validator
            is consulted DETERMINISTICALLY per candidate (same seed
            → same candidate sequence → same validator verdicts), so
            adding a validator does not break the
            ``(scene_seed, episode_idx)`` → ``(start, goal)``
            determinism guarantee.

        :returns: ``(start_pos, goal_pos)``, each ``np.ndarray`` of
            shape ``(3,)`` in world-frame metres.
        :raises RuntimeError: if no accepted draw appears within
            :data:`_REACH_SAMPLE_MAX_TRIES` iterations (means the
            requested ``[min, max]`` annulus is empty under the
            current sampler, OR the validator rejected too many
            candidates — widen ``reach_max_distance`` / shrink the
            annulus / loosen the validator tolerance).
        """
        if seed is None:
            seed = self._reach_task_seed
        rng = np.random.RandomState(int(seed))

        radius = self._reach_max_distance
        d_min = self._reach_min_distance
        d_max = self._reach_max_distance
        # Slack matches the JSON round-trip tolerance used by the tests;
        # avoids spurious rejections when ``d_0`` is at the floating-point
        # boundary of the annulus.
        tol = 1e-9

        def _sample_in_ball() -> np.ndarray:
            # Uniform-in-ball: random direction × uniform-in-radius
            # following ``r = R * u**(1/3)``.
            d = rng.normal(size=3)
            d = d / (np.linalg.norm(d) + 1e-12)
            r = radius * (rng.uniform(0.0, 1.0) ** (1.0 / 3.0))
            return d * r

        # Order matches the reach-task narrative: first goes to the
        # GOAL, then to the START (the legacy convention). Each
        # rejection draw consumes 6 RandomState calls (3 normal + 1
        # uniform per point), so 10k tries cap RNG consumption at
        # ~60k draws — negligible at construction time.
        validation_attempts = 0
        for _attempt in range(_REACH_SAMPLE_MAX_TRIES):
            goal_offset = _sample_in_ball()
            start_offset = _sample_in_ball()
            d0 = float(np.linalg.norm(goal_offset - start_offset))
            if d_min - tol <= d0 <= d_max + tol:
                start_world = self._reach_home_eef_pos + start_offset
                goal_world = self._reach_home_eef_pos + goal_offset
                if validator is None or validator(start_world, goal_world):
                    return start_world, goal_world
                validation_attempts += 1
                if validation_attempts >= _REACH_VALIDATION_MAX_TRIES:
                    raise RuntimeError(
                        f"_sample_reach_targets: validator rejected "
                        f"{_REACH_VALIDATION_MAX_TRIES} consecutive distance-"
                        f"accepted candidates (seed={seed}). Either the "
                        f"current scene is structurally infeasible for the "
                        f"rotation invariant or the tolerance is too tight."
                    )

        raise RuntimeError(
            f"_sample_reach_targets: no accepted draw after "
            f"{_REACH_SAMPLE_MAX_TRIES} tries (seed={seed}, "
            f"d_min={d_min}, d_max={d_max}, sample radius={radius}). "
            f"The requested annulus is likely empty under the current "
            f"sampler — widen ``reach_max_distance`` or shrink the "
            f"``reach_min_distance``/``reach_max_distance`` gap."
        )

    def _drive_eef_to_world_target(
        self,
        target_world_pos,
        max_steps: int = _REACH_DRIVE_MAX_STEPS,
        tol: float = _REACH_DRIVE_TOL,
        rot_tol: float = _GOAL_RETURN_HOME_ROT_TOL,
        axis_order: str = "xyz",
        stall_check: bool = False,
    ) -> float:
        """Closed-loop blocking drive of the EE to a world-frame XYZ target,
        actively holding the home arm rotation throughout the drive.

        Mirrors :meth:`_step_blocking_expert` but with an *absolute*
        world target (rather than a base-frame delta). Used by the
        reach-task reset to get the EE to ``start`` or ``goal``
        independently of the underlying ``EXPERT_ACTIONS`` waypoint
        table.

        Per OSC sub-step we send both a position corrective AND a
        rotation corrective:

        * ``corrective_pos = R_bw @ (target_world - eef_now_world)``
        * ``corrective_rot = R_bw @ rot_err_world``

        where ``rot_err_world = (R_home * R_now^{-1}).as_rotvec()`` and
        ``R_home`` is snapshotted from the EE pose at entry. This
        matches the active rotation feedback used by the
        ``return_home`` loop inside :meth:`_run_expert_to_goal` and
        keeps the EE world rotation invariant to the start/goal
        endpoint choice (verified by
        ``test_eef_rotation_matches_between_init_and_goal``).

        Without this, the controller would only see a zero rotation
        delta which it interprets as "hold current orientation" but
        doesn't actively correct: the wrist would drift by ~0.5 rad
        over the 0.3 m drive on a typical reach episode.

        Restores the mobile-base joints after every step (camera-
        stability lock — required so the base frame stays consistent
        with the snapshot taken at reset).

        :param axis_order: order in which to enable per-base-axis
            position corrections. ``"xyz"`` (default) drives all three
            axes simultaneously, preserving the legacy behaviour.
            ``"yxz"`` runs the OSC in three phases: first drives only
            the base-Y component (lateral) until ``|err_y_base| < tol``,
            then enables X (forward), then Z (up). Used by the
            obstacle-reach drive so the EEF goes AROUND the obstacle
            laterally before approaching it forward — without this the
            straight-line drive plows into the fixture and the wrist
            pushes for hundreds of sub-steps before giving up.
        :param stall_check: when True, abort the drive once the EEF
            world position hasn't moved by more than
            :data:`_REACH_DRIVE_STALL_TOL_M` over the last
            :data:`_REACH_DRIVE_STALL_WINDOW` sub-steps. Diagnoses the
            "wrist pushing geometry" failure mode. Off by default to
            preserve legacy reach-drive behaviour; the obstacle-reach
            validator turns it on so the rotation-invariance loop
            rejects unreachable pairs in seconds instead of minutes.

        :returns: residual world-frame Euclidean position error (m) at
            exit (either because both ``tol`` and ``rot_tol`` were
            reached, ``max_steps`` ran out, or the stall check fired).
        """
        target_world_pos = np.asarray(target_world_pos, dtype=np.float64)
        R_wb = self._get_base_world_R()
        R_bw = R_wb.T

        # Snapshot the home arm rotation so we can drive back to it on
        # every step. Quaternion-native (no Euler round-trip) so we
        # avoid the gimbal-lock-driven wrist flips that the
        # ``quat → euler → matrix`` chain triggers near pitch = ±π/2
        # (see :meth:`_get_eef_quat_world` for the rationale).
        home_quat = self._get_eef_quat_world()
        start_rot = Rotation.from_quat(home_quat)

        # Compute the per-base-frame-axis enable masks once. Each phase
        # is a (3,) ndarray of {0, 1}. Phases run sequentially: phase k
        # advances to phase k+1 once the gated axes have converged
        # (|err| < tol on each enabled axis). For the default
        # ``"xyz"`` order there is a single phase with all axes
        # enabled, recovering the legacy behaviour.
        axis_order = axis_order.lower()
        valid = set("xyz")
        if not set(axis_order).issubset(valid) or len(axis_order) != 3:
            raise ValueError(
                f"axis_order must be a permutation of 'xyz', got {axis_order!r}"
            )
        idx_map = {"x": 0, "y": 1, "z": 2}
        if axis_order == "xyz":
            axis_phases = [np.array([1.0, 1.0, 1.0], dtype=np.float64)]
        else:
            axis_phases = []
            mask = np.zeros(3, dtype=np.float64)
            for ch in axis_order:
                mask = mask.copy()
                mask[idx_map[ch]] = 1.0
                axis_phases.append(mask)
        phase_idx = 0

        # Stall tracking: a ring of recent EEF positions; when the
        # max-to-min spread on each axis is below the tol we know the
        # wrist is wedged. Sized to one window's worth so the rolling
        # check is O(1) per sub-step.
        stall_window: list[np.ndarray] = []

        residual = float("inf")
        for _ in range(max_steps):
            eef_now_world = self._get_eef_pos()
            err_world = target_world_pos - eef_now_world
            residual = float(np.linalg.norm(err_world))

            cur_quat = self._get_eef_quat_world(q_ref=home_quat)
            cur_rot = Rotation.from_quat(cur_quat)
            rot_err_world = (start_rot * cur_rot.inv()).as_rotvec()
            rot_err_mag = float(np.linalg.norm(rot_err_world))

            # Per-phase position gating: only the axes in the active
            # phase contribute to the position corrective; the rest are
            # zeroed so the OSC doesn't push along them yet.
            pos_err_base = R_bw @ err_world
            pos_err_base_gated = pos_err_base * axis_phases[phase_idx]

            # Phase advancement: enabled axes converged?
            gated_residual = float(np.linalg.norm(pos_err_base_gated))
            if gated_residual < tol and phase_idx < len(axis_phases) - 1:
                phase_idx += 1
                stall_window.clear()
                continue  # re-evaluate with the wider mask next iter

            # Termination: ALL axes converged AND rotation under tol.
            if (
                phase_idx == len(axis_phases) - 1
                and residual < tol
                and rot_err_mag < rot_tol
            ):
                break

            # Stall bail-out: the wrist is stuck (pushing geometry, IK
            # singular, etc.). Bail so the caller can reject the pair.
            if stall_check:
                stall_window.append(eef_now_world.copy())
                if len(stall_window) > _REACH_DRIVE_STALL_WINDOW:
                    stall_window.pop(0)
                if len(stall_window) >= _REACH_DRIVE_STALL_WINDOW:
                    stack = np.stack(stall_window, axis=0)
                    spread = float(np.max(np.linalg.norm(stack - stack[0], axis=1)))
                    if spread < _REACH_DRIVE_STALL_TOL_M:
                        break

            # Axis-angle rotation vectors transform between frames by
            # left-multiplying with the change-of-basis matrix; the
            # magnitude (angle) is frame-invariant.
            rot_err_base = R_bw @ rot_err_world
            self._env.step(_goal_prepare_axisangle(pos_err_base_gated, rot_err_base))
            if self._lock_base_in_arm_mode:
                self._restore_base_joints()

        return residual

    def _validate_rotation_for_targets(
        self,
        start_world: np.ndarray,
        goal_world: np.ndarray,
        tol: float = _REACH_ROT_INVARIANCE_TOL,
        axis_order: str = "xyz",
        stall_check: bool = False,
        max_steps: int = _REACH_DRIVE_MAX_STEPS,
    ) -> bool:
        """Return True iff driving the EE to ``start`` AND to ``goal``
        both leave the end-effector within ``tol`` rad (geodesic
        SO(3)) of the canonical home rotation snapshotted from the
        pre-drive state.

        Used by :meth:`reset` as the validator passed to
        :meth:`_sample_reach_targets` so the sampler rejects any
        ``(start, goal)`` pair whose actual driven EE rotations would
        deviate from home. Even though
        :meth:`_drive_eef_to_world_target` actively feeds back rotation
        error to the OSC controller, the controller cannot always
        satisfy both position and rotation constraints simultaneously
        (kinematic singularities near joint limits make the wrist
        rotation observably drift on a small fraction of scenes /
        episodes). Rather than tighten the controller (which would
        slow planning across the board), we simply reject the
        offending ``(start, goal)`` pair and let the sampler pick a
        different one in the same annulus.

        IMPORTANT — checks against the HOME rotation (not against each
        other). An earlier version only verified ``drift(R_start, R_goal)``
        which permitted a "double-flip" case where both drives ended at
        the same wrong rotation (the wrist flipped by π on both),
        producing a visually-broken init/goal pair that nevertheless
        passed the symmetric check. Pinning both endpoints to the home
        snapshot is the strict invariant we actually want.

        The method snapshots the full ``mujoco`` sim state at entry
        and restores it on exit (success OR failure), so the caller's
        env state is unchanged. A typical successful validation
        costs ``2 × _REACH_DRIVE_MAX_STEPS`` extra OSC sub-steps
        relative to a single drive; failures cost the same but on
        rejected candidates the sampler picks a different
        ``(start, goal)`` and tries again.
        """
        sim = self._env.sim
        state_before = sim.get_state()
        # The base-joints snapshot is mutated by ``_step_blocking_*``
        # helpers via ``_snapshot_base_joints``; copy by value so the
        # validation drives don't leak base-pose drift back into the
        # caller's state.
        base_qpos_snapshot = (
            np.array(self._base_joint_qpos, dtype=np.float64) if self._base_joint_qpos is not None else None
        )
        # Snapshot the canonical home rotation from the PRE-DRIVE state
        # (after _restore_state below this is what the EE rotation
        # reads back as). Quaternion-native, no Euler intermediate.
        home_quat = self._get_eef_quat_world()
        R_home = Rotation.from_quat(home_quat)
        # Snapshot the home arm joint configuration (init_qpos canonical
        # pose at the start of the reach-task reset). Used by the joint-
        # singularity rejection below to detect IK-branch flips that
        # leave a joint fully extended / at its physical limit even
        # when the EE rotation looks fine.
        home_qpos, joint_lower, joint_upper = self._get_arm_joint_pos_and_limits()

        def _restore_state() -> None:
            sim.set_state(state_before)
            sim.forward()
            if base_qpos_snapshot is not None:
                self._base_joint_qpos = base_qpos_snapshot.copy()

        def _drift_from_home(target_world: np.ndarray) -> tuple:
            """Drive to ``target_world`` and return
            ``(eef_rot_drift_rad, max_joint_deviation_rad,
            min_joint_limit_margin_rad)``.

            * ``eef_rot_drift_rad`` is the geodesic SO(3) distance
              between the post-drive EE rotation and ``R_home``.
            * ``max_joint_deviation_rad`` is the maximum absolute
              per-joint deviation between the post-drive arm qpos
              and ``home_qpos``.
            * ``min_joint_limit_margin_rad`` is the smallest gap
              between any joint and either its lower or upper
              physical limit (small ⇒ joint is fully extended /
              fully bent).
            """
            self._drive_eef_to_world_target(
                np.asarray(target_world, dtype=np.float64),
                axis_order=axis_order,
                stall_check=stall_check,
                max_steps=max_steps,
            )
            # Sign-continuous against the home quaternion so the
            # geodesic distance is never inflated by a q ↔ -q
            # hemisphere flip in the simulator's quat output.
            end_quat = self._get_eef_quat_world(q_ref=home_quat)
            R_end = Rotation.from_quat(end_quat)
            rot_drift = float(np.linalg.norm((R_home * R_end.inv()).as_rotvec()))
            end_qpos, _, _ = self._get_arm_joint_pos_and_limits()
            joint_dev = float(np.max(np.abs(end_qpos - home_qpos)))
            margin = float(np.min(np.minimum(end_qpos - joint_lower, joint_upper - end_qpos)))
            return rot_drift, joint_dev, margin

        try:
            # Start drive: must land within ``tol`` of home rotation AND
            # leave every joint within the deviation/limit-margin caps.
            start_rot_drift, start_joint_dev, start_margin = _drift_from_home(start_world)
            if start_rot_drift > tol:
                return False
            if start_joint_dev > _REACH_JOINT_DEVIATION_MAX_RAD:
                return False
            if start_margin < _REACH_JOINT_LIMIT_MARGIN_RAD:
                return False

            # Restore state for an independent goal drive that starts
            # from the home pose (not from where the start drive
            # finished).
            _restore_state()

            # Goal drive: same checks against the same home snapshots.
            goal_rot_drift, goal_joint_dev, goal_margin = _drift_from_home(goal_world)
            if goal_rot_drift > tol:
                return False
            if goal_joint_dev > _REACH_JOINT_DEVIATION_MAX_RAD:
                return False
            if goal_margin < _REACH_JOINT_LIMIT_MARGIN_RAD:
                return False

            return True
        finally:
            _restore_state()

    # ------------------------------------------------------------------
    # Reach-task workspace clearing (used by reset() when reach_task=True)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_structural_fixture(fixture) -> bool:
        """Return True iff the fixture's MRO contains a structural base
        class (cabinet / counter / wall / floor / appliance built into
        the kitchen architecture). These are kept; everything else
        (small countertop appliances such as StandMixer, Toaster,
        Blender, ...) is candidate for hiding by the reach-task
        workspace clearer.

        :seealso: :data:`_REACH_STRUCTURAL_BASE_CLASSES`.
        """
        for base in type(fixture).__mro__:
            if base.__name__ in _REACH_STRUCTURAL_BASE_CLASSES:
                return True
        return False

    def _body_subtree_ids(self, root_bid: int) -> list:
        """Return ``[root_bid, *descendant_body_ids]`` (in unspecified
        order) for ``root_bid`` in the underlying ``mujoco`` model.
        Used to walk every body that belongs to a fixture so we can
        reach all of its geoms (e.g. ``StandMixer`` has a separate
        body for the bowl, head, blender, knob_speed, ...).
        """
        model = self._env.sim.model
        out = [root_bid]
        for bid in range(model.nbody):
            if bid == root_bid:
                continue
            ancestor = bid
            # Walk up the parent chain. mujoco's worldbody has parent
            # id 0 (and is itself id 0), so the loop terminates.
            while ancestor != 0:
                parent = model.body_parentid[ancestor]
                if parent == ancestor:
                    break
                if parent == root_bid:
                    out.append(bid)
                    break
                ancestor = parent
        return out

    def _hide_nearby_appliances(self, eef_pos, radius: float, except_fixture_name: str | None = None) -> list:
        """Hide non-structural fixtures within ``radius`` (m) of
        ``eef_pos`` (world-frame XYZ) by zeroing every covered geom's
        rgba alpha + contype + conaffinity.

        Effects:
            * Visible: the geom no longer renders (alpha=0).
            * Physics: the geom no longer collides
              (contype=conaffinity=0). The body still exists in the
              kinematic tree (no joint surgery, so cabinets / drawers
              that depend on the larger fixture chain are unaffected).

        Why class-based filtering:
            See :data:`_REACH_STRUCTURAL_BASE_CLASSES` for the
            allow-list of structural classes that are NEVER hidden
            (cabinets, counters, walls, floor, stove, hood, sink,
            microwave, oven, dishwasher, fridge, stool, window,
            wall accessory, structural primitive boxes). This is
            scene-agnostic: it works against any robocasa kitchen and
            against any underlying ``--task_name`` because it
            classifies by ``type(fixture).__mro__`` rather than by a
            hard-coded fixture name.

        Why class+radius (instead of every non-structural fixture in
        the scene): the user only wants the *closest* clutter cleared
        \u2014 things on counters near the workspace. Distant decorations
        (fruit bowls on the far counter, etc.) stay in the scene.

        Why gated by ``reach_task``: this method is only ever called
        from :meth:`reset` when ``self._reach_task`` is True. The
        regular ``CloseStandMixerHead`` path (or any other robocasa
        task in normal mode) never sees this clearing, so the existing
        ``EXPERT_ACTIONS`` waypoint replays still work bit-identically.

        :param except_fixture_name: optional robocasa fixture name (the
            key into ``env.fixtures``) that should NOT be hidden even
            if it matches every other hide criterion. Used by the
            obstacle-reach task to keep ONE chosen obstacle visible
            and colliding while everything else in the radius is
            cleared.

        :returns: list of ``(fixture_name, fixture_class_name, dist_m)``
            tuples for fixtures that were hidden, sorted by distance.
        """
        sim = self._env.sim
        model = sim.model
        fixtures = getattr(self._env, "fixtures", None)
        if not fixtures:
            return []

        eef_pos = np.asarray(eef_pos, dtype=np.float64)
        hidden = []
        for name, fixture in fixtures.items():
            if except_fixture_name is not None and name == except_fixture_name:
                continue
            if self._is_structural_fixture(fixture):
                continue
            root_body_name = getattr(fixture, "root_body", None)
            if root_body_name is None:
                continue
            try:
                root_bid = model.body_name2id(root_body_name)
            except Exception:
                continue
            root_pos = np.asarray(sim.data.body_xpos[root_bid], dtype=np.float64)
            d = float(np.linalg.norm(root_pos - eef_pos))
            if d > radius:
                continue
            for bid in self._body_subtree_ids(root_bid):
                n_geoms = int(model.body_geomnum[bid])
                geom_start = int(model.body_geomadr[bid])
                for i in range(n_geoms):
                    gid = geom_start + i
                    model.geom_rgba[gid, 3] = 0.0
                    model.geom_contype[gid] = 0
                    model.geom_conaffinity[gid] = 0
            hidden.append((name, type(fixture).__name__, d))
        if hidden:
            sim.forward()
        hidden.sort(key=lambda t: t[2])
        return hidden

    def _run_expert_to_goal(self):
        """Drive ``self._env`` through the per-task expert action sequence.

        The keypoints in :data:`EXPERT_ACTIONS` are **base-frame** EE
        displacements. For each sub-step we:

        1. Convert the base-frame target into world for closed-loop
           comparison against the world-frame EE observation.
        2. Convert the residual world error back into the base frame and
           send it as a 12-D layout-B corrective action via
           :func:`_goal_prepare_action` (controller's
           ``input_ref_frame="base"``).
        3. Restore the mobile-base joint qpos after every sim step
           (camera-stability lock; required so the base frame stays
           consistent with the snapshot taken at the start).

        Bypasses the new layout-A :meth:`step` path because the
        simulator's robosuite controller consumes layout B directly.
        """
        task_name = self._config["task_name"]
        seed = self._config.get("seed")
        waypoints = _goal_chunk_waypoints(task_name, seed)

        # Snapshot base orientation once (base is locked through this loop).
        R_wb = self._get_base_world_R()
        R_bw = R_wb.T

        start_pose = self._get_eef_pose7()
        start_pos_world = start_pose[:3].copy()
        # Quaternion-native snapshot (avoids gimbal-lock at pitch=±π/2).
        home_quat = self._get_eef_quat_world()
        start_rot = Rotation.from_quat(home_quat)

        for label, actions in waypoints:
            if label == "return_home":
                for _ in range(_GOAL_RETURN_HOME_MAX_STEPS):
                    eef_now_world = self._get_eef_pos()
                    pos_err_world = start_pos_world - eef_now_world
                    cur_quat = self._get_eef_quat_world(q_ref=home_quat)
                    cur_rot = Rotation.from_quat(cur_quat)
                    rot_err_world = (start_rot * cur_rot.inv()).as_rotvec()
                    if (
                        np.linalg.norm(pos_err_world) < _GOAL_RETURN_HOME_POS_TOL
                        and np.linalg.norm(rot_err_world) < _GOAL_RETURN_HOME_ROT_TOL
                    ):
                        break
                    pos_err_base = R_bw @ pos_err_world
                    # Rotation error: axis-angle vector transforms from
                    # world to base by left-multiplying with R_bw (the
                    # rotation axis is a 3-vector that lives in space
                    # and must be re-expressed in the base frame). The
                    # angle (vector magnitude) is frame-invariant.
                    rot_err_base = R_bw @ rot_err_world
                    self._env.step(_goal_prepare_axisangle(pos_err_base, rot_err_base))
                    if self._lock_base_in_arm_mode:
                        self._restore_base_joints()
            else:
                # ``actions`` is (N, 7) of per-step base-frame deltas.
                for action7_base in actions:
                    self._step_blocking_expert(action7_base, R_wb, R_bw)

    def _step_blocking_expert(self, action7_base, R_wb, R_bw):
        """Per-step closed-loop blocking with explicit world↔base bridging.

        ``action7_base`` is ``[dx, dy, dz, droll, dpitch, dyaw, dgrip]``
        with the xyz delta expressed in the **base** frame. The closed
        loop measures error in the world frame (because the EE
        observation is world-frame), then transforms the error back to
        base before sending as the corrective.
        """
        action7_base = np.asarray(action7_base, dtype=np.float64)
        eef_before_world = self._get_eef_pos()
        target_world = eef_before_world + R_wb @ action7_base[:3]

        for i in range(_GOAL_BLOCKING_MAX_STEPS):
            eef_now_world = self._get_eef_pos()
            err_world = target_world - eef_now_world
            if np.linalg.norm(err_world) < _GOAL_BLOCKING_TOL:
                break
            corrective = np.zeros(7, dtype=np.float64)
            corrective[:3] = R_bw @ err_world
            if i == 0:
                corrective[3:6] = action7_base[3:6]
            corrective[6] = action7_base[6]
            self._env.step(_goal_prepare_action(corrective))
            if self._lock_base_in_arm_mode:
                self._restore_base_joints()

    def _reach_apply_orientation_hold(self, sub: np.ndarray) -> np.ndarray:
        """In-place rewrite one OSC sub-step's 12-D layout-A action so
        the EE world rotation is driven back to
        ``self._reach_home_eef_quat`` and the commanded position delta
        magnitude is bounded by :data:`_REACH_STEP_MAX_POS_NORM`.

        Used only by :meth:`step` when ``self._reach_task`` is true.

        The two corrections are independent of the planner's
        commanded rotation/gripper channels: ``sub[8:11]`` is fully
        overridden every sub-step, regardless of what the upstream
        action contained. This implements the per-step orientation
        invariant requested in the plan ("every step in the reach env
        should preserve the end-effector orientation").

        Reads ``robot0_eef_quat`` via :meth:`_get_eef_quat_world`
        (sign-continuous against the home reference, no Euler
        round-trip) so the rotation feedback never triggers the
        gimbal-lock-driven wrist flip near pitch = ±π/2.

        :param sub: one row of the canonical layout-A chunk
            (shape ``(12,)``, dtype float64). Modified in place AND
            returned for convenience.
        :returns: the modified sub-step.
        """
        if self._reach_home_eef_quat is None:
            return sub

        R_wb = self._get_base_world_R()
        R_bw = R_wb.T

        # 0) WORKSPACE BOUND. Project the commanded EE delta so the
        #    new EE world position stays inside a sphere of radius
        #    ``_REACH_WORKSPACE_R`` centred at ``_reach_home_eef_pos``.
        #    This is the only mechanism that reliably prevents the
        #    elbow-fully-straight singularity the planner used to
        #    drive the arm into (sc022, sc035, sc049 etc. on v3/v4/v5).
        #    The earlier "joint margin guard" below catches degenerate
        #    states once we are already in them; the workspace bound
        #    prevents getting there in the first place.
        if self._reach_home_eef_pos is not None:
            # Convert the commanded normalised delta into a metric
            # world-frame displacement.
            pos_norm_base = np.asarray(sub[5:8], dtype=np.float64)
            pos_meters_base = pos_norm_base * _GOAL_OSC_OUTPUT_MAX[:3]
            pos_meters_world = R_wb @ pos_meters_base

            eef_now = self._get_eef_pos()
            home = np.asarray(self._reach_home_eef_pos, dtype=np.float64)
            a = eef_now - home  # current offset from home
            d = pos_meters_world  # proposed displacement in world
            R = _REACH_WORKSPACE_R
            a_norm = float(np.linalg.norm(a))
            a_dot_d = float(np.dot(a, d))
            d_norm_sq = float(np.dot(d, d))

            if d_norm_sq < 1e-18:
                # Zero command — nothing to project.
                scale = 1.0
            elif a_norm >= R - 1e-9:
                # Already outside (or right at) the sphere. Allow only
                # inward motion; reject any outward push.
                scale = 0.0 if a_dot_d > 0 else 1.0
            else:
                # Inside. Find max t in [0, 1] such that
                # ||a + t * d|| <= R, i.e. solve the quadratic
                # d^2 t^2 + 2 (a · d) t + (a^2 - R^2) <= 0 for t.
                disc = a_dot_d * a_dot_d - d_norm_sq * (a_norm * a_norm - R * R)
                if disc < 0:
                    scale = 1.0  # numerically should not happen since C < 0
                else:
                    t_pos = (-a_dot_d + np.sqrt(disc)) / d_norm_sq
                    scale = float(max(0.0, min(1.0, t_pos)))

            if scale < 1.0:
                # Re-encode the projected delta back into the
                # normalised OSC action space.
                pos_meters_world_proj = d * scale
                pos_meters_base_proj = R_bw @ pos_meters_world_proj
                sub[5:8] = np.clip(pos_meters_base_proj / _GOAL_OSC_OUTPUT_MAX[:3], -1.0, 1.0)

        # 1) Sphere-clamp the commanded position delta (normalised
        #    OSC units). Direction is preserved; magnitude is bounded.
        #    Defensive — the workspace projection above usually
        #    already keeps things in check, but this guards against
        #    pathological cases where the planner sends a delta
        #    much larger than 1.0 in some component.
        pos = sub[5:8]
        pos_norm = float(np.linalg.norm(pos))
        if pos_norm > _REACH_STEP_MAX_POS_NORM:
            sub[5:8] = pos * (_REACH_STEP_MAX_POS_NORM / pos_norm)

        # 1b) Joint-singularity guard. If any arm joint is within
        #     ``_REACH_JOINT_LIMIT_MARGIN_RAD`` of its lower or upper
        #     physical limit, the OSC's IK becomes degenerate and the
        #     EE either freezes or thrashes (the "fully-straight arm"
        #     mode visible in the planner videos for sc032 ep2,
        #     sc035 ep8, sc049 ep8, etc.).
        #     When that happens, ZERO the commanded position delta —
        #     the rotation corrective below will still pull the wrist
        #     back toward ``_reach_home_eef_quat``, which mechanically
        #     relaxes the offending joint AWAY from its limit. Once
        #     the joint clears the margin (typically within 1-2
        #     sub-steps), normal position commands resume. With the
        #     workspace bound (step 0) in place this branch should
        #     almost never fire; it's kept as a defensive backstop.
        qpos, j_lower, j_upper = self._get_arm_joint_pos_and_limits()
        joint_margin = float(np.min(np.minimum(qpos - j_lower, j_upper - qpos)))
        if joint_margin < _REACH_JOINT_LIMIT_MARGIN_RAD:
            sub[5:8] = 0.0

        # 2) Override the rotation channel with a corrective toward
        #    the home quaternion. Quaternion-native math: no Euler.
        cur_quat = self._get_eef_quat_world(q_ref=self._reach_home_eef_quat)
        R_home = Rotation.from_quat(self._reach_home_eef_quat)
        R_now = Rotation.from_quat(cur_quat)
        rot_err_world = (R_home * R_now.inv()).as_rotvec()
        # World → base frame. R_bw was computed once at the top of
        # the function (step 0); reuse it here.
        rot_err_base = R_bw @ rot_err_world
        # Express in the planner-normalised range: divide by the OSC
        # output_max for the rotation channel, clip to [-1, +1].
        sub[8:11] = np.clip(rot_err_base / _GOAL_OSC_OUTPUT_MAX[3:6], -1.0, 1.0)
        return sub

    def step(self, action):
        """Execute one WM tick: ``chunk_size`` robosuite sub-steps.

        Parameters
        ----------
        action : dict | array-like
            One of:
            * components dict (preferred for the planner)
            * canonical layout A ``(chunk_size, 12)`` or
              ``(chunk_size * 12,)`` numpy array (legacy / test scripts)
            * 7-D legacy ``[dx, dy, dz, droll, dpitch, dyaw, d_grip]``
              numpy array

        Returns
        -------
        obs : dict       — observation after the last sub-step
        reward : float   — robosuite reward of the last sub-step
        done : bool      — robosuite done flag of the last sub-step
        info : dict      — ``{"success": bool, "sim_steps": int, "control_modes": list[int]}``
        """
        chunk = self._coerce_action_to_chunk(action).copy()

        reward = 0.0
        done = False
        modes = []

        for i in range(self._chunk_size):
            sub = chunk[i]
            # Reach-task only: clamp the commanded position to a safe
            # range and overwrite the rotation channel with a
            # corrective toward the home EE quaternion BEFORE the OSC
            # consumes the action. This enforces the per-step
            # orientation invariant asserted by
            # ``test_eef_rotation_held_across_steps``.
            if self._reach_task:
                sub = self._reach_apply_orientation_hold(sub)
            cm_value = +1.0 if sub[4] >= 0.5 else -1.0
            modes.append(int(cm_value))

            full = _layoutA_to_robosuite12(sub)
            _, reward, done, _ = self._env.step(full)

            # Camera-stability lock: in arm mode, restore the base qpos
            # we snapshotted last; in base mode, accept the new state.
            if self._lock_base_in_arm_mode and cm_value < 0:
                self._restore_base_joints()
            else:
                self._snapshot_base_joints()

        success = self._env._check_success()
        info = {
            "success": bool(success),
            "sim_steps": self._chunk_size,
            "control_modes": modes,
        }
        out_obs = self._build_obs()

        # Reach-task reward (overrides the underlying robocasa reward,
        # which is task-specific and meaningless when the env is being
        # used as a generic kitchen backdrop). Formula:
        #     r_t = max(0, 1 - ||eef_t - goal|| / d_0)
        # where ``d_0 = ||start - goal||`` was anchored at the most
        # recent ``reset(mode='init' | 'goal')``. r_0 = 0 (eef = start)
        # and r_T = 1 when the EEF reaches the goal. ``r_t`` is
        # monotonically increasing only for a perfectly straight-line
        # policy; in practice CEM rollouts overshoot and so r_t can
        # decrease step-to-step.
        if self._reach_task and self._reach_goal_world is not None:
            eef_now = self._get_eef_pos()
            dist_to_goal = float(np.linalg.norm(eef_now - self._reach_goal_world))
            if self._reach_d0 > 1e-9:
                reward = max(0.0, 1.0 - dist_to_goal / self._reach_d0)
            else:
                # Degenerate task (start == goal). Treat as already-solved.
                reward = 1.0
            out_obs["reach_distance_to_goal"] = dist_to_goal
            out_obs["reach_d0"] = self._reach_d0
            out_obs["reach_goal_world"] = self._reach_goal_world.copy()
            info["reach_distance_to_goal"] = dist_to_goal
            info["reach_d0"] = self._reach_d0
            info["reach_reward"] = float(reward)

        return out_obs, float(reward), bool(done), info

    def step_chunk(self, actions, dt=None):
        """Execute several WM ticks back-to-back.

        Parameters
        ----------
        actions : list | array-like
            One of:
            * ``list[dict]`` / ``list[ndarray]`` / ``list`` of mixed —
              dispatched element-wise through :meth:`step`. Preferred for
              the planner (logical-components dicts).
            * ``(N, chunk_size, 12)`` or ``(N, chunk_size * 12)`` numpy —
              canonical layout A (legacy / test scripts).
            * ``(N, 7)`` numpy — legacy DROID-style 7-D rows; each row is
              split via :func:`split_7d_action` and broadcast.
        dt : float or None
            Optional wall-clock spacing between ticks (seconds). ``None``
            (default) runs as fast as possible. Useful for live-stream
            paced playback.

        Returns
        -------
        obs : dict       — observation after the final WM tick
        reward : float   — reward at the final sub-step
        done : bool      — done flag at the final sub-step
        info : dict      — ``info["chunk_obs"]`` is a list of N obs dicts
                           (one per WM tick); plus the usual single-tick keys.
        """
        # 1) Normalise into an iterable of per-tick actions.
        if isinstance(actions, list) or (
            isinstance(actions, tuple) and len(actions) > 0 and not isinstance(actions[0], (int, float))
        ):
            tick_actions = list(actions)
        else:
            arr = np.asarray(actions, dtype=np.float64)
            if arr.ndim == 2:
                if arr.shape[1] == 7:  # legacy 7-D rows
                    tick_actions = [arr[i] for i in range(arr.shape[0])]
                else:
                    assert arr.shape[1] == self._chunk_size * ACTION_PER_STEP_DIM, (
                        f"flat row must be size 7 (legacy) or {self._chunk_size * ACTION_PER_STEP_DIM}, "
                        f"got {arr.shape[1]}"
                    )
                    arr = arr.reshape(-1, self._chunk_size, ACTION_PER_STEP_DIM)
                    tick_actions = [arr[i] for i in range(arr.shape[0])]
            elif arr.ndim == 3:
                assert arr.shape[1:] == (self._chunk_size, ACTION_PER_STEP_DIM), (
                    f"3-D actions must be (N, {self._chunk_size}, {ACTION_PER_STEP_DIM}), got {arr.shape}"
                )
                tick_actions = [arr[i] for i in range(arr.shape[0])]
            else:
                raise ValueError(f"actions must be list, 2-D, or 3-D; got ndim={arr.ndim}")

        # 2) Step through each tick, using the unified per-tick path.
        chunk_obs = []
        chunk_rewards = []
        chunk_infos = []
        obs = None
        reward = 0.0
        done = False
        info = {}
        for tick in tick_actions:
            t0 = time.perf_counter()
            obs, reward, done, info = self.step(tick)
            chunk_obs.append(obs)
            chunk_rewards.append(float(reward))
            chunk_infos.append(info)
            if dt is not None:
                left = dt - (time.perf_counter() - t0)
                if left > 0:
                    time.sleep(left)

        info = dict(info)
        info["chunk_obs"] = chunk_obs
        # Per-tick rewards (one entry per :meth:`step` call). Used by
        # the planner to compute the discounted return without losing
        # the intra-chunk granularity (``step_chunk`` only returns the
        # FINAL tick's reward in its top-level position).
        info["chunk_rewards"] = chunk_rewards
        return obs, reward, done, info

    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
