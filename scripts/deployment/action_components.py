# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Logical-component action helpers shared by env wrappers and the planner.

The "components dict" is the unified action format passed across the
``GymEnvServer`` boundary:

::

    {
        "eef_translation": (3,) float32,    # dx, dy, dz   (m)
        "eef_rotation":    (3,) float32,    # axis-angle delta (rad)
        "gripper":         (1,) float32,    # binary or continuous, robot-defined
        "joints":          (Nj,) float32,   # optional joint deltas
        # robocasa-specific extras (default = zeros, planner doesn't set)
        "base_motion":     (4,) float32,    # forward, side, yaw, torso
        "control_mode":    (1,) float32,    # ±1
    }

This module provides:

* :func:`split_7d_action` — convert the planner's 7-D
  ``[dx, dy, dz, droll, dpitch, dyaw, d_grip]`` (DROID-style, Euler
  rotation) to the components dict, running the closed-form Euler→
  axis-angle conversion (numpy mirror of
  ``app.robo_jepa.embodiment_preprocessing.euler_xyz_to_axis_angle``).
* :func:`merge_components` — inverse, for env wrappers that internally
  still want a flat raw vector of a given embodiment-specific layout.
  Note that this **converts the rotation back from axis-angle to the
  layout the embodiment expects**: the canonical PandaOmron layout A
  keeps axis-angle (matches the OSC controller's input space), while
  every other embodiment (DROID, franka via deoxys OSC_POSE, SO-101,
  mock) gets Euler XYZ — which is the legacy 7-D contract those env
  wrappers' ``step()`` methods were always written against.
* :func:`is_components_dict` — quick test used by the env wrappers' 7-D
  legacy shim.

The numpy implementations here are kept consistent (bit-for-bit when
possible) with the pytorch versions in
``app/robo_jepa/embodiment_preprocessing.py``.
"""

from __future__ import annotations

import numpy as np

# ── canonical PandaOmron layout A constants (mirror of preprocessor) ─────
ROBOCASA_ACTION_PER_STEP = 12


# ── Euler-XYZ → axis-angle (numpy, mirrors the pytorch version) ─────────


def _euler_xyz_to_matrix_np(euler: np.ndarray) -> np.ndarray:
    """Numpy ``(..., 3) → (..., 3, 3)`` Euler-XYZ rotation matrix.

    Extrinsic rotation (matches ``Rotation.from_euler('xyz', ...)`` with
    lowercase axis string), so ``R = Rz @ Ry @ Rx``.
    """
    e = np.asarray(euler, dtype=np.float64)
    rx, ry, rz = e[..., 0], e[..., 1], e[..., 2]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    zero = np.zeros_like(rx)
    one = np.ones_like(rx)

    Rx = np.stack(
        [
            np.stack([one, zero, zero], axis=-1),
            np.stack([zero, cx, -sx], axis=-1),
            np.stack([zero, sx, cx], axis=-1),
        ],
        axis=-2,
    )
    Ry = np.stack(
        [
            np.stack([cy, zero, sy], axis=-1),
            np.stack([zero, one, zero], axis=-1),
            np.stack([-sy, zero, cy], axis=-1),
        ],
        axis=-2,
    )
    Rz = np.stack(
        [
            np.stack([cz, -sz, zero], axis=-1),
            np.stack([sz, cz, zero], axis=-1),
            np.stack([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )
    return Rz @ Ry @ Rx


def _matrix_to_axis_angle_np(R: np.ndarray) -> np.ndarray:
    """Numpy ``(..., 3, 3) → (..., 3)`` rotation-matrix → axis-angle."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    skew = np.stack([rx, ry, rz], axis=-1)

    sin_theta = np.sin(theta)
    # avoid divide-by-zero in the masked-off branch (np.where computes both
    # sides eagerly); the result of the False branch is discarded.
    safe_denom = np.where(np.abs(sin_theta) > 1e-6, 2.0 * sin_theta, 1.0)
    safe = np.where(
        np.abs(sin_theta) > 1e-6,
        theta / safe_denom,
        0.5 + theta * theta / 12.0,
    )
    axis_angle = skew * safe[..., None]

    near_pi = theta > (np.pi - 1e-3)
    if np.any(near_pi):
        m = R + np.eye(3, dtype=R.dtype)
        diag = np.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], axis=-1).clip(min=0.0)
        ax = np.sqrt(diag * 0.5 + 1e-12)
        signs = np.stack(
            [np.sign(rx), np.sign(ry), np.sign(rz)],
            axis=-1,
        )
        signs = np.where(signs == 0.0, 1.0, signs)
        ax = ax * signs * theta[..., None]
        axis_angle = np.where(near_pi[..., None], ax, axis_angle)

    return axis_angle


def euler_xyz_to_axis_angle_np(euler: np.ndarray) -> np.ndarray:
    """Numpy mirror of
    :func:`app.robo_jepa.embodiment_preprocessing.euler_xyz_to_axis_angle`."""
    return _matrix_to_axis_angle_np(_euler_xyz_to_matrix_np(euler))


# ── Components dict helpers ──────────────────────────────────────────────


def is_components_dict(action) -> bool:
    """Return ``True`` if ``action`` is a logical-components dict.

    Used by env wrappers' 7-D legacy shim to decide whether to route the
    incoming action through :func:`split_7d_action`.
    """
    return isinstance(action, dict) and ("eef_translation" in action or "joints" in action)


def split_7d_action(action_7d) -> dict:
    """Split a planner 7-D ``[dx, dy, dz, droll, dpitch, dyaw, d_grip]``
    array into the unified components dict.

    The Euler rotation is converted to axis-angle so the env wrappers
    can pack it into whatever raw format their robot expects.

    Returns a dict with the keys::

        eef_translation (3,) float32
        eef_rotation    (3,) float32   axis-angle
        gripper         (1,) float32   raw value (env wrapper may
                                       binarise / clip)

    Optional keys (``joints``, ``base_motion``, ``control_mode``) are
    not set; env wrappers fill in their own defaults.
    """
    a = np.asarray(action_7d, dtype=np.float32).reshape(-1)
    assert a.shape == (7,), f"expected 7-D action, got shape {a.shape}"
    eef_translation = a[0:3].astype(np.float32)
    eef_rotation = euler_xyz_to_axis_angle_np(a[3:6]).astype(np.float32)
    gripper = a[6:7].astype(np.float32)
    return {
        "eef_translation": eef_translation,
        "eef_rotation": eef_rotation,
        "gripper": gripper,
    }


def _components_or_zeros(comp: dict, key: str, n: int) -> np.ndarray:
    val = comp.get(key)
    if val is None:
        return np.zeros(n, dtype=np.float32)
    return np.asarray(val, dtype=np.float32).reshape(-1)


def _binarise(x: float | np.ndarray, threshold: float = 0.5) -> float:
    """RoboCasa convention: thresholded ±1 binary signal."""
    v = float(np.asarray(x).reshape(-1)[0])
    return 1.0 if v >= threshold else -1.0


# RoboCasa OSC controller scaling. The training-time canonical action is
# ``[-1, 1]`` normalised; the controller multiplies the input by these
# ``output_max`` values to produce the actual per-OSC-sub-step delta in
# metres / radians. The same canonical 12-D action is broadcast over
# all ``chunk_size`` OSC sub-steps within a chunk, so the planner-side
# metric values are normalised by ``output_max * chunk_size`` (the
# per-WM-step physical motion ceiling). Keep this in sync with
# ``app.robo_jepa.embodiment_preprocessing`` (it's the same OSC
# controller on both sides).
_ROBOCASA_CHUNK_SIZE = 4
_ROBOCASA_OSC_POS_MAX = 0.05  # metres per OSC sub-step
_ROBOCASA_OSC_ROT_MAX = 0.5  # radians per OSC sub-step
_ROBOCASA_PER_WM_STEP_POS_MAX = _ROBOCASA_OSC_POS_MAX * _ROBOCASA_CHUNK_SIZE  # 0.20 m
_ROBOCASA_PER_WM_STEP_ROT_MAX = _ROBOCASA_OSC_ROT_MAX * _ROBOCASA_CHUNK_SIZE  # 2.0 rad


def merge_components(components: dict, embodiment: str) -> np.ndarray:
    """Pack a components dict into a flat raw vector for ``embodiment``.

    Currently implemented:

    * ``franka_omron`` — canonical PandaOmron layout A (12-D); applies
      RoboCasa server-side defaults (zero base_motion, ``control_mode = -1``)
      when the dict omits those keys, **normalises** the metric
      ``eef_translation`` (m) and ``eef_rotation`` (rad axis-angle) by
      the OSC controller's ``output_max`` and clips to ``[-1, 1]`` so the
      training-distribution canonical layout is preserved, and binarises
      the gripper. Rotation stays in axis-angle (the layout the OSC
      controller wants).

    Other embodiments fall back to a flat 7-D
    ``[eef_translation, eef_rotation, gripper]`` vector — matches the
    DROID / Franka (deoxys OSC_POSE) / SO-101 / mock contract, which is
    Euler XYZ in slots 3:6 (the DROID cartesian action / observation
    layout and the deoxys ``Rotation.from_euler('xyz', ...)`` call). The dict's axis-angle
    ``eef_rotation`` is therefore converted back to Euler XYZ here.
    """
    if embodiment == "franka_omron" or embodiment == "franka_omron_state_delta":
        base = _components_or_zeros(components, "base_motion", 4)
        cm = components.get("control_mode")
        if cm is None:
            control_mode = -1.0
        else:
            control_mode = _binarise(cm)
        eef_pos_m = _components_or_zeros(components, "eef_translation", 3)
        eef_rot_rad = _components_or_zeros(components, "eef_rotation", 3)
        # metric → normalised + clip (matches the training canonical action,
        # see _goal_prepare_action which does the same thing for the
        # expert-actions goal generator).
        eef_pos = np.clip(eef_pos_m / _ROBOCASA_OSC_POS_MAX, -1.0, 1.0)
        if embodiment == "franka_omron_state_delta":
            # Per the agreed temporary contract for the state-delta
            # variant: planner-side rotation slots must be zero, and the
            # env-side OSC controller is sent a rotation noop. (Will be
            # lifted later when proper Δquat handling is added.)
            rot_max = float(np.abs(eef_rot_rad).max()) if eef_rot_rad.size else 0.0
            assert rot_max < 1e-6, (
                "merge_components(franka_omron_state_delta): planner "
                "rotation slots must be zero; got "
                f"|eef_rotation|.max()={rot_max:.6e}"
            )
            eef_rot = np.zeros(3, dtype=eef_rot_rad.dtype)
        else:
            eef_rot = np.clip(eef_rot_rad / _ROBOCASA_OSC_ROT_MAX, -1.0, 1.0)
        gripper = components.get("gripper")
        if gripper is None:
            grip = -1.0
        else:
            grip = _binarise(gripper)
        return np.concatenate(
            [
                base,
                np.array([control_mode], dtype=np.float32),
                eef_pos,
                eef_rot,
                np.array([grip], dtype=np.float32),
            ]
        ).astype(np.float32)

    # default 7-D layout: convert eef_rotation back from axis-angle (the
    # contract documented at the top of this module) to Euler XYZ, since
    # all the legacy embodiments using this branch (droid, franka via
    # deoxys OSC_POSE, so101, mock) interpret slots 3:6 as Euler XYZ.
    # See e.g. DROID's cartesian action `[vx,vy,vz,ωx,ωy,ωz]` and cartesian
    # observation `[x,y,z,roll,pitch,yaw]`, and franka deoxys's
    # `Rotation.from_euler("xyz", commanded_pose[3:6])` in the OSC path.
    eef_pos = _components_or_zeros(components, "eef_translation", 3)
    eef_rot_axis_angle = _components_or_zeros(components, "eef_rotation", 3)
    if np.any(eef_rot_axis_angle):
        # scipy handles the |axis|>=π corner case; for the common small
        # rotations the small-angle approximation makes Euler ≈ rotvec
        # (which is why the bug stayed hidden for translation-only runs).
        from scipy.spatial.transform import Rotation as _R

        eef_rot = _R.from_rotvec(eef_rot_axis_angle.astype(np.float64)).as_euler("xyz").astype(np.float32)
    else:
        eef_rot = eef_rot_axis_angle  # all zeros either way
    gripper = _components_or_zeros(components, "gripper", 1)
    return np.concatenate([eef_pos, eef_rot, gripper]).astype(np.float32)
