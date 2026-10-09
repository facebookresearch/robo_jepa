# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-embodiment action / state preprocessing for the world model.

Single source of truth for "how a ``(B, T, K)`` raw action becomes the
``(B, T, action_dim)`` tensor the WM consumes" and the analogous mapping
for state.

All functions are pure-pytorch and fully batched (no Python loops over
``B`` or ``T``).

Design
------
``WorldModel.step`` receives a 7-D planner action plus a state object that
carries an embodiment id per batch element. Different embodiments expect
different action / state layouts inside the WM tensor. Rather than
hard-coding ``F.pad`` everywhere, ``WorldModel.step`` looks the embodiment
up in the registries below and applies the corresponding callable.

Two registries:

* ``ACTION_PREPROCESSORS[embodiment_name] -> callable``
* ``STATE_PREPROCESSORS[embodiment_name] -> callable``

Both callables receive ``(x, target_dim)`` and must return a tensor whose
last-dim equals ``target_dim``. Adding a new embodiment is a one-line
registry edit at the bottom of this file.

Currently registered:

* ``franka_omron``: RoboCasa PandaOmron (canonical "layout A" 12-D action,
  expanded over a chunk to (chunk_size * 12)). State input is zero-padded
  to ``state_dim`` to mirror the dataset's
  ``action_source='action_key'`` regime where state was identically zero
  during training.
* (default fallback): ``pad_preprocess`` — zero-pad/truncate the last
  dim. Matches the original DROID-style behaviour.
"""

from __future__ import annotations

from typing import Callable

import torch


# Canonical PandaOmron action layout (RoboCasa modality.json layout A):
#   [0:4]   base_motion (base_x, base_y, base_yaw, torso)
#   [4]     control_mode  (binary, ±1; -1 = arm, +1 = base)
#   [5:8]   ee_pos delta
#   [8:11]  ee_rot delta (axis-angle)
#   [11]    gripper      (binary, ±1)
ROBOCASA_ACTION_PER_STEP = 12
ROBOCASA_DEFAULT_CHUNK = 4
ROBOCASA_DEFAULT_PAD = ROBOCASA_ACTION_PER_STEP * ROBOCASA_DEFAULT_CHUNK  # 48

# RoboCasa OSC controller scaling. The training-time canonical action is
# ``[-1, 1]`` normalised; the controller multiplies the input by these
# ``output_max`` values to produce the actual per-OSC-sub-step delta in
# metres / radians. The WM consumes one chunked action per WM step,
# where the same per-sub-step value is broadcast over
# ``ROBOCASA_DEFAULT_CHUNK`` OSC sub-steps. So the **per-WM-step**
# physical motion ceiling is ``output_max * chunk_size``, and the
# planner's metric intent should be normalised by that combined factor
# (so a planner action of 0.20 m / WM step → canonical 1.0, fully
# saturated; planner action 0.10 m / WM step → canonical 0.5, well
# inside the training distribution). Keep this in sync with
# ``scripts.robo_jepa.action_components`` (it's the same OSC controller
# on both sides).
_ROBOCASA_OSC_POS_MAX = 0.05   # metres per *OSC* sub-step
_ROBOCASA_OSC_ROT_MAX = 0.5    # radians per *OSC* sub-step
_ROBOCASA_PER_WM_STEP_POS_MAX = _ROBOCASA_OSC_POS_MAX * ROBOCASA_DEFAULT_CHUNK   # 0.20 m
_ROBOCASA_PER_WM_STEP_ROT_MAX = _ROBOCASA_OSC_ROT_MAX * ROBOCASA_DEFAULT_CHUNK   # 2.0 rad


# ---------------------------------------------------------------------------
# RoboCasa "state_delta" embodiment: training data uses ``action_source =
# state_delta`` so the WM action is *literally* the consecutive-step diff of
# the 14-D RoboCasa proprio (NOT the OSC controller's chunked input). The
# state input is the same 14-D proprio in raw simulator units. Both are
# zero-padded to ``state_dim = action_dim = 25`` in this model's config.
#
# State (and Δstate) layout (as in the training data):
#   [0:3]   base_pos          (m, world)            — Δ usually 0 (base locked)
#   [3:7]   base_quat (xyzw)                        — Δ usually 0
#   [7:10]  ee_pos_rel        (m, base frame)
#   [10:13] ee_rot_rel quat[xyz] (first 3 of base-relative quat)
#   [13]    gripper_qpos[0]   (sim m, raw)
# Slots [14:25] are zero pad. There is NO chunking and NO [-1, 1]
# normalisation — values are kept in their raw simulator units.
ROBOCASA_STATE_DELTA_PROPRIO_DIM = 14
# Planner gripper lives in [-1, 1] with +1 = close, -1 = open.
# Δgripper_qpos is positive-open (wider fingers), so the scale is
# negative to flip the sign.  Magnitude 0.04 m ≈ full finger range.
ROBOCASA_STATE_DELTA_GRIPPER_SCALE = -0.04


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def pad_preprocess(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Zero-pad or truncate the last dim of ``x`` to ``target_dim``.

    This is the original DROID-style behaviour previously inlined as
    ``F.pad(action, (0, action_dim - action.shape[-1], 0, 0, 0, 0))``.
    """
    cur = x.shape[-1]
    if cur == target_dim:
        return x
    if cur > target_dim:
        return x[..., :target_dim]
    pad_n = target_dim - cur
    pad_zeros = x.new_zeros(*x.shape[:-1], pad_n)
    return torch.cat([x, pad_zeros], dim=-1)


# ---------------------------------------------------------------------------
# Euler-XYZ → axis-angle (vectorised, pure pytorch)
# ---------------------------------------------------------------------------


def _euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., 3]`` Euler-XYZ angles (rad) to ``[..., 3, 3]`` rotation
    matrices.

    Matches ``scipy.spatial.transform.Rotation.from_euler('xyz', ...).as_matrix()``,
    i.e. **extrinsic** rotations around the fixed X, then Y, then Z axes.
    The combined matrix is ``R = Rz @ Ry @ Rx``.
    """
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    zero = torch.zeros_like(rx)
    one = torch.ones_like(rx)

    Rx = torch.stack(
        [torch.stack([one, zero, zero], dim=-1),
         torch.stack([zero, cx, -sx], dim=-1),
         torch.stack([zero, sx, cx], dim=-1)],
        dim=-2,
    )
    Ry = torch.stack(
        [torch.stack([cy, zero, sy], dim=-1),
         torch.stack([zero, one, zero], dim=-1),
         torch.stack([-sy, zero, cy], dim=-1)],
        dim=-2,
    )
    Rz = torch.stack(
        [torch.stack([cz, -sz, zero], dim=-1),
         torch.stack([sz, cz, zero], dim=-1),
         torch.stack([zero, zero, one], dim=-1)],
        dim=-2,
    )
    return Rz @ Ry @ Rx


def _matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., 3, 3]`` rotation matrices to ``[..., 3]`` axis-angle
    (rotation-vector) representation.

    Closed-form ``log(R) = (theta / (2 sin theta)) (R - R^T)`` with safe
    handling of ``theta -> 0`` (small-angle Taylor expansion) and
    ``theta -> pi`` (use the diagonal of ``R + I`` to recover the axis).
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    # generic skew-symmetric components
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    skew = torch.stack([rx, ry, rz], dim=-1)

    sin_theta = torch.sin(theta)
    # safe scale: theta / (2 sin theta), with 1/2 limit at theta -> 0
    safe = torch.where(
        sin_theta.abs() > 1e-6,
        theta / (2.0 * sin_theta),
        0.5 + theta * theta / 12.0,
    )
    axis_angle = skew * safe.unsqueeze(-1)

    # near-pi correction (sin_theta ≈ 0 with theta ≈ pi): use the diagonal
    # to recover an unsigned axis, then sign by the off-diagonals.
    near_pi = (theta > (torch.pi - 1e-3))
    if near_pi.any():
        m = R + torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
        diag = torch.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], dim=-1).clamp(min=0.0)
        ax = torch.sqrt(diag * 0.5 + 1e-12)
        signs = torch.stack(
            [
                torch.sign(R[..., 2, 1] - R[..., 1, 2]),
                torch.sign(R[..., 0, 2] - R[..., 2, 0]),
                torch.sign(R[..., 1, 0] - R[..., 0, 1]),
            ],
            dim=-1,
        )
        signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
        ax = ax * signs * theta.unsqueeze(-1)
        axis_angle = torch.where(near_pi.unsqueeze(-1), ax, axis_angle)

    return axis_angle


def euler_xyz_to_axis_angle(euler: torch.Tensor) -> torch.Tensor:
    """Vectorised Euler-XYZ → axis-angle. Input ``[..., 3]`` → output ``[..., 3]``.

    Equivalent to
    ``Rotation.from_euler('xyz', euler).as_rotvec()`` from scipy.
    """
    R = _euler_xyz_to_matrix(euler)
    return _matrix_to_axis_angle(R)


# ---------------------------------------------------------------------------
# Planner-7D → canonical-12D
# ---------------------------------------------------------------------------


def to_canonical12_from_droid7(x: torch.Tensor) -> torch.Tensor:
    """Convert the planner's 7-D action to canonical PandaOmron 12-D layout A.

    Input layout (DROID-style): ``[dx, dy, dz, droll, dpitch, dyaw, d_grip]``
    in **metres / radians** (planner-side metric units).
    Output layout A:
    ``[base(4)=0, control_mode=-1, ee_pos(3), ee_rot_axisangle(3), gripper=±1]``
    in the **normalised [-1, 1]** training-canonical units.

    The Euler→axis-angle conversion is closed-form pytorch (see
    :func:`euler_xyz_to_axis_angle`). The gripper is binarised by sign of
    ``d_grip`` (>=0 → +1, <0 → -1) — the WM was trained on binary ±1
    gripper signals.

    The translation and rotation are **normalised** by the RoboCasa OSC
    controller's ``output_max`` (0.05 m / 0.5 rad) and clipped to
    ``[-1, 1]`` so the WM-side actions match the training distribution
    (RoboCasa logs the normalised input the OSC sees, not the metric
    delta the controller produces). This is symmetric with the env-side
    normalisation in
    :func:`scripts.robo_jepa.action_components.merge_components` for
    embodiment ``franka_omron``. Without this normalisation the WM is
    fed actions that are ~20× smaller in magnitude than what it was
    trained on, and predicts essentially no motion (the imagination
    viz looks like a single static frame).
    """
    assert x.shape[-1] == 7, f"expected last-dim=7, got {x.shape}"
    pos_m = x[..., 0:3]
    euler_rad = x[..., 3:6]
    grip = x[..., 6:7]

    axis_angle_rad = euler_xyz_to_axis_angle(euler_rad)
    # metric → normalised + clip (matches merge_components / OSC controller).
    pos = (pos_m / _ROBOCASA_OSC_POS_MAX).clamp(-1.0, 1.0)
    axis_angle = (axis_angle_rad / _ROBOCASA_OSC_ROT_MAX).clamp(-1.0, 1.0)
    base = x.new_zeros(*x.shape[:-1], 4)
    control_mode = x.new_full((*x.shape[:-1], 1), -1.0)
    # Binarise gripper at threshold 0.5 (matches
    # ``scripts.robo_jepa.action_components._binarise`` so the env-side
    # and WM-side franka_omron packers agree). With ``threshold=0.5``
    # the default planner-side ``grip=0`` maps to ``-1`` (closed) — the
    # convention used by env-vs-wm scripts (``a[11] = -1.0``) and by
    # the close_mixer training distribution. The previous
    # ``torch.where(grip >= 0.0, ...)`` used threshold 0 and produced
    # ``+1`` (open) for the default zero, which sat outside the WM's
    # close-gripper-during-arm-motion training regime and made the
    # transformer effectively ignore the action token.
    gripper = torch.where(grip >= 0.5, torch.ones_like(grip), -torch.ones_like(grip))

    return torch.cat([base, control_mode, pos, axis_angle, gripper], dim=-1)


# ---------------------------------------------------------------------------
# Per-embodiment preprocessors
# ---------------------------------------------------------------------------


def robocasa_action_preprocess(
    x: torch.Tensor,
    target_dim: int,
    chunk_size: int = ROBOCASA_DEFAULT_CHUNK,
    action_per_step: int = ROBOCASA_ACTION_PER_STEP,
) -> torch.Tensor:
    """Preprocess a planner / canonical action for the RoboCasa WM.

    The trained model consumes a flat ``(B, T, chunk_size * 12)`` action
    tensor where each row is a sequence of ``chunk_size`` canonical 12-D
    sub-actions. The function auto-detects the input layout based on the
    last dim:

    * ``shape[-1] == 7``  — planner's DROID-style action; converted to
      canonical 12-D via :func:`to_canonical12_from_droid7`.
    * ``shape[-1] == 12`` — already canonical layout A; used as-is.
    * other               — :class:`AssertionError`.

    The 12-D action is then broadcast over the chunk axis with
    ``unsqueeze(-2).expand(...)`` (no copies) and ``reshape`` to
    ``(..., chunk_size * 12)``.

    The result is zero-padded up to ``target_dim``, as the training loader pads
    actions to ``pad_size`` (e.g. 48 -> 64). Asserts ``target_dim >= chunk_size *
    action_per_step`` so a too-small model action dim fails loudly.
    """
    expected = chunk_size * action_per_step
    assert target_dim >= expected, (
        f"robocasa_action_preprocess: target_dim={target_dim} < "
        f"chunk_size*action_per_step={expected}. Check WM config."
    )
    cur = x.shape[-1]
    if cur == 7:
        x = to_canonical12_from_droid7(x)
    elif cur == action_per_step:
        pass
    else:
        raise AssertionError(
            f"robocasa_action_preprocess: expected last-dim 7 (planner) or "
            f"{action_per_step} (canonical), got {x.shape}"
        )
    # (..., 12) -> (..., chunk, 12) -> (..., chunk * 12)
    x = x.unsqueeze(-2).expand(*x.shape[:-1], chunk_size, action_per_step)
    return pad_preprocess(x.reshape(*x.shape[:-2], expected).contiguous(), target_dim)


def robocasa_state_preprocess(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Preprocess the state input for the RoboCasa WM.

    Training-time convention is ``state == zeros`` for RoboCasa under
    ``action_source='action_key'``; at inference we mirror that by
    zero-padding (or zero-filling) the planner-side state up to
    ``target_dim``. Implemented as a thin wrapper over
    :func:`pad_preprocess` so future embodiments can override.
    """
    return pad_preprocess(x, target_dim)


# ---------------------------------------------------------------------------
# RoboCasa franka_omron_state_delta preprocessors
# ---------------------------------------------------------------------------


def franka_omron_state_delta_action_preprocess(
    x: torch.Tensor,
    target_dim: int,
    proprio_dim: int = ROBOCASA_STATE_DELTA_PROPRIO_DIM,
) -> torch.Tensor:
    """Preprocess a planner action for the RoboCasa state-delta WM.

    The trained model consumes a per-WM-step ``Δstate`` of length
    ``ROBOCASA_STATE_DELTA_PROPRIO_DIM = 14``, padded to
    ``target_dim = state_dim = action_dim = 25``. ``Δstate`` slots are:

        [0:3]   Δbase_pos          (m, world)
        [3:7]   Δbase_quat (xyzw)
        [7:10]  Δee_pos_rel        (m, base frame)
        [10:13] Δee_rot_rel_quat[xyz]
        [13]    Δgripper_qpos[0]
        [14:25] zero pad

    The planner is currently 7-D ``[dx, dy, dz, droll, dpitch, dyaw,
    dgrip]`` in metric m / rad. **Translation is mapped directly into the
    base-frame slot** (the env-side OSC controller already consumes
    base-frame deltas, so the planner's ``dxyz`` is base-frame too — no
    ``R_bw`` rotation is needed). **Rotation is asserted to be zero**
    per the agreed contract; we pass ``Δee_rot_rel_quat[xyz] = 0`` to the
    WM (will be lifted later when we implement proper Δquat). **Gripper
    is rescaled** from the planner's ``[-1, 1]`` convention to metric
    Δgripper_qpos via ``ROBOCASA_STATE_DELTA_GRIPPER_SCALE`` (0.04 m),
    so that ±1 ≈ full close/open and 0.3 ≈ 0.012 m (in-distribution).

    Asserts:
      * ``target_dim >= proprio_dim`` (need at least 14 slots).
      * If ``x`` already has ``proprio_dim`` (or ``target_dim``) last-dim,
        we treat it as already-canonical and only pad/trim.
      * If ``x`` has 7 dims (planner format), the rotation slots
        ``x[..., 3:6]`` must be exactly zero.
    """
    assert target_dim >= proprio_dim, (
        f"franka_omron_state_delta_action_preprocess: target_dim={target_dim} "
        f"< proprio_dim={proprio_dim}"
    )
    cur = x.shape[-1]

    if cur == target_dim:
        return x.contiguous()
    if cur == proprio_dim:
        return pad_preprocess(x, target_dim)
    if cur != 7:
        raise AssertionError(
            f"franka_omron_state_delta_action_preprocess: expected last-dim "
            f"7 (planner) / {proprio_dim} (canonical) / {target_dim} (padded), "
            f"got {x.shape}"
        )

    # 7-D planner action layout: [dx, dy, dz, droll, dpitch, dyaw, dgrip].
    # Rotation slots must be zero (agreed-upon temporary contract).
    rot = x[..., 3:6]
    rot_max = rot.abs().max().item() if rot.numel() > 0 else 0.0
    assert rot_max < 1e-6, (
        f"franka_omron_state_delta_action_preprocess: planner rotation slots "
        f"must be zero (Δrotation support not yet implemented for the "
        f"state-delta path), got |rot|.max()={rot_max:.6e}"
    )

    pos = x[..., 0:3]                                     # (..., 3) base-frame Δee_pos
    grip = x[..., 6:7] * ROBOCASA_STATE_DELTA_GRIPPER_SCALE  # (..., 1) planner +1=close → negative Δqpos (narrower)
    base_pos = x.new_zeros(*x.shape[:-1], 3)              # Δbase_pos = 0
    base_quat = x.new_zeros(*x.shape[:-1], 4)             # Δbase_quat = 0
    ee_rot = x.new_zeros(*x.shape[:-1], 3)                # Δee_rot_rel_quat[xyz] = 0

    delta_state = torch.cat(
        [base_pos, base_quat, pos, ee_rot, grip], dim=-1
    )  # (..., 14)
    return pad_preprocess(delta_state, target_dim)


def franka_omron_state_delta_state_preprocess(
    x: torch.Tensor,
    target_dim: int,
    proprio_dim: int = ROBOCASA_STATE_DELTA_PROPRIO_DIM,
) -> torch.Tensor:
    """Preprocess the WM state input for the RoboCasa state-delta variant.

    Expects either:
      * already the full ``target_dim`` (passthrough), or
      * the raw ``proprio_dim = 14`` RoboCasa proprio vector
        (env-side ``obs['proprio_robocasa']``), in which case we
        zero-pad slots ``[14:target_dim]``.
    Anything else is a config drift bug — fail loudly.
    """
    cur = x.shape[-1]
    if cur == target_dim:
        return x.contiguous()
    if cur == proprio_dim:
        return pad_preprocess(x, target_dim)
    raise AssertionError(
        f"franka_omron_state_delta_state_preprocess: expected last-dim "
        f"{proprio_dim} (raw proprio) or {target_dim} (padded), got {x.shape}"
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


# default action / state preprocessors used when an embodiment has no entry
# in the registry below.
def _default_action(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    return pad_preprocess(x, target_dim)


def _default_state(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    return pad_preprocess(x, target_dim)


ACTION_PREPROCESSORS: dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    "franka_omron": robocasa_action_preprocess,
    "franka_omron_state_delta": franka_omron_state_delta_action_preprocess,
}

STATE_PREPROCESSORS: dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    "franka_omron": robocasa_state_preprocess,
    "franka_omron_state_delta": franka_omron_state_delta_state_preprocess,
}


def get_action_preprocessor(name: str | None) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Return the action preprocessor for ``name`` (or the default)."""
    if name is None:
        return _default_action
    return ACTION_PREPROCESSORS.get(name, _default_action)


def get_state_preprocessor(name: str | None) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Return the state preprocessor for ``name`` (or the default)."""
    if name is None:
        return _default_state
    return STATE_PREPROCESSORS.get(name, _default_state)


def resolve_embodiment_name(
    embodiment_ids: torch.Tensor | None,
    embodiment_list: list[str] | None,
) -> str | None:
    """Resolve a single embodiment name from the per-batch ids tensor.

    Asserts that all batch elements share the same embodiment so that the
    preprocessing path stays a single vectorised op (no per-element
    branching). If either ``embodiment_ids`` or ``embodiment_list`` is
    ``None`` this returns ``None`` (caller falls back to defaults).
    """
    if embodiment_ids is None or embodiment_list is None or len(embodiment_list) == 0:
        return None
    emb = embodiment_ids.reshape(-1)
    assert (emb == emb[0]).all().item(), (
        "all batch elements must belong to the same embodiment for "
        "vectorised preprocessing; got embodiment_ids = "
        f"{embodiment_ids.detach().cpu().tolist()}"
    )
    idx = int(emb[0].item())
    if idx < 0 or idx >= len(embodiment_list):
        return None
    return embodiment_list[idx]
