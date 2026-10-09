# Kuka iiwa (`kuka`: OXE QT-Opt + MT-Opt)

Kuka iiwa bin grasping from QT-Opt (OXE `kuka`) and MT-Opt (`mt_opt_rlds`). Both map to the same id 11; RoboNet's
Kuka is a separate id (`kuka_robonet`, see [robonet.md](robonet.md)).

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `OXE_qtopt` | `kuka` (11) | `qtopt` | 10 → 10 Hz | **0.1 s** (every frame) | `left` 0 (one view) | 0.0045 (1-view only) |
| `OXE_mtopt` | `kuka` (11) | `oxe/mtopt` | 5 → 5 Hz | **0.2 s** (every frame) | `left` 0 (one view) | 0.0123 (1-view only) |

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `kuka` (11) | state-delta | `cartesian_position[6]` ⊕ `gripper_position[1]` = 7 | Δxyz, relative rotation `R_{t+1}·R_tᵀ` as euler xyz, Δgripper = 7 |

## Stored h5 (`trajectory/…`)

| Field | Shape | QT-Opt | MT-Opt |
|---|---|---|---|
| `observation/robot_state/cartesian_position` | [T, 6] | `base_pose_tool_reached` xyz ⊕ its quaternion as euler xyz | `state_dense` xyz ⊕ its quaternion as euler xyz |
| `observation/robot_state/gripper_position` | [T, 1] | `gripper_closed` | `gripper_closed` (bool → float) |
| `joint_position`, `action/*` | | NaN | NaN |

## Typical values

| Signal | Units | QT-Opt range | MT-Opt range | \|Δ\| per WM step, p50 / p99 (QT-Opt; MT-Opt) |
|---|---|---|---|---|
| x, y, z | m | 0.47–0.67, −0.17–0.26, 0.06–0.60 | 0.47–0.69, −0.26–0.24, 0.08–0.55 | 0.005 / 0.077, 0.006 / 0.177, 0.014 / 0.159; 0.007 / 0.100, 0.010 / 0.101, 0.013 / 0.164 |
| roll, pitch, yaw | rad | roll near ±π always, yaw near ±π in 76% of frames; pitch −0.04–0.09 | roll near ±π always; pitch −0.06–0.04 | 0.001 / 0.016, 0.004 / 0.029, 0.032 / 0.521; 0.001 / 0.016, 0.003 / 0.033, 0.003 / 0.403 |
| gripper | flag | {0, 1} | {0, 1} | Δ ∈ {−1, 0, +1} |

With roll ≈ ±π and pitch ≈ 0 throughout, the orientation effectively varies only in yaw.

## Notes

* **Same id, different time steps**: QT-Opt is 0.1 s per WM step and MT-Opt 0.2 s, though the per-step magnitudes
  are of similar size.
* QT-Opt episodes are short (≈ 15 steps on average).
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): no preprocessor (zero-pad to 64); a
  7-D planner action is the training layout and the rollout state update is consistent.
