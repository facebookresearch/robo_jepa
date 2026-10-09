# OXE FMB (`franka_fmb`)

The Franka Panda of the Functional Manipulation Benchmark (FMB, Luo et al. 2024; OXE `fmb`), trained under its own
id, separate from the generic Franka.

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `OXE_fmb` | `franka_fmb` (17) | `oxe/fmb/` | 10 → 5 Hz | 0.2 s (2 frames) | `left` 0, `right` 1; one view matching `left\|right` | 0.0024 (1-view only) |

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `franka_fmb` (17) | state-delta | `cartesian_position[6]` ⊕ `gripper_position[1]` = 7 | Δxyz, relative rotation `R_{t+1}·R_tᵀ` as euler xyz, Δgripper = 7 |

`pose_dim: 6`, `gripper_dim: 1`, `joints_dim: 7`.

## Stored h5 (`trajectory/…`)

| Field | Shape | Content |
|---|---|---|
| `observation/robot_state/cartesian_position` | [T, 6] | `observation/eef_pose` xyz ⊕ its quaternion as euler xyz |
| `observation/robot_state/gripper_position` | [T, 1] | `observation/state_gripper_pose` |
| `observation/robot_state/joint_position` | [T, 7] | `observation/joint_pos`; not used |
| `action/*` | | NaN |

Videos (10 fps): `image_side_1` → `left`, `image_side_2` → `right`, `image_wrist_1/2` → `wrist_left` / `wrist_right`.

## Typical values

| Signal | Units | Range | \|Δ\| per WM step, p50 / p99 |
|---|---|---|---|
| x, y, z | m | 0.43–0.63, −0.31–0.19, 0.03–0.34 | 0.002 / 0.023, 0.001 / 0.049, 0.006 / 0.039 |
| roll, pitch, yaw | rad | roll near ±π; pitch −0.98–0.83; yaw −0.04–1.83 | 0.006 / 0.139, 0.006 / 0.041, 0.005 / 0.150 |
| gripper | flag | {0, 1} | Δ ∈ {−1, 0, +1} |

## Notes

* The wrist cameras are stored but never sampled (`cameras: ['left|right']`).
* The gripper polarity (whether 1 = closed) is not documented.
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): no preprocessor (zero-pad to 64); a
  7-D planner action is the training layout and the rollout state update is consistent.
