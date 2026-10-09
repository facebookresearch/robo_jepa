# RoboMind Franka (`robomind_franka`)

The Franka Panda part of RoboMind (Wu et al. 2024), trained under its own id, separate from the generic Franka.

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `Robomind_franka` | `robomind_franka` (9) | `robomind/robomind_franka` | 30 → 5 Hz | 0.2 s (6 frames) | `camera_left` 0, `camera_right` 1, `camera_top` 5; 2 at random | 0.0063 |

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `robomind_franka` (9) | state-delta | `cartesian_position[6]` ⊕ `gripper_position[1]` = 7 | Δxyz, relative rotation `R_{t+1}·R_tᵀ` as euler xyz, Δgripper = 7 |

`pose_dim: 6`, `gripper_dim: 1`, `joints_dim: 7`. Layout (from the config dimensions):
`cartesian_position` = RoboMind `end_effector` (x, y, z, roll, pitch, yaw), `gripper_position` =
`joint_position[7]`, `joint_position` = `joint_position[:7]`.

## Stored h5 (`trajectory/…`)

| Field | Shape | Content |
|---|---|---|
| `observation/robot_state/cartesian_position` | [T, 6] | EE xyz (m) + euler xyz (rad) |
| `observation/robot_state/gripper_position` | [T, 1] | gripper joint, [0, 0.99] |
| `observation/robot_state/joint_position` | [T, 7] | arm joints; not used |

## Typical values

| Signal | Units | Range | \|Δ\| per WM step, p50 / p99 |
|---|---|---|---|
| x, y, z | m | 0.45–0.76, −0.35–0.49, 0.28–0.69 | 0.007 / 0.068, 0.012 / 0.143, 0.026 / 0.143 |
| roll, pitch, yaw | rad | roll near ±π; pitch −1.20–0.56; yaw −1.23–0.47 | 0.013 / 0.17, 0.036 / 0.37, 0.027 / 0.22 |
| gripper | joint value | [0, 0.99] | p99 0.62 |

## Notes

* RoboMind stores JPEG frames inside HDF5; frames were decoded per frame, BGR → RGB for the real Franka streams
  and simulated-split quaternions were converted to xyz-euler (paper App. B.4).
* Roll near ±π does not leak into the action (the rotation difference goes through matrices).
* The gripper polarity (whether 1 = closed) is not documented.
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): no preprocessor (zero-pad to 64); a
  7-D planner action is the training layout and the rollout state update is consistent.
