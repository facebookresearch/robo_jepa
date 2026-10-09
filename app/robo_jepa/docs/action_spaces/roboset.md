# Roboset (`franka_roboset`)

The Franka Panda of Roboset (Bharadhwaj et al. 2024), trained under its own id, separate from the generic Franka. The
end-effector pose is forward kinematics of the stored joints.

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `Roboset` | `franka_roboset` (6) | `roboset/` | 5 → 1 Hz | 1 s (5 frames) | `left` 0, `right` 1, `wrist` 2, `top` 5; 2 at random (240×424; 1-view group: `left`/`right`/`top`) | 0.0847 |

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `franka_roboset` (6) | state-delta | `cartesian_position[6]` ⊕ `gripper_position[1]` = 7 | Δxyz, relative rotation `R_{t+1}·R_tᵀ` as euler xyz, Δgripper = 7 |

`pose_dim: 6`, `gripper_dim: 1`, `joints_dim: 7`. The stored `action/*` is not used.

## Stored h5 (`trajectory/…`)

Source: the Roboset trial arrays `qp_arm` [T, 7], `qp_ee` [T], `ctrl_arm` [T, 7], `ctrl_ee` [T, 1].

| Field | Shape | Content |
|---|---|---|
| `observation/robot_state/joint_position` | [T, 7] | `qp_arm`, with +π/2 on joints 6–7 |
| `observation/robot_state/cartesian_position` | [T, 6] | FK of those joints (Franka, `panda_virtual_ee_link`): xyz (m) + euler xyz (rad) |
| `observation/robot_state/gripper_position` | [T, 1] | `clip(qp_ee / 0.8305, 0, 1)`: closedness, **1 = closed** |
| `action/joint_position` | [≤T, 7] | `ctrl_arm`, +π/2 on joints 6–7, −π/2 on joint 4 |
| `action/cartesian_position` | [≤T, 6] | FK of those joints; stops at the first NaN row (T − 2 rows) |
| `action/gripper_position` | [T, 1] | `ctrl_ee / 2 + 0.5` (maps [−1, 1] → [0, 1], not clipped) |

## Typical values

| Signal | Units | Range | \|Δ\| per WM step, p50 / p99 |
|---|---|---|---|
| x, y, z | m | 0.55–0.61, −0.02–0.27, 0.24–0.48 | 0.007 / 0.035, 0.045 / 0.154, 0.034 / 0.109 |
| roll, pitch, yaw | rad | roll near ±π; pitch −0.03–0.61; yaw −1.51 … −0.73 | 0.09 / 0.24, 0.08 / 0.19, 0.11 / 0.57 |
| gripper | closedness | ≈ 0.46–1.0 (`qp_ee` 0.38–0.83) | 0.02 / 0.52 |

## Notes

* **Gripper.** High `qp_ee` = closed, so the stored value is a closedness (1 = closed); 0.8305 is the closed
  extreme, not the open width (as the paper labels it). Fully open still reads ≈ 0.46, not 0. The state and action
  grippers share a polarity but not a scale.
* The `/ 0.8305` + clip applies only when `qp_ee` is 1-D; a 2-D `qp_ee` is stored raw.
* The last 2 rows of `ctrl_arm` / `ctrl_ee` are NaN in every trial, so `action/cartesian_position` is 2 rows shorter
  than the state. This does not reach the WM.
* The joint-zero offsets (+π/2 on joints 6–7, −π/2 on action joint 4) are hard-coded to match the DRM Franka URDF.
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): no preprocessor (zero-pad to 64); a
  7-D planner action is the training layout and the rollout state update is consistent (its [0, 1] gripper clip
  matches this closedness scale).
