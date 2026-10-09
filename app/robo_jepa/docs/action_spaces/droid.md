# DROID (Franka Panda, 6 embodiment ids)

DROID 1.0.1 episodes are stored unchanged (the raw `trajectory.h5` under `data.h5:trajectory/`). It is the only
dataset trained under six embodiment ids: all six read the same episodes and differ only in which signal becomes the
WM action/state.

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `Droid` | `franka` (0) | `droid/1.0.1/` | 15 → 1.25 Hz (config: 60 → 5) | 0.8 s (12 frames) | `left` 0, `right` 1, `wrist` 2; 2 at random (1-view group: `left`/`right`) | 0.0282 |
| `Droid_joints` | `franka_joints` (1) | `droid/1.0.1/` | 15 → 1.25 Hz (config: 60 → 5) | 0.8 s (12 frames) | same | 0.0282 |
| `Droid_cart_action_key` | `franka_cart_action_key` (21) | `droid/1.0.1/` | 15 → 1.875 Hz (config: 60 → 7.5) | 0.53 s (8 commands) | same | 0.0282 |
| `Droid_joint_action_key` | `franka_joint_action_key` (22) | `droid/1.0.1/` | 15 → 1.875 Hz (config: 60 → 7.5) | 0.53 s (8 commands) | same | 0.0282 |
| `Droid_cart_vel_action_key` | `franka_cart_vel_action_key` (23) | `droid/1.0.1/` | 15 → 1.875 Hz (config: 60 → 7.5) | 0.53 s (8 commands) | same | 0.0282 |
| `Droid_joint_vel_action_key` | `franka_joint_vel_action_key` (24) | `droid/1.0.1/` | 15 → 1.875 Hz (config: 60 → 7.5) | 0.53 s (8 commands) | same | 0.0282 |

All six use the filter `filter_positive_negative_droid`, which drops episodes whose path contains `REAL` **and** whose
metadata building is "setup room" (and episodes without a metadata entry). DROID's total weight is 0.169.

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `franka` (0) | state-delta, cartesian | EE pose (6) + gripper = 7 | Δxyz, relative rotation `R_{t+1}·R_tᵀ` as euler xyz, Δgripper = 7 |
| `franka_joints` (1) | state-delta, joints | joints (7) + gripper = 8 | Δjoints, Δgripper = 8 |
| `franka_cart_action_key` (21) | action-key `droid_cart` | zeros | 8 × (absolute target xyz + absolute euler + gripper cmd) = 56 |
| `franka_joint_action_key` (22) | action-key `droid_joint` | zeros | 8 × (absolute target joints + gripper cmd) = 64 |
| `franka_cart_vel_action_key` (23) | action-key `droid_cart_vel` | zeros | 8 × (normalized EE velocity + gripper cmd) = 56 |
| `franka_joint_vel_action_key` (24) | action-key `droid_joint_vel` | zeros | 8 × (normalized joint velocity + gripper cmd) = 64 |

The action-key chunk is `action/<field>[t : t+8]` from the sampled frame, each step followed by
`action/gripper_position`, flattened. The position ids carry absolute targets (current pose plus motion); the
velocity ids carry only the motion. `franka` is the id the deployment planner targets (paper B.7).

## Stored h5 (`trajectory/…`)

| Field | Shape | Content |
|---|---|---|
| `observation/robot_state/cartesian_position` | [T, 6] | EE pose in the robot base frame: xyz (m) + **euler xyz** (rad); roll sits near ±π most of the time |
| `observation/robot_state/gripper_position` | [T] | measured gripper, [0, 0.99]; 0 = open, 1 = closed |
| `observation/robot_state/joint_positions` | [T, 7] | measured joint angles (rad); note the plural name |
| `action/cartesian_position` | [T, 6] | **absolute target pose** (not a delta) |
| `action/cartesian_velocity` | [T, 6] | commanded EE velocity, normalized to [−1, 1]; rotation part is a base-frame angular velocity |
| `action/joint_position` | [T, 7] | **absolute target joints** |
| `action/joint_velocity` | [T, 7] | commanded joint velocity, normalized to [−1, 1] |
| `action/gripper_position` | [T] | commanded gripper in [0, 1] (continuous) |
| `action/gripper_velocity` | [T] | in [−1, 1]; not used |
| `action/target_cartesian_position`, `action/target_gripper_position` | | a different signal from `action/cartesian_position`; not used |
| `action/robot_state/*` | | mirror of `observation/robot_state/*` |
| `observation/camera_extrinsics/{left,right,wrist}`, `timestamp/*`, `controller_info/*` | | not used by the WM |

`cameras/{left,right,wrist}` give `recordings/MP4/<serial>.mp4`. The mp4 has T − 1 frames for an h5 of length T.

## Typical values

| Signal | Units | Range | \|Δ\| per WM step, p50 / p99 |
|---|---|---|---|
| `franka` Δx, Δy, Δz | m | — | 0.018 / 0.177, 0.024 / 0.203, 0.029 / 0.199 |
| `franka` rotation x, y, z | rad | — | 0.037 / 0.358, 0.041 / 0.318, 0.058 / 0.457 |
| `franka` Δgripper | [0, 1] | [0, 0.99] | 0 / 0.837 |
| `franka_joints` Δjoints | rad | — | p99 0.30–0.70 |
| `cartesian_velocity`, `joint_velocity` | normalized | [−1, 1] | — |
| control period | s | median 0.071 (14.1 Hz) | — |

One normalized velocity unit moves the target about 9 cm, 0.18 rad of EE rotation (base frame, left-multiplied), or
0.24 rad per joint per 15 Hz step.

## Notes

* **Rates.** The configs say `raw_fps: 60`, but the videos are 15 fps recordings sped up 4× and the control stream
  is 15 Hz. Frames are indexed by count, so training is consistent; divide every config rate by 4. Paper App. B.7's
  "60 → 5 Hz, chunk of 12 commands" for the action-key path matches neither the config (8 commands) nor the real rate.
* **Roll near ±π.** Harmless for `franka` (the rotation difference goes through matrices). For
  `franka_cart_action_key`, about 11% of chunks contain a > π jump in roll between consecutive commands.
* **No normalization**: m, rad, [0, 1] gripper and normalized velocities are mixed in one vector as is.
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): none of the six ids has a
  preprocessor. For `franka` a 7-D planner action is the training layout and the rollout state update is
  consistent. The action-key ids keep a zero state, but the caller must build the 56/64-D chunk. For
  `franka_joints` the rollout update treats joints as xyz + euler + gripper, which is not consistent.
* **Eval-only** `DROID_objectstore` uses the same prefix and `franka` but inverts the filter: it keeps only the
  REAL-lab "setup room" episodes that training drops.
