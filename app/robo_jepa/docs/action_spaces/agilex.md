# RoboMind AgileX (`agilex`)

The AgileX dual-arm robot from RoboMind (Wu et al. 2024). The state concatenates both arms' end-effector poses and
grippers (paper App. B.4: "the left and right end-effector poses and the left and right arm joints, taking each
arm's last joint as its gripper").

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `robomind_agilex` | `agilex` (8) | `robomind/robomind_agilex` | 30 → 5 Hz | 0.2 s (6 frames) | `camera_front` 3, `camera_left_wrist` 2, `camera_right_wrist` 2; 2 at random (1-view group: `camera_front`) | 0.0125 |

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `agilex` (8) | state-delta | `cartesian_position[12]` ⊕ `gripper_position[2]` = 14 | element-wise `state[t+1] − state[t]` = 14 |

`pose_dim: 12`, `gripper_dim: 2`, `joints_dim: 12`. Layout (from the config dimensions and the paper):

| Idx | Content | Source (RoboMind) |
|---:|---|---|
| 0–5 | left EE xyz (m) + roll, pitch, yaw (rad) | `end_effector_left[:6]` |
| 6–11 | right EE xyz + roll, pitch, yaw | `end_effector_right[:6]` |
| 12–13 | left, right gripper (last arm joint) | `joint_position_{left,right}[6]` |

The public 7-D `end_effector` is labelled x, y, z, rx, ry, rz, rw, but behaves like xyz + roll/pitch/yaw + a
gripper-like value (the last entry ranges −0.13 … 5.09), not xyz + quaternion.

## Stored h5 (`trajectory/…`)

| Field | Shape | Content |
|---|---|---|
| `observation/robot_state/cartesian_position` | [T, 12] | both arms' EE xyz + rpy |
| `observation/robot_state/gripper_position` | [T, 2] | last joint of each arm |
| `observation/robot_state/joint_position` | [T, 12] | 6 joints per arm; not used |

## Typical values

| Signal | Units | Range | \|Δ\| per WM step, p99 |
|---|---|---|---|
| left EE x, y, z | m | −0.07–0.28, −0.32–0.01, −0.03–0.37 | 0.06–0.10 |
| right EE x, y, z | m | −0.11–0.27, −0.01–0.30, −0.02–0.45 | 0.06–0.10 |
| roll, pitch, yaw | rad | within ±1.93 (never within 0.5 rad of ±π) | 0.25–0.44 |
| gripper | joint value | [−0.013, 0.434] | 0.22 |

## Notes

* The layout is inferred from the config dimensions and the paper, and the roll/pitch/yaw reading of `rx, ry, rz`
  from their value ranges.
* The element-wise Euler difference is harmless here: the angles stay far from ±π, so no WM step contains a > π
  jump.
* The gripper may instead be the `rw` slot of `end_effector`; the two differ by an order of magnitude (≈ 0.4 vs ≈ 5).
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): no preprocessor (zero-pad to 64).
  The rollout state update keeps only 7 slots as xyz + euler + gripper, so it is **not** consistent with this layout.
