# Google robot (`google`: OXE RT-1 + BC-Z)

The Google mobile manipulator from RT-1 (OXE `fractal20220817_data`) and BC-Z (`bc_z`). Both map to the same id 10
and share one input projection.

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `OXE_RT1` | `google` (10) | `rt1` | 10 → 5 Hz | 0.2 s (2 frames) | `back` 3 (one view) | 0.0020 (1-view only) |
| `OXE_bcz` | `google` (10) | `oxe/bc_z/` | 10 → 5 Hz | 0.2 s (2 frames) | `back` 3 (one view) | 0.0031 (1-view only) |

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `google` (10) | state-delta | `cartesian_position[6]` ⊕ `gripper_position[1]` = 7 | Δxyz, relative rotation `R_{t+1}·R_tᵀ` as euler xyz, Δgripper = 7 |

## Stored h5 (`trajectory/…`)

| Field | Shape | RT-1 | BC-Z |
|---|---|---|---|
| `observation/robot_state/cartesian_position` | [T, 6] | `base_pose_tool_reached` xyz ⊕ its quaternion as euler xyz | `present/xyz` ⊕ **`present/axis_angle` as is** |
| `observation/robot_state/gripper_position` | [T, 1] | `gripper_closed` | `present/sensed_close` |
| `observation/robot_state/joint_position` | [T, 7] | NaN | NaN |
| `action/*` | | NaN | NaN |

The recorded OXE actions (`world_vector`, `rotation_delta`, RT-1 base commands, BC-Z future residuals) are not stored.

## Typical values

| Signal | Units | RT-1 range | BC-Z range | \|Δ\| per WM step, p50 / p99 (RT-1; BC-Z) |
|---|---|---|---|---|
| x, y, z | m (base frame) | 0.34–0.93, −0.35–0.19, 0.11–1.12 | −0.42–0.34, −0.14–0.34, 0.52–0.99 | 0.014 / 0.107, 0.012 / 0.091, 0.019 / 0.156; 0.002 / 0.056, 0.002 / 0.038, 0.002 / 0.033 |
| rotation slots | rad | euler xyz, roll near ±π in 85% of frames | axis-angle, each within ±1.9 | 0.038 / 0.368, 0.030 / 0.227, 0.034 / 0.278; 0.006 / 0.087, 0.005 / 0.081, 0.007 / 0.140 |
| gripper | closedness | [0, 1], continuous | [0.2, 1.0] | p99 0.541; 0.218 |

## Notes

* **BC-Z rotation is axis-angle in the euler slots.** The loader reads it as euler xyz, which differs from the true
  rotation by 0.31 rad median (0.76 rad p99). The relative-rotation action is therefore computed from misread
  rotations, though it stays small and smooth.
* **One id, two conventions.** RT-1 (euler, gripper [0, 1]) and BC-Z (axis-angle, gripper [0.2, 1]) share id 10.
* RT-1's mobile-base commands have no counterpart in the WM action. Roll near ±π does not leak into the action
  (matrix path).
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): no preprocessor (zero-pad to 64); a
  7-D planner action is the training layout and the rollout state update is consistent.
