# Language-Table (`language_table`)

A planar pusher: an arm whose end effector moves in a 2-D plane over a table and pushes blocks, with no gripper and no
orientation. Sim (OXE `language_table_sim`) and real (`language_table`) share one id.

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `OXE_language_table_sim` | `language_table` (16) | `oxe/language_table_sim/` | 5 → 5 Hz | 0.2 s (every frame) | `front` 6 (one view) | 0.0049 (1-view only) |
| `OXE_language_table_real` | `language_table` (16) | `oxe/language_table/` | 5 → 5 Hz | 0.2 s (every frame) | `front` 6 (one view) | 0.0074 (1-view only) |

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `language_table` (16) | state-delta | `cartesian_position[2]` = 2 (`gripper_dim: 0`) | element-wise `state[t+1] − state[t]` = 2: the realized planar displacement |

## Stored h5 (`trajectory/…`)

| Field | Shape | Content |
|---|---|---|
| `observation/robot_state/cartesian_position` | [T, 2] | `observation/effector_translation`: the measured planar EE position |
| `observation/robot_state/gripper_position` | [T, 1] | NaN (unused) |
| `observation/robot_state/joint_position`, `action/*` | | NaN |

The public `effector_target_translation` and the 2-D `action` are not stored.

## Typical values

| Signal | Units | Range (real; sim) | \|Δ\| per WM step, p50 / p99 / max (real; sim) |
|---|---|---|---|
| x | — (not documented) | 0.19–0.62; 0.15–0.60 | 0.010 / 0.077 / 0.117; 0.006 / 0.029 / 0.030 |
| y | — (not documented) | −0.31–0.35; −0.30–0.30 | 0.016 / 0.102 / 0.148; 0.007 / 0.030 / 0.030 |

## Notes

* **Two motion scales under one id.** Sim and real cover the same positions, but sim per-step motion never exceeds
  0.030 in either axis, while real reaches ≈ 0.15.
* 62 of the 64 padded action slots are always zero.
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): no preprocessor (zero-pad to 64);
  the rollout state update is approximately right if planner slots 2–6 are zero.
