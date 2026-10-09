# WidowX / Bridge (`bridge`: OXE Bridge v1 + Bridge V2)

WidowX 250 tabletop manipulation from Bridge v1 (OXE `bridge/0.1.0`) and Bridge V2 (`bridge_data_v2/0.0.1`). Both
map to the same id 7; RoboNet's WidowX is a separate id (`widowx_robonet`, see [robonet.md](robonet.md)).

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `OXE_Bridge` | `bridge` (7) | `oxe/bridge` | 5 → 5 Hz | 0.2 s (every frame) | `right` (one view) | 0.0009 (1-view only) |
| `OXE_Bridge2` | `bridge` (7) | `oxe/bridge2` | 5 → 5 Hz | 0.2 s (every frame) | `right_back` 1, `camera1` / `camera2` 4; 2 at random (256×256) | 0.0166 (both groups) |

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `bridge` (7) | state-delta | `cartesian_position[6]` ⊕ `gripper_position[1]` (closedness) = 7 | Δxyz, relative rotation `R_{t+1}·R_tᵀ` as euler xyz, Δgripper = 7 |

The recorded delta command in `action/*` is not used.

## Stored h5 (`trajectory/…`)

| Field | Shape | Bridge v1 | Bridge V2 |
|---|---|---|---|
| `observation/robot_state/cartesian_position` | [T, 6] | `observation/state[:6]`: EE xyz (m) + euler xyz (rad) | same |
| `observation/robot_state/gripper_position` | [T, 1] | `1 − clip(state[6], 0, 1)`: **1 = closed** | same |
| `action/cartesian_position` | [T, 6] | `world_vector` ⊕ `rotation_delta` (per-step EE delta command) | `action[:6]` (per-step EE delta command) |
| `action/gripper_position` | [T, 1] | NaN | `action[6]`, raw: **1 = open** |
| `observation/robot_state/joint_position`, `action/joint_position` | [T, 7] | NaN | NaN |

Bridge V2 videos: `right_back` and `camera1`/`camera2`/`wrist` (the latter only when not all-zero).

## Typical values

| Signal | Units | Range | \|Δ\| per WM step, p50 / p99 (V2) |
|---|---|---|---|
| x, y, z | m | v1: 0.16–0.45, −0.17–0.15, −0.07–0.22; V2: 0.12–0.45, −0.14–0.26, −0.03–0.24 | 0.004 / 0.034, 0.004 / 0.043, 0.006 / 0.043 |
| rotation x, y, z | rad | v1 within [−0.49, 0.66]; V2 yaw up to 2.88 (rarely near ±π) | 0.012 / 0.091, 0.013 / 0.116, 0.029 / 0.244 |
| gripper (state) | [0, 1] closedness | v1 [0, 0.94], V2 [0, 0.95] | p99 0.30 |
| action command | m, rad | position \|x\| ≤ 0.10, rotation \|x\| ≤ 0.27 | — |

Bridge v1 per-step magnitudes are similar (p99 0.031 / 0.049 / 0.038 m, 0.11 / 0.13 / 0.13 rad, Δgripper 0.31).

## Notes

* **Gripper polarity differs between action and state.** The stored state is a closedness (1 = closed, paper App.
  B.4); the recorded action gripper keeps Bridge's raw convention (1 = open). The WM only sees the state.
* **The action command is only partly realized.** `action[t, :3]` correlates with the next state change with gain
  0.6–0.94: a commanded delta, not a copy of the state.
* `OXE_Bridge2.view_map` has no `wrist` entry, although `cameras: ['.*', '.*']` can select a stored `wrist` video.
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): no preprocessor (zero-pad to 64); a
  7-D planner action is the training layout and the rollout state update is consistent.
