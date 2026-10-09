# OXE RoboNet (5 robots)

RoboNet (OXE `robo_net`) contributes five embodiment ids, one per robot. Episodes are stored per robot under
`robonet/<robot>/` (from the per-episode `episode_metadata/robot`).

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `OXE_robonet_franka` | `franka_robonet` (12) | `oxe/robonet/franka` | 1 → 1 Hz | 1 s (every frame) | `camera1..3`, one at random → 4 | 0.0037 |
| `OXE_robonet_baxter` | `baxter` (13) | `oxe/robonet/baxter` | 1 → 1 Hz | 1 s (every frame) | same | 0.0085 |
| `OXE_robonet_sawyer` | `sawyer` (14) | `oxe/robonet/sawyer` | 1 → 1 Hz | 1 s (every frame) | same | 0.0235 |
| `OXE_robonet_widowx` | `widowx_robonet` (15) | **`oxe/robonet/sawyer`** | 1 → 1 Hz | 1 s (every frame) | same | 0.0024 |
| `OXE_robonet_kuka` | `kuka_robonet` (18) | `oxe/robonet/kuka` | 1 → 1 Hz | 1 s (every frame) | same | 0.0008 |

All are in the 1-view group only; view id 4 is "default / unmapped".

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| all five | state-delta | `[x, y, z, θ, 0, 0]` ⊕ `0` = 7 | Δxyz, relative rotation `R_{t+1}·R_tᵀ` as euler xyz (only the first component, Δθ, is non-zero), Δgripper = 0 → 7 |

## Stored h5 (`trajectory/…`)

The public `observation/state` is 5-D, `[x, y, z, θ, gripper]`.

| Field | Shape | Content |
|---|---|---|
| `observation/robot_state/cartesian_position` | [T, 6] | `[x, y, z, θ, 0, 0]`: **θ (the yaw) sits in the euler roll slot**, pitch and yaw are 0 |
| `observation/robot_state/gripper_position` | [T, 1] | **identically 0**; the source gripper `state[4]` is dropped |
| `joint_position`, `action/*` | | NaN |

## Typical values

| Robot | x | y | z | θ (rad) | source gripper `state[4]` |
|---|---|---|---|---|---|
| baxter | −0.28–1.03 | −0.03–1.05 | −0.13–1.08 | 0.01–6.20 | 3.9–96.6 |
| franka | 0.00–1.00 | 0.14–1.00 | 0.00–1.00 | 0.01–6.28 | −1.57–1.57 |
| kuka | 0.58–1.01 | 0.14–1.01 | −0.21–0.96 | 0.93–4.92 | 0 |
| sawyer | −0.01–1.01 | −0.01–1.01 | 0.00–1.09 | 1.55–4.64 | −1.0–1.01 |
| widowx | 0.00–1.01 | −0.01–0.99 | 0.01–1.00 | 1.56–4.77 | −1.0–1.0 |

\|Δ\| per WM step, p50 / p99: Δx 0.048 / 0.23, Δy 0.044 / 0.19, **Δz 0.12 / 1.02**, rot x 0.10 / 2.0 rad; the other
rotation components and Δgripper are 0.

## Notes

* **Units.** xyz is workspace-normalized to ≈ [0, 1] per robot (not metres), θ ∈ [0, 2π). Δz reaches the full
  normalized range within one 1 s step at p99; 1 fps is the coarsest rate in the mixture.
* **`widowx_robonet` trains on Sawyer episodes**: its `s3_prefix` is `oxe/robonet/sawyer`, so id 15 sees the same
  data as id 14.
* **No gripper signal**: `gripper_position` is 0 for all five robots.
* **Yaw is encoded as roll**: consistent within RoboNet, but opposite to the other datasets, where roll is the
  near-±π component.
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): no preprocessor (zero-pad to 64);
  the rollout update is formally consistent (7-D layout), but metric planner deltas do not match RoboNet's normalized
  units.
