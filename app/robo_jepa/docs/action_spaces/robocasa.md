# RoboCasa (Franka + Omron mobile base, sim)

RoboCasa's PandaOmron is a Franka arm on an Omron mobile base, simulated in robosuite. Two embodiment ids read the
same data and differ only in how the h5 is interpreted: `franka_omron` uses the recorded OSC controller commands,
`franka_omron_state_delta` the difference of the 14-D proprio state.

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `RoboCasa` | `franka_omron` (2) | `robocasa/human_720p/` | 20 → 5 Hz | 0.2 s (4 sub-steps) | `robot0_agentview_left` 0, `robot0_agentview_right` 1, `robot0_eye_in_hand` 2; 2 at random (1-view group: an agentview) | — |
| `RoboCasa_state_delta` | `franka_omron_state_delta` (3) | `robocasa/human_720p/` | 20 → 5 Hz | 0.2 s (4 frames) | same | — |

Eval-only: `RoboCasa_target` (`franka_omron`, `robocasa/human_target_720p/`, same settings).

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `franka_omron` (2) | action-key `robocasa` | zeros | 4 × 12-D OSC commands ("layout A") = 48, normalized [−1, 1] |
| `franka_omron_state_delta` (3) | state-delta | `cartesian_position[13]` ⊕ `gripper_position[1]` = 14, raw sim units | element-wise `state[t+1] − state[t]` = 14 |

**Layout A (12-D per OSC sub-step)** = `action/cartesian_position[:, :11]` ⊕ `action/gripper_position`:

| Idx | Channel | Meaning | Active when |
|---:|---|---|---|
| 0–2 | `base_motion` (x, y, yaw) | mobile-base velocity command, normalized | base mode |
| 3 | `torso` | spine vertical delta, normalized | base mode |
| 4 | `control_mode` | ±1 (−1 = arm, +1 = base) | always |
| 5–7 | `ee_pos` | EE translation **delta**, base frame; ×0.05 m by the controller | arm mode |
| 8–10 | `ee_rot` | EE **axis-angle** rotation delta, base frame; ×0.5 rad | arm mode |
| 11 | `gripper` | ±1 (−1 = open, +1 = close) | always |

**State-delta proprio (14-D)**:

| Idx | Component | Units / frame | Notes |
|---:|---|---|---|
| 0–2 | `base_pos` (x, y, z) | m, world | z ≈ 0.700 (constant); moves only in navigation episodes |
| 3–6 | `base_quat` (x, y, z, w) | world, xyzw | x ≡ y ≡ 0 (yaw only); unit norm |
| 7–9 | `ee_pos_rel` (x, y, z) | m, base frame | the clean metric EE translation |
| 10–12 | `ee_rot_rel` (x, y, z) | xyz of the base-relative EE quaternion (w dropped) | — |
| 13 | gripper finger half-width | m | continuous opening, ≈ 0.018–0.040 |

## Stored h5 (`trajectory/…`)

| Field | Shape | Content |
|---|---|---|
| `action/cartesian_position` | [T, 13] | `base_motion(4)` + `control_mode(1)` + `ee_pos(3)` + `ee_rot(3)` + pad(2, always 0) |
| `action/gripper_position` | [T, 1] | gripper command, ±1 |
| `observation/robot_state/cartesian_position` | [T, 13] | proprio: `base_pos` + `base_quat` + `ee_pos_rel` + `ee_rot_rel` |
| `observation/robot_state/gripper_position` | [T, 1] | finger half-width (m) |

The two 13-D `cartesian_position` arrays have the same name but unrelated meanings (normalized commands vs absolute
proprio).

## Typical values

| Signal | Units | Range | \|Δ\| per WM step |
|---|---|---|---|
| `ee_pos` command | normalized | [−1, 1], saturates during reaching | — |
| `ee_rot` command | normalized | up to ≈ 0.52 | — |
| `base_pos` x, y | m | x 0.42–2.76, y −2.59 … −0.49 | up to 0.15 (navigation) |
| `ee_pos_rel` x, y, z | m | x 0.01–0.64, y −0.44–0.09, z 0.27–0.71 | ≈ ±0.06 (std ≈ 0.018) |
| `base_quat` z, `ee_rot_rel` x | — | [−1, 1] | spikes to ≈ ±2.0 (quaternion sign flips) |
| gripper half-width | m | ≈ 0.018–0.040 | ≈ ±0.012 |

## Notes

* **One mode per sub-step.** `control_mode` is a hard switch: in arm mode the base and torso channels are exactly 0,
  in base mode `ee_pos` / `ee_rot` are exactly 0. Episodes mix the modes over time (navigation is mostly base mode,
  pick-and-place only arm mode). The mode is part of the action because the same image motion can come from moving
  the arm or driving the base.
* **Commands are goal deltas, not displacements.** The arm runs OSC_POSE (`input_ref_frame = base`, `kp = 150`,
  `ramp_ratio = 0.2`), so one sub-step realizes only part of the commanded `±0.05 m`; one WM step (4 sub-steps)
  commands at most 0.20 m. The env's `--blocking_control` mode instead drives the EE to the metric target.
* **Why `franka_omron` has a zero state.** The normalized OSC command and the absolute proprio live in different
  spaces with no closed-form integration, so the WM is action-only and rolls out purely in latent space.
  `franka_omron_state_delta` makes the action a state delta so that `state_{t+1} ≈ state_t + action_t`.
* **Two gripper meanings.** `franka_omron`: a binary ±1 command. `franka_omron_state_delta`: Δ finger width in m
  (≈ ±0.01).
* **State-delta rotation is unusable.** Component-wise quaternion subtraction gives spurious ≈ ±2.0 jumps whenever a
  logged quaternion flips sign. At inference rotation is asserted zero, so this path is translation + gripper only.
* **Inference** (`embodiment_preprocessing.py`):
  * `franka_omron`: a 7-D planner action is converted to layout A (translation ÷ 0.05 m; rotation converted
    euler xyz → axis-angle, then ÷ 0.5 rad; clipped to [−1, 1]; base = 0, mode = −1, gripper binarized at 0.5),
    repeated over the 4 sub-steps and zero-padded to 64. A 12-D layout-A input is passed through as is.
  * `franka_omron_state_delta`: the 7-D action maps to the 14-D Δstate as Δ`base_pos` = Δ`base_quat` = 0,
    Δ`ee_pos_rel` = (dx, dy, dz), Δ`ee_rot_rel` = 0 (asserted), Δgripper = −0.04 · dgrip (planner +1 = close →
    narrower finger width); raw metric units, no chunking.
* **Env packing** (`--wm_embodiment` of the RoboCasa env server): `franka_omron` packs to layout A as above;
  `franka_omron_state_delta` uses the same layout but zeros the OSC rotation slots. Both drive the same controller.
