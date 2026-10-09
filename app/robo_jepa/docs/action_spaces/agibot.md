# AgiBot World (AgiBot G1 dual-arm)

AgiBot G1 is a dual-arm robot on a mobile base with a head and a waist. Training data comes from the public AgiBot
World Alpha and Beta releases (per-episode `proprio_stats.h5`). `agibot_robotic` covers the gripper robots,
`agibot_humanoid` the dexterous-hand ones.

## Dataset

| Config entry | Embodiment (id) | Data (`s3_prefix`) | Raw → WM rate | Time per WM step | Cameras → view id | Weight |
|---|---|---|---|---|---|---|
| `AGIBot_robotiq` | `agibot_robotic` (5) | `agibot/robotiq` | 30 → 1 Hz | 1 s (30 frames) | `head_color` 3, `hand_left_color` 2, `hand_right_color` 2; 2 at random (1-view group: `head_color`) | — |
| `AGIBot_humanoid` | `agibot_humanoid` (19) | `agibot/humanoid` | 30 → 1 Hz | 1 s (30 frames) | same | — |

Video: 640×480 at 30 fps.

## WM input

| Embodiment (id) | Source | State (dim) | Action per WM step (dim) |
|---|---|---|---|
| `agibot_robotic` (5) | state-delta | `cartesian_position[23]` ⊕ `gripper_position[2]` = 25 | element-wise `state[t+1] − state[t]` = 25 |
| `agibot_humanoid` (19) | state-delta | `cartesian_position[23]` ⊕ `gripper_position[12]` = 35 (layout assumed, see Notes) | element-wise difference = 35 |

The 23-D pose (same for both ids):

| Idx | Content | Units / range | Public source (`proprio_stats.h5`) |
|---:|---|---|---|
| 0–2 | left EE xyz | m; x 0.45–0.80, y 0.18–0.48, z 0.30–0.76 | `state/end/position[:, 0]` |
| 3–5 | right EE xyz | m; x 0.50–0.85, y −0.50 … −0.03, z 0.31–0.94 | `state/end/position[:, 1]` |
| 6–8 | left EE roll, pitch, yaw | rad, extrinsic xyz euler; roll hits ±π | `state/end/orientation[:, 0]` (xyzw quat) |
| 9–11 | right EE roll, pitch, yaw | rad, as above | `state/end/orientation[:, 1]` |
| 12–13 | head (yaw, pitch) | rad; 12 ≈ 0, 13 ∈ [0.31, 0.51] | `action/head/position` |
| 14–16 | mobile-base x, y, z | m; z ≡ 0; all zero in Alpha | `state/robot/position` |
| 17–20 | mobile-base quaternion (x, y, z, w) | x ≡ y ≡ 0 (yaw only); all zero in Alpha | `state/robot/orientation` |
| 21–22 | waist (pitch, lift) | rad, m; 21 ∈ [0.44, 0.70], 22 ∈ [0.14, 0.31]; constant per episode | `action/waist/position` (**commanded**) |

Gripper: `agibot_robotic` stores the measured opening per hand (`state/effector/position`, raw units ≈ 35–122);
for `agibot_humanoid` it is presumably six hand-joint angles per hand (see Notes).

## Stored h5 (`trajectory/…`)

For `agibot_robotic`:


| Field | Shape | Content |
|---|---|---|
| `observation/robot_state/cartesian_position` | [T, 23] | the pose above (state) |
| `observation/robot_state/gripper_position` | [T, 2] | measured gripper opening, raw units |
| `observation/robot_state/joint_position` | [T, 14] | 2 arms × 7 joints; not used (`joints_dim = 0`) |
| `action/cartesian_position` | [T, 18] | cols 0–13 and 16–17 copy state cols 0–13 and 21–22; cols 14–15 = base velocity command; not used |
| `action/gripper_position` | [T, 2] | commanded level in [0, 1]; not used |
| `action/joint_position` | [T, 14] | copy of the measured joints; not used |

Action and state are time-aligned (no look-ahead).

## Typical values

| Signal | Units | Range | \|Δ\| per WM step, median / p99 / max |
|---|---|---|---|
| left EE xyz (0–2) | m | see above | ≤ 2e-4 / 0.07–0.10 / 0.15 |
| right EE xyz (3–5) | m | see above | 0.006–0.009 / 0.08–0.12 / 0.15 |
| EE roll (6, 9) | rad | ±π | ≤ 0.05 / **5.5** / **6.25** |
| EE pitch, yaw (7–8, 10–11) | rad | — | ≤ 0.05 / 0.26–0.68 / 0.96 |
| head (12–13), waist (21–22) | rad, m | constant | 0 / 0 / ≤ 1e-4 |
| base pose (14–20) | m, — | zero in Alpha | 0 / ≤ 0.09 / 0.10 |
| gripper (23–24) | raw | ≈ 35–122 | 0 / 57–61 / 72.6 |

## Notes

* **Euler wrap.** EE roll sits near ±π for common grasp orientations, so the element-wise difference jumps by ≈ 2π
  when roll crosses the branch cut (about 2% of WM steps at `fps = 1`, dims 6 and 9 only). Wrap Δ into (−π, π]
  before treating it as a rotation; the trained WM has seen these spikes.
* **Scale mismatch.** Nothing is normalized: gripper deltas (p99 ≈ 60) are hundreds of times larger than position
  deltas (p99 ≈ 0.1 m).
* **Inert dims.** 16–18 are always 0; 12–13 and 21–22 are constant within an episode; 14–20 are zero in Alpha and
  move only slightly in Beta. Base motion reaches the WM only as a pose delta; the base velocity command is not used.
* **Alpha waist correction.** In some Alpha episodes the commanded waist pitch differs from the measured one by a
  constant Δ (e.g. 0.08–0.11 rad). The stored waist (21–22) is the commanded one, and the arm poses are expressed at
  the commanded pitch: `p = Ry(Δ)·(p_pub − c) + c`, `R = Ry(Δ)·R_pub`, with pivot
  `c = (0.131, 0, lift + 0.3485)` m. For Beta (and Alpha with Δ = 0) the public poses are used as is.
* **`agibot_humanoid`** layout is not confirmed. The likely convention is the 23-D pose unchanged and
  `gripper_position` = `state/effector/position` [12]: six raw hand-joint angles per hand (left 0–5, right 6–11,
  rad, ≈ [−0.2, 1.06]; index 5 / 11 constant), giving the 35-D state.
* **`configs/scaling/wm_100m_local_2k.yaml`** reads the public tree directly through `state_keys` / `gripper_keys`
  (`end/position` 6 ⊕ raw `end/orientation` quaternions 8 ⊕ head 2 ⊕ measured waist 2 ⊕ `robot/position` 3 ⊕
  `robot/orientation` 4 ⊕ effector 2 = **27-D**, no Euler conversion), still named `agibot_robotic`. Its states and
  actions are not interchangeable with the 25-D layout above.
* **Inference** ([README](README.md#inference-what-worldmodelstep-does-per-id)): no preprocessor (zero-pad to 64).
  The rollout state update keeps only 7 slots as xyz + euler + gripper, so it is **not** consistent with this layout.
