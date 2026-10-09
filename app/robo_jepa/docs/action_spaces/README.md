# Action spaces of the RoboJEPA training mixture

One document per robot, covering every embodiment id of the main training config `configs/scaling/wm_4b.yaml`
(paper Tables 5, 7, 8). Each document has the same four tables: **Dataset** (config entries, rates, cameras,
sampling weight), **WM input** (what state and action the world model sees), **Stored h5** (the episode file
layout) and **Typical values** (units and ranges), followed by short notes.

## Embodiment ids

The id is the index into `model.transition_model.config.embodiment_list`; it selects the per-embodiment
action/state input projection.

| id | `embodiment_name` | robot | datasets | document |
|---:|---|---|---|---|
| 0 | `franka` | Franka Panda | DROID (state-delta, cartesian) | [droid.md](droid.md) |
| 1 | `franka_joints` | Franka Panda | DROID (state-delta, joints) | [droid.md](droid.md) |
| 2 | `franka_omron` | Franka + Omron base (sim) | RoboCasa (action-key) | [robocasa.md](robocasa.md) |
| 3 | `franka_omron_state_delta` | Franka + Omron base (sim) | RoboCasa (state-delta) | [robocasa.md](robocasa.md) |
| 4 | `so-101` | SO-101 | LeRobot (cartesian via FK) | [so101.md](so101.md) |
| 5 | `agibot_robotic` | AgiBot G1 dual-arm (gripper) | AgiBot World | [agibot.md](agibot.md) |
| 6 | `franka_roboset` | Franka Panda | Roboset | [roboset.md](roboset.md) |
| 7 | `bridge` | WidowX 250 | OXE Bridge, Bridge V2 | [bridge.md](bridge.md) |
| 8 | `agilex` | AgileX dual-arm | RoboMind AgileX | [agilex.md](agilex.md) |
| 9 | `robomind_franka` | Franka Panda | RoboMind Franka | [robomind_franka.md](robomind_franka.md) |
| 10 | `google` | Google robot | OXE RT-1, BC-Z | [google_robot.md](google_robot.md) |
| 11 | `kuka` | Kuka iiwa | OXE QT-Opt, MT-Opt | [kuka.md](kuka.md) |
| 12 | `franka_robonet` | Franka | OXE RoboNet | [robonet.md](robonet.md) |
| 13 | `baxter` | Baxter | OXE RoboNet | [robonet.md](robonet.md) |
| 14 | `sawyer` | Sawyer | OXE RoboNet | [robonet.md](robonet.md) |
| 15 | `widowx_robonet` | WidowX | OXE RoboNet (**see caveat 2**) | [robonet.md](robonet.md) |
| 16 | `language_table` | planar pusher | OXE Language-Table sim + real | [language_table.md](language_table.md) |
| 17 | `franka_fmb` | Franka Panda | OXE FMB | [fmb.md](fmb.md) |
| 18 | `kuka_robonet` | Kuka | OXE RoboNet | [robonet.md](robonet.md) |
| 19 | `agibot_humanoid` | AgiBot G1 (dexterous hands) | AgiBot World | [agibot.md](agibot.md) |
| 20 | `1x_robot` | 1X EVE humanoid | 1X world-model data | [1x.md](1x.md) |
| 21 | `franka_cart_action_key` | Franka Panda | DROID (action-key, cartesian position) | [droid.md](droid.md) |
| 22 | `franka_joint_action_key` | Franka Panda | DROID (action-key, joint position) | [droid.md](droid.md) |
| 23 | `franka_cart_vel_action_key` | Franka Panda | DROID (action-key, cartesian velocity) | [droid.md](droid.md) |
| 24 | `franka_joint_vel_action_key` | Franka Panda | DROID (action-key, joint velocity) | [droid.md](droid.md) |
| 25 | `so-101_joints` | SO-101 | LeRobot (joints) | [so101.md](so101.md) |

## How the loader builds the WM action and state

All datasets go through `RobotVideoDataset` (`app/robo_jepa/datasets/robot_video_dataset.py`). The config's
`action_source` selects one of two pipelines:

* **`state_delta`** (default). The state is read at the sampled frames (stride `ceil(raw_fps / fps)` raw frames)
  and the action is the difference of consecutive states (`poses_to_diffs`):
  * `use_joint_state: true` → state = `joint_position[joints_dim]` ⊕ `gripper_position`, action = element-wise
    difference.
  * otherwise state = `cartesian_position[pose_dim]` ⊕ `gripper_position[gripper_dim]`, and
    * `pose_dim == 3`: Δxyz, then Δgripper (no id uses this);
    * `3 < pose_dim ≤ 6`: Δxyz, then the relative rotation `R_{t+1}·R_tᵀ` (world frame) as **euler xyz**, then
      Δgripper;
    * `pose_dim > 6` or `< 3` (AgiBot, AgileX, 1X, RoboCasa state-delta, Language-Table): a plain **element-wise**
      difference of the whole vector, Euler/quaternion components included.
* **`action_key`** (RoboCasa and the DROID action-key ids). The recorded controller stream under
  `trajectory/action/` is read at the raw rate; the `c = raw_fps // fps` sub-step commands of one WM step are
  concatenated into one vector, and the **state is all zeros** (the WM is action-only). Layouts:
  `ACTION_KEY_LAYOUTS` in the loader.

Action and state are right-zero-padded to `action_dim = state_dim = 64`. **Nothing is normalized**: every
dataset enters the WM in its own units.

## Inference: what `WorldModel.step` does per id

At planning time `WorldModel.step` receives the planner's action (normally 7-D
`[dx, dy, dz, droll, dpitch, dyaw, dgrip]` in m / rad) and the current state. It then:

1. **Preprocesses** both with `embodiment_preprocessing.py`. Only `franka_omron` and `franka_omron_state_delta`
   have dedicated preprocessors ([robocasa.md](robocasa.md)); every other id is zero-padded to 64 (or truncated if
   wider).
2. **Advances the state** with `compute_new_pose` (`world_model.py`):
   * `ACTION_ONLY_EMBODIMENTS` (`franka_omron` and the four DROID action-key ids): the state stays all zeros.
   * `STATE_DELTA_EMBODIMENTS` (`franka_omron_state_delta`): `state + action` on the un-padded prefix.
   * everything else: DROID-style integration of the **first 7 slots only**: `xyz += Δxyz`,
     `R ← R(Δeuler)·R`, `gripper = clip(gripper + Δg, 0, 1)`; the 7-D result is zero-padded back to 64.

| ids | consistent with training? |
|---|---|
| `franka`, `franka_roboset`, `robomind_franka`, `franka_fmb`, `bridge`, `google`, `kuka`, `franka_robonet`, `baxter`, `sawyer`, `widowx_robonet`, `kuka_robonet`, `so-101` | **Yes.** State = xyz + euler + gripper and action = its state-delta, so a 7-D planner action is the training layout and the integration inverts `poses_to_diffs`. The gripper clip assumes [0, 1]. |
| `franka_omron`, `franka_omron_state_delta` | Yes, via dedicated preprocessors ([robocasa.md](robocasa.md)). |
| `franka_cart_action_key`, `franka_joint_action_key`, `franka_cart_vel_action_key`, `franka_joint_vel_action_key` | State: yes (zeros). **Action: no preprocessor**; the caller must supply the full 56/64-D chunk. |
| `franka_joints`, `so-101_joints` | **No.** The state is joints ⊕ gripper, but the integration treats slots 3–5 as euler angles and slot 6 as a [0, 1] gripper. |
| `agibot_robotic` (25), `agibot_humanoid` (35), `agilex` (14), `1x_robot` (25) | **No.** The integration keeps only 7 slots and reads them as one xyz + euler + gripper. |
| `language_table` (2) | Approximately: slots 0–1 add correctly, and slots 2–6 stay zero if the planner leaves them zero. |

The paper's deployment (App. B.7) plans only with a Franka-style 7-D end-effector delta, i.e. the first row.

## Per-dataset summary (`wm_4b.yaml`, training)

| dataset | embodiment (id) | `s3_prefix` | source | state | action / WM step | raw → WM fps | raw frames per step |
|---|---|---|---|---|---|---|---|
| Droid | `franka` (0) | `droid/1.0.1/` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 15→1.25 (config ×4) | 12 |
| Droid_joints | `franka_joints` (1) | `droid/1.0.1/` | state_delta (joints) | joint[7]+grip[1] = 8 | elementwise Δ = 8 | 15→1.25 (config ×4) | 12 |
| Droid_cart_action_key | `franka_cart_action_key` (21) | `droid/1.0.1/` | action_key:droid_cart | zeros | 8×7 = 56 | 15→1.875 (config ×4) | 8 |
| Droid_joint_action_key | `franka_joint_action_key` (22) | `droid/1.0.1/` | action_key:droid_joint | zeros | 8×8 = 64 | 15→1.875 (config ×4) | 8 |
| Droid_cart_vel_action_key | `franka_cart_vel_action_key` (23) | `droid/1.0.1/` | action_key:droid_cart_vel | zeros | 8×7 = 56 | 15→1.875 (config ×4) | 8 |
| Droid_joint_vel_action_key | `franka_joint_vel_action_key` (24) | `droid/1.0.1/` | action_key:droid_joint_vel | zeros | 8×8 = 64 | 15→1.875 (config ×4) | 8 |
| Lerobot | `so-101` (4) | `lerobot/single_arm_labeled` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 30→5 | 6 |
| Lerobot_joints | `so-101_joints` (25) | `lerobot/single_arm_labeled` | state_delta (joints) | joint[5]+grip[1] = 6 | elementwise Δ = 6 | 30→5 | 6 |
| AGIBot_robotiq | `agibot_robotic` (5) | `agibot/robotiq` | state_delta (cartesian) | cart[23]+grip[2] = 25 | elementwise Δ = 25 | 30→1 | 30 |
| robomind_agilex | `agilex` (8) | `robomind/robomind_agilex` | state_delta (cartesian) | cart[12]+grip[2] = 14 | elementwise Δ = 14 | 30→5 | 6 |
| RoboCasa | `franka_omron` (2) | `robocasa/human_720p/` | action_key:robocasa | zeros | 4×12 = 48 | 20→5 | 4 |
| RoboCasa_state_delta | `franka_omron_state_delta` (3) | `robocasa/human_720p/` | state_delta (cartesian) | cart[13]+grip[1] = 14 | elementwise Δ = 14 | 20→5 | 4 |
| Roboset | `franka_roboset` (6) | `roboset/` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 5→1 | 5 |
| OXE_Bridge2 | `bridge` (7) | `oxe/bridge2` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 5→5 | 1 |
| Robomind_franka | `robomind_franka` (9) | `robomind/robomind_franka` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 30→5 | 6 |
| OXE_RT1 | `google` (10) | `rt1` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 10→5 | 2 |
| OXE_Bridge | `bridge` (7) | `oxe/bridge` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 5→5 | 1 |
| OXE_qtopt | `kuka` (11) | `qtopt` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 10→10 | 1 |
| OXE_mtopt | `kuka` (11) | `oxe/mtopt` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 5→5 | 1 |
| OXE_robonet_franka | `franka_robonet` (12) | `oxe/robonet/franka` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 1→1 | 1 |
| OXE_robonet_baxter | `baxter` (13) | `oxe/robonet/baxter` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 1→1 | 1 |
| OXE_robonet_sawyer | `sawyer` (14) | `oxe/robonet/sawyer` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 1→1 | 1 |
| OXE_robonet_widowx | `widowx_robonet` (15) | `oxe/robonet/sawyer` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 1→1 | 1 |
| OXE_robonet_kuka | `kuka_robonet` (18) | `oxe/robonet/kuka` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 1→1 | 1 |
| OXE_language_table_sim | `language_table` (16) | `oxe/language_table_sim/` | state_delta (cartesian) | cart[2]+grip[0] = 2 | elementwise Δ = 2 | 5→5 | 1 |
| OXE_language_table_real | `language_table` (16) | `oxe/language_table/` | state_delta (cartesian) | cart[2]+grip[0] = 2 | elementwise Δ = 2 | 5→5 | 1 |
| OXE_bcz | `google` (10) | `oxe/bc_z/` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 10→5 | 2 |
| OXE_fmb | `franka_fmb` (17) | `oxe/fmb/` | state_delta (cartesian) | cart[6]+grip[1] = 7 | xyz Δ + euler(R₊Rᵀ) + grip Δ = 7 | 10→5 | 2 |
| 1X | `1x_robot` (20) | `1x-data` | state_delta (cartesian) | cart[25]+grip[0] = 25 | elementwise Δ = 25 | 30→3 | 10 |
| AGIBot_humanoid | `agibot_humanoid` (19) | `agibot/humanoid` | state_delta (cartesian) | cart[23]+grip[12] = 35 | elementwise Δ = 35 | 30→1 | 30 |

Eval-only datasets in `data.eval` (not trained on): `DROID_objectstore` (`franka`, `droid/1.0.1/`),
`RoboCasa_target` (`franka_omron`, `robocasa/human_target_720p/`), `Lerobot` (`so-101`,
`lerobot_eval_data_labeled/`).

## Cross-cutting caveats

1. **DROID rates are reported ×4.** DROID configs say `raw_fps: 60`, but the videos are 15 fps recordings sped up
   4× and the control stream is 15 Hz. Frame indexing is consistent, so training is unaffected: divide every
   DROID rate by 4 (state-delta ids: 0.8 s per WM step; action-key ids: 8 commands = 0.53 s). See
   [droid.md](droid.md).
2. **`widowx_robonet` reads Sawyer data.** Both `OXE_robonet_sawyer` and `OXE_robonet_widowx` have
   `s3_prefix: oxe/robonet/sawyer`, so id 15 is trained on the same episodes as id 14.
3. **Paper vs config.**
   * Paper App. B.7 says DROID action-key runs at 5 Hz with 12-command chunks; the config uses `fps: 7.5`, i.e.
     8 commands.
   * App. B.7 says the state-delta rotation is computed in so(3); the code computes `R_{t+1}R_tᵀ` as euler xyz for
     `pose_dim ≤ 6`, and an element-wise difference above that.
   * App. B.7 calls the AgiBot action 23-D; it is 25-D (23 pose + 2 gripper).
4. **Gripper scale is not uniform** (paper App. B.4 says every gripper is a [0, 1] closedness):

   | gripper scale | datasets |
   |---|---|
   | [0, 1], 1 = closed | DROID, RT-1 (`gripper_closed`), QT-Opt / MT-Opt (binary), FMB (binary), Bridge v1/V2, Roboset (≥ 0.46 in practice), 1X hand flags |
   | [0.2, 1] | BC-Z (`sensed_close`) |
   | raw units | **AgiBot** (≈ 35–122), **SO-101** (0–1.745), AgileX / RoboMind (joint value) |
   | none | **RoboNet** (identically 0), Language-Table (no gripper) |

   Bridge's recorded *action* gripper keeps the raw 1 = open convention ([bridge.md](bridge.md)).
5. **Rotation encoding is not uniform.** Most sources store xyz-euler, but **BC-Z stores axis-angle** in the euler
   slots ([google_robot.md](google_robot.md)) and **RoboNet stores its yaw in the roll slot**
   ([robonet.md](robonet.md)). Roll sits near ±π for DROID, RT-1, QT-Opt, MT-Opt, FMB, Roboset and RoboMind Franka:
   harmless for the `pose_dim = 6` matrix path, but visible in raw states and in `franka_cart_action_key`.
6. **Units are not uniform.** RoboNet xyz is workspace-normalized to ≈ [0, 1] per robot, and Language-Table sim
   motion never exceeds 0.03 per step while real reaches 0.15. Nothing is normalized at load time.
