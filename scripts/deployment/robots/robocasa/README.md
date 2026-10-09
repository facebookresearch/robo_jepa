# RoboCasa Simulation Environment

WM-aligned RoboCasa wrapper for the robo_jepa framework. Lets a planner /
imagination loop drive the env using the **same canonical PandaOmron action
layout** the world model was trained on, so actions can be sent verbatim
without any frame / scaling / parametrisation conversion.

## Prerequisites

```bash
conda activate jepa_311
```

EGL rendering is configured automatically (`MUJOCO_GL=egl`).

## Quick Start

### Start the server

```bash
cd scripts/deployment/robots/robocasa
python robot_env_server.py --port 4754
```

### Test with random arm-mode actions (saves a video)

```bash
python test_env.py --mode random_arm --num_ticks 30 --output test_rollout.mp4
```

### Replay a real dataset episode (best alignment check)

```bash
python test_env.py \
    --mode dataset \
    --dataset_h5 /tmp/robocasa_inspect/data.h5 \
    --num_ticks 50 \
    --output dataset_rollout.mp4
```

## Action Format

The env accepts **three** equivalent action forms; all are coerced into
the canonical 12-D PandaOmron layout-A chunk before reaching the OSC
controller.

### 1. Canonical 12-D layout A (in [-1, 1])

| Index | Channel | Notes |
|------:|---|---|
| 0–2 | `base_motion` (x, y, yaw) | body-frame velocity command |
| 3   | `torso` | spine vertical delta |
| 4   | `control_mode` | binary ±1 (−1 = arm mode, +1 = base mode) |
| 5–7 | `ee_pos` delta | base frame, normalised by OSC `output_max=0.05 m` |
| 8–10 | `ee_rot` delta | axis-angle, base frame, normalised by OSC `output_max=0.5 rad` |
| 11  | `gripper` | binary ±1 (−1 open, +1 close) |

Per `env.step(action)` you send a chunk of `chunk_size` such 12-D vectors,
either as `(chunk_size, 12)` or flattened to `(chunk_size * 12,)`. The
wrapper steps the underlying robosuite env once per sub-step (so one
`env.step` = `chunk_size` OSC ticks).

`chunk_size` defaults to 4 (matches the WM training raw_fps / fps =
20 / 5). Override on the server with `--chunk_size`.

In this form the wrapper does **not** renormalise, **not** convert
Euler→axis-angle, and **not** override the robocasa default controller
config — what you send is exactly what the controller consumes.

### 2. Components dict (planner-side, metric units)

The planner / WM side sends an `action_components.split_7d_action`-style
dict (`{"eef_translation": (m, base frame), "eef_rotation": (rad,
axis-angle), "gripper": ±1, ...}`). The wrapper packs it into the
canonical 12-D vector via `merge_components(components, wm_embodiment)`,
which since the May-2026 fix:

* divides `eef_translation` (metres) by `0.05` and clips to `[-1, 1]`,
* divides `eef_rotation` (rad) by `0.5` and clips to `[-1, 1]`,
* applies the RoboCasa server-side defaults (`base_motion = 0`,
  `control_mode = -1`),
* binarises the gripper.

The single 12-D vector is then broadcast over the chunk (one canonical
action per WM step → `chunk_size` identical OSC sub-steps), matching the
convention the WM was trained on.

### 3. Legacy 7-D `[dx, dy, dz, droll, dpitch, dyaw, dgrip]`

A flat 7-D numpy array is treated as a planner DROID-style metric delta
in base frame (Euler XYZ in slots 3:6). It is converted to a components
dict via `split_7d_action` and then routed through the path above.

### `wm_embodiment`: `franka_omron` vs `franka_omron_state_delta`

The server's `--wm_embodiment` flag selects the variant of
`merge_components` used to pack components-dict / 7-D inputs:

* **`franka_omron`** (default): canonical layout-A 12-D action with the
  normalisation above.
* **`franka_omron_state_delta`**: same packed 12-D layout, but the
  rotation slots `[8:11]` are *asserted* zero on input and zeroed before
  the OSC controller. This matches the temporary contract for the
  state-delta WM variant (see
  `app/robo_jepa/embodiment_preprocessing.py`), which doesn't yet handle
  Δquat rotations.

Both variants produce the same env-side OSC execution; the difference
is only in the WM-side state/action representation (see Observation
Format below — `proprio_robocasa` is needed for the state-delta WM).

## Observation Format

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `pose` | `(7,)` | float64 | `[x, y, z, roll, pitch, yaw, gripper_closure]` (world frame). The legacy 7-D proprio used by `franka_omron`. |
| `proprio_robocasa` | `(14,)` | float64 | RoboCasa 14-D proprio used by the `franka_omron_state_delta` WM variant: `[base_pos(3), base_quat_xyzw(4), ee_pos_rel(3), ee_rot_rel_quat_xyz(3), gripper_qpos(1)]`. `ee_pos_rel` / `ee_rot_rel` are in the **base** frame (`R_bw @ (eef_world − base_pos)` for position; `R_bw @ R_we` quaternion[xyz] for rotation). Same layout as the training data. |
| `joints` | `(7,)` | float64 | Joint positions |
| `img_0` | `(H, W, 3)` | uint8 | First camera image |
| `img_N` | `(H, W, 3)` | uint8 | N-th camera image |
| `time_step_begin` | scalar | float | ms timestamp |
| `time_step_end`   | scalar | float | ms timestamp |
| `time_send_obs`   | scalar | float | ms timestamp |

`info` from `step()` contains `success`, `sim_steps` (= `chunk_size` in
fixed-step mode, or the variable convergence count when the server is
started with `--blocking_control`), and `control_modes` (the actual ±1
used for each sub-step).

## CLI Reference

### `robot_env_server.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `4754` | TCP port |
| `--task_name` | `CloseStandMixerHead` | RoboCasa task |
| `--camera_names` | `robot0_agentview_left` | Cameras (space-separated) |
| `--camera_width` | `256` | Image width |
| `--camera_height` | `256` | Image height |
| `--seed` | `None` | Random seed |
| `--robot` | `PandaOmron` | Robot configuration |
| `--no_init_home` | off | Skip resetting arm to home pose |
| `--chunk_size` | `4` | Sub-steps per WM tick |
| `--no_lock_base` | off | Disable camera-stability base-joint lock |
| `--wm_embodiment` | `franka_omron` | Selects the `merge_components` branch for components-dict / 7-D actions. Choose `franka_omron_state_delta` for the state-delta WM variant — both produce a 12-D layout-A action; the state-delta path additionally asserts zero planner rotation and zeros the rotation slots before the OSC controller. |
| `--blocking_control` | off | Wrap the env with `BlockingRoboCasaEnv`: each `env.step` interprets the planner's xyz delta as a **target offset** (base-frame metres) and runs OSC sub-steps until the EE converges to that target (or hits `--blocking_max_steps`). Without this flag the env runs exactly `chunk_size` OSC sub-steps and physically realises only a fraction of the commanded delta. |
| `--blocking_pos_tol` | `0.001` | Convergence tolerance (m, world-frame Euclidean) when `--blocking_control` is on. |
| `--blocking_max_steps` | `200` | Hard cap on OSC sub-steps per `env.step` in blocking mode. |

### `test_env.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--task_name` | `CloseStandMixerHead` | RoboCasa task |
| `--camera_names` | `robot0_agentview_left robot0_eye_in_hand` | Cameras |
| `--camera_width` | `256` | Image width |
| `--camera_height` | `256` | Image height |
| `--seed` | `42` | Random seed |
| `--chunk_size` | `4` | Sub-steps per WM tick |
| `--num_ticks` | `30` | Number of WM ticks (each = chunk_size sub-steps) |
| `--mode` | `random_arm` | One of `random_arm`, `random_base`, `dataset` |
| `--dataset_h5` | none | Path to a v0.1 `data.h5` (when `--mode dataset`) |
| `--mag` | `0.2` | Random action magnitude |
| `--output` | `test_rollout.mp4` | Output video path |
| `--fps` | `15` | Output video FPS |

## Client-Side Parameterization

The environment can be reconfigured at reset time via kwargs:

```python
from gym_tcp import GymEnvClient

env = GymEnvClient(host="localhost", port=4754)

obs, _ = env.reset()                                    # defaults
obs, _ = env.reset(task_name="PnPCounterToCab")
obs, _ = env.reset(camera_names=["robot0_agentview_left",
                                 "robot0_agentview_right"],
                   camera_width=512, camera_height=512)
obs, _ = env.reset(seed=42)                             # reproducibility
```

## Camera-Stability Base Lock

The mobile base can drift by a few mm per OSC tick under arm-mode commands
(numerical noise + wheel friction), which is visually distracting. The
wrapper snapshots the base joints after each sub-step and, if the
sub-step's `control_mode` was −1, restores them so the cameras stay still.
Base-mode (`control_mode = +1`) sub-steps execute normally and the snapshot
is updated. Disable with `--no_lock_base` (server) or
`lock_base_in_arm_mode=False` (constructor) if you don't want it.

## Blocking control mode (`BlockingRoboCasaEnv`)

`blocking_robot_env.py` provides a drop-in wrapper around `RoboCasaEnv`
for closed-loop, metric-target control. It is enabled with the
`--blocking_control` server flag.

With blocking control, each `env.step(action)` does **not** just run
`chunk_size` OSC sub-steps. Instead the wrapper:

1. Reads the planner's per-WM-step base-frame Δxyz from the action
   (whether that arrived as a components dict, a 7-D vector, or a
   canonical 12-D chunk — collapsed back to a single intent via the OSC
   `output_max=0.05 m`).
2. Converts it to a world-frame target `current + R_wb @ Δxyz`.
3. Runs OSC sub-steps until the EE is within `--blocking_pos_tol` of
   that target (or it hits `--blocking_max_steps`), clipping each
   sub-step's correction to the OSC controller's per-step capacity.

This closes the loop in the env so the realised motion matches the
planner's intent — important for the new `franka_omron_state_delta` WM,
which expresses actions in the metric scale of the in-distribution
14-D state delta. `info["sim_steps"]` reports the actual variable
sub-step count instead of a fixed `chunk_size`.

## Available Cameras

Common RoboCasa cameras:
- `robot0_agentview_left` (default)
- `robot0_agentview_right`
- `robot0_eye_in_hand`
- `robot0_agentview_center`

## Available Tasks

Any task registered in RoboCasa, e.g.:
- `CloseStandMixerHead`
- `PnPCounterToCab`
- `OpenSingleDoor`
- `CloseDoubleDoor`
- `TurnOnStove`

## Custom reach task

The wrapper also implements a **scene-agnostic reach task** that does
NOT depend on any specific robocasa task's success/waypoint definition.
When `--reach_task` is set, the underlying robocasa task (`--task_name`,
e.g. `CloseStandMixerHead`) is used only as a kitchen backdrop; the
EE is driven via closed-loop blocking control to:

* `reset(mode="goal")` → a random **goal** pose,
* `reset(mode="init")` → a random **start** pose.

Both poses are sampled uniformly inside a 3-D ball of radius
`--reach_max_distance` (default 0.20 m) around the home EEF position
(captured on the very first reset). Sampling is fully deterministic
from `--reach_task_seed` and uses a **dedicated** `np.random.RandomState`
that does NOT touch the underlying scene RNG (`--seed`), so:

* changing `--reach_task_seed` reseeds only the start/goal pair and
  leaves the kitchen layout / objects identical;
* changing `--seed` reseeds only the kitchen and leaves the start/goal
  pair identical.

The driving uses the same base-frame OSC corrective + camera-stability
base lock as the regular goal generator
(`_drive_eef_to_world_target` mirrors `_step_blocking_expert` but with
an absolute world-frame target). The returned obs gains three extra
keys for diagnostics:

| Key | Shape | Description |
|---|---|---|
| `reach_target_world` | `(3,)` | World-frame XYZ the EE was driven toward |
| `reach_home_world` | `(3,)` | The home EEF pos the offsets were sampled around |
| `reach_residual` | scalar | Final world-frame Euclidean error (m) at exit |

The `--blocking_control` server flag composes orthogonally with the
reach task — both can be enabled together (the reach drive happens at
reset time; `BlockingRoboCasaEnv` then handles per-step blocking
during the rollout).

Server example:

```bash
python robot_env_server.py --port 4754 \
    --task_name CloseStandMixerHead \
    --camera_names robot0_agentview_left robot0_eye_in_hand \
    --camera_width 320 --camera_height 240 \
    --seed 42 \
    --reach_task --reach_task_seed 0 --reach_max_distance 0.20 \
    --blocking_control --blocking_pos_tol 0.001 --blocking_max_steps 200
```

A standalone `reach_task_preview.py` script renders a 2×2 grid (start /
goal × two reach seeds) for visual sanity-checking:

```bash
python reach_task_preview.py \
    --task_name CloseStandMixerHead --seed 42 \
    --reach_seeds 0 1 --reach_max_distance 0.20 \
    --output reach_task_preview.png
```
