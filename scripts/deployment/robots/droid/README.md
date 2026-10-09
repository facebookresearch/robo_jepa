# DROID Robot — Setup & Deployment Guide

This directory contains the robo_jepa environment server for the
[DROID](https://droid-dataset.github.io/) robot platform (Franka Panda
controlled via the original DROID stack). The server wraps `droid.robot_env.RobotEnv`
into the unified Gym interface expected by all robo_jepa agents.

```
Agent cluster (N GPUs)                                        Robot PC (DROID platform)
──────────────────────────────────────────                    ──────────────────────────
                                                             ┌──────────────────────────────────────────────────────────┐
  Rank 0 ──────────────────────────────────┐                 │                                                          │
    │                                      │                 │                                                          │
    │ env.reset(goal) ── TCP/ZMQ ─────────>│────────────────>│  move to goal pose, capture goal image                   │
    │                 <── TCP/ZMQ ─────────│<────────────────│  goal obs (image + pose + joints)                        │
    │ env.reset(init) ── TCP/ZMQ ─────────>│────────────────>│  move to start pose                                      │
    │                 <── TCP/ZMQ ─────────│<────────────────│  initial obs (image + pose + joints)                     │
    │                                      │                 │                                                          │
    │ 1. encode(goal_image) → goal_features│                 │                                                          │
    │    encode(obs_image)  → obs_features │                 │                                                          │
    │ 2. dist.broadcast(goal_features) ───>│──> all ranks    │                                                          │
    │    dist.broadcast(obs_features) ────>│──> all ranks    │                                                          │
    │    dist.broadcast(obs_pose) ────────>│──> all ranks    │                                                          │
    │                                      │                 │                                                          │
  All ranks (CEM planning loop) ───────────┤                 │                                                          │
    │  for cem_iter in 1..15:              │                 │                                                          │
    │    • sample 16 action sequences      │                 │                                                          │
    │    • rollout WM for R steps          │                 │                                                          │
    │    • score vs goal features (L1)     │                 │                                                          │
    │    • dist.all_gather(rewards) <──────│──> all ↔ all    │                                                          │
    │    • dist.all_gather(actions) <──────│──> all ↔ all    │                                                          │
    │    • elite selection (top 2%)        │                 │                                                          │
    │    • update mean/std                 │                 │                                                          │
    │                                      │                 │                                                          │
  Rank 0 ──────────────────────────────────┘                 │                                                          │
    │                                                        │                                                          │
    │ best action                                            │                                                          │
    │                                                        │                                                          │
    │  env.step(action)  ─────────────── TCP/ZMQ ──────────> │  DroidEnvWrapper.step(action)                            │
    │                    <────────────── TCP/ZMQ ─────────── │  observation dict                                        │
    │                                                        │  │                                                       │
    │ (loop back to step 1)                                  │  └─> droid.robot_env.RobotEnv                            │
    │                                                        │        ├── ServerInterface / FrankaRobot                 │
                                                             │        ├── MultiCameraWrapper                            │
                                                             │        │     ├── ZedCamera (ZED SDK, stereo)             │
                                                             │        │     └── RealSenseCamera (pyrealsense2, custom)  │
                                                             │        └── calibration_utils                             │
                                                             └──────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. DROID software stack

The full DROID stack must be installed and functional on the robot PC. This
includes the Franka low-level controller (either running locally or on a NUC
via `ServerInterface`), ZED camera drivers, and calibration data.

Follow the [DROID installation guide](https://github.com/droid-dataset/droid)
to set up:

- **`droid` Python package** — must be importable (`from droid.robot_env import RobotEnv`).
  Follow the [DROID installation guide](https://github.com/droid-dataset/droid)
  for setup instructions.
- **Franka controller** — either a local `FrankaRobot` instance or a remote
  NUC running `zerorpc` (configured via `droid.misc.parameters.nuc_ip`).
- **ZED SDK + ZED cameras** — `MultiCameraWrapper` auto-discovers all
  connected ZED stereo cameras via `pyzed`. Make sure the ZED SDK is
  installed and cameras are plugged in before starting the server.
- **Intel RealSense camera** *(custom modification)* — in addition to
  the upstream DROID ZED-only setup, `gather_zed_cameras()` appends one
  `RealSenseCamera` instance that reads from an Intel RealSense device
  via `pyrealsense2`. It captures 640×480 BGR frames, pads them
  horizontally, and resizes to 1280×720 to match the ZED output
  resolution. The RealSense uses a fixed serial number `'0000000'` and
  its image appears as key `0000000_left` in the DROID observation.
  Install the RealSense SDK and Python bindings:
  ```bash
  pip install pyrealsense2
  ```
- **Camera calibration** — run the DROID calibration procedure to generate
  `calibration/` data (extrinsics for each camera). Without it
  `load_calibration_info()` will fail.

### 2. Python dependencies

The server additionally requires the packages used by `gym_tcp.py`:

```bash
pip install pyzmq msgpack-python
```

### 3. Network parameters

Edit `droid/misc/parameters.py` on the robot PC to match your hardware:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nuc_ip` | NUC IP address (set to `None` if running the controller locally) | `172.16.0.3` |
| `robot_ip` | Franka robot IP | `172.16.0.2` |
| `hand_camera_id` | Serial number of the wrist ZED camera | `12260683` |
| `varied_camera_1_id` | Serial number of the first external ZED camera | `""` |
| `varied_camera_2_id` | Serial number of the second external ZED camera | `""` |

> The RealSense camera uses a hardcoded serial `'0000000'` and does not
> need a parameter entry.

---

## Step 1: Start the Robot

Follow the standard DROID procedure to bring the Franka online:

1. Open the Franka dashboard and unlock the robot.
2. Activate FCI.
3. If using a NUC, SSH in and start the controller.

Verify the robot is reachable:

```bash
ping 172.16.0.2   # robot
ping 172.16.0.3   # NUC (if applicable)
```

---

## Step 2: Start the Environment Server

From the robot PC:

```bash
cd scripts/deployment/robots/droid
python robot_env_server.py --port 12345
```

You should see:

```
[GymEnvServer] listening on tcp://0.0.0.0:12345
```

The server is now waiting for agent connections.

### Server arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Network interface to bind to. |
| `--port` | `4754` | TCP port for agent connections. |
| `--action_space` | `cartesian_velocity` | DROID action space. One of: `cartesian_velocity`, `cartesian_position`, `joint_velocity`, `joint_position`. |
| `--gripper_action_space` | `None` | DROID gripper action space (e.g. `velocity`). `None` uses the DROID default. |
| `--camera_kwargs` | `{}` | JSON string forwarded to `MultiCameraWrapper` for per-camera-type settings (resolution, etc.). |
| `--no_reset` | `false` | Skip the initial `reset()` in the DROID `RobotEnv` constructor. Useful when you want to preserve the robot's current pose. |

### Examples

**Default (Cartesian velocity, all cameras, reset on start):**

```bash
python robot_env_server.py --port 12345
```

**Joint velocity control, no initial reset:**

```bash
python robot_env_server.py --port 12345 \
    --action_space joint_velocity \
    --no_reset
```

**Custom camera resolution:**

```bash
python robot_env_server.py --port 12345 \
    --camera_kwargs '{"hand_camera": {"resolution": [640, 480]}, "varied_camera": {"resolution": [1280, 720]}}'
```

---

## Step 3: Connect an Agent

### From the same machine

```python
from gym_tcp import GymEnvClient

env = GymEnvClient(host="localhost", port=12345)
obs, _ = env.reset()

for _ in range(100):
    action = your_policy(obs)  # shape (7,) for cartesian, (8,) for joint
    obs, reward, done, info = env.step(action)
    if done:
        obs, _ = env.reset()

env.close()
```

### From a remote cluster (SSH port forwarding)

Set up an SSH port-forwarding chain from the compute node to the robot PC:

```
cluster compute node  <-->  login node  <-->  dev server  <-->  robot PC
     port 12345            port 34567        port 23456      port 12345
```

Then the agent connects to `localhost:12345` on the compute node.

### World-model planner (multi-node SLURM)

```bash
cd scripts/deployment/agents/world_model
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    policy_client.py \
    --checkpoint_path /path/to/model \
    --total_steps 12 --action_substeps 1 \
    --use_gripper --translation_only \
    --embed_dim 1664 \
    --camera_id 0 --embodiment_id 0 \
    --use_cameras 0 \
    --use_cache --samples 16 --num_batches 1 \
    --cem 15 --step_cem 10 --rollout 2 --cutoff 20 \
    --maxgripper 0.1 --gripper_init_std 0.05 --robot_init_std 0.06 \
    --momentum_std 0.75 --momentum_mean_gripper 0.15 --momentum_std_gripper 0.45 \
    --execution_prefix 1 --closed_loop_context_len 4 \
    --port 12345 \
    --task_path tasks_droid/final/grasp \
    --output_filename ./robo_jepa/droid/grasp_cup/ep
```

See `agents/world_model/README.md` for the full argument reference and
multi-view examples.

---

## World-Model Planner — Process Scheme

The planner distributes CEM across all GPUs of the job. Only
rank 0 talks to the robot; all other ranks participate exclusively in
the CEM optimisation. The closed-loop MPC repeats the cycle below for
each of the `total_steps` environment steps.

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  EPISODE START (see top diagram for reset sequence)                       │
 │                                                                          │
 │  Rank 0:                                                                 │
 │    env.reset(goal) ── TCP ──> robot → goal obs                           │
 │    env.reset(init) ── TCP ──> robot → initial obs                        │
 │    encode goal image  → goal_features tensor                             │
 │    encode init image  → obs_features tensor                              │
 │                                                                          │
 │  dist.broadcast(goal_features, src=0)     rank 0 → all ranks            │
 │  dist.broadcast(obs_features, src=0)      rank 0 → all ranks            │
 │  dist.broadcast(obs_pose, src=0)          rank 0 → all ranks            │
 └──────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────────────────┐
 │  PER ENV STEP (repeated total_steps=12 times)                            │
 │                                                                          │
 │  ┌─ All ranks: WM context warm-up ─────────────────────────────────┐    │
 │  │  Feed last closed_loop_context_len=4 (obs, action) pairs into    │    │
 │  │  the transition model to prime the KV cache.                     │    │
 │  └──────────────────────────────────────────────────────────────────┘    │
 │                                                                          │
 │  ┌─ All ranks: CEM loop (15 iters at step 0, 10 after) ────────────┐   │
 │  │                                                                   │   │
 │  │  Per iteration:                                                   │   │
 │  │    1. Each rank samples 16 candidate action sequences             │   │
 │  │       (rollout=2 WM steps ahead, 7-D actions)                     │   │
 │  │                                                                   │   │
 │  │    2. Each rank unrolls the WM for 2 steps per candidate          │   │
 │  │       (16 parallel rollouts in bfloat16)                          │   │
 │  │                                                                   │   │
 │  │    3. Each rank scores candidates: L1(predicted, goal) in         │   │
 │  │       feature space                                               │   │
 │  │                                                                   │   │
 │  │    4. Communication:                                              │   │
 │  │         dist.all_gather(rewards)   all ↔ all                      │   │
 │  │         dist.all_gather(actions)   all ↔ all                      │   │
 │  │       → every rank now has all N×16 candidates                    │   │
 │  │                                                                   │   │
 │  │    5. Elite selection: keep the top 20% of candidates             │   │
 │  │       Update Gaussian mean + std (with momentum)                  │   │
 │  │                                                                   │   │
 │  └───────────────────────────────────────────────────────────────────┘   │
 │                                                                          │
 │  Result: refined action sequence of length rollout=2                     │
 │  Take the first execution_prefix=1 action from that sequence.            │
 │                                                                          │
 │  ┌─ Rank 0 only: execute + observe ────────────────────────────────┐    │
 │  │  env.step(action) ── TCP ──> robot                               │    │
 │  │                    <── TCP ── new observation                     │    │
 │  │  encode new obs image → obs_features                             │    │
 │  └──────────────────────────────────────────────────────────────────┘    │
 │                                                                          │
 │  dist.broadcast(obs_features, src=0)      rank 0 → all ranks            │
 │  dist.broadcast(obs_pose, src=0)          rank 0 → all ranks            │
 │                                                                          │
 └──────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────────────────┐
 │  EPISODE END                                                             │
 │                                                                          │
 │  Rank 0: send no-op (hold pose), save video + trajectory .npz            │
 │  dist.barrier()  — all ranks sync before next episode                    │
 └──────────────────────────────────────────────────────────────────────────┘
```

**Key compute numbers (typical DROID config):**

| Parameter | Value |
|---|---|
| Total GPUs | N (one CEM rank per GPU) |
| CEM candidates per iteration | 16 × N (16 samples per rank) |
| CEM refinement iterations | 15 (first step), 10 (subsequent) |
| WM rollout horizon | 2 steps |
| Actions executed per plan | 1 (`execution_prefix`) |
| WM context window | 4 frames |
| Env steps per episode | 12 |

**GPU communication per env step:**

| Call | Direction | When |
|---|---|---|
| `dist.broadcast(obs_features)` | rank 0 → all | after encoding new observation |
| `dist.broadcast(obs_pose)` | rank 0 → all | after encoding new observation |
| `dist.all_gather(rewards)` × N_cem | all ↔ all | end of each CEM iteration |
| `dist.all_gather(actions)` × N_cem | all ↔ all | end of each CEM iteration |

Total per step: **2 broadcasts + 2×N_cem all-gathers** (N_cem = 10 or 15).

---

## Action Format

The action dimensionality depends on the `--action_space` chosen when
starting the server (matching `RobotEnv.DoF`):

| Action space | DoF | Vector layout |
|---|---|---|
| `cartesian_velocity` | **7** | `[dx, dy, dz, droll, dpitch, dyaw, d_gripper]` |
| `cartesian_position` | **7** | `[dx, dy, dz, droll, dpitch, dyaw, d_gripper]` |
| `joint_velocity` | **8** | `[dj0, dj1, dj2, dj3, dj4, dj5, dj6, d_gripper]` |
| `joint_position` | **8** | `[dj0, dj1, dj2, dj3, dj4, dj5, dj6, d_gripper]` |

The last element is always the gripper command. The preceding elements are
interpreted by the DROID controller according to the action space — e.g.
Cartesian deltas in metres / radians, or joint-level commands.

For velocity action spaces, the DROID `RobotEnv` asserts that all values
are in [−1, 1].

The action is forwarded directly to `droid.robot_env.RobotEnv.step()`
without any remapping.

---

## Observation Format

Each `step()` and `reset()` returns an observation dictionary:

| Key | Shape | Description |
|-----|-------|-------------|
| `pose` | `(7,)` float64 | End-effector pose: `[x, y, z, roll, pitch, yaw, gripper_position]`. Position from the 4×4 homogeneous transform, orientation converted to Euler XYZ. |
| `joints` | `(7,)` float64 | Joint positions (radians). |
| `img_0`, `img_1`, … | `(H, W, 3)` uint8 | Camera images — one per ZED camera plus one RealSense camera. Sorted by camera serial number. RGBA images are converted to RGB. |
| `time_step_begin` | float | Timestamp (ms) when the observation read started. |
| `time_step_end` | float | Timestamp (ms) when the observation read finished. |
| `time_send_obs` | float | Timestamp (ms) when the observation was packed for sending. |

### Camera ordering

`MultiCameraWrapper` discovers cameras via `gather_zed_cameras()`
(`droid/camera_utils/camera_readers/zed_camera.py`), which returns:

1. **All connected ZED stereo cameras** — discovered through the ZED SDK
   (`sl.Camera.get_device_list()`). Each `ZedCamera` produces left and
   right images keyed by `<serial>_left` / `<serial>_right`.
2. **One Intel RealSense camera** *(custom modification)* — a
   `RealSenseCamera` instance that captures 640×480 BGR via
   `pyrealsense2`, pads horizontally, and resizes to 1280×720. Its image
   is keyed `0000000_left`.

Images are assigned to `img_0`, `img_1`, … in **sorted order of camera
serial number** (the RealSense serial `'0000000'` sorts first). The DROID
convention maps camera serial numbers to types:

| Type | Serial | `camera_id` (world model) | Description |
|------|--------|---------------------------|-------------|
| RealSense | `0000000` | — | External RGB camera (custom) |
| Hand camera | e.g. `12260683` | `2` (wrist) | ZED stereo, mounted on the gripper |
| Varied camera | configured in `parameters.py` | `0` (left) or `1` (right) | ZED stereo, external |

Use `--use_cameras` and `--camera_id` on the policy client to select which
images to feed to the world model and which identity tokens to assign.

---

## Architecture

```
robot_env_server.py
  │
  ├── DroidEnvWrapper          # Adapter: DROID obs → unified obs format
  │     │
  │     └── droid.robot_env.RobotEnv
  │           ├── ServerInterface (NUC) or FrankaRobot (local)
  │           ├── MultiCameraWrapper
  │           │     ├── ZedCamera (ZED SDK stereo cameras)
  │           │     └── RealSenseCamera (Intel RealSense, custom)
  │           ├── calibration_utils → camera extrinsics
  │           └── transformations → pose conversions
  │
  └── GymEnvServer (gym_tcp.py)
        └── ZMQ REP socket, msgpack serialisation
```

`DroidEnvWrapper` is a thin adapter that:

1. Constructs a `droid.robot_env.RobotEnv` with the CLI-specified parameters.
2. On `reset()`: calls `RobotEnv.reset()` and builds a unified observation.
3. On `step(action)`: validates the action dimensionality against
   `RobotEnv.DoF` (7 for cartesian, 8 for joint), forwards it to
   `RobotEnv.step()`, and builds a unified observation from
   `RobotEnv.get_observation()`.
4. Translates the DROID observation dict (nested `robot_state`, `image`,
   `camera_extrinsics`, `camera_intrinsics`, `camera_type`, `timestamp`)
   into the flat `{pose, joints, img_*, time_*}` format.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'droid'`**
- The `droid` package is not installed. Follow the
  [DROID installation guide](https://github.com/droid-dataset/droid) to
  set it up.

**`ModuleNotFoundError: No module named 'pyzed'`**
- Install the ZED SDK and its Python bindings. See
  [ZED SDK installation](https://www.stereolabs.com/docs/installation/).

**`ModuleNotFoundError: No module named 'pyrealsense2'`**
- Install the RealSense SDK Python bindings: `pip install pyrealsense2`.

**No cameras found / empty `img_*` keys**
- Verify ZED cameras are connected: `lsusb | grep -i stereo`.
- Check that the ZED SDK detects them: `python -c "import pyzed.sl as sl; print(sl.Camera.get_device_list())"`.
- For the RealSense, verify: `python -c "import pyrealsense2 as rs; print(rs.context().devices)"`.

**Robot does not move**
- If using a NUC (`nuc_ip` is set), verify the NUC is reachable and the
  `zerorpc` server is running.
- If running locally (`nuc_ip = None`), ensure `FrankaRobot` can connect to
  the robot at `robot_ip`.

**`ConnectionRefused` from the agent**
- Verify the server is running and the port matches.
- If connecting through SSH tunnels, check every leg of the forwarding chain.

**Calibration errors**
- Re-run the DROID camera calibration procedure.
- Ensure the calibration files exist in `droid/calibration/`.
