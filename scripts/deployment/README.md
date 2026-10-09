# robo_jepa — Robot Control & World-Model Planning

This directory contains everything needed to run trained world models (and other agents) on real robots. The design separates **robot environments** from **agents** with a network boundary: environments run on the robot PC and expose a Gym-compatible interface over TCP, while agents run anywhere with network access (a GPU cluster, a laptop, etc.) and interact through the standard `reset()` / `step()` loop.

```
Agent (cluster / laptop)                   Robot PC
─────────────────────────                  ──────────────────────────
GymEnvClient                               GymEnvServer
  env.reset()          ──── TCP ──────>      RobotEnv.reset()
  env.step(action)     ──── TCP ──────>      RobotEnv.step(action)
                       <─── TCP ──────       observation dict
```

## Quickstart

**1. Test the transport layer without any hardware:**

```bash
cd scripts/deployment

# Terminal 1 — fake robot
python mock_robot.py --port 4754

# Terminal 2 — fake agent
python mock_agent.py --port 4754 --mode choreography
```

**2. Run a robot environment (pick one):**

```bash
# DROID
cd scripts/deployment/robots/droid
python robot_env_server.py --port 12345

# RoboCasa (sim)
cd scripts/deployment/robots/robocasa
python robot_env_server.py --port 4754
# state-delta WM variant + closed-loop blocking control:
python robot_env_server.py --port 4754 \
    --wm_embodiment franka_omron_state_delta \
    --blocking_control --blocking_pos_tol 0.001 --blocking_max_steps 200
```

See the per-robot READMEs for hardware setup prerequisites.

**3. Connect an agent:**

```python
from gym_tcp import GymEnvClient

env = GymEnvClient(host="localhost", port=4754)
obs, _ = env.reset()
obs, reward, done, info = env.step(action)
```

**4. Run the world-model planner (multi-node, SLURM):**

```bash
# Real robot (DROID) — multi-node CEM
cd scripts/deployment/agents/world_model
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    policy_client.py \
    --checkpoint_path /path/to/model \
    --port 12345 --use_gripper --translation_only \
    --task_path tasks_droid/final/grasp \
    # ... see agents/world_model/README.md for full args

# RoboCasa simulator — one GPU; each script starts its own env server
bash scripts/deployment/examples/reproduce_reach.sh 50m 120k 0 1 2 3 4    # <size> <step> <task seeds...>
```

---

## Directory layout

```
scripts/deployment/
├── gym_tcp.py              # ZMQ-based Gym server/client (network transport layer)
├── mock_robot.py            # Synthetic robot env for hardware-free testing
├── mock_agent.py            # Scripted agent for verifying any env server
├── action_components.py     # Per-embodiment components-dict <-> flat-vector packing
├── robots/
│   ├── droid/               # DROID environment (Franka via the DROID stack)
│   └── robocasa/            # RoboCasa simulator (PandaOmron, mobile-base + arm)
└── agents/
    └── world_model/         # CEM planner using a trained JEPA world model
```

## Transport layer — `gym_tcp.py`

All communication between agents and robot environments goes through a lightweight ZMQ REQ/REP protocol. Messages are serialized with MessagePack; numpy arrays are encoded as tagged maps so non-Python clients can decode them.

- **`GymEnvServer`** — wraps any object with `reset()` / `step()` / `close()` methods and serves it on a TCP port. One request at a time (correct for robot control: the agent waits for the observation before sending the next action).
- **`GymEnvClient`** — connects to a `GymEnvServer` and provides the same `reset()` / `step()` / `close()` interface as a local Gym env, so agents need no code changes when switching between local and remote environments.

```python
# Server side (robot PC)
from gym_tcp import GymEnvServer
server = GymEnvServer(env, port=4754)
server.start()

# Client side (agent)
from gym_tcp import GymEnvClient
env = GymEnvClient(host="robot-pc", port=4754)
obs, _ = env.reset()
obs, reward, done, info = env.step(action)
```

## Unified interface

All environments (DROID, RoboCasa, mock) share the same
observation and action conventions:

**Action** — either a flat float vector (legacy / planner DROID-style)
or a `split_7d_action` components dict (RoboCasa). The flat-vector
dimensionality depends on the robot and action space:

| Environment | Action space | DoF | Layout |
|---|---|---|---|
| DROID | `cartesian_velocity` / `cartesian_position` | **7** | `[dx, dy, dz, droll, dpitch, dyaw, d_gripper]` (Euler XYZ) |
| DROID | `joint_velocity` / `joint_position` | **8** | `[dj0, dj1, …, dj6, d_gripper]` |
| RoboCasa | Canonical PandaOmron layout A | **12 × chunk_size** | `[base_motion(4), control_mode, ee_pos(3), ee_rot_axis_angle(3), gripper]` per OSC sub-step (in `[-1, 1]`) |
| Mock | any | **7** | `[dx, dy, dz, droll, dpitch, dyaw, d_gripper]` (Euler XYZ) |

The last element of the legacy 7-D layout is always the gripper command;
translation deltas are in metres and rotation deltas in radians (Euler
XYZ in slots `3:6`). For a planner that wants to share code across
embodiments, `action_components.merge_components(components, embodiment)`
packs a metric-units components dict into the right per-embodiment
flat-vector layout (axis-angle + normalisation for RoboCasa, Euler XYZ
for every legacy embodiment). See the per-robot READMEs for value
ranges, safety clamps, and the RoboCasa `--wm_embodiment` /
`--blocking_control` flags.

**Observation** — dictionary with:

| Key | Shape | Description |
|-----|-------|-------------|
| `pose` | `(7,)` float64 | EEF pose `[x, y, z, rx, ry, rz, gripper]` |
| `proprio_robocasa` | `(14,)` float64 | RoboCasa-only 14-D proprio used by the `franka_omron_state_delta` WM variant (`base_pos | base_quat | ee_pos_rel | ee_rot_rel_quat[xyz] | gripper_qpos`). |
| `joints` | `(6,)` or `(7,)` float64 | Joint angles in radians |
| `img_0`, `img_1`, … | `(H, W, 3)` uint8 | Camera images |

---

## Robot environments

### DROID — `robots/droid/`

A Franka Panda controlled via the original [DROID](https://droid-dataset.github.io/) software stack. Uses `droid.robot_env.RobotEnv` which manages the robot (locally or via a NUC `ServerInterface`), all connected ZED cameras (`MultiCameraWrapper`), and camera calibration.

Key components:
- `robot_env_server.py` — `DroidEnvWrapper` adapter + `GymEnvServer`. Configurable action space (`cartesian_velocity` / `cartesian_position` / `joint_velocity` / `joint_position`), gripper action space, and camera parameters via CLI.

The DROID README (`robots/droid/README.md`) covers DROID stack prerequisites, server arguments, camera ordering, and deployment examples.

### RoboCasa (simulator) — `robots/robocasa/`

RoboCasa kitchen simulator wrapped to match the WM training-time
canonical PandaOmron action layout. The env runs the underlying
robosuite OSC controller and exposes both the legacy 7-D `pose` proprio
and a 14-D `proprio_robocasa` proprio for the state-delta WM variant.

Key components:
- `robot_env.py` — `RoboCasaEnv`. `chunk_size`-broadcast canonical 12-D action; `wm_embodiment` parameter selects between `franka_omron` and `franka_omron_state_delta`.
- `blocking_robot_env.py` — `BlockingRoboCasaEnv` wrapper that runs OSC sub-steps until the EE converges to the planner's metric Δxyz target (replaces the fixed-chunk under-reach).
- `robot_env_server.py` — `GymEnvServer` wrapping either form, with `--wm_embodiment`, `--blocking_control`, `--blocking_pos_tol`, `--blocking_max_steps` CLI flags.
- `test_env.py` — standalone tester with `--mode random_arm`, `--mode random_base`, `--mode dataset` (replays real-dataset action chunks for the best WM↔env alignment check).

The RoboCasa README (`robots/robocasa/README.md`) covers the full action
format (canonical 12-D + components dict + legacy 7-D), the `wm_embodiment`
variants, blocking control, and the available cameras / tasks.

---

## Agents

### World-model planner — `agents/world_model/`

The main agent. Uses a trained JEPA world model to plan actions via CEM (cross-entropy method) in the latent feature space.

How it works:
1. Loads a world-model checkpoint (V-JEPA 2.1 encoder + transition model, optionally an image decoder).
2. Connects to a robot environment server via `GymEnvClient`.
3. At every step, imagines multiple future rollouts of candidate action sequences through the world model, scores them by feature-space distance to a goal image, refines with CEM, and executes the best first action on the robot (closed-loop MPC).

The planner is compute-intensive and is distributed across multiple GPU nodes via `torchrun` + SLURM. See `agents/world_model/README.md` for launch scripts, full argument reference, and example commands for single-view, multi-view, and RoboCasa (sim) configurations — including the `franka_omron_state_delta` variant that pairs with `--blocking_control` on the env server.

The planner picks the env-side action contract automatically from
`--embodiment_id`: it sends per-step **deltas** to RoboCasa
(`franka_omron` / `franka_omron_state_delta`) and per-step **absolute
poses** to the legacy embodiments (DROID, mock).

Key files:
- `policy_client.py` — main entry point; loads the model, runs the MPC loop, records videos.
- `tasks_droid/` — directory of task JSON files (goal images + metadata) for various manipulation tasks (grasp, lift, shelf placement, etc.).

---

## Testing without hardware

`mock_robot.py` and `mock_agent.py` let you test the full agent ↔ environment loop without any physical robot:

```bash
# Terminal 1: start a mock robot server
python mock_robot.py --port 4754

# Terminal 2: run the mock agent against it
python mock_agent.py --port 4754 --mode choreography
```

The mock robot returns synthetic observations (random images, a pose that drifts with applied actions) using the same observation schema as the real environments. The mock agent supports two modes:
- **random** — sends random delta-pose actions.
- **choreography** — lifts the end-effector 4 cm, then traces a horizontal circle. Useful for verifying that actions and observations are serialized correctly.

---

## Writing a new agent

```python
from gym_tcp import GymEnvClient

env = GymEnvClient(host="localhost", port=4754)
obs, _ = env.reset()

for _ in range(100):
    # obs["pose"]   -> (7,) current EEF pose
    # obs["img_0"]  -> (H, W, 3) camera image
    # obs["joints"] -> joint angles in radians

    action = your_policy(obs)  # shape (7,): [dx, dy, dz, drx, dry, drz, d_gripper]
    obs, reward, done, info = env.step(action)
    if done:
        obs, _ = env.reset()

env.close()
```

Start with `mock_robot.py` to validate your agent logic, then point it at a real robot server by changing the host/port.

## Writing a new robot environment

Implement a class with `reset(**kwargs)` and `step(action)` methods following the observation/action conventions above, then serve it:

```python
from gym_tcp import GymEnvServer

env = YourRobotEnv()
server = GymEnvServer(env, port=4754, host="0.0.0.0")
server.start()
```

All existing agents will work with the new environment without modification.
