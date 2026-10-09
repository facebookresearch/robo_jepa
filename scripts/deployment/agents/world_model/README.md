# World Model Planning across all supported robots

This directory contains scripts for deploying a trained JEPA world model
as a CEM-based planner. The same `policy_client.py` works against
**every** robot env server in `scripts/deployment/robots/` — DROID
(Franka via the DROID stack), the RoboCasa kitchen simulator, and the
`mock_robot.py` stub. The planner uses CEM (cross-entropy method) in the latent feature
space of the world model to find action sequences that drive the robot
toward a specified goal image.

Which env you talk to is decided by **two** things only:

* the `--port` (and optionally `--host`) you pass — this points at one
  of the `robot_env_server.py` instances under
  `scripts/deployment/robots/<embodiment>/`,
* the `--embodiment_id` you pass — this is resolved against the WM's
  `embodiment_list` to pick the env-side action contract
  (delta-into-components-dict for RoboCasa, absolute-pose for every
  other env) and the proprio key (`pose` for legacy embodiments,
  `proprio_robocasa` for the state-delta RoboCasa variant).

No per-embodiment branch in `policy_client.py` is keyed off any robot's
name — everything goes through the unified embodiment dispatch above.

## Robot env compatibility matrix

| Env | Server | Action contract sent by planner | Proprio obs key | Notes |
|---|---|---|---|---|
| DROID | `robots/droid/robot_env_server.py` | components dict OR 7-D `[x, y, z, roll, pitch, yaw, gripper]` (Euler XYZ); 8-D for joint action spaces — forwarded straight to `droid.robot_env.RobotEnv.step()`, semantics depend on `--action_space` (absolute for `cartesian_position`, normalised Δ for `cartesian_velocity`) | `pose` | action space picked by `--action_space` on the server |
| RoboCasa (`franka_omron`) | `robots/robocasa/robot_env_server.py` | components dict (preferred; metric m / rad axis-angle) OR a canonical 12-D layout-A vector — packed via `merge_components(…, "franka_omron")`, normalised by OSC `output_max=0.05 m / 0.5 rad` and clipped to `[−1, 1]` | `pose` | per-WM-step components dict is broadcast over `chunk_size` OSC sub-steps |
| RoboCasa (`franka_omron_state_delta`) | `robots/robocasa/robot_env_server.py --wm_embodiment franka_omron_state_delta --blocking_control` | components dict; rotation slots must be zero | `proprio_robocasa` (14-D) | use together with `BlockingRoboCasaEnv` so the env physically realises the metric Δxyz the planner intends |
| Mock | `mock_robot.py` | 7-D `[dx, dy, dz, droll, dpitch, dyaw, d_gripper]` | `pose` | for hardware-free smoke tests |

## Overview

`policy_client.py` is the main entry point.  It:

1. Loads a world-model checkpoint (transition model + encoder, optionally an
   image-decoding head).
2. Connects to a `robot_env_server` that streams live camera observations and
   executes actions on the robot or simulator.
3. Runs closed-loop MPC: at every environment step the planner imagines
   multiple future rollouts of candidate action sequences, scores them by
   feature-space distance to the goal, refines with CEM, and sends the best
   first action(s) to the env.

### Embodiment-conditional execution path

Every env exposes the **same** input contract — a
`split_7d_action` components dict (or, equivalently, a flat numpy
vector of the corresponding embodiment-specific shape). What differs is
how the env interprets the result of `merge_components`:

* **RoboCasa (`franka_omron` / `franka_omron_state_delta`)**: the env
  packs the components dict into a canonical 12-D layout-A vector
  (`merge_components(…, "franka_omron")`), normalises metric units by
  the OSC controller's `output_max`, clips to `[−1, 1]`, and broadcasts
  it over `chunk_size` OSC sub-steps. Each `env.step` is therefore one
  per-WM-step **delta** (see
  `scripts/deployment/robots/robocasa/README.md` § Action Format).
* **DROID**: same components-dict input; the resulting flat vector is
  forwarded straight to `droid.robot_env.RobotEnv.step()`, so the
  semantics (absolute vs normalised Δ) depend on the `--action_space`
  the server was started with.

On the planner side, `policy_client.py` currently picks between two
send strategies based on the resolved `embodiment_name`:

* **RoboCasa (`franka_omron` / `franka_omron_state_delta`)**: pack each
  per-step Δ into a `split_7d_action` components dict in metric units
  and send it through.
* **Every other embodiment** (DROID, mock): accumulate
  `commanded_pose` / `commanded_gripper` and send a flat 7-D `[x, y, z,
  roll, pitch, yaw, gripper]` *absolute* target. This restores the
  pre-refactor contract (commit `efff9504b` had silently broken it by
  sending raw deltas to every env). For DROID this matches the env's
  expectations.

  At end-of-episode the no-op terminator also follows this split: zero
  delta for RoboCasa; the current `commanded_pose` for legacy envs (so
  the robot stops in place rather than teleporting to the origin).

### `franka_omron_state_delta` WM variant

When `--embodiment_id` resolves to `franka_omron_state_delta`, the
planner additionally:

* reads the env's 14-D `obs["proprio_robocasa"]` instead of the legacy
  7-D `obs["pose"]` (so `obs_pose` has shape `(14,)` rather than `(7,)`),
* computes the WM-side action as a plain numerical diff of consecutive
  proprios (`pose_t − pose_{t-1}`) instead of the DROID-style xyz +
  Euler-rotation diff in `poses_to_diffs` (the 14-D state layout is
  incompatible with treating slots `[3:6]` as Euler).

This matches the dataset's `action_source = state_delta` regime the WM
was trained on. Use it together with `--blocking_control` on the env
server so the env physically realises the metric Δxyz the planner
intends (otherwise the OSC fixed-chunk mode under-reaches the
in-distribution scale).

### Other recent fixes

* **NCCL backend**: `init_distributed` now pins each rank's NCCL backend
  to a single explicit CUDA device. `dist.barrier()` was previously
  silently deadlocking when the device was inferred per rank.
* **`dist.broadcast` dtypes**: `goal_features` and per-step `obs_features`
  are explicitly cast to `float32` before broadcast — the encoder runs
  in bfloat16 autocast, but rank!=0 allocates float32 receive buffers,
  so a dtype mismatch was deadlocking the broadcast on multi-node runs.
  Redundant `dist.barrier()` calls around broadcasts have been removed
  (broadcast is itself a sync point).
* **AOT-compiled action preprocessor**: the bf16 / mp-bf16 dispatcher
  now calls the per-embodiment action / state preprocessor (mirrors
  `WorldModel.step`). Previously it just zero-padded the planner's
  7-D action up to the WM's 48-D RoboCasa layout, so under
  `--load_compiled bf16` the WM was being fed a wrong-layout,
  wrong-units action and barely moved.
* **MPC-vs-emb plot x-axis**: `process()` now plots `mpc_dists`
  (per-planner-step) on the shared env-step axis at
  `(k+1) * execution_prefix`, instead of `k+1`. Previous plots
  collapsed all CEM scores to the leftmost steps when
  `execution_prefix > 1`.

## Multi-node deployment with SLURM

The world model + CEM planner is compute-intensive, so the CEM samples are distributed across many GPUs: one
`torchrun` per node, launched with `srun` from inside a SLURM allocation (run from this directory, in the robo_jepa
env):

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    policy_client.py [ARGS...]
```

## Key arguments

| Argument | Description |
|---|---|
| `--checkpoint_path` | Path to the world-model directory (must contain `config.json`, `world_model.pth.tar`, `encoder.pth.tar`). |
| `--embed_dim` | Encoder embedding dimension (`1664` for ViT-G / vjepa 2.1). |
| `--image_size` | Input resolution, e.g. `256 256` or `240 320`. |
| `--aspect_ratio` | Crop aspect-ratio range, e.g. `1.0 1.333`. |
| `--use_cameras` | Which cameras to read from the **environment** (indices into the env's camera stream, e.g. `0 1` = first and second camera provided by the robot env server). |
| `--camera_id` | Camera identity tokens passed to the **world model**, following the DROID convention: `0`=left, `1`=right, `2`=wrist. These tell the model which viewpoint each observation corresponds to. |
| `--embodiment_id` | Embodiment index for multi-embodiment models. Resolved against the WM's `embodiment_list` to pick the env-side action contract (delta vs absolute) and the proprio key (`pose` vs `proprio_robocasa`). |
| `--rollout` | MPC prediction horizon (number of world-model steps to unroll). |
| `--samples` | Number of CEM action-sequence candidates per refinement iteration. |
| `--num_batches` | Multiplier for the total number of CEM samples (distributed across GPUs). |
| `--cem` | Number of initial CEM refinement iterations. |
| `--step_cem` | Number of CEM refinement iterations at each subsequent step. |
| `--cutoff` | Top-percentile of candidates kept during CEM elite selection. |
| `--total_steps` | Total environment steps to run the policy for. |
| `--episodes` | Number of episodes to run. |
| `--init_task` / `--init_ep` | Starting task and episode indices within `--task_path`. |
| `--task_path` | Path to a directory of task JSON files (DROID) **or** a single JSON file (RoboCasa) containing the goal/init spec. |
| `--use_gripper` | Enable gripper action dimension. |
| `--translation_only` | Restrict action search to 3D translation (no rotation). |
| `--maxnorm` | Maximum action norm for robot joints. |
| `--maxgripper` | Maximum action norm for the gripper. |
| `--robot_init_std` / `--gripper_init_std` | Initial standard deviation for CEM sampling. |
| `--momentum_std` / `--momentum_mean` | CEM momentum for smoothing across steps (robot). |
| `--momentum_std_gripper` / `--momentum_mean_gripper` | CEM momentum (gripper). |
| `--execution_prefix` | Number of actions from the planned sequence to execute before re-planning. |
| `--closed_loop_context_len` | Number of past observations to feed as context in closed-loop re-planning. |
| `--view_weights` | Optional per-view weighting of the CEM reward (e.g. `1.0 0.0` to score only the first view). |
| `--load_compiled` | AOT-compiled checkpoint mode: `bf16` (default), `mp-bf16`, or `false` for eager. The compiled paths now apply the per-embodiment action / state preprocessor; pre-fix runs with the bf16 path silently fed wrong-layout actions. |
| `--bypass_input` | Skip waiting for an Enter prompt before episode start (useful for non-interactive SLURM runs). |
| `--use_cache` / `--cache_path` | Enable `torch.compile` cache for faster model compilation on subsequent runs. |
| `--force_recache` | Force recompilation even if a cached compilation exists. |
| `--goal_name` / `--model_name` | Labels used in output videos and plots. |
| `--output_filename` | Prefix for output `.mp4` and `.npz` files. |
| `--port` | Port for connecting to the robot env server. |
| `--cem_imagination_viz` | After every CEM call (rank 0 only), imagine the chosen action sequence (`means[-1]`) through the WM, decode each predicted feature with the image head, and save a side-by-side mp4 (one column per view). Forces `load_decoder=True`. Output: `<output_filename>_cem_viz/{task}_ep{ep}_step{c:02d}.mp4`. |
| `--cem_imagination_denoising_steps` | Decoder denoising steps for the imagination viz (default `3`, fast). |
| `--cem_imagination_cfg_scale` | Decoder CFG scale for the imagination viz (default `1.0`). |
| `--cem_imagination_fps` | Output mp4 fps for the imagination viz (default `5`). |

## Example: real Franka (DROID platform)

The configuration used for all real-robot experiments in the paper (Grasp, Object Lift, Pick and Place). Two views,
external left camera and wrist (`--use_cameras 0 1`, WM view ids `--camera_id 0 2`), at 240×320. The goal is scored on
the left view only (`--view_weights 1.0 0.0`). The search covers 3D end-effector translation plus the gripper, with
rotation held fixed. Each replan uses a horizon of 5 world-model steps, a context of 2 observations, and executes 2
actions. The run is 20 env steps per episode. CEM does 15 iterations on the first step and 10 on later ones, keeps the
top 20 elites, and draws 4096 samples per iteration: 16 samples × 16 batches per GPU across 2 nodes × 8 GPUs. The
predictor runs AOT-compiled in bf16 (`--load_compiled bf16`, which needs the compiled programs in the checkpoint
dir, see `app/robo_jepa/inference/README.md`).

Start the robot env server (`scripts/deployment/robots/droid/`) on the robot PC, make its port reachable from the
cluster, then, inside a 2-node × 8-GPU allocation:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun --nodes=2 --ntasks-per-node=1 \
    torchrun --nnodes=2 --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    policy_client.py \
    --checkpoint_path /path/to/wm-<size> \
    --host <robot_host> --port 12345 \
    --task_path /path/to/tasks/<task> --goal_name "<task>" --model_name "RoboJEPA <size>" \
    --episodes 30 --init_task 0 --init_ep 0 --total_steps 20 --action_substeps 1 \
    --embed_dim 1664 --image_size 240 320 --use_cameras 0 1 --camera_id 0 2 --embodiment_id 0 \
    --view_weights 1.0 0.0 \
    --use_gripper --translation_only \
    --rollout 5 --closed_loop_context_len 2 --execution_prefix 2 \
    --cem 15 --step_cem 10 --cutoff 20 --samples 16 --num_batches 16 \
    --robot_init_std 0.15 --maxnorm 0.15 --momentum_mean 0.15 --momentum_std 0.70 \
    --gripper_init_std 0.8 --maxgripper 0.99 --momentum_mean_gripper 0.15 --momentum_std_gripper 0.45 \
    --load_compiled bf16 --use_cache \
    --output_filename /path/to/outputs/<size>/<task>/ep
```
