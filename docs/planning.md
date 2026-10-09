# Planning

## Architecture

Robots and agents are separate processes connected by a Gym-style TCP interface (`scripts/deployment/gym_tcp.py`). An
**env server** wraps a real robot or the simulator and serves `reset()` / `step()`. An **agent** connects to it as a
client from any machine with network access. The world-model agent is
`scripts/deployment/agents/world_model/policy_client.py`.

```
policy_client.py (GPU)                         robot_env_server.py
  encode goal image + current frames  <-- TCP -->  real robot (DROID) or RoboCasa
  CEM over action sequences in latent space
  send best first action                          execute, return new observation
```

To test the transport without hardware, run `python mock_robot.py --port 4754` and `python mock_agent.py --port 4754`
in `scripts/deployment/`.

## The planner

At each step, `policy_client.py`:

1. encodes the goal image and the last `--closed_loop_context_len` frames;
2. runs CEM: it samples `--samples x --num_batches` action sequences of length `--rollout` (the horizon), rolls each
   one through the predictor, and scores the predicted features by per-token L1 distance to the goal features
   (weighted across views with `--view_weights`). It refits a Gaussian to the `--cutoff` best sequences, running
   `--cem` iterations on the first step and `--step_cem` iterations after that, warm-started with the momentum terms;
3. executes the first `--execution_prefix` actions and re-plans.

Translation samples are bounded by `--maxnorm`. `--translation_only` turns rotation off, and `--use_gripper` adds a
gripper search (`--gripper_init_std`, `--maxgripper`). The sample budget can be split across GPUs: launch
`policy_client.py` with `srun` + `torchrun`, one process per GPU (see Real robots below).

## Checkpoint layout

`WorldModel.load_pretrained(path)` and `policy_client.py --checkpoint_path` expect a directory:

```
<model>/
  config.json            # predictor config: model.transition_model.config of the training YAML, plus encoder_version
  world_model.pth.tar    # predictor weights (for another file name, pass --weights-filename)
  encoder.pth.tar        # V-JEPA encoder weights (a symlink to vjepa2_1_vitG_384.pt works)
  decoder/               # optional, only for --cem_imagination_viz: config.json + wm_decoder.pth.tar
  compiled/              # optional AOT programs, see app/robo_jepa/inference/README.md
```

For the released checkpoints (see the README's model table), copy the size's `config.json` from
`configs/released/` into one directory with the downloaded weights, and add the encoder next to them.

**Compiled inference.** `python -m app.robo_jepa.inference.compile_checkpoint <model dir> --num-gpus 4` compiles
inference programs into `<model dir>/compiled/`, which run 2-12x faster than eager. Load them with
`WorldModel.load_pretrained(path, load_compiled=True)` or `policy_client.py --load_compiled bf16`.

## RoboCasa tasks

Four custom tasks on RoboCasa365 kitchens with the PandaOmron robot are defined in `scripts/deployment/robots/robocasa/`:

| Task | Goal | Steps per episode (paper) | Metric |
|---|---|---|---|
| Reach | move the end effector to a goal pose | 5 | normalised cumulative reward |
| ObjectReach | carry a pre-grasped object (banana, can, mug) to a goal pose | 5 | normalised cumulative reward |
| ObstacleReach | reach around an appliance placed between start and goal | 18 | final reward |
| ObjectPush | push a flour bag to a goal position by contact | 18 | final reward |

The per-step reward is `r_t = max(0, 1 - ||x_t - x_goal|| / d0)`, where `x` is the end effector (Reach,
ObstacleReach) or the object (ObjectReach, ObjectPush).

**Action space.** Each planner action is an end-effector translation target. With `--blocking_control`, the env server
drives the arm to it (1 mm tolerance, at most 200 sim steps) before returning the next observation. The server runs
with `--wm_embodiment franka_omron_state_delta`. On the client, ObstacleReach and ObjectPush plan with the state-delta
embodiment (`--embodiment_id 3`); Reach and ObjectReach default to `franka_omron` (`--embodiment_id 2`), and
`EMBODIMENT=3` switches them to state delta.

## Reproduction scripts

Each script plans one episode per task seed, on one GPU, and starts its own env server:

```bash
export REPO=$PWD CONDA_SH=<conda.sh> CONDA=<env> MODELS=<dir with wm-scaling-<size>-norms/> OUT=<results dir>
bash scripts/deployment/examples/reproduce_reach.sh          50m 120k 0 1 2 3 4    # <size> <step> <seeds...>
bash scripts/deployment/examples/reproduce_object_reach.sh   50m 120k 0 1 2 3 4
bash scripts/deployment/examples/reproduce_obstacle_reach.sh 50m 120k 0 1 2 3 4
bash scripts/deployment/examples/reproduce_push.sh           50m 120k 0 1 2 3 4
```

Each episode writes `<OUT>/<task>_repro/.../ep_ts<seed>_0.{npz,mp4}`. The `.npz` holds the per-step env rewards and
the env's per-step info (distance to goal, `d0`, success). `PORT` and `GPU` override the server port and the GPU;
`EMBODIMENT` overrides the WM embodiment for Reach and ObjectReach.

**Cost.** With `--samples 16 --num_batches 64` (1024 candidates, run as 64 sequential batches), one CEM iteration of
the eager 50M model takes about 20 s at horizon 1 and 40-60 s at horizon 5 on an H100. That is about 20 min per Reach
episode and 3 h per ObjectPush episode, so split the seeds across GPUs or use a compiled model.

**Differences from the paper.** Table 14 of the paper lists the full RoboCasa configuration. The shipped scripts
differ from it:
- ObstacleReach runs 5 steps instead of 18;
- ObjectReach does not search the gripper (no `--use_gripper`);
- Reach and ObjectReach use a pretraining kitchen instead of the held-out target kitchens;
- the start-goal distance ranges differ.

Check these before comparing numbers.

## Rendering on MIG slices

MIG slices have no graphics support, so on them the env server has to render with Mesa's software EGL device. Set
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json` and `MUJOCO_EGL_DEVICE_ID=<index of the
software device>`. robosuite asserts that the device index appears in `CUDA_VISIBLE_DEVICES`, so set e.g. `GPU=0,<index>`
for the reproduction scripts; CUDA stops at the first invalid index and still sees only the slice.

## Real robots

The DROID env server (Franka via the DROID stack) is in `scripts/deployment/robots/droid/`, with a README for the
hardware setup. Real-robot planning runs multi-node CEM:
inside a SLURM allocation, from `scripts/deployment/agents/world_model/`,

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    policy_client.py [ARGS...]
```

See `scripts/deployment/README.md` and
`scripts/deployment/agents/world_model/README.md` for the arguments.
