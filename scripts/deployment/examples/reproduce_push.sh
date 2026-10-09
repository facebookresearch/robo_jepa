#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Reproduce one OBJECT_PUSH planning run: ONE env, ONE GPU.
#   Usage: ./reproduce_push.sh <size> <step> <seed> [seed ...]   (env: SCENE=4 kitchen)
# Shared planner hyperparameters (identical across all 4 example tasks); only
# ROLLOUT / EMBODIMENT / TOTAL_STEPS + the task-server flags differ per task.
# PUSH: rollout=5, embodiment_id=3, total_steps=18.
# Global CEM budget 1024 = samples(16) * world_size(1) * num_batches(64).
set -e
SIZE=$1; STEP=$2; shift 2; SEEDS="$@"
SCENE=${SCENE:-4}                    # target kitchen id (env); task seed varies per episode
ROLLOUT=5; EMBODIMENT=3; TOTAL_STEPS=18
OBJECT=${OBJECT:-flourbag001}        # brown flour bag
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}   # repo root
MODELS=${MODELS:-/path/to/models}   # holds wm-scaling-<size>-norms/{config.json,cooldown_<size>_<step>.pth.tar}
CONDA=${CONDA:-robo_jepa}   # conda env (prefix or name)
CONDA_SH=${CONDA_SH:-/path/to/conda/etc/profile.d/conda.sh}
OUT=${OUT:-$REPO/results}                                          # results root
PORT=${PORT:-7300}                                                # env server port (unique per node)
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=${GPU:-0}   # GPU: device list for planner + env server
export PYTHONPATH="$REPO:$REPO/scripts/deployment:${PYTHONPATH:-}"
# policy_client always initialises torch.distributed: a 1-process group on this GPU
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=${MASTER_PORT:-29531}
source "$CONDA_SH" && conda activate "$CONDA"

# --- stage checkpoint ---
SRC=$MODELS/wm-scaling-$SIZE-norms
CKPT=/tmp/push_ckpt_${SIZE}_${STEP}; mkdir -p "$CKPT"
ln -sf "$SRC/cooldown_${SIZE}_${STEP}.pth.tar" "$CKPT/world_model.pth.tar"
ln -sf "$SRC/config.json" "$CKPT/config.json"
[ -e "$SRC/encoder.pth.tar" ] && ln -sf "$SRC/encoder.pth.tar" "$CKPT/encoder.pth.tar"

# --- one task JSON per env seed (start/goal pair); kitchen pinned to SCENE ---
TASKS=/tmp/push_tasks_$$; mkdir -p "$TASKS"
for s in $SEEDS; do
  printf '{"init":{"mode":"init","object_push_task_seed":%d},"goal":{"mode":"goal","object_push_task_seed":%d}}' "$s" "$s" > "$TASKS/ts$s.json"
done

# --- env server (GPU 0): forward-cone push of the pinned brown bag ---
cd "$REPO"
python scripts/deployment/robots/robocasa/robot_env_server.py \
  --port "$PORT" --camera_names robot0_agentview_left robot0_eye_in_hand \
  --camera_width 320 --camera_height 240 --robot PandaOmron --chunk_size 4 \
  --wm_embodiment franka_omron_state_delta --seed 0 --blocking_control \
  --blocking_pos_tol 0.001 --blocking_max_steps 200 \
  --object_push_task --object_push_target_kitchen_id "$SCENE" \
  --object_push_object_name "$OBJECT" --object_push_forward_cone_deg 60 \
  --object_push_clear_radius 1.0 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null' EXIT
sleep 45

# --- planner: single GPU, num_batches=64 -> global CEM 1024 ---
cd "$REPO/scripts/deployment/agents/world_model"
python policy_client.py \
  --checkpoint_path "$CKPT" --host localhost --port "$PORT" \
  --embed_dim 1664 --image_size 240 320 --use_cameras 0 1 --camera_id 0 2 \
  --embodiment_id $EMBODIMENT --task_path "$TASKS" --episodes 1 --init_ep 0 \
  --total_steps $TOTAL_STEPS --rollout $ROLLOUT --cem 15 --step_cem 10 --cutoff 20 \
  --samples 16 --num_batches 64 --maxnorm 0.3 --maxgripper 0.99 \
  --robot_init_std 0.3 --gripper_init_std 0.8 \
  --momentum_std 0.70 --momentum_mean_gripper 0.15 --momentum_std_gripper 0.45 \
  --execution_prefix 1 --closed_loop_context_len 2 --gamma 0.99 \
  --translation_only --use_cache --load_compiled false --action_substeps 1 \
  --bypass_input --view_weights 0.5 0.5 \
  --output_filename "$OUT/push_repro/${SIZE}_${STEP}_k${SCENE}/ep"
