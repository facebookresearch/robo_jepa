#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=reach_inv_golden
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --mem=480G
#SBATCH --output=slurm_logs/%j_reach_inv_golden.out
# ---------------------------------------------------------------------------
# Generate the reach-invariants golden file (and per-(scene, episode, mode)
# viz PNGs) by fanning the per-scene work out over `WORKERS` (default 20)
# concurrent xargs workers on a single H200 node.
#
# Supports two kitchen splits:
#   pretrain (default) — 49 pretraining seeds ({0..49}\{6}). Writes to
#                         scripts/deployment/robots/robocasa/reach_invariants_golden.json
#                         (legacy path; what the existing test reads).
#   target             — 10 held-out target kitchens, seeds 0..9 (each
#                         maps to a distinct (layout, style) pair via
#                         `seed % 10`). Writes to
#                         scripts/deployment/robots/robocasa/reach_invariants_golden_target.json.
#
# Wall-clock budget on this node:
#   pretrain ~6-10 min (49 / 20 ≈ 2.5 scenes / worker @ ~3 min/scene)
#   target   ~3-6  min (10 / 20 ≤ 1   scene  / worker)
#
# After all workers finish, the merge step consolidates every sc*.json
# under the partials dir into the final golden file.
#
# Usage (from the repo root, with slurm_logs/ created and the robo_jepa env active):
#   sbatch scripts/deployment/robots/robocasa/launch_reach_invariants_golden.sh
#   sbatch --export=ALL,KITCHEN_SPLIT=target scripts/deployment/robots/robocasa/launch_reach_invariants_golden.sh
#
# Optional env vars:
#   KITCHEN_SPLIT=pretrain|target   # default: pretrain
#   SEEDS="0 1 41 35"               # override the seed set
#   WORKERS=20                      # parallel worker count
#   NO_VIZ=1                        # skip image saving (faster, no PNGs)
# ---------------------------------------------------------------------------

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root

KITCHEN_SPLIT="${KITCHEN_SPLIT:-pretrain}"

# Per-split output paths. Pretrain keeps the legacy
# ``reach_invariants_golden.json`` filename (already committed); other
# splits write alongside with ``_<split>`` suffix.
    if [[ "${KITCHEN_SPLIT}" == "pretrain" ]]; then
    PARTIALS_DIR="${REPO_ROOT}/scripts/deployment/robots/robocasa/reach_invariants_golden_partials"
    VIZ_DIR="${REPO_ROOT}/scripts/deployment/robots/robocasa/reach_invariants_golden_viz"
    OUTPUT="${REPO_ROOT}/scripts/deployment/robots/robocasa/reach_invariants_golden.json"
else
    PARTIALS_DIR="${REPO_ROOT}/scripts/deployment/robots/robocasa/reach_invariants_golden_partials_${KITCHEN_SPLIT}"
    VIZ_DIR="${REPO_ROOT}/scripts/deployment/robots/robocasa/reach_invariants_golden_viz_${KITCHEN_SPLIT}"
    OUTPUT="${REPO_ROOT}/scripts/deployment/robots/robocasa/reach_invariants_golden_${KITCHEN_SPLIT}.json"
fi
SCRIPT="${REPO_ROOT}/scripts/deployment/robots/robocasa/generate_reach_invariants_golden.py"
WORKER_LOG_DIR="${REPO_ROOT}/slurm_logs/reach_inv_golden_${KITCHEN_SPLIT}_${SLURM_JOB_ID:-noslurm}"

# Per-split default seed sets. Pretrain: {0..49}\{6} (49 seeds matching
# the planner's INCLUDED_SEEDS). Target: 0..9 (one per target kitchen,
# via ``seed % 10`` inside RoboCasaEnv._create_env).
if [[ "${KITCHEN_SPLIT}" == "pretrain" ]]; then
    DEFAULT_SEEDS=(0 1 2 3 4 5 7 8 9 10 11 12 13 14 15 16 17 18 19 \
                   20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 \
                   40 41 42 43 44 45 46 47 48 49)
else
    DEFAULT_SEEDS=(0 1 2 3 4 5 6 7 8 9)
fi

if [[ -n "${SEEDS:-}" ]]; then
    read -r -a SEED_ARR <<< "${SEEDS}"
else
    SEED_ARR=("${DEFAULT_SEEDS[@]}")
fi
WORKERS="${WORKERS:-20}"

if [[ "${NO_VIZ:-0}" == "1" ]]; then
    VIZ_ARGS=""
else
    VIZ_ARGS="--viz-dir ${VIZ_DIR}"
fi

cd "${REPO_ROOT}"
mkdir -p "${PARTIALS_DIR}" "${WORKER_LOG_DIR}" "$(dirname "${OUTPUT}")"
if [[ "${NO_VIZ:-0}" != "1" ]]; then
    mkdir -p "${VIZ_DIR}"
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/scripts/deployment:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "[launcher] node=$(hostname) cpus=$(nproc) workers=${WORKERS}"
echo "[launcher] kitchen_split=${KITCHEN_SPLIT}"
echo "[launcher] gpus visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[launcher] processing ${#SEED_ARR[@]} seeds: ${SEED_ARR[*]}"
echo "[launcher] partials -> ${PARTIALS_DIR}"
echo "[launcher] viz      -> ${VIZ_DIR} (${VIZ_ARGS:-disabled})"
echo "[launcher] output   -> ${OUTPUT}"
echo "[launcher] per-worker logs -> ${WORKER_LOG_DIR}/sc<XXX>.log"
echo "[launcher] start: $(date -u +%FT%TZ)"

t_start=$(date +%s)

# Worker: process one seed end-to-end and tee its log to a per-seed file.
# We pin each worker to a single GPU (round-robin across 8 GPUs) so EGL
# context allocations are spread across all available devices instead of
# slamming GPU 0.
run_one() {
    local seed="$1"
    local gpu=$(( seed % 8 ))
    local log="${WORKER_LOG_DIR}/sc$(printf '%03d' "${seed}").log"
    {
        echo "[worker] seed=${seed} gpu=${gpu} split=${KITCHEN_SPLIT} start: $(date -u +%FT%TZ)"
        CUDA_VISIBLE_DEVICES="${gpu}" python -u "${SCRIPT}" \
            --split "${KITCHEN_SPLIT}" \
            --seeds "${seed}" \
            --partials-dir "${PARTIALS_DIR}" \
            ${VIZ_ARGS}
        echo "[worker] seed=${seed} end: $(date -u +%FT%TZ)"
    } > "${log}" 2>&1
    echo "[launcher] sc$(printf '%03d' "${seed}") DONE -> ${log}"
}

export -f run_one
export SCRIPT PARTIALS_DIR VIZ_ARGS WORKER_LOG_DIR KITCHEN_SPLIT

# Fan out: stream seeds into xargs -P ${WORKERS} -n 1.
printf '%s\n' "${SEED_ARR[@]}" | \
    xargs -P "${WORKERS}" -I{} bash -c 'run_one "$@"' _ {}

t_end=$(date +%s)
echo "[launcher] all ${#SEED_ARR[@]} workers DONE in $(( (t_end - t_start) / 60 ))m $(( (t_end - t_start) % 60 ))s"

# Sanity-check: every seed has a partial.
missing=()
for s in "${SEED_ARR[@]}"; do
    f="${PARTIALS_DIR}/sc$(printf '%03d' "${s}").json"
    if [[ ! -f "${f}" ]]; then
        missing+=("${s}")
    fi
done
if (( ${#missing[@]} > 0 )); then
    echo "[launcher] ERROR: ${#missing[@]} seeds have no partial JSON: ${missing[*]}" >&2
    echo "[launcher] check ${WORKER_LOG_DIR}/sc<XXX>.log for the failure tracebacks" >&2
    exit 3
fi

# Merge -> final golden.
echo "[launcher] merging ${#SEED_ARR[@]} partials -> ${OUTPUT}"
python -u "${SCRIPT}" --merge --partials-dir "${PARTIALS_DIR}" --output "${OUTPUT}"
echo "[launcher] DONE wall=$(( (t_end - t_start) / 60 ))m: ${OUTPUT}"
