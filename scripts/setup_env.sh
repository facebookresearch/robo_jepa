#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Create the robo_jepa conda env on the Alliance "Fir" cluster from environment.yml.
#
#   bash scripts/setup_env.sh
#   ROBOCASA_ASSETS=1 bash scripts/setup_env.sh     # also download the RoboCasa365 kitchen assets (~23 GB)
#
# One env covers training, eval and planning, including the RoboCasa365 simulator (robocasa/robosuite).
#
# The env and every package/download cache live on $SCRATCH ($HOME quota is small).
# The Compute Canada login environment is neutralised: PIP_CONFIG_FILE is blanked so pip
# ignores the CC wheelhouse ("+computecanada" wheels linked against CVMFS libs), PYTHONPATH is
# unset because CC injects site customisations that hide manylinux wheels, and the user site
# (~/.local, which has a stray torch) is disabled.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$SCRATCH/miniforge3}"
ENV_PREFIX="${ENV_PREFIX:-$CONDA_ROOT/envs/robo_jepa}"
CACHE="${CACHE:-$SCRATCH/cache}"
# cosmos-tokenizer hard-pins einops/mediapy/einx/ipdb, so it goes in with --no-deps after the env
# (its runtime deps are pinned in environment.yml).
COSMOS_TOKENIZER="git+https://github.com/NVIDIA/Cosmos-Tokenizer.git@3584ae752ce8ebdbe06a420bf60d7513c0e878cc"

export CONDA_PKGS_DIRS="$CACHE/conda_pkgs"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export PIP_CONFIG_FILE=/dev/null
unset PYTHONPATH
export PYTHONNOUSERSITE=1
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    tmp=$(mktemp -d)
    curl -sSL -o "$tmp/miniforge.sh" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$tmp/miniforge.sh" -b -p "$CONDA_ROOT"
    rm -rf "$tmp"
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
if [[ -d "$ENV_PREFIX" ]]; then
    conda env update -p "$ENV_PREFIX" -f "$REPO/environment.yml"
else
    conda env create -p "$ENV_PREFIX" -f "$REPO/environment.yml"
fi
conda activate "$ENV_PREFIX"
pip install --no-deps "$COSMOS_TOKENIZER"

# RoboCasa one-time setup. Run from a neutral dir: a `robocasa/` folder in the cwd would shadow the package.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
(
    cd "$CACHE"
    echo n | python -m robocasa.scripts.setup_macros >/dev/null   # "n" keeps an existing macros_private.py
    echo n | python -m robosuite.scripts.setup_macros >/dev/null
    if [[ "${ROBOCASA_ASSETS:-0}" == 1 ]]; then
        echo y | python -m robocasa.scripts.download_kitchen_assets
    fi
)

python -c "import torch, xformers, decord, cosmos_tokenizer; print('torch', torch.__version__, 'xformers', xformers.__version__, 'decord', decord.__version__)"
(cd "$CACHE" && python -c "import robocasa, robosuite, mujoco; print('robocasa ok, mujoco', mujoco.__version__)")
