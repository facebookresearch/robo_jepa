# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch
import torch.distributed as dist
import time
import datetime

from src.utils.logging import get_logger

logger = get_logger()
import subprocess


def init_distributed(port=37219, rank_and_world_size=(None, None)):
    # time.sleep(10)
    # hst = subprocess.run(['hostname'], capture_output=True, text=True, check=True).stdout
    # nvret = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).returncode
    # _world_size, _rank = int(os.environ.get('WORLD_SIZE', -1)), int(os.environ.get('RANK', -1))
    # print(f'found {_world_size=} {_rank=} at {hst=}. nvidia-smi retcode: {nvret}. CUDA available? {torch.cuda.is_available()} CUDA device count: {torch.cuda.device_count()}')
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size

    if (rank is None) or (world_size is None):
        if "LOCAL_RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
            logger.info("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank
        else:
            world_size, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    try:
        # Pin the NCCL backend to a single, explicit CUDA device per rank.
        # Without this, dist.barrier() infers the device from the current
        # context, which can be inconsistent across ranks (especially when
        # CUDA_VISIBLE_DEVICES is mutated post-startup) and silently
        # deadlocks the barrier even though all ranks are inside it.
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES restricts us to 1 device per rank.
            except Exception:
                pass
            device_id = torch.device("cuda", 0)
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(minutes=30),
                device_id=device_id,
            )
        else:
            torch.distributed.init_process_group(
                backend="nccl", world_size=world_size, rank=rank, timeout=datetime.timedelta(minutes=30)
            )
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"Rank: {rank}. Distributed training not available {e}")

    return world_size, rank
