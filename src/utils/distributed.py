# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.utils.logging import get_logger

logger = get_logger()


def init_distributed(port=37249, rank_and_world_size=(None, None), timeout=None):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ["MASTER_ADDR"] = "localhost"

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        except Exception:
            logger.info("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank, timeout=timeout)
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"Rank: {rank}. Distributed training not available {e}")

    if "SLURM_JOB_ID" in os.environ:
        # Use the job's node-local scratch dir (if the cluster provides one) instead of /tmp
        tmpdir = Path(os.environ.get("SLURM_TMPDIR", "/path/to/slurm_tmpdir"))
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    return world_size, rank
