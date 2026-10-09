# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-dataset initialisation for robot video training.

Provides :class:`MultiDataset` (a thin wrapper that concatenates multiple
:class:`RobotVideoDataset` instances into a single indexable dataset) and
:func:`init_datasets` (a factory that builds the combined dataset, weighted
sampler, and data-loader in one call).

Typical usage::

    from app.robo_jepa.datasets.init_data import init_datasets
    from app.robo_jepa.transforms import make_transforms

    loader, sampler = init_datasets(
        data_paths=[None, None],
        dataset_names=['Droid', 'Roboset'],
        dataset_weights=[0.6, 0.4],
        transform=make_transforms(crop_height=256, crop_width=256),
        common_kwargs=dict(
            s3_profile='<aws_profile>',
            s3_prefix='s3://<bucket>/data/robotics/',
            pad_size=35,
            frames_per_clip=16,
        ),
        dataset_kwargs={
            'Droid':   dict(embodiment_name='franka', s3_prefix='droid/1.0.1/',
                            raw_fps=60, fps=5, pose_dim=6, ...),
            'Roboset': dict(embodiment_name='franka', s3_prefix='roboset/',
                            raw_fps=5, fps=5, pose_dim=6, ...),
        },
        batch_size=4,
        num_workers=10,
    )

``common_kwargs`` are shared across all datasets. ``dataset_kwargs`` override
or extend them per-dataset. The per-dataset ``s3_prefix`` is joined with the
common ``s3_prefix`` (i.e. ``os.path.join(common, per_dataset)``).
"""

import os
from copy import deepcopy as copy
from logging import getLogger

import numpy as np
import torch
import torch.utils.data

from app.robo_jepa.datasets.robot_video_dataset import RobotVideoDataset
from src.datasets.utils.weighted_sampler import MemoryEfficientDistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


class MultiDataset(torch.utils.data.Dataset):
    """Concatenation of multiple :class:`RobotVideoDataset` instances.

    Indexes are contiguous: if dataset A has 100 samples and B has 200,
    indices 0–99 map to A and 100–299 map to B.  Use together with
    :class:`MemoryEfficientDistributedWeightedSampler` for weighted sampling.

    Args:
        *datasets: Variable number of :class:`RobotVideoDataset` instances.
        dataset_weights (list[float], optional): Sampling weights per dataset.
            Normalised internally to sum to 1. If None, weights are
            proportional to dataset length.
    """

    def __init__(self, *datasets, dataset_weights=None):
        self.datasets = datasets
        if dataset_weights is None:
            dataset_weights = [len(d) for d in datasets]
        total = sum(dataset_weights)
        dataset_weights = [w / total for w in dataset_weights]
        self.dataset_weights = dataset_weights
        self.dataset_sizes = np.array([len(d) for d in datasets])
        self.dataset_start_ids = np.cumsum(self.dataset_sizes) - self.dataset_sizes
        self.total = sum(len(d) for d in datasets)

    def __getitem__(self, index):
        dataset_mask = self.dataset_start_ids <= index
        # find the last maximum in dataset_mask i.e. the last dataset start_id that is lower than our index
        dataset_bin = len(dataset_mask) - 1 - np.argmax(dataset_mask[::-1])
        return self.datasets[dataset_bin][index - self.dataset_start_ids[dataset_bin]]

    def __len__(self):
        return self.total


def init_datasets(
    data_paths,
    dataset_names,
    dataset_weights,
    common_kwargs,
    dataset_kwargs,
    rank=0,
    world_size=1,
    batch_size=1,
    collator=None,
    transform=None,
    drop_last=False,
    num_workers=0,
    pin_mem=True,
    shuffle=True,
    persistent_workers=True,
    num_replicas_override=None,
    rank_override=None,
):
    """Build a multi-dataset data-loader with weighted distributed sampling.

    Creates one :class:`RobotVideoDataset` per entry in *dataset_names*,
    wraps them in a :class:`MultiDataset`, attaches a
    :class:`MemoryEfficientDistributedWeightedSampler`, and returns the
    resulting :class:`torch.utils.data.DataLoader`.

    Per-dataset ``s3_prefix`` values are **joined** with the common one::

        effective_s3_prefix = os.path.join(common_kwargs['s3_prefix'],
                                           dataset_kwargs[name]['s3_prefix'])

    Args:
        data_paths (list): One entry per dataset (legacy, pass None for each).
        dataset_names (list[str]): Dataset identifiers matching keys in
            *dataset_kwargs*.
        dataset_weights (list[float]): Sampling weights (unnormalised).
        common_kwargs (dict): Keyword arguments shared across all datasets
            (e.g. ``s3_profile``, ``pad_size``, ``frames_per_clip``).
        dataset_kwargs (dict[str, dict]): Per-dataset keyword arguments
            keyed by dataset name. Overrides/extends *common_kwargs*.
        rank (int): Current process rank for distributed training.
        world_size (int): Total number of processes.
        batch_size (int): Batch size per rank.
        collator (callable, optional): Custom collate function.
        transform (callable, optional): Video transform passed to every
            dataset (see :func:`transforms.make_transforms`).
        drop_last (bool): Drop the last incomplete batch.
        num_workers (int): DataLoader worker processes.
        pin_mem (bool): Pin memory for GPU transfer.
        shuffle (bool): Enable random sampling.
        persistent_workers (bool): Keep workers alive between epochs.

    Returns:
        tuple: ``(data_loader, dist_sampler)``.
    """
    datasets = []
    for data_path, dataset_name in zip(data_paths, dataset_names):
        current_kwargs = copy(common_kwargs)
        new_kwargs = copy(dataset_kwargs[dataset_name])
        print("current_kwargs s3 prefix:", current_kwargs.get("s3_prefix"))
        print("new_kwargs s3 prefix:", new_kwargs.get("s3_prefix"))
        new_kwargs["s3_prefix"] = os.path.join(current_kwargs["s3_prefix"], new_kwargs["s3_prefix"])
        current_kwargs.update(new_kwargs)
        dataset = RobotVideoDataset(
            dataset_name=dataset_name,
            data_path=data_path,
            transform=transform,
            **current_kwargs,
        )
        datasets.append(dataset)
    datasets = MultiDataset(*datasets, dataset_weights=dataset_weights)
    # Per-group sharding: when num_replicas_override / rank_override are given
    # (e.g. for view-group routing across DDP ranks), the sampler shuffles the
    # dataset only within the group so that each group sees an independent
    # shuffling/weighting universe even though DDP itself remains a single
    # global process group.
    sampler_num_replicas = num_replicas_override if num_replicas_override is not None else world_size
    sampler_rank = rank_override if rank_override is not None else rank
    dist_sampler = MemoryEfficientDistributedWeightedSampler(
        datasets, num_replicas=sampler_num_replicas, rank=sampler_rank, shuffle=shuffle
    )
    # dist_sampler = DistributedSampler(
    #     datasets, rank=rank, num_replicas=world_size, shuffle=True
    # )

    data_loader = torch.utils.data.DataLoader(
        datasets,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("VideoDataset unsupervised data loader created")
    return data_loader, dist_sampler
