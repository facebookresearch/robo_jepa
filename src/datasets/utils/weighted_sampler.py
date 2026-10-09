# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, Optional

import numpy as np
from torch.utils.data import DistributedSampler

from src.utils.logging import get_logger

logger = get_logger("WeightedSampler")


class MemoryEfficientDistributedWeightedSampler(DistributedSampler):
    """
    This class implements a memory efficient version of `DistributedWeightedSampler`.
    It shares the same interface as `DistributedWeightedSampler`.
    The effective change is just-in-time sampling of the indices, instead of pre-computing them.
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        logger.info(f"Using MemoryEfficientDistributedWeightedSampler with rank {rank} / {num_replicas}")
        assert hasattr(
            dataset, "dataset_weights"
        ), "Dataset must have dataset_weights property for using MemoryEfficientDistributedWeightedSampler"
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )

        self.dataset_weights = dataset.dataset_weights
        self.dataset_sizes = [len(d) for d in dataset.datasets]
        if len(self.dataset_sizes) != len(self.dataset_weights):
            raise ValueError(
                f"Number of datasets ({len(self.dataset_sizes)}) "
                f"does not match number of dataset weights ({len(self.dataset_weights)})"
            )

        if self.shuffle:
            self.rng = np.random.default_rng(self.seed + self.rank * 10000 + self.epoch)
            total_weights = sum(self.dataset_weights)
            self.dataset_probablities = np.array([w / total_weights for w in self.dataset_weights])
        else:
            if any([not isinstance(w, int) for w in self.dataset_weights]):
                raise ValueError("Dataset weights must be integers when shuffle is False")

            self.dataset_orders = []
            for i, w in enumerate(self.dataset_weights):
                self.dataset_orders.extend([i] * w)

            self.drawn_samples = 0

    def __iter__(self) -> Iterator:
        return self

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.shuffle:
            self.rng = np.random.default_rng(self.seed + self.rank * 1000 + self.epoch)

    def __next__(self) -> int:
        if self.shuffle:
            selected_dataset_idx = self.rng.choice(range(len(self.dataset_weights)), p=self.dataset_probablities)

            # In order to avoid sampling the same example multiple times between the ranks,
            # we limit each rank to a subset of the total number of samples in the dataset.
            # For example if our dataet is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], and we have 2 ranks,
            # then rank 0 will ONLY sample from [0, 2, 4, 6, 8], and rank 1 from [1, 3, 5, 7, 9].
            # In each iteration we first produce `in_rank_sample` which is the sample index in the rank,
            # based on the size of the subset which that rank can sample from.
            # Then we computer `sample_idx_in_dataset` for the indx of the sample in the whole dataset.
            # For the above example if we are sampling for rank 1, we have `self.rng.integers(5)`.
            # Let's assume the result is 2, then `in_rank_sample` is 2 (number "5" in the subset),
            # so the sample index in the whole dataset is
            # `in_rank_sample * self.num_replicas + self.rank`: 2 * 2 + 1 = 5.

            selected_dataset_size = self.dataset_sizes[selected_dataset_idx]
            # 1) Getting sample index in the rank.
            # NOTE: this may effectively drops the last batch,
            # but given the sample sizes that we use this sampler with, it should not be an issue.
            num_samples_in_rank = selected_dataset_size // self.num_replicas
            in_rank_sample = self.rng.integers(num_samples_in_rank)

            # 2) Getting sample index in the dataset.
            sample_idx_in_dataset = in_rank_sample * self.num_replicas + self.rank

        else:
            # Iterate through the dataset orders in a round-robin fashion, offset by the rank
            dataset_orders_idx = (self.rank + self.drawn_samples) % len(self.dataset_orders)
            selected_dataset_idx = self.dataset_orders[dataset_orders_idx]
            # Get the sample index in the selected dataset by skipping with the num_replicas * drawn_samples
            sample_idx_in_dataset = (self.drawn_samples * self.num_replicas + self.rank) % self.dataset_sizes[
                selected_dataset_idx
            ]
            self.drawn_samples += 1

        # Getting the index of the sample in the whole dataset
        # For example if the total dataset has 4 datasets with sizes [10, 20, 30, 5].
        # and our selected_dataset_idx=3 and sample_idx_in_dataset=5
        # then the index of the sample in the whole dataset is
        #   10 (for dataset 1) + 20 (for dataset 1) + 5 (for sample_idx_in_dataset) = 35
        # This is because the first 10 samples are from the first dataset, the next 20 are from the second dataset,
        # then we reach at the 3rd dataset which is the selected dataset, and the 5th sample in the 3rd dataset.
        sample_idx = 0
        for i, d in enumerate(self.dataset_sizes):
            if selected_dataset_idx == i:
                break
            sample_idx += d
        sample_idx += sample_idx_in_dataset

        return sample_idx


def safe_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None
