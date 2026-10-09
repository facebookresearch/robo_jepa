# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.utils.logging import get_logger

logger = get_logger("Datasets utils")


def dataset_paths():
    # Robot datasets are located via `data.s3_prefix` in the config, so no named dataset paths are registered.
    return {}


def get_dataset_paths(datasets: list):
    paths = []
    for d in datasets:
        try:
            path = dataset_paths().get(d)
        except Exception:
            raise Exception(f"Unknown dataset: {d}")
        paths.append(path)
    logger.info(f"Datapaths {paths}")
    return paths
