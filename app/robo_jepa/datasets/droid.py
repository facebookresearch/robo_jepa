# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import pandas as pd


def filter_droid(buildings, labs, instructions, samples, negative_filter=False, all_metadata=None):
    if len(instructions) != 0:
        assert (len(buildings) != 0) + (len(labs) != 0) == 0, "can only filter by instruction or by building+lab"
    if len(buildings) == 0 and len(labs) == 0 and len(instructions) == 0:
        return samples
    sample_in_lab = lambda s: any(lab in s for lab in labs)  # noqa: E731
    no_meta = 0
    if (buildings is not None and len(buildings) != 0) or (instructions is not None and len(instructions) != 0):
        filtered_samples = []
        for _, s in samples.iterrows():
            # convert_droid.py writes only `path`, which is also the metadata key
            curr_metadata = all_metadata.get(s.get("metadata", s.path))
            if curr_metadata is None:
                # filtered_samples.append(s)
                no_meta += 1
                continue
            current_instructions = {k: v for k, v in curr_metadata.items() if "language_instruction" in k}
            if buildings is not None and len(buildings) != 0:
                sample_in_building = any(curr_metadata["building"] == b for b in buildings)
                if negative_filter:
                    if not (sample_in_building and sample_in_lab(s.path)):
                        filtered_samples.append(s)
                        continue
                else:
                    if sample_in_building and sample_in_lab(s.path):
                        filtered_samples.append(s)
                        continue
            if instructions is not None and len(instructions) != 0 and current_instructions is not None:
                stop = False
                for current_instruction in current_instructions.values():
                    if negative_filter:
                        if all(inst not in current_instruction for inst in instructions):
                            filtered_samples.append(s)
                            stop = True
                    else:
                        if any(inst in current_instruction for inst in instructions):
                            filtered_samples.append(s)
                            stop = True
                    if stop:
                        break
        samples = filtered_samples
    print(f"metadata not found in {no_meta} trajs")
    return pd.concat(samples, axis=1).T


def filter_positive_negative_droid(
    samples,
    fs_handler,
    s3_prefix,
    filter=None,
    negative_filter=None,
):
    if filter is None:
        filter = {}
    filter_labs = filter.get("labs", [])
    filter_buildings = filter.get("buildings", [])
    filter_instructions = filter.get("instructions", [])
    if negative_filter is None:
        negative_filter = {}
    nfilter_labs = negative_filter.get("labs", [])
    nfilter_buildings = negative_filter.get("buildings", [])
    nfilter_instructions = negative_filter.get("instructions", [])
    all_metadata = os.path.join(s3_prefix, "aggregated_metadata.json")
    with fs_handler.open(all_metadata, "r") as f:
        all_metadata = json.load(f)
    samples = filter_droid(
        filter_buildings, filter_labs, filter_instructions, samples, negative_filter=False, all_metadata=all_metadata
    )
    samples = filter_droid(
        nfilter_buildings, nfilter_labs, nfilter_instructions, samples, negative_filter=True, all_metadata=all_metadata
    )
    return samples
