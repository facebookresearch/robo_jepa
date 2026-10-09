# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, patch_size=16, tubelet_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.proj(x.transpose(1, 2)).flatten(2).transpose(1, 2)
        return x
