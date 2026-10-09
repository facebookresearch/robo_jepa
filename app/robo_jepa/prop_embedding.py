# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import einops
import numpy as np
import torch
import torch.nn as nn

from app.robo_jepa.robo_jepa.models.utils.pos_embs import get_1d_sincos_pos_embed


class MultiEmbodimentLinear(nn.Module):
    def __init__(self, hidden_dim, num_embodiments, max_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_embodiments = num_embodiments
        self.max_size = max_size
        w_shape = (max_size, hidden_dim, num_embodiments)
        b_shape = (num_embodiments, hidden_dim)
        self.w = nn.Parameter(
            torch.zeros(
                w_shape,
            )
        )
        self.b = nn.Parameter(
            torch.zeros(
                b_shape,
            )
        )
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        to_onehot = torch.eye(num_embodiments)
        self.register_buffer("to_onehot", to_onehot)

    def forward(self, proprio, emb_id):
        if len(emb_id.shape) > 1:
            emb_id = emb_id.squeeze(1)
        emb_id = self.to_onehot[emb_id]
        return torch.einsum("b t p, b e, p h e -> b t h", proprio, emb_id, self.w) + (emb_id @ self.b).unsqueeze(1)


class ProprioceptiveEmbedding(nn.Module):
    def __init__(
        self,
        num_frames=16,
        tubelet_size=1,
        in_chans=7,  # in_chans=8,
        tokens_per_step=1,
        is_multiembodiment=False,
        num_embodiments=1,
        embed_dim=384,
        deep=False,
        use_3d_pos=False,
        use_embeddding=True,
        shift_input=True,
    ):
        super().__init__()
        # logging.info(f'using 3d prop position: {use_3d_pos}')

        # Map input to predictor dimension
        self.num_frames = num_frames
        self.use_embeddding = use_embeddding
        self.is_multiembodiment = is_multiembodiment
        self.num_embodiments = num_embodiments
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.shift_input = shift_input
        self.tokens_per_step = tokens_per_step

        if not is_multiembodiment:
            self.patch_embed = nn.Conv1d(
                in_chans, embed_dim * tokens_per_step, kernel_size=tubelet_size, stride=tubelet_size
            )
        else:
            self.patch_embed = MultiEmbodimentLinear(
                max_size=in_chans, num_embodiments=num_embodiments, hidden_dim=embed_dim
            )

        if deep:
            self.proprio_encoder = nn.Sequential(
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.proprio_encoder = nn.Identity()

        # since we drop last frame, reduce nb of dims by 1
        num_patches = num_frames // tubelet_size
        if self.shift_input:
            num_patches -= 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)

        if use_3d_pos:
            power = int(np.ceil(embed_dim / 6) * 2)
            res = embed_dim - power
            sincos = get_1d_sincos_pos_embed(power, num_patches, cls_token=False)
            sincos = np.concatenate([sincos, np.zeros([num_patches, res])], axis=1)
        else:
            sincos = get_1d_sincos_pos_embed(embed_dim, num_patches, cls_token=False)

        self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def resize_positional_embedding(self, new_t):
        if new_t == self.num_frames:
            return
        pe = nn.functional.interpolate(
            self.pos_embed.permute(0, 2, 1),
            scale_factor=new_t / self.pos_embed.shape[1],
            mode="linear",
        ).permute(0, 2, 1)
        self.pos_embed = nn.Parameter(pe, requires_grad=False)
        self.num_frames = new_t
        return self.pos_embed

    def forward(self, x, emb_id=None, starting_step=None, global_length=None):
        # x: proprioceptive vectors of shape [B T D] TODO: check cache here
        spatial = False
        if len(x.shape) == 4:
            assert self.tokens_per_step == 1
            spatial = True
            B_, T, V, D = x.shape
            x = einops.rearrange(x, "b t v d -> (b v) t d")
        B, T, D = x.shape
        if self.is_multiembodiment:
            x = self.patch_embed(x, emb_id)
        else:
            x = x.permute(0, 2, 1)
            x = self.patch_embed(x)
            x = x.permute(0, 2, 1)
        x = x.reshape(B, T, self.tokens_per_step, -1)
        x = self.proprio_encoder(x)
        if self.use_embeddding:
            pe = self.pos_embed
            if global_length is not None:
                if self.num_frames != global_length:
                    pe = self.resize_positional_embedding(global_length)
            if starting_step is not None or T < self.num_frames:
                t_start = 0 if starting_step is None else starting_step
                t_end = t_start + T
                assert t_end < self.num_frames, "positional embedding out of bounds. Resize it first"
                pe = pe[:, t_start:t_end]
            pe = pe.unsqueeze(2)
            x += pe
        if spatial:
            x = einops.rearrange(x.squeeze(2), "(b v) t d -> b t v d", b=B_, v=V)
        return x


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )
