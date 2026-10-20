# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import gc
import json
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Optional

import einops
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional as F

import app.robo_jepa.robo_jepa.models.predictor as vit_pred2
from app.robo_jepa.embodiment_preprocessing import (
    get_action_preprocessor,
    get_state_preprocessor,
    resolve_embodiment_name,
)
from app.robo_jepa.trainable_model import TrainableModel
from app.robo_jepa.utils import build_encoder
from app.robo_jepa.robo_jepa.models.utils.modules import KeyValueCache as WMKeyValueCache

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# Embodiments whose world model was trained with the proprio (state) input
# identically equal to zero — RoboCasa's PandaOmron robot under the dataset's
# ``action_source='action_key'`` convention, plus the DROID action_key
# variants (cart/joint × position/velocity) which also return states=zeros
# under ``action_source='action_key'``. For these, ``compute_new_pose``
# returns zeros (carrying the convention forward at inference) instead of
# running the DROID-style xyz+euler+gripper integration, which would
# otherwise misinterpret the chunked action vector as a 7-D pose delta and
# contaminate the next state token.
ACTION_ONLY_EMBODIMENTS = {
    "franka_omron",
    "franka_cart_action_key",
    "franka_joint_action_key",
    "franka_cart_vel_action_key",
    "franka_joint_vel_action_key",
}

# Embodiments whose world model was trained with ``action_source='state_delta'``
# — the action is *literally* the consecutive-step diff of the (raw,
# non-zero) state vector. ``compute_new_pose`` for these advances the state
# by simple addition (``new = pose + action``) on the un-padded prefix,
# matching training. Currently only RoboCasa's franka_omron variant.
STATE_DELTA_EMBODIMENTS = {"franka_omron_state_delta"}


def poses_to_diffs(poses, use_closedness=True):
    """Planner-side state-delta actions from absolute ``[T, 7]`` poses (xyz, euler xyz, gripper).

    Rotation deltas are ``R_{t+1} R_t^T`` as euler xyz, matching ``RobotVideoDataset.poses_to_diffs`` for
    ``3 < pose_dim <= 6`` and inverted by :func:`compute_new_pose`.
    """
    xyz = poses[:, :3]
    thetas = poses[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta.cpu().numpy(), degrees=False).as_matrix() for theta in thetas]
    xyz_diff = xyz[1:] - xyz[:-1]
    angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    angle_diff = torch.stack([torch.from_numpy(d).float() for d in angle_diff], dim=0)
    if use_closedness:
        closedness = poses[:, -1:]
        closedness_delta = closedness[1:] - closedness[:-1]
        return torch.cat([xyz_diff, angle_diff, closedness_delta], dim=1)
    else:
        return torch.cat([xyz_diff, angle_diff], dim=1)


@torch.compiler.disable
def compute_new_pose(pose, action, embodiment_ids=None, embodiment_list=None):
    """Advance proprio one step under each embodiment's own convention.

    For DROID-style embodiments the per-element math is the original one:
    XYZ delta + Euler-XYZ rotation composition + scalar gripper delta on the
    first 7 dims of the (already padded) state vector.

    For "action-only" embodiments — currently RoboCasa's ``franka_omron`` —
    the world model was trained with the proprio input identically equal to
    zero (see dataset's ``action_source='action_key'`` branch).  In that
    regime the integrated pose has no meaning and feeding back the
    DROID-style integration would produce garbage that contaminates the
    state token at the next ``wm.step`` iteration.  We therefore return an
    all-zero pose for those batch elements, matching the training-time
    convention exactly.

    For "state-delta" embodiments — currently
    ``franka_omron_state_delta`` — the dataset's ``action_source='state_delta'``
    means the action *is* the consecutive-step diff of the state, so we
    advance the state by simple addition: ``new_state = pose + action``
    on the un-padded prefix (slot widths defined by training, e.g. 14-D
    RoboCasa proprio for franka_omron_state_delta). The caller then
    F.pads the result back up to ``state_dim``. The DROID-style xyz +
    Euler integration is BYPASSED for these because the state vector
    layout (base_pos | base_quat | ee_pos_rel | ee_rot_rel_quat[xyz] |
    gripper_qpos) is incompatible with treating slots [0:6] as
    xyz + Euler.

    :param pose: ``[B, T=1, state_dim]`` (already F.padded to ``state_dim``).
    :param action: ``[B, T=1, action_dim]`` (already F.padded to ``action_dim``).
    :param embodiment_ids: optional ``[B, 1]`` long tensor of embodiment
        indices into ``embodiment_list``. If ``None`` (single-embodiment
        models), the DROID-style math is used for every element.
    :param embodiment_list: optional list of embodiment names matching
        ``embodiment_ids``. Should be ``self.model.embodiment_list``.
    :returns: ``[B, T=1, K]`` where K is 7 for DROID/action-only paths and
        ``min(state_dim, action_dim)`` (the un-padded state width) for
        state-delta paths; the caller F.pads back up to ``state_dim``.
    """
    device = pose.device
    pose_np = pose[:, 0].float().cpu().numpy()
    action_np = action[:, 0].cpu().numpy()
    B = pose_np.shape[0]

    # Per-element classification by embodiment.
    action_only_mask = np.zeros(B, dtype=bool)
    state_delta_mask = np.zeros(B, dtype=bool)
    if embodiment_ids is not None and embodiment_list is not None:
        emb_ids_np = embodiment_ids[:, 0].detach().cpu().numpy()
        for b in range(B):
            name = embodiment_list[int(emb_ids_np[b])]
            if name in ACTION_ONLY_EMBODIMENTS:
                action_only_mask[b] = True
            if name in STATE_DELTA_EMBODIMENTS:
                state_delta_mask[b] = True

    # Fast path: every element uses state-delta integration. Avoids the
    # heavy DROID-style scipy Rotation work entirely (which would crash
    # on the wider 14-D state vector anyway).
    if state_delta_mask.all() and B > 0:
        # new_state = pose + action, on the shared prefix length
        K = min(pose_np.shape[-1], action_np.shape[-1])
        new_state = pose_np[:, :K] + action_np[:, :K]
        return torch.from_numpy(new_state).to(device).float()[:, None]

    # DROID-style integration on the first 7 dims (xyz + euler-xyz + gripper).
    new_xyz = pose_np[:, :3] + action_np[:, :3]
    thetas = pose_np[:, 3:6]
    delta_thetas = action_np[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(B)]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack(angle_diff, axis=0)
    new_closedness = np.clip(pose_np[:, 6:7] + action_np[:, 6:7], 0, 1)
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)  # [B, 7]

    # Override action-only entries with zeros (matches training-time state).
    if action_only_mask.any():
        new_pose[action_only_mask, :] = 0.0

    # Per-element override: state-delta entries (when in a mixed-embodiment
    # batch) overwrite the DROID-style integration with the additive update
    # over the un-padded prefix.
    if state_delta_mask.any():
        K = min(pose_np.shape[-1], action_np.shape[-1])
        full = np.zeros((B, max(K, new_pose.shape[-1])), dtype=new_pose.dtype)
        full[:, : new_pose.shape[-1]] = new_pose
        full[state_delta_mask, :K] = pose_np[state_delta_mask, :K] + action_np[state_delta_mask, :K]
        new_pose = full

    return torch.from_numpy(new_pose).to(device).float()[:, None]


@dataclasses.dataclass
class WorldModelState:
    # the state is represented by the pair of input features and actions history
    features: Optional["tensor[batch, time, views, height, width, dims]"] = None  # noqa: F821
    states: Optional["tensor[batch, time, state_dim]"] = None  # noqa: F821
    camera_extrinsics: Optional["tensor[batch, 1, state_dim]"] = None  # noqa: F821
    actions: Optional["tensor[batch, time-1, action_dim]"] = None  # noqa: F821
    view_map: Optional["tensor[batch, views]"] = None  # noqa: F821
    dt: Optional["tensor[batch, time]"] = None  # noqa: F821
    embodiment_ids: Optional["tensor[batch,]"] = None  # noqa: F821
    # when cache is used, we need to determine what is the position id of the input
    feature_positions: Optional["tensor[batch]"] = None  # noqa: F821
    # or, equivalently, by the key-value cache representing both
    cache: Optional[WMKeyValueCache] = None

    def clone(self):
        return WorldModelState(
            features=self.features.clone() if self.features is not None else None,
            states=self.states.clone() if self.states is not None else None,
            camera_extrinsics=self.camera_extrinsics.clone() if self.camera_extrinsics is not None else None,
            actions=self.actions.clone() if self.actions is not None else None,
            view_map=self.view_map.clone() if self.view_map is not None else None,
            dt=self.dt.clone() if self.dt is not None else None,
            embodiment_ids=self.embodiment_ids.clone() if self.embodiment_ids is not None else None,
            feature_positions=self.feature_positions if self.feature_positions is not None else None,
            cache=self.cache.clone() if self.cache is not None else None,
        )

    @classmethod
    def from_cache(
        cls,
        cache,
        features,
        position=None,
        view_map=None,
        embodiment_ids=None,
        dt=None,
        states=None,
        camera_extrinsics=None,
    ):
        t = features.shape[1]
        v = features.shape[2]
        b = features.shape[0]
        if position is None:
            # TODO: check if layer 0 is fine here
            assert cache.pointer[0] <= cache.cache_len
            position = cache.pointer[0]
        assert t == 1, "inference with multiple input features and cache is not supported"
        if states is not None:
            assert states.shape[0] == b
            assert states.shape[1] == t
        if view_map is not None:
            assert view_map.shape[0] == b
            assert view_map.shape[1] == v
        if embodiment_ids is not None:
            assert embodiment_ids.shape[0] == b
        if dt is not None:
            assert dt.shape[0] == b
            assert dt.shape[1] == t
        actions = torch.zeros([b, 0, 7], device=features.device, dtype=features.dtype)
        position = torch.tensor(position, device=features.device)
        return WorldModelState(
            cache=cache,
            features=features,
            actions=actions,
            feature_positions=position,
            embodiment_ids=embodiment_ids,
            dt=dt,
            view_map=view_map,
            camera_extrinsics=camera_extrinsics,
            states=states,
        )

    @classmethod
    def from_features(
        cls,
        features,
        actions=None,
        states=None,
        view_map=None,
        embodiment_ids=None,
        dt=None,
    ):
        B, T, V, H, W, C = features.shape
        assert actions is None or actions.shape[1] == (T - 1)
        if actions is None:
            actions = torch.zeros((B, 0, 7), dtype=features.dtype, device=features.device)
        return WorldModelState(
            features=features,
            actions=actions,
            states=states,
            view_map=view_map,
            embodiment_ids=embodiment_ids,
            dt=dt,
        )

    def repeat(self, times: int):
        """repeats the batch dimension."""
        actions = self.actions.repeat(times, 1, 1)
        features = self.features.repeat(times, 1, 1, 1, 1, 1).float()
        if self.cache is not None:
            cache = WMKeyValueCache(
                layers=self.cache.layers,
                batch=self.cache.batch * times,
                frame_tokens=self.cache.frame_tokens,
                cache_len=self.cache.cache_len,
                num_heads=self.cache.num_heads,
                head_dim=self.cache.head_dim,
                dtype=self.cache.dtype,
                device=self.cache.device,
                pointer=self.cache.pointer,
                evicted=self.cache.evicted,
                keys=[k.repeat(times, 1, 1, 1, 1) for k in self.cache.keys],
                values=[v.repeat(times, 1, 1, 1, 1) for v in self.cache.values],
                # keys=self.cache.keys.repeat(1, times, 1, 1, 1, 1),
                # values=self.cache.values.repeat(1, times, 1, 1, 1, 1),
            )
        else:
            cache = None
        positions = self.feature_positions
        if self.states is not None:
            states = self.states.repeat(times, 1, 1)
        else:
            states = None
        if self.camera_extrinsics is not None:
            camera_extrinsics = self.camera_extrinsics.repeat(times, 1, 1)
        else:
            camera_extrinsics = None
        if self.view_map is not None:
            view_map = self.view_map.repeat(times, 1)
        else:
            view_map = None
        if self.embodiment_ids is not None:
            embodiment_ids = self.embodiment_ids.repeat(times, 1)
        else:
            embodiment_ids = None
        if self.dt is not None:
            dt = self.dt.repeat(times, 1)
        else:
            dt = None
        new_state = WorldModelState(
            actions=actions,
            features=features,
            cache=cache,
            feature_positions=positions,
            states=states,
            camera_extrinsics=camera_extrinsics,
            dt=dt,
            view_map=view_map,
            embodiment_ids=embodiment_ids,
        )
        # Preserve AOT cache for compiled model context replay.
        # CRITICAL: ``_aot_cache_keys/_values`` have shape
        # ``(depth, B, num_kv_heads, cache_len, frame_tokens, head_dim)``
        # — the BATCH dimension is **dim 1**, NOT dim 0. The previous
        # ``repeat(times, *([1] * (ndim-1)))`` broadcast was replicating
        # the depth axis (40 → 40*32) instead of the batch axis
        # (1 → 32) — which both (a) produced a malformed cache that the
        # compiled program either silently mis-read or paged into the
        # wrong layout, and (b) inflated host RSS by ~6 GB / cache /
        # rank per env step (the planner's ``wm_state.to('cpu')`` then
        # copies the bloated tensor to host) → 1.5 TB OOM after a few
        # env steps. Replicating the correct dim 1 keeps the cache at
        # the natural size (~6 GB on GPU at B=32, or ~196 MB at B=1).
        if hasattr(self, "_aot_cache_keys") and self._aot_cache_keys is not None:
            new_state._aot_cache_keys = self._aot_cache_keys.repeat(
                1, times, *([1] * (self._aot_cache_keys.ndim - 2))
            )
        if hasattr(self, "_aot_cache_values") and self._aot_cache_values is not None:
            new_state._aot_cache_values = self._aot_cache_values.repeat(
                1, times, *([1] * (self._aot_cache_values.ndim - 2))
            )
        return new_state

    def to(self, device):
        def maybe_to(tensor, device):
            return tensor.to(device) if tensor is not None else tensor

        if self.cache is not None:
            cache = WMKeyValueCache(
                layers=self.cache.layers,
                batch=self.cache.batch,
                frame_tokens=self.cache.frame_tokens,
                cache_len=self.cache.cache_len,
                num_heads=self.cache.num_heads,
                head_dim=self.cache.head_dim,
                dtype=self.cache.dtype,
                device=self.cache.device,
                pointer=self.cache.pointer,
                evicted=self.cache.evicted,
                keys=[k.to(device) for k in self.cache.keys],
                values=[v.to(device) for v in self.cache.values],
                # keys=self.cache.keys.to(device),
                # values=self.cache.values.to(device),
            )
        else:
            cache = None
        new_state = WorldModelState(
            actions=maybe_to(self.actions, device),
            features=maybe_to(self.features, device),
            cache=cache,
            feature_positions=self.feature_positions,
            states=maybe_to(self.states, device),
            dt=maybe_to(self.dt, device),
            view_map=maybe_to(self.view_map, device),
            embodiment_ids=maybe_to(self.embodiment_ids, device),
            camera_extrinsics=maybe_to(self.camera_extrinsics, device),
        )
        # Preserve AOT cache for compiled model context replay
        if hasattr(self, "_aot_cache_keys") and self._aot_cache_keys is not None:
            new_state._aot_cache_keys = self._aot_cache_keys.to(device)
        if hasattr(self, "_aot_cache_values") and self._aot_cache_values is not None:
            new_state._aot_cache_values = self._aot_cache_values.to(device)
        return new_state


class WorldModel(TrainableModel):
    def __init__(self, transition_model_config, train_config=None, heads=None, device="cpu"):
        if "grid_size" in transition_model_config:
            raise ValueError("Config key 'grid_size' is not supported. Use 'grid_height' and 'grid_width' instead.")
        transition_model = vit_pred2.__dict__["vit_predictor"](**transition_model_config)
        super().__init__(transition_model, train_config=train_config, device=device)
        self.transition_model = transition_model
        self.encoder_layernorm = transition_model_config["encoder_layernorm"]
        self.encoder = None
        self.encoder_version = None
        if heads is None:
            heads = {}
        self.heads = heads

        if train_config is not None:
            self.rollout_steps = self.train_config.get("rollout_steps")
            self.predictor_window_steps = self.train_config.get("predictor_window_steps")
            self.eval_rollout_steps = self.train_config.get("eval_rollout_steps")
            self.train_rollout_prefixes = self.train_config.get("train_rollout_prefixes")
            self.rollout_stop_gradient = self.train_config.get("rollout_stop_gradient")
            # Optional per-view L1 loss weighting: dict {int_view_id: float_weight}.
            # Missing IDs default to weight 1.0. Empty dict = no-op (current behavior).
            self.view_loss_weights = dict(self.train_config.get("view_loss_weights", {}) or {})
            # Fused autoregressive rollout backward (opt-in, default OFF). When True,
            # `rollout` generates all AR steps once under no_grad and runs a SINGLE
            # forward+backward over the full causal sequence instead of one
            # forward+backward per step. Mathematically equivalent to the per-step
            # path when feedback is detached (rollout_stop_gradient) and attention is
            # frame-causal; see _rollout_fused. The legacy per-step path stays the
            # default and is used as a fallback whenever the fused gate is not met.
            self.rollout_fused_backward = bool(self.train_config.get("rollout_fused_backward", False))
            # Loss aggregation for the rollout objective:
            #   "per_step" (default) — mean-of-means: each AR step normalized by its own
            #       valid-token count, then summed (bit-compatible with the legacy loop).
            #   "valid" — pooled: every valid token weighted equally across all AR steps;
            #       scaled by R so it coincides with "per_step" under uniform padding.
            self.rollout_loss_weighting = str(self.train_config.get("rollout_loss_weighting", "per_step"))
            if self.rollout_loss_weighting not in ("per_step", "valid"):
                raise ValueError(
                    f"rollout_loss_weighting must be 'per_step' or 'valid', got {self.rollout_loss_weighting!r}"
                )
            # Use KV-cache inference for the (no_grad) generation phase of the fused
            # rollout. Each AR step then processes only the new frame (O(R)) instead
            # of re-forwarding the growing prefix (O(R^2)). Only the generation is
            # cached; the single grad forward is unchanged. Opt-in, default OFF.
            self.rollout_fused_kv_cache = bool(self.train_config.get("rollout_fused_kv_cache", False))
        else:
            self.view_loss_weights = {}
            self.rollout_fused_backward = False
            self.rollout_loss_weighting = "per_step"
            self.rollout_fused_kv_cache = False

    # encoder functions

    def load_encoder(
        self,
        path,
        use_proj,
        device="cuda",
        version="v21",
        max_num_frames=16,
        compile=False,
    ):
        encoder = build_encoder(
            version=version,
            device=device,
            filename=path,
            use_proj=use_proj,
            max_num_frames=max_num_frames,
        )
        if compile:
            # fullgraph=False: the hub V-JEPA 2.1 attention enters `torch.backends.cuda.sdp_kernel()`, a context
            # manager Dynamo cannot trace, so the graph has to break there.
            self.encoder = torch.compile(encoder, dynamic=False, fullgraph=False)
        else:
            self.encoder = encoder
        self.encoder_version = version

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path,
        load_decoder=True,
        load_encoder=True,
        load_compiled=False,
        load_compiled_decoder=False,
        encoder_path=None,
        decoder_path=None,
        device="cuda",
        resolution=None,
        compiled_max_steps=None,
        weights_filename="world_model.pth.tar",
    ):
        """Load a pretrained WorldModel from a checkpoint directory.

        Args:
            weights_filename: Name of the world-model weights file inside
                ``checkpoint_path``. Defaults to ``world_model.pth.tar``.
            load_compiled: False to load eager model, True or "bf16" for bf16
                AOT programs, or "mp-bf16" for mixed-precision (fp32 weights +
                autocast bf16) AOT programs.
            load_compiled_decoder: False to use eager decoder, True or "bf16"
                for bf16 AOT decoder programs.  Requires load_decoder=True.
                Independent of load_compiled — you can load a compiled decoder
                with an eager WM or vice versa.  When set, the compiled decoder
                is installed as ``heads["image_head"]`` and the original eager
                head is backed up to ``heads["image_head_eager"]``.
            resolution: Optional (H, W) pixel resolution tuple for
                cross-resolution compiled inference. When specified with
                load_compiled=True, loads programs from the resolution-specific
                compiled subdirectory (e.g. compiled/bf16-256x256/).
        """
        # to avoid circular imports
        if load_decoder:
            from app.robo_jepa.world_model_heads import WorldModelDiTCosmosHead

            if decoder_path is None:
                decoder_path = os.path.join(checkpoint_path, "decoder")
            dec_config_path = os.path.join(decoder_path, "config.json")
            with open(dec_config_path, "r") as f:
                dec_config = json.load(f)
            dec_config["decoder_path"] = os.path.join(checkpoint_path, "decoder", dec_config["decoder_path"])

            # The Cosmos DiT is the only decoder; a config without "kind" is treated as one.
            decoder_type = dec_config.pop("kind", "dit")
            if decoder_type != "dit":
                raise ValueError(f"unsupported decoder kind {decoder_type!r} in {dec_config_path} (only 'dit')")
            diffusion_kwargs = dec_config.pop("diffusion_kwargs", None)
            wm_image_head = WorldModelDiTCosmosHead(
                head_config=dec_config,
                device=device,
                diffusion_kwargs=diffusion_kwargs,
            )
            heads = dict(image_head=wm_image_head)
        else:
            heads = None

        config = os.path.join(checkpoint_path, "config.json")
        state = torch.load(os.path.join(checkpoint_path, weights_filename), map_location="cpu")
        with open(config) as f:
            config = json.load(f)
        encoder_version = config.get("encoder_version", "v21")
        if "encoder_version" in config:
            config.pop("encoder_version")
        wm = WorldModel(transition_model_config=config, device=device, heads=heads)
        wm.encoder_version = encoder_version
        try:
            # to support old models before refactor
            wm.transition_model.load_state_dict(
                {k.replace("module.", ""): v for k, v in state["predictor"].items()},
                strict=False,
            )
        except KeyError:
            wm.transition_model.load_state_dict(
                {k.replace("module.", ""): v for k, v in state["model"].items()},
                strict=False,
            )
        for p in wm.transition_model.parameters():
            p.requires_grad = False
        wm.transition_model.eval()
        if encoder_path is None:
            encoder_path = os.path.join(checkpoint_path, "encoder.pth.tar")
        if load_encoder:
            wm.load_encoder(
                encoder_path,
                device=device,
                use_proj=False,
                max_num_frames=64,
                version=encoder_version,
            )
        if load_decoder:
            dec_state = torch.load(f"{decoder_path}/wm_decoder.pth.tar", map_location="cpu")
            wm.heads["image_head"].ema_model.load_state_dict(
                {k.replace("module.", ""): v for k, v in dec_state["model"].items()},
                strict=False,
            )
            for p in wm.heads["image_head"].ema_model.parameters():
                p.requires_grad = False
            wm.heads["image_head"].eval()
        if load_compiled:
            # Resolve precision: True -> "bf16", or explicit "bf16" / "mp-bf16"
            if load_compiled is True:
                precision = "bf16"
            else:
                precision = str(load_compiled)

            # Determine attn_impl from precision string.
            # Convention: "bf16-sdpa-cudnn" → precision="bf16", attn_impl="sdpa_cudnn"
            #             "bf16" → precision="bf16", attn_impl="fmha" (default)
            #             "mp-bf16-sdpa-cudnn" → precision="mp-bf16", attn_impl="sdpa_cudnn"
            attn_impl = "fmha"
            for suffix in ("sdpa_cudnn", "sdpa-cudnn", "sdpa"):
                normalized = suffix.replace("-", "_")
                if f"-{suffix}" in precision:
                    attn_impl = normalized
                    precision = precision.replace(f"-{suffix}", "")
                    break

            # Try precision subfolder first, fall back to flat compiled/ for
            # backward compatibility with existing checkpoints
            if attn_impl != "fmha":
                dir_suffix = f"{precision}-{attn_impl.replace('_', '-')}"
            else:
                dir_suffix = precision
            # Append resolution suffix if specified
            if resolution is not None:
                dir_suffix += f"-{resolution[0]}x{resolution[1]}"
            compiled_dir = os.path.join(checkpoint_path, "compiled", dir_suffix)
            if not os.path.isdir(compiled_dir):
                # Fallback: try without attn_impl suffix
                fallback_suffix = precision
                if resolution is not None:
                    fallback_suffix += f"-{resolution[0]}x{resolution[1]}"
                compiled_dir = os.path.join(checkpoint_path, "compiled", fallback_suffix)
            if not os.path.isdir(compiled_dir):
                # Fallback: flat compiled/ dir (old layout)
                compiled_dir = os.path.join(checkpoint_path, "compiled")
            if not os.path.isdir(compiled_dir):
                raise FileNotFoundError(
                    f"load_compiled={load_compiled!r} but no compiled/{dir_suffix}/ or "
                    f"compiled/ directory found in {checkpoint_path}. "
                    f"Run compile_checkpoint first."
                )
            from app.robo_jepa.inference.dispatcher import AOTWorldModelDispatcher

            # Pass resolution override to dispatcher
            res_grid_h = resolution[0] // 16 if resolution else None
            res_grid_w = resolution[1] // 16 if resolution else None
            dispatcher = AOTWorldModelDispatcher(
                wm,
                artifact_dir=compiled_dir,
                device=device,
                precision=precision,
                attn_impl=attn_impl,
                grid_height=res_grid_h,
                grid_width=res_grid_w,
            )
            dispatcher.load_programs(max_steps=compiled_max_steps)
            logger.info(f"Loaded AOT dispatcher with {len(dispatcher.compiled_variants)} variants from {compiled_dir}")

            result = dispatcher
        else:
            result = wm

        # ── Optionally load compiled decoder (independent of WM) ──────
        if load_compiled_decoder and load_decoder:
            decoder_precision = "bf16" if load_compiled_decoder is True else str(load_compiled_decoder)
            decoder_compiled_dir = os.path.join(checkpoint_path, "compiled", f"decoder-{decoder_precision}")
            if not os.path.isdir(decoder_compiled_dir):
                decoder_compiled_dir = os.path.join(checkpoint_path, "compiled", "decoder")
            if os.path.isdir(decoder_compiled_dir):
                from app.robo_jepa.inference.decoder_dispatcher import (
                    AOTDecoderDispatcher,
                )

                decoder_disp = AOTDecoderDispatcher.from_world_model(
                    wm,
                    artifact_dir=decoder_compiled_dir,
                    device=device,
                )
                # Back up the eager head and install the compiled dispatcher
                wm.heads["image_head_eager"] = wm.heads["image_head"]
                wm.heads["image_head"] = decoder_disp
                logger.info(
                    f"Installed compiled decoder as heads['image_head'] "
                    f"({len(decoder_disp.compiled_variants)} variants from "
                    f"{decoder_compiled_dir}). "
                    f"Eager head backed up to heads['image_head_eager']."
                )
            else:
                logger.warning(
                    f"load_compiled_decoder={load_compiled_decoder!r} but no "
                    f"decoder compiled directory found at {decoder_compiled_dir}"
                )

        return result

    def preprocess_frames(self, frames: np.ndarray, size: int | tuple[int]) -> torch.Tensor:
        """
        frames: [N, H, W, 3] numpy array valued in [0..255], uint8 dtype
        """
        buffer = torch.tensor(frames, dtype=torch.float32)
        buffer = buffer.permute(3, 0, 1, 2)
        buffer = torch.nn.functional.interpolate(
            buffer,
            size=(size, size) if isinstance(size, int) else size,
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None, None]
        buffer = buffer.to(torch.uint8).float() / 255.0
        buffer = buffer - mean
        buffer = buffer / std
        return buffer  # 3 N H W

    def encode(self, frames, size=(256, 256), device="cuda", dtype=torch.float32):
        video = self.preprocess_frames(frames, size).to(device)
        self.encoder.vit.embed_dim
        h_, w_ = video.shape[-2:]
        h_ //= self.encoder.vit.patch_size
        w_ //= self.encoder.vit.patch_size
        video = einops.repeat(video, "c b h w -> b c t h w", t=1)  # V-JEPA 2.1: one frame per image
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
                h = self.encoder(video.float())
        h = einops.rearrange(h, "b (h w) c -> b h w c", h=h_, w=w_)
        if self.encoder_layernorm:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    def encode_features(
        self,
        video,
        window=2,
        tubelet_size=2,
        grid_size=14,
        add_batch=True,
        device="cuda",
        dtype=torch.float32,
    ):
        z = []
        d = window
        dim = self.encoder.vit.embed_dim
        if add_batch:
            video = video[None]  # add batch dimension
        else:
            b = video.shape[0]
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        for T in range(tubelet_size, video.shape[2] + tubelet_size, tubelet_size):
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype, enabled=True):
                    features = self.encoder(video[:, :, max(0, T - d) : T].to(device))
            if add_batch:
                computed_features = (
                    features[0].reshape(min(T, d) // tubelet_size, *grid_size, dim).to(torch.float32)[-1]
                )
            else:
                computed_features = features.reshape(b, min(T, d) // tubelet_size, *grid_size, dim).to(torch.float32)[
                    -1
                ]
            z.append(computed_features)
        z = torch.stack(z, dim=0)  # []
        if self.encoder_layernorm:
            z = F.layer_norm(z, (z.size(-1),))
        return z

    def encode_images(
        self,
        video,
        window=2,
        tubelet_size=2,
        grid_size=14,
        device="cuda",
        dtype=torch.float32,
    ):
        z = []
        d = window
        dim = self.encoder.vit.embed_dim
        video = video[None]  # add batch dimension
        for T in range(tubelet_size, video.shape[2] + tubelet_size, tubelet_size):
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype, enabled=True):
                    features = self.encoder(video[:, :, max(0, T - d) : T].to(device))
            computed_features = (
                features[0].reshape(min(T, d) // tubelet_size, grid_size, grid_size, dim).to(torch.float32)[-1]
            )
            z.append(computed_features)
        return torch.stack(z, dim=0)  # []

    def delete_encoder(self):
        del self.encoder
        torch.cuda.empty_cache()
        gc.collect()

    def load_checkpoint(
        self,
        path,
        load_optimizer=True,
        strict=True,
        dont_load=None,
        load_heads=True,
        load_module=True,
    ):
        start_epoch = 0
        if load_heads:
            for name, head in self.heads.items():
                head_path = path.removesuffix(".pth.tar") + "_" + name + ".pth.tar"
                # Heads saved before this used `path.strip(".pth.tar")`, e.g. `world_model-lates_image_head.pth.tar`.
                # FSDP2 saves a DCP directory without the suffix, so check both forms.
                legacy_path = path.strip(".pth.tar") + "_" + name + ".pth.tar"
                saved = lambda p: os.path.exists(p) or os.path.isdir(p.removesuffix(".pth.tar"))  # noqa: E731
                if not saved(head_path) and saved(legacy_path):
                    logger.warning(f"loading head {name} from its pre-fix name {legacy_path}")
                    head_path = legacy_path
                start_epoch, _ = head.load_checkpoint(head_path, load_optimizer, strict, dont_load)
        if load_module:
            return super().load_checkpoint(path, load_optimizer, strict, dont_load)
        return start_epoch, None

    def save_checkpoint(self, epoch, path, stats=None, rank=0):
        for name, head in self.heads.items():
            head_path = path.removesuffix(".pth.tar") + "_" + name + ".pth.tar"
            # Forward rank as kwarg so the head's save_checkpoint sees the
            # caller's rank (its parent signature is `(epoch, path, stats, rank)`
            # — passing rank positionally would bind it to `stats`).
            head.save_checkpoint(epoch, head_path, rank=rank)
        # Pass rank as kwarg.  TrainableModel.save_checkpoint's signature is
        # `(self, epoch, path, stats=None, rank=0)`, so a positional 3rd arg
        # would silently bind to `stats` and `rank` would stay 0 on every
        # process — causing every rank to enter the rank-0-only makedirs and
        # race on FileExistsError under HSDP.
        return super().save_checkpoint(epoch, path, stats=stats, rank=rank)

    @torch.compiler.disable
    def step(
        self,
        action: torch.Tensor,
        state: WorldModelState,
        use_cache=False,
        no_grad=True,
        batch=100,
        device="cuda",
        pos_embedding_resize_t=None,
    ) -> WorldModelState:
        """
        the main forward pass function for working with the world model.
        This method runs the world model in an RNN style:

        observation, state = world_model.step(action, state)

        here observation is the predicted encoder feature, state is an
        instance of `WorldModelState`, and action is a 7-dim vector

        Args:
            action ([B, 1, K]): planner action. ``K`` may be the model's
                ``action_dim`` (already preprocessed) or smaller. Smaller
                shapes are routed through the embodiment-keyed
                preprocessors registered in
                :mod:`app.robo_jepa.embodiment_preprocessing`. The
                planner-side default is the DROID 7-D
                ``[dx, dy, dz, droll, dpitch, dyaw, d_grip]`` layout.
            state: the WorldModelState object representing
                * actions_history ([B, T - 1, K]): same shape contract as ``action``.
                * video_features_history ([B, T, V, H, W, C]): video features tensor.
                    V is the number of cameras, H/W are token grid dims, C the model dim.
                * (Optional) states ([B, T, K_s]): robot poses for each step.
                    The WM does not necessarily predict the robot pose so the
                    user has to update the state manually.
                * (Optional) camera_extrinsics ([B, 1, 6]): camera extrinsics.
                * (Optional) embodiment_ids ([B, 1]): per-batch embodiment
                    index into ``self.model.embodiment_list``.

        Embodiment dispatch
        -------------------
        The action and state tensors are passed through per-embodiment
        preprocessors before being fed to the predictor. **All batch
        elements must share the same embodiment** (asserted) so the
        preprocessing stays a single vectorised op. Currently registered:

        * ``franka_omron`` (RoboCasa PandaOmron): converts the 7-D planner
          action to canonical 12-D layout A and broadcasts over a
          chunk of size 4, yielding 48-D. State is zero-padded.
        * (default fallback): zero-pad / truncate to ``action_dim`` and
          ``state_dim`` (matches the DROID-style behaviour).

        See :mod:`app.robo_jepa.embodiment_preprocessing` for details.

        Returns:
            new_state:
            new_history: a tuple of
                * actions_history ([B, T, action_dim]):
                * video_features_history ([B, T+1, V, H, W, C]):
        """
        actions_history, video_features_history = state.actions, state.features
        B, T, V, H, W, C = video_features_history.shape
        b, t, _ = action.shape
        assert t == 1, "only one step at a time is supported"

        # ── embodiment-keyed preprocessing dispatch ─────────────────────
        embodiment_ids = state.embodiment_ids
        embodiment_list = getattr(self.model, "embodiment_list", None)
        embodiment_name = resolve_embodiment_name(embodiment_ids, embodiment_list)
        action_preproc = get_action_preprocessor(embodiment_name)
        state_preproc = get_state_preprocessor(embodiment_name)

        if action.shape[-1] != self.model.action_dim:
            action = action_preproc(action, self.model.action_dim)
        if actions_history.shape[-1] != self.model.action_dim:
            # Always-empty pre-history (B, 0, action_dim); preprocessor not needed.
            actions_history = torch.zeros([actions_history.shape[0], 0, self.model.action_dim])
        actions = torch.cat([actions_history.to(device), action.to(device)], dim=1)  # .float()
        poses = state.states
        if poses is not None and poses.shape[-1] != self.model.state_dim:
            poses = state_preproc(poses, self.model.state_dim)
        view_map = state.view_map
        dt = state.dt
        camera_extrinsics0 = state.camera_extrinsics
        if camera_extrinsics0 is not None:
            camera_extrinsics = camera_extrinsics0.repeat(1, T, 1)
        else:
            camera_extrinsics = None
        # note that here we are running the full attention process, including for the history
        # in future we'll change that to KV-cache of history, right now it has a limited support on v100s
        cache = None
        position = None
        if no_grad:
            if use_cache:
                if state.cache is not None:
                    cache = state.cache
                    position = state.feature_positions
                    cache_size = state.cache.pointer[0]
                else:
                    position = 0
                    cache_size = 0
                out, _, _, cache = self.forward(
                    video_features_history.to(device),
                    actions.to(device),
                    use_cache=True,
                    states=poses,
                    camera_extrinsics=camera_extrinsics,
                    dt=dt,
                    view_map=view_map,
                    cache_size=cache_size,
                    embodiment_ids=embodiment_ids,
                    cache=cache,
                    input_position=position,
                    pos_embedding_resize_t=pos_embedding_resize_t,
                )
                # TODO: think about how to better perform state initialization
                position += video_features_history.shape[1]
                features = out[:, -1:]
                actions = actions[:, :0]
            else:
                out = self.batched_forward(
                    video_features_history,
                    actions,
                    states=poses,
                    camera_extrinsics=camera_extrinsics,
                    batch=batch,
                    pos_embedding_resize_t=pos_embedding_resize_t,
                    device=device,
                    view_map=view_map,
                    embodiment_ids=embodiment_ids,
                    dt=dt,
                )
                features = torch.cat([video_features_history.cpu(), out.cpu()], dim=1)
        else:
            out, _, _, _ = self.forward(
                video_features_history.to(device),
                actions.to(device),
                states=poses,
                camera_extrinsics=camera_extrinsics,
                view_map=view_map,
                embodiment_ids=embodiment_ids,
                dt=dt,
            )
            out = out[:, -1:]
            features = torch.cat([video_features_history.cpu(), out.cpu()], dim=1)
        if self.model.state_tokens > 0:
            assert poses is not None
            new_pose = compute_new_pose(
                poses[:, -1:],
                action,
                embodiment_ids=embodiment_ids,
                embodiment_list=getattr(self.model, "embodiment_list", None),
            )
            if poses.shape[-1] != new_pose.shape[-1]:
                new_pose = state_preproc(new_pose, self.model.state_dim)
            poses = torch.cat([poses, new_pose], dim=1)
        if use_cache and poses is not None:
            poses = poses[:, -1:]
        return (
            out,
            WorldModelState(
                actions=actions.cpu(),
                features=features.cpu(),
                cache=cache,
                feature_positions=position,
                camera_extrinsics=camera_extrinsics0,
                embodiment_ids=embodiment_ids,
                dt=dt,
                states=poses,
                view_map=view_map,
            ),
        )

    def forward(
        self,
        video_features,
        actions,
        states=None,
        camera_extrinsics=None,
        view_map=None,
        embodiment_ids=None,
        dt=None,
        use_cache=False,
        cache=None,
        cache_size=None,
        input_position=None,
        pos_embedding_resize_t=None,
    ):
        """
        Wrapper method for the inner model forward call
        Args:
            video_features:
            actions:
            states:
            camera_extrinsics:
        Returns:

        """
        video_features, action_features, next_states, cache = self.transition_model(
            video_features,
            actions=actions,
            states=states,
            camera_extrinsics=camera_extrinsics,
            use_cache=use_cache,
            view_map=view_map,
            embodiment_ids=embodiment_ids,
            dt=dt,
            cache=cache,
            input_position=input_position,
            pos_embedding_resize_t=pos_embedding_resize_t,
            cache_size=cache_size,
        )
        return video_features, action_features, next_states, cache

    def select_history(
        self,
        video_features,
        actions,
        predicted_features,
        states,
        predicted_states,
        camera_extrinsics,
        prefix_len: int,
        dt=None,
        suffix_len: int = None,
        pad_to_length: int = None,
    ) -> WorldModelState:
        """
        Assuming we have a sequence of encoder states and a sequence of predicted features
        where predicted_features[:, t] is predicted from history of video_features[:, :t+1] from encoder,
        this method constructs a new history of the first `prefix_len` steps from encoder and one predictor step.

        Args:
            video_features [B, T, V, H, W, C]:
            actions, [B, T, 7]:
            predicted_features [B, T, V, H, W, C]:
            prefix_len (int): how many encoder steps to take
            suffix_len (int): how many predictor steps to take
            pad_to_length (int): if set, pad feature_sequence and actions along the time
                dimension to this fixed length so that the compiled model always sees the
                same tensor shape.  A ``compile_padding_mask`` is stored on the returned
                state so that downstream code can ignore padded positions.
        """
        B, T, V, H, W, C = predicted_features.shape
        t = prefix_len
        if suffix_len is None:
            d = self.predictor_window_steps
        else:
            d = suffix_len
        current_history = video_features[:, :t]
        if dt is not None:
            dt_sequence = dt[:, : t + d]
        else:
            dt_sequence = None
        next_steps = predicted_features[:, t - 1 : t - 1 + d]
        state_sequence = None
        if self.model.decode_state:
            current_state_history = states[:, :t]
            next_states = predicted_states[:, t - 1 : t - 1 + d]
            state_sequence = torch.cat([current_state_history, next_states], dim=1)
        elif self.model.state_tokens > 0:
            state_sequence = states[:, : t + d]
        ce_sequence = None
        if self.model.camera_extrinsics_tokens > 0:
            # all camera extrinsics params are supposed to be fixed along the seuqence
            # so we'll just repeat them in the forward pass
            ce_sequence = camera_extrinsics[:, :1]
        actions = actions[:, : t + d - 1]  # we have t states from encoder and d from predictor.
        # Since we need to have 1 action less than state we select t actions
        feature_sequence = torch.cat([current_history, next_steps], dim=1)

        real_len = feature_sequence.shape[1]

        if pad_to_length is not None and real_len < pad_to_length:
            pad_t = pad_to_length - real_len
            feature_sequence = F.pad(feature_sequence, (0, 0, 0, 0, 0, 0, 0, 0, 0, pad_t))
            actions = F.pad(actions, (0, 0, 0, pad_t))
            if state_sequence is not None:
                state_sequence = F.pad(state_sequence, (0, 0, 0, pad_t))
            if dt_sequence is not None:
                dt_sequence = F.pad(dt_sequence, (0, pad_t))

        return WorldModelState(
            actions=actions,
            features=feature_sequence,
            states=state_sequence,
            camera_extrinsics=ce_sequence,
            dt=dt_sequence,
        )

    def rollout(
        self,
        state: WorldModelState,
        actions_suffix,
        features_suffix=None,
        states_suffix=None,
        padding_mask=None,
        action_noise=0.0,
        view_map=None,
        embodiment_ids=None,
        dt=None,
        loss_weight=1.0,
        suffix_len=None,
        run_backward=False,
        rollout_steps=None,
        pad_to_length=None,
        apply_view_loss_weights=False,
    ):
        if suffix_len is None:
            d = self.predictor_window_steps
        else:
            d = suffix_len
        # todo (rollout_len - 1) was in the orig code
        if rollout_steps is None:
            rollout_steps = self.rollout_steps - 1
        rollout_steps = min(rollout_steps, actions_suffix.shape[1])
        loss_weight = loss_weight / (rollout_steps + 1)

        # ── Fused single-backward path (opt-in via rollout_fused_backward) ──────
        # Generate all AR steps once under no_grad, then run ONE forward+backward
        # over the full causal sequence instead of one forward+backward per step.
        # Mathematically equivalent to the per-step loop below because (a) the AR
        # feedback is detached (rollout_stop_gradient) so per-step grads are
        # disjoint and additive, and (b) temporal attention is strictly causal so
        # each output position's receptive field equals the matching per-step
        # input. Gated to exactly the settings where this holds; otherwise fall
        # through to the legacy loop. See _rollout_fused.
        use_fused = (
            self.rollout_fused_backward
            and run_backward
            and action_noise == 0
            and bool(self.rollout_stop_gradient)
            and d == 1
            and not self.model.decode_state
            and features_suffix is not None
            and rollout_steps >= 1
        )
        if use_fused:
            return self._rollout_fused(
                state,
                actions_suffix=actions_suffix,
                features_suffix=features_suffix,
                states_suffix=states_suffix,
                padding_mask=padding_mask,
                view_map=view_map,
                embodiment_ids=embodiment_ids,
                dt=dt,
                loss_weight=loss_weight,
                d=d,
                rollout_steps=rollout_steps,
                pad_to_length=pad_to_length,
                apply_view_loss_weights=apply_view_loss_weights,
            )

        rollout_losses = defaultdict(list)
        for h in range(rollout_steps):
            logger.debug(f"Autoregressive step {h}")
            new_action = actions_suffix[:, h : h + 1]
            if action_noise > 0:
                new_action = torch.randn(*new_action.shape).to(new_action.device).to(new_action.dtype) * action_noise
            features, actions = state.features, state.actions
            dt_prev = state.dt
            poses = state.states
            actions = torch.cat([actions, new_action], dim=1)
            ce = state.camera_extrinsics
            if ce is not None:
                # there is only one camera extrinsic vector for the video so we repeat it for every frame
                ce = ce.repeat(1, actions.shape[1], 1, 1)

            # -- pad inputs to fixed length for torch.compile ----------------
            real_t = features.shape[1]
            if pad_to_length is not None and real_t < pad_to_length:
                _pad = pad_to_length - real_t
                features_fwd = F.pad(features, (0, 0, 0, 0, 0, 0, 0, 0, 0, _pad))
                actions_fwd = F.pad(actions, (0, 0, 0, _pad))
                poses_fwd = F.pad(poses, (0, 0, 0, _pad)) if poses is not None else None
                dt_fwd = F.pad(dt_prev, (0, _pad)) if dt_prev is not None else None
                if ce is not None:
                    ce_fwd = F.pad(ce, (0, 0, 0, 0, 0, _pad))
                else:
                    ce_fwd = None
            else:
                features_fwd = features
                actions_fwd = actions
                poses_fwd = poses
                dt_fwd = dt_prev
                ce_fwd = ce
            # ----------------------------------------------------------------

            next_steps, _, next_poses, _ = self.forward(
                features_fwd.detach() if self.rollout_stop_gradient else features_fwd,
                actions_fwd,
                states=poses_fwd.detach() if poses_fwd is not None else None,
                camera_extrinsics=ce_fwd,
                view_map=view_map,
                embodiment_ids=embodiment_ids,
                dt=dt_fwd,
                use_cache=False,
            )

            # -- slice back to real length after compiled forward -------------
            if pad_to_length is not None and real_t < pad_to_length:
                next_steps = next_steps[:, :real_t]
                if next_poses is not None:
                    next_poses = next_poses[:, :real_t]
            # ----------------------------------------------------------------

            # we don't need anything but the last step (which corresponds to the prediction after everything except )
            next_steps = next_steps[:, -d:]
            # next_step = next_steps[:, -1:]
            f_len = features.shape[1]
            features = features[:, : f_len - d + 1]
            features = torch.cat(
                [
                    features.detach() if self.rollout_stop_gradient else features,
                    next_steps,
                ],
                dim=1,
            )  # [B T+1 V H W C]
            if self.model.decode_state:
                # if we predict the pose via the model, we concat history with the newly predicted pose
                poses = torch.cat(
                    [
                        poses.detach() if self.rollout_stop_gradient else poses,
                        next_poses[:, -1:],
                    ],
                    dim=1,
                )
            elif self.model.state_tokens > 0 and states_suffix is not None:
                # if we don't predict the pose via the model, we concat history with the pose from the training sequence  # noqa: E501
                # TODO: change that to action integration (accumulation), for better eval metrics
                poses = torch.cat(
                    [
                        poses.detach() if self.rollout_stop_gradient else poses,
                        states_suffix[:, h : h + 1],
                    ],
                    dim=1,
                )
            elif self.model.state_tokens > 0:
                raise NotImplementedError()  # TODO: integrate the state manually
            else:
                poses = None
            if self.model.camera_extrinsics_tokens > 0:
                ce = ce[:, :1]
            else:
                ce = None
            if dt_prev is not None:
                if dt is None:
                    raise ValueError("dt is not provided")
                dt_prev = torch.cat([dt_prev, dt[:, h : h + 1]], dim=1)
            state = WorldModelState(
                features=features,
                actions=actions,
                states=poses,
                camera_extrinsics=ce,
                dt=dt_prev,
                view_map=view_map,
                embodiment_ids=embodiment_ids,
            )
            if features_suffix is not None:
                targets = features_suffix[:, h : h + d]
                if self.model.decode_state:
                    target_poses = states_suffix[:, h : h + d]
                    next_poses = next_poses[:, -d:]
                else:
                    target_poses = None
                losses = self.compute_loss(
                    next_steps,
                    targets,
                    target_poses,
                    next_poses,
                    padding_mask[:, h : h + d],
                    shift=0,
                    view_map=view_map,
                    apply_view_loss_weights=apply_view_loss_weights,
                )
                logger.debug(f"Current padding mask: {padding_mask[:, h : h + d]}")
                for k in losses:
                    rollout_losses[k].append(losses[k].detach() if hasattr(losses[k], "detach") else losses[k])
                loss = losses["loss"] * loss_weight
                logger.debug(f"Finish autoregressive step {h}. The loss is {loss.item()}. Computing gradients...")
                if run_backward:
                    self.backward(loss)
                    nan_params = [
                        n
                        for n, p in self.model.named_parameters()
                        if p.grad is not None and torch.isnan(p.grad).any().item()
                    ]
                    none_params = [n for n, p in self.model.named_parameters() if p.grad is None]
                    if len(nan_params) == 0:
                        suff = "No NaNs found!"
                    else:
                        suff = f"Found NaNs in parameters: {nan_params}"
                    suff2 = f"None grads: {none_params}"
                    logger.debug(f"Finish autoregressive step {h}. Done computing gradients. {suff}. {suff2}")
        for k in rollout_losses:
            rollout_losses[k] = (
                torch.stack(rollout_losses[k].detach())
                if hasattr(rollout_losses[k], "detach")
                else torch.tensor(rollout_losses[k], device=loss.device)
            )
        return rollout_losses, state

    def _rollout_fused(
        self,
        state: WorldModelState,
        actions_suffix,
        features_suffix,
        states_suffix,
        padding_mask,
        view_map,
        embodiment_ids,
        dt,
        loss_weight,
        d,
        rollout_steps,
        pad_to_length,
        apply_view_loss_weights,
    ):
        """Fused single-backward equivalent of the per-step ``rollout`` loop.

        Preconditions (enforced by the ``use_fused`` gate in ``rollout``):
        ``rollout_stop_gradient`` is True, ``d == 1``, ``decode_state`` is False,
        ``run_backward`` is True, ``action_noise == 0``, ``features_suffix`` is not
        None and ``rollout_steps >= 1``. ``loss_weight`` is already divided by
        ``(rollout_steps + 1)`` by the caller (matching the legacy path).

        Strategy:
          1. Generate the AR sequence F_{R-1} once under ``no_grad`` by replaying the
             legacy loop with ``features_suffix=None`` (no loss/backward) for ``R-1``
             steps. This assembles features/actions/states/dt exactly as the per-step
             path would have them entering its last iteration.
          2. Run ONE grad forward over F_{R-1} (== the legacy last iteration's
             forward). By causality the per-frame outputs at positions
             ``[start_pos : start_pos + R]`` equal the ``R`` per-step predictions.
          3. Aggregate the loss (``per_step`` or ``valid``) and call ``backward``
             once. Per-step diagnostics are always recorded so train.py logging is
             unchanged.
        """
        # Clamp defensively to the available targets/mask (legacy implicitly
        # assumes the caller provides enough; in normal operation R == rollout_steps).
        R = min(rollout_steps, features_suffix.shape[1], padding_mask.shape[1])
        start_pos = state.features.shape[1] - 1  # index of the last input frame in F0

        # ── 1. Generation: advance to F_{R-1} under no_grad ────────────────────
        if R > 1:
            # KV-cache generation uses use_cache=True. Under tensor parallelism the
            # cache is head-sharded (predictor.forward allocates it with the per-rank
            # head count) and the SP all-gather hands attention the full sequence, so
            # the cached path composes with TP+SP — provided the per-frame token count
            # N is divisible by tp (so the SP Shard(1) scatter divides evenly without
            # the whole-frame padding that the use_cache path skips). Otherwise fall
            # back to the no-cache full-forward generation.
            tp_mesh = getattr(self.model, "_tp_mesh", None)
            if tp_mesh is not None:
                _, _, _Vg, _Hg, _Wg, _ = state.features.shape
                _N = _Vg * _Hg * _Wg + self.model.action_tokens + self.model.state_tokens
                tp_ok = (_N % int(getattr(self.model, "_tp_size", 1))) == 0
            else:
                tp_ok = True
            use_kv = (
                self.rollout_fused_kv_cache
                and self.model.camera_extrinsics_tokens == 0
                and (self.model.state_tokens == 0 or states_suffix is not None)
                and tp_ok
            )
            if use_kv:
                gen_state = self._rollout_generate_cached(
                    state,
                    actions_suffix=actions_suffix,
                    states_suffix=states_suffix,
                    dt=dt,
                    view_map=view_map,
                    embodiment_ids=embodiment_ids,
                    R=R,
                )
            else:
                # TP-safe / default: replay the legacy loop without loss/backward
                # (full forwards) to advance the state to F_{R-1}.
                with torch.no_grad():
                    _, gen_state = self.rollout(
                        state,
                        actions_suffix=actions_suffix,
                        features_suffix=None,  # skip loss/backward -> pure generation
                        states_suffix=states_suffix,
                        padding_mask=padding_mask,
                        action_noise=0.0,
                        view_map=view_map,
                        embodiment_ids=embodiment_ids,
                        dt=dt,
                        suffix_len=d,
                        run_backward=False,
                        rollout_steps=R - 1,
                        pad_to_length=pad_to_length,
                    )
        else:
            gen_state = state

        # ── 2. Single grad forward over F_{R-1} (mirrors one legacy iteration) ──
        new_action = actions_suffix[:, R - 1 : R]
        features, actions = gen_state.features, gen_state.actions
        dt_prev = gen_state.dt
        poses = gen_state.states
        actions = torch.cat([actions, new_action], dim=1)
        ce = gen_state.camera_extrinsics
        if ce is not None:
            ce = ce.repeat(1, actions.shape[1], 1, 1)

        real_t = features.shape[1]
        if pad_to_length is not None and real_t < pad_to_length:
            _pad = pad_to_length - real_t
            features_fwd = F.pad(features, (0, 0, 0, 0, 0, 0, 0, 0, 0, _pad))
            actions_fwd = F.pad(actions, (0, 0, 0, _pad))
            poses_fwd = F.pad(poses, (0, 0, 0, _pad)) if poses is not None else None
            dt_fwd = F.pad(dt_prev, (0, _pad)) if dt_prev is not None else None
            ce_fwd = F.pad(ce, (0, 0, 0, 0, 0, _pad)) if ce is not None else None
        else:
            features_fwd = features
            actions_fwd = actions
            poses_fwd = poses
            dt_fwd = dt_prev
            ce_fwd = ce

        # The no_grad generation above (and any prior no_grad forward, e.g. the
        # teacher-forcing pass) populates the autocast weight cache with *detached*
        # bf16 casts. If the grad forward below reused those, gradients would never
        # reach the cast (Linear/Conv) params — only the non-cached norm params would
        # get grads. Clear the cache so the grad forward creates fresh grad-linked
        # casts. No-op when autocast is disabled (fp32).
        torch.clear_autocast_cache()

        out, _, _, _ = self.forward(
            features_fwd.detach() if self.rollout_stop_gradient else features_fwd,
            actions_fwd,
            states=poses_fwd.detach() if poses_fwd is not None else None,
            camera_extrinsics=ce_fwd,
            view_map=view_map,
            embodiment_ids=embodiment_ids,
            dt=dt_fwd,
            use_cache=False,
        )
        if pad_to_length is not None and real_t < pad_to_length:
            out = out[:, :real_t]

        # Outputs at [start_pos : start_pos + R] are the R per-step predictions
        # (each predicting features_suffix[:, h]). With d == 1 this is the tail.
        preds_all = out[:, start_pos : start_pos + R]
        assert preds_all.shape[1] == R, f"fused rollout produced {preds_all.shape[1]} preds, expected {R}"
        targets_all = features_suffix[:, 0:R]
        pm_all = padding_mask[:, 0:R]

        # ── 3. Loss aggregation + single backward ──────────────────────────────
        rollout_losses = defaultdict(list)

        def _record(losses_dict):
            for k in losses_dict:
                v = losses_dict[k]
                rollout_losses[k].append(v.detach() if hasattr(v, "detach") else v)

        if self.rollout_loss_weighting == "valid":
            # Pooled normalization: every valid token weighted equally. The (× R)
            # factor makes this coincide with per_step under uniform padding.
            losses_all = self.compute_loss(
                preds_all,
                targets_all,
                None,
                None,
                pm_all,
                shift=0,
                view_map=view_map,
                apply_view_loss_weights=apply_view_loss_weights,
            )
            self.backward(losses_all["loss"] * loss_weight * R)
            # Per-step diagnostics for logging parity (detached, no extra graph).
            with torch.no_grad():
                for h in range(R):
                    _record(
                        self.compute_loss(
                            preds_all[:, h : h + 1].detach(),
                            targets_all[:, h : h + 1],
                            None,
                            None,
                            pm_all[:, h : h + 1],
                            shift=0,
                            view_map=view_map,
                            apply_view_loss_weights=apply_view_loss_weights,
                        )
                    )
        else:  # "per_step" — exact backward-compat: sum of per-step mean-of-means
            total_loss = None
            for h in range(R):
                losses_h = self.compute_loss(
                    preds_all[:, h : h + 1],
                    targets_all[:, h : h + 1],
                    None,
                    None,
                    pm_all[:, h : h + 1],
                    shift=0,
                    view_map=view_map,
                    apply_view_loss_weights=apply_view_loss_weights,
                )
                _record(losses_h)
                total_loss = losses_h["loss"] if total_loss is None else total_loss + losses_h["loss"]
            # Single backward over the summed objective. By linearity this equals
            # the legacy sum of per-step backward(loss_h * loss_weight) calls.
            self.backward(total_loss * loss_weight)

        for k in rollout_losses:
            rollout_losses[k] = torch.tensor(rollout_losses[k], device=preds_all.device)

        # ── Best-effort final state (F_R), for return-contract parity. The
        # training caller ignores it; eval callers never take the fused path.
        final_features = torch.cat([gen_state.features, preds_all[:, -1:].detach()], dim=1)
        poses_final = gen_state.states
        if self.model.state_tokens > 0 and states_suffix is not None and poses_final is not None:
            poses_final = torch.cat([poses_final, states_suffix[:, R - 1 : R]], dim=1)
        dt_final = gen_state.dt
        if dt_final is not None and dt is not None:
            dt_final = torch.cat([dt_final, dt[:, R - 1 : R]], dim=1)
        final_state = WorldModelState(
            features=final_features,
            actions=actions,
            states=poses_final,
            camera_extrinsics=gen_state.camera_extrinsics if self.model.camera_extrinsics_tokens > 0 else None,
            dt=dt_final,
            view_map=view_map,
            embodiment_ids=embodiment_ids,
        )
        return rollout_losses, final_state

    def _inner_predictor(self):
        """Unwrap DDP / torch.compile to the bare predictor module.

        Used for the no_grad KV-cache generation: DDP plays no role without a
        backward, and the compiled program (dynamic=False, in-place KV-cache) is
        not built for the incremental T=1 cache path — so we run generation
        eagerly on the underlying module. Weights are shared, and the generated
        frames are detached context for the (compiled) grad forward, so eager-vs-
        compiled numerical differences here are irrelevant.
        """
        m = self.transition_model
        for _ in range(4):
            if hasattr(m, "_orig_mod"):  # torch.compile OptimizedModule
                m = m._orig_mod
            elif hasattr(m, "module"):  # DistributedDataParallel
                m = m.module
            else:
                break
        return m

    @torch.no_grad()
    def _rollout_generate_cached(self, state, actions_suffix, states_suffix, dt, view_map, embodiment_ids, R):
        """KV-cache equivalent of the no_grad generation phase (d == 1).

        Produces the same context F_{R-1} (features/actions/states/dt) the legacy
        no_grad loop would, but each AR step processes only the new frame using the
        predictor's KV-cache (cache_len == local temporal window, so this is
        faithful to the windowed full forward). Returns a WorldModelState.
        """
        inner = self._inner_predictor()
        use_state = self.model.state_tokens > 0
        use_dt = self.model.use_dt

        feats0 = state.features
        P0 = feats0.shape[1]

        # Priming: process the whole prefix F0 (P0 frames, P0 actions) -> pred_0.
        # A P0-frame prefix carries P0-1 actions in the state; appending the first
        # rollout action gives P0 actions.
        prime_actions = torch.cat([state.actions, actions_suffix[:, 0:1]], dim=1)
        out, _, _, cache = inner(
            feats0,
            prime_actions,
            states=state.states if use_state else None,
            use_cache=True,
            cache=None,
            input_position=0,
            cache_size=0,
            view_map=view_map,
            embodiment_ids=embodiment_ids,
            dt=state.dt if use_dt else None,
        )
        position = P0
        pred = out[:, -1:]
        preds = [pred]

        # Incremental steps: feed only the previous prediction (1 frame). The frame
        # processed at step i is pred_{i-1}; in the legacy full forward it pairs with
        # action actions_suffix[:, i] and state/dt index i-1.
        for i in range(1, R - 1):
            out, _, _, cache = inner(
                pred,
                actions_suffix[:, i : i + 1],
                states=states_suffix[:, i - 1 : i] if use_state else None,
                use_cache=True,
                cache=cache,
                input_position=position,
                cache_size=cache.pointer[0],
                view_map=view_map,
                embodiment_ids=embodiment_ids,
                dt=dt[:, i - 1 : i] if use_dt else None,
            )
            position += 1
            pred = out[:, -1:]
            preds.append(pred)

        features = torch.cat([feats0, *preds], dim=1)
        actions = torch.cat([state.actions, actions_suffix[:, 0 : R - 1]], dim=1)
        states = torch.cat([state.states, states_suffix[:, 0 : R - 1]], dim=1) if use_state else None
        dt_seq = torch.cat([state.dt, dt[:, 0 : R - 1]], dim=1) if use_dt else None
        return WorldModelState(
            features=features,
            actions=actions,
            states=states,
            camera_extrinsics=state.camera_extrinsics,
            dt=dt_seq,
            view_map=view_map,
            embodiment_ids=embodiment_ids,
        )

    @torch.compiler.disable
    def batched_forward(
        self,
        video_features,
        actions,
        states=None,
        camera_extrinsics=None,
        view_map=None,
        embodiment_ids=None,
        dt=None,
        batch=100,
        verbose=False,
        pos_embedding_resize_t=None,
        device="cuda",
    ):
        result = []
        with torch.no_grad():
            for j in range(int(math.ceil(len(actions) / batch))):
                out, _, _, _ = self.forward(
                    video_features[j * batch : (j + 1) * batch].to(device),
                    actions[j * batch : (j + 1) * batch].to(device),
                    states=states[j * batch : (j + 1) * batch].to(device) if states is not None else None,
                    camera_extrinsics=(
                        camera_extrinsics[j * batch : (j + 1) * batch].to(device)
                        if camera_extrinsics is not None
                        else None
                    ),
                    embodiment_ids=(
                        embodiment_ids[j * batch : (j + 1) * batch].to(device) if embodiment_ids is not None else None
                    ),
                    dt=dt[j * batch : (j + 1) * batch].to(device) if dt is not None else None,
                    pos_embedding_resize_t=pos_embedding_resize_t,
                    view_map=view_map[j * batch : (j + 1) * batch].to(device),
                )
                result.append(out[:, -1:].cpu())
        return torch.cat(result, dim=0)

    def compute_loss(
        self,
        predicted_features,
        targets,
        state,
        pred_state,
        padding_mask,
        shift=0,
        view_map=None,
        apply_view_loss_weights=False,
    ):
        B, T, V, H, W, C = predicted_features.shape

        # Keep V as its own axis so per-view loss weighting is a simple broadcast.
        predicted_features = predicted_features.reshape(B, T, V, H * W, C)
        targets = targets.reshape(B, T, V, H * W, C)
        if shift != 0:
            targets_ = targets[:, shift:]
            features = predicted_features[:, :-shift]
            padding_mask = padding_mask[:, :-shift]
            if self.model.decode_state:
                pred_state = pred_state[:, :-shift]
                state = state[:, shift:]
        else:
            targets_ = targets
            features = predicted_features

        cos_loss = -(
            features * targets_ / (features.norm(dim=-1, keepdim=True) * targets_.norm(dim=-1, keepdim=True))
        ).sum(-1)

        if self.model.decode_state:
            state_loss = torch.abs(pred_state - state)
            state_loss_comps = {
                "state_loss_y": state_loss[..., 1].mean(),
                "state_loss_x": state_loss[..., 2].mean(),
                "state_loss_z": state_loss[..., 3].mean(),
                "state_loss_th1": state_loss[..., 4].mean(),
                "state_loss_th2": state_loss[..., 5].mean(),
                "state_loss_th3": state_loss[..., 6].mean(),
                "state_loss_gr": state_loss[..., 0].mean(),
            }
            state_loss = state_loss.mean()
        else:
            state_loss = 0.0
            state_loss_comps = {}

        mae = torch.mean(torch.abs(features - targets_), dim=-1)  # (B, T_eff, V, H*W)

        # Optional per-view L1 weighting. Only mae is weighted (cos_loss and state_loss
        # are left unchanged by design). Opt-in via apply_view_loss_weights to keep all
        # eval / inference call sites (and pre-feature checkpoints) numerically
        # unweighted by default.
        if apply_view_loss_weights and view_map is not None and len(self.view_loss_weights) > 0:
            # view_map: (B, V) of integer semantic view IDs.
            # Special key -1 sets the default weight for any view ID not otherwise
            # listed (e.g. {-1: 0.5, 2: 5.0} -> wrist=5x, everything else 0.5x).
            default_weight = float(self.view_loss_weights.get(-1, 1.0))
            view_weights = torch.full((B, V), default_weight, device=mae.device, dtype=mae.dtype)
            for view_id, weight in self.view_loss_weights.items():
                if int(view_id) == -1:
                    continue
                view_weights = torch.where(
                    view_map == int(view_id),
                    torch.full_like(view_weights, float(weight)),
                    view_weights,
                )
            # Broadcast (B, V) -> (B, 1, V, 1) against mae (B, T_eff, V, H*W).
            mae = mae * view_weights.unsqueeze(1).unsqueeze(-1)

        padding_mask = padding_mask.unsqueeze(-1).unsqueeze(-1)  # (B, T_eff, 1, 1)

        cos_loss = cos_loss * padding_mask
        cos_loss = cos_loss.mean(dim=(-1, -2)).sum() / padding_mask.sum()
        mae = mae * padding_mask
        # Per-view L1 (diagnostic only): keep V axis, then average exactly like
        # the scalar `mae` below — sum over (B, T_eff, H*W) where padding_mask
        # is 1, divide by (#valid (B,T) frames * H*W). Yields a tensor of shape
        # (V,). Exposed as scalar keys `l1_loss_view_<v>` so downstream loops
        # over `losses` (in train.py logging and eval scripts) pick them up
        # automatically. Existing `l1_loss` semantics are unchanged.
        per_view_l1 = mae.sum(dim=(0, 1, -1)) / (padding_mask.sum() * mae.shape[-1])
        mae = mae.mean(dim=(-1, -2)).sum() / padding_mask.sum()
        if self.model.decode_state:
            state_loss = state_loss * padding_mask.squeeze(-1).squeeze(-1)
            state_loss = state_loss.sum() / padding_mask.sum()
        out = {
            "loss": mae + state_loss,
            "cos_loss": cos_loss,
            "l1_loss": mae,
            "state_loss": state_loss,
            **state_loss_comps,
        }
        for v_idx in range(per_view_l1.shape[0]):
            out[f"l1_loss_view_{v_idx}"] = per_view_l1[v_idx]
        return out

    def to(self, device):
        self.transition_model.to(device)
        if self.encoder is not None:
            self.encoder.to(device)
        for head in self.heads.values():
            head.to(device)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")


class WorldModelHead:
    def __init__(self, head_config):
        pass
