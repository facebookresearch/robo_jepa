# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP2 sharding helpers for the robo_jepa world-model training stack.

This module is the single integration point for FSDP2 (the composable
`torch.distributed._composable.fsdp.fully_shard` API).  Everything here is
opt-in: the legacy DDP path in `train.py` stays the default.

Sharding layout (per the FSDP2 plan):
  - HSDP 2-D mesh of shape (num_nodes, gpus_per_node) with mesh dim names
    ("replicate", "shard").  Within a node, params/grads/optimizer are
    sharded; across nodes, replicated.  This matches one shard group per
    node (typically 8 H200), which is what the 8B predictor needs to fit.
  - Per-`Block` `fully_shard` inside `predictor_blocks` plus a single outer
    `fully_shard` on the top-level transition_model (or head.model).  Same
    treatment for each head.
  - Optional activation checkpointing via `checkpoint_wrapper` *replacing*
    the in-forward `torch.utils.checkpoint` branch in `predictor.py` (we
    flip `module.use_activation_checkpointing = False` after wrapping so
    the per-Block recompute is not done twice).
  - Mixed precision: bf16 params/comp, fp32 reduce-scatter — matches the
    current `meta.dtype: bfloat16, mixed_precision: true` numerical
    posture.
  - Optional CPU offload (off by default) for future >8B scaling.

Checkpointing:
  - Save: torch.distributed.checkpoint (DCP) directory.  Atomic-rename
    pattern at the directory level, with a `.prev/` backup.  Optimizer
    state is sharded on disk; epoch / scaler / stats live in a small
    rank-0 `metadata.pt` inside the same dir.
  - Load auto-detects format: a directory uses DCP; a `.pth.tar` file
    falls back to a legacy load (rank-0 torch.load + broadcast via
    `set_model_state_dict(broadcast_from_rank0=True, full_state_dict=True)`).
    This lets new FSDP2 runs warm-start from any prod `.pth.tar`.

The module deliberately does NOT depend on app.robo_jepa.* models so it
stays cheap to import and unit-test.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #


def make_fsdp_mesh(
    world_size: int,
    gpus_per_node: int,
    layout: str = "hsdp",
) -> DeviceMesh:
    """Build the device mesh that FSDP2 shards over.

    layout="hsdp"        -> 2-D mesh ("replicate"=num_nodes, "shard"=gpus_per_node)
    layout="shard_only"  -> 1-D mesh ("shard"=world_size)            [pure FSDP]
    layout="replicate_only" -> 1-D mesh ("replicate"=world_size)     [pure DDP-ish]

    The HSDP layout matches the prod cluster topology (8 H200 per node) and
    is what the FSDP plan asks for: shard intra-node, replicate cross-node.
    """
    assert world_size > 0 and gpus_per_node > 0, f"bad mesh dims world_size={world_size} gpus_per_node={gpus_per_node}"
    if layout == "hsdp":
        if world_size % gpus_per_node != 0:
            raise ValueError(
                f"HSDP requires world_size divisible by gpus_per_node ({world_size=} % {gpus_per_node=} != 0)"
            )
        num_nodes = world_size // gpus_per_node
        mesh = init_device_mesh(
            "cuda",
            (num_nodes, gpus_per_node),
            mesh_dim_names=("replicate", "shard"),
        )
    elif layout == "shard_only":
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))
    elif layout == "replicate_only":
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("replicate",))
    else:
        raise ValueError(f"Unknown FSDP mesh layout: {layout!r}")
    logger.info(f"[fsdp] built mesh layout={layout!r} shape={tuple(mesh.shape)}")
    return mesh


# --------------------------------------------------------------------------- #
# fully_shard application
# --------------------------------------------------------------------------- #


def _select_shard_mesh(mesh: DeviceMesh) -> DeviceMesh:
    """Pick the sub-mesh that fully_shard should shard over.

    For HSDP we shard along the 'shard' dim and rely on FSDP2 to detect the
    'replicate' dim from the parent mesh and emit all-reduce on it (HSDP
    semantics).  Passing the full 2-D mesh to fully_shard is the supported
    way to opt into HSDP — fully_shard inspects mesh_dim_names and treats
    a 'replicate' dim as the cross-node replicate group.
    """
    return mesh


def apply_fsdp2(
    module: nn.Module,
    mesh: DeviceMesh,
    *,
    mp_policy: MixedPrecisionPolicy | None = None,
    use_activation_checkpointing: bool = False,
    cpu_offload: bool = False,
    blocks_attr: str = "predictor_blocks",
) -> nn.Module:
    """Apply FSDP2 to `module` in place: per-Block + outer wrap.

    - If `module` has an `nn.ModuleList` named `blocks_attr` (default
      "predictor_blocks"), each Block is wrapped individually so that
      gradient reduce-scatter overlaps with the next Block's all-gather.
    - When `use_activation_checkpointing` is True, each Block is replaced
      by `checkpoint_wrapper(block, NO_REENTRANT)` *before* `fully_shard`
      so the standard FSDP2+AC composition applies (recompute happens
      after the all-gather, on the unsharded params).  We also flip
      `module.use_activation_checkpointing = False` so the in-forward
      branch in `predictor.py` becomes a pass-through (no double recompute).
    - The outer module is then wrapped with a single `fully_shard` so any
      top-level params (view_tokens, view_assignment_tokens, embeddings,
      embed/proj, norms, prop encoders) are also sharded.

    Returns the same `module` object (mutated in place).
    """
    offload = CPUOffloadPolicy() if cpu_offload else None
    fsdp_kwargs: dict = {"mesh": _select_shard_mesh(mesh)}
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy
    if offload is not None:
        fsdp_kwargs["offload_policy"] = offload

    blocks = getattr(module, blocks_attr, None)
    n_wrapped_blocks = 0
    if blocks is not None and isinstance(blocks, nn.ModuleList):
        for i, block in enumerate(blocks):
            wrapped = block
            if use_activation_checkpointing:
                wrapped = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            fully_shard(wrapped, **fsdp_kwargs)
            blocks[i] = wrapped
            n_wrapped_blocks += 1
    else:
        logger.warning(
            f"[fsdp] module {type(module).__name__} has no '{blocks_attr}' "
            f"ModuleList — skipping per-Block wrap, only outer wrap will run."
        )

    # Outer wrap: covers top-level params not consumed by the per-block wraps.
    fully_shard(module, **fsdp_kwargs)

    # Disable in-forward AC branch so we don't double-checkpoint.  The
    # predictor's forward at robo_jepa/models/predictor.py:546 reads this attr at call time, so flipping it here is enough.
    if use_activation_checkpointing and hasattr(module, "use_activation_checkpointing"):
        module.use_activation_checkpointing = False

    # Stash the mesh + a "first-replicate-slot" save sub-PG on the module so
    # save_dcp_checkpoint can restrict the DCP write to one node's worth of
    # ranks under HSDP (otherwise both replicate slots race on the same
    # checkpoint dir and produce duplicate shards).  dist.new_group is
    # collective — it must be called by EVERY rank, even non-members — so we
    # build it here once at wrap time and cache.
    save_pg, is_saver, n_savers = _build_first_node_save_pg(mesh)
    module._fsdp2_mesh = mesh
    module._fsdp2_save_pg = save_pg
    module._fsdp2_is_saver = is_saver
    module._fsdp2_num_savers = n_savers

    logger.info(
        f"[fsdp] wrapped {type(module).__name__}: blocks={n_wrapped_blocks} "
        f"AC={use_activation_checkpointing} cpu_offload={cpu_offload} "
        f"savers_per_ckpt={n_savers}"
    )
    return module


def _build_first_node_save_pg(
    mesh: DeviceMesh,
) -> tuple[dist.ProcessGroup | None, bool, int]:
    """Build the process group used for HSDP-aware DCP save.

    Under HSDP the model parameters are *replicated* across the 'replicate'
    dim (typically across nodes) and *sharded* along the 'shard' dim (the 8
    GPUs of a node).  All replicate slots hold identical data, so the
    checkpoint should be written by exactly one of them — otherwise N
    replicate slots race on the same dir, producing N copies of every shard
    (or worse, half-overwriting each other).

    For HSDP this returns a sub-PG containing only the first replicate
    slot's ranks (i.e. the first node).  For 1-D meshes (pure FSDP / pure
    DP) this returns the world PG.

    `dist.new_group` is itself a collective: every rank in the world must
    call it (even non-members), so this MUST be invoked by all ranks at
    the same point in execution.

    Returns (process_group, is_member, num_savers).
    """
    has_replicate = (
        mesh.mesh_dim_names is not None and "replicate" in mesh.mesh_dim_names and mesh["replicate"].size() > 1
    )
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if not has_replicate:
        return None, True, world_size

    gpus_per_node = mesh["shard"].size()
    first_node_ranks = list(range(gpus_per_node))
    pg = dist.new_group(ranks=first_node_ranks)
    is_member = dist.get_rank() in first_node_ranks
    return pg, is_member, gpus_per_node


# --------------------------------------------------------------------------- #
# FSDP2 detection + no_sync
# --------------------------------------------------------------------------- #


def is_fsdp2_module(module: nn.Module) -> bool:
    """Detect whether `module` was wrapped with `fully_shard`.

    fully_shard installs a `set_requires_gradient_sync` method on the
    module via composable state — checking for that attribute is the
    documented detection idiom.
    """
    return hasattr(module, "set_requires_gradient_sync")


@contextlib.contextmanager
def fsdp_or_ddp_no_sync(module: nn.Module):
    """Context manager that disables gradient sync inside the with-block.

    - DDP module: enters the legacy `module.no_sync()` context manager.
    - FSDP2 module: toggles `set_requires_gradient_sync(False)` on enter
      and restores `True` on exit.
    - Plain (unwrapped) module: no-op.

    This lets the microbatch / grad-accum loop in train.py use a single
    helper for both wrap kinds.
    """
    if is_fsdp2_module(module):
        module.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            module.set_requires_gradient_sync(True)
    elif hasattr(module, "no_sync"):
        # DistributedDataParallel
        with module.no_sync():
            yield
    else:
        # Plain unwrapped module (e.g. single-process debug runs).
        yield


# --------------------------------------------------------------------------- #
# DCP / legacy checkpoint helpers
# --------------------------------------------------------------------------- #


_PTH_TAR_SUFFIX = ".pth.tar"


def _to_dcp_dir(path: str) -> str:
    """Normalize a checkpoint path to a directory path.

    The training pipeline historically uses `<folder>/world_model-latest.pth.tar`.
    For DCP we want a directory; strip the suffix if present.  This is
    safer than `str.strip(".pth.tar")`, which would also remove leading/
    trailing characters in the set.
    """
    if path.endswith(_PTH_TAR_SUFFIX):
        return path[: -len(_PTH_TAR_SUFFIX)]
    return path


def _atomic_replace_dir(tmp_dir: str, dest_dir: str, backup_dir: str) -> None:
    """Atomically replace `dest_dir` with `tmp_dir`, keeping the previous
    contents at `backup_dir`.  Mirrors the .pth.tar/.prev pattern used
    in `TrainableModel.save_checkpoint`."""
    if os.path.exists(dest_dir):
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
        os.rename(dest_dir, backup_dir)
    os.rename(tmp_dir, dest_dir)


def _looks_like_dcp_dir(path: str) -> bool:
    """Heuristic: a DCP checkpoint is a directory containing a `.metadata`
    file (DCP writes `.metadata` at the root of the checkpoint dir)."""
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, ".metadata")) or os.path.exists(os.path.join(path, "metadata.pt"))
    )


def _is_legacy_checkpoint(path: str) -> bool:
    """A legacy checkpoint is a regular file ending in `.pth.tar`."""
    return os.path.isfile(path) and path.endswith(_PTH_TAR_SUFFIX)


def save_dcp_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    epoch: int,
    stats,
    path: str,
    rank: int = 0,
) -> None:
    """Write an FSDP2-aware checkpoint to a directory.

    Layout under `<dir>/`:
      - `.metadata`, `__N_M.distcp` ...   (DCP writer output, sharded across
                                            the FIRST node's ranks only)
      - `metadata.pt`                     (rank-0 only: epoch, scaler, stats)

    Atomicity / crash safety:
      Rank 0 moves any existing `<dir>/` to `<dir>.prev/` before the new
      save begins, so a crash mid-save leaves the previous good ckpt in
      `<dir>.prev/` (the partially-written `<dir>/` is still on disk and
      can be inspected).  No `<dir>.tmp/` intermediate is used — that
      pattern raced under HSDP because every rank tried to manage the dir.

    HSDP semantics:
      Under HSDP the FSDP-wrapped module has param replicas in N replicate
      slots (typically N nodes) holding identical sharded data.  Only the
      FIRST replicate slot writes the checkpoint; the other slots are
      silent on disk and just synchronize via the post-save barrier.  The
      sub-process-group that does the write is built once at FSDP wrap
      time (apply_fsdp2 -> _build_first_node_save_pg) and stashed on the
      module as `_fsdp2_save_pg`.

    Layout-agnostic:
      For 1-D meshes (pure FSDP / pure DP) every rank participates in the
      DCP write via the world PG.  For non-FSDP modules this function
      should not be called (TrainableModel.save_checkpoint dispatches via
      `is_fsdp2_module`).
    """
    dest_dir = _to_dcp_dir(path)
    backup_dir = dest_dir + ".prev"

    # Rank 0: prep destination dir.  Move any existing ckpt out of the way.
    # This is rank-0-only — no other rank ever touches the filesystem here.
    if rank == 0:
        if os.path.isdir(dest_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
            os.rename(dest_dir, backup_dir)
        os.makedirs(dest_dir, exist_ok=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Build the sharded state dict on every rank.  This is local-only; it
    # collects DTensor refs without comm.
    sd_options = StateDictOptions(full_state_dict=False)
    model_sd = get_model_state_dict(model, options=sd_options)
    state_dict: dict = {"model": model_sd}
    if optimizer is not None:
        optim_sd = get_optimizer_state_dict(model, optimizer, options=sd_options)
        state_dict["optim"] = optim_sd

    # Restrict the DCP collective to the first replicate slot (HSDP) — see
    # _build_first_node_save_pg.  For 1-D meshes save_pg is None and every
    # rank is a saver.
    save_pg = getattr(model, "_fsdp2_save_pg", None)
    is_saver = bool(getattr(model, "_fsdp2_is_saver", True))
    n_savers = int(
        getattr(
            model,
            "_fsdp2_num_savers",
            dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
        )
    )

    if is_saver:
        logger.info(f"[fsdp] saving DCP checkpoint to {dest_dir} (epoch={epoch}, savers={n_savers})...")
        dcp.save(state_dict, checkpoint_id=dest_dir, process_group=save_pg)

    # Wait for the (first-node) DCP write to land before rank 0 writes the
    # metadata.pt sidecar and before non-saver ranks proceed.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        meta = {
            "epoch": int(epoch),
            "scaler": None if scaler is None else scaler.state_dict(),
            "stats": stats,
        }
        torch.save(meta, os.path.join(dest_dir, "metadata.pt"))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        logger.info(f"[fsdp] DCP checkpoint committed at {dest_dir}")


def _load_legacy_full_to_sharded(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    *,
    load_optimizer: bool,
    strict: bool,
    dont_load: Iterable[str] | None,
) -> tuple[int, dict]:
    """Load a legacy `.pth.tar` checkpoint into an FSDP2-wrapped model.

    rank-0 loads the file (CPU mmap), drops `module.` prefixes and
    `dont_load` keys, then `set_model_state_dict` broadcasts and
    re-shards into DTensor placements on every rank.  Optimizer state
    is loaded the same way when present and requested.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    if rank == 0:
        checkpoint = torch.load(path, map_location="cpu", mmap=True)
    else:
        checkpoint = None

    # Broadcast the epoch / stats / scaler / model_sd / optim_sd via
    # set_model_state_dict (which broadcasts the model_sd), and a tiny
    # `dist.broadcast_object_list` for the scalar metadata.
    epoch = 0
    stats = {}
    if rank == 0:
        epoch = int(checkpoint.get("epoch", 0))
        stats = checkpoint.get("stats", {}) or {}
    if dist.is_available() and dist.is_initialized():
        meta_obj = [epoch, stats]
        dist.broadcast_object_list(meta_obj, src=0)
        epoch, stats = meta_obj[0], meta_obj[1]

    # --- model ---
    model_sd = checkpoint["model"] if rank == 0 else {}
    if rank == 0:
        # Strip legacy `module.` prefix introduced by older DDP saves.
        model_sd = {k.replace("module.", ""): v for k, v in model_sd.items()}
        if dont_load:
            for k in dont_load:
                model_sd.pop(k, None)

    set_model_state_dict(
        model,
        model_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=strict,
        ),
    )

    # --- optimizer ---
    if optimizer is not None and load_optimizer and rank == 0 and checkpoint.get("opt"):
        optim_sd = checkpoint["opt"]
    else:
        optim_sd = {} if rank == 0 else {}

    if optimizer is not None and load_optimizer:
        set_optimizer_state_dict(
            model,
            optimizer,
            optim_sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=strict,
            ),
        )

    logger.info(f"[fsdp] loaded legacy ckpt {path} (epoch={epoch}) into FSDP2 model")
    return epoch, stats


def _load_dcp_to_sharded(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    *,
    load_optimizer: bool,
    strict: bool,
) -> tuple[int, dict]:
    """Load a DCP-format checkpoint into an FSDP2-wrapped model."""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    # Build empty sharded state dicts as targets so DCP knows the
    # expected DTensor placements per rank.
    sd_options = StateDictOptions(full_state_dict=False)
    model_sd = get_model_state_dict(model, options=sd_options)
    state_dict: dict = {"model": model_sd}
    if optimizer is not None and load_optimizer:
        optim_sd = get_optimizer_state_dict(model, optimizer, options=sd_options)
        state_dict["optim"] = optim_sd

    logger.info(f"[fsdp] loading DCP checkpoint from {path}...")
    dcp.load(state_dict, checkpoint_id=path)

    set_model_state_dict(
        model,
        state_dict["model"],
        options=StateDictOptions(full_state_dict=False, strict=strict),
    )
    if optimizer is not None and load_optimizer and "optim" in state_dict:
        set_optimizer_state_dict(
            model,
            optimizer,
            state_dict["optim"],
            options=StateDictOptions(full_state_dict=False, strict=strict),
        )

    # Read the small metadata file (epoch / scaler / stats) on every rank.
    meta_path = os.path.join(path, "metadata.pt")
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu")
        epoch = int(meta.get("epoch", 0))
        stats = meta.get("stats", {}) or {}
    else:
        epoch, stats = 0, {}
    logger.info(f"[fsdp] loaded DCP ckpt {path} (epoch={epoch})")
    return epoch, stats


def load_dcp_or_legacy_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    *,
    load_optimizer: bool = True,
    strict: bool = True,
    dont_load: Iterable[str] | None = None,
) -> tuple[int, dict]:
    """Auto-detect the checkpoint format at `path` and load it.

    Resolution order:
      1. If the literal `path` is a DCP directory, load from it.
      2. If `path.endswith(".pth.tar")`, treat as legacy file.
      3. Else, also try `_to_dcp_dir(path)` (i.e. strip `.pth.tar`) — this
         lets callers pass the legacy-style `.pth.tar` path even when the
         on-disk artifact is a DCP directory at the suffix-stripped name.
    Returns (epoch, stats) on success.  Raises FileNotFoundError if
    nothing is found.
    """
    # 1. Literal path is a DCP dir.
    if os.path.isdir(path) and _looks_like_dcp_dir(path):
        return _load_dcp_to_sharded(
            model,
            optimizer,
            path,
            load_optimizer=load_optimizer,
            strict=strict,
        )
    # 2. Legacy .pth.tar file.
    if _is_legacy_checkpoint(path):
        return _load_legacy_full_to_sharded(
            model,
            optimizer,
            path,
            load_optimizer=load_optimizer,
            strict=strict,
            dont_load=dont_load,
        )
    # 3. Try the suffix-stripped DCP directory name.
    candidate_dir = _to_dcp_dir(path)
    if candidate_dir != path and os.path.isdir(candidate_dir) and _looks_like_dcp_dir(candidate_dir):
        return _load_dcp_to_sharded(
            model,
            optimizer,
            candidate_dir,
            load_optimizer=load_optimizer,
            strict=strict,
        )
    raise FileNotFoundError(f"[fsdp] no DCP dir or legacy .pth.tar found at {path!r} (also tried {candidate_dir!r})")


# --------------------------------------------------------------------------- #
# Convenience: build the standard mp policy used by robo_jepa
# --------------------------------------------------------------------------- #


def default_mp_policy(dtype: torch.dtype) -> MixedPrecisionPolicy:
    """bf16 params/comp, fp32 reduce-scatter — matches train.py's existing
    `cfgs_meta.dtype: bfloat16, mixed_precision: true` posture."""
    if dtype == torch.bfloat16:
        return MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    if dtype == torch.float16:
        return MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
    return MixedPrecisionPolicy()  # fp32 throughout
