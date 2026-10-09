# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor + Sequence Parallelism (TP+SP) for the robo_jepa predictor.

Megatron-style TP+SP applied to ``VisionTransformerPredictor`` so a group of
GPUs collaborates on ONE sample, cutting the per-sample forward+backward latency
of the v5 cooldown's gating cell (two-view, high-res, batch-size-1). The DP
(gradient-reduction) dimension is handled separately by DDP over the ``dp`` mesh
dim; TP shards parameters/activations within the ``tp`` mesh dim.

Design (mirrors the torchtitan TP+SP recipe, adapted to the custom
``RoPEAttention`` whose RoPE + xformers ``fmha`` kernel only accept plain
tensors):

* Per ``Block``: ``norm1``/``norm2`` -> ``SequenceParallel`` (activations sharded
  on the sequence dim). ``attn``/``mlp`` get a ``PrepareModuleInput`` that
  all-gathers the sequence back to full length (``use_local_output=True`` so the
  custom attention/MLP run on plain tensors). ``qkv``/``q``/``kv``/``fc1``/``fc2``
  are column-parallel; ``proj``/``fc3`` are row-parallel with ``output_layouts=
  Shard(1)`` so the row-reduce becomes a reduce-scatter back to sequence-sharded.
* Each ``attn``'s ``num_heads``/``num_kv_heads`` are divided by the TP degree so
  the local reshape sees ``num_heads/tp`` heads (head_dim is unchanged).
* The predictor's residual stream is scattered to ``Shard(1)`` at the block-stack
  entry and gathered back to full length before ``predictor_norm`` — done inside
  ``VisionTransformerPredictor.forward`` guarded by ``self._tp_mesh``.

Because SP all-gathers the sequence before attention, the space-time RoPE and the
block-sparse local-window mask run over the full sequence unchanged — no RoPE or
mask surgery is required.
"""

from __future__ import annotations

import logging
import os
import shutil

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

logger = logging.getLogger(__name__)


def make_parallel_mesh(
    world_size: int,
    gpus_per_node: int,
    tp_size: int,
    *,
    fsdp_enabled: bool = False,
    fsdp_layout: str = "hsdp",
    device_type: str = "cuda",
) -> DeviceMesh:
    """Build the parallelism ``DeviceMesh``.

    * ``fsdp_enabled=False`` (default): 2-D ``(dp, tp)`` mesh. ``tp`` is the inner
      (intra-node, NVLink) dim; ``dp`` is the gradient-reduction dim handled by
      DDP over ``mesh["dp"]``. Parameters are NOT sharded on ``dp`` (pure
      replicate) — TP alone provides the memory relief.
    * ``fsdp_enabled=True``: 3-D ``(dp_replicate, dp_shard, tp)`` mesh so FSDP2
      can additionally shard parameters on ``dp_shard`` while TP shards on ``tp``.
      ``dp_shard`` is the remaining intra-node degree after TP
      (``gpus_per_node // tp_size``); ``dp_replicate`` spans nodes.

    ``tp`` is always the last (fastest-varying) dim so a TP group is a contiguous
    block of ranks within a node.
    """
    if world_size % tp_size != 0:
        raise ValueError(f"world_size={world_size} not divisible by tp_size={tp_size}")
    if gpus_per_node % tp_size != 0:
        raise ValueError(f"gpus_per_node={gpus_per_node} not divisible by tp_size={tp_size}")

    if not fsdp_enabled:
        dp_size = world_size // tp_size
        mesh = init_device_mesh(
            device_type,
            (dp_size, tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        logger.info(f"[tp] built 2-D mesh (dp={dp_size}, tp={tp_size})")
        return mesh

    if fsdp_layout != "hsdp":
        raise ValueError(f"fsdp+tp only supports hsdp layout, got {fsdp_layout!r}")
    num_nodes = world_size // gpus_per_node
    dp_shard = gpus_per_node // tp_size
    mesh = init_device_mesh(
        device_type,
        (num_nodes, dp_shard, tp_size),
        mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
    )
    logger.info(f"[tp] built 3-D mesh (dp_replicate={num_nodes}, dp_shard={dp_shard}, tp={tp_size})")
    return mesh


def _block_parallelize_plan(block: nn.Module) -> dict:
    """Per-``Block`` TP+SP plan for ``parallelize_module``.

    ``attn``/``mlp`` use ``PrepareModuleInput(use_local_output=True)`` so the
    sequence is all-gathered to full length and handed to the custom
    attention/MLP as a plain tensor (the RoPE + ``fmha`` kernel reject DTensor).
    Only the first positional arg (the hidden state) is transformed; the attn
    kwargs (attn_mask, runtime_grid_*, view_ids, ...) pass through untouched.
    """
    plan: dict = {
        "norm1": SequenceParallel(),
        "norm2": SequenceParallel(),
        "attn": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,
        ),
        # use_local_output=False keeps the residual stream as a DTensor sharded
        # on the sequence dim, so uneven sequence lengths (T*N not divisible by
        # tp, which happens during autoregressive rollout) shard correctly.
        "attn.proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "mlp": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,
        ),
    }

    attn = block.attn
    if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
        plan["attn.qkv"] = ColwiseParallel()
    else:  # grouped-query attention: separate q / kv projections
        plan["attn.q"] = ColwiseParallel()
        plan["attn.kv"] = ColwiseParallel()

    mlp = block.mlp
    if hasattr(mlp, "fc3"):  # SwiGLUFFN: fc1 + fc2 gate (both column), fc3 (row)
        plan["mlp.fc1"] = ColwiseParallel()
        plan["mlp.fc2"] = ColwiseParallel()
        plan["mlp.fc3"] = RowwiseParallel(output_layouts=Shard(1), use_local_output=False)
    else:  # plain MLP: fc1 (column), fc2 (row)
        plan["mlp.fc1"] = ColwiseParallel()
        plan["mlp.fc2"] = RowwiseParallel(output_layouts=Shard(1), use_local_output=False)

    return plan


@torch.no_grad()
def _repack_fused_to_head_major(linear: nn.Linear, groups: int, n_heads: int, head_dim: int) -> None:
    """Permute a fused projection's output rows from component-major
    ``(groups, n_heads, head_dim)`` to head-major ``(n_heads, groups, head_dim)``,
    in place and value-preserving.

    The fused ``qkv`` (groups=3) and ``kv`` (groups=2) projections store rows
    grouped by component first (all Q heads, then all K, then all V). A
    contiguous ``ColwiseParallel`` shard of that layout would split *components*
    across ranks (rank0=all-K, rank1=all-V) instead of splitting *heads*.
    Re-packing to head-major makes a contiguous shard equal a per-head shard.
    The matching head-major reshape in ``RoPEAttention.forward`` is gated on
    ``attn._tp_head_major``.
    """
    out = groups * n_heads * head_dim
    if linear.weight.shape[0] != out:
        raise ValueError(f"unexpected fused proj out={linear.weight.shape[0]} != groups*n_heads*head_dim={out}")
    w = linear.weight.data.view(groups, n_heads, head_dim, linear.weight.shape[1])
    linear.weight.data = w.permute(1, 0, 2, 3).reshape(out, -1).contiguous()
    if linear.bias is not None:
        b = linear.bias.data.view(groups, n_heads, head_dim)
        linear.bias.data = b.permute(1, 0, 2).reshape(out).contiguous()


def _unrepack_head_major_to_fused(t: torch.Tensor, groups: int, n_heads: int, head_dim: int) -> torch.Tensor:
    """Inverse of ``_repack_fused_to_head_major`` for a full (gathered) tensor.

    Maps rows from head-major ``(n_heads, groups, head_dim)`` back to the
    standard component-major ``(groups, n_heads, head_dim)`` so a saved
    checkpoint matches the non-TP / pretrain layout. The permutation swaps the
    leading two dims and is its own structural inverse.
    """
    out = groups * n_heads * head_dim
    if t.shape[0] != out:
        raise ValueError(f"unexpected fused proj rows={t.shape[0]} != groups*n_heads*head_dim={out}")
    if t.dim() == 2:  # weight [out, in]
        return t.view(n_heads, groups, head_dim, t.shape[1]).permute(1, 0, 2, 3).reshape(out, -1).contiguous()
    return t.view(n_heads, groups, head_dim).permute(1, 0, 2).reshape(out).contiguous()  # bias [out]


def tp_full_state_dict(predictor: nn.Module) -> dict:
    """Gather a TP predictor's (replicate-wrapped) params to a COMPLETE, full,
    component-major state dict — byte-compatible with a non-TP model / the
    pretrain checkpoint.

    Uses ``get_model_state_dict(full_state_dict=True)`` rather than a raw
    ``module.state_dict()``: at scale the composable ``replicate`` / DDP reducer
    buckets the parameters, which silently drops the TP DTensor params out of a
    plain ``state_dict()`` (observed at dp=224 — the checkpoint then contained
    only the ~non-parallelized params). ``get_model_state_dict`` reconstructs the
    full params from the distributed state. We then invert the head-major qkv/kv
    repack, and assert the param count matches the pre-shard total so a broken
    gather raises instead of writing a corrupt checkpoint. Collective: every rank
    must call it in the same order.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    full = dict(get_model_state_dict(predictor, options=StateDictOptions(full_state_dict=True, cpu_offload=True)))
    for path, groups, n_heads, head_dim in getattr(predictor, "_tp_repack_info", []):
        for suffix in ("weight", "bias"):
            key = f"{path}.{suffix}"
            if full.get(key) is not None:
                full[key] = _unrepack_head_major_to_fused(full[key], groups, n_heads, head_dim)
    expected = getattr(predictor, "_tp_full_numel", None)
    got = sum(v.numel() for v in full.values() if hasattr(v, "numel"))
    # get_model_state_dict(full_state_dict=True) materializes the full dict on
    # rank 0 only (others get an empty dict by design — they don't write). So the
    # completeness check is meaningful ONLY on rank 0: there `got` must equal the
    # pre-shard total (catches the reducer-dropped-params failure where rank 0
    # saw a partial dict). Non-rank-0 legitimately have got==0.
    import torch.distributed as dist

    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    if is_rank0 and expected is not None and got != expected:
        raise RuntimeError(
            f"TP checkpoint gather is INCOMPLETE on rank 0: got {got} params, expected {expected}. "
            "Refusing to save a corrupt checkpoint (the parallelized DTensor params were dropped)."
        )
    return full


def tp_full_optim_state_dict(model: nn.Module, optimizer) -> dict:
    """Gather the optimizer state (Adam m/v, sharded as tp DTensors) to a FULL
    optimizer state dict on rank 0 — so a TP checkpoint can be optimizer-resumed.
    Collective: every rank must call it. The state is kept in its native
    (head-major) layout: it's only ever reloaded into the same TP optimizer (the
    resume re-runs apply_tp_sp, so head-major is symmetric), never into a plain
    model, so no un-repack is needed (unlike the weights, which eval loads plain).

    We DON'T use ``get_optimizer_state_dict(full_state_dict=True, cpu_offload=True)``
    for the gather: at 8B that torch path (``_gather_state_dict`` → ``.cpu()``
    offload of the full exp_avg/exp_avg_sq) native-segfaults on rank 0. Instead we
    fetch the FQN-keyed *sharded* state (DTensor values, no gather) and all-gather
    each moment ourselves via ``DTensor.full_tensor()`` over the tp mesh — a plain
    4-way all-gather — then offload to CPU. The result is byte-identical to what
    ``full_state_dict=True`` returns (FQN-keyed dict of full plain tensors), so the
    existing :func:`tp_load_optim_state_dict` (``set_optimizer_state_dict`` with
    ``broadcast_from_rank0=True``) reshards it unchanged.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
    from torch.distributed.tensor import DTensor
    import torch.distributed as dist

    # full_state_dict=False → per-rank sharded DTensors, NO cross-rank gather here.
    sharded = get_optimizer_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=False)
    )

    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

    def _gather(obj):
        # DTensor.full_tensor() is a collective (all-gather over the DTensor's
        # mesh); every rank must reach each call in the same order. The tp
        # sharding is symmetric across ranks and dict/list traversal is
        # deterministic, so all ranks issue the identical sequence of collectives.
        # Only rank 0 KEEPS the gathered full tensor; the other ranks still
        # participate in the collective but immediately discard the result. This
        # bounds peak host memory to ~1× the full optimizer (on rank 0) instead of
        # world_size× (materializing the full 8B Adam state redundantly on every
        # rank caused a host-memory crash during the save), and matches the load
        # contract: tp_load_optim_state_dict expects the full dict on rank 0 and
        # {} on every other rank.
        if isinstance(obj, DTensor):
            full = obj.full_tensor()
            return full.cpu() if is_rank0 else None
        if torch.is_tensor(obj):
            return obj.detach().cpu() if is_rank0 else None
        if isinstance(obj, dict):
            return {k: _gather(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_gather(v) for v in obj)
        return obj

    gathered = _gather(sharded)
    return gathered if is_rank0 else {}


def tp_load_optim_state_dict(model: nn.Module, optimizer, full_optim_sd: dict) -> None:
    """Load a FULL optimizer state dict (from rank 0) back into the tp-sharded
    optimizer, re-sharding to each rank. Collective: rank 0 passes the full dict
    (read from the checkpoint), other ranks pass ``{}`` and receive it via
    broadcast_from_rank0. Must be called AFTER init_opt + apply_tp_sp so the
    optimizer's params/layout match what was saved.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

    set_optimizer_state_dict(
        model, optimizer, optim_state_dict=full_optim_sd,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )


def _tp_optim_dir(path: str) -> str:
    """DCP directory for the tp-sharded optimizer that sits next to the plain
    model ``.pth.tar``. E.g. ``.../world_model-latest.pth.tar`` ->
    ``.../world_model-latest_optim``."""
    from app.robo_jepa.fsdp_utils import _to_dcp_dir

    return _to_dcp_dir(path) + "_optim"


def tp_save_optim_dcp(model: nn.Module, optimizer, path: str, rank: int = 0) -> None:
    """Save the tp-sharded optimizer state as a DCP directory — NO all-gather.

    The full-optimizer all-gather (``tp_full_optim_state_dict`` /
    ``get_optimizer_state_dict(full_state_dict=True)``) native-segfaults inside
    the NCCL all-gather at 8B, so we instead write each rank's LOCAL shard via
    ``torch.distributed.checkpoint`` — the same mechanism the FSDP2 path uses
    (proven robust at 8B). Directory: ``<path w/o .pth.tar>_optim/``.

    TP + DDP-replicate: the params/optimizer are TP-sharded (DTensors on the tp
    submesh) and replicated across the dp dim (each dp slot is an identical
    tp-sharded copy). Only the FIRST dp slot's tp group writes — mirroring the
    HSDP first-replicate-slot save (``save_dcp_checkpoint``). The dp slot is
    ``global_rank // tp_size``; slot 0 is global ranks ``[0, tp_size)``. Other dp
    slots skip the write and just barrier. Collective: EVERY rank must call this
    (for the barriers); only slot-0 ranks issue the DCP write over the tp PG.
    """
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    if optimizer is None:
        return

    dest_dir = _tp_optim_dir(path)
    tmp_dir = dest_dir + ".tmp"
    backup_dir = dest_dir + ".prev"

    # ATOMIC save: write the new checkpoint into a fresh `_optim.tmp/`, then
    # promote it with rename(s). `dcp.save` writes in place, so writing directly
    # into `dest_dir` (the old behaviour) left it half-written and MISSING its
    # `.metadata` if the job was killed mid-write — an unresumable corrupt ckpt.
    # With tmp+rename, `dest_dir` is only ever a COMPLETE directory: a crash
    # during the write only affects `_optim.tmp/`, leaving the last good ckpt in
    # `dest_dir` (and the one before in `.prev`).
    if rank == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Local sharded optimizer state (DTensor refs) — collected without comm.
    optim_sd = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=False))

    tp_mesh = getattr(model, "_tp_mesh", None)
    tp_size = int(getattr(model, "_tp_size", 1))
    global_rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    is_saver = (global_rank // max(tp_size, 1)) == 0  # first dp slot only
    save_pg = tp_mesh.get_group() if tp_mesh is not None else None

    if is_saver:
        logger.info(f"[tp] saving sharded optimizer (DCP) to {tmp_dir} (tp={tp_size})...")
        dcp.save({"optim": optim_sd}, checkpoint_id=tmp_dir, process_group=save_pg)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    # rank-0-only atomic promote: current dest -> .prev, tmp -> dest. Both are
    # same-filesystem renames (atomic). Only reached after every saver finished
    # writing tmp (the barrier above), so tmp is guaranteed complete here.
    if rank == 0:
        if os.path.isdir(dest_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
            os.rename(dest_dir, backup_dir)
        os.rename(tmp_dir, dest_dir)
        logger.info("[tp] sharded optimizer saved")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def tp_load_optim_dcp(model: nn.Module, optimizer, path: str, strict: bool = False) -> bool:
    """Load a DCP sharded optimizer (written by :func:`tp_save_optim_dcp`) back
    into the tp-sharded optimizer, resharding to each rank. Returns True if a
    checkpoint dir was found and loaded, False otherwise (caller then keeps fresh
    Adam moments). Collective: every rank participates (DCP reads the stored
    shards and distributes them to every rank, replicating across the dp dim).
    Must be called AFTER init_opt + apply_tp_sp so placements match."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    if optimizer is None:
        return False
    src_dir = _tp_optim_dir(path)
    backup_dir = src_dir + ".prev"

    # Pick the newest VALID checkpoint dir. A DCP dir is only loadable if its
    # `.metadata` is present (written by dcp.save on completion); a dir left
    # half-written by a mid-save crash lacks it. Prefer the primary, fall back to
    # `.prev` — so a corrupt latest optimizer degrades to a 1-epoch-older resume
    # instead of crashing the job (which is what happened when dcp.load hit a
    # metadata-less dir). Filesystem state is identical across ranks, so every
    # rank selects the same dir → the dcp.load collective stays consistent.
    def _valid(d: str) -> bool:
        return os.path.isdir(d) and os.path.isfile(os.path.join(d, ".metadata"))

    if _valid(src_dir):
        load_dir = src_dir
    elif _valid(backup_dir):
        load_dir = backup_dir
        logger.warning(f"[tp] primary optimizer ckpt {src_dir} is missing/corrupt; loading from {backup_dir}")
    else:
        logger.warning(f"[tp] no valid sharded optimizer ckpt at {src_dir} or {backup_dir}; starting fresh moments")
        return False

    # Empty sharded target tells DCP the expected DTensor placements per rank.
    optim_sd = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=False))
    logger.info(f"[tp] loading sharded optimizer (DCP) from {load_dir}...")
    dcp.load({"optim": optim_sd}, checkpoint_id=load_dir)
    set_optimizer_state_dict(
        model, optimizer, optim_sd,
        options=StateDictOptions(full_state_dict=False, strict=strict),
    )
    return True


def apply_tp_sp(predictor: nn.Module, tp_mesh: DeviceMesh) -> nn.Module:
    """Apply Megatron TP+SP to ``predictor.predictor_blocks`` in place.

    ``tp_mesh`` must be a 1-D mesh over the tensor-parallel dimension (e.g.
    ``mesh["tp"]``). Stores ``predictor._tp_mesh`` / ``predictor._tp_size`` so the
    predictor's forward scatters the residual stream to ``Shard(1)`` at the block
    entry and gathers it back before ``predictor_norm``.
    """
    if tp_mesh.ndim != 1:
        raise ValueError(f"apply_tp_sp expects a 1-D tp mesh, got ndim={tp_mesh.ndim}")
    tp_size = tp_mesh.size()
    if tp_size <= 1:
        return predictor

    blocks = getattr(predictor, "predictor_blocks", None)
    if blocks is None or not isinstance(blocks, nn.ModuleList):
        raise ValueError("predictor has no `predictor_blocks` ModuleList to parallelize")

    # Total full param+buffer count BEFORE sharding — used by tp_full_state_dict
    # to assert the save-time gather recovered everything (guards against the
    # reducer silently dropping DTensor params). Count state_dict() (params AND
    # buffers) so it matches what get_model_state_dict returns at save time.
    predictor._tp_full_numel = sum(v.numel() for v in predictor.state_dict().values() if hasattr(v, "numel"))

    repack_info: list = []  # (state_dict path, groups, n_heads, head_dim) for component-major save
    for i, block in enumerate(blocks):
        attn = block.attn
        nh = int(attn.num_heads)
        nkv = int(attn.num_kv_heads)
        if nh % tp_size != 0:
            raise ValueError(f"block {i}: num_heads={nh} not divisible by tp_size={tp_size}")
        if nkv % tp_size != 0:
            raise ValueError(f"block {i}: num_kv_heads={nkv} not divisible by tp_size={tp_size}")

        # Re-pack fused projections to head-major BEFORE parallelizing so the
        # contiguous column-parallel shard splits heads (not q/k/v). q (GQA) is
        # already head-major, so only the fused qkv / kv are re-packed. The
        # matching head-major reshape in the forward is gated on _tp_head_major.
        head_dim = int(attn.head_dim)
        if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
            _repack_fused_to_head_major(attn.qkv, groups=3, n_heads=nh, head_dim=head_dim)
            repack_info.append((f"predictor_blocks.{i}.attn.qkv", 3, nh, head_dim))
        else:
            _repack_fused_to_head_major(attn.kv, groups=2, n_heads=nkv, head_dim=head_dim)
            repack_info.append((f"predictor_blocks.{i}.attn.kv", 2, nkv, head_dim))
        attn._tp_head_major = True

        parallelize_module(block, tp_mesh, _block_parallelize_plan(block))

        # Localize head counts so the in-attn reshape / kernel see num_heads/tp
        # heads (head_dim is fixed and was computed from the global count at
        # construction time, so it stays correct).
        attn.num_heads = nh // tp_size
        attn.num_kv_heads = nkv // tp_size

    predictor._tp_mesh = tp_mesh
    predictor._tp_size = tp_size
    predictor._tp_repack_info = repack_info
    logger.info(f"[tp] applied TP+SP to {len(blocks)} predictor blocks (tp={tp_size})")
    return predictor
