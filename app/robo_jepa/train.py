# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import math
import os
import random

import torch.distributed

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import time
from collections import defaultdict

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.robo_jepa.datasets.init_data import init_datasets
from app.robo_jepa.flops_utils import (
    compute_ar_flops_for_prefix_lens,
    compute_predictor_iter_flops,
    compute_teacher_forced_flops_per_microbatch,
    compute_world_predictor_iter_flops,
)
from src.datasets.utils.utils import get_dataset_paths
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger, gpu_timer

# FSDP2 plumbing.  These are no-ops unless `meta.fsdp.enabled: true` in
# the YAML config; the legacy DDP path stays the default for prod runs.
from app.robo_jepa.fsdp_utils import (
    apply_fsdp2,
    default_mp_policy,
    fsdp_or_ddp_no_sync,
    is_fsdp2_module,
    load_dcp_or_legacy_checkpoint,
    make_fsdp_mesh,
)

# from app.robo_jepa.tactile_video import init_data as init_raw_data
from app.robo_jepa.transforms import make_transforms
from app.robo_jepa.utils import init_wandb
from app.robo_jepa.world_model import WorldModel
from app.robo_jepa.world_model_heads import WorldModelDiTCosmosHead

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
DEFAULT_EVAL_FREQ = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
random.seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def _maybe_make_profiler(folder, rank):
    """Optional torch.profiler for a dedicated debug run (env ROBO_PROFILE=1).

    Rank-0 only. Profiles a few training steps then logs an aggregated op table
    (self CUDA time) — the compact, safe-to-emit breakdown that identifies the
    bottleneck — plus a by-shape table. A full chrome trace is only exported when
    ROBO_PROFILE_TRACE=1 (it can be many GB for an 8B/rollout step). Returns None
    (no-op) in normal runs so production is unaffected.
    """
    if os.environ.get("ROBO_PROFILE") != "1" or rank != 0:
        return None
    from torch.profiler import ProfilerActivity, profile, schedule

    export_trace = os.environ.get("ROBO_PROFILE_TRACE") == "1"

    def _on_ready(p):
        try:
            logger.info(
                "[profile] top ops by self_cuda_time_total:\n%s",
                p.key_averages().table(sort_by="self_cuda_time_total", row_limit=45),
            )
            logger.info(
                "[profile] top ops by input shape:\n%s",
                p.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total", row_limit=30
                ),
            )
            if export_trace:
                trace_path = os.path.join(folder, "profile_trace_rank0.json")
                p.export_chrome_trace(trace_path)
                logger.info("[profile] chrome trace -> %s", trace_path)
        except Exception as e:  # noqa: BLE001
            logger.info("[profile] dump failed: %s", e)

    wait = int(os.environ.get("ROBO_PROFILE_WAIT", "1"))
    warmup = int(os.environ.get("ROBO_PROFILE_WARMUP", "1"))
    active = int(os.environ.get("ROBO_PROFILE_ACTIVE", "2"))
    logger.info(f"[profile] ROBO_PROFILE=1: wait={wait} warmup={warmup} active={active} trace={export_trace}")
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=_on_ready,
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    )


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    folder = args.get("folder")
    # -- META
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    load_checkpoint_strict = cfgs_meta.get("load_checkpoint_strict", True)
    load_optimizer = cfgs_meta.get("load_optimizer", True)
    dont_load = cfgs_meta.get("dont_load", [])
    pretrained = cfgs_meta.get("pretrained", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    compile_model = cfgs_meta.get("compile", False)
    # ── FSDP2 sharding (opt-in via `meta.fsdp:` block) ──────────────────
    # When `enabled: true`, the predictor + each head are wrapped with
    # torch.distributed._composable.fsdp.fully_shard (HSDP layout: shard
    # intra-node, replicate cross-node).  See app/robo_jepa/fsdp_utils.py.
    cfgs_fsdp = cfgs_meta.get("fsdp", {}) or {}
    fsdp_enabled = bool(cfgs_fsdp.get("enabled", False))
    fsdp_cpu_offload = bool(cfgs_fsdp.get("cpu_offload", False))
    fsdp_mesh_layout = cfgs_fsdp.get("mesh_layout", "hsdp")
    # ── Tensor + Sequence Parallelism (opt-in via `meta.tp:` block) ─────
    # When `size > 1`, the predictor is wrapped with Megatron TP+SP
    # (app/robo_jepa/parallel_utils.py): a group of `size` consecutive ranks
    # collaborates on ONE sample. Data-parallel gradient reduction runs over the
    # remaining `dp` mesh dim via DDP(device_mesh=dp). FSDP+TP (3D) is not yet
    # supported — run TP with `meta.fsdp.enabled: false`.
    cfgs_tp = cfgs_meta.get("tp", {}) or {}
    tp_size = int(cfgs_tp.get("size", 1))
    if tp_size > 1 and fsdp_enabled:
        raise NotImplementedError(
            "meta.tp.size>1 with meta.fsdp.enabled=true (FSDP+TP / 3D) is not supported yet; "
            "run TP with meta.fsdp.enabled=false."
        )
    # ── Cooldown phase (opt-in via `meta.cooldown:` block) ──────────────
    # When `enabled: true`, the predictor's optimizer is built with the
    # `LinearCooldownSchedule` (LR linearly decays from `optimizer.ref_lr`
    # → 0 over `num_steps` optimizer steps; WD held constant).  Intended
    # use: load an intermediate scaling-WM checkpoint that was trained
    # with a stable high LR and finish it with a clean cooldown.
    #
    # Knobs:
    #   * `enabled`         — switch the LR/WD pair to LinearCooldownSchedule.
    #   * `num_steps`       — duration of the linear LR decay.
    #   * `reset_on_load`   — controls how the scheduler / outer-epoch
    #     counter behave when a checkpoint is loaded:
    #       - True  (default): on a *fresh* cooldown start (load_checkpoint
    #         points at the source training intermediate), reset start_epoch
    #         to 0 so the cooldown runs the full num_steps from ref_lr → 0.
    #         On a *preempt resume* (resume_preempt=True), we fast-forward
    #         the scheduler by start_epoch * ipe so the cooldown continues
    #         where it left off — i.e. we DON'T re-do a full cooldown from
    #         scratch on top of an already-partially-cooled model.
    #       - False: NEVER reset on any load.  Always fast-forward.  Use
    #         only when the load_checkpoint path points at a partially-
    #         cooled model that you want to keep cooling from where it
    #         stopped; do NOT use with a freshly-trained source ckpt
    #         (start_epoch would be that source's training epoch and the
    #         scheduler would fast-forward past num_steps → LR ≈ 0 from
    #         the very first iter).
    # The optimizer state from the checkpoint IS reloaded (so AdamW moments
    # carry over) in every case — only schedulers and the epoch counter
    # are affected by these knobs.
    cfgs_cooldown = cfgs_meta.get("cooldown", {}) or {}
    cooldown_enabled = bool(cfgs_cooldown.get("enabled", False))
    cooldown_num_steps = int(cfgs_cooldown.get("num_steps", 5000))
    cooldown_reset_on_load = bool(cfgs_cooldown.get("reset_on_load", True))
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    cfgs_model = args.get("model")
    model_name = cfgs_model.get("model_name")  # noqa: F841
    encoder_pretrained_path = cfgs_model.get("encoder_pretrained_path")
    encoder_version = cfgs_model.get("encoder_version", "v21")
    pretrained_path = cfgs_model.get("pretrained_path", "none")
    encoder_layernorm = cfgs_model["transition_model"]["config"].get("encoder_layernorm", False)

    main_optimizer = cfgs_model["main_optimizer"]
    num_frames = cfgs_model["transition_model"]["config"].get("num_frames", 16)
    # Tracks whether compiled padding should be active.  Always True when
    # compile_model is True — we keep the compiled model for both train and
    # eval, but pad the eval batch to the training batch size so the compiled
    # model never sees a new shape.
    _use_compiled_padding = [compile_model]

    # -- DATA
    cfgs_data = args.get("data")
    cfg_train_data = cfgs_data.get("train", [])
    # Optional per-rank "view groups" routing — partitions ranks into groups
    # that train on different view counts (V) with their own per-group
    # dataset subset and grad-accumulation factor. When absent, we fall
    # back to the legacy flat config (single global dataset list / weights /
    # batch shape across all ranks).
    train_view_groups = cfg_train_data.get("view_groups", None)
    train_dataset_names = [d for d in cfg_train_data.keys() if d not in ["weights", "kwargs", "view_groups"]]
    train_dataset_paths = get_dataset_paths(train_dataset_names)
    train_dataset_kwargs = cfg_train_data.get("kwargs", {})
    train_dataset_weights = cfg_train_data.get("weights", None)
    augs = train_dataset_kwargs.get("augmentations", {})
    if "crop_size" in augs:
        raise ValueError(
            "Config key 'crop_size' is not supported in augmentations. Use 'crop_height' and 'crop_width' instead."
        )
    train_batch_size = train_dataset_kwargs.get("batch_size", 1)
    train_num_workers = train_dataset_kwargs.get("num_workers", 0)
    if "augmentations" in train_dataset_kwargs:
        # NOTE: del train_dataset_kwargs['augmentations'] SHOULD NOT BE USED HERE!!!!
        # The reason is that this breaks the preemption recovery!!!
        train_dataset_kwargs = {
            k: v for k, v in train_dataset_kwargs.items() if k not in ["augmentations", "batch_size", "num_workers"]
        }
    train_kwargs_per_dataset = {
        d: kws for d, kws in cfg_train_data.items() if d not in ["weights", "kwargs", "view_groups"]
    }


    train_dataset_kwargs["s3_profile"] = cfgs_data.get("s3_profile", None)
    train_dataset_kwargs["s3_prefix"] = os.path.expandvars(cfgs_data.get("s3_prefix", ""))

    # -- OPTIMIZATION
    if main_optimizer == "transition_model":
        num_epochs = cfgs_model["transition_model"]["optimizer"]["num_epochs"]
        ipe = cfgs_model["transition_model"]["optimizer"]["iterations_per_epoch"]
        train_predictor = True
        train_heads_on_predictor = False
    elif main_optimizer == "image_head":
        num_epochs = cfgs_model["heads"]["image_head"]["optimizer"]["num_epochs"]
        ipe = cfgs_model["heads"]["image_head"]["optimizer"]["iterations_per_epoch"]

        image_head_type = cfgs_model["heads"]["image_head"]["kind"]
        train_predictor = cfgs_model["heads"]["train_predictor"]
        train_heads_on_predictor = cfgs_model["heads"]["train_heads_on_predictor"]
    else:
        raise ValueError(f"main_optimizer must be 'transition_model' or 'image_head', got {main_optimizer!r}")
    # NOTE: add new heads here
    train_heads = "image_head" in cfgs_model["heads"]
    # ── Cooldown phase: clip the outer training loop to exactly
    # `ceil(cooldown_num_steps / ipe)` epochs so the schedule reaches
    # LR=0 right as training ends.  This overrides whatever
    # `optimizer.num_epochs` was set to in the YAML — convenient because
    # the user typically inherits the optimizer block from the original
    # (long) `wm_*.yaml` config and only flips on the cooldown block.
    if cooldown_enabled:
        cooldown_num_epochs = max(1, math.ceil(cooldown_num_steps / max(int(ipe), 1)))
        logger.info(
            f"[cooldown] overriding num_epochs={num_epochs} -> "
            f"{cooldown_num_epochs} so the outer loop runs "
            f"{cooldown_num_epochs} * {ipe} = {cooldown_num_epochs * ipe} "
            f"iters >= cooldown_num_steps={cooldown_num_steps}."
        )
        num_epochs = cooldown_num_epochs
    # -- WANDB
    cfg_wandb = args.get("wandb")
    wdb_entity = cfg_wandb.get("entity")
    wdb_project = cfg_wandb.get("project")
    wdb_run_name = cfg_wandb.get("name")
    wdb_run_id = cfg_wandb.get("run_id")

    # -- LOGGING
    cfgs_logging = args.get("logging")
    tag = cfgs_logging.get("write_tag")
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass
    # -- init torch distributed backend

    # sometimes the script gets "address already in use" if it restarts too quickly after preemption
    # therefore, we do this
    add_to_port = int(os.environ.get("SLURM_JOB_ID", 9)) % 10
    nccl_timeout_minutes = cfgs_meta.get("nccl_timeout_minutes", 120)
    world_size, rank = init_distributed(
        port=37368 + add_to_port,
        timeout=datetime.timedelta(minutes=nccl_timeout_minutes),
    )
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # ----------------------------------------------------------------------- #
    # Resolve which view-group this rank belongs to (when the new
    # `data.train.view_groups` schema is provided).  Each rank ends up with
    # its own dataset subset, weights, view-count V, and grad-accumulation
    # factor K.  When absent, we fall back to the legacy flat config: every
    # rank trains on the same dataset list with K=1.
    # ----------------------------------------------------------------------- #
    if train_view_groups is not None:
        # Validate: groups must partition range(world_size).
        covered = [None] * world_size
        for gname, gcfg in train_view_groups.items():
            r0, r1 = int(gcfg["rank_range"][0]), int(gcfg["rank_range"][1])
            for r in range(r0, r1):
                if r >= world_size:
                    raise ValueError(f"view_groups['{gname}'] rank_range=[{r0}, {r1}) exceeds world_size={world_size}")
                if covered[r] is not None:
                    raise ValueError(f"view_groups overlap on rank {r}: '{covered[r]}' and '{gname}'")
                covered[r] = gname
        for r, gname in enumerate(covered):
            if gname is None:
                raise ValueError(f"view_groups do not cover rank {r}; ranks must form a complete partition")

        my_group_name = covered[rank]
        my_group_cfg = train_view_groups[my_group_name]
        my_num_views = int(my_group_cfg["num_views"])
        my_grad_accum = int(my_group_cfg.get("grad_accum_steps", 1))
        my_batch_size = int(my_group_cfg.get("batch_size", train_batch_size))
        my_rank_range = (int(my_group_cfg["rank_range"][0]), int(my_group_cfg["rank_range"][1]))
        my_group_size = my_rank_range[1] - my_rank_range[0]
        my_rank_offset = rank - my_rank_range[0]
        my_datasets = list(my_group_cfg["datasets"])
        my_weights = list(my_group_cfg.get("weights", [1.0] * len(my_datasets)))
        if len(my_weights) != len(my_datasets):
            raise ValueError(
                f"view_groups['{my_group_name}']: len(weights)={len(my_weights)} does not match "
                f"len(datasets)={len(my_datasets)}"
            )
        # Predictor's RoPE V-axis allocation is fixed at construction by
        # cfgs_model[...].num_views and MUST be >= the max V across all
        # view groups, so DDP parameter shapes match across mixed-V ranks.
        max_group_v = max(int(g["num_views"]) for g in train_view_groups.values())
        configured_num_views = int(cfgs_model["transition_model"]["config"].get("num_views", 1))
        if configured_num_views < max_group_v:
            raise ValueError(
                f"model.transition_model.config.num_views={configured_num_views} is smaller than "
                f"max(view_groups[*].num_views)={max_group_v}. Bump num_views to >= {max_group_v}."
            )
        # Pre-warm the attn_mask cache for every V used by any group, so no
        # GPU allocation happens during forward (which would race with
        # NCCL all-reduce). If user set view_groups_v explicitly, validate it
        # covers every group's V; otherwise inject the union automatically.
        group_v_values = {int(g["num_views"]) for g in train_view_groups.values()}
        configured_view_groups_v = cfgs_model["transition_model"]["config"].get("view_groups_v", None)
        if configured_view_groups_v is not None:
            missing_v = group_v_values - set(int(v) for v in configured_view_groups_v)
            if missing_v:
                raise ValueError(
                    f"model.transition_model.config.view_groups_v={configured_view_groups_v} does not "
                    f"cover the V values used by view_groups: missing {sorted(missing_v)}."
                )
        else:
            cfgs_model["transition_model"]["config"]["view_groups_v"] = sorted(group_v_values)
        # ---------------------------------------------------------------- #
        # Per-rank RESOLUTION routing via per-view-group SUBGROUPS.          #
        # A view group fixes V + datasets/weights for its whole rank_range.  #
        # Its optional `augmentations` block sub-partitions that rank_range  #
        # into named resolution subgroups, each specifying only:            #
        #     rank_range, crop_height, crop_width, grad_accum_steps          #
        #   view_groups:                                                     #
        #     two_view:                                                      #
        #       num_views: 2                                                 #
        #       rank_range: [0, 128]                                         #
        #       datasets: [...]; weights: [...]                             #
        #       augmentations:                                               #
        #         hi: { rank_range: [0, 64],  crop_height: 240, crop_width: 320, grad_accum_steps: 1 }
        #         lo: { rank_range: [64, 128], crop_height: 144, crop_width: 192, grad_accum_steps: 2 }
        # A view group that omits `augmentations` trains its whole range at  #
        # the global resolution with its own `grad_accum_steps` (legacy).    #
        # crop_*//patch sets the encoder grid; RoPE is parameter-free w.r.t. #
        # grid size so DDP all-reduce is unaffected across mixed-res ranks;  #
        # the predictor pins the runtime (H,W) per batch (predictor.forward  #
        # -> RoPEAttention).                                                 #
        # ---------------------------------------------------------------- #
        # Derive the encoder patch size from the global config invariant
        # (config grid_height == global_crop_height // patch). This avoids
        # needing the not-yet-built encoder object here.
        _cfg_gh = int(cfgs_model["transition_model"]["config"]["grid_height"])
        _cfg_gw = int(cfgs_model["transition_model"]["config"]["grid_width"])
        _global_ch = int(augs.get("crop_height")) if augs.get("crop_height") is not None else _cfg_gh
        _global_cw = int(augs.get("crop_width")) if augs.get("crop_width") is not None else _cfg_gw
        if _global_ch % _cfg_gh != 0 or _global_cw % _cfg_gw != 0:
            raise ValueError(
                f"Cannot derive encoder patch: global crop ({_global_ch}x{_global_cw}) not "
                f"divisible by config grid ({_cfg_gh}x{_cfg_gw})."
            )
        _patch_h = _global_ch // _cfg_gh
        _patch_w = _global_cw // _cfg_gw
        if _patch_h != _patch_w:
            raise ValueError(f"Non-square encoder patch derived ({_patch_h} vs {_patch_w}); unsupported.")
        _enc_patch = _patch_h

        def _grid_for_crop(ch, cw, where):
            ch, cw = int(ch), int(cw)
            if ch % _enc_patch != 0 or cw % _enc_patch != 0:
                raise ValueError(f"{where}: crop ({ch}x{cw}) must be a multiple of the encoder patch ({_enc_patch}).")
            return (ch // _enc_patch, cw // _enc_patch)

        def _subgroup_augs(sg):
            # Everything except routing knobs becomes an augmentation override.
            _skip = ("rank_range", "grad_accum_steps", "batch_size")
            return {**augs, **{k: v for k, v in sg.items() if k not in _skip}}

        # Flatten (view-group x resolution-subgroup) into synthetic entries for
        # FLOPs accounting + view_groups_hw pre-warm, validating that the
        # subgroups partition each view group's rank_range exactly.
        flat_flop_groups: dict = {}
        flat_flop_grids: dict = {}
        for gname, gcfg in train_view_groups.items():
            gr0, gr1 = int(gcfg["rank_range"][0]), int(gcfg["rank_range"][1])
            gV = int(gcfg["num_views"])
            subs = gcfg.get("augmentations", None)
            if not subs:
                flat_flop_groups[gname] = {
                    "rank_range": [gr0, gr1],
                    "num_views": gV,
                    "grad_accum_steps": int(gcfg.get("grad_accum_steps", 1)),
                    "batch_size": int(gcfg.get("batch_size", train_batch_size)),
                }
                flat_flop_grids[gname] = _grid_for_crop(_global_ch, _global_cw, f"view_groups['{gname}']")
                continue
            if any(k in subs for k in ("crop_height", "crop_width", "rank_range", "crop_size")):
                raise ValueError(
                    f"view_groups['{gname}'].augmentations must be a dict of NAMED resolution subgroups "
                    f"(each a dict with rank_range/crop_height/crop_width/grad_accum_steps), not a flat "
                    f"augmentations dict."
                )
            sub_cov: dict = {}
            for sname, scfg in subs.items():
                sr0, sr1 = int(scfg["rank_range"][0]), int(scfg["rank_range"][1])
                if sr0 < gr0 or sr1 > gr1:
                    raise ValueError(
                        f"view_groups['{gname}'].augmentations['{sname}'] rank_range=[{sr0},{sr1}) is "
                        f"outside the view group rank_range=[{gr0},{gr1})."
                    )
                for r in range(sr0, sr1):
                    if r in sub_cov:
                        raise ValueError(
                            f"view_groups['{gname}'] subgroup overlap on rank {r}: "
                            f"'{sub_cov[r]}' and '{sname}'."
                        )
                    sub_cov[r] = sname
                fname = f"{gname}/{sname}"
                flat_flop_groups[fname] = {
                    "rank_range": [sr0, sr1],
                    "num_views": gV,
                    "grad_accum_steps": int(scfg.get("grad_accum_steps", 1)),
                    "batch_size": int(scfg.get("batch_size", gcfg.get("batch_size", train_batch_size))),
                }
                flat_flop_grids[fname] = _grid_for_crop(
                    scfg.get("crop_height", _global_ch),
                    scfg.get("crop_width", _global_cw),
                    f"view_groups['{gname}'].augmentations['{sname}']",
                )
            for r in range(gr0, gr1):
                if r not in sub_cov:
                    raise ValueError(
                        f"view_groups['{gname}'].augmentations subgroups do not cover rank {r}; "
                        f"subgroups must partition the view group rank_range=[{gr0},{gr1})."
                    )

        # Resolve THIS rank's resolution subgroup (within its view group).
        _my_subs = my_group_cfg.get("augmentations", None)
        if not _my_subs:
            my_subgroup_name = None
            my_augs = dict(augs)
            my_grad_accum = int(my_group_cfg.get("grad_accum_steps", 1))
            my_batch_size = int(my_group_cfg.get("batch_size", train_batch_size))
            my_grid_h, my_grid_w = _grid_for_crop(_global_ch, _global_cw, f"view_groups['{my_group_name}']")
        else:
            my_subgroup_name = None
            for sname, scfg in _my_subs.items():
                sr0, sr1 = int(scfg["rank_range"][0]), int(scfg["rank_range"][1])
                if sr0 <= rank < sr1:
                    my_subgroup_name = sname
                    _my_sub_cfg = scfg
                    break
            if my_subgroup_name is None:
                raise ValueError(
                    f"rank {rank} is not covered by any augmentations subgroup of "
                    f"view_groups['{my_group_name}']."
                )
            my_augs = _subgroup_augs(_my_sub_cfg)
            my_grad_accum = int(_my_sub_cfg.get("grad_accum_steps", 1))
            my_batch_size = int(
                _my_sub_cfg.get("batch_size", my_group_cfg.get("batch_size", train_batch_size))
            )
            my_grid_h, my_grid_w = _grid_for_crop(
                my_augs.get("crop_height", _global_ch),
                my_augs.get("crop_width", _global_cw),
                f"view_groups['{my_group_name}'].augmentations['{my_subgroup_name}']",
            )

        # Pre-warm the attn_mask cache for every (H,W) used by any subgroup,
        # plus the construction grid, so no rank allocates a mask mid-forward.
        group_hw_values = set(flat_flop_grids.values()) | {(_cfg_gh, _cfg_gw)}
        configured_view_groups_hw = cfgs_model["transition_model"]["config"].get("view_groups_hw", None)
        if configured_view_groups_hw is not None:
            cfg_hw = {(int(p[0]), int(p[1])) for p in configured_view_groups_hw}
            missing_hw = group_hw_values - cfg_hw
            if missing_hw:
                raise ValueError(
                    f"model.transition_model.config.view_groups_hw={configured_view_groups_hw} does not "
                    f"cover the (H,W) grids used by view_groups/eval: missing {sorted(missing_hw)}."
                )
        else:
            cfgs_model["transition_model"]["config"]["view_groups_hw"] = [list(hw) for hw in sorted(group_hw_values)]
        logger.info(
            f"[rank {rank}] view_groups['{my_group_name}'] subgroup={my_subgroup_name} "
            f"V={my_num_views} K={my_grad_accum} B={my_batch_size} "
            f"crop=({my_augs.get('crop_height')}x{my_augs.get('crop_width')}) grid=({my_grid_h}x{my_grid_w}); "
            f"view_groups_hw={cfgs_model['transition_model']['config']['view_groups_hw']}"
        )
        # Cell label (view-group/resolution-subgroup) used for per-cell timing
        # + FLOPs reporting, matching the flat_flop_groups keys.
        my_cell_name = f"{my_group_name}/{my_subgroup_name}" if my_subgroup_name is not None else my_group_name
        # Restrict the global dataset name list to this rank's group's subset.
        train_dataset_names = my_datasets
        train_dataset_paths = get_dataset_paths(train_dataset_names)
        train_dataset_weights = my_weights
        # Per-group dataset overrides: each view-group can selectively
        # override fields of a globally-defined dataset block. This lets you
        # reuse a single dataset (e.g., Droid) across two_view (V=2) and
        # one_view (V=1) groups by overriding `cameras` per group:
        #   view_groups:
        #     two_view:
        #       datasets: [Droid]
        #       # No overrides → uses Droid block as defined globally
        #     one_view:
        #       datasets: [Droid]
        #       dataset_overrides:
        #         Droid:
        #           cameras: ['.*']        # ← overrides V=2 → V=1 here
        # Override is a top-level shallow merge (override keys replace
        # global keys). Nested dicts (e.g. view_map) are replaced wholesale.
        import copy as _copy

        my_overrides = my_group_cfg.get("dataset_overrides", {}) or {}
        train_kwargs_per_dataset = _copy.deepcopy(train_kwargs_per_dataset)
        for ds_name, ovr in my_overrides.items():
            if ds_name not in train_kwargs_per_dataset:
                raise ValueError(
                    f"view_groups['{my_group_name}'].dataset_overrides references "
                    f"unknown dataset '{ds_name}'. Globally defined datasets: "
                    f"{list(train_kwargs_per_dataset.keys())}"
                )
            if ds_name not in my_datasets:
                logger.warning(
                    f"view_groups['{my_group_name}'].dataset_overrides has '{ds_name}' "
                    f"but it's not in this group's datasets list — override is a no-op."
                )
                continue
            train_kwargs_per_dataset[ds_name] = {
                **train_kwargs_per_dataset[ds_name],
                **dict(ovr),
            }
            logger.info(f"[rank {rank}] view_groups['{my_group_name}'] override on '{ds_name}': {dict(ovr)}")
        logger.info(
            f"[rank {rank}] view_groups: group='{my_group_name}' V={my_num_views} K={my_grad_accum} "
            f"rank_range={my_rank_range} group_size={my_group_size} rank_offset={my_rank_offset} "
            f"datasets={my_datasets}"
        )
    else:
        my_group_name = "default"
        my_num_views = int(cfgs_model["transition_model"]["config"].get("num_views", 1))
        my_grad_accum = 1
        my_batch_size = train_batch_size
        my_group_size = world_size
        my_rank_offset = rank
        # Legacy flat config: single global resolution for every rank.
        my_augs = augs
        my_grid_h = int(cfgs_model["transition_model"]["config"]["grid_height"])
        my_grid_w = int(cfgs_model["transition_model"]["config"]["grid_width"])
        flat_flop_groups = None
        flat_flop_grids = None
        my_cell_name = "default"
        logger.info(f"[rank {rank}] view_groups not configured; using legacy flat dataloader with K=1")

    # ----------------------------------------------------------------------- #
    # Tensor-parallel (TP) data routing.  With tp_size>1, a group of `tp_size`
    # CONSECUTIVE ranks forms one TP group that collaborates on ONE sample, so
    # the data sampler must shard on the DP axis only (all tp ranks in a group
    # draw the same shard) and the loaded batch is broadcast across the TP group
    # (below in the train loop) to neutralize per-rank augmentation/temporal RNG.
    #
    # rank_range boundaries (view-group AND resolution-subgroup) must be aligned
    # to tp_size so a TP group never straddles a group/subgroup boundary.
    # ----------------------------------------------------------------------- #
    if tp_size > 1:
        if world_size % tp_size != 0:
            raise ValueError(f"world_size={world_size} not divisible by tp_size={tp_size}")
        if train_view_groups is not None:
            for gname, gcfg in train_view_groups.items():
                gr0, gr1 = int(gcfg["rank_range"][0]), int(gcfg["rank_range"][1])
                if gr0 % tp_size != 0 or (gr1 - gr0) % tp_size != 0:
                    raise ValueError(
                        f"view_groups['{gname}'] rank_range=[{gr0},{gr1}) is not tp_size={tp_size}-aligned"
                    )
                for sname, scfg in (gcfg.get("augmentations", None) or {}).items():
                    sr0, sr1 = int(scfg["rank_range"][0]), int(scfg["rank_range"][1])
                    if sr0 % tp_size != 0 or (sr1 - sr0) % tp_size != 0:
                        raise ValueError(
                            f"view_groups['{gname}'].augmentations['{sname}'] rank_range=[{sr0},{sr1}) "
                            f"is not tp_size={tp_size}-aligned"
                        )
        if my_group_size % tp_size != 0:
            raise ValueError(f"group '{my_group_name}' size={my_group_size} not divisible by tp_size={tp_size}")
        my_tp_rank = my_rank_offset % tp_size
        my_dp_rank = my_rank_offset // tp_size
        my_dp_size = my_group_size // tp_size
    else:
        my_tp_rank = 0
        my_dp_rank = my_rank_offset
        my_dp_size = my_group_size

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")  # noqa: F841
    latest_file = f"{tag}-latest.pth.tar"
    latest_path = os.path.join(folder, latest_file)

    # Auto-detect existing checkpoint for resumption (handles scontrol requeue,
    # NODE FAILURE requeue, and any other restart path that bypasses submitit's
    # checkpoint() method — none of which set resume_preempt).
    # IMPORTANT: under FSDP2, save_dcp_checkpoint writes a DIRECTORY at
    # `<tag>-latest` (no `.pth.tar` suffix), so plain `os.path.exists` on
    # `<tag>-latest.pth.tar` would return False even when a fully valid DCP
    # checkpoint exists at `<tag>-latest/`.  Treat either form as "exists".
    def _ckpt_exists(p: str) -> bool:
        if not p:
            return False
        if os.path.exists(p):
            return True
        # FSDP2 DCP dir: strip `.pth.tar` suffix and check for directory.
        dcp = p[: -len(".pth.tar")] if p.endswith(".pth.tar") else p
        return os.path.isdir(dcp)

    # Prefer an IN-FOLDER checkpoint whenever one exists, EVEN when
    # `load_checkpoint: True` (which points read_checkpoint at a pretrain/source
    # ckpt). A fresh submission always lands in a NEW timestamped run folder
    # (update_folder_with_timestamp), so no in-folder latest exists yet and we
    # correctly start from the source. But a requeue (node failure / scontrol)
    # re-runs the initial path with resume_preempt=False in the SAME folder,
    # where world_model-latest now exists — without this, the run would ignore
    # its own progress and RESTART the cooldown from the source at step 0.
    # Detecting the in-folder latest here forces a proper preempt-resume
    # (fast-forwarding the cooldown scheduler) instead of a fresh restart.
    if _ckpt_exists(latest_path):
        load_model = True
        resume_preempt = True
        logger.info(f"Auto-detected in-folder checkpoint at {latest_path}, resuming (preempt-resume)")
    load_path = None
    # Whether a preempt-resume actually loaded an IN-FOLDER cooldown checkpoint
    # (whose epoch is cooldown-relative) vs. fell back to the SOURCE checkpoint
    # (whose epoch is pretrain-relative). Only the former may fast-forward the
    # cooldown scheduler; the latter must be treated as a fresh cooldown start.
    resumed_from_infolder_ckpt = False
    if load_model:
        # load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if resume_preempt and _ckpt_exists(latest_path):
            load_path = latest_path
            load_optimizer = True
            dont_load = []
            resumed_from_infolder_ckpt = True
            logger.info(f"Resuming from preemption! Continuing from checkpoint {load_path}")
        elif resume_preempt and r_file is not None:
            # Preempted before any in-folder save was written. Fall back to the
            # source checkpoint specified in meta.read_checkpoint so we don't
            # silently re-randomize weights and restart the LR sweep. Because the
            # source epoch is pretrain-relative (NOT cooldown-relative), the
            # cooldown logic below treats this as a FRESH start (start_epoch=0)
            # rather than fast-forwarding past the cooldown horizon.
            load_path = os.path.join(folder, r_file)
            load_optimizer = True
            dont_load = []
            logger.warning(
                f"[resume_preempt] in-folder latest checkpoint {latest_path!r} not found "
                f"(preempted before first save). Falling back to source checkpoint {load_path}."
            )
        elif r_file is not None:
            load_path = os.path.join(folder, r_file)
            logger.info(
                f"Starting the training from checkpoint {load_path}; the path to save next checkpoints in future in this job is {latest_path}"  # noqa: E501
            )
        else:
            logger.info("load_checkpoint is set to true but no checkpoint file specified! ")
            load_path = None
        # FSDP2-aware existence check (see `_ckpt_exists` above).  The previous
        # `os.path.exists(load_path)` silently turned `load_model` off whenever
        # the on-disk artifact was a DCP directory rather than a `.pth.tar`
        # file — which is the case for every 8B run.  That made every
        # preempt-resume start from random-init FSDP2 weights.
        if not _ckpt_exists(load_path):
            logger.warning(
                f"load_path={load_path!r} not found on disk (checked both file and DCP-dir form); "
                f"falling back to no-load."
            )
            load_path = None
            load_model = False
    if pretrained is not None:
        pretrained = os.path.join(folder, pretrained)

    # -- init model
    heads = {}
    if train_heads:
        if "image_head" in cfgs_model["heads"]:
            if image_head_type is not None and image_head_type.lower() != "none":
                if image_head_type != "dit":
                    raise ValueError(f"image_head.kind must be 'dit' (the only decoder), got {image_head_type!r}")
                if image_head_type == "dit":
                    diffusion_kwargs = cfgs_model["heads"]["image_head"]["diffusion_kwargs"]
                    decoder = WorldModelDiTCosmosHead(
                        head_config=dict(cfgs_model["heads"]["image_head"]["config"]),
                        device=device,
                        diffusion_kwargs=diffusion_kwargs,
                    )
                    denoising_steps = cfgs_model["heads"]["image_head"]["config"].get("denoising_steps", [1000])
                    from app.robo_jepa.robo_jepa.diffusion import create_diffusion

                    diffusion_models = [create_diffusion(str(s), **diffusion_kwargs) for s in denoising_steps]

                heads["image_head"] = decoder
    world_model = WorldModel(
        transition_model_config=dict(cfgs_model["transition_model"]["config"]),
        train_config=dict(cfgs_model["transition_model"]["trainer"]),
        device=device,
        heads=heads,
    )

    # ── FSDP2 vs legacy DDP wrap ────────────────────────────────────────
    # When FSDP2 is enabled we must:
    #   (a) wrap BEFORE init_opt so the optimizer's lazy state allocation
    #       sees DTensor-shaped params (otherwise we OOM at the first
    #       step() with a non-sharded state).
    #   (b) wrap BEFORE the pretrained_path warm-start so the warm-start
    #       can broadcast & re-shard via set_model_state_dict.
    # The legacy DDP path keeps the old order (pretrained -> init_opt ->
    # compile -> DDP) verbatim so production runs are byte-stable.

    loaded_transition_model = False
    _tp_loaded_epoch = 0  # epoch captured by the TP pre-wrap load (for cooldown fast-forward on resume)

    # ── Build the TP+SP parallel mesh (FSDP-off path) ───────────────────
    # tp_mesh feeds apply_tp_sp; dp_mesh drives DDP's gradient reduction;
    # tp_group is used to broadcast each batch across the TP group in the loop.
    tp_mesh = None
    dp_mesh = None
    tp_group = None
    if tp_size > 1:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("meta.tp.size>1 requires a distributed group (launch with app.main_distributed).")
        from app.robo_jepa.parallel_utils import make_parallel_mesh

        _gpus_per_node = int(args.get("tasks_per_node", torch.cuda.device_count() or 8))
        parallel_mesh = make_parallel_mesh(dist.get_world_size(), _gpus_per_node, tp_size, fsdp_enabled=False)
        tp_mesh = parallel_mesh["tp"]
        dp_mesh = parallel_mesh["dp"]
        tp_group = tp_mesh.get_group()

    if fsdp_enabled:
        # --- FSDP2 branch ---------------------------------------------------
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "meta.fsdp.enabled=true requires a distributed group (launch with app.main / app.main_distributed)."
            )
        world_size_for_mesh = dist.get_world_size()
        gpus_per_node = int(args.get("tasks_per_node", torch.cuda.device_count() or 8))
        mesh = make_fsdp_mesh(world_size_for_mesh, gpus_per_node, layout=fsdp_mesh_layout)
        mp_policy = default_mp_policy(dtype)
        use_ac = bool(cfgs_model["transition_model"]["config"].get("use_activation_checkpointing", False))

        if train_predictor:
            apply_fsdp2(
                world_model.transition_model,
                mesh,
                mp_policy=mp_policy,
                use_activation_checkpointing=use_ac,
                cpu_offload=fsdp_cpu_offload,
            )
        else:
            for p in world_model.transition_model.parameters():
                p.requires_grad = False

        for name in world_model.heads.keys():
            apply_fsdp2(
                world_model.heads[name].model,
                mesh,
                mp_policy=mp_policy,
                use_activation_checkpointing=use_ac,
                cpu_offload=fsdp_cpu_offload,
            )

        # Warm-start from a non-sharded `.pth.tar` (or DCP dir) AFTER the
        # FSDP wrap, so the loader can broadcast + re-shard into DTensor.
        if pretrained_path != "none" and (not load_model or main_optimizer != "transition_model"):
            logger.info(f"[fsdp] warm-start predictor from {pretrained_path} (post-FSDP, broadcast+reshard)")
            try:
                load_dcp_or_legacy_checkpoint(
                    world_model.transition_model,
                    optimizer=None,  # warm-start only loads weights
                    path=pretrained_path,
                    load_optimizer=False,
                    strict=False,
                    dont_load=("predictor_pos_embed",),
                )
                loaded_transition_model = True
            except Exception as e:
                logger.warning(f"[fsdp] warm-start load failed (continuing): {e}")

        # init_opt AFTER FSDP wrap so the optimizer's lazy state alloc
        # in step() is DTensor-shaped from the start.
        if train_predictor:
            world_model.init_opt(
                **dict(cfgs_model["transition_model"]["optimizer"]),
                dtype=dtype,
                cooldown_enabled=cooldown_enabled,
                cooldown_num_steps=cooldown_num_steps,
            )
        for name, head in world_model.heads.items():
            head.init_opt(**dict(cfgs_model["heads"][name]["optimizer"]), dtype=dtype)

        # torch.compile AFTER fully_shard
        # cleanly when this order is used).  The DDPOptimizer flag is
        # not relevant under FSDP2 — leave torch._dynamo.config untouched.
        if compile_model and train_predictor:
            logger.info("[fsdp] compiling transition_model with torch.compile(dynamic=False) post-FSDP wrap")
            world_model.transition_model = torch.compile(world_model.transition_model, dynamic=False)
    else:
        # --- legacy DDP branch (unchanged from prod) ------------------------
        if tp_size > 1 and train_predictor:
            # Tensor parallel: do ALL predictor loading HERE, into the PLAIN
            # model, BEFORE apply_tp_sp (TP re-packs qkv to head-major + shards
            # into DTensors). The source to load is `load_path` resolved above:
            #   * fresh start         -> read_checkpoint (the source pretrain ckpt)
            #   * preempt-resume      -> folder/world_model-latest (this run's own
            #                            component-major TP save) — so the cooldown
            #                            CONTINUES with the LR fast-forwarded
            #                            (see the post-wrap block) instead of
            #                            restarting from the source.
            # Falls back to pretrained_path if load_path is unset.
            _tp_src = load_path if (load_model and load_path is not None) else (
                pretrained_path if pretrained_path != "none" else None
            )
            if _tp_src is not None:
                logger.info(f"[tp] loading {_tp_src} into plain predictor before apply_tp_sp")
                _tp_ep, _ = load_dcp_or_legacy_checkpoint(
                    world_model.transition_model,
                    optimizer=None,
                    path=_tp_src,
                    load_optimizer=False,
                    strict=load_checkpoint_strict,
                )
                loaded_transition_model = True
                _tp_loaded_epoch = int(_tp_ep or 0)
            from app.robo_jepa.parallel_utils import apply_tp_sp

            apply_tp_sp(world_model.transition_model, tp_mesh)
        else:
            if pretrained_path != "none" and (not load_model or main_optimizer != "transition_model"):
                logger.info(f"pretrained path is not None! loading the model from: {pretrained_path}")
                if os.path.isdir(pretrained_path):
                    # DCP directory (FSDP-saved). Load into the predictor here.
                    load_dcp_or_legacy_checkpoint(
                        world_model.transition_model,
                        optimizer=None,
                        path=pretrained_path,
                        load_optimizer=False,
                        strict=False,
                    )
                    loaded_transition_model = True
                else:
                    checkpoint = torch.load(pretrained_path, map_location="cpu", mmap=True)
                    pred_pretrained_dict = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
                    if "predictor_pos_embed" in pred_pretrained_dict:
                        del pred_pretrained_dict["predictor_pos_embed"]
                    try:
                        msg = world_model.transition_model.load_state_dict(pred_pretrained_dict, strict=False)
                        logger.info(f"loaded pretrained predictor with msg: {msg}")
                        loaded_transition_model = True
                    except RuntimeError:
                        pass  # we ignore some keys that did not load

        # -- init optimizer and scheduler
        if train_predictor:
            world_model.init_opt(
                **dict(cfgs_model["transition_model"]["optimizer"]),
                dtype=dtype,
                cooldown_enabled=cooldown_enabled,
                cooldown_num_steps=cooldown_num_steps,
            )
            # -- torch.compile the transition model before DDP wrapping
            if compile_model:
                logger.info("Compiling transition_model with torch.compile(dynamic=False)...")
                # Disable the DDPOptimizer graph-splitting backend.  The default
                # optimize_ddp=True splits the compiled graph at DDP bucket
                # boundaries and re-traces each sub-graph through Inductor.
                # During that re-trace the block-sparse attention mask tensors
                # (captured from self.attn_mask_cache) lose their device metadata,
                # causing "RuntimeError: tensor does not have a device".
                # With optimize_ddp=False DDP uses its standard C++ reducer and
                # the compiled graph stays intact.
                torch._dynamo.config.optimize_ddp = False
                world_model.transition_model = torch.compile(world_model.transition_model, dynamic=False)
            if dist.is_available() and dist.is_initialized():
                if tp_size > 1:
                    # TP + NO SHARDING: do NOT wrap the predictor. DDP/replicate
                    # /FSDP all flatten the TP DTensor params at scale (dp=224),
                    # which breaks checkpoint extraction. The model fits with
                    # headroom (~47 GB free on the tightest ranks), so we keep
                    # params/grads/opt replicated over dp and average gradients
                    # over the dp group MANUALLY in optimization_step (see
                    # TrainableModel). This keeps the params as clean TP DTensors
                    # so the checkpoint gather (over tp only) works. Stash the dp
                    # process group for the manual all-reduce.
                    world_model.transition_model._tp_dp_group = dp_mesh.get_group()
                    logger.info(
                        f"[tp] no-sharding mode: predictor unwrapped; manual dp grad all-reduce over "
                        f"dp group of size {dp_mesh.size()} (params stay replicated DTensors)."
                    )
                else:
                    # broadcast_buffers=False: critical at large world_size for any
                    # code path where some ranks call DDP forward() different numbers
                    # of times than others.
                    # DDP forward() with broadcast_buffers=True invokes a global
                    # `dist.broadcast` per call. Eval ranks with no inner-loop work
                    # skip those calls entirely, so the collective counts diverge
                    # across ranks and the broadcast deadlocks. Predictor has zero
                    # registered buffers, so disabling this is functionally a no-op
                    # for both training and eval (and a tiny perf win in training).
                    world_model.transition_model = DistributedDataParallel(
                        world_model.transition_model,
                        static_graph=False,
                        find_unused_parameters=False,
                        broadcast_buffers=False,
                    )
        else:
            for p in world_model.transition_model.parameters():
                p.requires_grad = False
        for name, head in world_model.heads.items():
            head.init_opt(**dict(cfgs_model["heads"][name]["optimizer"]), dtype=dtype)
        for name in world_model.heads.keys():
            world_model.heads[name].model = DistributedDataParallel(
                world_model.heads[name].model,
                static_graph=False,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

    start_epoch = 0
    _loaded_ckpt_stats: dict = {}  # captures the `stats` dict returned by load_checkpoint
    world_model.load_encoder(
        encoder_pretrained_path,
        device="cuda",
        version=encoder_version,
        use_proj=False,
        max_num_frames=64,
        compile=True,
    )
    if load_model and not int(os.environ.get("VSCODE_DEBUG", "0")):
        if tp_size > 1:
            # TP: the predictor was already loaded into the PLAIN model pre-wrap
            # (see the TP branch above). Seed start_epoch from the epoch captured
            # by that pre-wrap load so the cooldown reset/fast-forward logic below
            # behaves like the non-TP path (fresh source -> reset to 0; in-folder
            # resume -> fast-forward the LR schedule).
            start_epoch = _tp_loaded_epoch
            # Warm-start reset (mirrors the non-TP `if not load_optimizer` branch
            # below): a fresh warm-start from a SOURCE checkpoint (not an in-folder
            # preempt-resume) must begin at epoch 0 so the warmup is honored and the
            # full num_epochs schedule runs. Without this, seeding start_epoch from
            # the source epoch (e.g. 250) makes `range(start_epoch, num_epochs)` empty
            # and fast-forwards the LR schedule past the warmup.
            preempt_resume_infolder = resume_preempt and resumed_from_infolder_ckpt
            if not preempt_resume_infolder and not load_optimizer:
                start_epoch = 0
            # Optimizer-state resume: on a real in-folder preempt-resume, reload
            # the saved Adam moments into the (already init_opt'd) tp optimizer so
            # momentum isn't lost across requeues. rank 0 reads the full opt from
            # the in-folder .pth.tar; other ranks pass {} and get it by broadcast
            # (collective). Fresh starts (source ckpt) keep fresh moments.
            if (
                resume_preempt
                and resumed_from_infolder_ckpt
                and load_path is not None
                and train_predictor
                and world_model.optimizer is not None
            ):
                from app.robo_jepa.parallel_utils import tp_load_optim_dcp

                # Optimizer is saved sharded via DCP next to the model .pth.tar
                # (see TrainableModel.save_checkpoint TP branch). Load it back,
                # resharding to this run's tp layout. Collective: every rank
                # participates. Returns False if no sharded optim dir is present
                # (e.g. preempted before the first save) -> keep fresh moments.
                if tp_load_optim_dcp(world_model.model, world_model.optimizer, load_path):
                    logger.info("[tp] resumed optimizer state (Adam moments) from sharded DCP checkpoint")
                else:
                    logger.info("[tp] no sharded optimizer checkpoint; starting with fresh Adam moments")
        else:
            start_epoch, _loaded_ckpt_stats = world_model.load_checkpoint(
                load_path,
                load_heads=True,
                load_module=not loaded_transition_model,
                strict=load_checkpoint_strict,
                dont_load=dont_load,
                load_optimizer=load_optimizer,
            )
            if _loaded_ckpt_stats is None:
                _loaded_ckpt_stats = {}
            if not load_optimizer:
                start_epoch = 0
        # ── Cooldown phase: decide whether to reset the scheduler / epoch
        # counter to step 0 (fresh cooldown start), or fast-forward them
        # so the cooldown continues where it left off (preempt resume).
        #
        # Behavior matrix:
        #   cooldown_enabled  resume_preempt(in-folder)  reset_on_load  → action
        #   ────────────────  ───────────────────────── ─────────────  → ──────────
        #   False             *                          *              → fast-forward (legacy non-cooldown path)
        #   True              False                      True           → RESET     (fresh cooldown start, default)
        #   True              False                      False          → fast-forward (manual override; risky)
        #   True              True (in-folder ckpt)      *              → fast-forward (resume; continue)
        #
        # CRITICAL: a preempt-resume that had to fall back to the SOURCE
        # checkpoint (no in-folder cooldown save existed yet) is NOT a real
        # resume — its epoch is pretrain-relative (e.g. 251), so fast-forwarding
        # by it would skip the entire cooldown horizon (range(251, 25) → empty)
        # and the job would "complete" instantly with zero training. We detect
        # this via `resumed_from_infolder_ckpt` and treat it as a fresh start.
        #
        # The auto-detect on `resume_preempt` is the important fix:
        # historically every preempt-resume restarted the cooldown LR
        # sweep at ref_lr → 0 on top of an already-partially-cooled
        # model, leading to multiple stacked cooldowns.  Now resume
        # picks up the LinearCooldownSchedule at the right step.
        preempt_resume_infolder = resume_preempt and resumed_from_infolder_ckpt
        do_reset = cooldown_enabled and cooldown_reset_on_load and (not preempt_resume_infolder)
        if do_reset:
            logger.info(
                f"[cooldown] fresh start: discarding checkpoint epoch={start_epoch}; scheduler resumes from step 0."
            )
            start_epoch = 0
        else:
            if cooldown_enabled:
                if preempt_resume_infolder:
                    logger.info(
                        f"[cooldown] preempt-resume: fast-forwarding scheduler by "
                        f"start_epoch={start_epoch} * ipe={ipe} = {start_epoch * ipe} steps "
                        f"so the cooldown continues from where it stopped."
                    )
                else:
                    logger.info(
                        f"[cooldown] reset_on_load=False override: fast-forwarding scheduler by "
                        f"start_epoch={start_epoch} * ipe={ipe} = {start_epoch * ipe} steps."
                    )
            for _ in range(start_epoch * ipe):
                if train_predictor:
                    world_model.scheduler.step()
                    world_model.wd_scheduler.step()
                for name in world_model.heads.keys():
                    world_model.heads[name].scheduler.step()
                    world_model.heads[name].wd_scheduler.step()
    if rank == 0:
        wandb_logger = init_wandb(
            resume_preempt=resume_preempt,
            folder=folder,
            project=wdb_project,
            entity=wdb_entity,
            config=args,
            name=wdb_run_name,
            run_id=wdb_run_id,
        )
    else:
        wandb_logger = None

        # -- init data-loaders/samplers
    # Build the TRAINING transform from this rank's (possibly group-specific)
    # augmentations so mixed-resolution view-groups crop to their own size.
    # The global `augs` still drives the decoder/eval/MPC paths below.
    transform = make_transforms(**my_augs)
    unsupervised_loader, unsupervised_sampler = init_datasets(
        train_dataset_paths,
        train_dataset_names,
        train_dataset_weights,
        train_dataset_kwargs,
        train_kwargs_per_dataset,
        rank=rank,
        world_size=world_size,
        transform=transform,
        batch_size=my_batch_size,
        num_workers=train_num_workers,
        # Per-group sharding: when view_groups partition the world, each
        # group shuffles/weights its dataset independently within the group.
        # With tensor parallelism, shard on the DP axis only (my_dp_size /
        # my_dp_rank) so all tp ranks in a TP group draw the SAME shard; the
        # batch is then broadcast across the TP group in the train loop.
        num_replicas_override=(my_dp_size if (train_view_groups is not None or tp_size > 1) else None),
        rank_override=(my_dp_rank if (train_view_groups is not None or tp_size > 1) else None),
    )
    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    if start_epoch > 0:
        logger.info(f"Starting epoch at: {start_epoch}")
        unsupervised_sampler.set_epoch(start_epoch)

    logger.info("Initializing loader...")
    train_loader = iter(unsupervised_loader)
    # Dataloader-spawn sync. iter(...) above forks 10 worker procs per rank
    # for the train loader and per eval loader. With many datasets in
    # data.train (more dataset metadata in the parent => bigger forked memory
    # state) and 256 ranks each forking simultaneously, the fork can take
    # 5 min on fast nodes but >20 min on slow ones — empirically observed at
    # ws=256 with 13 train datasets (job 6478577 init spread = 26 min, vs
    # 3 min on the same setup with 8 train datasets in job 6441386).
    #
    # Without this barrier, fast ranks proceed straight into the first
    # training iter (data fetch + encoder forward + Inductor compile of the
    # backward graph). Inductor compile is single-threaded Python and holds
    # the GIL for 5–10 min. While that GIL is held, the NCCL watchdog
    # thread (which needs the GIL to monitor enqueued collectives) cannot
    # make progress and fires after `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` (480 s
    # by default) → terminates the whole job. The watchdog message
    # "only active collectives: 0" confirms we never reach the all-reduce.
    #
    # Synchronizing here forces all ranks past the slow dataloader spawn
    # before any rank starts the GIL-holding compile. The barrier itself
    # is a single C-level NCCL collective; no GIL churn between ranks.
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        logger.info("Waiting for all ranks to finish dataloader spawn (barrier)...")
        dist.barrier()
        logger.info("All ranks past dataloader-spawn barrier; entering training loop.")

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        unsupervised_sampler.set_epoch(start_epoch)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                udata = next(train_loader)
            except Exception:
                train_loader = iter(unsupervised_loader)
                udata = next(train_loader)  # noqa: F841

    # ------------------------------------------------------------------ #
    # Predictor FLOPs counter (scaling-laws total-compute axis).         #
    # ------------------------------------------------------------------ #
    # Per-rank per-iter forward+backward predictor FLOPs.
    #
    # The teacher-forced part is constant per rank (forward at full T_base).
    # The AR rollout part is stochastic because train.py samples a random
    # prefix per microbatch (`train_rollout_prefixes="random"`); we stamp the
    # *realized* AR FLOPs into local_metrics by tracking the sampled prefixes
    # via `_realized_ar_prefix_lens` (cleared each iter, appended by
    # step_model after each torch.randint).
    _predictor_cfg = cfgs_model["transition_model"]["config"]
    _trainer_cfg = cfgs_model["transition_model"]["trainer"]
    # Per-rank predictor cfg view with this rank's (group) resolution so the
    # per-rank FLOPs reflect mixed-resolution training. grid_h/grid_w are in
    # encoder-token units (predictor patch_size == 1).
    _predictor_cfg_rank = {**_predictor_cfg, "grid_height": int(my_grid_h), "grid_width": int(my_grid_w)}
    teacher_forced_flops_per_microbatch = int(
        compute_teacher_forced_flops_per_microbatch(
            predictor_cfg=_predictor_cfg_rank,
            batch_size=int(my_batch_size),
            runtime_v=int(my_num_views),
            include_backward=True,
        )
    )
    # Expected per-iter FLOPs, used (a) for the [flops] init log and (b) as
    # the world-flops basis for the resume-fallback cumulative seed.
    predictor_flops_iter = int(
        compute_predictor_iter_flops(
            predictor_cfg=_predictor_cfg_rank,
            trainer_cfg=_trainer_cfg,
            batch_size=int(my_batch_size),
            runtime_v=int(my_num_views),
            grad_accum_steps=int(my_grad_accum),
            include_backward=True,
        )
    )
    # Per-microbatch accumulator for the prefix values sampled by step_model.
    # Cleared at the start of each outer iter; appended by step_model on each
    # microbatch.
    _realized_ar_prefix_lens: list = []
    # On rank 0, also compute the constant per-iter world FLOPs (sum over
    # ranks, accounting for the mixed-V partition) up-front so that resume
    # from a checkpoint produces an exact cumulative counter without having
    # to thread a new field through the checkpoint format. Per-rank flops
    # are constant across iters by construction.
    world_flops_iter = 0
    per_group_flops_per_rank: dict = {}
    if rank == 0:
        world_flops_iter, per_group_flops_per_rank = compute_world_predictor_iter_flops(
            predictor_cfg=_predictor_cfg,
            trainer_cfg=_trainer_cfg,
            batch_size=int(train_batch_size),
            view_groups_cfg=flat_flop_groups,
            world_size=int(world_size),
            fallback_num_views=int(my_num_views),
            fallback_grad_accum=int(my_grad_accum),
            include_backward=True,
            view_group_grids=flat_flop_grids,
        )
        # Resume the realized cumulative-FLOPs counter from the checkpoint's
        # `stats` dict if present; otherwise fall back to the expected-value
        # seed `start_epoch * ipe * world_flops_iter`.  The counter is
        # written to `stats["cumulative_flops"]` at save-time (see the
        # save_checkpoint call near the bottom of the epoch loop).
        _ckpt_cum = None
        if isinstance(_loaded_ckpt_stats, dict):
            _ckpt_cum = _loaded_ckpt_stats.get("cumulative_flops")
        if _ckpt_cum is not None:
            cumulative_flops = int(_ckpt_cum)
            logger.info(
                f"[flops] resumed cumulative from checkpoint stats: "
                f"{cumulative_flops:.3e} FLOPs (runtime start_epoch={start_epoch})"
            )
        else:
            cumulative_flops = int(start_epoch) * int(ipe) * int(world_flops_iter)
            if start_epoch > 0:
                logger.info(
                    f"[flops] no cumulative_flops in checkpoint; seeding from "
                    f"expected per-iter * start_epoch * ipe: {cumulative_flops:.3e}"
                )
        logger.info(f"[flops] predictor per-rank per-iter FLOPs (group={my_group_name}): {predictor_flops_iter:.3e}")
        logger.info(
            f"[flops] world per-iter FLOPs (expected): {world_flops_iter:.3e} "
            f"({world_flops_iter / 1e15:.3f} PF/iter); "
            f"per-group per-rank: { {k: f'{v:.3e}' for k, v in per_group_flops_per_rank.items()} }; "
            f"cumulative at start_epoch={start_epoch} ipe={ipe}: {cumulative_flops:.3e}"
        )
    else:
        cumulative_flops = 0

    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)
        _robo_prof = _maybe_make_profiler(folder, rank)
        if _robo_prof is not None:
            _robo_prof.start()
        for itr in range(ipe):
            itr_start_time = time.time()

            def get_batch():
                logger.debug(f"start reading batch")  # noqa: F541
                nonlocal train_loader
                try:
                    (
                        rgb_video,
                        raw_rgb_video,
                        action,
                        state,
                        camera_extrinsics,
                        padding_mask,
                        view_list,
                        dt,
                        start,
                        traj_names,
                        embodiment_names,
                        index,
                    ) = next(train_loader)
                except StopIteration as e:
                    logger.info(f"Exception {e=}")
                    logger.info("Exhausted data loaders. Refreshing...")
                    train_loader = iter(unsupervised_loader)
                    (
                        rgb_video,
                        raw_rgb_video,
                        action,
                        state,
                        camera_extrinsics,
                        padding_mask,
                        view_list,
                        dt,
                        start,
                        traj_names,
                        embodiment_names,
                        index,
                    ) = next(train_loader)

                action = action.to(device, dtype=dtype, non_blocking=True)
                padding_mask = padding_mask.to(device, dtype=dtype, non_blocking=True)
                view_list = view_list.to(device, non_blocking=True)
                state = state.to(device, dtype=dtype, non_blocking=True)
                rgb_video = rgb_video.to(device, dtype=dtype, non_blocking=True)
                dt = dt.to(device, dtype=dtype, non_blocking=True)
                raw_rgb_video = raw_rgb_video.to(device, dtype=dtype, non_blocking=True)
                camera_extrinsics = camera_extrinsics.to(device, dtype=dtype, non_blocking=True)
                logger.debug(f"done reading batch (elements: {index}; names: {traj_names})")
                return (
                    rgb_video,
                    raw_rgb_video,
                    action,
                    state,
                    camera_extrinsics,
                    padding_mask,
                    view_list,
                    dt,
                    start,
                    embodiment_names,
                    index,
                )

            def forward_target_encoder(video):
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        # video.shape == [b t v h w c]
                        # video.permute(0, 2, 1, 3, 4) == [b t c h w]
                        (
                            b,
                            _,
                            v,
                            t,
                        ) = video.shape[:4]
                        h_, w_ = video.shape[-2:]
                        h_ //= world_model.encoder.vit.patch_size
                        w_ //= world_model.encoder.vit.patch_size
                        video = einops.rearrange(video, "b c v t h w -> (b t v) c h w")
                        video = einops.repeat(video, "b c h w -> b c t h w", t=1)  # V-JEPA 2.1: one frame per image
                        h = world_model.encoder(video.float())
                        h = einops.rearrange(h, "(b t v) (h w) c -> b t v h w c", b=b, t=t, v=v, h=h_, w=w_)
                        # h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
                        return h

            def step_model(
                video_features,
                rgb_video,
                action,
                state,
                camera_extrinsics,
                padding_mask,
                dt,
                view_list,
                embodiment_ids,
                precomputed_rates=None,
                grad_scale=1.0,
                microbatch_mode="final",
            ):
                rates = defaultdict(float)
                # When the outer-iter loop pre-advances the LR/WD schedulers
                # (per-iter, not per-microbatch), it passes the resulting rates
                # in via `precomputed_rates`.  Otherwise (eval, or
                # back-compat) we leave LR/WD logging at 0.0 as before.
                if precomputed_rates is not None:
                    for k, v in precomputed_rates.items():
                        rates[k] = v
                else:
                    rates["info/transition_model/lr"] = 0.0
                    rates["info/transition_model/wd"] = 0.0
                    for name, head in world_model.heads.items():
                        rates[f"info/{name}/lr"] = 0.0
                        rates[f"info/{name}/wd"] = 0.0
                # --

                # Step 1. Forward
                video_f = video_features.to(dtype)
                if encoder_layernorm:
                    video_f = F.layer_norm(video_f, (video_f.size(-1),))
                    video_features = F.layer_norm(video_features, (video_features.size(-1),))
                video_f_target = video_f
                curr_action = action
                curr_state = state
                total_stats = {}
                # Summary of the code below (almost every step is optional):
                # 1. Train heads on encoder features
                # 2. Train predictor to predict one step with teacher forcing of encoder features
                # 3. Optionally, train heads on features from predictor
                # 4. Select a history (T steps from encoder + 1 step from predictor);
                #       train predictor to autoregressively predict the second step, third, fourth, etc.
                #       with its own previous prediction as input. By default, 2 rollout steps
                #       are done during training
                # (No evaluation happens here: forward-prediction eval runs as a standalone script.)

                # Any combination of predictor, image head, state head can be trained simultaneously
                # however, for efficiency the configs specify only one model to be trained per training job

                # 1. TRAIN HEADS ON FEATURES FROM ENCODER
                if "image_head" in world_model.heads:
                    if "image_head" in world_model.heads and image_head_type == "dit":
                        diffusion_model = random.choice(diffusion_models)
                        world_model.heads["image_head"].model.module.gaussian_diffusion = diffusion_model
                    target_rgb_video = world_model.heads["image_head"].preprocess_rgb(rgb_video)
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        encoder_image_losses = world_model.heads["image_head"].compute_loss(
                            video_f, target_rgb_video, padding_mask
                        )
                    rates["info/image_head/examples_seen"] += video_f.shape[0] * video_f.shape[1]
                    encoder_image_losses = {k: v.mean() for k, v in encoder_image_losses.items()}
                    # the weight is like that because we train the head on encoder features and potentially on predictor too  # noqa: E501
                    loss = encoder_image_losses["loss"] / (train_heads_on_predictor * world_model.rollout_steps + 1)
                    world_model.heads["image_head"].backward(loss * grad_scale)
                    encoder_image_losses = {"encoder_image_" + k: v.item() for k, v in encoder_image_losses.items()}
                    total_stats.update(encoder_image_losses)
                if (
                    train_predictor
                    or train_heads_on_predictor
                ):
                    # 2. TRAIN PREDICTOR TO PREDICT ONE STEP IN THE FUTURE USING TEACHER FORCING
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        # NOTE: curr_state is proprio and it is used optionally, based on "state_tokens" arg
                        logger.debug(f"start computing forward pass of the predictor")  # noqa: F541
                        if video_f.shape[1] == (curr_action.shape[1] + 1):
                            # Temporary workaround - for some configs (e.g. decoder training)
                            # we have one less action than the num. of frames fo we cut the last state
                            video_f = video_f[:, :-1]
                            video_f_target = video_f_target[:, :-1]
                            padding_mask = padding_mask[:, :-1]
                        (predicted_video_features, _, predicted_states, cache) = world_model.forward(
                            video_f,
                            curr_action,
                            curr_state,
                            camera_extrinsics,
                            view_map=view_list,
                            dt=dt,
                            embodiment_ids=embodiment_ids,
                        )
                        losses = world_model.compute_loss(
                            predicted_video_features,
                            video_f_target,
                            state=curr_state,
                            pred_state=predicted_states,
                            padding_mask=padding_mask,
                            shift=1,
                            view_map=view_list,
                            apply_view_loss_weights=True,
                        )
                        logger.debug(f"done computing forward pass of the predictor")  # noqa: F541
                    loss = losses["loss"]
                    # Step 2. Backward & step
                    if train_predictor:
                        logger.debug(f"start computing backward pass of the predictor")  # noqa: F541
                        # Use no_sync() for the teacher-forcing backward since we
                        # will do another forward-backward (rollout) before the
                        # optimizer step.  This defers the DDP all-reduce to the
                        # rollout backward, avoiding a redundant gradient sync and
                        # eliminating a window where in-flight NCCL comms can race
                        # with subsequent CUDA kernels (e.g. attention mask
                        # generation) at multi-node scale.
                        if world_model.rollout_steps > 1 and hasattr(world_model.transition_model, "no_sync"):
                            with world_model.transition_model.no_sync():
                                world_model.backward((loss / 2) * grad_scale)
                        else:
                            world_model.backward((loss / 2) * grad_scale)
                        logger.debug(f"done computing backward pass of the predictor")  # noqa: F541

                    # 3. TRAIN HEADS ON FEATURES FROM PREDICTOR
                    if train_heads_on_predictor:
                        if "image_head" in world_model.heads:
                            target_rgb_video = world_model.heads["image_head"].preprocess_rgb(rgb_video)
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                tubelet = world_model.heads["image_head"].unwrapped.tubelet_size
                                slice_ = tubelet
                                predictor_image_losses = world_model.heads["image_head"].compute_loss(
                                    predicted_video_features[:, :-1].detach(),
                                    target_rgb_video[:, slice_:],
                                    padding_mask,
                                )
                            rates["info/image_head/examples_seen"] += predicted_video_features.shape[0] * (
                                predicted_video_features.shape[1] - 1
                            )
                            predictor_image_losses = {k: v.mean() for k, v in predictor_image_losses.items()}
                            world_model.heads["image_head"].backward(predictor_image_losses["loss"] * grad_scale)
                            predictor_image_losses = {
                                "predictor_image_" + k: v.item() for k, v in predictor_image_losses.items()
                            }
                            total_stats.update(predictor_image_losses)
                    # 4. TRAIN PREDICTOR ON FURTHER AUTOREGRESSIVE ROLLOUT
                    # TODO: add training heads on predictor here
                    if world_model.rollout_steps > 1 and train_predictor:
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            total_steps = int(padding_mask.sum(1).min().item())
                            suffix_len = world_model.predictor_window_steps
                            if world_model.train_rollout_prefixes == "random":
                                # we choose one prefix randomly
                                logger.debug(
                                    f"selecting prefix between 0 and {total_steps + 1 - world_model.rollout_steps - (suffix_len - 1)}"  # noqa: E501
                                )
                                prefixes = torch.randint(
                                    total_steps + 1 - world_model.rollout_steps - (suffix_len - 1), size=(1,)
                                )
                            elif world_model.train_rollout_prefixes == "all":
                                # we interate over all possible prefixes
                                prefixes = list(range(total_steps + 1 - world_model.rollout_steps - (suffix_len - 1)))
                            # Record the realized prefix lengths (= t+1 for each
                            # sampled t) for the FLOPs counter, so the wandb log
                            # reflects the ACTUAL compute done this microbatch
                            # (not the expected value over the uniform prefix
                            # distribution). See the `_realized_ar_prefix_lens`
                            # accumulator declared just before the training loop.
                            for _p in prefixes:
                                _realized_ar_prefix_lens.append(int(_p) + 1)
                            stats = defaultdict(list)
                            logger.debug(
                                f"running autoregressive training with prefixes {prefixes}; padding mask: {padding_mask.sum(1)}"  # noqa: E501
                            )
                            for t in prefixes:
                                # here we select t steps from encoder and one step from predictor and concat them over time  # noqa: E501
                                wm_state = world_model.select_history(
                                    video_f,
                                    curr_action,
                                    predicted_video_features,
                                    states=curr_state,
                                    predicted_states=predicted_states,
                                    camera_extrinsics=camera_extrinsics,
                                    dt=dt,
                                    prefix_len=t + 1,
                                )
                                # current_cache = cache.select_history
                                # wm_state = WorldModelState.from_cache(cache=current_cache, features=predicted_video_features[:, t:t+1])  # noqa: E501
                                # TODO: stopped here
                                rollout_losses, _ = world_model.rollout(
                                    wm_state,
                                    # actions for rollout
                                    actions_suffix=curr_action[:, t + 1 :],
                                    # target states for loss (we start from the t+2 th since we have t+1 from encoder and one from predictor as input  # noqa: E501
                                    # so the next step predictor predicts is the t+2 th)
                                    features_suffix=video_f[:, t + 2 :],
                                    states_suffix=curr_state[:, t + 2 :],
                                    dt=dt[:, t + 2 :],
                                    action_noise=0.0,
                                    run_backward=train_predictor,
                                    # we have `len(prefixes)` loss terms so weight them equally
                                    # grad_scale folds in any per-microbatch grad-accumulation factor.
                                    loss_weight=grad_scale * 1.0 / (len(prefixes) * 2),
                                    # here we slice t+1 to the second last because
                                    # padding mask is the input mask not the output mask.
                                    # to make it the output mask we need to shift it forward by 1 (via slice [:-1])
                                    # we slice from t+1 because features suffix is sliced from t+2 but since shift the mask  # noqa: E501
                                    # forward we slice from t+1
                                    padding_mask=padding_mask[:, t + 1 : -1],
                                    view_map=view_list,
                                    embodiment_ids=embodiment_ids,
                                    pad_to_length=num_frames if _use_compiled_padding[0] else None,
                                    apply_view_loss_weights=True,
                                )
                                for k in rollout_losses:
                                    stats[k].append(rollout_losses[k])
                            stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                            rollout_result = {}
                            for k in stats:
                                for j in range(len(stats[k])):
                                    rollout_result[f"train_rollout/{k}/{j + 2}"] = stats[k][j].item()
                    else:
                        rollout_result = {}
                    total_stats.update(rollout_result)
                else:
                    loss = 0.0
                    losses = {}
                grad_stats, optim_stats = {}, {}
                # so far, we only computed losses and ran .backward() so accumulate gradients on several objectives. Below is the  # noqa: E501
                # only place where we perform an actual optimization step
                # When training with grad accumulation across microbatches, we
                # only call optimization_step on the FINAL microbatch of the
                # outer iter; intermediate microbatches just accumulate grads.
                if microbatch_mode == "final":
                    for name in world_model.heads.keys():
                        # todo: per head grad and optim stats
                        grad_stats[name], optim_stats[name] = world_model.heads[name].optimization_step()
                    if train_predictor:
                        logger.debug(f"start computing optimization step of the predictor")  # noqa: F541
                        grad_stats["optim/grad_norm/transition_model"], optim_stats["transition_model"] = (
                            world_model.optimization_step()
                        )
                        logger.debug(f"done computing optimization step of the predictor")  # noqa: F541
                    for key in list(grad_stats.keys()):
                        grad_stats[f"optim/grad_norm/{key}"] = (
                            grad_stats[key].global_norm if grad_stats[key] is not None else 0.0
                        )
                        del grad_stats[key]
                    for key in list(optim_stats.keys()):
                        optim_stats[f"optim/{key}/first_moment"] = (
                            optim_stats[key].get("exp_avg").avg if optim_stats[key] is not None else 0.0
                        )
                        optim_stats[f"optim/{key}/second_moment"] = (
                            optim_stats[key].get("exp_avg_sq").avg if optim_stats[key] is not None else 0.0
                        )
                        del optim_stats[key]
                total_stats.update(grad_stats)
                total_stats.update(optim_stats)
                total_stats.update(dict(rates))
                total_stats.update({"sequence_length": padding_mask.sum(1).mean().item()})
                # this is to monitor that we did not accidentally add padded sequences in
                # sometimes vjepa has features with 0. activations so we do a more robust check here
                total_stats.update(
                    {
                        "observed_length": ((video_f != 0.0).sum(-1) > 1400)
                        .all(-1)
                        .all(-1)
                        .all(-1)
                        .float()
                        .sum(-1)
                        .mean()
                        .item()
                    }
                )
                total_stats.update(
                    {
                        "min_observed_length": ((video_f != 0.0).sum(-1) > 1400)
                        .all(-1)
                        .all(-1)
                        .all(-1)
                        .float()
                        .sum(-1)
                        .min()
                        .item()
                    }
                )
                return (
                    float(loss),
                    losses,
                    total_stats,
                )

            # ---------------------------------------------------------------
            # Per-outer-iter scheduler advance — schedulers must step exactly
            # ONCE per optimizer step, never per microbatch, so that LR/WD
            # stay in lockstep across DDP ranks regardless of each rank's
            # per-group K (grad_accum_steps).
            # ---------------------------------------------------------------
            precomputed_rates = {}
            if train_predictor:
                precomputed_rates["info/transition_model/lr"] = world_model.scheduler.step()
                precomputed_rates["info/transition_model/wd"] = world_model.wd_scheduler.step()
            for name, head in world_model.heads.items():
                precomputed_rates[f"info/{name}/lr"] = head.scheduler.step()
                precomputed_rates[f"info/{name}/wd"] = head.wd_scheduler.step()

            # ---------------------------------------------------------------
            # K microbatches per outer iter.  Microbatches 1..K-1 run under
            # transition_model.no_sync() (and head.model.no_sync() for any
            # DDP-wrapped head) so that NO bucket all-reduce is enqueued for
            # them.  Microbatch K runs without no_sync() — its (rollout)
            # backward triggers the single set of bucket all-reduces for the
            # whole outer iter.  See the plan §4 for the ordering proof.
            # ---------------------------------------------------------------
            from contextlib import ExitStack

            grad_scale_per_mb = 1.0 / float(my_grad_accum)
            encoder_time_total_ms = 0.0
            fwdbwd_time_total_ms = 0.0
            microbatch_times_ms = []
            last_loss = 0.0
            last_losses = {}
            last_rollout_stats = {}
            # Clear the per-iter realized-AR-prefix accumulator; step_model
            # appends one entry per `prefixes` element on each call.
            _realized_ar_prefix_lens.clear()
            for mb_idx in range(my_grad_accum):
                is_last_mb = mb_idx == my_grad_accum - 1
                mb_start = time.time()
                (
                    rgb_video,
                    raw_rgb_video,
                    action,
                    state,
                    camera_extrinsics,
                    padding_mask,
                    view_list,
                    dt,
                    start,
                    embodiment_names,
                    index,
                ) = get_batch()
                # TP+SP: all tp ranks in a TP group must process the IDENTICAL
                # sample. The sampler already gives them the same shard index,
                # but per-rank augmentation / temporal-start RNG can still
                # diverge, so broadcast the frame tensors from the group's
                # tp_rank-0 (global rank = rank - my_tp_rank). Non-root ranks
                # still advanced their loaders in lockstep above; we just
                # overwrite their decoded tensors with the root's.
                if tp_size > 1:
                    _tp_src = rank - my_tp_rank
                    _bcast = [rgb_video, raw_rgb_video, action, state, camera_extrinsics, padding_mask, dt]
                    for _bi, _bt in enumerate(_bcast):
                        if torch.is_tensor(_bt):
                            _btc = _bt.contiguous()
                            dist.broadcast(_btc, src=_tp_src, group=tp_group)
                            _bcast[_bi] = _btc
                    rgb_video, raw_rgb_video, action, state, camera_extrinsics, padding_mask, dt = _bcast
                if world_model.model.is_multiembodiment:
                    embodiment_ids = []
                    for emb in embodiment_names:
                        if emb not in world_model.model.embodiment_list:
                            raise ValueError(f"Unknown embodiment: {emb}")
                        embodiment_ids.append(world_model.model.embodiment_list.index(emb))
                    embodiment_ids = np.array(embodiment_ids)
                    embodiment_ids = torch.from_numpy(embodiment_ids).to(rgb_video.device)
                else:
                    embodiment_ids = None
                video_features, encoder_t = gpu_timer(lambda: forward_target_encoder(rgb_video))
                encoder_time_total_ms += encoder_t

                # Build the no_sync() context stack for non-final microbatches.
                # `fsdp_or_ddp_no_sync` picks the correct mechanism per module:
                #   - FSDP2-wrapped: set_requires_gradient_sync(False)
                #   - DDP-wrapped:    module.no_sync()
                #   - plain (debug):  no-op
                ctx_stack = ExitStack()
                if not is_last_mb:
                    ctx_stack.enter_context(fsdp_or_ddp_no_sync(world_model.transition_model))
                    for hname in world_model.heads.keys():
                        hmodel = world_model.heads[hname].model
                        ctx_stack.enter_context(fsdp_or_ddp_no_sync(hmodel))
                with ctx_stack:
                    try:
                        (mb_loss, mb_losses, mb_rollout_stats), mb_gpu_etime_ms = gpu_timer(
                            lambda: step_model(
                                video_features,
                                raw_rgb_video,
                                action,
                                state,
                                camera_extrinsics,
                                padding_mask,
                                dt,
                                view_list,
                                embodiment_ids,
                                precomputed_rates=precomputed_rates,
                                grad_scale=grad_scale_per_mb,
                                microbatch_mode="final" if is_last_mb else "intermediate",
                            )
                        )
                    except RuntimeError as e:
                        logger.info(f"Got runtime error: {e}; saving the batch")
                        torch.save(
                            {
                                "video_features": video_features,
                                "rgb_video": rgb_video,
                                "raw_rgb_video": raw_rgb_video,
                                "action": action,
                                "state": state,
                                "camera_extrinsics": camera_extrinsics,
                                "padding_mask": padding_mask,
                                "view_list": view_list,
                            },
                            os.path.join(folder, "runtime_error_batch.pth"),
                        )
                        raise e
                fwdbwd_time_total_ms += mb_gpu_etime_ms
                microbatch_times_ms.append((time.time() - mb_start) * 1000.0)
                last_loss = mb_loss
                last_losses = mb_losses
                last_rollout_stats = mb_rollout_stats

            loss = last_loss
            losses = last_losses
            rollout_stats = last_rollout_stats
            encoder_time = encoder_time_total_ms
            gpu_etime_ms = fwdbwd_time_total_ms
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            # ---------------------------------------------------------------
            # Per-view-group timing metrics (one all_gather_object per outer
            # iter; never per microbatch — preserves the "exactly one set of
            # collectives per outer iter" invariant for the predictor's DDP
            # bucket all-reduces; this gather is a separate object collective
            # on its own).  Only rank 0 logs aggregated values to wandb.
            # ---------------------------------------------------------------
            microbatches_total_ms = float(sum(microbatch_times_ms))
            microbatch_ms_max = float(max(microbatch_times_ms)) if microbatch_times_ms else 0.0
            microbatch_ms_min = float(min(microbatch_times_ms)) if microbatch_times_ms else 0.0
            microbatch_ms_mean = (
                microbatches_total_ms / float(len(microbatch_times_ms)) if microbatch_times_ms else 0.0
            )
            # The optimizer step happens inside the LAST microbatch's
            # step_model call (gpu_timer-wrapped). Approximate optim time as
            # the difference between the last microbatch's wall clock and
            # the mean of the prior microbatches (which had no optim step).
            if len(microbatch_times_ms) >= 2:
                prior_mean = sum(microbatch_times_ms[:-1]) / float(len(microbatch_times_ms) - 1)
                optim_step_ms = max(microbatch_times_ms[-1] - prior_mean, 0.0)
            else:
                optim_step_ms = 0.0
            sync_wait_ms = max(iter_elapsed_time_ms - microbatches_total_ms, 0.0)
            # ---------------------------------------------------------------
            # Realized predictor FLOPs for THIS iter on THIS rank.
            # Teacher-forced contribution is constant; AR contribution depends
            # on the prefix lengths step_model actually sampled this iter
            # (captured in `_realized_ar_prefix_lens`). Falls back to the
            # expected value (`predictor_flops_iter`) if the iter wasn't a
            # training iter (e.g., eval/skip).
            # ---------------------------------------------------------------
            if my_grad_accum > 0:
                _tf_iter = int(my_grad_accum) * int(teacher_forced_flops_per_microbatch)
                _ar_iter = int(
                    compute_ar_flops_for_prefix_lens(
                        predictor_cfg=_predictor_cfg,
                        trainer_cfg=_trainer_cfg,
                        batch_size=int(my_batch_size),
                        runtime_v=int(my_num_views),
                        prefix_lens=_realized_ar_prefix_lens,
                        include_backward=True,
                    )
                )
                predictor_flops_iter_realized = _tf_iter + _ar_iter
            else:
                predictor_flops_iter_realized = int(predictor_flops_iter)
            local_metrics = {
                "group": my_cell_name,
                "microbatches_total_ms": microbatches_total_ms,
                "microbatch_ms_max": microbatch_ms_max,
                "microbatch_ms_min": microbatch_ms_min,
                "microbatch_ms_mean": microbatch_ms_mean,
                "sync_wait_ms": sync_wait_ms,
                "optim_step_ms": float(optim_step_ms),
                "encoder_ms": float(encoder_time_total_ms),
                "predictor_fwd_bwd_ms": float(fwdbwd_time_total_ms),
                "iter_elapsed_ms": float(iter_elapsed_time_ms),
                "grad_accum_steps": int(my_grad_accum),
                "num_views": int(my_num_views),
                "world_size": int(my_group_size),
                "examples_seen_per_iter": int(my_grad_accum * my_batch_size * my_num_views),
                "predictor_flops_iter": int(predictor_flops_iter_realized),
            }
            per_group_wandb_dict = {}
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, local_metrics)
            else:
                gathered = [local_metrics]
            if rank == 0:
                # Group by group name and reduce.
                groups_data: dict = {}
                for g in gathered:
                    if g is None:
                        continue
                    gn = g["group"]
                    groups_data.setdefault(gn, []).append(g)
                for gn, glist in groups_data.items():
                    n = len(glist)

                    def _mean(key):
                        return sum(float(x[key]) for x in glist) / n

                    def _max(key):
                        return max(float(x[key]) for x in glist)

                    def _min(key):
                        return min(float(x[key]) for x in glist)

                    per_group_wandb_dict[f"time/{gn}/microbatches_total_ms"] = _mean("microbatches_total_ms")
                    per_group_wandb_dict[f"time/{gn}/microbatch_ms_max"] = _max("microbatch_ms_max")
                    per_group_wandb_dict[f"time/{gn}/microbatch_ms_min"] = _min("microbatch_ms_min")
                    per_group_wandb_dict[f"time/{gn}/microbatch_ms_mean"] = _mean("microbatch_ms_mean")
                    per_group_wandb_dict[f"time/{gn}/sync_wait_ms"] = _mean("sync_wait_ms")
                    per_group_wandb_dict[f"time/{gn}/optim_step_ms"] = _mean("optim_step_ms")
                    per_group_wandb_dict[f"time/{gn}/encoder_ms"] = _mean("encoder_ms")
                    per_group_wandb_dict[f"time/{gn}/predictor_fwd_bwd_ms"] = _mean("predictor_fwd_bwd_ms")
                    per_group_wandb_dict[f"time/{gn}/iter_elapsed_ms"] = _mean("iter_elapsed_ms")
                    per_group_wandb_dict[f"info/{gn}/grad_accum_steps"] = int(glist[0]["grad_accum_steps"])
                    per_group_wandb_dict[f"info/{gn}/num_views"] = int(glist[0]["num_views"])
                    per_group_wandb_dict[f"info/{gn}/world_size"] = int(glist[0]["world_size"])
                    per_group_wandb_dict[f"info/{gn}/examples_seen_per_iter"] = int(glist[0]["examples_seen_per_iter"])
                    # Per-group per-rank FLOPs are now stochastic (AR prefix is
                    # sampled per microbatch on each rank). Report the mean and
                    # min/max within the group so the value is interpretable.
                    per_group_wandb_dict[f"info/{gn}/flops_per_iter_per_rank"] = _mean("predictor_flops_iter")
                    per_group_wandb_dict[f"info/{gn}/flops_per_iter_per_rank_min"] = _min("predictor_flops_iter")
                    per_group_wandb_dict[f"info/{gn}/flops_per_iter_per_rank_max"] = _max("predictor_flops_iter")
                # ---- per-group load-balance summary (console; visible with
                # wandb off). `compute` = sum of this group's microbatch times;
                # `idle` = sync_wait (time waiting at the DDP all-reduce for the
                # slowest rank). Balanced K across resolution/view groups
                # minimizes the max idle.
                if (itr % log_freq) == 0:
                    _bal = []
                    for gn in sorted(groups_data.keys()):
                        _bal.append(
                            f"{gn}(K={int(per_group_wandb_dict[f'info/{gn}/grad_accum_steps'])}): "
                            f"mb_min={per_group_wandb_dict[f'time/{gn}/microbatch_ms_min']:.0f}ms "
                            f"compute={per_group_wandb_dict[f'time/{gn}/microbatches_total_ms']:.0f}ms "
                            f"idle={per_group_wandb_dict[f'time/{gn}/sync_wait_ms']:.0f}ms "
                            f"iter={per_group_wandb_dict[f'time/{gn}/iter_elapsed_ms']:.0f}ms"
                        )
                    logger.info("[balance] " + " | ".join(_bal))
                # ---------------- cumulative predictor FLOPs ---------------- #
                # Sum across ALL gathered ranks (not per group) so the counter
                # reflects total predictor work done by the whole world this
                # iter. This automatically respects the mixed-V partition.
                world_flops_iter_gathered = int(sum(int(g["predictor_flops_iter"]) for g in gathered if g is not None))
                cumulative_flops += world_flops_iter_gathered
                per_group_wandb_dict["train/total_flops"] = int(cumulative_flops)
                # PetaFLOPS-day = 1e15 FLOP/s * 86400 s = 8.64e19 FLOPs.
                # Canonical compute axis for scaling-laws plots
                # (Kaplan et al., Hoffmann et al.).
                per_group_wandb_dict["train/total_pflops_days"] = float(cumulative_flops) / 8.64e19
                per_group_wandb_dict["train/flops_per_iter"] = int(world_flops_iter_gathered)
            if wandb_logger is not None:
                wandb_logger.log(
                    {
                        "epoch": epoch + 1,
                        "itr": itr,
                        "global_step": itr + ipe * epoch,
                        **losses,
                        **rollout_stats,
                        **per_group_wandb_dict,
                        "gpu_etime_ms": gpu_etime_ms,
                        "iter_elapsed_time_ms": iter_elapsed_time_ms,
                        "encoder_time_ms": encoder_time,
                    },
                    step=itr + ipe * epoch,
                )
            if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                logger.info(
                    f"[{epoch + 1}, {itr:5d}] loss: {loss:.3f} | "
                    f"[mem: {torch.cuda.max_memory_allocated() / 1024.0**2:.2e}] "
                    f"[enc: {encoder_time:.1f} ms]"
                    f"[gpu: {gpu_etime_ms:.1f} ms]"
                    f"[wall: {iter_elapsed_time_ms:.1f} ms]"
                )
            assert not np.isnan(loss), "loss is nan"
            if _robo_prof is not None:
                _robo_prof.step()
        if _robo_prof is not None:
            _robo_prof.stop()
        # -- Save Last
        # NOTE: save_checkpoint is called on EVERY rank (not just rank 0).
        # Under the legacy DDP path the function early-returns for rank!=0,
        # preserving the old behaviour (only rank 0 writes the .pth.tar).
        # Under FSDP2 the underlying torch.distributed.checkpoint.save is
        # a collective and *requires* every rank to participate; previously
        # gating this on rank==0 caused a hang at the first save.
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            # Stash cumulative_flops inside the checkpoint's `stats` dict so it
            # is saved inline next to model/opt/epoch under both .pth.tar and
            # FSDP2 metadata.pt code paths (TrainableModel.save_checkpoint
            # writes `stats` as a top-level key in the saved dict).  Only
            # rank 0 has the authoritative counter; others save stats=None.
            _ckpt_stats = {"cumulative_flops": int(cumulative_flops)} if rank == 0 else None
            world_model.save_checkpoint(epoch + 1, latest_path, stats=_ckpt_stats, rank=rank)
            # if 'SLURM_JOB_ID' in os.environ:
            #     subprocess.check_call(['scancel', '--signal=USR2', os.environ['SLURM_JOB_ID']])
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"{tag}-e{epoch}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                world_model.save_checkpoint(epoch + 1, save_every_path, stats=_ckpt_stats, rank=rank)
