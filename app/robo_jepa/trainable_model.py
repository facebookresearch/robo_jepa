# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import torch

from src.utils.adamw import AdamW as RAdamW
from src.utils.logging import adamw_logger, grad_logger
from src.utils.schedulers import (
    ConstantWDSchedule,
    CosineWDSchedule,
    LinearCooldownSchedule,
    WarmupCosineSchedule,
)

# FSDP2-aware checkpoint helpers (used when self.model has been wrapped
# with `fully_shard`).  Imports are cheap; the helpers themselves are
# called only when `is_fsdp2_module(self.model)` is true.
from app.robo_jepa.fsdp_utils import (
    is_fsdp2_module,
    load_dcp_or_legacy_checkpoint,
    save_dcp_checkpoint,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class TrainableModel(ABC):
    def __init__(self, model, train_config=None, device="cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = None
        self.scaler = None
        self.scheduler = None
        self.wd_scheduler = None
        self.mixed_precision = None
        self.clip_grad = None
        self.use_radamw = None
        self.train_config = train_config

    def init_opt(
        self,
        use_radamw=False,
        betas=(0.9, 0.999),
        eps=1e-8,
        ipe_scale=1.25,
        weight_decay=1e-6,
        final_weight_decay=1e-6,
        final_lr=0.0,
        start_lr=0.0,
        ref_lr=1e-3,
        warmup=2,
        num_epochs=90,
        iterations_per_epoch=1000,
        mixed_precision=True,
        clip_grad=1.0,
        dtype=None,
        cooldown_enabled=False,
        cooldown_num_steps=5000,
    ):
        self.mixed_precision = mixed_precision
        self.clip_grad = clip_grad
        self.use_radamw = use_radamw

        # ── μP-aware param-group construction.  When the underlying module
        # exposes `mup_enabled = True` (set by VisionTransformerPredictor when
        # `model.transition_model.config.mup_base_width` is configured) we
        # build 4 param groups with explicit `lr_scale` fields; otherwise we
        # fall back to the legacy 2-group construction (bit-stable).
        # The scheduler reads `lr_scale` and applies it on every `step()`.
        mup_enabled = bool(getattr(self.model, "mup_enabled", False))
        mup_width_mult = float(getattr(self.model, "mup_width_mult", 1.0))
        # Defensive guard against the head double-shrink bug:
        # the optimizer-side `lr_scale = 1/m` we apply to the `head_decay`
        # group below ONLY composes correctly with `mup_head_mode = "init_only"`
        # (where the head's `1/m` lives entirely in init).  With
        # `fwd_mult` or `both`, the predictor *also* multiplies the head
        # output by `1/m` at forward, which combined with our optimizer
        # `lr_scale = 1/m` would double-shrink the head's effective update.
        # See the cross-check against Microsoft's MuAdam (`mup/optim.py`):
        # MuReadout is classified as vector-like there and gets NO LR scaling
        # — the 1/m happens entirely at fwd via MuReadout.  Our `init_only`
        # path is the dual reformulation: 1/m at init AND 1/m on LR.  The
        # other modes need a different optimizer config.  Until that is
        # implemented, refuse to build the optimizer with them.
        mup_head_mode = str(getattr(self.model, "mup_head_mode", "init_only"))
        if mup_enabled and mup_head_mode != "init_only":
            raise NotImplementedError(
                f"mup_head_mode={mup_head_mode!r} is not supported by the "
                f"optimizer-side LR shrink in TrainableModel.init_opt. "
                f"The current 'head_decay' group has lr_scale = 1/m, which "
                f"composes correctly ONLY with mup_head_mode='init_only' "
                f"(init-only 1/m, no forward-time multiplier). Modes "
                f"'fwd_mult' and 'both' would double-shrink the head update "
                f"by combining the forward 1/m with the optimizer 1/m. "
                f"To enable them, drop the head group's lr_scale to 1.0 "
                f"(matching Microsoft MuAdam's MuReadout-as-vector-like rule)."
            )

        # Helper: classify (name, param) into one of {input, hidden, head, no_decay}.
        # Bias / 1-D parameters always end up in `no_decay`.  For trunk
        # parameters we look up the parent module's `_mup_role` attribute
        # (set by VisionTransformerPredictor._tag_mup_roles); modules that
        # weren't tagged (e.g. heads outside the predictor) default to
        # "input" treatment.
        named_modules = dict(self.model.named_modules())

        def _role_for(name: str, p: torch.Tensor) -> str:
            if ("bias" in name) or (len(p.shape) == 1):
                return "no_decay"
            # `name` looks like "predictor_blocks.0.attn.qkv.weight" — we
            # want the parent module name, which is everything before the
            # final dotted segment.
            parent_name = name.rsplit(".", 1)[0]
            mod = named_modules.get(parent_name)
            return getattr(mod, "_mup_role", "input") if mod is not None else "input"

        if mup_enabled:
            # Materialise the per-name classification once — generators
            # are exhausted on first iter and we need to enumerate twice
            # (logging + optimizer state count).
            buckets = {"input": [], "hidden": [], "head": [], "no_decay": []}
            for n, p in self.model.named_parameters():
                buckets[_role_for(n, p)].append(p)
            param_groups = [
                {
                    "params": buckets["input"],
                    "lr_scale": 1.0,
                    "_mup_group": "input_decay",
                },
                {
                    "params": buckets["hidden"],
                    "lr_scale": 1.0 / mup_width_mult,
                    "_mup_group": "hidden_decay",
                },
                {
                    "params": buckets["head"],
                    "lr_scale": 1.0 / mup_width_mult,
                    "_mup_group": "head_decay",
                },
                {
                    "params": buckets["no_decay"],
                    "lr_scale": 1.0,
                    "WD_exclude": True,
                    "weight_decay": 0,
                    "_mup_group": "no_decay",
                },
            ]
            # Drop empty groups (e.g. no `state_proj` when decode_state=False
            # leaves the `head` bucket empty for non-predictor inner models).
            param_groups = [g for g in param_groups if len(g["params"]) > 0]
            logger.info(
                f"[μP] enabled with width_mult={mup_width_mult:.4f}; "
                f"param-group sizes "
                f"input={len(buckets['input'])}, hidden={len(buckets['hidden'])}, "
                f"head={len(buckets['head'])}, no_decay={len(buckets['no_decay'])}"
            )
        else:
            param_groups = []
            param_groups += [
                {"params": (p for n, p in self.model.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
            ]
            param_groups += [
                {
                    "params": (p for n, p in self.model.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                    "WD_exclude": True,
                    "weight_decay": 0,
                },
            ]
        logger.info(f"optimizer epsilon: {eps}")
        if use_radamw:
            logger.info(f"Using Rescaled-AdamW")  # noqa: F541
            optimizer = RAdamW(param_groups, betas=betas, eps=eps)
        else:
            logger.info(f"Using AdamW")  # noqa: F541
            # TP no-sharding mode mixes sharded DTensor (block) params with plain
            # (non-block) params; the optimizer's `foreach` path can't mix the two
            # (aten._foreach_* rejects mixed Tensor/DTensor), so use the per-param
            # path. No-op for the non-TP path (foreach=None = torch default).
            _tp = getattr(self.model, "_tp_size", 1) > 1
            optimizer = torch.optim.AdamW(
                param_groups, betas=betas, eps=float(eps), foreach=(False if _tp else None)
            )
        logger.info(
            f"total trainable params: {sum([np.prod(p.shape) for gr in optimizer.param_groups for p in gr['params']]).item()}"  # noqa: E501
        )
        # ── LR / WD scheduler dispatch ─────────────────────────────────────
        # Default: WSD-style stable phase via WarmupCosineSchedule + CosineWDSchedule
        # (the historical training recipe).
        # Cooldown phase: replace BOTH with the linear-cooldown pair so the
        # LR linearly decays from `ref_lr` → 0 over `cooldown_num_steps`
        # steps and weight-decay is held constant.  Used to take an
        # intermediate scaling-WM checkpoint (saved during the stable phase)
        # and finish it with a clean cooldown — see `meta.cooldown:` in the
        # YAML.  Per-group `lr_scale` (μP) is applied inside both
        # WarmupCosineSchedule and LinearCooldownSchedule.
        if cooldown_enabled:
            logger.info(
                f"[cooldown] enabled — LR will linearly decay from "
                f"ref_lr={ref_lr:g} -> 0 over {cooldown_num_steps} steps; "
                f"WD held constant at {weight_decay:g} (warmup/num_epochs/"
                f"iterations_per_epoch / final_lr / final_wd / ipe_scale "
                f"are IGNORED in cooldown mode)."
            )
            scheduler = LinearCooldownSchedule(
                optimizer,
                num_steps=int(cooldown_num_steps),
                ref_lr=ref_lr,
            )
            wd_scheduler = ConstantWDSchedule(
                optimizer,
                ref_wd=weight_decay,
            )
        else:
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=int(warmup * iterations_per_epoch),
                start_lr=start_lr,
                ref_lr=ref_lr,
                final_lr=final_lr,
                T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
            )
            wd_scheduler = CosineWDSchedule(
                optimizer,
                ref_wd=weight_decay,
                final_wd=final_weight_decay,
                T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
            )
        # GradScaler is only needed for float16; bfloat16 does not suffer from
        # gradient underflow so the scaler is unnecessary overhead and creates
        # graph breaks under torch.compile.
        use_gradscaler = mixed_precision and (dtype == torch.float16)
        scaler = torch.cuda.amp.GradScaler() if use_gradscaler else None
        if mixed_precision and scaler is None:
            logger.info("mixed_precision is enabled but GradScaler is disabled (not float16)")
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.wd_scheduler = wd_scheduler
        return optimizer, scaler, scheduler, wd_scheduler

    def load_checkpoint(self, path, load_optimizer=True, strict=True, dont_load=None):
        # FSDP2 path: auto-detect DCP-dir vs legacy .pth.tar and load via
        # DTensor-aware helpers (set_model_state_dict broadcast_from_rank0).
        if is_fsdp2_module(self.model):
            epoch, stats = load_dcp_or_legacy_checkpoint(
                self.model,
                self.optimizer,
                path,
                load_optimizer=load_optimizer,
                strict=strict,
                dont_load=dont_load,
            )
            return (epoch, stats)

        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        epoch = 0
        epoch = checkpoint["epoch"]

        # -- loading predictor
        pretrained_dict = checkpoint["model"]
        if dont_load is not None:
            for k in dont_load:
                if k in pretrained_dict:
                    del pretrained_dict[k]
        # Drop keys that don't exist in the current model. Without this,
        # strict=True would raise RuntimeError when an old checkpoint contains
        # parameters that have since been removed (e.g. `view_tokens` when the
        # predictor is now built with use_view_tokens=False). We log the
        # filtered keys so silent drops are visible.
        own_keys = set(self.model.state_dict().keys())
        filtered_keys = [k for k in pretrained_dict.keys() if k not in own_keys]
        if filtered_keys:
            logger.info(
                f"Filtering {len(filtered_keys)} key(s) from checkpoint that are absent from the current model: "
                f"{filtered_keys[:8]}{' ...' if len(filtered_keys) > 8 else ''}"
            )
            for k in filtered_keys:
                del pretrained_dict[k]
        msg = self.model.load_state_dict(pretrained_dict, strict=strict)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")
        weights_sum = sum(p.sum().item() for p in self.model.parameters())
        logger.info(f"sum of weights after loading the model: {weights_sum:.5f}")

        # -- loading optimizer
        if self.optimizer is not None and load_optimizer:
            self.optimizer.load_state_dict(
                checkpoint["opt"],
            )
            # first_moment = sum(p['exp_avg'].sum().item() for p in self.optimizer.state.values())
            # second_moment = sum(p['exp_avg_sq'].sum().item() for p in self.optimizer.state.values())
            # step_min = min(p['step'].item() for p in self.optimizer.state.values())
            # step_max = max(p['step'].item() for p in self.optimizer.state.values())
            if self.scaler is not None:
                self.scaler.load_state_dict(
                    checkpoint["scaler"],
                )
            # logger.info(f'loaded optimizers from epoch {epoch}')
            # logger.info(f'loaded optimizer; first moment: {first_moment:.5E}; second moment: {second_moment:.5E}; min step: {step_min}; max step: {step_max}')  # noqa: E501
            logger.info(f"read-path: {path}")
        stats = checkpoint.get("stats", {})
        return (epoch, stats)

    def save_checkpoint(self, epoch, path, stats=None, rank=0):
        # Tensor-parallel (FSDP off) path FIRST: the predictor's params are TP
        # DTensors that we serialize as a plain, eval-loadable `.pth.tar` (not a
        # DCP dir). Gather to a full, component-major (un-repacked) weight dict
        # AND gather the full optimizer state — both are collectives, every rank
        # must participate in the same order. With opt state saved, a TP
        # checkpoint is warm-start / eval / resume(weights) AND optimizer-resumable.
        from torch.distributed.tensor import DTensor  # noqa: F401  (kept for type checks below)

        if getattr(self.model, "_tp_mesh", None) is not None:
            from app.robo_jepa.parallel_utils import tp_full_state_dict, tp_save_optim_dcp

            full_model_sd = tp_full_state_dict(self.model)
            if rank == 0:
                save_dict = {
                    "model": full_model_sd,
                    # Optimizer is NOT gathered here — the full-optimizer all-gather
                    # native-segfaults at 8B. It is saved sharded via DCP below (see
                    # tp_save_optim_dcp), next to this file as `<name>_optim/`.
                    "opt": None,
                    "scaler": None if self.scaler is None else self.scaler.state_dict(),
                    "epoch": epoch,
                    "stats": stats,
                }
                logger.info(f"saving the (TP-gathered) model at epoch {epoch}...")
                try:
                    tmp_path = path + ".tmp"
                    torch.save(save_dict, tmp_path)
                    backup_path = path + ".prev"
                    if os.path.exists(path):
                        if os.path.exists(backup_path):
                            os.remove(backup_path)
                        os.rename(path, backup_path)
                    os.rename(tmp_path, path)
                    logger.info("the model is saved")
                except Exception as e:
                    logger.info(f"Encountered exception when saving checkpoint: {e}")
            # Sharded optimizer save (collective: every rank participates). Robust
            # at 8B — no all-gather. Skipped cleanly when optimizer is None.
            tp_save_optim_dcp(self.model, self.optimizer, path, rank=rank)
            return

        # FSDP2 path: sharded DCP checkpoint under <path>-without-suffix/. Every
        # rank participates (DCP is a collective); rank 0 writes the metadata.pt
        # sidecar. (TP is handled above — its replicate() wrap also satisfies
        # is_fsdp2_module, so the TP check MUST come first.)
        if is_fsdp2_module(self.model):
            save_dcp_checkpoint(
                self.model,
                self.optimizer,
                self.scaler,
                epoch=epoch,
                stats=stats,
                path=path,
                rank=rank,
            )
            return

        if rank != 0:
            return
        save_dict = {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "epoch": epoch,
            "stats": stats,
        }
        logger.info(f"saving the model at epoch {epoch}...")
        try:
            # Atomic save: write to temp file, then rename to avoid corruption
            # if preemption kills the process mid-write
            tmp_path = path + ".tmp"
            torch.save(save_dict, tmp_path)
            # Keep previous checkpoint as backup before overwriting
            backup_path = path + ".prev"
            if os.path.exists(path):
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(path, backup_path)
            os.rename(tmp_path, path)
            logger.info(f"the model is saved")  # noqa: F541
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    def delete_opt(self):
        del self.optimizer
        del self.scaler
        del self.scheduler
        del self.wd_scheduler
        torch.cuda.empty_cache()
        gc.collect()

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass

    def backward(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def _tp_clip_grad_norm(self, max_norm):
        """Global grad-norm clip for the TP no-sharding model, where params are a
        MIX of tp-sharded DTensors (block params) and plain replicated tensors
        (non-block params) — which torch's foreach clip_grad_norm_ rejects.

        Global norm^2 = sum over params of ||grad||^2:
          * DTensor (Shard on tp): each rank holds a different shard, so summing
            local-shard sq-norms across the tp group gives the full param's norm^2.
          * plain (replicated on tp): identical on every tp rank, so divide its
            sq-norm by tp_size to count it exactly once after the tp all-reduce.
        Then scale every grad in place by min(1, max_norm/total_norm).
        """
        import torch.distributed as dist
        from torch.distributed.tensor import DTensor

        tp_group = self.model._tp_mesh.get_group()
        tp_size = int(self.model._tp_size)
        params = [p for p in self.model.parameters() if p.grad is not None]
        if not params:
            return torch.zeros(())
        device = params[0].grad.to_local().device if isinstance(params[0].grad, DTensor) else params[0].grad.device
        local_sq = torch.zeros((), device=device, dtype=torch.float32)
        for p in params:
            g = p.grad
            if isinstance(g, DTensor):
                local = g.to_local()
                local_sq += local.float().pow(2).sum()
            else:
                local_sq += g.float().pow(2).sum() / tp_size
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=tp_group)  # -> global grad norm^2
        total_norm = local_sq.sqrt()
        clip_coef = float((max_norm / (total_norm + 1e-6)).clamp(max=1.0).item())
        if clip_coef < 1.0:
            for p in params:
                g = p.grad
                (g.to_local() if isinstance(g, DTensor) else g).mul_(clip_coef)
        return total_norm

    def optimization_step(self):
        # Tensor-parallel, no-sharding mode: the predictor is NOT wrapped in
        # DDP/replicate/FSDP (those flatten the TP DTensor params at scale and
        # break checkpointing). Instead we average gradients over the
        # data-parallel group MANUALLY here, on the final microbatch (grads are
        # fully accumulated). Params/grads/opt stay replicated over dp (the model
        # fits with headroom), so the params remain clean TP DTensors and the
        # checkpoint gather works. Grad of a DTensor param is sharded on tp and
        # replicated on dp, so reducing the LOCAL shard across dp averages
        # correctly.
        dp_group = getattr(self.model, "_tp_dp_group", None)
        if dp_group is not None:
            import torch.distributed as dist
            from torch.distributed.tensor import DTensor

            dp_size = dist.get_world_size(dp_group)
            if dp_size > 1:
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
                    dist.all_reduce(local, op=dist.ReduceOp.SUM, group=dp_group)
                    local.div_(dp_size)

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.clip_grad:
            dp_group = getattr(self.model, "_tp_dp_group", None)
            if dp_group is not None:
                # TP no-sharding: params are MIXED (sharded DTensor block params +
                # plain non-block params); torch's foreach clip_grad_norm_ can't
                # mix them. Compute the global grad norm manually (all-reduced over
                # the tp group) and scale in place.
                _grad_norm = self._tp_clip_grad_norm(self.clip_grad)
            else:
                _grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            if self.use_radamw and (_grad_norm > self.clip_grad):
                logger.info(f"Gradient spike... skipping update {_grad_norm=}")
                self.optimizer.skip_step()
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.clip_grad:
            grad_stats = grad_logger(self.model.named_parameters())
            grad_stats.global_norm = float(_grad_norm)
        else:
            grad_stats = None
        self.optimizer.zero_grad()
        # print(f'gradient norm after step: {_grad_norm}')
        optim_stats = adamw_logger(self.optimizer)
        return grad_stats, optim_stats
