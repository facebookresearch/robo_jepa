# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math


class WarmupCosineSchedule(object):

    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, last_epoch=-1, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(
                self.final_lr,
                self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        # Per-group LR multiplier (μP).  When `lr_scale` is absent (legacy
        # 2-group construction) the default is 1.0, so behaviour is
        # bit-stable with the pre-μP code.  When set by the μP-aware
        # `TrainableModel.init_opt` (hidden / head groups carry
        # `lr_scale = 1/m`), every step writes `new_lr * lr_scale` into
        # the corresponding param-group.
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr * float(group.get("lr_scale", 1.0))

        return new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd


class LinearCooldownSchedule(object):
    """Linear LR cooldown from `ref_lr` down to 0 over `num_steps`.

    Used for the "cooldown" phase of WSD-style training: load an
    intermediate world-model checkpoint that was trained with a stable
    high LR, then continue training while linearly annealing the LR to
    zero over a fixed number of optimizer steps.

    Step mechanics (matching `WarmupCosineSchedule`'s convention that
    `step()` returns the LR that was just set, with `_step=0` before
    the first call):

      * call #1 (i.e. _step transitions 0 -> 1): writes `ref_lr * (1 - 1/num_steps)`
        — the LR begins decaying immediately.
      * call #num_steps:                        writes 0.
      * later calls:                            stay at 0 (clipped).

    Per-group `lr_scale` (set by μP-aware `TrainableModel.init_opt`) is
    multiplied through, so under μP the hidden / head groups still
    receive their `1/m` reduction during cooldown.
    """

    def __init__(self, optimizer, num_steps, ref_lr):
        if num_steps <= 0:
            raise ValueError(f"LinearCooldownSchedule requires num_steps > 0, got {num_steps}")
        self.optimizer = optimizer
        self.num_steps = int(num_steps)
        self.ref_lr = float(ref_lr)
        self._step = 0

    def step(self):
        self._step += 1
        progress = float(min(self._step, self.num_steps)) / float(self.num_steps)
        new_lr = max(0.0, self.ref_lr * (1.0 - progress))
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr * float(group.get("lr_scale", 1.0))
        return new_lr


class ConstantWDSchedule(object):
    """No-op WD scheduler: holds `weight_decay` constant at `ref_wd`.

    Used during cooldown — `LinearCooldownSchedule` only modulates LR;
    weight decay is kept at the value the model was trained with so the
    cooldown phase is a pure LR sweep, not a joint LR + WD sweep.
    Mirrors `CosineWDSchedule`'s `_step` / `step()` API for drop-in use.
    """

    def __init__(self, optimizer, ref_wd):
        self.optimizer = optimizer
        self.ref_wd = float(ref_wd)
        self._step = 0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = self.ref_wd
        return self.ref_wd
