# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import sys

import torch


def gpu_timer(closure, log_timings=True):
    """Helper to time gpu-time to execute closure()"""
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad
def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    selected_params = []
    first_layer_id = last_layer_id = None
    i = 0
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith(".bias") or len(p.shape) == 1):
            # FSDP2 stores params/grads as DTensor.  Diagnostic norms work on
            # the local shard — they are reported per-rank to wandb anyway.
            g = p.grad
            if hasattr(g, "to_local"):
                g = g.to_local()
            selected_params.append(g)
            if "qkv" in n:
                last_layer_id = i
                if first_layer_id is None:
                    first_layer_id = i
            i += 1
    if not selected_params:
        stats.update(0.0)
        stats.first_layer = stats.last_layer = 0.0
        return stats
    norms = torch.stack(torch._foreach_norm(selected_params, 2.0))
    mean_norm = norms.mean()
    stats.update(mean_norm.item())  # TODO: investigate how to remove this sync point
    if last_layer_id is None or first_layer_id is None:
        stats.first_layer = stats.last_layer = 0.0
    else:
        stats.first_layer = norms[first_layer_id].item()
        stats.last_layer = norms[last_layer_id].item()
    return stats


def adamw_logger(optimizer):
    """logging magintude of first and second momentum buffers in adamw"""
    # TODO: assert that optimizer is instance of torch.optim.AdamW
    state = optimizer.state_dict().get("state")
    exp_avg_stats = AverageMeter()
    exp_avg_sq_stats = AverageMeter()
    exp_avg_t = []
    exp_avg_sq_t = []
    numels = []
    for key in state:
        s = state.get(key)
        ea = s.get("exp_avg")
        eas = s.get("exp_avg_sq")
        # FSDP2: optimizer state lives in DTensors.  Use the local shard for
        # the diagnostic norm — it's already a per-rank average and we don't
        # want a global all-gather here.
        if hasattr(ea, "to_local"):
            ea = ea.to_local()
        if hasattr(eas, "to_local"):
            eas = eas.to_local()
        exp_avg_t.append(ea)
        exp_avg_sq_t.append(eas)
        numels.append(ea.numel())
    if not exp_avg_t:
        return {"exp_avg": exp_avg_stats, "exp_avg_sq": exp_avg_sq_stats}
    n1 = torch.stack(torch._foreach_norm(exp_avg_t, 1.0))
    n2 = torch.stack(torch._foreach_norm(exp_avg_sq_t, 1.0))

    # sync point! maybe consider different way by doing global avg instead
    numels = torch.as_tensor(numels, dtype=n1.dtype, device=n1.device)
    n1 /= numels
    n2 /= numels
    exp_avg_stats.update(n1.mean().item())  # TODO: fix
    exp_avg_sq_stats.update(n2.mean().item())  # TODO: fix
    return {"exp_avg": exp_avg_stats, "exp_avg_sq": exp_avg_sq_stats}


def jepa_rootpath():
    this_file = os.path.abspath(__file__)
    return "/".join(this_file.split("/")[:-3])


def git_information():
    jepa_root = jepa_rootpath()
    try:
        resp = (
            subprocess.check_output(["git", "-C", jepa_root, "rev-parse", "HEAD", "--abbrev-ref", "HEAD"])
            .decode("ascii")
            .strip()
        )
        commit, branch = resp.split("\n")
        return f"branch: {branch}\ncommit: {commit}\n"
    except Exception:
        return "unknown"
