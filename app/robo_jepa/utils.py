# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch import nn


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# Public V-JEPA 2.1 code (github.com/facebookresearch/vjepa2) via torch.hub. The hub cache lives under
# $TORCH_HOME/hub, so compute nodes without internet reuse it. Override the repo with VJEPA2_HUB_REPO
# (a "owner/name[:ref]" GitHub spec or a local clone).
VJEPA2_HUB_REPO = os.environ.get("VJEPA2_HUB_REPO", "facebookresearch/vjepa2")
VJEPA2_1_ENCODER = "vjepa2_1_vit_gigantic_384"
_REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _hide_local_packages():
    """The vjepa2 repo also has top-level `app`/`src` packages. Hide ours while its hubconf imports them.

    Our `src` is a regular package, so it would shadow the hub's namespace packages even with the hub dir first on
    sys.path; drop every sys.path entry inside our repo for the duration as well.
    """

    def ours(name):
        return name.split(".")[0] in ("app", "src", "evals")

    saved_modules = {k: sys.modules.pop(k) for k in list(sys.modules) if ours(k)}
    saved_path = sys.path[:]
    sys.path[:] = [p for p in sys.path if not Path(p or ".").resolve().is_relative_to(_REPO_ROOT)]
    try:
        yield
    finally:
        # Drop the hub's modules that would shadow ours (e.g. its `src.utils.*`), but keep the ones whose names
        # do not exist in this repo (e.g. `app.vjepa_2_1.*`): torch.compile re-imports them by name when it
        # traces the encoder, and `import_module` finds them in sys.modules before looking at our `app` package.
        for k in [k for k in sys.modules if ours(k)]:
            if k.split(".")[0] in ("app", "src", "evals") and "." not in k or _exists_in_repo(k):
                del sys.modules[k]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _exists_in_repo(module_name):
    """True if `module_name` (dotted) names a package or module inside this repo."""
    p = _REPO_ROOT.joinpath(*module_name.split("."))
    return p.is_dir() or p.with_suffix(".py").is_file()


def load_vjepa2_1_encoder():
    """Build the public V-JEPA 2.1 ViT-G/384 encoder (random init; weights come from build_encoder)."""
    source = "local" if os.path.isdir(VJEPA2_HUB_REPO) else "github"
    with _hide_local_packages():
        # pretrained=False: the hub's download URL is not usable; the same checkpoint is loaded from disk.
        encoder, _ = torch.hub.load(
            VJEPA2_HUB_REPO, VJEPA2_1_ENCODER, source=source, pretrained=False, trust_repo=True, verbose=False
        )
    return encoder


class EncoderWithProj(nn.Module):
    """The frozen V-JEPA 2.1 ViT-G/384 encoder (``version="v21"``, the only supported encoder).

    ``use_proj`` must be False: the projection head belonged to the removed ViT-L (``v1``) encoder.
    """

    def __init__(self, version="v21", use_proj=False, max_num_frames=16):
        super().__init__()
        if version != "v21":
            raise ValueError(f"unsupported encoder version {version!r} (only 'v21')")
        if use_proj:
            raise ValueError("use_proj=True is unsupported: the projection head belonged to the removed v1 encoder")
        self.vit = load_vjepa2_1_encoder()

    def forward(self, x):
        return self.vit(x)


def build_encoder(version="v21", device="cuda", use_proj=False, filename=None, max_num_frames=16):
    encoder = EncoderWithProj(version=version, use_proj=use_proj, max_num_frames=max_num_frames)
    encoder = encoder.to(device)

    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.eval()
    state_dict = torch.load(filename, map_location="cpu", mmap=True)
    state_dict["target_encoder"] = {
        k.replace("module.backbone.", ""): v for k, v in state_dict["target_encoder"].items()
    }
    encoder.vit.load_state_dict(state_dict["target_encoder"])
    return encoder


def sync_dict(ddict):
    keys = sorted(ddict.keys())
    vals = np.array([ddict[k] for k in keys])
    vals = torch.from_numpy(vals).float().cuda()
    dist.barrier()
    dist.all_reduce(vals, op=dist.ReduceOp.AVG)
    vals = vals.cpu().numpy().tolist()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in zip(keys, vals)}


def init_wandb(entity, project, config, folder, name, resume_preempt=False, run_id=None):
    wandb_run_id = Path(folder) / "wandb_run.json"
    wandb_id = None
    if resume_preempt:
        if not wandb_run_id.exists():
            raise Warning("Pre-empted run but previous wandb id not found, would create new run!")
        else:
            wandb_id = json.load(open(wandb_run_id))["wandb_id"]
    elif run_id is not None:
        wandb_id = run_id
    def _do_init(**extra):
        return wandb.init(
            dir=folder,
            name=name,
            entity=entity,
            project=project,
            config=config,
            id=wandb_id,
            resume="allow" if wandb_id is not None else None,
            **extra,
        )

    # wandb.init must NEVER kill training: a transient network timeout reaching
    # the wandb server (CommError "Run initialization has timed out") previously
    # crashed a preemption-resume. Use a generous init_timeout, then fall back to
    # offline mode (logs locally, syncable later), then to a disabled no-op run.
    try:
        wandb_logger = _do_init(settings=wandb.Settings(init_timeout=300))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[wandb] online init failed ({e}); retrying in offline mode")
        try:
            wandb_logger = _do_init(mode="offline", settings=wandb.Settings(init_timeout=300))
        except Exception as e2:  # noqa: BLE001
            logger.warning(f"[wandb] offline init also failed ({e2}); disabling wandb for this run")
            wandb_logger = wandb.init(mode="disabled")
    run_id = wandb_logger.id
    with open(wandb_run_id, "w") as f:
        json.dump({"wandb_id": run_id}, f)
    return wandb_logger
