# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AOT Inductor inference engine for RoboJEPA world model.

Provides `AOTWorldModelDispatcher` — a drop-in replacement for `WorldModel.step()`
that dispatches to pre-compiled AOT Inductor `.pt2` programs keyed by (step_idx, batch_size).

Also provides `AOTDecoderDispatcher` for the DiT diffusion decoder, dispatching
compiled programs keyed by (cache_step, update_cache, batch_size).

Key features:
  - Compiles code-only .pt2 artifacts (no baked-in weights, ~3-30 MB each)
  - Shares a single copy of model weights across all program variants
  - Instant cold start (~45 ms load + 5 ms weight injection per program)

Usage:
    from app.robo_jepa.world_model import WorldModel

    # Load both compiled WM and compiled decoder in one call:
    dispatcher = WorldModel.load_pretrained(
        model_dir, device="cuda",
        load_compiled=True, load_compiled_decoder=True,
    )

    # WM rollout
    out, new_state = dispatcher.step(action, state)

    # Decode features → RGB via heads['image_head'] (compiled decoder installed)
    head = dispatcher.heads["image_head"]  # AOTDecoderDispatcher
    rgb, cache = head.decode(
        features, cfg_scale=1.0, denoising_steps=4,
        use_cache=True, cache=cache,
    )

    # Original eager head available at heads['image_head_eager']
"""

from app.robo_jepa.inference.decoder_dispatcher import AOTDecoderDispatcher
from app.robo_jepa.inference.dispatcher import AOTWorldModelDispatcher

__all__ = ["AOTWorldModelDispatcher", "AOTDecoderDispatcher"]
