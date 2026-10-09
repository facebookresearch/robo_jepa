# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AOT Inductor dispatcher for DiT diffusion decoder inference.

Drop-in replacement for ``wm.heads["image_head"].decode()``:

    # Eager:
    rgb, cache = wm.heads["image_head"].decode(
        features, cfg_scale=1.0, denoising_steps=4, use_cache=True, cache=cache,
    )

    # Compiled — same API, same return type:
    rgb, cache = decoder_dispatcher.decode(
        features, cfg_scale=1.0, denoising_steps=4, use_cache=True, cache=cache,
    )

Manages pre-compiled .pt2 programs keyed by (cache_step, update_cache, batch_size),
with shared decoder weights across all variants.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                   AOTDecoderDispatcher                          │
    │                                                                 │
    │  .decode(features, denoising_steps, use_cache, cache, ...)     │
    │       │                                                         │
    │       ├─ Rearrange features     (same as WorldModelDiTHead)     │
    │       ├─ Create noise / q_sample (same as SimpleDiT.sample)     │
    │       ├─ ddim_sample_loop(compiled_model_fn, ...)               │
    │       │     └─ GaussianDiffusion handles all DDIM math          │
    │       │        compiled_model_fn dispatches to .pt2 programs     │
    │       ├─ Rearrange output       (same as WorldModelDiTHead)     │
    │       └─ postprocess_rgb        (Cosmos decode via head)        │
    └─────────────────────────────────────────────────────────────────┘

Artifact directory layout:
    artifact_dir/
        manifest.json
        decoder_weights.pt
        decoder_step0_update0_b1.pt2
        decoder_step0_update1_b1.pt2
        decoder_step4_update0_b16.pt2
        ...
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from app.robo_jepa.inference.aot_compile import load_shared_constants
from app.robo_jepa.inference.decoder_aot_compile import (
    DecoderStepModule,
    build_decoder_example_inputs,
    get_decoder_config,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DEFAULT_CACHE_STEPS = [0, 1, 2, 3, 4]


class _CompiledModelFn:
    """Callable that wraps compiled .pt2 programs to match SimpleDiT.forward().

    GaussianDiffusion calls ``model(x, t, **model_kwargs)`` inside
    ``p_mean_variance``.  This class translates those calls into dispatches
    to the correct compiled program based on (cache_step, update_cache, B).

    It converts between the ``KeyValueCache`` object that GaussianDiffusion
    passes around and the flat (cache_keys, cache_values) tensors that the
    compiled programs expect.
    """

    def __init__(self, dispatcher: "AOTDecoderDispatcher"):
        self._dispatcher = dispatcher

    # ddim_sample_loop_progressive uses ``next(model.parameters()).device``
    # as a fallback device.  We expose a dummy parameter so that call works.
    def parameters(self):
        yield torch.empty(0, device=self._dispatcher._device)

    def __call__(
        self,
        x,
        t,
        condition,
        cache=None,
        use_cache=False,
        update_cache=False,
        cache_size=None,
        **kwargs,
    ):
        d = self._dispatcher

        # ── Determine cache_step ──────────────────────────────────────
        if cache_size is not None:
            cs = int(cache_size) if not isinstance(cache_size, int) else cache_size
        else:
            cs = 0

        B = x.shape[0]

        # ── Convert KeyValueCache → flat tensors ─────────────────────
        if cache is not None:
            cache_keys = torch.stack(cache.keys)  # (depth, B, heads, len, tokens, dim)
            cache_values = torch.stack(cache.values)
        else:
            cache_keys = torch.zeros(
                d._depth,
                B,
                d._num_kv_heads,
                d._cache_len,
                d._frame_tokens,
                d._head_dim,
                device=x.device,
                dtype=torch.bfloat16,
            )
            cache_values = torch.zeros_like(cache_keys)

        # ── Dispatch: torch.compile or AOTI ──────────────────────────
        if d._use_torch_compile:
            if d._use_cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            program = d._get_compiled_fn(cs, update_cache, B)
            target_B = B  # torch.compile doesn't need batch padding
        else:
            program, target_B = d._get_program(cs, update_cache, B)

        # Pad if needed
        if target_B > B:
            pad = lambda t, dim=0: d._pad_to(t, target_B, dim)  # noqa: E731
            x_in = pad(x)
            t_in = pad(t)
            cond_in = pad(condition)
            ck_in = pad(cache_keys, dim=1)
            cv_in = pad(cache_values, dim=1)
        else:
            x_in, t_in, cond_in = x, t, condition
            ck_in, cv_in = cache_keys, cache_values

        # frames rolled out of the cache before this call; an updating call on a full cache rolls one more
        evicted = int(cache.evicted[0]) if cache is not None else 0
        inputs = [
            x_in.to(torch.bfloat16).contiguous(),
            t_in.contiguous(),
            cond_in.to(torch.bfloat16).contiguous(),
            ck_in.to(torch.bfloat16).contiguous(),
            cv_in.to(torch.bfloat16).contiguous(),
        ]
        if d._use_torch_compile or d._time_offset_input:
            inputs.append(torch.tensor(evicted, dtype=torch.long, device=x.device))
        output, new_ck, new_cv = program(*inputs)

        # Clone outputs when using CUDA graphs to prevent overwriting
        # by the next graph replay within the DDIM loop
        if d._use_cudagraphs:
            output = output.clone()
            new_ck = new_ck.clone()
            new_cv = new_cv.clone()

        # Unpad
        if target_B > B:
            output = output[:B]
            new_ck = new_ck[:, :B]
            new_cv = new_cv[:, :B]

        # ── Convert flat tensors → KeyValueCache ─────────────────────
        if use_cache:
            from app.robo_jepa.robo_jepa.models.utils.modules import KeyValueCache

            new_pointer = [cs + 1] if update_cache else [cs]
            new_cache = KeyValueCache(
                layers=d._depth,
                batch=B,
                frame_tokens=d._frame_tokens,
                cache_len=d._cache_len,
                num_heads=d._num_kv_heads,
                head_dim=d._head_dim,
                keys=[new_ck[i] for i in range(d._depth)],
                values=[new_cv[i] for i in range(d._depth)],
                pointer=new_pointer,
                evicted=[evicted + int(update_cache and cs >= d._cache_len)] * d._depth,
            )
            return output, new_cache

        return output


class AOTDecoderDispatcher:
    """AOT Inductor dispatcher for the DiT diffusion decoder.

    Drop-in replacement for ``WorldModelDiTCosmosHead``: call ``.decode()``
    with the same arguments and get the same output.  Internally dispatches
    the DiT forward pass to compiled .pt2 programs while delegating all
    DDIM math to ``GaussianDiffusion.ddim_sample_loop()``.

    Args:
        head: The ``WorldModelDiTCosmosHead`` instance (provides Cosmos
              tokenizer, postprocess_rgb, diffusion config, etc.).
        decoder_config: Config dict from ``get_decoder_config()``.
        artifact_dir: Directory containing .pt2 artifacts.
        device: CUDA device for inference.
    """

    def __init__(
        self,
        head,
        decoder_config: dict,
        artifact_dir: str,
        device: str = "cuda",
        use_torch_compile: bool = False,
        compile_cache_dir: Optional[str] = None,
        use_cudagraphs: bool = False,
    ):
        self._head = head
        self._dit = head.ema_model
        self._decoder_config = decoder_config
        self._artifact_dir = artifact_dir
        self._device = device
        self._use_torch_compile = use_torch_compile
        self._compile_cache_dir = compile_cache_dir
        self._use_cudagraphs = use_cudagraphs

        if compile_cache_dir is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = compile_cache_dir
            logger.info(f"Set TORCHINDUCTOR_CACHE_DIR={compile_cache_dir}")

        self._cache_len = decoder_config["cache_len"]
        self._depth = decoder_config["depth"]
        self._num_kv_heads = decoder_config["num_kv_heads"]
        self._head_dim = decoder_config["head_dim"]
        self._grid_height = decoder_config["grid_height"]
        self._grid_width = decoder_config["grid_width"]
        self._frame_tokens = self._grid_height * self._grid_width

        # Loaded programs: (cache_step, update_cache, batch_size) → callable
        self._programs: Dict[Tuple[int, bool, int], object] = {}
        # Programs take the cache's time offset as a last input. False for .pt2 artifacts compiled before
        # that input existed (manifest without "time_offset_input"); those number cached frames from 0.
        self._time_offset_input = True

        # torch.compile cache: (cache_step, update_cache, batch_size) → compiled fn
        self._compiled_fns: Dict[Tuple[int, bool, int], object] = {}

        # Shared weights (extracted once, injected into all programs)
        self._weights: Optional[Dict[str, torch.Tensor]] = None

        # Compiled cosmos tokenizer decode (lazily created)
        self._compiled_tokenizer_decode = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def decoder_config(self) -> dict:
        return self._decoder_config

    @property
    def cache_len(self) -> int:
        return self._cache_len

    @property
    def compiled_variants(self) -> List[Tuple[int, bool, int]]:
        return sorted(self._programs.keys())

    @property
    def artifact_dir(self) -> str:
        return self._artifact_dir

    @property
    def last_output(self):
        return self._head.last_output

    @last_output.setter
    def last_output(self, value):
        self._head.last_output = value

    # ── Weight extraction ─────────────────────────────────────────────

    def _extract_weights(self) -> Dict[str, torch.Tensor]:
        if self._weights is not None:
            return self._weights

        module = (
            DecoderStepModule(self._dit, cache_size=0, update_cache=False)
            .eval()
            .to(torch.bfloat16)
        )
        self._weights = {k: v.to(self._device) for k, v in module.state_dict().items()}
        n_params = sum(v.numel() for v in self._weights.values())
        size_mb = (
            sum(v.numel() * v.element_size() for v in self._weights.values()) / 1e6
        )
        logger.info(
            f"Extracted {len(self._weights)} weight tensors "
            f"({n_params / 1e6:.0f}M params, {size_mb:.0f} MB)"
        )
        del module
        return self._weights

    # ── Program naming ────────────────────────────────────────────────

    @staticmethod
    def _tag(cache_step: int, update_cache: bool, batch_size: int) -> str:
        return f"decoder_step{cache_step}_update{int(update_cache)}_b{batch_size}"

    def _pt2_path(self, cache_step: int, update_cache: bool, batch_size: int) -> str:
        return os.path.join(
            self._artifact_dir,
            f"{self._tag(cache_step, update_cache, batch_size)}.pt2",
        )

    # ── Loading ───────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_dir: str,
        artifact_dir: str,
        device: str = "cuda",
        batch_sizes: Optional[List[int]] = None,
        cache_steps: Optional[List[int]] = None,
    ) -> "AOTDecoderDispatcher":
        """Load a pre-compiled decoder dispatcher from disk.

        Args:
            model_dir: Path to the WorldModel checkpoint (for decoder weights).
            artifact_dir: Directory containing .pt2 artifacts and manifest.json.
            device: CUDA device.
            batch_sizes: If set, only load programs for these batch sizes.
            cache_steps: If set, only load programs for these cache steps.
        """
        from app.robo_jepa.world_model import WorldModel

        logger.info(f"Loading AOTDecoderDispatcher from {artifact_dir}")

        wm = WorldModel.load_pretrained(model_dir, device=device, load_decoder=True)
        wm.eval()

        return cls.from_world_model(wm, artifact_dir, device, batch_sizes, cache_steps)

    @classmethod
    def from_world_model(
        cls,
        wm,
        artifact_dir: str,
        device: str = "cuda",
        batch_sizes: Optional[List[int]] = None,
        cache_steps: Optional[List[int]] = None,
        use_torch_compile: bool = False,
        compile_cache_dir: Optional[str] = None,
    ) -> "AOTDecoderDispatcher":
        """Create dispatcher from an already-loaded WorldModel.

        Args:
            wm: Loaded WorldModel instance.
            artifact_dir: Directory containing .pt2 artifacts.
            device: CUDA device.
            batch_sizes: If set, only load programs for these batch sizes.
            cache_steps: If set, only load programs for these cache steps.
            use_torch_compile: Use torch.compile instead of AOTI (for FP8).
            compile_cache_dir: Inductor cache dir for torch.compile persistence.
        """
        decoder_config = get_decoder_config(wm)
        head = wm.heads["image_head"]

        dispatcher = cls(
            head=head,
            decoder_config=decoder_config,
            artifact_dir=artifact_dir or "",
            device=device,
            use_torch_compile=use_torch_compile,
            compile_cache_dir=compile_cache_dir,
        )
        if not use_torch_compile:
            dispatcher.load_programs(batch_sizes=batch_sizes, cache_steps=cache_steps)
        return dispatcher

    def load_programs(
        self,
        batch_sizes: Optional[List[int]] = None,
        cache_steps: Optional[List[int]] = None,
    ):
        """Load .pt2 programs from artifact_dir and inject shared weights."""
        # Load or extract weights
        weights_path = os.path.join(self._artifact_dir, "decoder_weights.pt")
        if os.path.exists(weights_path):
            logger.info(f"Loading shared decoder weights from {weights_path}")
            t0 = time.perf_counter()
            self._weights = {
                k: v.to(self._device)
                for k, v in torch.load(
                    weights_path, map_location=self._device, weights_only=True
                ).items()
            }
            logger.info(
                f"  Loaded {len(self._weights)} tensors in "
                f"{(time.perf_counter() - t0) * 1000:.0f} ms"
            )
        else:
            logger.info("No decoder_weights.pt found, extracting from model...")
            self._extract_weights()

        # Discover variants from manifest or filesystem
        manifest_path = os.path.join(self._artifact_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            programs_meta = manifest.get("programs", {})
            self._time_offset_input = manifest.get("time_offset_input", False)
            variants = [
                (v["cache_step"], v["update_cache"], v["batch_size"])
                for v in programs_meta.values()
                if v.get("exists", True)
            ]
        else:
            self._time_offset_input = False
            variants = self._discover_variants_from_filesystem()

        # Filter
        if batch_sizes is not None:
            variants = [(cs, uc, bs) for cs, uc, bs in variants if bs in batch_sizes]
        if cache_steps is not None:
            variants = [(cs, uc, bs) for cs, uc, bs in variants if cs in cache_steps]

        logger.info(f"Loading {len(variants)} decoder .pt2 programs...")
        t_total = time.perf_counter()

        for cs, uc, bs in variants:
            pt2_path = self._pt2_path(cs, uc, bs)
            if not os.path.exists(pt2_path):
                logger.warning(
                    f"  [{self._tag(cs, uc, bs)}] .pt2 not found: {pt2_path}"
                )
                continue

            t0 = time.perf_counter()
            loaded = torch._inductor.aoti_load_package(pt2_path)
            load_ms = (time.perf_counter() - t0) * 1000

            fqns = loaded.get_constant_fqns()
            aligned = {fqn: self._weights[fqn] for fqn in fqns if fqn in self._weights}
            missing = [fqn for fqn in fqns if fqn not in self._weights]
            if missing:
                logger.warning(
                    f"  [{self._tag(cs, uc, bs)}] {len(missing)}/{len(fqns)} FQNs "
                    f"NOT found in weights! First 5: {missing[:5]}"
                )
            else:
                logger.info(f"  [{self._tag(cs, uc, bs)}] all {len(fqns)} FQNs matched")

            t0 = time.perf_counter()
            load_shared_constants(loaded, aligned)
            inject_ms = (time.perf_counter() - t0) * 1000

            self._programs[(cs, uc, bs)] = loaded
            logger.info(
                f"  [{self._tag(cs, uc, bs)}] loaded ({load_ms:.0f} ms) + "
                f"injected ({inject_ms:.0f} ms)"
            )

        logger.info(
            f"All decoder programs loaded in "
            f"{(time.perf_counter() - t_total) * 1000:.0f} ms"
        )

    def _discover_variants_from_filesystem(self) -> List[Tuple[int, bool, int]]:
        """Discover (cache_step, update_cache, batch_size) from .pt2 filenames."""
        variants = []
        for fname in sorted(os.listdir(self._artifact_dir)):
            if not fname.startswith("decoder_step") or not fname.endswith(".pt2"):
                continue
            base = fname.replace(".pt2", "")
            parts = base.split("_")
            try:
                cs = int(parts[1].replace("step", ""))
                uc = parts[2] == "update1"
                bs = int(parts[3].replace("b", ""))
                variants.append((cs, uc, bs))
            except (IndexError, ValueError):
                continue
        return variants

    # ── Dispatch helpers ──────────────────────────────────────────────

    def _get_program(self, cache_step: int, update_cache: bool, batch_size: int):
        """Look up the compiled program, with cache_step clamping and batch padding."""
        effective_step = min(cache_step, self._cache_len)
        key = (effective_step, update_cache, batch_size)

        if key in self._programs:
            return self._programs[key], batch_size

        available_bs = sorted(
            bs
            for (cs, uc, bs) in self._programs
            if cs == effective_step and uc == update_cache
        )
        for avail_bs in available_bs:
            if avail_bs >= batch_size:
                return self._programs[
                    (effective_step, update_cache, avail_bs)
                ], avail_bs

        raise RuntimeError(
            f"No compiled decoder program for cache_step={effective_step}, "
            f"update_cache={update_cache}, batch_size={batch_size}. "
            f"Available: {self.compiled_variants}"
        )

    def _pad_to(
        self, tensor: torch.Tensor, target_B: int, dim: int = 0
    ) -> torch.Tensor:
        B = tensor.shape[dim]
        if B >= target_B:
            return tensor
        shape = list(tensor.shape)
        shape[dim] = target_B - B
        padding = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=dim)

    # ── decode() — drop-in for WorldModelDiTCosmosHead.decode() ───────

    def decode(
        self,
        features,
        denoising_steps=128,
        cfg_scale=1.0,
        samples_prefix=None,
        verbose=False,
        cache=None,
        use_cache=False,
        return_raw=False,
        prefix_noise_timestamp=0,
        skip_postprocess=False,
    ):
        """Decode V-JEPA features → RGB video.

        Drop-in replacement for ``WorldModelDiTCosmosHead.decode()``.
        Same signature, same return type.  The DiT forward pass is
        dispatched to compiled .pt2 programs; all DDIM math is handled
        by ``GaussianDiffusion.ddim_sample_loop()``.

        Args:
            features: (B, T, V, H, W, C) encoded V-JEPA features.
            denoising_steps: Number of DDIM denoising steps.
            cfg_scale: Classifier-free guidance scale (1.0 = no CFG).
            samples_prefix: Optional prefix samples for diffusion forcing.
            verbose: Print progress.
            cache: KV cache from previous decode call (``KeyValueCache``).
            use_cache: Whether to use/return KV cache for online decoding.
            return_raw: Return raw Cosmos latents instead of RGB.
            prefix_noise_timestamp: Noise timestamp for prefix samples.

        Returns:
            If use_cache:  (rgb, cache)
            If return_raw: (rgb, raw_samples)
            Otherwise:     rgb
        """
        assert cfg_scale == 1.0, (
            "AOT decoder dispatcher does not support cfg_scale != 1.0 "
            "(would require 2× batch programs).  Use eager decoder for CFG."
        )
        assert int(return_raw) + int(use_cache) <= 1, (
            "return_raw and use_cache are mutually exclusive."
        )

        head = self._head
        dit = self._dit

        # Ensure diffusion schedule matches requested steps
        if dit.gaussian_diffusion.original_num_steps != denoising_steps:
            from app.robo_jepa.robo_jepa.diffusion import create_diffusion

            dit.gaussian_diffusion = create_diffusion(
                str(denoising_steps), **head.diffusion_kwargs
            )

        B, T, V, H, W, C = features.shape

        # ── Rearrange features (identical to WorldModelDiTHead.decode) ──
        if dit.decode_video:
            condition = rearrange(features, "b t v h w c -> (b v) (t h w) c")
        else:
            condition = rearrange(features, "b t v h w c -> (b t v) (h w) c")

        # ── Create noise & t_init (identical to SimpleDiT.sample) ───────
        B_eff = condition.shape[0]
        batch_mul = 1  # cfg_scale == 1.0

        if dit.decode_video:
            if not use_cache:
                num_frames = dit.num_frames
                if dit.use_cosmos:
                    num_frames += 1
            else:
                num_frames = 1
                assert samples_prefix is None, "prefix cannot be used with cache"
            z = torch.zeros(
                B_eff * batch_mul,
                num_frames,
                dit.in_chans,
                dit.img_size[0],
                dit.img_size[1],
                device=condition.device,
            )
            t_init = torch.zeros(
                (B_eff * batch_mul, num_frames),
                device=z.device,
            ).long()
            t_init += dit.gaussian_diffusion.num_timesteps - 1
            if samples_prefix is not None:
                z[:, : samples_prefix.shape[1]] = samples_prefix
                t_init[:, : samples_prefix.shape[1]] = prefix_noise_timestamp
            z = dit.gaussian_diffusion.q_sample(z, t_init)

            def next_t(t_):
                t_ = t_.clone()
                if samples_prefix is not None:
                    t_[:, samples_prefix.shape[1] :] -= 1
                else:
                    t_ -= 1
                return t_
        else:
            z = torch.randn(
                B_eff * batch_mul,
                dit.in_chans,
                dit.img_size[0],
                dit.img_size[1],
                device=condition.device,
            )
            t_init = None
            next_t = None

        # ── Build model_kwargs (identical to SimpleDiT.sample) ──────────
        kwargs = {"condition": condition}
        if use_cache:
            kwargs["use_cache"] = True
            kwargs["cache"] = cache
            if cache is not None:
                kwargs["cache_size"] = (
                    cache.pointer[0]
                    if isinstance(cache.pointer[0], int)
                    else cache.pointer[0].item()
                )

        # ── Run DDIM loop with compiled model fn ────────────────────────
        compiled_fn = _CompiledModelFn(self)

        samples, kv_cache = dit.gaussian_diffusion.ddim_sample_loop(
            compiled_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=kwargs,
            progress=verbose,
            device=condition.device,
            t_init=t_init,
            next_t_func=next_t,
        )

        # ── Post-process (identical to WorldModelDiTHead.decode) ────────
        T = T * dit.tubelet_size
        samples_ = samples
        samples = rearrange(samples, "(b v) t c h w -> b t v c h w", b=B, v=V)

        if skip_postprocess:
            if use_cache:
                return samples, kv_cache
            return samples

        if self._use_torch_compile and hasattr(head, "tokenizer"):
            rgb = self._postprocess_rgb_compiled(samples, B, V, online=use_cache)
        else:
            rgb = head.postprocess_rgb(samples, online=use_cache)

        if return_raw:
            return rgb, samples_
        elif use_cache:
            return rgb, kv_cache
        return rgb

    # ── Cache management ──────────────────────────────────────────────

    def make_empty_cache(
        self, batch_size: int, device: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create empty KV cache tensors for the decoder.

        Returns:
            (cache_keys, cache_values) both of shape
            [depth, B, num_kv_heads, cache_len, frame_tokens, head_dim]
        """
        if device is None:
            device = self._device
        shape = (
            self._depth,
            batch_size,
            self._num_kv_heads,
            self._cache_len,
            self._frame_tokens,
            self._head_dim,
        )
        cache_keys = torch.zeros(shape, device=device, dtype=torch.bfloat16)
        cache_values = torch.zeros_like(cache_keys)
        return cache_keys, cache_values

    # ── torch.compile mode (FP8, avoids AOTI multi-program bug) ─────

    def _get_compiled_fn(self, cache_step: int, update_cache: bool, batch_size: int):
        """Get a torch.compiled function for (cache_step, update_cache, batch_size).

        Lazily creates and caches compiled functions. When a torch.compile
        cache dir is set, the Inductor cache provides cross-process persistence.
        """
        effective_step = min(cache_step, self._cache_len)
        key = (effective_step, update_cache, batch_size)

        if key not in self._compiled_fns:
            logger.info(
                f"Creating torch.compiled decoder function for "
                f"step={effective_step}, update={update_cache}, B={batch_size}"
            )
            func_mod = (
                DecoderStepModule(
                    self._dit,
                    cache_size=effective_step,
                    update_cache=update_cache,
                )
                .eval()
                .to(torch.bfloat16)
            )

            # Disable CUTLASS templates which have a stride assertion bug
            # with _scaled_mm on col-major FP8 tensors in PyTorch 2.9
            import torch._inductor.config as inductor_config

            inductor_config.max_autotune_gemm_backends = "TRITON"
            inductor_config.coordinate_descent_tuning = True

            compile_kwargs = dict(
                fullgraph=True,
                dynamic=False,
            )
            if self._use_cudagraphs:
                compile_kwargs["mode"] = "reduce-overhead"
            compiled_fn = torch.compile(func_mod, **compile_kwargs)

            # Warm up to trigger actual compilation (and cache the .so)
            inputs = build_decoder_example_inputs(
                self._decoder_config,
                effective_step,
                batch_size,
                device=self._device,
            )
            with torch.no_grad():
                compiled_fn(*inputs)
            torch.cuda.synchronize()

            self._compiled_fns[key] = compiled_fn
            logger.info(
                f"  torch.compiled decoder step={effective_step}, "
                f"update={update_cache}, B={batch_size} ready"
            )

        return self._compiled_fns[key]

    def pre_warm_compile_cache(
        self,
        batch_sizes: List[int] = (1, 2, 4, 8, 16, 32),
        cache_steps: Optional[List[int]] = None,
    ):
        """Pre-warm the torch.compile Inductor cache for all decoder variants.

        This triggers compilation for every (cache_step, update_cache, batch_size)
        combination and saves the resulting .so files to TORCHINDUCTOR_CACHE_DIR.
        Subsequent runs with the same cache dir will skip compilation entirely.
        """
        if cache_steps is None:
            cache_steps = DEFAULT_CACHE_STEPS

        total = len(cache_steps) * 2 * len(batch_sizes)
        done = 0
        t_start = time.perf_counter()

        for cs in cache_steps:
            for uc in [False, True]:
                for B in batch_sizes:
                    t0 = time.perf_counter()
                    self._get_compiled_fn(cs, uc, B)
                    elapsed = time.perf_counter() - t0
                    done += 1
                    logger.info(
                        f"  [{done}/{total}] step={cs}, update={uc}, B={B}: "
                        f"{elapsed:.1f}s"
                    )

        total_time = time.perf_counter() - t_start
        logger.info(
            f"Pre-warmed {total} decoder programs in {total_time:.0f}s "
            f"(cache dir: {self._compile_cache_dir})"
        )

    # ── Compiled cosmos tokenizer decode ─────────────────────────────

    def _get_compiled_tokenizer_decode(self):
        """Lazily compile the cosmos tokenizer decode function."""
        if self._compiled_tokenizer_decode is None:
            head = self._head
            logger.info("Compiling cosmos tokenizer decode...")
            compile_kwargs = dict(dynamic=False)
            if self._use_cudagraphs:
                compile_kwargs["mode"] = "reduce-overhead"
            self._compiled_tokenizer_decode = torch.compile(
                head.tokenizer.decode, **compile_kwargs
            )
        return self._compiled_tokenizer_decode

    def _postprocess_rgb_compiled(self, samples, B, V, online=False):
        """postprocess_rgb with compiled cosmos tokenizer decode."""
        head = self._head
        b, _, v = samples.shape[:3]
        latents = rearrange(samples, "b t v c h w -> (b v) c t h w")

        head.online = online
        if online:
            if head.last_output is None:
                head.last_output = latents
            else:
                head.last_output = torch.cat([head.last_output, latents], dim=2)
                head.last_output = head.last_output[:, :, -8:]
            decode_input = head.last_output
        else:
            decode_input = latents

        if self._use_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        tokenizer_decode = self._get_compiled_tokenizer_decode()
        decoded = tokenizer_decode(decode_input)

        if self._use_cudagraphs:
            decoded = decoded.clone()

        if online:
            decoded = decoded[:, :, -4:]

        decoded = rearrange(decoded, "(b v) c t h w -> b t v h w c", b=b, v=v)
        decoded = (decoded + 1) / 2
        return (decoded * 255.0).clip(0.0, 255.0).to(torch.uint8)

    # ── Compilation ───────────────────────────────────────────────────

    def compile_one(
        self,
        cache_step: int,
        update_cache: bool,
        batch_size: int,
        force: bool = False,
    ) -> str:
        """Compile a single (cache_step, update_cache, batch_size) variant."""
        from app.robo_jepa.inference.compile_decoder_checkpoint import (
            _ensure_cuda_env,
            compile_one,
        )

        _ensure_cuda_env()
        os.makedirs(self._artifact_dir, exist_ok=True)

        return compile_one(
            dit_model=self._dit,
            decoder_config=self._decoder_config,
            cache_step=cache_step,
            update_cache=update_cache,
            batch_size=batch_size,
            artifact_dir=self._artifact_dir,
            force=force,
            device=self._device,
        )

    def compile(
        self,
        batch_sizes: Optional[List[int]] = None,
        cache_steps: Optional[List[int]] = None,
        force: bool = False,
    ):
        """Compile .pt2 programs for all (cache_step, update_cache, batch_size) combos."""
        if batch_sizes is None:
            batch_sizes = DEFAULT_BATCH_SIZES
        if cache_steps is None:
            cache_steps = DEFAULT_CACHE_STEPS

        total = len(cache_steps) * 2 * len(batch_sizes)
        logger.info(
            f"Compiling {total} decoder programs: "
            f"cache_steps={cache_steps}, batch_sizes={batch_sizes}"
        )

        t_total = time.perf_counter()
        for cs in cache_steps:
            for uc in [False, True]:
                for bs in batch_sizes:
                    self.compile_one(cs, uc, bs, force=force)

        dt = time.perf_counter() - t_total
        logger.info(f"Compilation complete: {total} programs in {dt:.0f}s")

    # ── Save ──────────────────────────────────────────────────────────

    def save(self, save_weights: bool = True):
        """Save manifest and optionally shared weights to artifact_dir."""
        from app.robo_jepa.inference.compile_decoder_checkpoint import (
            save_manifest,
            save_weights as _save_weights,
        )

        os.makedirs(self._artifact_dir, exist_ok=True)

        if save_weights:
            _save_weights(self._dit, self._artifact_dir)

        batch_sizes = sorted(
            {bs for _, _, bs in self._discover_variants_from_filesystem()}
        )
        cache_steps_found = sorted(
            {cs for cs, _, _ in self._discover_variants_from_filesystem()}
        )

        save_manifest(
            self._decoder_config,
            self._artifact_dir,
            batch_sizes or DEFAULT_BATCH_SIZES,
            cache_steps_found or DEFAULT_CACHE_STEPS,
        )
