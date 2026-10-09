# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AOT Inductor dispatcher for RoboJEPA world model inference.

Manages pre-compiled .pt2 programs keyed by (step_idx, batch_size), with shared
model weights across all variants. Each .pt2 is compiled without baked-in weights
(~30 MB code-only), and weights are injected once at load time from a single copy.

Architecture:
    ┌──────────────────────────────────────────────────┐
    │              AOTWorldModelDispatcher              │
    │                                                  │
    │  weights (GPU, shared) ─────┬──────┬──────┐      │
    │                             │      │      │      │
    │  programs: {                ▼      ▼      ▼      │
    │    (0, 16): loaded_pt2 ← weights                 │
    │    (0, 32): loaded_pt2 ← weights                 │
    │    (7, 16): loaded_pt2 ← weights                 │
    │    (7, 32): loaded_pt2 ← weights                 │
    │    ...                                           │
    │  }                                               │
    │                                                  │
    │  step(action, state) → dispatch by (step, B)     │
    └──────────────────────────────────────────────────┘

Artifact directory layout:
    artifact_dir/
        manifest.json          # metadata: model config, compiled variants
        weights.pt             # shared bf16 state_dict (saved once)
        step0_b16.pt2          # code-only compiled programs
        step0_b32.pt2
        step7_b16.pt2
        step7_b32.pt2
        ...
"""

import json
import logging
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from app.robo_jepa.inference.aot_compile import (
    FunctionalStepModule,
    build_example_inputs,
    get_model_config,
    load_shared_constants,
)
from app.robo_jepa.inference.mem_debug import log_rss

# Default compilation grid
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DEFAULT_STEPS = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 0-7 = unique cache sizes, 8 = rolling
from app.robo_jepa.world_model import WorldModel, WorldModelState, compute_new_pose
from app.robo_jepa.embodiment_preprocessing import (
    get_action_preprocessor,
    get_state_preprocessor,
    resolve_embodiment_name,
)

logger = logging.getLogger(__name__)


def _ensure_cuda_env():
    """Ensure CUDA headers are available for AOT Inductor compilation."""
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        # Try to infer from sys.executable
        conda_prefix = os.path.dirname(os.path.dirname(sys.executable))

    targets_include = os.path.join(conda_prefix, "targets", "x86_64-linux", "include")
    if os.path.isdir(targets_include):
        existing = os.environ.get("CPLUS_INCLUDE_PATH", "")
        if targets_include not in existing:
            os.environ["CPLUS_INCLUDE_PATH"] = f"{targets_include}:{existing}"

    if "CUDA_HOME" not in os.environ and os.path.isdir(conda_prefix):
        os.environ["CUDA_HOME"] = conda_prefix


# ── Parallel compilation worker (runs in a subprocess) ────────────────────────


def _compile_worker(
    gpu_id: int,
    combos: List[Tuple[int, int]],
    model_dir: str,
    artifact_dir: str,
    cache_len: int,
    force: bool = False,
    precision: str = "bf16",
    attn_impl: str = "fmha",
    fp8_scales_path: str = None,
    weights_filename: str = "world_model.pth.tar",
):
    """Subprocess worker: compiles assigned (step, batch) combos on one GPU.

    Each worker process sees only one GPU via CUDA_VISIBLE_DEVICES, so it
    always compiles for cuda:0. This avoids device index mismatch issues
    where a .pt2 compiled on cuda:3 fails when loaded on cuda:0.

    Args:
        precision: "bf16" (model+inputs bf16) or "mp-bf16" (model fp32, autocast).
        attn_impl: Attention backend to use ("fmha", "sdpa", "sdpa_cudnn").
    """
    import gc
    import os
    import time

    import torch

    # Force this process to see only its assigned GPU as cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    torch.cuda.set_device(0)

    _ensure_cuda_env()

    from app.robo_jepa.inference.aot_compile import (
        FunctionalStepModule,
        MixedPrecisionStepModule,
        build_example_inputs,
        get_model_config,
        swap_attn_impl,
        _ensure_libcuda_symlink,
    )
    from app.robo_jepa.world_model import WorldModel

    print(
        f"  [GPU {gpu_id}] Worker started, {len(combos)} combos to compile "
        f"(precision={precision}, attn_impl={attn_impl})",
        flush=True,
    )

    wm = WorldModel.load_pretrained(
        model_dir, device=device, load_decoder=False, weights_filename=weights_filename
    )
    wm.eval()
    # Disable training-only activation checkpointing so the functional KV-cache
    # path populates the new-cache tensors during AOT export (see dispatcher
    # __init__ for details).
    if hasattr(wm.transition_model, "use_activation_checkpointing"):
        wm.transition_model.use_activation_checkpointing = False
    model_config = get_model_config(wm)

    if attn_impl != "fmha":
        n = swap_attn_impl(wm.transition_model, attn_impl)
        print(
            f"  [GPU {gpu_id}] Swapped attn_impl to '{attn_impl}' on {n} modules",
            flush=True,
        )

    # Apply static FP8 quantization if scales are provided
    if fp8_scales_path is not None:
        from app.robo_jepa.inference.static_fp8 import apply_static_fp8

        scales = torch.load(fp8_scales_path, map_location=device)
        wm.transition_model, n_fp8 = apply_static_fp8(wm.transition_model, scales)
        print(f"  [GPU {gpu_id}] Applied static FP8 to {n_fp8} layers", flush=True)

    # Force cuDNN attention lowering in Inductor when sdpa_cudnn is requested
    if attn_impl == "sdpa_cudnn":
        import torch._inductor.config as inductor_config

        if hasattr(inductor_config, "force_cudnn_attention"):
            inductor_config.force_cudnn_attention = True
            print(f"  [GPU {gpu_id}] Enabled force_cudnn_attention=True", flush=True)
        else:
            print(
                f"  [GPU {gpu_id}] force_cudnn_attention not available in PyTorch {torch.__version__}",
                flush=True,
            )

    results = []
    for step_idx, batch_size in combos:
        tag = f"step{step_idx}_b{batch_size}"
        pt2_path = os.path.join(artifact_dir, f"{tag}.pt2")

        if os.path.exists(pt2_path) and not force:
            print(f"  [GPU {gpu_id}][{tag}] exists, skipping", flush=True)
            results.append((step_idx, batch_size, pt2_path, 0.0, None))
            continue

        cache_size = min(step_idx, cache_len)
        print(
            f"  [GPU {gpu_id}][{tag}] Compiling (cache_size={cache_size})...",
            flush=True,
        )
        t0 = time.perf_counter()

        try:
            if precision == "mp-bf16":
                # Mixed precision: fp32 model, autocast in forward
                func_mod = (
                    MixedPrecisionStepModule(
                        wm.transition_model,
                        cache_size=cache_size,
                        input_position=step_idx,
                    )
                    .eval()
                    .float()
                )
            else:
                # Pure bf16: model + inputs all bf16
                func_mod = (
                    FunctionalStepModule(
                        wm.transition_model,
                        cache_size=cache_size,
                        input_position=step_idx,
                    )
                    .eval()
                    .to(torch.bfloat16)
                )

            example = build_example_inputs(model_config, step_idx, batch_size, cache_len, device)

            if precision == "mp-bf16":
                # For mp-bf16: video_features, actions, states stay fp32;
                # cache tensors are bf16 (attention outputs bf16 under autocast)
                pass  # build_example_inputs already returns fp32 + bf16 cache
            else:
                # For bf16: convert all floating-point inputs to bf16
                example = tuple(t.to(torch.bfloat16) if t.is_floating_point() else t for t in example)

            with torch.no_grad():
                exported = torch.export.export(func_mod, example)

            os.makedirs(artifact_dir, exist_ok=True)
            _ensure_libcuda_symlink()
            torch._inductor.aoti_compile_and_package(
                exported,
                package_path=pt2_path,
                inductor_configs={
                    "aot_inductor.package_constants_in_so": False,
                },
            )

            dt = time.perf_counter() - t0
            size_mb = os.path.getsize(pt2_path) / (1024 * 1024)
            print(
                f"  [GPU {gpu_id}][{tag}] Done in {dt:.0f}s ({size_mb:.1f} MB)",
                flush=True,
            )
            results.append((step_idx, batch_size, pt2_path, dt, None))

            del func_mod, exported
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"  [GPU {gpu_id}][{tag}] FAILED in {dt:.0f}s: {e}", flush=True)
            import traceback as tb

            tb.print_exc()
            results.append((step_idx, batch_size, None, dt, str(e)))

        gc.collect()
        torch.cuda.empty_cache()

    del wm
    return results


class AOTWorldModelDispatcher:
    """AOT Inductor dispatcher for RoboJEPA world model.

    Compiles, saves, loads, and dispatches pre-compiled .pt2 programs for
    different (step_idx, batch_size) combinations. All programs share a
    single copy of the model weights on GPU.

    Args:
        wm: Loaded WorldModel instance (used for compilation and weight extraction).
        artifact_dir: Directory to store/load .pt2 artifacts and weights.
        device: CUDA device for inference.
    """

    def __init__(
        self,
        wm: WorldModel,
        artifact_dir: str,
        device: str = "cuda",
        precision: str = "bf16",
        attn_impl: str = "fmha",
        use_torch_compile: bool = False,
        compile_cache_dir: Optional[str] = None,
        use_cudagraphs: bool = False,
        grid_height: Optional[int] = None,
        grid_width: Optional[int] = None,
    ):
        self._wm = wm
        # Activation checkpointing is a training-only memory optimization. It must
        # be off for the functional KV-cache path used by AOT compile / compiled
        # inference, otherwise the predictor skips populating the new-cache tensors
        # (torch.stack on an empty list during export). Inference never needs it.
        if wm is not None and hasattr(wm.transition_model, "use_activation_checkpointing"):
            wm.transition_model.use_activation_checkpointing = False
        self._artifact_dir = artifact_dir
        self._device = device
        self._precision = precision
        self._attn_impl = attn_impl
        self._use_torch_compile = use_torch_compile
        self._compile_cache_dir = compile_cache_dir
        self._use_cudagraphs = use_cudagraphs
        self._model_config = get_model_config(wm, grid_height=grid_height, grid_width=grid_width)

        # Set up torch.compile cache directory if provided
        if compile_cache_dir is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = compile_cache_dir
            logger.info(f"Set TORCHINDUCTOR_CACHE_DIR={compile_cache_dir}")

        # Swap attention backend on the model if different from default
        if attn_impl != "fmha" and wm is not None:
            from app.robo_jepa.inference.aot_compile import swap_attn_impl

            n = swap_attn_impl(wm.transition_model, attn_impl)
            logger.info(f"Swapped attn_impl to '{attn_impl}' on {n} attention modules")

        # Computed from model config
        self._cache_len = self._model_config["cache_len"]
        self._action_dim = self._model_config["action_dim"]
        self._state_dim = self._model_config["state_dim"]
        self._depth = self._model_config["depth"]

        num_kv_heads = self._model_config["num_kv_heads"] or self._model_config["num_heads"]
        num_heads = self._model_config["num_heads"]
        pred_dim = self._model_config["predictor_embed_dim"]
        head_dim = pred_dim // num_heads
        H = self._model_config["grid_height"]
        W = self._model_config["grid_width"]
        n_views = self._model_config["num_views"] or 1
        action_tokens = self._model_config["action_tokens"]
        state_tokens = self._model_config["state_tokens"]

        self._frame_tokens = n_views * H * W + action_tokens + state_tokens
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._state_tokens = state_tokens

        # Loaded programs: (step_idx, batch_size) -> callable
        self._programs: Dict[Tuple[int, int], object] = {}
        # Programs take the cache's time offset as a last input. False for .pt2 artifacts compiled before
        # that input existed (manifest without "time_offset_input"); those number cached frames from 0.
        self._time_offset_input = True
        self._loaded_batch_size: Optional[int] = None  # currently loaded B

        # Available variants discovered from manifest/filesystem
        self._available_variants: List[dict] = []

        # Shared weights (extracted once, injected into all programs)
        self._weights: Optional[Dict[str, torch.Tensor]] = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def model(self):
        return self._wm.model

    @property
    def transition_model(self):
        return self._wm.transition_model

    @property
    def encoder(self):
        return self._wm.encoder

    @property
    def model_config(self) -> dict:
        return self._model_config

    @property
    def heads(self):
        return self._wm.heads

    @property
    def compiled_variants(self) -> List[Tuple[int, int]]:
        """List of (step_idx, batch_size) variants currently loaded."""
        return sorted(self._programs.keys())

    # ── Delegate to WorldModel for non-step operations ────────────────

    def to(self, device):
        self._wm.to(device)
        self._device = str(device)
        if self._weights is not None:
            self._weights = {k: v.to(device) for k, v in self._weights.items()}
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def eval(self):
        self._wm.eval()
        return self

    def preprocess_frames(self, *args, **kwargs):
        return self._wm.preprocess_frames(*args, **kwargs)

    def encode_features(self, *args, **kwargs):
        return self._wm.encode_features(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self._wm.encode(*args, **kwargs)

    # ── Weight extraction ─────────────────────────────────────────────

    def _extract_weights(self, step_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Extract weights for FunctionalStepModule.

        For bf16 precision: weights are converted to bf16.
        For mp-bf16 precision: weights stay fp32 (model runs in fp32 with autocast).
        """
        if self._weights is not None:
            return self._weights

        weight_dtype = torch.bfloat16 if self._precision == "bf16" else torch.float32
        func_mod = (
            FunctionalStepModule(self._wm.transition_model, cache_size=0, input_position=0).eval().to(weight_dtype)
        )
        self._weights = {k: v.to(self._device) for k, v in func_mod.state_dict().items()}
        logger.info(
            f"Extracted {len(self._weights)} weight tensors "
            f"(dtype={weight_dtype}, "
            f"{sum(v.numel() * v.element_size() for v in self._weights.values()) / 1e6:.0f} MB)"
        )
        del func_mod
        return self._weights

    # ── Compilation ───────────────────────────────────────────────────

    def _pt2_path(self, step_idx: int, batch_size: int) -> str:
        return os.path.join(self._artifact_dir, f"step{step_idx}_b{batch_size}.pt2")

    def _weights_path(self) -> str:
        return os.path.join(self._artifact_dir, "weights.pt")

    def _manifest_path(self) -> str:
        return os.path.join(self._artifact_dir, "manifest.json")

    def compile_one(self, step_idx: int, batch_size: int, force: bool = False) -> str:
        """Compile a single (step_idx, batch_size) variant to a code-only .pt2.

        Args:
            step_idx: Cache step index (0 = empty cache, up to cache_len).
            batch_size: Batch size to compile for.
            force: If True, recompile even if .pt2 already exists.

        Returns:
            Path to the compiled .pt2 file.
        """
        _ensure_cuda_env()
        from app.robo_jepa.inference.aot_compile import _ensure_libcuda_symlink

        pt2_path = self._pt2_path(step_idx, batch_size)
        if os.path.exists(pt2_path) and not force:
            logger.info(f"  [{step_idx}, B={batch_size}] .pt2 exists: {pt2_path}")
            return pt2_path

        cache_size = min(step_idx, self._cache_len)
        logger.info(f"  [{step_idx}, B={batch_size}] Compiling (cache_size={cache_size})...")

        func_mod = (
            FunctionalStepModule(
                self._wm.transition_model,
                cache_size=cache_size,
                input_position=step_idx,
            )
            .eval()
            .to(torch.bfloat16)
        )

        example = build_example_inputs(self._model_config, step_idx, batch_size, self._cache_len, self._device)
        bf16_example = tuple(t.to(torch.bfloat16) if t.is_floating_point() else t for t in example)

        t0 = time.perf_counter()
        with torch.no_grad():
            exported = torch.export.export(func_mod, bf16_example)
        export_time = time.perf_counter() - t0
        logger.info(f"    export: {export_time:.1f}s")

        t0 = time.perf_counter()
        os.makedirs(self._artifact_dir, exist_ok=True)
        _ensure_libcuda_symlink()
        torch._inductor.aoti_compile_and_package(
            exported,
            package_path=pt2_path,
            inductor_configs={
                "aot_inductor.package_constants_in_so": False,
            },
        )
        compile_time = time.perf_counter() - t0
        size_mb = os.path.getsize(pt2_path) / (1024 * 1024)
        logger.info(f"    compile: {compile_time:.1f}s → {size_mb:.1f} MB")

        del func_mod, exported
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        return pt2_path

    def compile(
        self,
        batch_sizes: Optional[List[int]] = None,
        steps: Optional[List[int]] = None,
        force: bool = False,
    ):
        """Compile .pt2 programs for all (step, batch_size) combinations.

        Args:
            batch_sizes: List of batch sizes to compile for.
                Defaults to DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32].
            steps: List of step indices. Defaults to DEFAULT_STEPS = [0..8].
                Steps beyond cache_len use rolling cache (same compiled program
                as cache_len, but we still compile a separate artifact so the
                dispatcher can look it up directly).
            force: If True, recompile even if .pt2 already exists.
        """
        _ensure_cuda_env()

        if batch_sizes is None:
            batch_sizes = DEFAULT_BATCH_SIZES
        if steps is None:
            steps = DEFAULT_STEPS

        total = len(steps) * len(batch_sizes)

        logger.info(f"Compiling {total} programs: steps={steps}, batch_sizes={batch_sizes}")

        t_total = time.perf_counter()
        compiled = []
        for step_idx in steps:
            for bs in batch_sizes:
                pt2_path = self.compile_one(step_idx, bs, force=force)
                compiled.append((step_idx, bs, pt2_path))

        compile_total = time.perf_counter() - t_total
        logger.info(f"Compilation complete: {len(compiled)} programs in {compile_total:.0f}s")

    def compile_parallel(
        self,
        model_dir: str,
        batch_sizes: Optional[List[int]] = None,
        steps: Optional[List[int]] = None,
        num_gpus: Optional[int] = None,
        force: bool = False,
        precision: str = "bf16",
        attn_impl: str = "fmha",
        fp8_scales_path: str = None,
        weights_filename: str = "world_model.pth.tar",
    ):
        """Compile .pt2 programs in parallel across multiple GPUs.

        Args:
            model_dir: Path to WorldModel checkpoint (workers load it fresh).
            batch_sizes: List of batch sizes. Defaults to DEFAULT_BATCH_SIZES.
            steps: List of step indices. Defaults to DEFAULT_STEPS.
            num_gpus: Number of GPUs to use. Defaults to torch.cuda.device_count().
            force: If True, recompile even if .pt2 already exists.
            precision: "bf16" (all bf16) or "mp-bf16" (fp32 model + autocast).
            attn_impl: Attention backend ("fmha", "sdpa", "sdpa_cudnn").
        """
        import torch.multiprocessing as mp

        _ensure_cuda_env()

        if batch_sizes is None:
            batch_sizes = DEFAULT_BATCH_SIZES
        if steps is None:
            steps = DEFAULT_STEPS
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        combos = [(s, b) for s in steps for b in batch_sizes]
        total = len(combos)
        os.makedirs(self._artifact_dir, exist_ok=True)

        logger.info(f"Parallel compilation: {total} programs across {num_gpus} GPUs")
        logger.info(f"  steps={steps}, batch_sizes={batch_sizes}")

        # Distribute combos round-robin across GPUs
        gpu_combos: List[List[Tuple[int, int]]] = [[] for _ in range(num_gpus)]
        for i, combo in enumerate(combos):
            gpu_combos[i % num_gpus].append(combo)

        for g in range(num_gpus):
            logger.info(f"  GPU {g}: {len(gpu_combos[g])} combos")

        t_total = time.perf_counter()

        # Use spawn to avoid CUDA context fork issues
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_gpus) as pool:
            worker_results = pool.starmap(
                _compile_worker,
                [
                    (
                        gpu_id,
                        gpu_combos[gpu_id],
                        model_dir,
                        self._artifact_dir,
                        self._cache_len,
                        force,
                        precision,
                        attn_impl,
                        fp8_scales_path,
                        weights_filename,
                    )
                    for gpu_id in range(num_gpus)
                    if gpu_combos[gpu_id]
                ],
            )

        # Flatten results
        all_results = [r for batch in worker_results for r in batch]
        n_ok = sum(1 for _, _, path, _, err in all_results if err is None)
        n_fail = sum(1 for _, _, path, _, err in all_results if err is not None)
        total_time = time.perf_counter() - t_total

        logger.info(
            f"Parallel compilation done: {n_ok} succeeded, {n_fail} failed, "
            f"{total_time:.0f}s wall-clock ({total_time / 60:.1f} min)"
        )

        if n_fail > 0:
            for step_idx, bs, path, dt, err in all_results:
                if err is not None:
                    logger.error(f"  FAILED [{step_idx}, B={bs}]: {err}")

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, save_weights: bool = True):
        """Save manifest (and optionally shared weights) to artifact_dir.

        The .pt2 files are already saved during compile(). This method
        additionally saves a manifest describing all compiled variants,
        and optionally the shared weights.

        Args:
            save_weights: If True (default), save weights.pt alongside
                the .pt2 files. Set to False when embedding compiled programs
                in a model checkpoint — weights will be extracted from the
                loaded model at runtime instead.
        """
        os.makedirs(self._artifact_dir, exist_ok=True)

        # Save weights (optional — skip when embedding in a model checkpoint)
        weights_mb = 0.0
        if save_weights:
            weights = self._extract_weights()
            weights_path = self._weights_path()
            torch.save(weights, weights_path)
            weights_mb = os.path.getsize(weights_path) / (1024 * 1024)
            logger.info(f"Saved weights: {weights_path} ({weights_mb:.0f} MB)")
        else:
            logger.info("Skipping weights.pt (will be extracted from model at load time)")

        # Discover all .pt2 files in artifact_dir
        variants = []
        for fname in sorted(os.listdir(self._artifact_dir)):
            if fname.endswith(".pt2"):
                # Parse step{X}_b{Y}.pt2
                base = fname.replace(".pt2", "")
                parts = base.split("_")
                try:
                    step_idx = int(parts[0].replace("step", ""))
                    batch_size = int(parts[1].replace("b", ""))
                    variants.append(
                        {
                            "step": step_idx,
                            "batch_size": batch_size,
                            "file": fname,
                            "size_mb": os.path.getsize(os.path.join(self._artifact_dir, fname)) / (1024 * 1024),
                        }
                    )
                except (IndexError, ValueError):
                    continue

        # Save manifest
        manifest = {
            "model_config": {
                k: v for k, v in self._model_config.items() if isinstance(v, (int, float, str, bool, type(None), list))
            },
            "device": self._device,
            "variants": variants,
            "weights_file": "weights.pt",
            "weights_size_mb": weights_mb,
            "time_offset_input": True,
        }
        manifest_path = self._manifest_path()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved manifest: {manifest_path} ({len(variants)} variants)")

    @classmethod
    def load(
        cls,
        model_dir: str,
        artifact_dir: str,
        device: str = "cuda",
        batch_sizes: Optional[List[int]] = None,
        max_steps: Optional[int] = None,
        load_decoder: bool = False,
    ) -> "AOTWorldModelDispatcher":
        """Load a pre-compiled dispatcher from disk.

        Args:
            model_dir: Path to the WorldModel checkpoint (for encoder + weights).
            artifact_dir: Directory containing .pt2 artifacts, weights.pt, manifest.json.
            device: CUDA device.
            batch_sizes: If set, only load programs for these batch sizes.
            max_steps: If set, only load programs for steps 0..max_steps-1.
            load_decoder: Whether to load the decoder (typically False for inference).

        Returns:
            Loaded AOTWorldModelDispatcher ready for inference.
        """
        logger.info(f"Loading AOTWorldModelDispatcher from {artifact_dir}")

        wm = WorldModel.load_pretrained(model_dir, device=device, load_decoder=load_decoder)
        wm.eval()

        dispatcher = cls(wm, artifact_dir=artifact_dir, device=device)
        dispatcher.load_programs(batch_sizes=batch_sizes, max_steps=max_steps)

        return dispatcher

    # ----------------------------------------------------------------
    # torch.compile mode (used for FP8 to avoid AOTI multi-program bug)
    # ----------------------------------------------------------------

    def _get_compiled_fn(self, step_idx: int, batch_size: int):
        """Get a torch.compiled function for (step_idx, batch_size).

        Lazily creates and caches compiled functions. When a torch.compile
        cache dir is set, the Inductor cache provides cross-process persistence.
        """
        cache_len = self._model_config["cache_len"]
        effective_step = min(step_idx, cache_len)
        key = (effective_step, batch_size)

        if not hasattr(self, "_compiled_fns"):
            self._compiled_fns = {}

        if key not in self._compiled_fns:
            logger.info(f"Creating torch.compiled function for step={effective_step}, B={batch_size}")
            func_mod = FunctionalStepModule(
                self._wm.transition_model,
                cache_size=effective_step,
                input_position=step_idx,
            )
            if self._precision == "bf16":
                func_mod = func_mod.eval().to(torch.bfloat16)
            else:
                func_mod = func_mod.eval()

            # Disable CUTLASS templates which have a stride assertion bug
            # with _scaled_mm on col-major FP8 tensors in PyTorch 2.9
            import torch._inductor.config as inductor_config

            inductor_config.max_autotune_gemm_backends = "TRITON"

            compile_kwargs = dict(
                fullgraph=True,
                dynamic=False,
            )
            if self._use_cudagraphs:
                compile_kwargs["mode"] = "reduce-overhead"
            compiled_fn = torch.compile(func_mod, **compile_kwargs)

            # Warm up to trigger actual compilation (and cache the .so)
            inputs = build_example_inputs(
                self._model_config,
                step_idx,
                batch_size,
                cache_len,
                self._device,
            )
            if self._precision == "bf16":
                inputs = tuple(t.to(torch.bfloat16) if t.is_floating_point() else t for t in inputs)
            with torch.no_grad():
                compiled_fn(*inputs)
            torch.cuda.synchronize()

            self._compiled_fns[key] = compiled_fn
            logger.info(f"  torch.compiled step={effective_step}, B={batch_size} ready")

        return self._compiled_fns[key]

    def pre_warm_compile_cache(
        self,
        batch_sizes: List[int] = (1, 2, 4, 8, 16, 32),
        max_steps: Optional[int] = None,
    ):
        """Pre-warm the torch.compile Inductor cache for all (step, batch_size).

        This triggers compilation for every combination and saves the resulting
        .so files to TORCHINDUCTOR_CACHE_DIR. Subsequent runs with the same
        cache dir will skip compilation entirely.
        """
        cache_len = self._model_config["cache_len"]
        n_steps = cache_len + 1  # steps 0..cache_len (unique programs)
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)

        total = n_steps * len(batch_sizes)
        done = 0
        t_start = time.perf_counter()

        for step_idx in range(n_steps):
            for B in batch_sizes:
                t0 = time.perf_counter()
                self._get_compiled_fn(step_idx, B)
                elapsed = time.perf_counter() - t0
                done += 1
                logger.info(f"  [{done}/{total}] step={step_idx}, B={B}: {elapsed:.1f}s")

        total_time = time.perf_counter() - t_start
        logger.info(f"Pre-warmed {total} programs in {total_time:.0f}s (cache dir: {self._compile_cache_dir})")

    def load_programs(
        self,
        batch_sizes: Optional[List[int]] = None,
        max_steps: Optional[int] = None,
    ):
        """Load .pt2 programs from artifact_dir and inject shared weights.

        Args:
            batch_sizes: If set, only load programs for these batch sizes.
            max_steps: If set, only load programs for steps 0..max_steps-1.
        """
        # Load weights
        weights_path = self._weights_path()
        if os.path.exists(weights_path):
            logger.info(f"Loading shared weights from {weights_path}")
            t0 = time.perf_counter()
            self._weights = {
                k: v.to(self._device)
                for k, v in torch.load(weights_path, map_location=self._device, weights_only=True).items()
            }
            logger.info(f"  Loaded {len(self._weights)} tensors in {(time.perf_counter() - t0) * 1000:.0f} ms")
        else:
            # Offload transition model to CPU FIRST to free GPU memory,
            # then extract bf16 weights from CPU and move to GPU.
            logger.info("No weights.pt found, extracting from model...")
            if self._wm is not None and hasattr(self._wm, "transition_model"):
                self._wm.transition_model.cpu()
                torch.cuda.empty_cache()
                logger.info("Offloaded transition_model to CPU before weight extraction")
            self._extract_weights()

        # Load manifest to discover variants
        manifest_path = self._manifest_path()
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            variants = manifest.get("variants", [])
            self._time_offset_input = manifest.get("time_offset_input", False)
        else:
            # Discover from filesystem
            self._time_offset_input = False
            variants = []
            for fname in sorted(os.listdir(self._artifact_dir)):
                if fname.endswith(".pt2"):
                    base = fname.replace(".pt2", "")
                    parts = base.split("_")
                    try:
                        step_idx = int(parts[0].replace("step", ""))
                        batch_size_val = int(parts[1].replace("b", ""))
                        variants.append(
                            {
                                "step": step_idx,
                                "batch_size": batch_size_val,
                                "file": fname,
                            }
                        )
                    except (IndexError, ValueError):
                        continue

        # Filter variants
        filtered = variants
        if batch_sizes is not None:
            filtered = [v for v in filtered if v["batch_size"] in batch_sizes]
        if max_steps is not None:
            filtered = [v for v in filtered if v["step"] < max_steps]

        logger.info(f"Loading {len(filtered)} .pt2 programs...")
        log_rss("dispatcher.load_programs:before_aot_load")
        t_total = time.perf_counter()

        for v in filtered:
            step_idx = v["step"]
            bs = v["batch_size"]
            pt2_path = os.path.join(self._artifact_dir, v["file"])

            if not os.path.exists(pt2_path):
                logger.warning(f"  [{step_idx}, B={bs}] .pt2 not found: {pt2_path}")
                continue

            t0 = time.perf_counter()
            loaded = torch._inductor.aoti_load_package(pt2_path)
            load_ms = (time.perf_counter() - t0) * 1000
            log_rss(f"dispatcher.load_programs:loaded[step{step_idx}_b{bs}]")

            # Inject shared weights
            fqns = loaded.get_constant_fqns()
            aligned = {fqn: self._weights[fqn] for fqn in fqns if fqn in self._weights}

            t0 = time.perf_counter()
            load_shared_constants(loaded, aligned)
            inject_ms = (time.perf_counter() - t0) * 1000
            log_rss(f"dispatcher.load_programs:injected[step{step_idx}_b{bs}]")

            self._programs[(step_idx, bs)] = loaded
            logger.info(f"  [{step_idx}, B={bs}] loaded ({load_ms:.0f} ms) + injected ({inject_ms:.0f} ms)")

        logger.info(f"All programs loaded in {(time.perf_counter() - t_total) * 1000:.0f} ms")

        # Offload the eager transition model to CPU — step() uses compiled .pt2
        # programs which hold their own bf16 weights via load_constants().
        # The encoder stays on GPU for encode() calls.
        if self._wm is not None and hasattr(self._wm, "transition_model"):
            self._wm.transition_model.cpu()
            torch.cuda.empty_cache()
            logger.info("Offloaded transition_model to CPU (compiled programs hold bf16 weights)")

        log_rss("dispatcher.load_programs:done")

    # ── Inference ─────────────────────────────────────────────────────

    def _get_program(self, step_idx: int, batch_size: int):
        """Look up the compiled program for (step_idx, batch_size).

        Steps beyond cache_len all use the same program (rolling cache).
        """
        effective_step = min(step_idx, self._cache_len)
        key = (effective_step, batch_size)

        if key in self._programs:
            return self._programs[key]

        # If exact batch size not found, try to find one we can pad to
        available_bs = sorted(bs for (s, bs) in self._programs if s == effective_step)
        for avail_bs in available_bs:
            if avail_bs >= batch_size:
                return self._programs[(effective_step, avail_bs)]

        raise RuntimeError(
            f"No compiled program for step={effective_step}, batch_size={batch_size}. "
            f"Available: {self.compiled_variants}"
        )

    def _get_padded_batch_size(self, step_idx: int, batch_size: int) -> int:
        """Get the compiled batch size that will be used (may be larger than actual)."""
        effective_step = min(step_idx, self._cache_len)
        key = (effective_step, batch_size)
        if key in self._programs:
            return batch_size

        available_bs = sorted(bs for (s, bs) in self._programs if s == effective_step)
        for avail_bs in available_bs:
            if avail_bs >= batch_size:
                return avail_bs

        raise RuntimeError(f"No compiled program for step={effective_step}, batch_size={batch_size}.")

    def _pad_to(self, tensor: torch.Tensor, target_B: int, dim: int = 0) -> torch.Tensor:
        """Pad tensor along batch dim to target_B."""
        B = tensor.shape[dim]
        if B >= target_B:
            return tensor
        shape = list(tensor.shape)
        shape[dim] = target_B - B
        padding = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=dim)

    def step(
        self,
        action: torch.Tensor,
        state: WorldModelState,
        use_cache: bool = True,
        no_grad: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, WorldModelState]:
        """Run one inference step using the AOT-compiled program.

        Drop-in replacement for WorldModel.step(). Returns (out, new_state)
        where out is the predicted features (B, 1, V, H, W, C) and new_state
        is the updated WorldModelState with cache.

        Args:
            action: (B, 1, action_dim) action tensor.
            state: Current WorldModelState.
            use_cache: Must be True (AOT programs always use cache).
            no_grad: Ignored (always runs under torch.no_grad).

        Returns:
            (out, new_state): Predicted features and updated state.
        """
        video_features = state.features[:, -1:]  # (B, 1, V, H, W, C)
        B_actual = video_features.shape[0]
        step_idx = state.actions.shape[1]  # number of actions so far = step index

        # Look up compiled program
        if self._use_torch_compile:
            if self._use_cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            program = self._get_compiled_fn(step_idx, B_actual)
            target_B = B_actual  # torch.compile doesn't need batch padding
        else:
            program = self._get_program(step_idx, B_actual)
            target_B = self._get_padded_batch_size(step_idx, B_actual)

        # Initialize AOT cache if not present on state
        if not hasattr(state, "_aot_cache_keys") or state._aot_cache_keys is None:
            state._aot_cache_keys = torch.zeros(
                self._depth,
                target_B,
                self._num_kv_heads,
                self._cache_len,
                self._frame_tokens,
                self._head_dim,
                device=self._device,
                dtype=torch.bfloat16,
            )
            state._aot_cache_values = torch.zeros_like(state._aot_cache_keys)

        # Prepare inputs on device, padded to target_B
        # For bf16: all floating-point inputs are bf16
        # For mp-bf16: video_features/actions/states are fp32, cache stays bf16
        float_dtype = torch.bfloat16 if self._precision == "bf16" else torch.float32

        vf = self._pad_to(video_features.to(device=self._device, dtype=float_dtype), target_B)

        act = action.to(device=self._device, dtype=float_dtype)
        # ── Embodiment-keyed action preprocessor (mirrors WorldModel.step) ──
        # The eager :class:`WorldModel.step` runs the planner-side action
        # through ``get_action_preprocessor(embodiment_name)`` to convert
        # 7-D droid layout / metric units into the model's canonical
        # ``self._action_dim`` (e.g. 48-D RoboCasa chunked layout A in
        # normalised [-1, 1] space). The compiled dispatcher previously
        # SKIPPED this step and just zero-padded — so when
        # ``--load_compiled bf16`` was on the WM was being fed a
        # mostly-zero, wrong-layout, wrong-units action vector and barely
        # moved. Applying the same preprocessor here keeps the eager
        # and compiled paths bit-identical from the planner's POV.
        if act.shape[-1] != self._action_dim:
            wm_for_embodiments = self._wm
            embodiment_list = getattr(
                getattr(wm_for_embodiments, "model", wm_for_embodiments),
                "embodiment_list",
                None,
            )
            embodiment_name = resolve_embodiment_name(state.embodiment_ids, embodiment_list)
            action_preproc = get_action_preprocessor(embodiment_name)
            act = action_preproc(act, self._action_dim)
            # Diagnostic (one-shot gated, opt-in): log what action actually
            # reaches the AOT program. Useful for debugging the
            # planner-side imagination viz.
            if os.environ.get("RJP_LOG_AOT_ACTIONS", "0") == "1":
                with torch.no_grad():
                    a0 = act[0, 0].detach().float().cpu().numpy()
                logger.info(
                    f"[dispatcher.step] embodiment={embodiment_name} "
                    f"step_idx={step_idx} canonical_action[0,0]={a0.round(3).tolist()}"
                )
        # Defensive fallback (should be a no-op if the preprocessor returned
        # the right size; kept for embodiments without a registered
        # preprocessor).
        if act.shape[-1] != self._action_dim:
            act = F.pad(act, (0, self._action_dim - act.shape[-1]))
        act = self._pad_to(act, target_B)

        st = state.states[:, -1:].to(device=self._device, dtype=float_dtype)
        if st.shape[-1] != self._state_dim:
            wm_for_embodiments = self._wm
            embodiment_list = getattr(
                getattr(wm_for_embodiments, "model", wm_for_embodiments),
                "embodiment_list",
                None,
            )
            embodiment_name = resolve_embodiment_name(state.embodiment_ids, embodiment_list)
            state_preproc = get_state_preprocessor(embodiment_name)
            st = state_preproc(st, self._state_dim)
        if st.shape[-1] != self._state_dim:
            st = F.pad(st, (0, self._state_dim - st.shape[-1]))
        st = self._pad_to(st, target_B)

        view_map = self._pad_to(state.view_map.to(self._device), target_B)
        if state.embodiment_ids is not None:
            embodiment_ids = self._pad_to(state.embodiment_ids.to(self._device), target_B)
        else:
            embodiment_ids = torch.zeros(target_B, 1, dtype=torch.long, device=self._device)

        inputs = [
            vf.contiguous(),
            act.contiguous(),
            st.contiguous(),
            state._aot_cache_keys.contiguous(),
            state._aot_cache_values.contiguous(),
            view_map.contiguous(),
            embodiment_ids.contiguous(),
        ]
        if self._use_torch_compile or self._time_offset_input:
            # frames rolled out of the cache before this step (programs past cache_len roll it by one)
            inputs.append(torch.tensor(max(0, step_idx - self._cache_len), dtype=torch.long, device=self._device))
        # Run AOT compiled model
        with torch.no_grad():
            out, new_ck, new_cv = program(*inputs)

        # Slice back to actual batch size
        out = out[:B_actual]
        # CRITICAL OOM FIX: slice the AOT KV cache back to ``B_actual``
        # before attaching to the returned state. The compiled program
        # always returns the full padded ``target_B`` (e.g. 32) — even
        # when called with B=1 (warmup loop). If we attach the unsliced
        # B=32 caches to a B=1 ``wm_state``, the planner's subsequent
        # ``wm_state.to('cpu')`` copies B=32 worth of cache (~6+ GB at
        # depth=40, num_kv_heads=8, cache_len=8, frame_tokens=600,
        # head_dim=64, bf16) to host RSS — once per env step per rank
        # × 8 ranks → host OOM at 1.5 TB by step ~5. ``.contiguous()``
        # forces a real copy of just the B_actual slice so the underlying
        # full-B storage can be freed by the CUDA caching allocator on
        # the next call.
        state._aot_cache_keys = new_ck[:, :B_actual].contiguous()
        state._aot_cache_values = new_cv[:, :B_actual].contiguous()

        # Reshape AOT output (B, 1, V*H*W, D) -> (B, 1, V, H, W, D) to match
        # state.features layout
        V = self._model_config["num_views"] or 1
        H = self._model_config["grid_height"]
        W = self._model_config["grid_width"]
        logger.debug(f"AOT out shape: {out.shape}, state.features shape: {state.features.shape}, V={V}, H={H}, W={W}")
        if out.ndim == 4:
            # out is (B, 1, V*H*W, D) — reshape to (B, 1, V, H, W, D)
            out = out.view(B_actual, 1, V, H, W, -1)
        elif out.ndim == 6:
            # out is already (B, 1, V, H, W, D)
            pass
        else:
            raise ValueError(f"Unexpected AOT output ndim={out.ndim}, shape={out.shape}")

        # Build new state (matching WorldModel.step() contract)
        new_features = torch.cat([state.features, out.to(state.features.device)], dim=1)
        # Use the *preprocessed* ``act`` (already padded to
        # ``self._action_dim`` by the action_preproc dispatch above)
        # rather than the raw planner-side ``action`` for the actions
        # history. Mirrors :class:`WorldModel.step` (line ~803): both
        # tensors in the cat must be width ``action_dim`` or the cat
        # blows up with a size-mismatch RuntimeError.
        prev_actions = state.actions.to(self._device)
        if prev_actions.shape[-1] != self._action_dim:
            # Always-empty pre-history (B, 0, action_dim) — mirrors
            # ``WorldModel.from_features`` which initialises actions as
            # (B, 0, 7) regardless of the model's actual action_dim.
            prev_actions = torch.zeros(
                (prev_actions.shape[0], prev_actions.shape[1], self._action_dim),
                dtype=prev_actions.dtype,
                device=self._device,
            )
        new_actions = torch.cat([prev_actions, act[:B_actual]], dim=1).float()

        # Compute new pose (same as WorldModel.step)
        poses = state.states
        if self._state_tokens > 0 and poses is not None:
            wm_for_embodiments = self._wm
            embodiment_list_for_pose = getattr(
                getattr(wm_for_embodiments, "model", wm_for_embodiments),
                "embodiment_list",
                None,
            )
            # Pre-pad ``poses`` to ``state_dim`` (mirrors the eager
            # ``WorldModel.step`` path at line ~809). Without this the
            # AOT dispatcher feeds the raw planner-side proprio (e.g.
            # 14-D ``proprio_robocasa`` for ``franka_omron_state_delta``)
            # into ``compute_new_pose``, which can return a width that
            # doesn't match ``poses.shape[-1]`` (DROID-style returns 7;
            # state-delta returns ``min(state_dim, action_dim)``). The
            # subsequent ``F.pad`` then pads ``new_pose`` up to
            # ``state_dim`` (e.g. 25), but ``poses`` is still 14, so
            # the final ``cat`` blows up with
            # ``Sizes of tensors must match except in dimension 1.
            # Expected size 14 but got size 25``.
            if poses.shape[-1] != self._state_dim:
                state_preproc_pre = get_state_preprocessor(
                    resolve_embodiment_name(state.embodiment_ids, embodiment_list_for_pose)
                )
                poses = state_preproc_pre(poses, self._state_dim)
            new_pose = compute_new_pose(
                poses[:, -1:],
                action,
                embodiment_ids=state.embodiment_ids,
                embodiment_list=embodiment_list_for_pose,
            )
            if poses.shape[-1] != new_pose.shape[-1]:
                new_pose = F.pad(new_pose, (0, self._state_dim - new_pose.shape[-1], 0, 0, 0, 0))
            poses = torch.cat([poses, new_pose], dim=1)
        if use_cache and poses is not None:
            poses = poses[:, -1:]

        new_state = WorldModelState(
            features=new_features,
            actions=new_actions,
            states=poses,
            view_map=state.view_map,
            embodiment_ids=state.embodiment_ids,
            dt=state.dt,
            camera_extrinsics=state.camera_extrinsics,
        )
        new_state._aot_cache_keys = state._aot_cache_keys
        new_state._aot_cache_values = state._aot_cache_values

        # Return (last_predicted_frame, new_state) — matching wm.step() contract
        return out, new_state
