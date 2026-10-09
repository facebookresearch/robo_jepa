#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AOT compile all 54 predictor programs (9 steps × 6 batch sizes).

Two-phase pipeline:
  Phase 1 (sequential, GPU): torch.export.export() each (step, batch) combo
  Phase 2 (parallel, CPU):   torch._export.aot_compile() each exported program

If torch.export fails (likely for complex models), falls back to:
  Fallback (sequential, GPU): torch.compile(dynamic=False) + inductor fx_graph_cache

Usage:
    python -u aot_compile_all.py --output-dir /path/to/compiled
    python -u aot_compile_all.py --output-dir /path/to/compiled --steps 0,1,2 --batches 1,32
"""

import argparse
import gc
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import torch
import torch.nn as nn
import torch._inductor.config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def _ensure_libcuda_symlink():
    """Ensure the conda build environment can compile AOTI programs.

    Fixes two issues with conda-packaged compilers on cluster nodes:

    1. **Linker**: ``ld`` cannot find ``-lcuda`` because it doesn't search
       CUDA driver stubs.  We symlink ``libcuda.so`` into the conda lib dir.

    2. **Compiler**: ``c++`` cannot find ``<nv/target>`` (included transitively
       by ``cuda_bf16.h``).  We add the conda ``targets/`` include dir to
       ``CPLUS_INCLUDE_PATH``.

    Both fixes are idempotent and safe to call multiple times.
    """
    lib_dir = os.path.join(sys.prefix, "lib")

    # --- Fix 1: libcuda.so symlink ---
    target = os.path.join(lib_dir, "libcuda.so")
    if not os.path.exists(target):
        for candidate in [
            os.path.join(lib_dir, "stubs", "libcuda.so"),
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib64/libcuda.so",
            "/usr/local/cuda/lib64/stubs/libcuda.so",
        ]:
            if os.path.exists(candidate):
                try:
                    os.symlink(candidate, target)
                except OSError:
                    pass
                break

    # --- Fix 2: CUDA include path for <nv/target> ---
    cuda_include = os.path.join(sys.prefix, "targets", "x86_64-linux", "include")
    if os.path.isdir(cuda_include):
        existing = os.environ.get("CPLUS_INCLUDE_PATH", "")
        if cuda_include not in existing:
            os.environ["CPLUS_INCLUDE_PATH"] = f"{cuda_include}:{existing}" if existing else cuda_include


from app.robo_jepa.world_model import WorldModel
from app.robo_jepa.robo_jepa.models.utils.modules import KeyValueCache


def load_shared_constants(loaded, constants):
    """Inject externally held weights into an AOTI package built with ``package_constants_in_so=False``.

    Shares the tensors (``user_managed=True``) where torch supports it; torch 2.7 has no such argument and copies
    them instead.
    """
    try:
        loaded.load_constants(constants, check_full_update=False, user_managed=True)
    except TypeError:
        loaded.load_constants(constants, check_full_update=False)


def swap_attn_impl(model, attn_impl: str):
    """Recursively swap attn_impl on all RoPEAttention/Attention modules.

    This allows changing the attention backend on a model that was loaded
    with a different attn_impl (e.g., from "fmha" to "sdpa_cudnn").
    Weights are unaffected since attn_impl only controls the backend dispatch.
    """
    from app.robo_jepa.robo_jepa.models.utils.modules import RoPEAttention, Attention

    count = 0
    for module in model.modules():
        if isinstance(module, (RoPEAttention, Attention)):
            module.attn_impl = attn_impl
            count += 1
    # Also update attn_impl on predictor-level (affects blockwise mask logic)
    if hasattr(model, "attn_impl"):
        model.attn_impl = attn_impl
    # Handle PredictorMultiSeqWrapper → backbone
    if hasattr(model, "backbone") and hasattr(model.backbone, "attn_impl"):
        model.backbone.attn_impl = attn_impl
    return count


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ── StepModule: nn.Module wrapper for torch.export ───────────────────────────


class StepModule(nn.Module):
    """Wraps transition_model as a proper nn.Module with flat tensor I/O.

    Bakes cache_size and input_position as constants so torch.export
    sees fully static shapes — no symbolic variables.
    """

    def __init__(self, transition_model, cache_size, input_position):
        super().__init__()
        self.transition_model = transition_model
        self.cache_size = cache_size
        self.input_position = input_position

    def forward(
        self,
        video_features,
        actions,
        states,
        cache_keys,
        cache_values,
        view_map,
        embodiment_ids,
        time_offset,
    ):
        # time_offset: 0-dim int64, frames rolled out of the cache before this step (KeyValueCache.evicted).
        # A runtime input because every step past cache_len shares one program.
        layers = cache_keys.shape[0]
        # Match cache dtype for flash attention compatibility
        target_dtype = cache_keys.dtype
        cache = KeyValueCache(
            layers=layers,
            batch=cache_keys.shape[1],
            frame_tokens=cache_keys.shape[4],
            cache_len=cache_keys.shape[3],
            num_heads=cache_keys.shape[2],
            head_dim=cache_keys.shape[5],
            dtype=target_dtype,
            device=cache_keys.device,
            keys=[cache_keys[i].clone() for i in range(layers)],
            values=[cache_values[i].clone() for i in range(layers)],
            pointer=[self.cache_size] * layers,
            evicted=time_offset,
        )

        out, _act, _state, cache = self.transition_model(
            video_features,
            actions=actions,
            states=states,
            camera_extrinsics=None,
            use_cache=True,
            cache=cache,
            cache_size=self.cache_size,
            input_position=self.input_position,
            view_map=view_map,
            embodiment_ids=embodiment_ids,
            dt=None,
        )

        return out, torch.stack(cache.keys, dim=0), torch.stack(cache.values, dim=0)


class FunctionalStepModule(nn.Module):
    """Mutation-free StepModule for CUDA graph and TensorRT compatibility.

    Unlike StepModule which clones cache tensors and relies on in-place
    mutations inside KeyValueCache.update(), this version passes cache
    tensors as read-only inputs through a functional path. Each attention
    layer returns new key/value tensors instead of mutating in place.

    This eliminates the ~80 in-place mutations that block CUDA graph capture.
    """

    def __init__(self, transition_model, cache_size, input_position):
        super().__init__()
        self.transition_model = transition_model
        self.cache_size = cache_size
        self.input_position = input_position

    def forward(
        self,
        video_features,
        actions,
        states,
        cache_keys,
        cache_values,
        view_map,
        embodiment_ids,
        time_offset,
    ):
        # time_offset: 0-dim int64, frames rolled out of the cache before this step (KeyValueCache.evicted).
        # A runtime input because every step past cache_len shares one program.
        layers = cache_keys.shape[0]
        target_dtype = cache_keys.dtype
        # Create a cache object (needed for metadata like cache_len), but
        # the actual K/V data flows through functional_cache_tensors.
        cache = KeyValueCache(
            layers=layers,
            batch=cache_keys.shape[1],
            frame_tokens=cache_keys.shape[4],
            cache_len=cache_keys.shape[3],
            num_heads=cache_keys.shape[2],
            head_dim=cache_keys.shape[5],
            dtype=target_dtype,
            device=cache_keys.device,
            keys=[cache_keys[i] for i in range(layers)],
            values=[cache_values[i] for i in range(layers)],
            pointer=[self.cache_size] * layers,
            evicted=time_offset,
        )

        result = self.transition_model(
            video_features,
            actions=actions,
            states=states,
            camera_extrinsics=None,
            use_cache=True,
            cache=cache,
            cache_size=self.cache_size,
            input_position=self.input_position,
            view_map=view_map,
            embodiment_ids=embodiment_ids,
            dt=None,
            functional_cache_tensors=(cache_keys, cache_values),
        )
        # In functional mode, the predictor returns 6 values:
        # (out, action_tokens, state_tokens, cache, new_cache_keys, new_cache_values)
        out = result[0]
        new_cache_keys = result[4]
        new_cache_values = result[5]

        return out, new_cache_keys, new_cache_values


class MixedPrecisionStepModule(nn.Module):
    """Mixed-precision (mp-bf16) variant of FunctionalStepModule.

    Keeps model weights in fp32 and wraps the forward pass in
    torch.amp.autocast(dtype=bfloat16). This gives bf16 matmul speed
    with fp32 accumulation — matching the behavior of eager inference
    under autocast.

    Inputs: video_features, actions, states are fp32; cache tensors are bf16.
    Outputs: predictions are bf16; cache tensors are bf16.
    """

    def __init__(self, transition_model, cache_size, input_position):
        super().__init__()
        self.transition_model = transition_model
        self.cache_size = cache_size
        self.input_position = input_position

    def forward(
        self,
        video_features,
        actions,
        states,
        cache_keys,
        cache_values,
        view_map,
        embodiment_ids,
        time_offset,
    ):
        # time_offset: 0-dim int64, frames rolled out of the cache before this step (KeyValueCache.evicted).
        # A runtime input because every step past cache_len shares one program.
        layers = cache_keys.shape[0]
        target_dtype = cache_keys.dtype
        cache = KeyValueCache(
            layers=layers,
            batch=cache_keys.shape[1],
            frame_tokens=cache_keys.shape[4],
            cache_len=cache_keys.shape[3],
            num_heads=cache_keys.shape[2],
            head_dim=cache_keys.shape[5],
            dtype=target_dtype,
            device=cache_keys.device,
            keys=[cache_keys[i] for i in range(layers)],
            values=[cache_values[i] for i in range(layers)],
            pointer=[self.cache_size] * layers,
            evicted=time_offset,
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = self.transition_model(
                video_features,
                actions=actions,
                states=states,
                camera_extrinsics=None,
                use_cache=True,
                cache=cache,
                cache_size=self.cache_size,
                input_position=self.input_position,
                view_map=view_map,
                embodiment_ids=embodiment_ids,
                dt=None,
                functional_cache_tensors=(cache_keys, cache_values),
            )

        out = result[0]
        new_cache_keys = result[4]
        new_cache_values = result[5]

        return out, new_cache_keys, new_cache_values


# ── Build example inputs ─────────────────────────────────────────────────────


def build_example_inputs(model_config, step_idx, batch_size, cache_len, device="cuda"):
    """Build concrete example inputs for a given (step_idx, batch_size)."""
    B = batch_size
    V = model_config["num_views"] if model_config["num_views"] else 1
    H = model_config["grid_height"]
    W = model_config["grid_width"]
    embed_dim = model_config["encoder_embed_dim"]
    action_dim = model_config["action_dim"]
    state_dim = model_config["state_dim"]
    depth = model_config["depth"]
    num_heads = model_config["num_kv_heads"] or model_config["num_heads"]
    head_dim = model_config["predictor_embed_dim"] // model_config["num_heads"]
    frame_tokens = V * H * W + model_config["action_tokens"]
    if model_config["state_tokens"] > 0:
        frame_tokens += model_config["state_tokens"]

    cache_size = min(step_idx, cache_len)

    video_features = torch.randn(B, 1, V, H, W, embed_dim, device=device, dtype=torch.float32)
    actions = torch.randn(B, 1, action_dim, device=device, dtype=torch.float32)
    states = torch.randn(B, 1, state_dim, device=device, dtype=torch.float32)
    view_map = torch.zeros(B, V, dtype=torch.long, device=device)
    embodiment_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
    cache_keys = torch.zeros(
        depth,
        B,
        num_heads,
        cache_len,
        frame_tokens,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    cache_values = torch.zeros(
        depth,
        B,
        num_heads,
        cache_len,
        frame_tokens,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    time_offset = torch.tensor(max(0, step_idx - cache_len), dtype=torch.long, device=device)

    return (
        video_features,
        actions,
        states,
        cache_keys,
        cache_values,
        view_map,
        embodiment_ids,
        time_offset,
    )


def get_model_config(wm, grid_height=None, grid_width=None):
    """Extract model config needed for building example inputs.

    Args:
        wm: WorldModel instance.
        grid_height: Override grid height (for cross-resolution compilation).
        grid_width: Override grid width (for cross-resolution compilation).
    """
    m = wm.model
    encoder_dim = wm.encoder.vit.embed_dim if wm.encoder else m.predictor_embed_dim
    return {
        "view_map": m.view_map,
        "num_views": m.num_views,
        "grid_height": grid_height if grid_height is not None else m.grid_height,
        "grid_width": grid_width if grid_width is not None else m.grid_width,
        "encoder_embed_dim": encoder_dim,
        "action_dim": m.action_dim,
        "state_dim": m.state_dim,
        "depth": m.depth,
        "num_heads": m.num_heads,
        "num_kv_heads": m.num_kv_heads,
        "predictor_embed_dim": m.predictor_embed_dim,
        "action_tokens": m.action_tokens,
        "state_tokens": m.state_tokens,
        "cache_len": m.attn_depth,
    }


# ── Phase 1: Export ──────────────────────────────────────────────────────────


def export_one(wm, model_config, step_idx, batch_size, output_dir, device="cuda"):
    """Export one (step, batch) program. Returns path to .pt2 file or None on failure."""
    cache_len = model_config["cache_len"]
    cache_size = min(step_idx, cache_len)
    tag = f"step{step_idx}_b{batch_size}"

    flush_print(f"  [{tag}] Exporting (cache_size={cache_size})...")
    t0 = time.perf_counter()

    module = StepModule(wm.transition_model, cache_size=cache_size, input_position=step_idx)
    module.eval()

    example_inputs = build_example_inputs(model_config, step_idx, batch_size, cache_len, device)

    # Dry run to catch runtime errors before export
    with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16):
        _out = module(*example_inputs)

    try:
        exported = torch.export.export(module, example_inputs)
        pt2_path = os.path.join(output_dir, f"{tag}.pt2")
        torch.export.save(exported, pt2_path)
        dt = time.perf_counter() - t0
        flush_print(f"  [{tag}] Exported in {dt:.1f}s -> {pt2_path}")
        return pt2_path
    except Exception as e:
        flush_print(f"  [{tag}] Export FAILED: {e}")
        return None


# ── Phase 2: AOT compile (CPU-parallel) ──────────────────────────────────────


def aot_compile_one(pt2_path, output_dir):
    """Load a .pt2 file and AOT compile to .so. CPU-only, can run in parallel."""
    tag = os.path.basename(pt2_path).replace(".pt2", "")
    so_path = os.path.join(output_dir, f"{tag}.so")

    try:
        exported = torch.export.load(pt2_path)
        so_path_out = torch._export.aot_compile(exported.module(), ())
        # Move to our output directory
        import shutil

        shutil.move(so_path_out, so_path)
        return tag, so_path, None
    except Exception as e:
        return tag, None, str(e)


# ── Fallback: torch.compile with inductor cache (PARALLEL multi-GPU) ─────────


def _compile_worker(gpu_id, combos, model_dir, model_config, output_dir):
    """Worker process: loads model on assigned GPU, compiles its combos."""
    import torch
    import torch._inductor.config

    torch._inductor.config.fx_graph_cache = True
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(output_dir, "inductor_cache")

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    flush_print(f"  [GPU {gpu_id}] Loading model ({len(combos)} combos)...")
    wm = WorldModel.load_pretrained(model_dir, device=device, load_decoder=False)
    wm.eval()
    flush_print(f"  [GPU {gpu_id}] Model loaded. Starting compilation.")

    cache_len = model_config["cache_len"]
    results = []
    for step_idx, batch_size in combos:
        cache_size = min(step_idx, cache_len)
        tag = f"step{step_idx}_b{batch_size}"
        flush_print(f"  [GPU {gpu_id}][{tag}] Compiling (cache_size={cache_size})...")
        t0 = time.perf_counter()

        try:
            module = StepModule(wm.transition_model, cache_size=cache_size, input_position=step_idx)
            module.eval()
            compiled = torch.compile(module, mode="default", dynamic=False)
            example_inputs = build_example_inputs(model_config, step_idx, batch_size, cache_len, device)

            with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16):
                for _ in range(2):
                    _out = compiled(*example_inputs)
            torch.cuda.synchronize()

            dt = time.perf_counter() - t0
            flush_print(f"  [GPU {gpu_id}][{tag}] Compiled in {dt:.1f}s")

            marker = os.path.join(output_dir, f"{tag}.compiled")
            with open(marker, "w") as f:
                f.write(f"compiled_at={time.time()}\nduration_s={dt:.1f}\ngpu={gpu_id}\n")
            results.append((tag, dt, None))
        except Exception as e:
            dt = time.perf_counter() - t0
            flush_print(f"  [GPU {gpu_id}][{tag}] FAILED in {dt:.1f}s: {e}")
            results.append((tag, dt, str(e)))

        gc.collect()
        torch.cuda.empty_cache()

    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="AOT compile RoboJEPA predictor programs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="compiled_programs",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="WorldModel checkpoint dir (config.json + weights)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step indices (default: 0..cache_len)",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for AOT compile phase",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    batch_sizes = [int(b) for b in args.batches.split(",")]

    flush_print("=" * 70)
    flush_print("AOT Compile — RoboJEPA Predictor Programs")
    flush_print(f"GPU: {torch.cuda.get_device_name()}")
    flush_print(f"PyTorch: {torch.__version__}")
    flush_print(f"Output: {args.output_dir}")
    flush_print("=" * 70)

    # Enable inductor fx_graph_cache (fallback path)
    torch._inductor.config.fx_graph_cache = True
    cache_dir = os.path.join(args.output_dir, "inductor_cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    # Load model
    flush_print("\nLoading model...")
    wm = WorldModel.load_pretrained(args.model_dir, device=args.device, load_decoder=False)
    wm.eval()
    model_config = get_model_config(wm)
    cache_len = model_config["cache_len"]
    flush_print(
        f"  cache_len={cache_len}, depth={model_config['depth']}, "
        f"heads={model_config['num_kv_heads'] or model_config['num_heads']}"
    )

    if args.steps is None:
        step_indices = list(range(cache_len + 1))  # 0..cache_len
    else:
        step_indices = [int(s) for s in args.steps.split(",")]

    combos = [(s, b) for s in step_indices for b in batch_sizes]
    flush_print(f"\nWill compile {len(combos)} programs: steps={step_indices}, batches={batch_sizes}")

    # ── Phase 1: Try torch.export ────────────────────────────────────
    flush_print(f"\n{'=' * 70}")
    flush_print("Phase 1: torch.export (sequential, GPU)")
    flush_print(f"{'=' * 70}")

    export_succeeded = []
    export_failed = []

    # Try exporting the first combo to see if torch.export works at all
    test_step, test_batch = combos[0]
    pt2 = export_one(wm, model_config, test_step, test_batch, args.output_dir, args.device)
    if pt2 is not None:
        export_succeeded.append((test_step, test_batch, pt2))
        flush_print(f"\ntorch.export works! Exporting remaining {len(combos) - 1} programs...")
        for step_idx, batch_size in combos[1:]:
            pt2 = export_one(wm, model_config, step_idx, batch_size, args.output_dir, args.device)
            if pt2:
                export_succeeded.append((step_idx, batch_size, pt2))
            else:
                export_failed.append((step_idx, batch_size))
    else:
        flush_print("\ntorch.export failed on first combo. Skipping export phase.")
        export_failed = combos

    # ── Phase 2: AOT compile exported programs (parallel, CPU) ───────
    if export_succeeded:
        flush_print(f"\n{'=' * 70}")
        flush_print(f"Phase 2: aot_compile ({len(export_succeeded)} programs, {args.workers} workers)")
        flush_print(f"{'=' * 70}")

        pt2_paths = [pt2 for _, _, pt2 in export_succeeded]
        aot_results = {}

        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(aot_compile_one, pt2, args.output_dir): pt2 for pt2 in pt2_paths}
                for future in as_completed(futures):
                    tag, so_path, error = future.result()
                    if so_path:
                        flush_print(f"  [{tag}] AOT compiled -> {so_path}")
                        aot_results[tag] = so_path
                    else:
                        flush_print(f"  [{tag}] AOT compile FAILED: {error}")
        except Exception as e:
            flush_print(f"  Parallel AOT compile failed: {e}")
            traceback.print_exc()

    # ── Fallback: torch.compile + inductor cache (parallel multi-GPU) ──
    fallback_combos = export_failed
    if not export_succeeded:
        fallback_combos = combos

    if fallback_combos:
        n_gpus = torch.cuda.device_count()
        flush_print(f"\n{'=' * 70}")
        flush_print(f"Fallback: torch.compile + inductor cache ({len(fallback_combos)} programs, {n_gpus} GPUs)")
        flush_print(f"{'=' * 70}")

        # Free the model on GPU 0 to make room for workers
        del wm
        gc.collect()
        torch.cuda.empty_cache()

        if n_gpus > 1:
            # Split combos across GPUs round-robin
            gpu_combos = [[] for _ in range(n_gpus)]
            for i, combo in enumerate(fallback_combos):
                gpu_combos[i % n_gpus].append(combo)

            import torch.multiprocessing as mp

            mp.set_start_method("spawn", force=True)

            with mp.Pool(n_gpus) as pool:
                worker_results = pool.starmap(
                    _compile_worker,
                    [
                        (
                            gpu_id,
                            gpu_combos[gpu_id],
                            args.model_dir,
                            model_config,
                            args.output_dir,
                        )
                        for gpu_id in range(n_gpus)
                        if gpu_combos[gpu_id]
                    ],
                )
            all_results = [r for batch in worker_results for r in batch]
        else:
            # Single GPU — run sequentially
            all_results = _compile_worker(0, fallback_combos, args.model_dir, model_config, args.output_dir)

        n_ok = sum(1 for _, _, err in all_results if err is None)
        n_fail = sum(1 for _, _, err in all_results if err is not None)
    else:
        all_results = []
        n_ok = 0
        n_fail = 0

    # ── Summary ──────────────────────────────────────────────────────
    flush_print(f"\n{'=' * 70}")
    flush_print("COMPILATION SUMMARY")
    flush_print(f"{'=' * 70}")
    flush_print(f"  Total combos: {len(combos)}")
    if export_succeeded:
        flush_print(f"  torch.export succeeded: {len(export_succeeded)}")
        if aot_results:
            flush_print(f"  AOT .so compiled: {len(aot_results)}")
    if fallback_combos:
        flush_print(f"  Inductor cache compiled: {len(fallback_combos)}")
    flush_print(f"  Output directory: {args.output_dir}")
    flush_print(f"  Inductor cache: {cache_dir}")


if __name__ == "__main__":
    main()
