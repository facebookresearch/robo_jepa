#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compile AOT inference programs for the DiT diffusion decoder.

This script takes a checkpoint directory (must contain a decoder/),
and compiles all (cache_step × update_cache × batch_size) .pt2 programs
into a compiled/decoder-bf16/ subdirectory inside the checkpoint.

Program matrix: 5 cache_steps × 2 update_cache × 6 batch_sizes = 60 programs.

Usage:
    # Single GPU (sequential, ~2 hours):
    python -m app.robo_jepa.inference.compile_decoder_checkpoint /path/to/checkpoint

    # Multi-GPU (parallel, ~30 min on 4 GPUs):
    python -m app.robo_jepa.inference.compile_decoder_checkpoint /path/to/checkpoint --num-gpus 4

    # Custom batch sizes / cache steps:
    python -m app.robo_jepa.inference.compile_decoder_checkpoint /path/to/checkpoint --batch-sizes 1,4,16

    # Force recompilation:
    python -m app.robo_jepa.inference.compile_decoder_checkpoint /path/to/checkpoint --force

Example SLURM submission (recommended):
    sbatch --gpus-per-node=4 --cpus-per-task=64 --mem=512G --time=06:00:00 \\
        --wrap="cd <repo_root> && \\
        python -m app.robo_jepa.inference.compile_decoder_checkpoint /path/to/checkpoint --num-gpus 4"
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger("compile_decoder")

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DEFAULT_CACHE_STEPS = [0, 1, 2, 3, 4]


def _ensure_cuda_env():
    """Ensure CUDA headers and libcuda symlink are available."""
    conda_prefix = os.environ.get("CONDA_PREFIX", os.path.dirname(os.path.dirname(sys.executable)))
    targets_include = os.path.join(conda_prefix, "targets", "x86_64-linux", "include")
    if os.path.isdir(targets_include):
        existing = os.environ.get("CPLUS_INCLUDE_PATH", "")
        if targets_include not in existing:
            os.environ["CPLUS_INCLUDE_PATH"] = f"{targets_include}:{existing}"
    if "CUDA_HOME" not in os.environ and os.path.isdir(conda_prefix):
        os.environ["CUDA_HOME"] = conda_prefix

    from app.robo_jepa.inference.aot_compile import _ensure_libcuda_symlink

    _ensure_libcuda_symlink()


def compile_one(
    dit_model,
    decoder_config: dict,
    cache_step: int,
    update_cache: bool,
    batch_size: int,
    artifact_dir: str,
    force: bool = False,
    device: str = "cuda",
) -> str:
    """Compile a single decoder .pt2 program.

    Returns the path to the compiled .pt2 file.
    """
    from app.robo_jepa.inference.decoder_aot_compile import (
        DecoderStepModule,
        build_decoder_example_inputs,
    )

    cache_len = decoder_config["cache_len"]
    cache_size = min(cache_step, cache_len)
    tag = f"decoder_step{cache_step}_update{int(update_cache)}_b{batch_size}"
    pt2_path = os.path.join(artifact_dir, f"{tag}.pt2")

    if os.path.exists(pt2_path) and not force:
        logger.info(f"  {tag}: already exists, skipping")
        return pt2_path

    logger.info(f"  {tag} (cache_size={cache_size})...")
    t0 = time.perf_counter()

    module = DecoderStepModule(dit_model, cache_size=cache_size, update_cache=update_cache).eval().to(torch.bfloat16)
    example = build_decoder_example_inputs(decoder_config, cache_step, batch_size, device)

    # Dry run to populate lazy init
    with torch.no_grad():
        module(*example)

    with torch.no_grad():
        exported = torch.export.export(module, example)

    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=pt2_path,
        inductor_configs={"aot_inductor.package_constants_in_so": False},
    )

    dt = time.perf_counter() - t0
    size_mb = os.path.getsize(pt2_path) / 1e6
    logger.info(f"  {tag}: {size_mb:.0f}MB in {dt:.1f}s")

    del module, exported
    gc.collect()
    torch.cuda.empty_cache()

    return pt2_path


def _worker_compile(args):
    """Worker function for parallel compilation on a specific GPU."""
    gpu_id, combos, model_dir, decoder_config, artifact_dir, force = args

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    worker_logger = logging.getLogger(f"compile_decoder.gpu{gpu_id}")
    worker_logger.info(f"GPU {gpu_id}: compiling {len(combos)} programs")

    from app.robo_jepa.world_model import WorldModel

    # Load to CPU, move only decoder to target GPU to avoid OOM
    wm = WorldModel.load_pretrained(model_dir, device="cpu", load_decoder=True)
    wm.eval()
    dit = wm.heads["image_head"].ema_model.to(device)
    wm.encoder = None
    wm.transition_model = None
    del wm
    gc.collect()
    torch.cuda.empty_cache()

    _ensure_cuda_env()

    for cache_step, update_cache, batch_size in combos:
        try:
            compile_one(
                dit,
                decoder_config,
                cache_step,
                update_cache,
                batch_size,
                artifact_dir,
                force=force,
                device=device,
            )
        except Exception as e:
            worker_logger.error(f"GPU {gpu_id}: FAILED step{cache_step}_update{int(update_cache)}_b{batch_size}: {e}")

    del dit
    torch.cuda.empty_cache()
    worker_logger.info(f"GPU {gpu_id}: done")


def compile_all(
    dit_model,
    decoder_config: dict,
    artifact_dir: str,
    batch_sizes: List[int],
    cache_steps: List[int],
    force: bool = False,
    device: str = "cuda",
):
    """Compile all programs sequentially on one GPU."""
    os.makedirs(artifact_dir, exist_ok=True)

    combos = [(cs, uc, bs) for cs in cache_steps for uc in [False, True] for bs in batch_sizes]
    total = len(combos)
    logger.info(f"Compiling {total} programs sequentially")

    t_total = time.perf_counter()
    for i, (cs, uc, bs) in enumerate(combos, 1):
        logger.info(f"[{i}/{total}]")
        compile_one(
            dit_model,
            decoder_config,
            cs,
            uc,
            bs,
            artifact_dir,
            force=force,
            device=device,
        )

    dt = time.perf_counter() - t_total
    logger.info(f"Compilation complete: {total} programs in {dt:.0f}s ({dt / 60:.1f} min)")


def compile_parallel(
    model_dir: str,
    decoder_config: dict,
    artifact_dir: str,
    batch_sizes: List[int],
    cache_steps: List[int],
    num_gpus: int,
    force: bool = False,
):
    """Compile all programs in parallel across multiple GPUs."""
    import torch.multiprocessing as mp

    os.makedirs(artifact_dir, exist_ok=True)

    combos = [(cs, uc, bs) for cs in cache_steps for uc in [False, True] for bs in batch_sizes]
    total = len(combos)

    # Round-robin distribute across GPUs
    gpu_combos = [[] for _ in range(num_gpus)]
    for i, combo in enumerate(combos):
        gpu_combos[i % num_gpus].append(combo)

    for g in range(num_gpus):
        logger.info(f"  GPU {g}: {len(gpu_combos[g])} programs")

    t_total = time.perf_counter()

    worker_args = [(g, gpu_combos[g], model_dir, decoder_config, artifact_dir, force) for g in range(num_gpus)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_gpus) as pool:
        pool.map(_worker_compile, worker_args)

    dt = time.perf_counter() - t_total
    logger.info(f"Parallel compilation done: {total} programs in {dt:.0f}s ({dt / 60:.1f} min)")


def save_weights(dit_model, artifact_dir: str):
    """Save shared decoder weights for weight injection at load time."""
    from app.robo_jepa.inference.decoder_aot_compile import DecoderStepModule

    module = DecoderStepModule(dit_model, cache_size=0, update_cache=False).eval().to(torch.bfloat16)
    weights = {k: v.cpu() for k, v in module.state_dict().items()}
    weights_path = os.path.join(artifact_dir, "decoder_weights.pt")
    torch.save(weights, weights_path)
    logger.info(f"  Saved decoder weights: {os.path.getsize(weights_path) / 1e6:.0f}MB")
    del module, weights
    gc.collect()


def save_manifest(
    decoder_config: dict,
    artifact_dir: str,
    batch_sizes: List[int],
    cache_steps: List[int],
):
    """Save manifest.json with compilation metadata."""
    manifest = {
        "type": "decoder_aot",
        "decoder_config": decoder_config,
        "batch_sizes": batch_sizes,
        "cache_steps": cache_steps,
        "update_cache_variants": [False, True],
        "time_offset_input": True,
        "programs": {},
    }
    for cs in cache_steps:
        for uc in [False, True]:
            for bs in batch_sizes:
                tag = f"decoder_step{cs}_update{int(uc)}_b{bs}"
                pt2_path = os.path.join(artifact_dir, f"{tag}.pt2")
                manifest["programs"][tag] = {
                    "cache_step": cs,
                    "update_cache": uc,
                    "batch_size": bs,
                    "file": f"{tag}.pt2",
                    "exists": os.path.exists(pt2_path),
                    "size_mb": round(os.path.getsize(pt2_path) / 1e6, 1) if os.path.exists(pt2_path) else 0,
                }

    manifest_path = os.path.join(artifact_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Saved manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile AOT inference programs for the DiT diffusion decoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the model checkpoint directory (must contain decoder/)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for parallel compilation. Default: 1 (sequential).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes. Default: 1,2,4,8,16,32",
    )
    parser.add_argument(
        "--cache-steps",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated cache step indices. Default: 0,1,2,3,4",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <checkpoint>/compiled/decoder-bf16/",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Override pixel resolution for cross-resolution compilation, e.g. '256x256'. "
        "Converts to encoder grid dims via patch_size=16. Output goes to "
        "compiled/decoder-bf16-<HxW>/. Default: use training resolution.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompilation even if .pt2 files already exist.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isdir(checkpoint):
        logger.error(f"Checkpoint directory not found: {checkpoint}")
        sys.exit(1)

    decoder_dir = os.path.join(checkpoint, "decoder")
    if not os.path.isdir(decoder_dir):
        logger.error(f"No decoder/ subdirectory in {checkpoint}")
        sys.exit(1)

    dir_suffix = "decoder-bf16"
    if args.resolution:
        dir_suffix += f"-{args.resolution}"
    artifact_dir = args.output_dir or os.path.join(checkpoint, "compiled", dir_suffix)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    cache_steps = [int(x) for x in args.cache_steps.split(",")]
    num_gpus = args.num_gpus or 1

    total_programs = len(cache_steps) * 2 * len(batch_sizes)

    logger.info("=" * 80)
    logger.info("AOT Compilation for DiT Diffusion Decoder")
    logger.info("=" * 80)
    logger.info(f"  Checkpoint:    {checkpoint}")
    logger.info(f"  Output:        {artifact_dir}")
    logger.info(f"  Batch sizes:   {batch_sizes}")
    logger.info(f"  Cache steps:   {cache_steps}")
    logger.info(f"  Total:         {total_programs} programs")
    logger.info(f"  GPUs:          {num_gpus} ({'parallel' if num_gpus > 1 else 'sequential'})")
    logger.info(f"  Force:         {args.force}")
    if torch.cuda.is_available():
        logger.info(f"  GPU:           {torch.cuda.get_device_name()}")
    logger.info(f"  PyTorch:       {torch.__version__}")
    logger.info("=" * 80)

    _ensure_cuda_env()

    from app.robo_jepa.inference.decoder_aot_compile import get_decoder_config
    from app.robo_jepa.world_model import WorldModel

    # Load model to CPU first, then move only decoder to GPU to avoid OOM
    logger.info("Loading model (to CPU)...")
    t0 = time.perf_counter()
    wm = WorldModel.load_pretrained(checkpoint, device="cpu", load_decoder=True)
    wm.eval()
    logger.info(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    # Parse resolution override
    res_grid_h, res_grid_w = None, None
    if args.resolution:
        parts = args.resolution.lower().split("x")
        if len(parts) != 2:
            logger.error(f"Invalid --resolution format: {args.resolution}. Expected HxW, e.g. 256x256")
            sys.exit(1)
        pixel_h, pixel_w = int(parts[0]), int(parts[1])
        res_grid_h = pixel_h // 16
        res_grid_w = pixel_w // 16
        logger.info(f"  Resolution override: {pixel_h}x{pixel_w} -> encoder grid {res_grid_h}x{res_grid_w}")

    decoder_config = get_decoder_config(wm, grid_height=res_grid_h, grid_width=res_grid_w)
    logger.info(
        f"  Decoder: depth={decoder_config['depth']}, "
        f"embed_dim={decoder_config['decoder_embed_dim']}, "
        f"heads={decoder_config['num_heads']}/{decoder_config['num_kv_heads']}, "
        f"cache_len={decoder_config['cache_len']}, "
        f"grid={decoder_config['grid_height']}x{decoder_config['grid_width']}"
    )

    # Move only decoder to GPU, discard encoder/transition_model
    dit = wm.heads["image_head"].ema_model.to("cuda:0")
    wm.encoder = None
    wm.transition_model = None
    del wm
    gc.collect()
    torch.cuda.empty_cache()
    mem_mb = torch.cuda.memory_allocated() / 1e6
    logger.info(f"  Decoder on GPU ({mem_mb:.0f}MB), encoder/transition_model discarded")

    os.makedirs(artifact_dir, exist_ok=True)

    # Compile
    t_start = time.perf_counter()
    if num_gpus > 1:
        del dit
        gc.collect()
        torch.cuda.empty_cache()

        compile_parallel(
            model_dir=checkpoint,
            decoder_config=decoder_config,
            artifact_dir=artifact_dir,
            batch_sizes=batch_sizes,
            cache_steps=cache_steps,
            num_gpus=num_gpus,
            force=args.force,
        )

        # Reload decoder only for saving weights
        wm = WorldModel.load_pretrained(checkpoint, device="cpu", load_decoder=True)
        wm.eval()
        dit = wm.heads["image_head"].ema_model
    else:
        compile_all(
            dit_model=dit,
            decoder_config=decoder_config,
            artifact_dir=artifact_dir,
            batch_sizes=batch_sizes,
            cache_steps=cache_steps,
            force=args.force,
        )

    total_time = time.perf_counter() - t_start
    logger.info(f"Compilation done in {total_time:.0f}s ({total_time / 60:.1f} min)")

    # Save weights + manifest
    logger.info("Saving weights and manifest...")
    save_weights(dit, artifact_dir)
    save_manifest(decoder_config, artifact_dir, batch_sizes, cache_steps)

    # Report
    n_pt2 = 0
    total_size = 0
    for f in os.listdir(artifact_dir):
        fp = os.path.join(artifact_dir, f)
        if os.path.isfile(fp):
            total_size += os.path.getsize(fp)
            if f.endswith(".pt2"):
                n_pt2 += 1

    logger.info("=" * 80)
    logger.info("Done!")
    logger.info(f"  Programs:  {n_pt2} .pt2 files + decoder_weights.pt + manifest.json")
    logger.info(f"  Size:      {total_size / (1024 * 1024):.0f} MB")
    logger.info(f"  Location:  {artifact_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
