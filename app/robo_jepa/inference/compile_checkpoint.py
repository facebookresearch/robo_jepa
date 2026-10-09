#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compile AOT inference programs for a RoboJEPA world model checkpoint.

This script takes a checkpoint directory, auto-detects the model config
(resolution, num_views, etc.), and compiles all 54 (step × batch_size)
.pt2 programs into a compiled/ subdirectory inside the checkpoint.

Usage:
    # Single GPU (sequential, ~2-3 hours):
    python -m app.robo_jepa.inference.compile_checkpoint /path/to/checkpoint

    # Multi-GPU (parallel, ~40-60 min on 4 GPUs):
    python -m app.robo_jepa.inference.compile_checkpoint /path/to/checkpoint --num-gpus 4

    # Custom batch sizes / steps:
    python -m app.robo_jepa.inference.compile_checkpoint /path/to/checkpoint --batch-sizes 1,4,16

    # Force recompilation of all programs:
    python -m app.robo_jepa.inference.compile_checkpoint /path/to/checkpoint --force

Example SLURM submission (recommended):
    sbatch --gpus-per-node=4 --cpus-per-task=64 --mem=512G --time=06:00:00 \\
        --wrap="cd <repo_root> && \\
        python -m app.robo_jepa.inference.compile_checkpoint /path/to/checkpoint --num-gpus 4"
"""

import argparse
import json
import logging
import os
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Compile AOT inference programs for a RoboJEPA checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the model checkpoint directory (must contain config.json + world_model.pth.tar)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for parallel compilation. Default: 1 (sequential). "
        "Set to match your SLURM --gpus-per-node for parallel compilation.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes to compile for. Default: 1,2,4,8,16,32",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="0,1,2,3,4,5,6,7,8",
        help="Comma-separated step indices to compile for. Default: 0,1,2,3,4,5,6,7,8",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for compiled artifacts. Default: <checkpoint>/compiled/<precision>/",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "mp-bf16"],
        help="Compilation precision. bf16: all bf16 weights+compute. "
        "mp-bf16: fp32 weights with autocast bf16 (fp32 accumulation). Default: bf16",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="fmha",
        choices=["fmha", "sdpa", "sdpa_cudnn"],
        help="Attention implementation backend. fmha: xformers (default). "
        "sdpa: PyTorch SDPA. sdpa_cudnn: cuDNN attention (fastest on Hopper/H200).",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Override pixel resolution for cross-resolution compilation, e.g. '256x256'. "
        "Converts to encoder grid dims via patch_size=16. Output goes to "
        "compiled/<precision>-<HxW>/. Default: use training resolution from config.json.",
    )
    parser.add_argument(
        "--weights-filename",
        type=str,
        default="world_model.pth.tar",
        help="Name of the world-model weights file inside the checkpoint dir. "
        "Default: world_model.pth.tar",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompilation even if .pt2 files already exist.",
    )
    parser.add_argument(
        "--fp8",
        type=str,
        default=None,
        help="Path to pre-calibrated FP8 activation scales (.pt file). "
        "When provided, all nn.Linear layers are converted to static FP8 "
        "before compilation. Generate scales with static_fp8.calibrate_activation_scales().",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("compile_checkpoint")

    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isdir(checkpoint):
        logger.error(f"Checkpoint directory not found: {checkpoint}")
        sys.exit(1)

    config_path = os.path.join(checkpoint, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"config.json not found in {checkpoint}")
        sys.exit(1)

    weights_path = os.path.join(checkpoint, args.weights_filename)
    if not os.path.exists(weights_path):
        logger.error(f"{args.weights_filename} not found in {checkpoint}")
        sys.exit(1)

    # Parse config for display
    with open(config_path) as f:
        config = json.load(f)

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
        logger.info(f"  Resolution override: {pixel_h}x{pixel_w} -> grid {res_grid_h}x{res_grid_w}")

    # Build output dir name: bf16, mp-bf16, bf16-sdpa-cudnn, bf16-256x256, etc.
    attn_impl = args.attn_impl
    if attn_impl != "fmha":
        dir_suffix = f"{args.precision}-{attn_impl}"
    else:
        dir_suffix = args.precision
    if args.fp8:
        dir_suffix += "-fp8"
    if args.resolution:
        dir_suffix += f"-{args.resolution}"
    artifact_dir = args.output_dir or os.path.join(checkpoint, "compiled", dir_suffix)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    steps = [int(x) for x in args.steps.split(",")]
    num_gpus = args.num_gpus or 1
    precision = args.precision

    total_programs = len(batch_sizes) * len(steps)

    logger.info("=" * 80)
    logger.info("AOT Compilation for RoboJEPA World Model")
    logger.info("=" * 80)
    logger.info(f"  Checkpoint:    {checkpoint}")
    logger.info(f"  Output:        {artifact_dir}")
    logger.info(f"  num_views:     {config.get('num_views', 1)}")
    logger.info(f"  image grid:    {config.get('grid_height', '?')}x{config.get('grid_width', '?')}")
    logger.info(f"  predictor_dim: {config.get('predictor_embed_dim', '?')}")
    logger.info(f"  depth:         {config.get('depth', '?')}")
    logger.info(f"  action_dim:    {config.get('action_dim', '?')}")
    logger.info(f"  Batch sizes:   {batch_sizes}")
    logger.info(f"  Steps:         {steps}")
    logger.info(f"  Total:         {total_programs} programs")
    logger.info(f"  GPUs:          {num_gpus} ({'parallel' if num_gpus > 1 else 'sequential'})")
    logger.info(f"  Precision:     {precision}")
    logger.info(f"  Attn impl:     {attn_impl}")
    logger.info(f"  Force:         {args.force}")
    if torch.cuda.is_available():
        logger.info(f"  GPU:           {torch.cuda.get_device_name()}")
    logger.info(f"  PyTorch:       {torch.__version__}")
    logger.info("=" * 80)

    # Ensure CUDA headers are available
    conda_prefix = os.environ.get("CONDA_PREFIX", os.path.dirname(os.path.dirname(sys.executable)))
    targets_include = os.path.join(conda_prefix, "targets", "x86_64-linux", "include")
    if os.path.isdir(targets_include):
        existing = os.environ.get("CPLUS_INCLUDE_PATH", "")
        if targets_include not in existing:
            os.environ["CPLUS_INCLUDE_PATH"] = f"{targets_include}:{existing}"
    if "CUDA_HOME" not in os.environ and os.path.isdir(conda_prefix):
        os.environ["CUDA_HOME"] = conda_prefix

    # Import after env setup
    from app.robo_jepa.inference.dispatcher import AOTWorldModelDispatcher
    from app.robo_jepa.world_model import WorldModel

    # Load model
    logger.info("Loading model...")
    t0 = time.perf_counter()
    wm = WorldModel.load_pretrained(
        checkpoint, device="cuda:0", load_decoder=False, weights_filename=args.weights_filename
    )
    wm.eval()
    logger.info(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    # Create dispatcher (with optional resolution override)
    dispatcher = AOTWorldModelDispatcher(
        wm,
        artifact_dir=artifact_dir,
        device="cuda:0",
        precision=precision,
        attn_impl=attn_impl,
        grid_height=res_grid_h,
        grid_width=res_grid_w,
    )
    logger.info(f"  Cache len:     {dispatcher._cache_len}")
    logger.info(f"  Frame tokens:  {dispatcher._frame_tokens}")

    # Compile
    t_start = time.perf_counter()
    if num_gpus > 1:
        # Free model from GPU — workers load their own copy
        del wm
        torch.cuda.empty_cache()

        logger.info(f"Starting parallel compilation on {num_gpus} GPUs...")
        dispatcher.compile_parallel(
            model_dir=checkpoint,
            batch_sizes=batch_sizes,
            steps=steps,
            num_gpus=num_gpus,
            force=args.force,
            precision=precision,
            attn_impl=attn_impl,
            fp8_scales_path=args.fp8,
            weights_filename=args.weights_filename,
        )

        # Reload model for saving manifest
        dispatcher._wm = WorldModel.load_pretrained(
            checkpoint, device="cuda:0", load_decoder=False, weights_filename=args.weights_filename
        )
        dispatcher._wm.eval()
    else:
        logger.info("Starting sequential compilation...")
        dispatcher.compile(
            batch_sizes=batch_sizes,
            steps=steps,
            force=args.force,
        )

    total_time = time.perf_counter() - t_start
    logger.info(f"Compilation done in {total_time:.0f}s ({total_time / 60:.1f} min)")

    # Save manifest (no weights — extracted from model at load time)
    dispatcher.save(save_weights=False)

    # Report
    n_files = 0
    total_size = 0
    for f in os.listdir(artifact_dir):
        fp = os.path.join(artifact_dir, f)
        if os.path.isfile(fp):
            n_files += 1
            total_size += os.path.getsize(fp)

    logger.info("=" * 80)
    logger.info("Done!")
    logger.info(f"  Programs:  {n_files - 1} .pt2 files + manifest.json")
    logger.info(f"  Size:      {total_size / (1024 * 1024):.0f} MB")
    logger.info(f"  Location:  {artifact_dir}")
    logger.info("")
    logger.info("To use in code:")
    logger.info(f'  wm = WorldModel.load_pretrained("{checkpoint}", load_compiled=True)')
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
