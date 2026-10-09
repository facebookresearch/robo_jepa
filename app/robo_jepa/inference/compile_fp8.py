#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-warm FP8 torch.compile Inductor cache for the WM and/or decoder.

Calibrates FP8 scales, converts nn.Linear → StaticFloat8Linear, then runs
torch.compile for every (step × batch_size) variant. The compiled .so files
are saved to TORCHINDUCTOR_CACHE_DIR for instant loading on subsequent runs.

Usage:
    # WM only:
    python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint --component wm

    # Decoder only:
    python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint --component decoder

    # Both:
    python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint --component both

    # Custom cache dir + batch sizes:
    TORCHINDUCTOR_CACHE_DIR=/path/to/cache \\
    python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint \\
        --batch-sizes 1,4 --cache-dir /path/to/inductor_cache

    # Save calibrated FP8 scales for reuse:
    python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint \\
        --save-scales /path/to/checkpoint/fp8_scales.pt

    # Load pre-calibrated scales (skip calibration):
    python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint \\
        --load-scales /path/to/checkpoint/fp8_scales.pt
"""

import argparse
import gc
import logging
import os
import sys
import time

import torch

logger = logging.getLogger("compile_fp8")


def _ensure_cuda_env():
    conda_prefix = os.environ.get(
        "CONDA_PREFIX", os.path.dirname(os.path.dirname(sys.executable))
    )
    targets_include = os.path.join(conda_prefix, "targets", "x86_64-linux", "include")
    if os.path.isdir(targets_include):
        existing = os.environ.get("CPLUS_INCLUDE_PATH", "")
        if targets_include not in existing:
            os.environ["CPLUS_INCLUDE_PATH"] = f"{targets_include}:{existing}"
    if "CUDA_HOME" not in os.environ and os.path.isdir(conda_prefix):
        os.environ["CUDA_HOME"] = conda_prefix


def calibrate_wm(wm, model_config, cache_len, device, dtype):
    """Calibrate FP8 for the WM transition model."""
    from app.robo_jepa.inference.aot_compile import (
        FunctionalStepModule,
        build_example_inputs,
    )
    from app.robo_jepa.inference.static_fp8 import calibrate_and_quantize

    def calibration_fn(model):
        for step in range(min(cache_len + 1, 9)):
            inputs = build_example_inputs(
                model_config,
                step,
                batch_size=1,
                cache_len=cache_len,
                device=device,
            )
            inputs = tuple(t.to(dtype) if t.is_floating_point() else t for t in inputs)
            module = FunctionalStepModule(
                model,
                cache_size=min(step, cache_len),
                input_position=step,
            )
            module.eval()
            with torch.no_grad():
                module(*inputs)
            del module

    model, n = calibrate_and_quantize(
        wm.transition_model,
        calibration_fn,
        output_dtype=dtype,
    )
    wm.transition_model = model
    return n


def calibrate_decoder(wm, decoder_config, device, dtype):
    """Calibrate FP8 for the decoder DiT."""
    from app.robo_jepa.inference.decoder_aot_compile import (
        DecoderStepModule,
        build_decoder_example_inputs,
    )
    from app.robo_jepa.inference.static_fp8 import calibrate_and_quantize

    dit = wm.heads["image_head"].ema_model

    def calibration_fn(model):
        for cs in [0, 2, 4]:
            for uc in [False, True]:
                inputs = build_decoder_example_inputs(decoder_config, cs, 1, device)
                module = DecoderStepModule(model, cache_size=cs, update_cache=uc)
                module.eval()
                with (
                    torch.no_grad(),
                    torch.amp.autocast(device, dtype=dtype, enabled=True),
                ):
                    module(*inputs)
                del module

    dit, n = calibrate_and_quantize(dit, calibration_fn, output_dtype=dtype)
    wm.heads["image_head"].ema_model = dit
    return n


def pre_warm_wm(wm, model_config, batch_sizes, device, dtype, use_cudagraphs=False):
    """Pre-warm WM torch.compile cache."""
    from app.robo_jepa.inference.dispatcher import AOTWorldModelDispatcher

    dispatcher = AOTWorldModelDispatcher(
        wm,
        artifact_dir="",
        device=device,
        use_torch_compile=True,
        use_cudagraphs=use_cudagraphs,
    )
    dispatcher.pre_warm_compile_cache(
        batch_sizes=batch_sizes,
    )
    return dispatcher


def pre_warm_decoder(wm, decoder_config, batch_sizes, device, use_cudagraphs=False):
    """Pre-warm decoder torch.compile cache."""
    from app.robo_jepa.inference.decoder_aot_compile import (
        build_decoder_example_inputs,
        get_decoder_config,
    )
    from app.robo_jepa.inference.decoder_dispatcher import AOTDecoderDispatcher

    head = wm.heads["image_head"]
    dispatcher = AOTDecoderDispatcher(
        head=head,
        decoder_config=decoder_config,
        artifact_dir="",
        device=device,
        use_torch_compile=True,
        use_cudagraphs=use_cudagraphs,
    )

    cache_len = decoder_config["cache_len"]
    cache_steps = list(range(cache_len + 1))
    total = len(cache_steps) * 2 * len(batch_sizes)
    done = 0
    t_start = time.perf_counter()

    for cs in cache_steps:
        for uc in [False, True]:
            for B in batch_sizes:
                t0 = time.perf_counter()
                dispatcher._get_compiled_fn(cs, uc, B)
                elapsed = time.perf_counter() - t0
                done += 1
                logger.info(
                    f"  [{done}/{total}] step={cs}, update={uc}, B={B}: {elapsed:.1f}s"
                )

    total_time = time.perf_counter() - t_start
    logger.info(f"Pre-warmed {total} decoder programs in {total_time:.0f}s")
    return dispatcher


def main():
    parser = argparse.ArgumentParser(
        description="Pre-warm FP8 torch.compile Inductor cache for WM and/or decoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="both",
        choices=["wm", "decoder", "both"],
        help="Which component to compile. Default: both",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1",
        help="Comma-separated batch sizes. Default: 1",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Inductor cache directory. Default: uses TORCHINDUCTOR_CACHE_DIR env var.",
    )
    parser.add_argument(
        "--save-scales",
        type=str,
        default=None,
        help="Save calibrated FP8 activation scales to this path (.pt file).",
    )
    parser.add_argument(
        "--load-scales",
        type=str,
        default=None,
        help="Load pre-calibrated FP8 scales from this path (skip calibration).",
    )
    parser.add_argument(
        "--cudagraphs",
        action="store_true",
        help="Use CUDA graphs (mode='reduce-overhead') for lower kernel launch overhead.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    checkpoint = os.path.abspath(args.checkpoint)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    do_wm = args.component in ("wm", "both")
    do_decoder = args.component in ("decoder", "both")
    device = "cuda"
    dtype = torch.bfloat16

    if args.cache_dir:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = args.cache_dir
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "~/.cache/torch_inductor")

    logger.info("=" * 70)
    logger.info("FP8 torch.compile Pre-warming")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint:    {checkpoint}")
    logger.info(f"  Component:     {args.component}")
    logger.info(f"  Batch sizes:   {batch_sizes}")
    logger.info(f"  Cache dir:     {cache_dir}")
    logger.info(f"  GPU:           {torch.cuda.get_device_name()}")
    logger.info(f"  PyTorch:       {torch.__version__}")
    logger.info("=" * 70)

    _ensure_cuda_env()

    from app.robo_jepa.inference.aot_compile import get_model_config
    from app.robo_jepa.inference.decoder_aot_compile import get_decoder_config
    from app.robo_jepa.world_model import WorldModel

    # Load model
    logger.info("Loading model...")
    t0 = time.perf_counter()
    load_decoder = do_decoder
    wm = WorldModel.load_pretrained(
        checkpoint, device=device, load_decoder=load_decoder
    )
    wm.eval()
    logger.info(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    model_config = get_model_config(wm)
    cache_len = model_config["cache_len"]
    decoder_config = get_decoder_config(wm) if do_decoder else None

    # Calibrate FP8
    if args.load_scales:
        logger.info(f"Loading pre-calibrated FP8 scales from {args.load_scales}")
        from app.robo_jepa.inference.static_fp8 import apply_static_fp8

        scales = torch.load(args.load_scales, map_location=device)

        if do_wm:
            wm_scales = {k: v for k, v in scales.items() if k.startswith("wm.")}
            wm_scales = {k[3:]: v for k, v in wm_scales.items()}  # strip "wm." prefix
            n_wm = apply_static_fp8(wm.transition_model, wm_scales, output_dtype=dtype)
            logger.info(f"  WM: applied FP8 to {n_wm} layers from saved scales")

        if do_decoder:
            dec_scales = {k: v for k, v in scales.items() if k.startswith("dec.")}
            dec_scales = {
                k[4:]: v for k, v in dec_scales.items()
            }  # strip "dec." prefix
            dit = wm.heads["image_head"].ema_model
            n_dec = apply_static_fp8(dit, dec_scales, output_dtype=dtype)
            logger.info(f"  Decoder: applied FP8 to {n_dec} layers from saved scales")
    else:
        if do_wm:
            logger.info("Calibrating FP8 for WM transition model...")
            t0 = time.perf_counter()
            n_wm = calibrate_wm(wm, model_config, cache_len, device, dtype)
            logger.info(f"  WM: {n_wm} layers → FP8 in {time.perf_counter() - t0:.1f}s")

        if do_decoder:
            logger.info("Calibrating FP8 for decoder DiT...")
            t0 = time.perf_counter()
            n_dec = calibrate_decoder(wm, decoder_config, device, dtype)
            logger.info(
                f"  Decoder: {n_dec} layers → FP8 in {time.perf_counter() - t0:.1f}s"
            )

    # Save scales if requested
    if args.save_scales:
        from app.robo_jepa.inference.static_fp8 import StaticFloat8Linear

        all_scales = {}
        if do_wm:
            for name, mod in wm.transition_model.named_modules():
                if isinstance(mod, StaticFloat8Linear):
                    all_scales[f"wm.{name}"] = mod.activation_scale.cpu()
        if do_decoder:
            dit = wm.heads["image_head"].ema_model
            for name, mod in dit.named_modules():
                if isinstance(mod, StaticFloat8Linear):
                    all_scales[f"dec.{name}"] = mod.activation_scale.cpu()
        torch.save(all_scales, args.save_scales)
        logger.info(f"  Saved {len(all_scales)} FP8 scales to {args.save_scales}")

    # Pre-warm torch.compile
    if do_wm:
        logger.info(
            f"\nPre-warming WM torch.compile ({cache_len + 1} steps × {len(batch_sizes)} batches)..."
        )
        t0 = time.perf_counter()
        pre_warm_wm(
            wm, model_config, batch_sizes, device, dtype, use_cudagraphs=args.cudagraphs
        )
        logger.info(f"  WM pre-warm done in {time.perf_counter() - t0:.0f}s")

    if do_decoder:
        # Free WM from GPU if we're done with it
        if do_wm:
            wm.transition_model = wm.transition_model.cpu()
            if wm.encoder is not None:
                wm.encoder = wm.encoder.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        logger.info(
            f"\nPre-warming decoder torch.compile ({decoder_config['cache_len'] + 1} steps × 2 × {len(batch_sizes)} batches)..."
        )
        t0 = time.perf_counter()
        pre_warm_decoder(
            wm, decoder_config, batch_sizes, device, use_cudagraphs=args.cudagraphs
        )
        logger.info(f"  Decoder pre-warm done in {time.perf_counter() - t0:.0f}s")

    logger.info("\n" + "=" * 70)
    logger.info("Done! Inductor cache populated.")
    logger.info(f"  Cache dir: {cache_dir}")
    logger.info("  Subsequent runs with the same cache dir will skip compilation.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
