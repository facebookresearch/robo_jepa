#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark autoregressive (AR) rollout speed of a RoboJEPA world model.

Runs N autoregressive ``world_model.step`` calls in EAGER mode and in
COMPILED (AOT ``.pt2``) mode and reports per-step / total latency plus the
eager→compiled speedup. Mirrors the ``wm.step(action, state, use_cache=True)``
RNN-style loop used by ``scripts/deployment/agents/world_model/policy_client.py``.

Only the world-model transition model (the predictor) is benchmarked. The
frozen V-JEPA encoder is NOT loaded or timed — the AR loop is seeded with
synthetic random encoder features of the correct shape (auto-detected from
``config.json``), which is exactly the input the WM consumes at each step.

Usage (run inside a GPU job, e.g. via ``srun --overlap --jobid=<JID>``):

    python -m app.robo_jepa.inference.benchmark_ar_steps \
        /path/to/wm-8b \
        --weights-filename cooldown_8b_200k.pth.tar \
        --num-steps 5 --batch-size 1

The checkpoint dir must contain ``config.json`` + the weights file and a
``compiled/<precision>/`` dir produced by ``compile_checkpoint`` (for the
compiled mode). Compiled programs only exist for the batch sizes / steps that
were compiled (this 8B checkpoint was compiled at batch size 1, steps 0-8).
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time

import torch

logger = logging.getLogger("benchmark_ar_steps")


def build_inputs(cfg, batch_size, num_steps, seed, device):
    """Synthesize a fixed (seeded) AR rollout: seed feature + per-step actions.

    Returns the pieces needed to (re)build a fresh WorldModelState plus the
    action sequence. Generated on CPU in fp32 so eager (autocast) and compiled
    (internal bf16 cast) paths receive identical numerical inputs.
    """
    g = torch.Generator().manual_seed(seed)
    V = cfg["num_views"]
    H = cfg["grid_height"]
    W = cfg["grid_width"]
    C = cfg["embed_dim"]

    features = torch.randn(batch_size, 1, V, H, W, C, generator=g)
    # 7-D planner-side pose / action (DROID layout); embodiment preprocessors
    # inside step() expand these to the model's state_dim / action_dim.
    states = torch.zeros(batch_size, 1, 7)
    actions = 0.1 * torch.randn(batch_size, num_steps, 7, generator=g)
    # camera identities per view slot (valid indices into the model's view_map)
    view_map = torch.arange(V, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    # embodiment 0 == "franka" (index into config embodiment_list)
    embodiment_ids = torch.zeros(batch_size, 1, dtype=torch.long)

    return dict(
        features=features.to(device),
        states=states.to(device),
        actions=actions.to(device),
        view_map=view_map.to(device),
        embodiment_ids=embodiment_ids.to(device),
    )


def make_state(inputs):
    from app.robo_jepa.world_model import WorldModelState

    return WorldModelState.from_features(
        features=inputs["features"].clone(),
        states=inputs["states"].clone(),
        view_map=inputs["view_map"].clone(),
        embodiment_ids=inputs["embodiment_ids"].clone(),
    )


def run_rollout(model, inputs, num_steps, collect_outputs=False):
    """Run one AR rollout, timing each step. Returns (per_step_seconds, outputs)."""
    state = make_state(inputs)
    actions = inputs["actions"]
    per_step = []
    outputs = [] if collect_outputs else None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        for k in range(num_steps):
            action = actions[:, k : k + 1]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, state = model.step(action, state, use_cache=True, no_grad=True)
            torch.cuda.synchronize()
            per_step.append(time.perf_counter() - t0)
            if collect_outputs:
                outputs.append(out[:, -1:].detach().float().cpu())

    return per_step, outputs


def benchmark_mode(model, inputs, num_steps, warmup, trials):
    """Warm up then time ``trials`` rollouts. Returns per-step stats (ms)."""
    for _ in range(warmup):
        run_rollout(model, inputs, num_steps)

    all_trials = []  # list[ list[per-step seconds] ]
    for _ in range(trials):
        per_step, _ = run_rollout(model, inputs, num_steps)
        all_trials.append(per_step)

    # mean per step across trials (ms)
    per_step_ms = [
        1000.0 * statistics.mean(all_trials[t][k] for t in range(trials)) for k in range(num_steps)
    ]
    totals_ms = [1000.0 * sum(all_trials[t]) for t in range(trials)]
    return {
        "per_step_ms": per_step_ms,
        "total_mean_ms": statistics.mean(totals_ms),
        "total_std_ms": statistics.pstdev(totals_ms) if trials > 1 else 0.0,
    }


def fmt_table(num_steps, eager, compiled):
    lines = []
    header = f"{'step':>6} | {'eager ms':>12} | {'compiled ms':>14} | {'speedup':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for k in range(num_steps):
        e = eager["per_step_ms"][k] if eager else float("nan")
        c = compiled["per_step_ms"][k] if compiled else float("nan")
        sp = (e / c) if (eager and compiled and c > 0) else float("nan")
        lines.append(f"{k:>6} | {e:>12.2f} | {c:>14.2f} | {sp:>8.2f}x")
    lines.append("-" * len(header))
    if eager and compiled:
        e_tot = eager["total_mean_ms"]
        c_tot = compiled["total_mean_ms"]
        sp = e_tot / c_tot if c_tot > 0 else float("nan")
        lines.append(
            f"{'TOTAL':>6} | {e_tot:>10.2f}±{eager['total_std_ms']:.1f} | "
            f"{c_tot:>12.2f}±{compiled['total_std_ms']:.1f} | {sp:>8.2f}x"
        )
    elif eager:
        lines.append(f"{'TOTAL':>6} | {eager['total_mean_ms']:>12.2f} | {'-':>14} | {'-':>9}")
    elif compiled:
        lines.append(f"{'TOTAL':>6} | {'-':>12} | {compiled['total_mean_ms']:>14.2f} | {'-':>9}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark eager vs compiled AR rollout speed of a RoboJEPA WM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", help="Path to the model checkpoint directory.")
    parser.add_argument(
        "--weights-filename",
        default="world_model.pth.tar",
        help="Weights file inside the checkpoint dir. Default: world_model.pth.tar",
    )
    parser.add_argument("--num-steps", type=int, default=5, help="AR steps per rollout. Default: 5")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Default: 1")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup rollouts. Default: 3")
    parser.add_argument("--trials", type=int, default=10, help="Timed rollouts. Default: 10")
    parser.add_argument(
        "--precision",
        default="bf16",
        help="Compiled artifact precision dir suffix (compiled/<precision>/). Default: bf16",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["eager", "compiled", "both"],
        help="Which path(s) to benchmark. Default: both",
    )
    parser.add_argument("--device", default="cuda", help="Device. Default: cuda")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for inputs. Default: 0")
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="When mode=both, compare eager vs compiled step outputs on identical inputs.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    if not torch.cuda.is_available():
        logger.error("CUDA is required for this benchmark.")
        sys.exit(1)

    checkpoint = os.path.abspath(args.checkpoint)
    config_path = os.path.join(checkpoint, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"config.json not found in {checkpoint}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)

    # Ensure CUDA headers/home are set for any lazy compile fallback paths.
    conda_prefix = os.environ.get("CONDA_PREFIX", os.path.dirname(os.path.dirname(sys.executable)))
    if "CUDA_HOME" not in os.environ and os.path.isdir(conda_prefix):
        os.environ["CUDA_HOME"] = conda_prefix

    from app.robo_jepa.world_model import WorldModel

    logger.info("=" * 78)
    logger.info("RoboJEPA AR rollout benchmark")
    logger.info("=" * 78)
    logger.info(f"  Checkpoint:   {checkpoint}")
    logger.info(f"  Weights:      {args.weights_filename}")
    logger.info(f"  num_views:    {cfg.get('num_views')}  grid: {cfg.get('grid_height')}x{cfg.get('grid_width')}")
    logger.info(f"  embed_dim:    {cfg.get('embed_dim')}  predictor_dim: {cfg.get('predictor_embed_dim')}  depth: {cfg.get('depth')}")
    logger.info(f"  AR steps:     {args.num_steps}   batch_size: {args.batch_size}")
    logger.info(f"  warmup/trials:{args.warmup}/{args.trials}   precision: {args.precision}   mode: {args.mode}")
    logger.info(f"  GPU:          {torch.cuda.get_device_name()}   torch {torch.__version__}")
    logger.info("=" * 78)

    inputs = build_inputs(cfg, args.batch_size, args.num_steps, args.seed, args.device)

    eager_stats = compiled_stats = None
    eager_outputs = compiled_outputs = None

    if args.mode in ("eager", "both"):
        logger.info("Loading EAGER model...")
        t0 = time.perf_counter()
        wm = WorldModel.load_pretrained(
            checkpoint,
            device=args.device,
            load_decoder=False,
            load_encoder=False,
            weights_filename=args.weights_filename,
        )
        wm.eval()
        logger.info(f"  loaded in {time.perf_counter() - t0:.1f}s")
        if args.check_correctness:
            _, eager_outputs = run_rollout(wm, inputs, args.num_steps, collect_outputs=True)
        logger.info("Benchmarking EAGER...")
        eager_stats = benchmark_mode(wm, inputs, args.num_steps, args.warmup, args.trials)
        del wm
        torch.cuda.empty_cache()

    if args.mode in ("compiled", "both"):
        logger.info("Loading COMPILED dispatcher...")
        t0 = time.perf_counter()
        disp = WorldModel.load_pretrained(
            checkpoint,
            device=args.device,
            load_decoder=False,
            load_encoder=False,
            load_compiled=args.precision,
            compiled_max_steps=args.num_steps,
            weights_filename=args.weights_filename,
        )
        logger.info(f"  loaded in {time.perf_counter() - t0:.1f}s "
                    f"({len(getattr(disp, 'compiled_variants', []))} variants)")
        if args.check_correctness:
            _, compiled_outputs = run_rollout(disp, inputs, args.num_steps, collect_outputs=True)
        logger.info("Benchmarking COMPILED...")
        compiled_stats = benchmark_mode(disp, inputs, args.num_steps, args.warmup, args.trials)
        del disp
        torch.cuda.empty_cache()

    logger.info("")
    logger.info("Per-step latency (mean over %d trials):", args.trials)
    print("\n" + fmt_table(args.num_steps, eager_stats, compiled_stats) + "\n")

    if eager_stats and compiled_stats:
        sp = eager_stats["total_mean_ms"] / compiled_stats["total_mean_ms"]
        logger.info(
            f"SUMMARY: {args.num_steps}-step rollout  eager={eager_stats['total_mean_ms']:.1f} ms  "
            f"compiled={compiled_stats['total_mean_ms']:.1f} ms  speedup={sp:.2f}x"
        )

    if args.check_correctness and eager_outputs and compiled_outputs:
        logger.info("Correctness (eager vs compiled, identical seeded inputs):")
        for k in range(args.num_steps):
            e = eager_outputs[k]
            c = compiled_outputs[k]
            max_abs = (e - c).abs().max().item()
            mean_abs = (e - c).abs().mean().item()
            denom = e.abs().mean().item() + 1e-8
            logger.info(f"  step {k}: max_abs_diff={max_abs:.4e}  mean_abs_diff={mean_abs:.4e}  "
                        f"rel={mean_abs / denom:.4e}")


if __name__ == "__main__":
    main()
