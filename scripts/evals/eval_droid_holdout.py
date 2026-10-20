# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DROID holdout forward-prediction eval (per-step rollout L1) for the checkpoints of one model size.

Reproduces the training-time `eval_data/DROID_objectstore/rollout/l1_loss/<step>` metric of the scaling runs: 9-step
latent rollouts on the DROID holdout (REAL lab, building == "setup room", which the scaling runs exclude from
training), L1 against the frozen-encoder features per step, averaged over all rollout prefixes of each clip.

  * the same --n-clips holdout clips (fixed seed) are encoded once and reused for every checkpoint, so checkpoints
    and sizes are compared on identical inputs.
  * attn_impl is forced to sdpa (fmha does not support the local_attn_window mask).
  * clips are batched by their number of valid frames, which leaves the per-clip average unchanged.
  * --kv-cache rolls out through the predictor's KV cache (WorldModel.step style) instead of full re-forwards.
  * one JSON per checkpoint in --out-dir; checkpoints with a JSON are skipped on rerun.

    python scripts/evals/eval_droid_holdout.py --size 50m --models-dir /data/robojepa --data /data/droid \\
        --encoder /data/vjepa2_1_vitG_384.pt --out-dir results/wm-50m
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.robo_jepa.datasets.robot_video_dataset import RobotVideoDataset
from app.robo_jepa.transforms import make_transforms
from app.robo_jepa.world_model import WorldModel

DTYPE = torch.bfloat16

# data.eval.DROID_objectstore (+ data.eval.kwargs) of the scaling configs, with DROID's real 15 fps
# (the configs' 60 / 5 / 30 assumed 60 fps videos; 15 / 1.25 keeps the same 12-frame step)
EMBODIMENT = "franka"
RAW_FPS, FPS, INTERMEDIATE_FRAMES_FPS = 15, 1.25, 7.5
POSE_DIM, GRIPPER_DIM, JOINTS_DIM = 6, 1, 7
VIEW_MAP = {"left": 0, "right": 1, "wrist": 2}
CAMERAS = ["left|right", "wrist"]
PAD_SIZE, FRAMES_PER_CLIP, MIN_LENGTH_TO_DROP, MIN_SIZE = 64, 16, 16, 1024
EVAL_ROLLOUT_STEPS = 10  # steps 1..9 are reported (step 10 has no target with predictor_window_steps=1)
HOLDOUT_LAB, HOLDOUT_BUILDING = "REAL", "setup room"
FILTER_SAMPLES_FUNC = "app.robo_jepa.datasets.droid.filter_positive_negative_droid"
FILTER_SAMPLES_FUNC_KWARGS = {"filter": {"labs": [HOLDOUT_LAB], "buildings": [HOLDOUT_BUILDING]}}
EVAL_AUGS = {
    "random_horizontal_flip": False, "random_resize_aspect_ratio": [1.333, 1.777], "random_resize_scale": [1.0, 1.0],
    "reprob": 0.0, "auto_augment": False, "motion_shift": False, "crop_height": 240, "crop_width": 320,
}


def checkpoints(models_dir, size):
    paths = glob.glob(os.path.join(models_dir, f"wm-{size}", f"cooldown_{size}_*k.pth.tar"))
    return sorted(paths, key=lambda p: int(re.search(r"_(\d+)k\.pth\.tar$", p).group(1)))


def load_dir(models_dir, size, ckpt, encoder):
    """Temp dir WorldModel.load_pretrained can read: config.json (attn_impl -> sdpa), the checkpoint, the encoder."""
    tmp = tempfile.mkdtemp(prefix=f"wm_{size}_")
    with open(os.path.join(models_dir, f"wm-{size}", "config.json")) as f:
        config = json.load(f)
    config["attn_impl"] = "sdpa"
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    os.symlink(os.path.abspath(ckpt), os.path.join(tmp, "world_model.pth.tar"))
    os.symlink(os.path.abspath(encoder), os.path.join(tmp, "encoder.pth.tar"))
    return tmp


def ensure_metadata(data):
    """The lab/building filter reads <data>/aggregated_metadata.json ({episode: DROID metadata}). convert_droid.py
    keeps each episode's metadata_*.json; add those the file lacks. All episodes, not only the holdout lab: training
    on the same root reads this file for its negative filter and drops episodes without metadata. Existing entries
    are kept, so a file written for the holdout lab alone is completed rather than trusted."""
    path = os.path.join(data, "aggregated_metadata.json")
    metadata = {}
    if os.path.exists(path):
        with open(path) as f:
            metadata = json.load(f)
    index = pd.read_csv(os.path.join(data, "dataset_index.csv"))
    before = len(metadata)
    # convert_droid.py writes only `path`; filter_droid falls back to it as the key
    for episode in index["metadata" if "metadata" in index else "path"]:
        if not isinstance(episode, str) or episode in metadata:
            continue
        for f in glob.glob(os.path.join(data, episode, "metadata_*.json")):
            with open(f) as fh:
                metadata[episode] = json.load(fh)
    added = len(metadata) - before
    if not added:
        return
    try:
        # write then rename, so an interrupted or concurrent run never leaves a truncated file
        fd, tmp = tempfile.mkstemp(dir=data, prefix=".aggregated_metadata.", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(metadata, f)
        os.replace(tmp, path)
    except OSError as e:
        print(f"could not add {added} episodes to {path} ({e}); the building filter may drop them", flush=True)
        return
    print(f"added {added} episodes to {path} ({len(metadata)} total)", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_clips(encoder, data, n_clips, seed, device):
    """The same n_clips holdout clips every time (fixed seeds), with their frozen-encoder features."""
    dataset = RobotVideoDataset(
        dataset_name="DROID_objectstore", data_path=None, cameras=CAMERAS, view_map=VIEW_MAP,
        embodiment_name=EMBODIMENT, raw_fps=RAW_FPS, fps=FPS, pose_dim=POSE_DIM, gripper_dim=GRIPPER_DIM,
        joints_dim=JOINTS_DIM, intermediate_frames_fps=INTERMEDIATE_FRAMES_FPS, s3_profile=None,
        s3_prefix=data.rstrip("/") + "/", pad_size=PAD_SIZE, frames_per_clip=FRAMES_PER_CLIP,
        min_length_to_drop=MIN_LENGTH_TO_DROP, min_size=MIN_SIZE, transform=make_transforms(**EVAL_AUGS),
        read_intermediate_frames=False, action_source="state_delta",
        filter_samples_func=FILTER_SAMPLES_FUNC, filter_samples_func_kwargs=FILTER_SAMPLES_FUNC_KWARGS,
    )
    n_episodes = len(dataset.samples)
    print(f"holdout: {n_episodes} episodes (dataset length {len(dataset)} with min_size={MIN_SIZE})", flush=True)
    seed_all(seed)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n_clips].tolist()
    clips, t0 = [], time.time()
    for i, idx in enumerate(order):
        (rgb_video, _, action, state, extrinsics, padding_mask, view_list, dt, start, traj_names, emb_names,
         index) = dataset[idx]
        video = rgb_video.unsqueeze(0).to(device, dtype=DTYPE)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            b, _, v, t = video.shape[:4]
            h_, w_ = (s // encoder.vit.patch_size for s in video.shape[-2:])
            flat = einops.repeat(einops.rearrange(video, "b c v t h w -> (b t v) c h w"), "b c h w -> b c t h w", t=1)
            feats = einops.rearrange(encoder(flat.float()), "(b t v) (h w) c -> b t v h w c", b=b, t=t, v=v, h=h_,
                                     w=w_)
        clips.append({
            "features": feats.to(DTYPE).cpu(), "action": action.unsqueeze(0), "state": state.unsqueeze(0),
            "extrinsics": extrinsics.unsqueeze(0), "padding_mask": padding_mask.unsqueeze(0),
            "view_list": view_list.unsqueeze(0), "dt": dt.unsqueeze(0),
            "embodiment": [emb_names] if isinstance(emb_names, str) else list(emb_names),
            "id": f"{traj_names}@{int(start)}",
        })
        if (i + 1) % 128 == 0:
            print(f"  encoded {i + 1}/{len(order)} clips ({time.time() - t0:.0f}s)", flush=True)
    return clips, n_episodes


def embodiment_id(world_model, name):
    emb_list = list(world_model.model.embodiment_list)
    for cand in (name, name + "_state_delta", name.replace("_state_delta", "")):
        if cand in emb_list:
            return emb_list.index(cand)
    raise ValueError(f"embodiment {name!r} not in {emb_list}")


def batches(clips, batch_size):
    """Clips grouped by their number of valid frames, then concatenated in chunks of batch_size. Within a batch all
    clips have the same valid length, so batch-mean losses weighted by batch size equal the per-clip average."""
    groups = defaultdict(list)
    for clip in clips:
        groups[int(clip["padding_mask"].sum())].append(clip)
    for _, group in sorted(groups.items()):
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            out = {k: torch.cat([c[k] for c in chunk]) for k in ("features", "action", "state", "extrinsics",
                                                                   "padding_mask", "view_list", "dt")}
            out["embodiment"] = [e for c in chunk for e in c["embodiment"]]
            yield out


def cached_rollout(world_model, history, action, state, dt, video_f, padding_mask, t, view_list, embodiment_ids):
    """WorldModel.rollout(...) for one prefix, but stepping through the predictor's KV cache like WorldModel.step:
    one forward over the history fills the cache, then each step feeds only the newest (predicted) frame with its
    action, pose and dt. Same inputs per frame, same targets, masks and loss as the uncached rollout."""
    actions_suffix, states_suffix, dt_suffix = action[:, t + 1:], state[:, t + 2:], dt[:, t + 2:]
    features_suffix, mask_suffix = video_f[:, t + 2:], padding_mask[:, t + 1:-1]
    steps = min(EVAL_ROLLOUT_STEPS - 1, actions_suffix.shape[1])
    if features_suffix.shape[1] < EVAL_ROLLOUT_STEPS:
        return {}
    losses_per_step = defaultdict(list)
    cache, position, pred = None, 0, None
    for h in range(steps):
        if h == 0:  # prefill: encoder history + the teacher-forced next step, as select_history builds it
            inp = (history.features, torch.cat([history.actions, actions_suffix[:, :1]], dim=1), history.states,
                   history.dt)
            cache_size = 0
        else:  # the frame predicted at the previous step, with its dataset action / pose / dt
            inp = (pred, actions_suffix[:, h:h + 1], states_suffix[:, h - 1:h], dt_suffix[:, h - 1:h])
            # frames held in the cache: the pointer keeps counting past cache_len, but the cache rolls at cache_len
            # (= attn_depth, the local attention window), so cap it to attend over the same window as the mask
            cache_size = min(cache.pointer[0], cache.cache_len)
        out, _, next_poses, cache = world_model.forward(
            inp[0], inp[1], states=inp[2], camera_extrinsics=None, view_map=view_list,
            embodiment_ids=embodiment_ids, dt=inp[3], use_cache=True, cache=cache, cache_size=cache_size,
            input_position=position)
        position += inp[0].shape[1]
        pred = out[:, -1:]
        losses = world_model.compute_loss(pred, features_suffix[:, h:h + 1], None, next_poses,
                                          mask_suffix[:, h:h + 1], shift=0, view_map=view_list,
                                          apply_view_loss_weights=False)
        for k, v in losses.items():
            losses_per_step[k].append(v.detach() if hasattr(v, "detach") else torch.tensor(v))
    return {k: torch.stack(v) for k, v in losses_per_step.items()}


def eval_checkpoint(world_model, clips, device, batch_size=1, kv_cache=False):
    sums, cnts = defaultdict(float), defaultdict(int)
    for clip in batches(clips, batch_size):
        n = clip["features"].shape[0]
        to = lambda x: x.to(device, dtype=DTYPE)  # noqa: E731
        action, state, extrinsics, dt = to(clip["action"]), to(clip["state"]), to(clip["extrinsics"]), to(clip["dt"])
        padding_mask, view_list = to(clip["padding_mask"]), clip["view_list"].to(device)
        embodiment_ids = None
        if world_model.model.is_multiembodiment:
            embodiment_ids = torch.tensor([embodiment_id(world_model, e) for e in clip["embodiment"]], device=device)
        video_f = clip["features"].to(device)
        if world_model.encoder_layernorm:
            video_f = F.layer_norm(video_f, (video_f.size(-1),))
        if video_f.shape[1] == action.shape[1] + 1:
            video_f, padding_mask = video_f[:, :-1], padding_mask[:, :-1]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            predicted, _, predicted_states, _ = world_model.forward(
                video_f, action, state, extrinsics, view_map=view_list, dt=dt, embodiment_ids=embodiment_ids)
            losses = world_model.compute_loss(predicted, video_f, state=state, pred_state=predicted_states,
                                              padding_mask=padding_mask, shift=1, view_map=view_list,
                                              apply_view_loss_weights=False)
            for k, v in losses.items():
                sums[f"teacher_forcing/{k}"] += float(v) * n
                cnts[f"teacher_forcing/{k}"] += n
            total_steps = int(padding_mask.sum(1).min().item())
            stats = defaultdict(list)
            for t in range(total_steps + 1 - EVAL_ROLLOUT_STEPS):  # suffix_len = predictor_window_steps = 1
                wm_state = world_model.select_history(
                    video_f, action, predicted, states=state, predicted_states=predicted_states,
                    camera_extrinsics=extrinsics, dt=dt, prefix_len=t + 1, suffix_len=1)
                if kv_cache:
                    rollout_losses = cached_rollout(world_model, wm_state, action, state, dt, video_f, padding_mask, t,
                                                    view_list, embodiment_ids)
                    for k, v in rollout_losses.items():
                        stats[k].append(v)
                    continue
                features_suffix = video_f[:, t + 2:]
                rollout_losses, _ = world_model.rollout(
                    wm_state, actions_suffix=action[:, t + 1:],
                    features_suffix=features_suffix if features_suffix.shape[1] >= EVAL_ROLLOUT_STEPS else None,
                    states_suffix=state[:, t + 2:], dt=dt[:, t + 2:], action_noise=0.0, suffix_len=1,
                    run_backward=False, padding_mask=padding_mask[:, t + 1:-1], rollout_steps=EVAL_ROLLOUT_STEPS - 1,
                    view_map=view_list, embodiment_ids=embodiment_ids)
                for k, v in rollout_losses.items():
                    stats[k].append(v)
            for k, v in stats.items():
                for j, x in enumerate(torch.stack(v).mean(0).tolist()):
                    sums[f"rollout/{k}/{j + 1}"] += x * n
                    cnts[f"rollout/{k}/{j + 1}"] += n
    return {k: sums[k] / cnts[k] for k in sorted(sums)}, dict(cnts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", required=True, help="model size, e.g. 22m, 50m, 1b")
    parser.add_argument("--models-dir", required=True,
                        help="dir with wm-<size>/{config.json,cooldown_<size>_<step>k.pth.tar}, as downloaded")
    parser.add_argument("--data", required=True, help="DROID root from scripts/data_processing/convert_droid.py")
    parser.add_argument("--encoder", required=True, help="V-JEPA 2.1 ViT-G/384 weights (vjepa2_1_vitG_384.pt)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-clips", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16, help="clips per forward (same valid length)")
    parser.add_argument("--kv-cache", action="store_true", help="roll out through the KV cache (WorldModel.step)")
    parser.add_argument("--steps", default=None, help="training steps in k, e.g. 100,150 (default: all)")
    args = parser.parse_args()
    device = "cuda"
    torch.cuda.set_device(0)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpts = checkpoints(args.models_dir, args.size)
    if args.steps:
        keep = {f"_{s}k.pth.tar" for s in args.steps.split(",")}
        ckpts = [c for c in ckpts if any(c.endswith(k) for k in keep)]
    if not ckpts:
        pattern = os.path.join(args.models_dir, f"wm-{args.size}", f"cooldown_{args.size}_*k.pth.tar")
        sys.exit(f"no checkpoints match {pattern}" + (f" with --steps {args.steps}" if args.steps else ""))
    todo = [c for c in ckpts if not os.path.exists(os.path.join(args.out_dir, f"{os.path.basename(c)[:-8]}.json"))]
    print(f"wm-{args.size}: {len(ckpts)} checkpoints, {len(todo)} to evaluate", flush=True)
    if not todo:
        return
    ensure_metadata(args.data)

    encoder = None
    clips = None
    for ckpt in todo:
        name = os.path.basename(ckpt)[: -len(".pth.tar")]
        t0 = time.time()
        tmp = load_dir(args.models_dir, args.size, ckpt, args.encoder)
        try:
            world_model = WorldModel.load_pretrained(tmp, load_decoder=False, load_encoder=encoder is None,
                                                     device=device)
        finally:
            shutil.rmtree(tmp)
        if encoder is None:
            encoder = world_model.encoder
            clips, n_episodes = encode_clips(encoder, args.data, args.n_clips, args.seed, device)
        else:
            world_model.encoder = encoder
        world_model.eval()
        world_model.predictor_window_steps = 1
        world_model.eval_rollout_steps = EVAL_ROLLOUT_STEPS
        world_model.rollout_steps = 2
        world_model.rollout_stop_gradient = True
        seed_all(args.seed)
        metrics, counts = eval_checkpoint(world_model, clips, device, args.batch_size, args.kv_cache)
        per_step = [metrics.get(f"rollout/l1_loss/{i}", float("nan")) for i in range(1, EVAL_ROLLOUT_STEPS)]
        result = {
            "size": args.size, "checkpoint": ckpt, "step_k": int(re.search(r"_(\d+)k$", name).group(1)),
            "dataset": f"DROID holdout ({HOLDOUT_LAB} lab, {HOLDOUT_BUILDING})", "holdout_episodes": n_episodes,
            "n_clips": len(clips), "seed": args.seed, "batch_size": args.batch_size, "kv_cache": args.kv_cache,
            "eval_rollout_steps": EVAL_ROLLOUT_STEPS, "rollout_l1_per_step": per_step,
            "rollout_l1_mean_1_9": float(np.mean(per_step)), "step9_rollout_l1": per_step[8],
            "metrics": metrics, "counts": counts,
            "clip_ids_sha": hashlib.sha256("\n".join(c["id"] for c in clips).encode()).hexdigest(),
            "seconds": time.time() - t0,
        }
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"{name}: step-9 L1 {per_step[8]:.4f}, mean 1-9 {result['rollout_l1_mean_1_9']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del world_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
