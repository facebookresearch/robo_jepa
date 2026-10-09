# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
import sys
import time

sys.path.insert(0, "../../../../")
sys.path.insert(0, "../")
import gc
import time
import json
import logging
from argparse import ArgumentParser
from collections import defaultdict, deque
from torchvision.transforms import functional as F
import mediapy
import cv2

from PIL import Image, ImageFont, ImageDraw

font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 25)
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})
import numpy as np
import numpy as np
from scipy.spatial.transform import Rotation

import numpy as np
import io
import lz4.frame
import torch
import torch.distributed as dist
from app.robo_jepa.world_model import WorldModel, WorldModelState, poses_to_diffs
from app.robo_jepa.mpc import mpc_utils
from app.robo_jepa.inference.mem_debug import log_rss, reset_cuda_peak
from distributed import init_distributed


def log_gpu_mem(tag):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"[GPU mem] {tag}: alloc={alloc:.2f}GB peak={peak:.2f}GB")


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from gym_tcp import GymEnvClient
from action_components import split_7d_action

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
# try:

# except Exception:
#     pass

view_name_map = {0: "left camera", 1: "wrist camera"}


def process(
    buffer,
    df,
    start_frame=100,
    episode=1,
    total_eps=3,
    goal="go up",
    color=(128, 64, 0),
    comment="",
    l=0.3,
    emb_dists=None,
    mpc_dists=None,
    execution_prefix=1,
):
    bottom_images = []
    for ts in range(buffer.shape[0]):
        f, ax = plt.subplots(1, 4, figsize=(20, 5))
        plt.tight_layout(pad=3)
        for j, n in enumerate(["x", "y", "z"]):
            ax.flat[j].plot(df[f"curr_pose_{n}"] - df[f"goal_pose_{n}"])
            ax.flat[j].set_ylim(-0.2, 0.2)
            ax.flat[j].grid()
            ax.flat[j].axvline(x=ts, ymin=0, ymax=1, color="red")
            ax.flat[j].set_title(f"{n} - {n}_goal")
            ax.flat[j].set_xlabel("step")
            ax.flat[j].set_ylabel("difference")
        diff = (
            df[["curr_pose_x", "curr_pose_y", "curr_pose_z"]].values
            - df[["goal_pose_x", "goal_pose_y", "goal_pose_z"]].values
        )
        ax.flat[3].plot(np.linalg.norm(diff, axis=1))
        ax.flat[3].set_ylim(0.0, l)
        ax.flat[j].axvline(x=ts, ymin=0, ymax=1, color="red")
        ax.flat[3].grid()
        ax.flat[3].set_title("pose to goal\ndistance (3D L2)")
        ax.flat[3].set_xlabel("step")
        ax.flat[3].set_ylabel("pose distance")
        f.canvas.draw()
        data = np.frombuffer(f.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape(f.canvas.get_width_height()[::-1] + (4,))[..., 1:]
        bottom_w = int(256 * buffer.shape[2] / 1024)
        bottom_image = cv2.resize(data, (buffer.shape[2], bottom_w))
        plt.close()
        bottom_images.append(bottom_image)
    bottom_images = np.stack(bottom_images)

    right_images = []
    # mpc_dists is recorded once per *planner* step; emb_dists once per
    # *env* step. Each planner step executes ``execution_prefix`` env
    # steps (the prefix of the CEM-output action sequence that's
    # actually applied to the env). So the invariant is:
    #     n_env_steps == n_planner_steps * execution_prefix
    # On a shared env-step x-axis, the k-th planner-step measurement
    # belongs at env-step (k+1) * execution_prefix (it's the score the
    # planner achieved AFTER applying its k-th action chunk).
    mpc_x = [(k + 1) * execution_prefix for k in range(len(mpc_dists))]
    for ts in range(buffer.shape[0]):
        right_image = []
        for v in range(mpc_dists.shape[1]):
            f = plt.figure(figsize=(6, 9))
            plt.tight_layout(pad=5)
            plt.plot(emb_dists[:, v], label="emb. distance")
            plt.plot(
                mpc_x,
                mpc_dists[:, v],
                label="mpc best result",
                marker="o",
            )
            plt.grid()
            y_max = max(max(emb_dists[:, v]), max(mpc_dists[:, v]))
            y_min = min(min(emb_dists[:, v]), min(mpc_dists[:, v]))
            y_max = y_max + abs(y_max) * 0.05
            y_min = y_min - abs(y_min) * 0.05
            plt.ylim(y_min, y_max)
            plt.legend()
            plt.axvline(x=ts, ymin=0, ymax=1, color="red")
            plt.xlabel("env step")
            plt.ylabel("feature L1 distance")
            plt.title(view_name_map[v])
            f.canvas.draw()
            right_data = np.frombuffer(f.canvas.tostring_argb(), dtype=np.uint8)
            right_data = right_data.reshape(f.canvas.get_width_height()[::-1] + (4,))[..., 1:]
            h_ = buffer.shape[1] + bottom_w + 75
            resized_right_image = cv2.resize(right_data, (int(400 * h_ / 800), h_))
            right_image.append(resized_right_image)
            plt.close()
        right_images.append(np.concatenate(right_image, 1))
    right_images = np.stack(right_images)

    # 320 + 75 + 640
    img = Image.new("RGB", (buffer.shape[2], 75), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), f"Goal: {goal}; Episode: {episode}/{total_eps}", (0, 0, 0), font=font)
    if comment != "":
        draw.text((550, 0), comment, (128, 64, 0), font=font)
        comment
    draw.text((0, 40), f"Goal image", (0, 0, 0), font=font)
    draw.text((int(517 * buffer.shape[2] / 1024), 40), f"Current image", (0, 0, 0), font=font)
    np.array(img)
    buffer = np.concatenate(
        [
            np.tile(np.array(img)[None], (buffer.shape[0], 1, 1, 1)),
            buffer,
            bottom_images,
        ],
        axis=1,
    )
    buffer = np.concatenate([buffer, right_images], axis=2)
    return buffer[start_frame:]


def ar_sample(state, actions, t=1, T=15, cache=True, device="cuda"):
    images_ = []
    features = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
        for k in range(0, T):
            action = actions[:, k : k + 1]
            # state.features += 1
            t = time.time()
            feature, state = wm.step(action, state, use_cache=cache, no_grad=True, pos_embedding_resize_t=64)
            d = time.time() - t
            print(f"step {k}: {d:.3f}s")
            features.append(feature[:, -1:].cuda())
            if len(features) == 8:
                frames = wm.heads["image_head"].decode(torch.cat(features, dim=1), cfg_scale=1.0, denoising_steps=32)
                images_ += [frames[0, :, 0].cpu().numpy()]
                features = []
            # image = wm.heads['image_head'].decode(feature.to(device))
            # images_.append(image[0,-1,0].cpu().numpy())
    return np.stack(images_)


def imagine_cem_decoded_video(
    wm,
    seed_wm_state,
    action_sequence,
    out_path,
    n_views,
    step_func=None,
    denoising_steps=3,
    cfg_scale=1.0,
    fps=5,
    device="cuda",
):
    """Imagine the CEM-output action sequence through the world model and
    render it via the image head (per-view decode with kv cache), then
    save a side-by-side mp4 (one column per view).

    Parameters
    ----------
    wm : WorldModel
        Loaded world model with ``heads["image_head"]`` populated.
    seed_wm_state : WorldModelState
        Same starting state CEM was given (CEM does not mutate the
        outer wm_state — the ``.repeat(samples)`` is a fresh copy
        internal to ``mpc_utils.cem``).
    action_sequence : Tensor of shape ``(rollout, action_dim)``
        ``means[-1]`` from CEM (the chosen action sequence the planner
        will execute). Will be reshaped to ``(1, rollout, action_dim)``.
    out_path : str
        Destination mp4 path.
    n_views : int
        Number of views to decode (concatenated horizontally).
    step_func : callable
        ``wm.step`` (or its compiled equivalent). Pass the same
        ``step_func`` policy_client uses so the action preprocessor in
        ``WorldModel.step`` runs identically to the CEM rollout.
    """
    import mediapy

    if step_func is None:
        step_func = wm.step

    # autocast wants a *string* device type (e.g. "cuda"), not a
    # torch.device — handle both forms.
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]

    # Move seed state to device. Don't mutate the caller's copy.
    state = seed_wm_state.to(device)
    actions_3d = action_sequence[None].to(device)  # (1, rollout, A)

    # Diagnostic — log what this call actually got.
    logger.info(
        f"[imagine] out={os.path.basename(out_path)}; "
        f"actions_3d.shape={tuple(actions_3d.shape)}; "
        f"per-step xyz: {actions_3d[0, :, :3].cpu().numpy().round(4).tolist()}; "
        f"sum={actions_3d.abs().sum().item():.4f}"
    )

    # Turn on the dispatcher's per-call canonical-action logger ONLY
    # for the duration of this imagination call (CEM also goes through
    # dispatcher.step and would flood the logs).
    _prev_log_env = os.environ.get("RJP_LOG_AOT_ACTIONS")
    os.environ["RJP_LOG_AOT_ACTIONS"] = "1"

    # 1) Predictor rollout:
    predicted = [state.features[:, -1:].to(device)]
    with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.float16, enabled=True):
        for k in range(actions_3d.shape[1]):
            feature, state = step_func(
                actions_3d[:, k : k + 1],
                state,
                use_cache=True,
                no_grad=True,
            )
            predicted.append(feature[:, -1:].to(device))
    features = torch.cat(predicted, dim=1)  # (1, rollout+1, V, H, W, C)

    # ── Diagnostic: are predicted features actually changing? ──────
    # If the WM is "stuck" (cache contamination, action-too-small,
    # preprocessor mis-routed) consecutive predicted features will be
    # near-identical and the decoded video will look static.
    seed_feat = predicted[0].float()
    diffs = [(predicted[i].float() - seed_feat).norm().item() for i in range(len(predicted))]
    seed_norm = seed_feat.norm().item()
    logger.info(
        f"[cem_imagination_viz] action mag (per-step l2): "
        f"{[round(actions_3d[0, k].norm().item(), 4) for k in range(actions_3d.shape[1])]}"
    )
    logger.info(
        f"[cem_imagination_viz] feat-vs-seed l2: "
        f"{[round(d, 3) for d in diffs]} (seed |f|={seed_norm:.2f}); "
        f"so per-step relative drift = "
        f"{[round(d / max(seed_norm, 1e-6), 4) for d in diffs]}"
    )

    # 2) Per-view decode with kv-cache (mirrors decode_view).
    per_view_videos = []
    T = features.shape[1]
    for view_idx in range(n_views):
        out = []
        with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.float16, enabled=True):
            kv_cache = None
            wm.heads["image_head"].last_output = None
            for t in range(T):
                frames, kv_cache = wm.heads["image_head"].decode(
                    features[:, t : t + 1, view_idx : view_idx + 1],
                    cfg_scale=cfg_scale,
                    denoising_steps=denoising_steps,
                    use_cache=True,
                    cache=kv_cache,
                )
                out.append(frames)
        decoded = torch.cat(out, dim=1)  # (1, T, 1, H, W, 3)
        arr = decoded[0, :, 0].cpu().numpy()  # (T, H, W, 3)
        per_view_videos.append(np.clip(arr, 0, 255).astype(np.uint8))

    # 3) Side-by-side concat along the width axis and write mp4.
    side_by_side = np.concatenate(per_view_videos, axis=2)  # (T, H, W * V, 3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mediapy.write_video(out_path, side_by_side, fps=fps)
    # Restore the dispatcher AOT-action logging flag.
    if _prev_log_env is None:
        os.environ.pop("RJP_LOG_AOT_ACTIONS", None)
    else:
        os.environ["RJP_LOG_AOT_ACTIONS"] = _prev_log_env
    return out_path


if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 64
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=4754)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--weights-filename",
        dest="weights_filename",
        type=str,
        default="world_model.pth.tar",
        help=(
            "Name of the transition-model weights file inside --checkpoint_path. "
            "Default: world_model.pth.tar. Use this when the packaged predictor "
            "weights use a non-standard name (e.g. cooldown_8b_200k.pth.tar) to "
            "avoid symlinking."
        ),
    )
    parser.add_argument(
        "--intermediate_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional raw .pth.tar to override the transition_model weights AFTER "
            "WorldModel.load_pretrained. The packaged --checkpoint_path provides the "
            "architecture, encoder, decoder, heads, and (if --load_compiled is set) the "
            "AOT-compiled .pt2 programs; this flag only swaps the predictor weights. "
            "Useful for evaluating intermediate / in-training checkpoints with the same "
            "compiled programs as the final-epoch packaging."
        ),
    )
    parser.add_argument("--goal_name", type=str, default="some task")
    parser.add_argument("--model_name", type=str, default="world model")
    # parser.add_argument('--encoder_path', type=str, default='/code/models/world_models/vitg16-vjepa2.pth.tar')
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--decoder_path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--init_ep", type=int, default=0)
    parser.add_argument(
        "--resume_existing",
        action="store_true",
        default=False,
        help=(
            "Skip any episode whose output .npz already exists. Enables "
            "requeue-on-timeout to continue from the last finished episode."
        ),
    )
    parser.add_argument(
        "--reach_episode_idx",
        type=int,
        default=None,
        help=(
            "Opt-in (reach/obstacle/object-reach tasks): pin every env.reset in this "
            "run to a SPECIFIC reach episode index instead of the env's auto-"
            "incrementing per-seed counter. Forwarded verbatim into the goal/init "
            "reset kwargs so a single-episode process reproduces the bit-identical "
            "(start, goal) draw of sequential episode K. Default None = auto-counter "
            "(unchanged legacy behaviour). Typically set equal to --init_ep when "
            "running one SLURM job per episode."
        ),
    )
    parser.add_argument("--aspect_ratio", type=float, default=(1.0, 1.0), nargs="+")
    parser.add_argument("--use_cameras", type=int, default=(0,), nargs="+")
    parser.add_argument("--init_task", type=int, default=0)
    parser.add_argument("--embed_dim", type=int, default=1408)
    parser.add_argument("--task_path", type=str, default="tasks/pickup_task.json")
    parser.add_argument(
        "--bypass_input",
        action="store_true",
        default=False,
    )
    parser.add_argument("--cem", type=int, default=1, help="number of initial cem iterations")
    parser.add_argument("--step_cem", type=int, default=1, help="number of per-step2 cem iterations")
    parser.add_argument(
        "--translation_only",
        action="store_true",
        default=False,
        help="whether to only search within 3D translation space of actions",
    )
    parser.add_argument("--use_gripper", action="store_true", default=False)
    parser.add_argument(
        "--load_image_head",
        action="store_true",
        default=False,
        help="whether to load image head (useful for debugging)",
    )
    parser.add_argument("--rollout", type=int, default=1, help="predictor prediction horizon")
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="number of CEM samples in one refinement iteration",
    )
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--objective", type=str, default="l1", help="cos | l1")
    parser.add_argument("--maxnorm", type=float, default=0.1, help="action search max norm")
    parser.add_argument("--robot_init_std", type=float, default=0.05, help="action search init_std")
    parser.add_argument("--maxgripper", type=float, default=0.5, help="action search max norm")
    parser.add_argument("--gripper_init_std", type=float, default=0.1, help="action search init_std")
    parser.add_argument("--cutoff", type=float, default=0.98, help="action search top percentile")
    parser.add_argument(
        "--view_weights",
        type=float,
        nargs="+",
        default=None,
        help="how to weight rewards generated by each of the views",
    )
    parser.add_argument("--momentum_mean", type=float, default=0.15)
    parser.add_argument("--momentum_std", type=float, default=0.25)
    parser.add_argument("--momentum_mean_gripper", type=float, default=0.25)
    parser.add_argument("--momentum_std_gripper", type=float, default=0.25)
    parser.add_argument(
        "--action_substeps",
        type=int,
        default=5,
        help="if >1 will divide the action by N and execute it N times",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="a multiplier for the number of cem samples",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=35,
        help="total envitonment steps to run the policy for",
    )
    parser.add_argument(
        "--closed_loop_context_len",
        type=int,
        default=1,
        help="how many encoder context step to initialize the predictor with in any mpc run",
    )
    parser.add_argument(
        "--execution_prefix",
        type=int,
        default=-1,
        help="how many how many resulting mpc steps to run on the robot (defaults to all steps)",
    )
    parser.add_argument("--output_filename", type=str, default="wm_deployment")
    parser.add_argument("--cache_path", type=str, default="wm_rope/compute_cache.bin")
    parser.add_argument("--force_recache", action="store_true", default=False)
    parser.add_argument("--use_cache", action="store_true", default=False)
    parser.add_argument(
        "--image_size",
        type=int,
        nargs="+",
        default=(256,),
        help="the size of the image observation",
    )
    parser.add_argument("--center_crop", action="store_true", default=False)
    parser.add_argument(
        "--load_compiled",
        type=str,
        default="bf16",
        help="Compiled checkpoint mode: 'bf16', 'mp-bf16', or 'false' for eager (default: bf16)",
    )
    parser.add_argument(
        "--compiled_max_steps",
        type=int,
        default=None,
        help=(
            "When --load_compiled is set, only load AOT programs for steps\n"
            "0..compiled_max_steps-1 (the dispatcher uses programs keyed by\n"
            "the cache size at call time, so for a planner that pre-runs\n"
            "closed_loop_context_len-1 cache-init steps and then a CEM with\n"
            "--rollout R the highest cache index needed is\n"
            "closed_loop_context_len-1 + R, i.e.\n"
            "compiled_max_steps = closed_loop_context_len + R is sufficient).\n"
            "Loading fewer programs cuts host RAM by ~each-program's-CPU-residency\n"
            "per skipped step (the AOT inductor runtime can be ~10-20 GB/program\n"
            "per rank for 1B models on 240p inputs). Default: load all\n"
            "variants present in the compiled/<precision>/ dir."
        ),
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=[None],
        nargs="+",
        help="camera id to provide to the WM. Follows the droid formulation with 0 being left camera, 1 for the right one, and 2 for wirst",
    )
    parser.add_argument(
        "--embodiment_id",
        type=int,
        default=None,
        help="embodiment id to provide to the WM. Follows the droid formulation with 1 being left camera and 2 for the right one",
    )
    parser.add_argument(
        "--cem_imagination_viz",
        action="store_true",
        default=False,
        help=(
            "After every CEM call, imagine the chosen action sequence "
            "(means[-1]) through the world model from the same starting "
            "state CEM was given, decode each predicted feature with the "
            "image head, and save a side-by-side (side cam | wrist cam) "
            "mp4 per planner step. Output: <output_filename>_cem_viz/"
            "ep{ep}_step{c:02d}.mp4. Requires the decoder to be loaded "
            "(forces load_decoder=True)."
        ),
    )
    parser.add_argument(
        "--cem_imagination_denoising_steps",
        type=int,
        default=3,
        help="Decoder denoising steps for the imagination viz (default 3, fast).",
    )
    parser.add_argument(
        "--cem_imagination_cfg_scale",
        type=float,
        default=1.0,
        help="Decoder CFG scale for the imagination viz (default 1.0).",
    )
    parser.add_argument(
        "--cem_imagination_fps",
        type=int,
        default=5,
        help="Output mp4 fps for the imagination viz (default 5).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help=(
            "Discount factor for the gamma-discounted return logged at\n"
            "the end of each episode. The return is computed over the\n"
            "per-env-step rewards reported by the env\n"
            "(``info['chunk_rewards']`` from ``env.step_chunk`` —\n"
            "populated by RoboCasaEnv when reach_task=True; the\n"
            "top-level last-tick reward is used as a fallback for envs\n"
            "that do not provide it). Default 0.99."
        ),
    )
    parser.add_argument(
        "--save_gpu_memory",
        action="store_true",
        default=False,
        help=(
            "Shuttle encoder and WM transition_model between CPU and GPU to "
            "reduce peak VRAM. The WM is kept in fp16 on CPU and moved to "
            "GPU only for warmup+CEM planning; the encoder is moved to GPU "
            "only for image encoding. Adds latency from transfers but allows "
            "large models (e.g. 4B) to run on 16 GB GPUs."
        ),
    )
    args = parser.parse_args()

    fold, fl = os.path.split(args.output_filename)
    os.makedirs(fold, exist_ok=True)

    assert args.execution_prefix != 0, "the robot needs to execute at least one action!"
    world_size, rank = init_distributed()
    print(f"Initialized (rank/world-size) {rank}/{world_size}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if len(args.image_size) == 1:
        args.image_size = (args.image_size[0], args.image_size[0])
    image_size_pil = (args.image_size[1], args.image_size[0])

    if args.objective == "cos":
        objective = mpc_utils.cos
    elif args.objective == "l1":
        objective = mpc_utils.l1
    else:
        raise ValueError("unsupported objective")
    if os.path.isdir(args.task_path):
        task_files = []
        for task_file in sorted(os.listdir(args.task_path)):
            if task_file.endswith(".json"):
                with open(os.path.join(args.task_path, task_file), "r") as f:
                    task_info = json.load(f)
                task_files.append((task_file[: -len(".json")], task_info))
        if args.init_task > 0:
            task_files = task_files[args.init_task :]
    else:
        with open(args.task_path, "r") as f:
            task_info = json.load(f)
        task_files = [(args.task_path[: -len(".json")], task_info)]
    logger.info(f"rank {rank}/{world_size}: here")
    if rank == 0:
        print("Running Action Env client")
        logger.info(f"rank {rank}/{world_size}: creating env")
        # Server-side ``env.reset(mode='goal')`` runs the rotation +
        # joint-singularity rejection validator (up to
        # ``_REACH_VALIDATION_MAX_TRIES = 100`` candidates × ~2 OSC
        # drives each), which can take many minutes on pathological
        # scenes. 1.8M ms = 30 min gives the validator comfortable
        # headroom and still bounds infinite hangs.
        env = GymEnvClient(host=args.host, port=args.port, timeout_ms=1_800_000)
    global_stats = defaultdict(list)
    load_compiled = args.load_compiled if args.load_compiled.lower() != "false" else False
    log_rss("policy_client:before_load_pretrained")
    wm_load_device = "cpu" if args.save_gpu_memory else "cuda"
    wm = WorldModel.load_pretrained(
        args.checkpoint_path,
        load_encoder=rank == 0,
        load_decoder=args.cem_imagination_viz,
        device=wm_load_device,
        load_compiled=load_compiled,
        compiled_max_steps=args.compiled_max_steps,
        weights_filename=args.weights_filename,
    )
    log_rss("policy_client:after_load_pretrained")

    # ── Optional: swap transition_model weights with an intermediate (in-training)
    # raw .pth.tar checkpoint. The encoder/decoder/heads/AOT programs from the
    # packaged --checkpoint_path stay in place; only the predictor weights are
    # replaced. Works for both eager (WorldModel) and AOT
    # (AOTWorldModelDispatcher) returns from load_pretrained.
    #
    # Under --load_compiled, the dispatcher already injected the packaged-final
    # bf16 weights into every loaded .pt2 program via load_shared_constants().
    # A bare load_state_dict on the offloaded eager
    # transition_model is therefore a NO-OP at inference; we additionally
    # re-extract bf16 weights and re-call load_constants on every program.
    # This is O(ms) and avoids the ~5h recompile per checkpoint.
    if args.intermediate_checkpoint is not None:
        from app.robo_jepa.inference.dispatcher import AOTWorldModelDispatcher
        from app.robo_jepa.inference.aot_compile import load_shared_constants

        if rank == 0:
            logger.info(f"Loading intermediate transition_model weights from {args.intermediate_checkpoint}")
        inter_state = torch.load(args.intermediate_checkpoint, map_location="cpu")
        # Save format: {model, opt, scaler, epoch, stats}; predictor under "model".
        # Older runs used the "predictor" key. Mirror load_pretrained's try/except.
        if "predictor" in inter_state:
            inter_sd = inter_state["predictor"]
        elif "model" in inter_state:
            inter_sd = inter_state["model"]
        else:
            raise RuntimeError(
                f"--intermediate_checkpoint {args.intermediate_checkpoint!r} has neither "
                f"'predictor' nor 'model' top-level key (keys: {list(inter_state)})"
            )
        inter_sd = {k.replace("module.", ""): v for k, v in inter_sd.items()}
        # Drop the optimizer state from the loaded dict to avoid keeping ~3x model
        # size in host memory longer than necessary.
        for _drop in ("opt", "scaler"):
            inter_state.pop(_drop, None)

        underlying_wm = wm._wm if isinstance(wm, AOTWorldModelDispatcher) else wm
        msg = underlying_wm.transition_model.load_state_dict(inter_sd, strict=False)
        # Validate: any missing PARAMETER (vs. buffer) is a hard error — we
        # don't want a silent partially-loaded predictor.
        param_names = {n for n, _ in underlying_wm.transition_model.named_parameters()}
        missing_params = [k for k in msg.missing_keys if k in param_names]
        if missing_params:
            raise RuntimeError(
                f"--intermediate_checkpoint missing PARAMETERS (not just buffers): {missing_params[:10]}"
            )
        if rank == 0:
            logger.info(
                f"intermediate load_state_dict: missing={len(msg.missing_keys)} "
                f"unexpected={len(msg.unexpected_keys)} "
                f"epoch_in_ckpt={inter_state.get('epoch')}"
            )
        for p in underlying_wm.transition_model.parameters():
            p.requires_grad = False
        underlying_wm.transition_model.eval()

        if isinstance(wm, AOTWorldModelDispatcher):
            # Re-extract bf16 weights and re-inject into every loaded .pt2
            # program. user_managed=True keeps tensor references, so the second
            # load_constants call is what makes the swap visible at
            # inference. The dispatcher's _extract_weights() handles the
            # bf16 cast + GPU placement; we just have to invalidate its
            # cached self._weights so it rebuilds from the new state_dict.
            underlying_wm.transition_model.to(device)  # need GPU for bf16 extraction
            wm._weights = None
            new_weights = wm._extract_weights()
            for (step_idx, bs), prog in wm._programs.items():
                fqns = prog.get_constant_fqns()
                aligned = {fqn: new_weights[fqn] for fqn in fqns if fqn in new_weights}
                load_shared_constants(prog, aligned)
            underlying_wm.transition_model.cpu()  # re-offload, mirror dispatcher policy
            torch.cuda.empty_cache()
            if rank == 0:
                logger.info(
                    f"Re-injected {len(new_weights)} intermediate weights into "
                    f"{len(wm._programs)} compiled .pt2 programs"
                )

        # Sanity fingerprint — single float that should differ from the
        # baseline if the swap actually took effect.
        if rank == 0:
            fp = sum(p.detach().float().sum().item() for p in underlying_wm.transition_model.parameters())
            logger.info(f"intermediate transition_model weight-sum fingerprint: {fp:.4f}")

    if args.save_gpu_memory:
        wm.transition_model.half()
        n_params = sum(p.numel() for p in wm.transition_model.parameters()) / 1e9
        logger.info(
            f"[save_gpu_memory] transition_model converted to fp16 on CPU "
            f"({n_params:.2f}B params, "
            f"{sum(p.numel() * p.element_size() for p in wm.transition_model.parameters()) / 1e9:.2f}GB)"
        )
        if wm.encoder is not None:
            enc_params = sum(p.numel() for p in wm.encoder.parameters()) / 1e9
            logger.info(f"[save_gpu_memory] encoder on CPU ({enc_params:.2f}B params)")
        logger.info("[save_gpu_memory] Skipping wm.to(device) — models stay on CPU until needed")
    else:
        wm.to(device)
    log_rss("policy_client:after_wm.to(device)")
    feature_encoder = wm.encode_features
    # feature_encoder = torch.compile(wm.encode_features, dynamic=False, fullgraph=True)
    # step_func = wm.step
    # if args.use_cache:
    # step_func = torch.compile(wm.forward, dynamic=False, )#fullgraph=True)
    # wm.forward = torch.compile(wm.forward, dynamic=False, fullgraph=True)
    # else:
    step_func = wm.step

    W = np.eye(7, dtype=np.float32)
    use_cameras = [str(c) for c in args.use_cameras]
    for task_name, task_info in task_files:
        if rank == 0:
            print(f"running task {task_name}! Continue?")
            if not args.bypass_input:
                input()
            else:
                print("yes!")
        for ep in range(args.init_ep, args.episodes):
            if args.resume_existing and os.path.exists(f"{args.output_filename}_{task_name}_{ep}.npz"):
                if rank == 0:
                    print(f"[resume] {args.output_filename}_{task_name}_{ep}.npz exists — skipping episode {ep}")
                continue
            frames = []
            emb_dists = []
            mpc_dists = []
            if rank == 0:
                print(f"running episode {ep}")
            stats = defaultdict(list)
            done = False
            # Per-episode env-reward bookkeeping. Per-step rewards are
            # recorded once per env tick (one per ``info['chunk_rewards']``
            # entry from ``env.step_chunk``); the discounted return is
            # ``Σ_t γ^t * r_t`` with ``t`` indexing env steps from 0.
            episode_rewards: list[float] = []
            episode_step_infos: list[dict] = []
            episode_step_index = 0  # ``t`` for the gamma discount
            episode_disc_return = 0.0
            intermediate_goals = task_info.get("intermediate_goals")
            goals_list = None
            if rank == 0:
                if args.save_gpu_memory:
                    t_shuttle = time.perf_counter()
                wm.encoder.to(device)
                if args.save_gpu_memory:
                    dt_shuttle = (time.perf_counter() - t_shuttle) * 1000
                    log_gpu_mem(f"encoder→GPU for initial encoding ({dt_shuttle:.0f}ms)")
                # Opt-in per-episode pinning: forward reach_episode_idx so the
                # env draws sequential episode ``ep``'s exact (start, goal).
                # ``args.reach_episode_idx`` is the index of the loop's FIRST
                # episode (== --init_ep in per-episode SLURM jobs); it advances
                # with the loop so a multi-episode run reproduces every episode.
                _goal_kw = dict(task_info["goal"])
                if args.reach_episode_idx is not None:
                    _goal_kw["reach_episode_idx"] = args.reach_episode_idx + (ep - args.init_ep)
                goal, goal_obs_list = env.reset(**_goal_kw)
                img_keys = [k for k in goal.keys() if k.startswith("img_") and any(c in k for c in use_cameras)]
                if intermediate_goals is not None:
                    goals_list = {}
                    intermediate_goal_feature = []
                    for img_key in img_keys:
                        for k, v in intermediate_goals.items():
                            try:
                                k_ = int(k)
                            except:
                                raise ValueError("intermediate goals keys should be integers")
                            goals_list[v] = goal_obs_list[k_]
                            goal_images = wm.preprocess_frames(
                                np.stack([goals_list[v][img_key], goals_list[v][img_key]]),
                                size=args.image_size,
                            )
                            goal_features = feature_encoder(
                                goal_images.to(device),
                                window=2,
                                tubelet_size=2,
                                grid_size=16,
                                dtype=torch.float16,
                                device=device,
                            )
                            goals_list[v] = goal_features
                        intermediate_goal_feature = torch.cat(
                            [goals_list[v] for v in sorted(list(goals_list.keys()))],
                            dim=0,
                        ).unsqueeze(1)
                        intermediate_goal_features.append(intermediate_goal_feature)
                    intermediate_goal_features = torch.cat(intermediate_goal_features, dim=1)
                print("computing goal feature...")

                t = time.time()
                goal_features = []
                for img_key, ar in zip(img_keys, args.aspect_ratio):
                    goal_image = goal[img_key]
                    h_, w_ = goal_image.shape[:2]
                    if args.aspect_ratio is not None:
                        new_h = h_
                        new_w = int(w_ // ar)
                        goal_image = (
                            F.resize(
                                torch.from_numpy(goal_image).permute(2, 0, 1),
                                [new_h, new_w],
                            )
                            .permute(1, 2, 0)
                            .numpy()
                        )
                    if args.center_crop:
                        img_goal = np.array(
                            F.center_crop(Image.fromarray(goal_image), min(*goal_image.shape[:2])).resize(
                                image_size_pil
                            )
                        )
                    else:
                        img_goal = np.array(Image.fromarray(goal_image).resize(image_size_pil))
                    goal_feature = wm.encode(
                        img_goal[None],
                        dtype=torch.float16,
                        device=device,
                        size=args.image_size,
                    )[:, None, None]
                    goal_features.append(goal_feature)
                # NCCL broadcast requires matching dtypes across ranks.
                # `wm.encode` returns bfloat16 (autocast output, no cast back),
                # but rank!=0 allocates the receive buffer as float32 below,
                # which deadlocks the dist.broadcast() / dist.barrier() pair.
                goal_features = torch.cat(goal_features, dim=2).to(torch.float32)
                d = time.time() - t
                print(f"done computing goal feature. spent {d * 1000:.3f}ms")
            else:
                patch_h = args.image_size[0] // 16
                patch_w = args.image_size[1] // 16
                goal_features = torch.zeros(
                    (1, 1, len(args.camera_id), patch_h, patch_w, args.embed_dim),
                    dtype=torch.float32,
                    device=device,
                )
                if intermediate_goals is not None and len(intermediate_goals) > 0:
                    intermediate_goal_features = torch.zeros(
                        (
                            1,
                            len(intermediate_goals),
                            len(args.camera_id),
                            patch_h,
                            patch_w,
                            args.embed_dim,
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
            if world_size != 1:
                # Note: shape sanity-check broadcast was tried earlier but
                # caused a deadlock specifically on LOCAL_RANK=0 of every
                # non-master node (likely a CUDA/NCCL stream-sync edge case
                # with the int64 device tensor + .tolist()). Reverted: rely
                # on rank!=0 allocating the correct shape from
                # len(args.camera_id) — your env returns matching n_views.
                logger.info(
                    f"DEBUG rank {rank}: BEFORE broadcast(goal_features) shape={tuple(goal_features.shape)} dtype={goal_features.dtype} device={goal_features.device}"
                )
                dist.broadcast(goal_features, src=0)
                logger.info(
                    f"DEBUG rank {rank}: AFTER broadcast(goal_features) (broadcast is a sync point, no extra barrier needed)"
                )
                if intermediate_goals is not None:
                    dist.broadcast(intermediate_goal_features, src=0)
                    if rank != 0:
                        goals_list = {
                            v: intermediate_goal_features[j]
                            for j, v in enumerate(sorted(list(intermediate_goals.values())))
                        }
            if intermediate_goals is not None and rank == 0:
                for v in goals_list:
                    logger.info(f"rank {rank}/{world_size}: goal {v}: features: {goals_list[v].sum().item():.4f}")
            if rank == 0:
                logger.info(f"rank {rank}/{world_size}: features: {goal_features.sum().item():.4f}")
            goal_features = goal_features.to(device)

            # Resolve embodiment name (used to pick the per-embodiment
            # obs key + dim BEFORE the first env-reset block reads obs).
            wm_emb_ids = None
            embodiment_name = None
            if wm.transition_model.is_multiembodiment:
                if args.embodiment_id is not None:
                    wm_emb_ids = torch.from_numpy(np.array([[args.embodiment_id]]))
                    embodiment_list = getattr(wm.transition_model, "embodiment_list", None)
                    if embodiment_list is not None and 0 <= args.embodiment_id < len(embodiment_list):
                        embodiment_name = embodiment_list[args.embodiment_id]
                    if rank == 0:
                        logger.info(f"resolved embodiment id={args.embodiment_id} -> name={embodiment_name!r}")
                else:
                    raise ValueError("embodiment is not Provided!")
            # Per-embodiment proprio: the franka_omron_state_delta variant
            # was trained on a 14-D RoboCasa proprio (base_pose + ee-in-base
            # + gripper qpos) — see app/robo_jepa/embodiment_preprocessing.py
            # ROBOCASA_STATE_DELTA_PROPRIO_DIM. The env exposes that vector
            # under obs["proprio_robocasa"]. All other embodiments use the
            # legacy 7-D obs["pose"].
            if embodiment_name == "franka_omron_state_delta":
                proprio_obs_key = "proprio_robocasa"
                proprio_dim = 14
                use_state_delta_path = True
            else:
                proprio_obs_key = "pose"
                proprio_dim = 7
                use_state_delta_path = False

            d_robot_move = 0.0
            d_image_post_proc = 0.0
            d_obs_send = 0.0
            d_compute_action = 0.0
            d_encode_obs = 0.0
            raw_images = defaultdict(list)
            images = []
            poses = []
            joints = []
            if rank == 0:
                poses.append(goal["pose"].copy())
                joints.append(goal["joints"].copy())
                _init_kw = dict(task_info["init"])
                if args.reach_episode_idx is not None:
                    _init_kw["reach_episode_idx"] = args.reach_episode_idx + (ep - args.init_ep)
                obs, init_obs_list = env.reset(**_init_kw)
                commanded_pose = obs["pose"][:6].copy()
                commanded_gripper = float(obs["pose"][6])
                commanded_pose = obs["pose"][:6].copy()
                commanded_gripper = float(obs["pose"][6])
                print("computing obs feature...")
                for img_key in img_keys:
                    raw_images[img_key].append(goal[img_key].copy())
                    raw_images[img_key].append(obs[img_key].copy())
                goal_images = np.concatenate(
                    [F.resize(Image.fromarray(goal[img_key]), (480, 640)) for img_key in img_keys],
                    axis=0,
                )
                obs_images = np.concatenate(
                    [F.resize(Image.fromarray(obs[img_key]), (480, 640)) for img_key in img_keys],
                    axis=0,
                )
                images.append(np.concatenate([goal_images, obs_images], axis=1))
                stats["goal_pose_x"].append(goal["pose"][0])
                stats["goal_pose_y"].append(goal["pose"][1])
                stats["goal_pose_z"].append(goal["pose"][2])
                stats["curr_pose_x"].append(obs["pose"][0])
                stats["curr_pose_y"].append(obs["pose"][1])
                stats["curr_pose_z"].append(obs["pose"][2])
                stats["episode"].append(ep)
                stats["step"].append(-1)
                # Initialise per-step env-reward columns so they stay
                # length-aligned with the other ``stats`` columns once
                # the per-step ``env.step_chunk`` loop starts appending
                # to them. The ``step=-1`` row corresponds to the
                # initial reset, BEFORE any env step, so its reward is
                # 0 by definition (and contributes 0 to the cumulative
                # / discounted return).
                stats["env_reward"].append(0.0)
                stats["env_reward_cumsum"].append(0.0)
                poses.append(obs["pose"].copy())
                joints.append(obs["joints"].copy())

                t = time.time()
                obs_features = []
                for img_key, ar in zip(img_keys, args.aspect_ratio):
                    image = obs[img_key]
                    h_, w_ = image.shape[:2]
                    if args.aspect_ratio is not None:
                        new_h = h_
                        new_w = int(w_ // ar)
                        image = (
                            F.resize(torch.from_numpy(image).permute(2, 0, 1), [new_h, new_w]).permute(1, 2, 0).numpy()
                        )
                    if args.center_crop:
                        img = np.array(
                            F.center_crop(Image.fromarray(image), min(*goal_image.shape[:2])).resize(image_size_pil)
                        )
                    else:
                        img = np.array(Image.fromarray(image).resize(image_size_pil))
                    obs_feature = wm.encode(
                        img[None],
                        dtype=torch.float16,
                        device=device,
                        size=args.image_size,
                    )[:, None, None]
                    obs_features.append(obs_feature)
                # Match rank!=0 receive-buffer dtype (float32) for dist.broadcast.
                obs_features = torch.cat(obs_features, dim=2).to(torch.float32)
                d = time.time() - t
                obs_pose = torch.from_numpy(obs[proprio_obs_key]).float().to(device)
                print(f"done computing obs feature. spent {d * 1000:.2f}ms")
                print(f"current pose: {obs['pose']}")
                if args.save_gpu_memory and wm.encoder is not None:
                    wm.encoder.to("cpu")
                    torch.cuda.empty_cache()
                    log_gpu_mem("after initial encoding, encoder→CPU")

                last_time_send_obs = 0  # obs['time_send_obs']
                last_time_step_begin = 0  # obs['time_step_begin']
                last_time_step_end = 0  # obs['time_step_end']
            else:
                patch_h = args.image_size[0] // 16
                patch_w = args.image_size[1] // 16
                obs_features = torch.zeros(
                    (1, 1, len(args.camera_id), patch_h, patch_w, args.embed_dim),
                    dtype=torch.float32,
                    device=device,
                )
                obs_pose = torch.zeros(proprio_dim, dtype=torch.float32, device=device)
            if world_size != 1:
                dist.broadcast(obs_features, src=0)
                dist.broadcast(obs_pose, src=0)
            if rank == 0:
                logger.info(f"rank {rank}/{world_size}: features: {obs_features.sum().item():.4f}")
            c = 0
            j = 0
            wm_uses_proprio = wm.transition_model.state_tokens > 0
            wm_uses_camera_extrinsics = wm.transition_model.camera_extrinsics_tokens > 0
            wm_uses_camera_tokens = wm.transition_model.view_map is not None
            wm_camera_ids = None
            if wm_uses_camera_tokens and args.camera_id[0] is not None:
                wm_camera_ids = torch.from_numpy(np.array([args.camera_id]))

            features_sequence = deque(maxlen=args.closed_loop_context_len)
            poses_sequence = deque(maxlen=args.closed_loop_context_len)
            features_sequence.append(obs_features.cpu())
            emb_dists.append(-objective(obs_features.cpu(), goal_features.cpu()).mean(dim=(0, 1, 3, 4)).cpu().numpy())
            poses_sequence.append(obs_pose.cpu())
            prev_actions = None
            prev_stds = None
            while c < args.total_steps:
                if args.save_gpu_memory:
                    t_shuttle = time.perf_counter()
                    wm.transition_model.to(device)
                    torch.cuda.empty_cache()
                    dt_shuttle = (time.perf_counter() - t_shuttle) * 1000
                    log_gpu_mem(f"WM→GPU [ep{ep},c={c}] ({dt_shuttle:.0f}ms)")
                if len(poses_sequence) == 1:
                    wm_poses = None
                    if wm_uses_proprio:
                        wm_poses = poses_sequence[-1][None, None]
                    wm_state = WorldModelState.from_features(
                        features=torch.cat(list(features_sequence), dim=1),
                        states=wm_poses,
                        view_map=wm_camera_ids,
                        embodiment_ids=wm_emb_ids,
                    ).to("cpu")
                    # HERE
                else:
                    # b /code/app/droid_wm/vjepa2/models/utils/modules.py:187
                    # b /code/app/droid_wm/vjepa2/models/utils/attention.py:40

                    curr_features = torch.cat(list(features_sequence), dim=1)
                    if use_state_delta_path:
                        # For franka_omron_state_delta the WM action *is*
                        # the per-step diff of the 14-D RoboCasa proprio
                        # (no rotation-matrix Euler-diff math is needed —
                        # both base and EE deltas are just numerical
                        # subtraction, mirroring the dataset's
                        # ``action_source = state_delta`` path).
                        poses_stacked = torch.stack(list(poses_sequence))
                        curr_actions = (poses_stacked[1:] - poses_stacked[:-1])[None]
                    else:
                        curr_actions = poses_to_diffs(torch.stack(list(poses_sequence)))[None]

                    if wm_uses_proprio:
                        wm_poses = torch.stack(list(poses_sequence))[None]
                    wm_state = WorldModelState.from_features(
                        features=curr_features[:, :1],
                        states=wm_poses[:, :1],
                        view_map=wm_camera_ids,
                        embodiment_ids=wm_emb_ids,
                    )
                    wm_state = wm_state.to(device)
                    log_rss(f"policy_client:before_warmup[ep{ep},c={c}]")
                    with (
                        torch.no_grad(),
                        torch.amp.autocast("cuda", dtype=torch.float16, enabled=True),
                    ):
                        for k in range(curr_actions.shape[1]):
                            # a very ugly way to initialize kv cache with encoder features
                            # but sparse attention does not work here.
                            feature, wm_state = step_func(
                                curr_actions[:, k : k + 1],
                                wm_state,
                                no_grad=True,
                                device=device,
                                use_cache=True,
                                batch=100,
                            )
                            wm_state.features = curr_features[:, k + 1 : k + 2].to(device)
                            if wm_uses_proprio:
                                wm_state.states = wm_poses[:, k + 1 : k + 2].to(device)
                    # OOM mitigation (bf16 path): keep wm_state on GPU
                    # rather than ``.to("cpu")``. The post-warmup state
                    # carries the AOT KV cache (bf16, depth=40,
                    # num_kv_heads=8, cache_len=8, frame_tokens=600,
                    # head_dim=64). Moving it to CPU and back doubles
                    # the bookkeeping; worse, the next CEM call does
                    # ``wm_state.repeat(samples=32)`` BEFORE
                    # ``.to(device)``, so the repeat materialises the
                    # full B=32 cache (~12 GB for keys+values) on
                    # **host** RSS — that is the actual leak that pushes
                    # 8 ranks past 1.5 TB by env step ~5. Keeping the
                    # state on GPU lets ``repeat`` allocate via the
                    # CUDA caching allocator (which we already pay for)
                    # and bypass the host bloat entirely.
                    log_rss(f"policy_client:after_warmup[ep{ep},c={c}]")
                # act0=np.zeros((1,8,7));act0[...,0]=0.02;imgs = ar_sample(wm_state.to('cuda'), torch.from_numpy(act0).cuda().float(), T=8);mediapy.write_video('motion.mp4', imgs[0], fps=5)
                t_start_action = time.perf_counter() * 1000
                if wm.encoder is not None:
                    wm.encoder.to("cpu")
                torch.cuda.empty_cache()
                cem_steps = args.cem if c == 0 else args.step_cem
                log_rss(f"policy_client:before_cem[ep{ep},c={c}]")
                log_gpu_mem(f"before CEM [ep{ep},c={c}]")
                reset_cuda_peak()
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                    rewards, means, stds = mpc_utils.cem(
                        wm,
                        wm_state,
                        goal_features if goals_list is None else goals_list,
                        rollout=args.rollout,
                        samples=args.samples,
                        cem_steps=cem_steps,
                        cutoff=args.cutoff,
                        momentum_mean=args.momentum_mean,
                        momentum_std=args.momentum_std,
                        momentum_mean_gripper=args.momentum_mean_gripper,
                        momentum_std_gripper=args.momentum_std_gripper,
                        verbose=True,
                        maxnorm=args.maxnorm,
                        maxgripper=args.maxgripper,
                        objective=objective,
                        robot_init_std=args.robot_init_std,
                        gripper_init_std=args.gripper_init_std,
                        prev_actions=prev_actions,
                        prev_stds=prev_stds,
                        action_repeat=args.action_repeat,
                        global_step=c,
                        view_weights=args.view_weights,
                        total_steps=args.total_steps,
                        use_gripper=args.use_gripper,
                        num_batches=args.num_batches,
                        translation_only=args.translation_only,
                        device=device,
                        use_cache=args.use_cache,
                        world_size=world_size,
                        rank=rank,
                        step_func=step_func,
                    )
                    mpc_dists.append(-rewards[-1].cpu().numpy())
                log_rss(f"policy_client:after_cem[ep{ep},c={c}]")
                log_gpu_mem(f"after CEM [ep{ep},c={c}]")
                # ── OOM mitigation: drop CEM intermediate buffers and flush
                # the CUDA caching allocator. Without this the host RSS
                # grew ~11 GB / rank / env step → 1.5 TB OOM by step ~5.
                # gc.collect() forces the Python garbage collector to
                # reclaim the per-rollout closures (rewards_sequence,
                # action_stats, history wm_states) that the dispatcher
                # holds via reference cycles through tensors;
                # empty_cache() releases GPU caching reserve back to the
                # CUDA driver so it does not bloat per-rank pinned host
                # bookkeeping over many env steps.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log_rss(f"policy_client:after_cem_gc[ep{ep},c={c}]")
                if args.save_gpu_memory:
                    t_shuttle = time.perf_counter()
                    wm.transition_model.to("cpu")
                    torch.cuda.empty_cache()
                    dt_shuttle = (time.perf_counter() - t_shuttle) * 1000
                    log_gpu_mem(f"WM→CPU [ep{ep},c={c}] ({dt_shuttle:.0f}ms)")
                if wm.encoder is not None:
                    if args.save_gpu_memory:
                        t_shuttle = time.perf_counter()
                    wm.encoder.to(device)
                    if args.save_gpu_memory:
                        dt_shuttle = (time.perf_counter() - t_shuttle) * 1000
                        log_gpu_mem(f"encoder→GPU for re-encoding [ep{ep},c={c}] ({dt_shuttle:.0f}ms)")
                action = means[-1]
                if args.execution_prefix != -1:
                    prev_actions = action[args.execution_prefix :]
                    prev_stds = stds[-1][args.execution_prefix :]
                if rank == 0:
                    logger.info(f"rank {rank}/{world_size}: mpc final action {action}")

                # ── CEM imagination viz (rank 0 only) ────────────────────
                # Take the same starting state CEM was given (note: ``mpc_utils.cem`` does
                # ``wm_state.repeat(samples)`` internally, so the outer
                # ``wm_state`` is unchanged), roll out the chosen action
                # sequence ``means[-1]`` through ``wm.step``, decode each
                # predicted feature with the image head, and save a
                # side-by-side mp4 (one column per view) so the user can
                # see what the planner thinks the action sequence looks
                # like.
                if rank == 0 and args.cem_imagination_viz:
                    n_views = len(args.camera_id) if args.camera_id[0] is not None else 1
                    viz_dir = f"{args.output_filename}_cem_viz"
                    viz_path = os.path.join(viz_dir, f"{task_name}_ep{ep}_step{c:02d}.mp4")
                    try:
                        imagine_cem_decoded_video(
                            wm,
                            wm_state,
                            action,
                            viz_path,
                            n_views=n_views,
                            step_func=step_func,
                            denoising_steps=args.cem_imagination_denoising_steps,
                            cfg_scale=args.cem_imagination_cfg_scale,
                            fps=args.cem_imagination_fps,
                            device=device,
                        )
                        logger.info(f"[cem_imagination_viz] wrote {viz_path}")
                    except Exception as exc:
                        logger.warning(f"[cem_imagination_viz] failed at step {c}: {exc}")
                # action = (means[-1][:1, :6])[0]
                # images_ = ar_sample(wm_state.to(device), means[-1][None], T=means[-1].shape[0], cache=True)
                # mediapy.write_video('reach2.mp4', images_, fps=4)
                sub_action = (action / args.action_substeps).cpu().numpy().astype(np.float32)
                # sub_action = np.concatenate([sub_action, np.zeros(1)])
                if c >= 1:
                    d_compute_action += time.perf_counter() * 1000 - t_start_action
                exec_prefix = sub_action[: args.execution_prefix if args.execution_prefix != -1 else len(sub_action)]
                n_exec = len(exec_prefix)
                if rank == 0 and n_exec > 0:
                    actions_list = []
                    # Robocasa (franka_omron / franka_omron_state_delta) is a
                    # delta-controlled OSC environment that expects normalised
                    # delta actions in its components-dict / 7-D split form.
                    # All other embodiments (DROID, franka via deoxys
                    # OSC_POSE, SO-101, mock) are absolute-pose-controlled —
                    # the env's step() takes a 7-D
                    # [x, y, z, roll, pitch, yaw, gripper] ABSOLUTE target.
                    # That matches the pre-refactor planner behaviour
                    # (commit efff9504b changed it to send raw deltas, which
                    # silently broke every legacy env). Restore the
                    # embodiment-conditional path.
                    send_absolute = embodiment_name not in (
                        "franka_omron",
                        "franka_omron_state_delta",
                    )
                    for s_act in exec_prefix:
                        calibrated = (W @ s_act).astype(np.float32)
                        delta_arm = calibrated[:6]
                        delta_gripper = calibrated[6]

                        # Accumulate the absolute commanded pose / gripper
                        # (used for plotting / logging on every embodiment,
                        # and as the actual command for legacy envs).
                        commanded_pose[:3] += delta_arm[:3]
                        R_start = Rotation.from_euler("xyz", commanded_pose[3:6])
                        R_delta = Rotation.from_euler("xyz", delta_arm[3:6])
                        commanded_pose[3:6] = (R_delta * R_start).as_euler("xyz")
                        commanded_gripper = float(np.clip(commanded_gripper + delta_gripper, 0.0, 1.0))

                        if send_absolute:
                            # Legacy 7-D absolute-pose contract (DROID, franka
                            # deoxys, SO-101, mock). Bypass the components
                            # dict — env wrappers' is_components_dict check
                            # falls through to their legacy 7-D path which
                            # treats this as the absolute target pose.
                            abs_action = np.concatenate([commanded_pose.copy(), [commanded_gripper]]).astype(
                                np.float32
                            )
                            for _ in range(args.action_substeps):
                                actions_list.append(abs_action)
                        else:
                            # Robocasa: components dict carries the per-step
                            # delta in metric units. Each delta is repeated
                            # ``action_substeps`` times to preserve the
                            # effective control rate.
                            delta_action = np.concatenate([delta_arm, [delta_gripper]]).astype(np.float32)
                            comp = split_7d_action(delta_action)
                            for _ in range(args.action_substeps):
                                actions_list.append(comp)
                    logger.debug(f"calibration matrix {W}")
                    logger.info(
                        f"sending {len(actions_list)} actions via env.step_chunk"
                        f" (mode={'absolute' if send_absolute else 'delta'},"
                        f" embodiment={embodiment_name!r})"
                    )
                    obs, reward, done, info = env.step_chunk(actions_list)
                    log_rss(f"policy_client:after_step_chunk[ep{ep},c={c}]")
                    commanded_pose = obs["pose"][:6].copy()
                    commanded_gripper = float(obs["pose"][6])
                    chunk_obs = info["chunk_obs"]
                    chunk_infos = info.get("chunk_infos", [info] * len(chunk_obs))
                    for ci in chunk_infos:
                        step_info = {
                            k: v
                            for k, v in ci.items()
                            if k not in ("chunk_obs", "chunk_rewards", "chunk_infos")
                            and isinstance(v, (bool, int, float))
                        }
                        episode_step_infos.append(step_info)

                    # Per-tick env rewards. RoboCasaEnv populates
                    # ``chunk_rewards`` (one float per :meth:`step` call
                    # inside :meth:`step_chunk`); other envs may omit it,
                    # in which case we fall back to the top-level last
                    # reward broadcast across the chunk.
                    chunk_rewards = info.get("chunk_rewards")
                    if chunk_rewards is None:
                        # Some env servers don't compute a scalar reward
                        # (return ``reward=None`` and omit ``chunk_rewards``).
                        # Reward is only used for logging here (CEM plans with
                        # the WM objective, not the env reward), so fall back to
                        # 0.0 rather than crashing on ``float(None)``.
                        reward_val = 0.0 if reward is None else float(reward)
                        chunk_rewards = [reward_val] * len(chunk_obs)
                    for r_t in chunk_rewards:
                        episode_rewards.append(float(r_t))
                        episode_disc_return += (args.gamma**episode_step_index) * float(r_t)
                        episode_step_index += 1
                    logger.info(
                        f"[ep {ep}] step_chunk rewards: "
                        f"{[round(r, 4) for r in chunk_rewards]} "
                        f"(running sum={sum(episode_rewards):.4f}, "
                        f"disc_return(γ={args.gamma})={episode_disc_return:.4f})"
                    )

                    for chunk_ob in chunk_obs:
                        c += 1
                        stats["goal_pose_x"].append(goal["pose"][0])
                        stats["goal_pose_y"].append(goal["pose"][1])
                        stats["goal_pose_z"].append(goal["pose"][2])
                        stats["curr_pose_x"].append(chunk_ob["pose"][0])
                        stats["curr_pose_y"].append(chunk_ob["pose"][1])
                        stats["curr_pose_z"].append(chunk_ob["pose"][2])
                        stats["episode"].append(ep)
                        poses.append(chunk_ob["pose"].copy())
                        joints.append(chunk_ob["joints"].copy())
                        stats["step"].append(j)
                        j += 1
                        for img_key in img_keys:
                            raw_images[img_key].append(chunk_ob[img_key].copy())
                        goal_images = np.concatenate(
                            [F.resize(Image.fromarray(goal[img_key]), (480, 640)) for img_key in img_keys],
                            axis=0,
                        )
                        obs_images = np.concatenate(
                            [F.resize(Image.fromarray(chunk_ob[img_key]), (480, 640)) for img_key in img_keys],
                            axis=0,
                        )
                        images.append(np.concatenate([goal_images, obs_images], axis=1))

                    # Per-step env reward + cumulative trackers in the
                    # stats frame (kept aligned with the chunk_obs loop
                    # above so a row of ``stats`` corresponds to one env
                    # tick).
                    for r_t in chunk_rewards:
                        stats["env_reward"].append(float(r_t))
                        stats["env_reward_cumsum"].append(float(sum(episode_rewards[: len(stats["env_reward"])])))

                    print(f"current pose: {obs['pose'][:3].round(3)}")
                    t_encode = time.perf_counter() * 1000

                    obs_features = []
                    for img_key in img_keys:
                        image = obs[img_key]
                        h_, w_ = image.shape[:2]
                        if args.aspect_ratio is not None:
                            new_h = h_
                            new_w = int(w_ // ar)
                            image = (
                                F.resize(
                                    torch.from_numpy(image).permute(2, 0, 1),
                                    [new_h, new_w],
                                )
                                .permute(1, 2, 0)
                                .numpy()
                            )
                        if args.center_crop:
                            img = np.array(
                                F.center_crop(
                                    Image.fromarray(image),
                                    min(*goal_image.shape[:2]),
                                ).resize(image_size_pil)
                            )
                        else:
                            img = np.array(Image.fromarray(image).resize(image_size_pil))
                        obs_feature = wm.encode(
                            img[None],
                            dtype=torch.float16,
                            device=device,
                            size=args.image_size,
                        )[:, None, None]
                        obs_features.append(obs_feature)
                    # Match rank!=0 receive-buffer dtype (float32) for dist.broadcast.
                    obs_features = torch.cat(obs_features, dim=2).to(torch.float32)
                    obs_pose = torch.from_numpy(obs[proprio_obs_key]).float().to(device)
                    if c > 1:
                        d_encode_obs += time.perf_counter() * 1000 - t_encode
                    time_receive_obs = time.perf_counter() * 1000
                    last_time_send_obs = 0
                    last_time_step_begin = 0
                    last_time_step_end = 0
                    if c > 1:
                        d_robot_move += last_time_step_end - last_time_step_begin
                        d_obs_send += time_receive_obs - last_time_send_obs
                        d_image_post_proc += last_time_send_obs - last_time_step_end
                    if c % 10 == 9:
                        print(
                            f"Timings. move: {d_robot_move / (c - 1):.1f}ms; proc obs: {d_image_post_proc / (c - 1):.1f}ms; communicate: {d_obs_send / (c - 1):.1f}ms; encode obs: {d_encode_obs / (c - 1):.1f}ms; compute action: {d_compute_action / (c - 1):.1f}ms"
                        )
                else:
                    c += n_exec * args.action_substeps
                    patch_h = args.image_size[0] // 16
                    patch_w = args.image_size[1] // 16
                    obs_features = torch.zeros(
                        (
                            1,
                            1,
                            len(args.camera_id),
                            patch_h,
                            patch_w,
                            args.embed_dim,
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                    obs_pose = torch.zeros(proprio_dim, dtype=torch.float32, device=device)
                if args.save_gpu_memory and wm.encoder is not None:
                    wm.encoder.to("cpu")
                    torch.cuda.empty_cache()
                    log_gpu_mem(f"encoder→CPU after re-encoding [ep{ep},c={c}]")
                if world_size != 1:
                    dist.broadcast(obs_features, src=0)
                    dist.broadcast(obs_pose, src=0)
                features_sequence.append(obs_features.cpu())
                emb_dists.append(
                    -objective(obs_features.cpu(), goal_features.cpu()).mean(dim=(0, 1, 3, 4)).cpu().numpy()
                )
                poses_sequence.append(obs_pose.cpu())
            if rank == 0:
                if c == args.total_steps:
                    # No-op terminator: stay at the current commanded pose.
                    # For robocasa (delta-controlled) this is a zero-delta
                    # components dict; for the legacy absolute-pose envs we
                    # send the current commanded_pose so the robot stops in
                    # place rather than teleporting to the origin.
                    if embodiment_name in ("franka_omron", "franka_omron_state_delta"):
                        no_op = split_7d_action(np.zeros(7, dtype=np.float32))
                    else:
                        no_op = np.concatenate([commanded_pose.copy(), [commanded_gripper]]).astype(np.float32)
                    obs, reward, done, info = env.step(no_op)
                #     images.append(np.concatenate([
                #         np.array(Image.fromarray(goal['img_0']).resize((512, 512))),
                #         np.array(Image.fromarray(obs['img_0']).resize((512, 512))),
                #     ], axis=1))
                # for _ in range(5):
                #     final_action = np.zeros(7)
                #     obs, reward, done, info = env.step(final_action)
                #     images.append(np.concatenate([
                #         np.array(Image.fromarray(goal['img_0']).resize((512, 512))),
                #         np.array(Image.fromarray(obs['img_0']).resize((512, 512))),
                #     ], axis=1))
                images = np.stack(images)
                for k in stats:
                    global_stats[k] += stats[k]
                stats = pd.DataFrame(stats)
                emb_dists = np.stack(emb_dists)
                mpc_dists = np.stack(mpc_dists)
                buffer = process(
                    images,
                    stats,
                    start_frame=0,
                    episode=ep + 1,
                    total_eps=args.episodes,
                    goal=args.goal_name,
                    comment=args.model_name,
                    emb_dists=emb_dists,
                    mpc_dists=mpc_dists,
                    execution_prefix=args.execution_prefix,
                    l=0.15,
                )
                frames.append(buffer)
                # frames.append(np.concatenate([frames[-1][-1:]] * 5))
            if rank == 0:
                # final_action = np.zeros(7)
                # final_action[-1] = -1
                # obs, reward, done, info = env.step(final_action)
                frames = np.concatenate(frames)
                for img_key in img_keys:
                    raw_images[img_key] = np.stack(raw_images[img_key])
                poses = np.stack(poses)
                joints = np.stack(joints)
                global_stats = pd.DataFrame(global_stats)
                fold, fl = os.path.split(args.output_filename)
                os.makedirs(fold, exist_ok=True)
                save_prefix = f"{args.output_filename}_{task_name}_{ep}"
                os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
                # global_stats.to_csv(f'{args.output_filename}.csv', index=False)
                # ── Per-episode env reward / discounted-return summary ──
                episode_total_reward = float(sum(episode_rewards))
                logger.info(
                    f"[ep {ep}] DONE: n_env_steps={len(episode_rewards)} "
                    f"sum_reward={episode_total_reward:.4f} "
                    f"discounted_return(γ={args.gamma})={episode_disc_return:.4f}"
                )
                np.savez(
                    f"{save_prefix}.npz",
                    poses=poses,
                    joints=joints,
                    env_rewards=np.asarray(episode_rewards, dtype=np.float64),
                    env_total_reward=np.float64(episode_total_reward),
                    env_disc_return=np.float64(episode_disc_return),
                    env_gamma=np.float64(args.gamma),
                    **{f"env_info_{k}": np.array([si[k] for si in episode_step_infos]) for k in episode_step_infos[0]}
                    if episode_step_infos
                    else {},
                )
                mediapy.write_video(f"{save_prefix}.mp4", frames, fps=4)
                for img_key in img_keys:
                    mediapy.write_video(
                        f"{save_prefix}_{img_key}_raw.mp4",
                        raw_images[img_key],
                        fps=5,
                    )
                print(f"done episode {ep}! Continue?")
                if not args.bypass_input:
                    input()
                else:
                    print("yes!")
            dist.barrier()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
