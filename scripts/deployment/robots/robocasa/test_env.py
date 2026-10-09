# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for the WM-aligned RoboCasa environment wrapper.

Sends canonical-PandaOmron action chunks (layout A) to the env and records
a video. Useful for sanity-checking the WM↔env alignment.

Two action sources are supported:

* ``--mode random_arm``    : per WM tick, sample one 12-D action with
                              ``control_mode=-1`` and small random EE
                              deltas (other channels zero), repeat ``c``
                              times to form the chunk.
* ``--mode random_base``   : same but with ``control_mode=+1`` and small
                              random base velocities.
* ``--mode dataset``       : replay action chunks from a real dataset
                              episode (path supplied via ``--dataset_h5``).

Usage
-----
    conda activate jepa_311
    cd scripts/deployment/robots/robocasa
    python test_env.py --mode random_arm --num_ticks 30 --output test_rollout.mp4
"""

import os
import sys
from argparse import ArgumentParser

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from robot_env import (
    ACTION_PER_STEP_DIM,
    DEFAULT_CHUNK_SIZE,
    RoboCasaEnv,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _collect_frame(obs):
    img_keys = sorted(k for k in obs if k.startswith("img_"))
    if not img_keys:
        return None
    return np.concatenate([obs[k] for k in img_keys], axis=1)


def _save_video(frames, path, fps=15):
    import imageio.v3 as iio
    print(f"Saving {len(frames)} frames to {path} at {fps} fps...")
    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264")
    size_kb = os.path.getsize(path) / 1024
    print(f"Video saved: {path} ({size_kb:.1f} KB)")


def _make_arm_action(chunk_size, mag_pos=0.2, mag_rot=0.0, mag_grip=-1.0, rng=None):
    """One WM tick of arm-mode action: control_mode=-1, random small EE delta,
    constant within the chunk."""
    rng = rng or np.random.default_rng()
    a = np.zeros(ACTION_PER_STEP_DIM, dtype=np.float64)
    a[4]    = -1.0                                  # control_mode = arm
    a[5:8]  = rng.normal(scale=mag_pos, size=3)     # ee_pos
    a[8:11] = rng.normal(scale=mag_rot, size=3)     # ee_rot
    a[11]   = mag_grip                              # gripper
    return np.tile(a[None, :], (chunk_size, 1))


def _make_base_action(chunk_size, mag=0.4, rng=None):
    """One WM tick of base-mode action: control_mode=+1, random base velocity,
    constant within the chunk."""
    rng = rng or np.random.default_rng()
    a = np.zeros(ACTION_PER_STEP_DIM, dtype=np.float64)
    a[0:3] = rng.normal(scale=mag, size=3)          # base_x, base_y, base_yaw
    a[3]   = 0.0                                    # torso (training has it 0)
    a[4]   = +1.0                                   # control_mode = base
    a[11]  = -1.0                                   # gripper open
    return np.tile(a[None, :], (chunk_size, 1))


def _load_dataset_chunks(h5_path, chunk_size, num_ticks):
    """Read N consecutive action chunks from a v0.1 RoboCasa data.h5.
    Reconstructs the canonical 12-D layout A from the stored
    ``cartesian_position`` (drop pad indices 11–12) + ``gripper_position``.
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        cart = np.array(f["trajectory/action/cartesian_position"][:])  # [T, 13]
        grip = np.array(f["trajectory/action/gripper_position"][:])    # [T, 1]
    if grip.ndim == 1:
        grip = grip[:, None]
    # Drop the 2 explicit pad dims (indices 11, 12), then append gripper -> 12
    a12 = np.concatenate([cart[:, :11], grip[:, :1]], axis=1).astype(np.float64)
    needed = num_ticks * chunk_size
    if len(a12) < needed:
        raise ValueError(
            f"Dataset has only {len(a12)} raw actions; need {needed} "
            f"(num_ticks={num_ticks} * chunk_size={chunk_size})"
        )
    return a12[:needed].reshape(num_ticks, chunk_size, ACTION_PER_STEP_DIM)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="Test the WM-aligned RoboCasa env wrapper")
    parser.add_argument("--task_name", type=str, default="CloseStandMixerHead")
    parser.add_argument("--camera_names", type=str, nargs="+",
                        default=["robot0_agentview_left", "robot0_eye_in_hand"])
    parser.add_argument("--camera_width", type=int, default=256)
    parser.add_argument("--camera_height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--num_ticks", type=int, default=30,
                        help="Number of WM ticks (each = chunk_size sub-steps)")
    parser.add_argument("--mode", type=str, default="random_arm",
                        choices=["random_arm", "random_base", "dataset"])
    parser.add_argument("--dataset_h5", type=str, default=None,
                        help="Path to a v0.1 data.h5 (only used when --mode dataset)")
    parser.add_argument("--mag", type=float, default=0.2,
                        help="Random action magnitude (random modes)")
    parser.add_argument("--output", type=str, default="test_rollout.mp4")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    print(f"=== test_env: mode={args.mode}, task={args.task_name}, "
          f"chunk_size={args.chunk_size}, num_ticks={args.num_ticks} ===")
    env = RoboCasaEnv(
        task_name=args.task_name,
        camera_names=args.camera_names,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    obs, _ = env.reset()
    print(f"  Initial pose:   {np.round(obs['pose'], 4)}")
    print(f"  Initial joints: {np.round(obs['joints'], 4)}")
    print(f"  Image keys:     {sorted(k for k in obs if k.startswith('img_'))}")

    frames = []
    f0 = _collect_frame(obs)
    if f0 is not None:
        frames.append(f0)

    rng = np.random.default_rng(args.seed)

    # Build/iterate the action source
    if args.mode == "random_arm":
        action_iter = (_make_arm_action(args.chunk_size, mag_pos=args.mag, rng=rng)
                       for _ in range(args.num_ticks))
    elif args.mode == "random_base":
        action_iter = (_make_base_action(args.chunk_size, mag=args.mag, rng=rng)
                       for _ in range(args.num_ticks))
    elif args.mode == "dataset":
        if not args.dataset_h5:
            raise SystemExit("--mode dataset requires --dataset_h5 PATH")
        ticks = _load_dataset_chunks(args.dataset_h5, args.chunk_size, args.num_ticks)
        action_iter = (ticks[i] for i in range(args.num_ticks))
    else:
        raise SystemExit(f"unknown mode {args.mode}")

    for i, action_chunk in enumerate(action_iter):
        obs, reward, done, info = env.step(action_chunk)
        f = _collect_frame(obs)
        if f is not None:
            frames.append(f)
        if (i + 1) % 5 == 0 or i == args.num_ticks - 1:
            cm = info.get("control_modes", [])
            print(f"  tick {i+1:3d}/{args.num_ticks} | reward={reward:.3f} "
                  f"| success={info.get('success', False)} | cm={cm}")

    if frames:
        _save_video(frames, args.output, fps=args.fps)
    env.close()
    print("=== done ===")


if __name__ == "__main__":
    main()
