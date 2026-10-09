#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert AgiBot World (Alpha or Beta) episodes into the episode format the robo_jepa dataloader reads.

Input: an extracted copy of ``agibot-world/AgiBotWorld-Alpha`` or ``-Beta`` (Hugging Face, gated), laid out as in the
release once its tars are unpacked::

    observations/<task_id>/<episode_id>/videos/<camera>.mp4      # e.g. head_color.mp4
    proprio_stats/<task_id>/<episode_id>/proprio_stats.h5
    task_info/task_<task_id>.json                     # language annotations (optional)

Output, under ``--dst``::

    dataset_index.csv                                 # column "path": episode dirs relative to --dst
    dropped_episodes.tsv                              # episode <TAB> reason, for every episode left out
    <task_id>/episode_<episode_id>/
        data.h5
        <camera>.mp4                                  # the source video, unchanged (AV1, 640x480, 30 fps)

``data.h5``::

    cameras/<camera>                                  # "<camera>" (the loader appends ".mp4"); the release's own
                                                      # names (head_color, ...), which the configs' view_map matches
    trajectory/timestamp
    trajectory/observation/robot_state/...            # the source's state/ tree
    trajectory/action/...                             # the source's action/ tree, sparse streams filled in (below)
    trajectory/observation/robot_state/{cartesian_position [T,23], gripper_position [T,G], joint_position [T,14]}
    trajectory/action/{cartesian_position [T,18], gripper_position [T,G], joint_position [T,14]}
    trajectory/annotation/...                         # task name, scene text, per-segment action text and skill
    trajectory/provenance/{task_id, episode_id, n_frames, empty_in_source, notes}

Configs read either the source tree (``state_keys`` / ``gripper_keys``, e.g. ``end/position``) or the derived
``cartesian_position`` arrays (``pose_dim: 23``); both are written. The derived arrays follow
``app/robo_jepa/docs/action_spaces/agibot.md``: per arm, end-effector position and extrinsic-xyz euler angles
re-expressed for the commanded waist pitch, then head, base pose (state) or base velocity (action), and waist.

Only the ``head_color``, ``hand_left_color`` and ``hand_right_color`` cameras are kept (``--cameras``); fisheye,
depth and tactile streams are not. Dexterous-hand tasks have fisheye wrist cameras, so their episodes keep
``head_color`` only.

Source defects, handled as in the original conversion:
  * signals the release ships as zero-length arrays are left out (listed in ``provenance/empty_in_source``);
  * an action stream shorter than the state is a sparse recording: it is scattered back into a copy of the matching
    state signal at its ``index`` rows (``end/`` has no index of its own; ``joint/index`` is used);
  * an action stream longer than the state, or an episode without video or ``proprio_stats.h5``, is dropped.

    python scripts/data_processing/convert_agibot.py --src /data/AgiBotWorld-Alpha --dst /data/agibot --workers 32

Configs then point ``data.s3_prefix`` at ``--dst`` with ``raw_fps: 30``.
"""

import argparse
import csv
import json
import os
import shutil
from multiprocessing import Pool

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

PIVOT_X, PIVOT_Z_ABOVE_LIFT = 0.131, 0.3485  # waist-pitch pivot (agibot.md section 3): z rides the lift joint
OPTIONAL = ("state/robot/position", "state/robot/orientation", "action/robot/velocity")  # zeros when absent


def leaves(group, prefix=""):
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Group):
            yield from leaves(item, path)
        else:
            yield path, np.asarray(item)


def read_proprio(path):
    """(state, action, timestamp, n, notes, empty) from one proprio_stats.h5; raises on a corrupt episode."""
    with h5py.File(path, "r") as f:
        state = dict(leaves(f["state"])) if "state" in f else {}
        action = dict(leaves(f["action"])) if "action" in f else {}
        timestamp = np.asarray(f["timestamp"]) if "timestamp" in f else None
    empty = sorted(f"state/{p}" for p, v in state.items() if v.size == 0)
    empty += sorted(f"action/{p}" for p, v in action.items() if v.size == 0)
    state = {p: v for p, v in state.items() if v.size}
    action = {p: v for p, v in action.items() if v.size}
    if not state:
        raise ValueError("no state signals")
    n, notes = max(len(v) for v in state.values()), []
    for path, val in list(action.items()):
        if path.endswith("/index") or len(val) == n:
            continue
        if len(val) > n:
            raise ValueError(f"corrupt: action {path} has {len(val)} rows > state length {n}")
        group = path.rsplit("/", 1)[0]
        index = action.get(f"{group}/index", action.get("joint/index"))
        if index is None or index.size != len(val):
            raise ValueError(f"cannot fill sparse {path}: {len(val)} rows and no matching index")
        ref = state.get(path)
        if ref is not None and ref.shape[1:] == val.shape[1:] and len(ref) == n:
            filled = np.array(ref, dtype=np.float32, copy=True)
        else:
            filled = np.zeros((n,) + val.shape[1:], dtype=np.float32)
            notes.append(f"nostate:{path}")
        filled[index] = val
        action[path] = filled
        notes.append(f"padded:{path}({len(val)}->{n})")
    return state, action, timestamp, n, notes, empty


def derived_arrays(state, action, n):
    """The six arrays configs with pose_dim 23 read (agibot.md sections 1-3)."""
    def get(key, width):
        group, path = (state, key[len("state/"):]) if key.startswith("state/") else (action, key[len("action/"):])
        if path in group and len(group[path]) == n:
            return group[path].reshape(n, width)
        if key in OPTIONAL:
            return np.zeros((n, width))
        raise ValueError(f"missing {key}")

    waist = get("action/waist/position", 2)  # commanded waist (pitch, lift)
    pitch = waist[:, 0] - get("state/waist/position", 2)[:, 0]
    tilt = Rotation.from_euler("y", pitch[:, None])
    pivot = np.stack([np.full(n, PIVOT_X), np.zeros(n), waist[:, 1] + PIVOT_Z_ABOVE_LIFT], 1)
    get("state/end/position", 6), get("state/end/orientation", 8)  # required
    pos = [tilt.apply(state["end/position"][:, arm] - pivot) + pivot for arm in (0, 1)]
    quats = state["end/orientation"]  # [T, 2 arms, xyzw]
    euler = [(tilt * Rotation.from_quat(quats[:, arm])).as_euler("xyz") for arm in (0, 1)]
    common = [pos[0], pos[1], euler[0], euler[1], get("action/head/position", 2)]
    width = state["effector/position"].reshape(n, -1).shape[1]  # 2 (gripper) or 12 (dexterous hands)
    joints = get("state/joint/position", 14)
    return {
        "observation/robot_state/cartesian_position": np.concatenate(
            common + [get("state/robot/position", 3), get("state/robot/orientation", 4), waist], 1),
        "action/cartesian_position": np.concatenate(common + [get("action/robot/velocity", 2), waist], 1),
        "observation/robot_state/gripper_position": get("state/effector/position", width),
        "action/gripper_position": get("action/effector/position", width),
        "observation/robot_state/joint_position": joints,
        "action/joint_position": joints,
    }


def write_tree(group, tree):
    for path, arr in tree.items():
        kw = {"compression": "gzip", "compression_opts": 4} if arr.size > 1024 else {}
        group.create_dataset(path, data=arr, **kw)


def strings(values):
    return np.array([str(v) for v in values], dtype=h5py.string_dtype("utf-8"))


def convert_episode(args):
    src, dst_root, task, episode, cameras, annotation = args
    rel = f"{task}/episode_{episode}"
    videos = {c: os.path.join(src, "observations", task, episode, "videos", f"{c}.mp4") for c in cameras}
    videos = {c: p for c, p in videos.items() if os.path.isfile(p)}
    if not videos:
        return rel, "no_video"
    proprio = os.path.join(src, "proprio_stats", task, episode, "proprio_stats.h5")
    if not os.path.isfile(proprio):
        return rel, "no_proprio"
    try:
        state, action, timestamp, n, notes, empty = read_proprio(proprio)
        derived = derived_arrays(state, action, n)
    except ValueError as e:
        return rel, str(e)

    dst = os.path.join(dst_root, rel)
    os.makedirs(dst, exist_ok=True)
    for cam, path in videos.items():
        shutil.copyfile(path, os.path.join(dst, f"{cam}.mp4"))
    tmp = os.path.join(dst, "data.h5.tmp")
    with h5py.File(tmp, "w") as f:
        for cam in sorted(videos):
            f.create_dataset(f"cameras/{cam}", data=cam.encode())
        traj = f.create_group("trajectory")
        if timestamp is not None:
            traj.create_dataset("timestamp", data=timestamp)
        write_tree(traj.create_group("observation/robot_state"), state)
        write_tree(traj.require_group("action"), action)
        for path, arr in derived.items():
            traj.create_dataset(path, data=arr.astype(np.float64))
        ann = traj.create_group("annotation")
        for key in ("task_name", "init_scene_text"):
            if isinstance(annotation.get(key), str):
                ann.create_dataset(key, data=annotation[key].encode())
        segments = (annotation.get("label_info") or annotation.get("lable_info") or {}).get("action_config") or []
        if segments:
            seg = ann.create_group("action_config")
            for key, cast in (("start_frame", int), ("end_frame", int)):
                seg.create_dataset(key, data=np.array([cast(s.get(key, -1)) for s in segments], dtype=np.int64))
            for key in ("action_text", "skill"):
                seg.create_dataset(key, data=strings(s.get(key, "") for s in segments))
        prov = traj.create_group("provenance")
        prov.create_dataset("task_id", data=np.int64(task))
        prov.create_dataset("episode_id", data=np.int64(episode))
        prov.create_dataset("n_frames", data=np.int64(n))
        prov.create_dataset("empty_in_source", data=strings(empty))
        prov.create_dataset("notes", data=strings(notes))
    os.replace(tmp, os.path.join(dst, "data.h5"))
    return rel, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="extracted release root (observations/, proprio_stats/, ...)")
    parser.add_argument("--dst", required=True, help="output root")
    parser.add_argument("--cameras", default="head_color,hand_left_color,hand_right_color",
                        help="comma-separated cameras to keep (file stems under videos/)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    cameras = args.cameras.split(",")

    jobs = []
    obs_root = os.path.join(args.src, "observations")
    for task in sorted(os.listdir(obs_root)):
        annotations = {}
        info = os.path.join(args.src, "task_info", f"task_{task}.json")
        if os.path.isfile(info):
            with open(info) as f:
                annotations = {str(r["episode_id"]): r for r in json.load(f)}
        for episode in sorted(os.listdir(os.path.join(obs_root, task))):
            jobs.append((args.src, args.dst, task, episode, cameras, annotations.get(episode, {})))
    print(f"{len(jobs)} episodes under {obs_root}")

    kept, dropped = [], []
    with Pool(args.workers) as pool:
        for i, (rel, reason) in enumerate(pool.imap_unordered(convert_episode, jobs, chunksize=4)):
            (dropped.append((rel, reason)) if reason else kept.append(rel))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(jobs)} done, {len(dropped)} dropped", flush=True)

    os.makedirs(args.dst, exist_ok=True)
    with open(os.path.join(args.dst, "dataset_index.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["path"]] + [[rel] for rel in sorted(kept)])
    with open(os.path.join(args.dst, "dropped_episodes.tsv"), "w") as f:
        f.writelines(f"{rel}\t{reason}\n" for rel, reason in sorted(dropped))
    print(f"kept {len(kept)} episodes, dropped {len(dropped)}; index: {args.dst}/dataset_index.csv")


if __name__ == "__main__":
    main()
