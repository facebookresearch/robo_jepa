#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert raw DROID 1.0.1 episodes into the episode format the robo_jepa dataloader reads.

Input: the raw DROID release (``gs://gresearch/robotics/droid_raw/1.0.1``), laid out as
``<lab>/<success|failure>/<date>/<episode>/`` with ``trajectory.h5``, ``metadata_*.json`` and
``recordings/MP4/<serial>.mp4`` (SVO and stereo recordings are not needed).

Output, mirroring the input tree under ``--dst``::

    dataset_index.csv                       # column "path": episode dirs relative to --dst
    dropped_episodes.tsv                    # episode <TAB> reason, for every episode left out
    <lab>/<outcome>/<date>/<episode>/
        data.h5                             # cameras/ + trajectory/ at the root
        metadata_*.json                     # copied unchanged
        recordings/MP4/<serial>.mp4         # the source video, relabelled to 15 fps (frames untouched)

``data.h5``:
    cameras/{left,right,wrist}              # "recordings/MP4/<serial>" (the loader appends ".mp4"); roles come
                                            # from the metadata's *_mp4_path, so ext1/ext2 are NOT assumed to be
                                            # left/right (they are swapped in ~38% of episodes)
    trajectory/action/...                   # DROID's own action/ group, unchanged
    trajectory/observation/...              # DROID's own observation/ group, unchanged, plus
                                            # camera_extrinsics/{left,right,wrist} aliases of <serial>_left

An episode is dropped (and listed in dropped_episodes.tsv) when it has no MP4 video, no metadata, metadata files
that disagree, or a video whose serial the metadata does not name.

DROID records one video frame per 15 Hz control step, but the raw MP4s declare 60 fps. The dataloader derives its
frame skip and ``dt`` from the declared rate, so each video is relabelled to exactly 15 fps by stretching its
timestamps and frame durations (ffmpeg ``setts`` with ``-c copy``): no re-encoding, every frame kept bit-identical.
Resolution does not matter to the loader (it random-resize-crops every frame), so videos keep their native size.

    python scripts/data_processing/convert_droid.py --src /data/droid_raw/1.0.1 --dst /data/droid --workers 32

Configs then point ``data.s3_prefix`` at ``--dst`` and select the cameras with ``cameras: ['left|right|wrist']``.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
from fractions import Fraction
from multiprocessing import Pool

import h5py

ROLES = ("left", "right", "wrist")
MP4 = re.compile(r"^(\d+)\.mp4$")
FPS = 15  # DROID's control rate: one video frame per trajectory step


def probe(path):
    """(nominal fps, average fps, frame count) of the first video stream. The loader reads the average one."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=r_frame_rate,avg_frame_rate,nb_read_packets", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    nominal, average, frames = out.stdout.strip().split(",")
    return Fraction(nominal), Fraction(average), int(frames)


def convert_video(src, dst):
    nominal, _, frames = probe(src)
    if nominal == FPS:
        shutil.copyfile(src, dst)
        return
    # The raw MP4s are labelled 60 fps: stretch every timestamp AND frame duration by 60/15 = 4, so the declared rate
    # (and the average rate decord reports) is exactly 15. Stream copy: frames and their order are untouched.
    k = nominal / FPS
    tmp = dst + ".tmp.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-c", "copy", "-bsf:v",
         f"setts=pts=PTS*{k.numerator}/{k.denominator}:dts=DTS*{k.numerator}/{k.denominator}"
         f":duration=DURATION*{k.numerator}/{k.denominator}", tmp], check=True)
    _, average, new_frames = probe(tmp)
    if average != FPS or new_frames != frames:
        raise RuntimeError(f"relabelling {src}: {float(average):.4f} fps / {new_frames} frames, want {FPS} / {frames}")
    os.replace(tmp, dst)


def roles_from_metadata(ep_dir):
    """{role: serial} from the episode's metadata_*.json, or a drop reason."""
    named = set()
    for path in glob.glob(os.path.join(ep_dir, "metadata_*.json")):
        with open(path) as f:
            meta = json.load(f)
        named.add(tuple(os.path.basename(meta.get(f"{r}_mp4_path") or "")[: -len(".mp4")] for r in ROLES))
    if not named:
        return None, "no_metadata"
    if len(named) > 1:
        return None, "conflicting_metadata"
    return {r: s for r, s in zip(ROLES, next(iter(named))) if s}, None


def convert_episode(args):
    src_root, dst_root, rel = args
    src, dst = os.path.join(src_root, rel), os.path.join(dst_root, rel)
    mp4_dir = os.path.join(src, "recordings", "MP4")
    videos = {m.group(1) for n in (os.listdir(mp4_dir) if os.path.isdir(mp4_dir) else []) if (m := MP4.match(n))}
    if not videos:
        return rel, "no_video"
    roles, reason = roles_from_metadata(src)
    if reason:
        return rel, reason
    if not videos <= set(roles.values()):
        return rel, "metadata_mismatch"
    cameras = {r: s for r, s in roles.items() if s in videos}  # a role whose video is missing is left out

    os.makedirs(os.path.join(dst, "recordings", "MP4"), exist_ok=True)
    for serial in cameras.values():
        convert_video(os.path.join(mp4_dir, f"{serial}.mp4"), os.path.join(dst, "recordings", "MP4", f"{serial}.mp4"))
    for path in glob.glob(os.path.join(src, "metadata_*.json")):
        shutil.copyfile(path, os.path.join(dst, os.path.basename(path)))
    tmp = os.path.join(dst, "data.h5.tmp")
    with h5py.File(os.path.join(src, "trajectory.h5"), "r") as fin, h5py.File(tmp, "w") as fout:
        traj = fout.create_group("trajectory")
        for group in ("action", "observation"):
            fin.copy(fin[group], traj, name=group)
        extrinsics = traj["observation/camera_extrinsics"]
        for role, serial in cameras.items():
            if f"{serial}_left" in extrinsics:
                extrinsics.create_dataset(role, data=extrinsics[f"{serial}_left"][()])
        cams = fout.create_group("cameras")
        for role, serial in cameras.items():
            cams.create_dataset(role, data=f"recordings/MP4/{serial}".encode())
    os.replace(tmp, os.path.join(dst, "data.h5"))
    return rel, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="raw DROID 1.0.1 root (<lab>/<outcome>/<date>/<episode>/)")
    parser.add_argument("--dst", required=True, help="output root")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    episodes = sorted(
        os.path.relpath(d, args.src) for d, _, files in os.walk(args.src) if "trajectory.h5" in files)
    print(f"{len(episodes)} episodes with a trajectory.h5 under {args.src}")
    kept, dropped = [], []
    with Pool(args.workers) as pool:
        for i, (rel, reason) in enumerate(pool.imap_unordered(
                convert_episode, [(args.src, args.dst, rel) for rel in episodes], chunksize=4)):
            (dropped.append((rel, reason)) if reason else kept.append(rel))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(episodes)} done, {len(dropped)} dropped", flush=True)

    os.makedirs(args.dst, exist_ok=True)
    with open(os.path.join(args.dst, "dataset_index.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["path"]] + [[rel] for rel in sorted(kept)])
    with open(os.path.join(args.dst, "dropped_episodes.tsv"), "w") as f:
        f.writelines(f"{rel}\t{reason}\n" for rel, reason in sorted(dropped))
    reasons = {r: sum(1 for _, x in dropped if x == r) for r in sorted({x for _, x in dropped})}
    print(f"kept {len(kept)} episodes, dropped {len(dropped)} {reasons}; index: {args.dst}/dataset_index.csv")


if __name__ == "__main__":
    main()
