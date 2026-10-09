# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render per-kitchen rollout videos for any OBJECT_REACH object.

One-off diagnostic script that builds the OBJECT_REACH env with the
specified object on each of the 9 working target kitchens (including
any combo currently marked failing in `OBJECT_KITCHEN_SKIPS`) and
captures every Nth OSC sub-step's ``robot0_agentview_left`` camera
frame during ``reset(mode='init')`` into a single MP4 per kitchen.

The frame-capture works by monkey-patching ``Kitchen.step`` (at the
CLASS level, so the rebuild inside ``reset()`` inherits the patch) to
read the offscreen camera after each physics tick. This shows the
full grasp + lift + carry sequence — including failure modes where
the object is knocked over before the gripper closes.

Object is selected via ``OBJECT_NAME`` env var; defaults to ``"can"``.
Output: ``tasks_robocasa_object_reach_videos/<object>_t{K}.mp4`` for
K in {1, 2, 3, 4, 5, 6, 8, 9, 10}.

Usage::

    cd <repo_root>
    MUJOCO_GL=egl PYTHONPATH="$PWD:$PWD/scripts/deployment:$PYTHONPATH" \\
        OBJECT_NAME=mug \\
        python \\
        scripts/deployment/robots/robocasa/generate_kitchen_videos.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from robots.robocasa.blocking_robot_env import (  # noqa: E402
    _OBJECT_REACH_INNER_TASK_NAME,
    ObjectReachBlockingRoboCasaEnv,
)
from robots.robocasa.robot_env import RoboCasaEnv  # noqa: E402

TARGET_KITCHENS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 9, 10)
OBJECT_NAME: str = os.environ.get("OBJECT_NAME", "can")
SCENE_SEED: int = 0
# Wrist camera. The side ``robot0_agentview_left`` view shows the
# kitchen context but obscures the exact fingertip-vs-object alignment
# at the moment of grasp; the wrist camera makes the grasp pose
# unambiguous (and is what the user requested when iterating on the
# can/mug per-kitchen biases). Override via ``CAMERA_NAME`` env var.
CAMERA_NAME: str = os.environ.get("CAMERA_NAME", "robot0_eye_in_hand")
CAMERA_WIDTH: int = 320
CAMERA_HEIGHT: int = 240
FPS: int = 30
# Render the camera only every N sub-steps to keep file sizes
# reasonable. With N=4 and ~700 sub-steps/reset, output is ~175 frames
# ≈ 5.8 s at 30 fps per kitchen.
FRAME_STRIDE: int = 4

_THIS_DIR = Path(__file__).resolve().parent
# Repo-root-relative output dir (next to ``tasks_robocasa_object_reach``
# and ``tasks_robocasa_object_reach_previews``). ``_THIS_DIR`` is
# .../scripts/deployment/robots/robocasa/ → 4 parents up = repo root.
_OUT_DIR = _THIS_DIR.parent.parent.parent.parent / "tasks_robocasa_object_reach_videos"


def _build_env(kitchen_id: int):
    inner_env = RoboCasaEnv(
        task_name=_OBJECT_REACH_INNER_TASK_NAME,
        camera_names=[CAMERA_NAME],
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        seed=SCENE_SEED,
        robot="PandaOmron",
        init_robot_to_home=True,
        reach_task=False,
        layout_and_style_ids=[(kitchen_id, kitchen_id)],
    )
    env = ObjectReachBlockingRoboCasaEnv(
        inner_env,
        object_reach_object_name=OBJECT_NAME,
        object_reach_task_seed=0,
        object_reach_max_distance=0.15,
        object_reach_clear_radius=1.0,
        object_reach_max_retries=3,
    )
    return env


def _install_frame_capture_hook(env, frames: list[np.ndarray]) -> callable:
    """Monkey-patch ``Kitchen.step`` (CLASS level) to capture a camera
    frame after every OSC sub-step. Returns the original ``step`` so
    the caller can restore it on teardown.

    Class-level patching is required because the wrapper's
    :meth:`reset` calls ``_maybe_recreate_env(force_rebuild=True)`` which
    constructs a new kitchen instance after the patch is installed —
    instance-level patching would only affect the pre-rebuild instance
    and silently produce zero frames.

    ``frames`` is mutated in place: every Nth sub-step's RGB array is
    appended (stride = :data:`FRAME_STRIDE`).
    """
    inner_env_initial = env._env._env  # robosuite Kitchen (pre-rebuild)
    kitchen_cls = type(inner_env_initial)
    orig_step = kitchen_cls.step
    counter = {"i": 0}

    def _patched_step(self, action):
        out = orig_step(self, action)
        counter["i"] += 1
        if counter["i"] % FRAME_STRIDE == 0:
            try:
                frame = self.sim.render(
                    camera_name=CAMERA_NAME,
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    depth=False,
                )
                # robosuite renders are upside-down (origin top-left vs
                # bottom-left); flip vertically.
                frame = frame[::-1].copy()
                frames.append(frame)
            except Exception as e:
                print(
                    f"[generate_can_kitchen_videos] frame capture failed "
                    f"at substep {counter['i']}: {type(e).__name__}: {e}",
                    flush=True,
                )
        return out

    kitchen_cls.step = _patched_step
    # Return the (cls, orig_step) pair so caller can restore correctly.
    return kitchen_cls, orig_step


def _write_mp4(frames: list[np.ndarray], out_path: Path, fps: int = FPS) -> None:
    """Encode RGB frames to an MP4 via cv2. Frames must all be the same
    HxWx3 uint8."""
    import cv2

    if not frames:
        raise RuntimeError("no frames captured; aborting")
    h, w = frames[0].shape[:2]
    # mp4v is widely playable; H.264 would be preferred but isn't
    # always available in OpenCV builds. fall back to mp4v.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {out_path}")
    for f in frames:
        writer.write(f[:, :, ::-1])  # RGB -> BGR for cv2
    writer.release()


def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[generate_kitchen_videos] writing -> {_OUT_DIR}",
        flush=True,
    )
    print(
        f"[generate_kitchen_videos] kitchens = {list(TARGET_KITCHENS)} | "
        f"object={OBJECT_NAME} | camera={CAMERA_NAME} | "
        f"{CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {FPS} fps "
        f"(stride={FRAME_STRIDE})",
        flush=True,
    )

    for kitchen_id in TARGET_KITCHENS:
        t0 = time.perf_counter()
        print(
            f"[generate_kitchen_videos] === object={OBJECT_NAME} kitchen {kitchen_id} === starting",
            flush=True,
        )
        env = _build_env(kitchen_id)
        frames: list[np.ndarray] = []
        kitchen_cls, orig_step = _install_frame_capture_hook(env, frames)
        try:
            obs, _ = env.reset(mode="init", object_reach_task_seed=0)
            grasp_failed = bool(obs.get("object_reach_grasp_failed", True))
            tag = "FAIL" if grasp_failed else "OK"
            # Restore so close() doesn't double-capture.
            kitchen_cls.step = orig_step
            # Write video.
            out_path = _OUT_DIR / f"{OBJECT_NAME}_t{kitchen_id}.mp4"
            _write_mp4(frames, out_path)
            print(
                f"[generate_kitchen_videos] object={OBJECT_NAME} kitchen {kitchen_id} "
                f"done in {time.perf_counter() - t0:.1f}s — grasp={tag}, "
                f"{len(frames)} frames -> {out_path}",
                flush=True,
            )
        except Exception as e:
            # Always restore the original step before raising.
            try:
                kitchen_cls.step = orig_step
            except Exception:
                pass
            print(
                f"[generate_kitchen_videos] object={OBJECT_NAME} kitchen {kitchen_id} "
                f"FAILED after {time.perf_counter() - t0:.1f}s: {e!r}",
                flush=True,
            )
            raise
        finally:
            try:
                env.close()
            except Exception:
                pass
    print(
        f"[generate_kitchen_videos] done: {len(TARGET_KITCHENS)} videos in "
        f"{_OUT_DIR}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
