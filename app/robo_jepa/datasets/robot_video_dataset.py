# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone multi-embodiment robot video dataset.

Loads robot trajectories (video + proprioception) from S3 or local storage.
Each sample yields synchronised video frames, end-effector states, and
computed actions for one or more camera views.

Data format (v0.1)
------------------
::

    <s3_prefix>/
        dataset_index.csv              # rows: column "path" → episode dir
        aggregated_metadata.json       # per-trajectory metadata (filtering)
        <episode_dir>/
            data.h5                    # HDF5 with state + camera mapping
            <video_subdir>/video.mp4   # one or more video streams

HDF5 layout (data.h5)::

    cameras/                           # camera name → relative mp4 path
        left:  "subdir_1/video.mp4"
        right: "subdir_2/video.mp4"
    trajectory/
        observation/
            robot_state/
                cartesian_position  [T, pose_dim]    # XYZRPY
                gripper_position    [T, gripper_dim]  # 0=open, 1=closed
                joint_position      [T, joints_dim]   # NOT loaded
            camera_extrinsics/                        # optional
                <camera_name>       [T, 6]
        action/                        # Used iff action_source='action_key'
            cartesian_position  [T, 13]    # base_motion(4) + control_mode(1)
                                           # + ee_pos(3) + ee_rot(3) + pad(2)
                                           # (RoboCasa); DROID has [T, 6]
            cartesian_velocity  [T, 6]     # DROID only
            gripper_position    [T, 1]
            joint_position      [T, 7]     # DROID singular (vs observation plural)
            joint_velocity      [T, 7]     # DROID only

Actions are computed at load time according to ``action_source``:

* ``'state_delta'`` (default, legacy DROID-style): finite differences of
  consecutive ``cartesian_position`` states (rotation diffs via rotation
  matrices when ``3 < pose_dim ≤ 6``), concatenated with gripper deltas.
* ``'action_key'`` (RoboCasa & DROID): the actual recorded controller actions
  are read from ``trajectory/action`` and chunked at the raw FPS so that one
  WM step at training FPS contains the full sub-step controller trajectory.
  The exact action layout is selected by ``action_key_layout``:

  ===================  ======================  ==========  =========  ============
  action_key_layout    field (under action/)   keep dims   +gripper   per-step dim
  ===================  ======================  ==========  =========  ============
  ``'robocasa'``       cartesian_position[13]  ``[:11]``   yes        12
  ``'droid_cart'``     cartesian_position[6]   full        yes        7
  ``'droid_joint'``    joint_position[7]       full        yes        8
  ``'droid_cart_vel'`` cartesian_velocity[6]   full        yes        7
  ``'droid_joint_vel'``joint_velocity[7]       full        yes        8
  ===================  ======================  ==========  =========  ============

  With ``c = raw_fps // fps`` the per-WM-step action becomes a flat
  ``c * per_step_dim``-vector. The ``states`` tensor is filled with zeros
  in this mode (the WM is action-only).

Batch shapes
------------
Each ``__getitem__`` returns a 12-tuple:

==============  =================  ==========================================
Field           Shape              Description
==============  =================  ==========================================
buffer          [C, V, T, H, W]   Normalised video (ImageNet mean/std)
prenorm_buffer  [T, V, H, W, C]   Pre-normalisation video (raw 0–255 float)
actions         [T, pad_size]      Action diffs, zero-padded
states          [T, pad_size]      Robot states, zero-padded
extrinsics      [T, V, 6]         Camera extrinsics (NaN if absent)
padding_mask    [T]                1.0 = valid, 0.0 = padding
view_ids        [V]                Integer view IDs from ``view_map``
dt              [T]                Time delta per frame (seconds)
start_frame     scalar             Start index in the raw video
path            str                Episode dir relative to ``s3_prefix``
embodiment_name str                Robot embodiment identifier
index           int                Sample index in the dataset
==============  =================  ==========================================

``buffer`` is the encoder input (channels-first, normalised).
``prenorm_buffer`` is the reconstruction target for generative decoders
(channels-last, unnormalised).

Temporal alignment (``read_intermediate_frames=True``)
------------------------------------------------------
**Why this exists.** Consider an action-conditioned world model that predicts
V-JEPA-level features at a low frame rate (e.g. 5 fps).  The encoder sees
``buffer`` — a sparse sequence of key frames.  On top of that, we want a DiT
diffusion decoder that takes those latent features and decodes them into a
*high frame rate* video (e.g. 20–40 fps).  The decoder needs ground-truth
high-fps frames as reconstruction targets — that is ``prenorm_buffer``.

So the two tensors serve complementary roles:

- ``buffer``  (low fps)  → input to the V-JEPA encoder / world model.
- ``prenorm_buffer`` (high fps) → ground-truth target for the diffusion decoder.

In practice, ``mul`` is chosen to match the **temporal compression ratio of the
video diffusion tokeniser** (e.g. Cosmos Tokenizer).  A tokeniser that maps
``T`` input frames to ``T / mul + 1`` latent tokens requires exactly this
alignment so that each buffer key frame corresponds to one latent token.
Common values are ``mul = 4`` or ``mul = 8``.

**How it works.**

1. All frames are loaded at ``intermediate_frames_fps`` spacing.  The raw-frame
   step is ``i_fstp = ceil(raw_fps / intermediate_frames_fps)``, producing
   ``fpc * mul + 1`` frames (the ``+1`` is the extra prediction-target tail
   frame).

2. Both ``buffer`` and ``prenorm_buffer`` start as the same tensor containing
   all ``fpc * mul + 1`` loaded frames (after the spatial transform).

3. ``buffer = buffer[:, :, ::skip]`` (where ``skip = mul = fstp // i_fstp``)
   keeps only every ``mul``-th frame → ``fpc + 1`` key frames.
   ``prenorm_buffer`` is **not** subsampled — it retains all
   ``fpc * mul + 1`` frames.

4. Because ``mul`` equals the tokeniser's temporal compression, the diffusion
   decoder receives ``fpc * mul + 1`` prenorm frames and produces ``fpc + 1``
   latent tokens — exactly one per buffer key frame.

**Shape invariant** (checked at runtime)::

    buffer.shape[T]  == prenorm_buffer.shape[T] // mul + 1
    (fpc + 1)        == (fpc * mul + 1)         // mul + 1

**Concrete example** (``raw_fps=60, fps=5, intermediate_frames_fps=20,
frames_per_clip=16``)::

    fstp   = 60 / 5  = 12     (raw-frame step for key frames)
    i_fstp = 60 / 20 = 3      (raw-frame step for intermediate frames)
    mul    = 12 / 3  = 4      (= tokeniser temporal compression ratio)

    Loaded frames: fpc * mul + 1 = 16 * 4 + 1 = 65 at intermediate_frames_fps

    prenorm_buffer (all 65 frames at 20 fps):

     pos:  0  1  2  3  4  5  6  7  8  ...  60 61 62 63 64
          [■][□][□][□][■][□][□][□][■] ... [■][□][□][□][■]
           ↑           ↑           ↑        ↑           ↑
           buffer (17 key frames = every 4th position)

     ■ = key frame  (in BOTH buffer and prenorm_buffer)
     □ = intermediate frame (ONLY in prenorm_buffer)

    buffer.shape[T]         = 17  (fpc + 1)
    prenorm_buffer.shape[T] = 65  (fpc * mul + 1)
    Invariant: 17 == 65 // 4 + 1  ✓

The ``+1`` tail frame exists in both tensors as the final prediction target.
``actions`` and ``states`` have length ``fpc`` (16), aligning with the first 16
key frames in ``buffer`` (the 17th frame is the prediction target, not paired
with an action).  The tokeniser compresses the 65 prenorm frames into 17 latent
tokens — one per buffer key frame — so the encoder and the decoder operate in
the same temporal grid.

Camera selection
----------------
=================================  ==========================================
cameras argument                   Behaviour
=================================  ==========================================
``['left|right']``                 Single-view: randomly pick left or right
``['left', 'right']``              Two-view: left first, right second
``['left|right', 'left|right']``   Two-view: random without replacement
``['.*']``                         Single-view: any available camera
=================================  ==========================================
"""

import concurrent.futures
import importlib
import json
import os
import re
import shutil
import tempfile
from logging import getLogger
from math import ceil

import einops
import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from filelock import FileLock
from scipy.spatial.transform import Rotation

from src.utils.file_system_handler import FileSystemHandler

logger = getLogger()
MAX_RETRIES = 40


# Per-layout descriptor table for action_source='action_key'.
# Maps layout-name -> (action field name, keep_dims, gripper field name, per_step_dim).
#   * action field is the key under trajectory/action/ to read.
#   * keep_dims is None to use the full last dimension of the action field, or
#     an int N to take only the first N dims (used by RoboCasa to drop the
#     2 explicit pad dims at indices 11/12 of the 13-D cartesian_position).
#   * gripper field is appended to give the per-step action vector.
#   * per_step_dim is the resulting (action + gripper) per-step width and
#     drives the pad_size validator (need pad_size >= action_chunk_size *
#     per_step_dim).
ACTION_KEY_LAYOUTS = {
    "robocasa": ("cartesian_position", 11, "gripper_position", 12),
    "droid_cart": ("cartesian_position", None, "gripper_position", 7),
    "droid_joint": ("joint_position", None, "gripper_position", 8),
    "droid_cart_vel": ("cartesian_velocity", None, "gripper_position", 7),
    "droid_joint_vel": ("joint_velocity", None, "gripper_position", 8),
}


class ConfigurationError(Exception):
    pass


class RobotVideoDataset(torch.utils.data.Dataset):
    """A dataset that loads robot trajectories (video + proprioception) from S3/local storage.

    All data is stored in the v0.1 format. The expected layout on S3 (or local fs)::

        <s3_prefix>/
            dataset_index.csv              # rows of episode paths (column: "path")
            aggregated_metadata.json       # per-trajectory metadata (used by filter funcs)
            <episode_dir>/
                data.h5
                <video_subdir>/video.mp4
                ...

    Expected HDF5 structure per episode (v0.1 format, data.h5)::

        cameras/                           # maps informal camera names → relative mp4 paths
            left:  "subdir_1/video.mp4"
            right: "subdir_2/video.mp4"
            ...
        trajectory/
            observation/
                robot_state/
                    cartesian_position:  float32 [T, pose_dim]  # end-effector XYZRPY
                    gripper_position:    float32 [T, gripper_dim] # 0=open, 1=closed
                    joint_position:      float32 [T, joints_dim]  # NOT loaded by this dataloader
                camera_extrinsics/     # [OPTIONAL]
                    <camera_name>:       float32 [T, 6]  # XYZRPY per camera
            action/                    # Used iff action_source='action_key'
                cartesian_position:    float32 [T, 13]
                gripper_position:      float32 [T, 1]
                joint_position:        float32 [T, 7]

    Actions are NOT read from the HDF5 by default. Instead they are computed at
    load time as finite differences of consecutive ``cartesian_position``
    (with rotation diffs via rotation matrices for 3 < pose_dim <= 6)
    concatenated with gripper deltas. To use the actual recorded controller
    actions instead, set ``action_source='action_key'`` (RoboCasa-style; the
    ``states`` tensor will then be all zeros and ``actions`` will be the
    canonical 12-D PandaOmron action chunked at ``c = raw_fps // fps``).

    Each ``__getitem__`` returns a tuple of::

        (buffer,           # [C, V, T, H, W]  normalized video frames
         prenorm_buffer,   # [T, V, H, W, C]  pre-normalization video frames
         actions,          # [T, pad_size]     action diffs (zero-padded if pad_size set)
         states,           # [T, pad_size]     robot states (zero-padded if pad_size set)
         extrinsics,       # [T, V, 6]        camera extrinsics (NaN if unavailable)
         padding_mask,     # [T]              1.0 for valid frames, 0.0 for padding
         view_ids,         # [V]              integer view IDs from view_map
         dt,               # [T]              time delta per frame in seconds
         start_frame,      # scalar            starting frame index in the raw video
         path,             # str               episode path relative to s3_prefix
         embodiment_name,  # str               robot embodiment identifier
         index)            # int               sample index in the dataset

    Args:
        data_path (str, optional): Unused legacy argument kept for API compatibility
            with init_datasets. Default: None.
        cameras (list[str]): List of regex patterns used to match camera names in the
            ``cameras/`` group of each episode's HDF5 file. For each pattern one
            camera is randomly sampled (without replacement) from the matches.
            Examples:
              - ['left|right']              → single-view: randomly picks either "left"
                                              or "right" for each sample
              - ['left', 'right']           → two-view: first view is always "left",
                                              second view is always "right"
              - ['left|right', 'left|right'] → two-view: each view randomly picks from
                                              {"left","right"} without replacement, so
                                              you get either (left, right) or (right, left)
              - ['.*']                       → single-view: sample any one camera
        view_map (dict[str, int], optional): Maps camera-name regex patterns to integer
            view IDs used downstream by the model (e.g. for learned view embeddings).
            Example: {'left': 0, 'right': 1, 'top': 2}. If None every view gets ID 0.
        dataset_name (str): Human-readable name for the dataset, used in error messages
            and logging (e.g. 'Droid', 'Roboset').
        raw_fps (int): Native frame rate of the source mp4 videos. Used to compute the
            frame-skip factor. Must be >= ``fps``. Default: 60.
        frames_per_clip (int): Number of frames (at the target ``fps``) to sample per
            clip. This determines the temporal length of each training example.
            Default: 16.
        fps (int): Target frame rate for training. Frames are sub-sampled from the
            raw video by skipping every ``ceil(raw_fps / fps)`` frames. Default: 5.
        min_fps (int, optional): When set, enables stochastic FPS augmentation.
            A frame-skip factor is uniformly sampled such that the effective FPS lies
            in [min_fps, fps]. If None, a fixed skip factor is used. Default: None.
        min_length_to_drop (int): Minimum number of valid frames (at target fps)
            required for a clip. Episodes shorter than this are skipped.
            Default: 16.
        read_intermediate_frames (bool): If True, loads additional intermediate frames
            between the key-frames (at ``intermediate_frames_fps``). Used for training
            generative decoders (e.g. DiT) that operate at a higher temporal resolution.
            Default: False.
        intermediate_frames_fps (int): FPS at which intermediate frames are sampled
            when ``read_intermediate_frames=True``. Default: 30.
        transform (callable, optional): A video transform applied per-view to each
            loaded frame buffer. Should accept a numpy array of shape [T, H, W, C]
            and return a tuple (normalized_buffer, prenorm_buffer). If None, raw
            frames are returned. Default: None.
        s3_profile (str, optional): AWS CLI profile name used to authenticate with S3.
            Passed to the FileSystemHandler. Set to None for local-only access.
            Default: None.
        s3_prefix (str): Full S3 (or local) path prefix to the dataset root where
            ``dataset_index.csv``, ``aggregated_metadata.json``, and all episode
            sub-directories reside.
            Example: 's3://<bucket>/data/robotics/droid/1.0.1/'.
        debug_log (bool): If True, emits verbose per-element loading logs at INFO
            level instead of DEBUG. Useful for debugging data loading issues.
            Default: False.
        min_size (int, optional): If set, artificially inflates ``__len__`` to at
            least this value (indices wrap around). Useful to ensure each GPU gets
            enough samples during distributed evaluation. Default: None.
        n_arms (int): Number of robot arms. Affects how end-effector poses are
            decomposed into position + rotation diffs. Default: 1.
        pose_dim (int): Dimensionality of the ``cartesian_position`` vector per arm.
            For standard 6-DoF XYZRPY, set to 6; rotation diffs are then computed
            via rotation matrices. For pose_dim > 6 or < 3, a simple element-wise
            difference is used instead. Default: 6.
        gripper_dim (int): Dimensionality of the ``gripper_position`` vector per arm.
            Typically 1 (scalar open/close where 0=open, 1=closed). When 0, gripper
            data is excluded from the action/state vectors. Default: 1.
        joints_dim (int): Number of joint dimensions in ``joint_position``. Stored as
            metadata only — joint positions are NOT loaded by the dataloader.
            Default: 7.
        pad_size (int, optional): If set, actions and states are zero-padded along
            the feature dimension to this size. This enables batching across datasets
            with different proprioceptive dimensions (e.g. 7-DoF franka vs 25-DoF 1X).
            Default: None (no padding).
        filter_samples_func (callable or str, optional): A function (or dotted import
            path string like 'droid.filter_positive_negative_droid') used to filter
            the dataset_index.csv rows at init time. The function signature must be:
            ``func(samples: pd.DataFrame, fs_handler, s3_prefix, **kwargs) -> pd.DataFrame``.
            Default: None (no filtering).
        filter_samples_func_kwargs (dict, optional): Extra keyword arguments forwarded
            to ``filter_samples_func``. Default: None.
        embodiment_name (str): Identifier for the robot embodiment (e.g. 'franka',
            'kuka', 'so-101'). Returned per-sample so the model can condition on it.
            Must not be None.
        use_joint_state (bool): If True, use raw joint positions instead of
            ``cartesian_position`` as the state vector (and compute "actions"
            as element-wise diffs). Default: False.
        state_keys (list[str], optional): Datasets whose robot_state has no
            ``cartesian_position`` (e.g. AgiBot) list the keys to concatenate instead,
            relative to ``trajectory/observation/robot_state`` (e.g. ``['end/position',
            'end/orientation']``). Each is flattened to ``[T, -1]``; actions are the usual
            state deltas. Default: None (use ``cartesian_position``).
        gripper_keys (list[str], optional): Keys appended after *state_keys* to act as the
            gripper part of the state (e.g. ``['effector/position']``). Default: None.
        action_source (str): Where to source the action vector from. Either
            ``'state_delta'`` (default, legacy: actions are diffs of state) or
            ``'action_key'`` (RoboCasa: read the recorded controller actions
            from ``trajectory/action`` and chunk them at the raw FPS). When
            ``'action_key'`` is set: (a) ``min_fps`` must be unset or equal to
            ``fps`` (variable FPS is unsupported and will raise); (b)
            ``raw_fps`` must be divisible by ``fps``; (c) ``pad_size`` must be
            at least ``(raw_fps // fps) * 12``; (d) the ``states`` returned
            from ``__getitem__`` is all zeros. Default: ``'state_delta'``.
    """

    def __init__(
        self,
        data_path=None,
        cameras=None,
        view_map=None,
        dataset_name=None,
        raw_fps=60,
        frames_per_clip=16,
        fps=5,
        min_fps=None,
        min_length_to_drop=16,
        read_intermediate_frames=False,
        intermediate_frames_fps=30,
        transform=None,
        s3_profile=None,
        s3_prefix=None,
        debug_log=False,
        min_size=None,
        n_arms=1,
        pose_dim=6,
        gripper_dim=1,
        joints_dim=7,
        pad_size=None,
        filter_samples_func=None,
        filter_samples_func_kwargs=None,
        embodiment_name=None,
        use_joint_state=False,
        action_source="state_delta",
        action_key_layout="robocasa",
        state_keys=None,
        gripper_keys=None,
    ):
        self.s3_prefix = s3_prefix
        self.embodiment_name = embodiment_name
        self.frames_per_clip = frames_per_clip
        self.intermediate_frames_fps = intermediate_frames_fps
        self.min_length_to_drop = min_length_to_drop
        self.min_size = min_size
        self.fps = fps
        self.raw_fps = raw_fps
        self.view_map = view_map
        self.pad_size = pad_size
        self.dataset_name = dataset_name
        self.state_keys = state_keys
        self.gripper_keys = gripper_keys
        self.use_joint_state = use_joint_state
        if action_source not in ("state_delta", "action_key"):
            raise ValueError(f"action_source must be 'state_delta' or 'action_key', got {action_source!r}")
        self.action_source = action_source
        if self.action_source == "action_key":
            # Action chunking requires a fixed FPS so that every sample has
            # the same chunk size c = raw_fps // fps. Variable FPS would
            # produce a variable per-sample action width which cannot be
            # batched. We assert this up-front (min_fps must be unset or
            # equal to fps), and we also assert that fps divides raw_fps.
            # Note: fps may be a float (e.g. 7.5 with raw_fps=60) — the
            # divisibility check uses float modulo and the chunk size is
            # cast back to int below.
            if action_key_layout not in ACTION_KEY_LAYOUTS:
                raise ValueError(
                    f"action_key_layout must be one of {sorted(ACTION_KEY_LAYOUTS)}, got {action_key_layout!r}"
                )
            self.action_key_layout = action_key_layout
            (
                self.action_key_field,
                self.action_key_keep_dims,
                self.action_key_gripper_field,
                self.action_per_step_dim,
            ) = ACTION_KEY_LAYOUTS[action_key_layout]
            if min_fps is not None and min_fps != fps:
                raise ValueError(
                    f"action_source='action_key' requires fixed fps "
                    f"(min_fps must be None or == fps). "
                    f"Got min_fps={min_fps}, fps={fps}."
                )
            if (raw_fps % fps) != 0:
                raise ValueError(
                    f"action_source='action_key' requires raw_fps to be "
                    f"divisible by fps. Got raw_fps={raw_fps}, fps={fps}."
                )
            # int() handles the float-fps case (e.g. 60 // 7.5 == 8.0).
            self.action_chunk_size = int(raw_fps // fps)
            required_pad = self.action_chunk_size * self.action_per_step_dim
            if pad_size is None or pad_size < required_pad:
                raise ValueError(
                    f"action_source='action_key' (layout={action_key_layout!r}) "
                    f"needs pad_size >= action_chunk_size * per_step_dim = "
                    f"{self.action_chunk_size} * {self.action_per_step_dim} = "
                    f"{required_pad}, got pad_size={pad_size}."
                )
        else:
            self.action_chunk_size = None
            self.action_per_step_dim = None
            self.action_key_layout = None
            self.action_key_field = None
            self.action_key_keep_dims = None
            self.action_key_gripper_field = None
        assert fps <= self.raw_fps, (
            f"requested to train at {fps} fps but in {self.dataset_name} videos are {self.raw_fps} fps."
        )
        assert embodiment_name is not None
        self.min_fps = min_fps
        self.n_arms = n_arms
        self.pose_dim = pose_dim
        self.joints_dim = joints_dim
        self.gripper_dim = gripper_dim
        self.s3_profile = s3_profile
        self.transform = transform
        self.read_intermediate_frames = read_intermediate_frames
        self.h5_reader = None
        self.debug_log = debug_log
        self.cameras = cameras  # cameras is the list of regex
        self.fs_handler = FileSystemHandler(s3_profile=self.s3_profile)
        with self.fs_handler.open(os.path.join(self.s3_prefix, "dataset_index.csv"), "r") as csv_file:
            samples = pd.read_csv(csv_file)
        if filter_samples_func is not None:
            if isinstance(filter_samples_func, str):
                mod_name, func_name = filter_samples_func.rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                filter_samples_func = getattr(mod, func_name)
            assert callable(filter_samples_func), "filter_samples_func be callable"
        if filter_samples_func_kwargs is None:
            filter_samples_func_kwargs = {}
        if callable(filter_samples_func):
            self.samples = filter_samples_func(
                samples, fs_handler=self.fs_handler, s3_prefix=self.s3_prefix, **filter_samples_func_kwargs
            )
        else:
            self.samples = samples
        self.indices = np.arange(len(self.samples))
        self.total_indices = len(self.indices)

    def filter_samples(self, samples):
        return samples

    def next_index(self, index, view_no=None):
        return (index + 1) % self.total_indices

    def run_with_timeout(self, func, args=(), kwargs=None, timeout=60):
        if self.h5_reader is None:
            self.h5_reader = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        if kwargs is None:
            kwargs = {}
        future = self.h5_reader.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)

    def __getitem__(self, index):
        if self.min_size is not None:
            index = index % len(self.indices)
        log_func = logger.info if self.debug_log else logger.debug
        index = self.indices[index]
        # index = 58533
        path = self.samples.iloc[index]
        winfo = torch.utils.data.get_worker_info()
        wid = winfo.id if winfo is not None else 0
        # -- keep trying to load videos until you find a valid sample
        loaded_video = False
        attempts = 0
        while not loaded_video:
            try:
                log_func(f"[worker id {wid}, index {index}]reading element {path.path}")
                trajectory_id = path.path[5:].replace("/", ".")
                lock = FileLock(f"/tmp/{trajectory_id[-200:]:}.lock")  # the slice is to avoid too long filenames
                log_func(f"[worker id {wid}, index {index}]acquiring lock file: /tmp/{trajectory_id}.lock")
                with lock:
                    ret = self.load_element(path, index)
                    log_func(f"[worker id {wid}, index {index}]releasing lock file: /tmp/{trajectory_id}.lock")
                log_func(f"[worker id {wid}, index {index}]finished reading element {path.path}")
                loaded_video = True
            # except ConfigurationError as:
            #     # raise
            #     logger.info(
            #         f"[worker id {wid}, index {index}]Ecountered exception when loading video {path.path} {e=}"
            #     )
            except Exception as e:
                if isinstance(e, TimeoutError):
                    del self.fs_handler
                    self.fs_handler = FileSystemHandler(s3_profile=self.s3_profile)
                logger.info(
                    f"[worker id {wid}, index {index}]Ecountered exception when loading video {path.path} {e=}"
                )
                attempts += 1
                if attempts >= MAX_RETRIES:
                    raise RuntimeError(f"Failed retrieving item after {attempts=}")
                loaded_video = False
                index = np.random.randint(self.__len__())
                if self.min_size is not None:
                    index = index % len(self.indices)
                index = self.indices[index]
                path = self.samples.iloc[index]

        return *ret, index

    def get_json(self, directory):
        fs = self.fs_handler.get_from_path(directory)
        for filename in fs.listdir(directory, detail=False):
            # for filename in os.listdir(directory):
            if filename.endswith(".json"):
                try:
                    with fs.open(filename, "r") as f:
                        # with open(os.path.join(directory,filename), "r") as f:
                        return filename, json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
                except Exception as e:
                    print(f"An unexpected error occurred while processing {filename}: {e}")

    def poses_to_diffs(self, poses):
        # TODO: change this to handle dual arm systems
        # return poses[1:] - poses[:-1]
        if self.pose_dim > 6 or self.pose_dim < 3:
            return poses[1:] - poses[:-1]
        poses_arms = []
        # import pdb; pdb.set_trace()
        for arm in range(self.n_arms):
            xyz = poses[:, arm * 6 : arm * 6 + 3]  # shape [T, 3]
            xyz_diff = xyz[1:] - xyz[:-1]
            if poses.shape[1] > (arm * 6) and self.pose_dim > 3:
                thetas = poses[:, arm * 6 + 3 : (arm + 1) * 6]  # euler angles, shape [T, 3]
                matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
                angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
                angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
                angle_diff = np.stack([d for d in angle_diff], axis=0)
                poses_diff = np.concatenate([xyz_diff, angle_diff], axis=1)
            else:
                poses_diff = xyz_diff
            poses_arms.append(poses_diff)
        poses_diff = np.concatenate(poses_arms, axis=1)
        if self.gripper_dim > 0:
            closedness = poses[:, -(self.gripper_dim * self.n_arms) :]
            closedness_delta = closedness[1:] - closedness[:-1]
            return np.concatenate([poses_diff, closedness_delta], axis=1)
        else:
            return poses_diff

    def load_element(self, path, index):
        log_func = logger.info if self.debug_log else logger.debug
        winfo = torch.utils.data.get_worker_info()
        wid = winfo.id if winfo is not None else 0

        # -- load trajectory info

        tpath = os.path.join(self.s3_prefix, path.path, "data.h5")
        log_func(f"[worker id {wid}, index {index}]open file for h5")
        # with opener(tpath, 'rb') as f:
        with tempfile.NamedTemporaryFile() as temp_file:
            with self.fs_handler.open(tpath, "rb") as f_src, open(temp_file.name, "wb") as f_tmp:
                shutil.copyfileobj(f_src, f_tmp)
            log_func(f"[worker id {wid}, index {index}]start open h5")
            data = h5py.File(temp_file.name)
            log_func(f"[worker id {wid}, index {index}]finish open h5")
            trajectory = data["trajectory"]
            camera_mapping = data["cameras"]

            # -- randomly sample a camera view

            view_list = {}
            cam_names = list(camera_mapping.keys())
            if len(cam_names) == 0:
                raise ValueError(f"no videos found for episode: {path.path}")

            for camera_match in self.cameras:
                curr_match = [
                    re.match(camera_match, c).group() for c in cam_names if re.match(camera_match, c) is not None
                ]
                if len(curr_match) == 0:
                    raise ConfigurationError(f'Unable to match any camera name with expression: "{camera_match}"')
                curr_camera = np.random.choice(curr_match)
                cam_names.remove(curr_camera.item())  # to sample without replacement
                mp4_name = np.array(camera_mapping[curr_camera]).item().decode()
                # cameras/<name> may store the video path without its extension (file = value + ".mp4")
                view_list[curr_camera.item()] = mp4_name if mp4_name.endswith(".mp4") else mp4_name + ".mp4"

            if self.use_joint_state:
                # Naming-convention drift across datasets: DROID stores
                # ``joint_positions`` (plural) under observation/robot_state,
                # while LeRobot (and most others) use ``joint_position``
                # (singular). Try the plural first (legacy), then fall back
                # to singular. The action group's ``joint_position`` is
                # always singular across all datasets.
                rs = trajectory["observation"]["robot_state"]
                if "joint_positions" in rs:
                    joint_pos = np.array(rs["joint_positions"])
                elif "joint_position" in rs:
                    joint_pos = np.array(rs["joint_position"])
                else:
                    raise KeyError(
                        f"observation/robot_state has neither 'joint_positions' "
                        f"nor 'joint_position' (use_joint_state=True). "
                        f"Available keys: {list(rs.keys())}"
                    )
                gripper_pos = np.array(rs["gripper_position"])
                if len(gripper_pos.shape) == 1:
                    gripper_pos = gripper_pos[:, None]
                if self.gripper_dim != 0 and gripper_pos.shape[1] != 0:
                    states = np.concatenate([joint_pos, gripper_pos], axis=1)  # [T, joints_dim + 1]
                else:
                    states = joint_pos
            elif self.state_keys is not None:
                # Datasets that ship their own state tree (e.g. AgiBot: end/position,
                # end/orientation, ...) instead of a single cartesian_position. Each key is
                # flattened to [T, -1] and concatenated, gripper keys last.
                rs = trajectory["observation"]["robot_state"]
                parts = [np.array(rs[k]) for k in list(self.state_keys) + list(self.gripper_keys or [])]
                states = np.concatenate([p.reshape(p.shape[0], -1) for p in parts], axis=1)
            else:
                cartesian_pos = np.array(trajectory["observation"]["robot_state"]["cartesian_position"])
                gripper_pos = np.array(trajectory["observation"]["robot_state"]["gripper_position"])
                if len(gripper_pos.shape) == 1:
                    gripper_pos = gripper_pos[:, None]
                if self.gripper_dim != 0 and gripper_pos.shape[1] != 0:
                    states = np.concatenate([cartesian_pos, gripper_pos], axis=1)  # [T, 7]
                else:
                    states = cartesian_pos
            buffers = []
            prenorm_buffers = []
            view_names = []
            extrinsics_buffers = []
            start_frame, end_frame, end_frame_state, indices, state_indices = None, None, None, None, None
            for view_name, mp4_name in view_list.items():
                vpath = os.path.join(self.s3_prefix, path.path, mp4_name)
                log_func(f"[worker id {wid}, index {index}]open video file")
                with self.fs_handler.open(vpath, "rb") as video_file:
                    log_func(f"[worker id {wid}, index {index}]create video FD")
                    # num_threads=0 is decord's "auto"; -1 is passed through to the codec and breaks dav1d (AV1)
                    vr = VideoReader(video_file, num_threads=0, ctx=cpu(0))
                    log_func(f"[worker id {wid}, index {index}]finish create video FD")
                    # --
                    vfps = vr.get_avg_fps()
                    if vfps < (self.raw_fps // 2):
                        raise ValueError(
                            f"in {self.dataset_name} fps should be {self.raw_fps}! Found a video with {vfps} FPS. path={vpath}"  # noqa: E501
                        )
                    fpc = self.frames_per_clip
                    if self.min_fps is not None:
                        frame_skip_range = [
                            frame_skip
                            for frame_skip in range(1, int(ceil(vfps)) + 1)
                            if (self.min_fps * frame_skip) <= vfps <= (self.fps * frame_skip)
                        ]
                        current_frame_skip = np.random.choice(frame_skip_range)
                        fps = vfps / current_frame_skip
                        dt = current_frame_skip / vfps
                    else:
                        fps = self.fps if self.fps is not None else vfps
                        current_frame_skip = ceil(vfps / fps)
                        dt = current_frame_skip / vfps
                    fstp = current_frame_skip
                    i_fstp = ceil(vfps / self.intermediate_frames_fps)

                    i_skip_rate = self.intermediate_frames_fps / fps
                    nframes = int(fpc * fstp)
                    if self.read_intermediate_frames:
                        nframes += i_fstp  # to add one extra tail frame if we are training DiT
                    vlen = len(vr)
                    states_len = states.shape[0]

                    min_frames = int(self.min_length_to_drop * fstp)
                    if vlen < min_frames:
                        raise ValueError(f"Video is too short {vpath=}, {nframes=}, {min_frames=}, {vlen=}")

                    # sample a random window of nframes
                    if start_frame is None:
                        start_frame = np.random.randint(0, max(vlen - int((fpc + 1) * fstp), fstp))
                        end_frame = min(start_frame + nframes, vlen)
                        end_frame_state = min(start_frame + int((fpc + 1) * fstp), states_len)
                        indices = np.arange(
                            start_frame, end_frame, fstp if not self.read_intermediate_frames else i_fstp
                        ).astype(np.int64)
                        state_indices = np.arange(start_frame, end_frame_state, fstp).astype(np.int64)
                        states = states[state_indices, :]
                        # extrinsics = extrinsics[state_indices, :]
                        if len(states) < 2:
                            raise ValueError(f"Video is too short {vpath=}, {nframes=}, {vlen=}")
                        if self.action_source == "action_key":
                            # --- Action-key path (RoboCasa, DROID, ...) -----------------
                            # We assume a fixed FPS (verified at __init__).  Every
                            # WM step covers a chunk of `c = action_chunk_size`
                            # raw-rate actions.  The exact action field, optional
                            # keep-slice, and per-step dim are configured by
                            # `action_key_layout` (see ACTION_KEY_LAYOUTS).
                            c = self.action_chunk_size
                            if int(fstp) != c:
                                raise ValueError(
                                    f"action_source='action_key' expects fixed "
                                    f"frame_skip == action_chunk_size, got "
                                    f"fstp={fstp} vs c={c} for {vpath=}"
                                )
                            if "action" not in trajectory:
                                raise ValueError(f"trajectory has no 'action' group: {tpath}")
                            field = self.action_key_field
                            grip_field = self.action_key_gripper_field
                            keep_dims = self.action_key_keep_dims
                            if field not in trajectory["action"]:
                                raise ValueError(
                                    f"trajectory['action'] has no '{field}' key "
                                    f"(action_key_layout={self.action_key_layout!r}): {tpath}"
                                )
                            if grip_field not in trajectory["action"]:
                                raise ValueError(
                                    f"trajectory['action'] has no '{grip_field}' key "
                                    f"(action_key_layout={self.action_key_layout!r}): {tpath}"
                                )
                            raw_act_arm = np.array(trajectory["action"][field])  # [T_raw, D_arm]
                            raw_act_grip = np.array(trajectory["action"][grip_field])  # [T_raw] or [T_raw, 1]
                            if raw_act_grip.ndim == 1:
                                raw_act_grip = raw_act_grip[:, None]
                            if keep_dims is not None:
                                if raw_act_arm.shape[1] < keep_dims:
                                    raise ValueError(
                                        f"action.{field} has only {raw_act_arm.shape[1]} dims "
                                        f"(need >= keep_dims={keep_dims} for layout "
                                        f"{self.action_key_layout!r}): {tpath}"
                                    )
                                raw_act_arm = raw_act_arm[:, :keep_dims]
                            raw_act = np.concatenate([raw_act_arm, raw_act_grip[:, :1]], axis=1).astype(
                                np.float32
                            )  # [T_raw, per_step_dim]
                            if raw_act.shape[1] != self.action_per_step_dim:
                                raise ValueError(
                                    f"action_key_layout={self.action_key_layout!r}: "
                                    f"expected per_step_dim={self.action_per_step_dim} "
                                    f"(field={field}, keep_dims={keep_dims}, "
                                    f"gripper={grip_field}), got "
                                    f"{raw_act.shape[1]} for {tpath}"
                                )

                            n_chunks = len(state_indices) - 1  # one chunk per inter-frame interval
                            needed = start_frame + n_chunks * c
                            if raw_act.shape[0] < needed:
                                raise ValueError(
                                    f"Not enough actions in {tpath}: have "
                                    f"{raw_act.shape[0]}, need {needed} "
                                    f"(start_frame={start_frame}, n_chunks={n_chunks}, c={c})"
                                )
                            chunked = raw_act[start_frame : start_frame + n_chunks * c]
                            chunked = chunked.reshape(n_chunks, c, self.action_per_step_dim)
                            actions = chunked.reshape(n_chunks, c * self.action_per_step_dim)
                            # State input is intentionally zeros under action_key
                            # mode (see design notes): the WM is action-only.
                            states = np.zeros(
                                (len(state_indices), c * self.action_per_step_dim),
                                dtype=np.float32,
                            )
                        elif self.use_joint_state:
                            actions = states[1:] - states[:-1]
                        else:
                            actions = self.poses_to_diffs(states)
                        if self.pad_size is not None:
                            actions = np.pad(actions, ((0, 0), (0, self.pad_size - actions.shape[1])))
                            states = np.pad(states, ((0, 0), (0, self.pad_size - states.shape[1])))

                    if "camera_extrinsics" in trajectory["observation"]:
                        camera_extrinsics = trajectory["observation"]["camera_extrinsics"]
                        # DROID keys extrinsics by camera serial (<serial>_left) rather than by camera name
                        extrinsics_name = (
                            view_name
                            if view_name in camera_extrinsics
                            else os.path.basename(mp4_name)[: -len(".mp4")] + "_left"
                        )
                        extrinsics = np.array(camera_extrinsics[extrinsics_name])
                        extrinsics = extrinsics[indices, :]
                    else:
                        extrinsics = np.full((len(indices), 6), fill_value=np.nan)
                    log_func(f"[worker id {wid}, index {index}]start reading frames")
                    vr.seek(0)  # go to start of video before sampling frames
                    buffer = vr.get_batch(indices).asnumpy()
                    log_func(f"[worker id {wid}, index {index}]finish reading frames")
                    if self.transform is not None:
                        # TODO: for multiview use the same transform parameters
                        # for different views, but only for those which makes sense e.g. random hue
                        log_func(f"[worker id {wid}]start apply augs")
                        buffer, prenorm_buffer = self.transform(buffer)
                        log_func(f"[worker id {wid}]finish apply augs")
                    else:
                        prenorm_buffer = buffer
                    buffers.append(buffer)
                    prenorm_buffers.append(prenorm_buffer)
                    extrinsics_buffers.append(extrinsics)
                    view_names.append(view_name)
            buffer = torch.stack(buffers, dim=1)
            dt = torch.ones(self.frames_per_clip, dtype=torch.float32) * dt
            prenorm_buffer = torch.stack(prenorm_buffers, dim=1)
            extrinsics = np.stack(extrinsics_buffers, axis=1)
            if not self.read_intermediate_frames:
                padding_mask = torch.ones(self.frames_per_clip, dtype=torch.float32)
            else:
                padding_mask = torch.ones(self.frames_per_clip + 1, dtype=torch.float32)
            actions = torch.from_numpy(actions).float()
            states = torch.from_numpy(states).float()[:-1]
            extrinsics = torch.from_numpy(extrinsics).float()
            view_names = np.array(view_names)
            if self.read_intermediate_frames:
                skip = int(
                    fstp / i_fstp
                )  # i_fstp is used in the decoder training and is usually equal to cosmos tokenizer temporal pool
                if skip != i_skip_rate:
                    raise ValueError(f"incompatible data: path={vpath}")
                buffer = buffer[:, :, ::skip]
            if buffer.shape[2] > actions.shape[0] and not self.read_intermediate_frames:
                # i.e. we have long enough video to get all frames but not long enough
                # state sequence to get one extra state.
                buffer = buffer[:, :, :-1]
                prenorm_buffer = prenorm_buffer[:, :, :-1]
            mul = 1 if not self.read_intermediate_frames else (fstp // i_fstp)
            if buffer.shape[2] < self.min_length_to_drop:
                raise ValueError(
                    f"Unable to load video with enough frames: path={vpath} {nframes=}, {min_frames=}, {vlen=}"
                )
            padding_mask[buffer.shape[2] - 1 :] = 0.0
            if buffer.shape[2] < (self.frames_per_clip + int(self.read_intermediate_frames)):
                buffer = torch.cat(
                    [
                        buffer,
                        1e-4
                        + torch.zeros(*buffer.shape[:2], self.frames_per_clip - buffer.shape[2], *buffer.shape[3:]),
                    ],
                    dim=2,
                )
                prenorm_buffer = torch.cat(
                    [
                        prenorm_buffer,
                        1e-4
                        + torch.zeros(
                            *prenorm_buffer.shape[:2],
                            self.frames_per_clip * mul - prenorm_buffer.shape[2],
                            *prenorm_buffer.shape[3:],
                        ),
                    ],
                    dim=2,
                )
                actions = torch.cat(
                    [actions, 1e-4 + torch.zeros(self.frames_per_clip - actions.shape[0], *actions.shape[1:])]
                )
                states = torch.cat(
                    [states, 1e-4 + torch.zeros(self.frames_per_clip - states.shape[0], *states.shape[1:])]
                )
                extrinsics = torch.cat(
                    [extrinsics, 1e-4 + torch.zeros(self.frames_per_clip - extrinsics.shape[0], *extrinsics.shape[1:])]
                )
            prenorm_buffer = einops.rearrange(prenorm_buffer, "c v t h w -> t v h w c")
            if (
                # this tests for certain shape invariants that should hold for the loaded data
                # 1) we are not in the "DiT mode" -> we need an equal number of actions and frames
                (buffer.shape[2] != actions.shape[0] and not self.read_intermediate_frames)
                # 2) we are in the "DiT mode" -> we need one extra frame compared to actions
                or (buffer.shape[2] != (actions.shape[0] + 1) and self.read_intermediate_frames)
                # 3) we are in the "DiT mode" -> if T is pre-tokenization frames and t is post-tokenization frames then T//4 + 1 == t  # noqa: E501
                or (buffer.shape[2] != (prenorm_buffer.shape[0] // mul + 1) and self.read_intermediate_frames)
            ):
                raise ValueError(f"bad data id path={path}, star_frame={start_frame}, vlen={vlen}, nframes={nframes}")
        if padding_mask.sum().item() < (self.min_length_to_drop - 1):
            raise ValueError(
                f"Unable to load video with enough frames: path={vpath} {nframes=}, {min_frames=}, {vlen=}"
            )
        if self.view_map is not None:
            view_ids = []
            for v in view_names:
                views_no = len(view_ids)
                for k, idx in self.view_map.items():
                    if re.match(k, v) is not None:
                        view_ids.append(idx)
                        break
                if len(view_ids) == views_no:
                    raise ValueError(f"no view macthed: {v}")
            view_ids = torch.from_numpy(np.array(view_ids))
        else:
            view_ids = torch.from_numpy(np.array([0 for _ in view_names])).long()
        embodiment_name = self.embodiment_name
        return (
            buffer,  # [3, V, T, H, W]
            prenorm_buffer,  # [T, V, H, W, 3]
            actions,  # [T, 7]
            states,  # [T, 7]
            extrinsics,  # [T, V, 6]
            padding_mask,  # [T]
            view_ids,  # [v]
            dt,  # [T]
            start_frame,  # []
            path.path,  # string
            embodiment_name,  # string
        )

    def __len__(self):
        if self.min_size is not None:
            return max(len(self.indices), self.min_size)
        return len(self.indices)
