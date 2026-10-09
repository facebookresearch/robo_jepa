# Data

## Format

Every dataset is converted to one common on-disk format: one directory per episode with an HDF5 file and one mp4 per
camera.

```
<dataset root>/
  dataset_index.csv                  # column "path": episode dirs relative to the root
  <episode_dir>/
    data.h5
    <camera videos>.mp4
```

```
data.h5
  cameras/<name>                                   # relative path of the camera's video (without .mp4)
  trajectory/observation/robot_state/
    cartesian_position  [T, pose_dim]              # end-effector pose
    gripper_position    [T, gripper_dim]           # 0 = open, 1 = closed
  trajectory/observation/camera_extrinsics/<camera>  [T, 6]   # optional
  trajectory/action/...                            # recorded controller actions (RoboCasa, DROID)
```

By default, actions are finite differences of consecutive end-effector states (`action_source: state_delta`). For
RoboCasa and DROID, the recorded controller actions can be used instead (`action_source: action_key`). The docstring of
`app/robo_jepa/datasets/robot_video_dataset.py` has the full layout and the batch tuple. Per-dataset action and state
conventions are documented in `app/robo_jepa/docs/action_spaces/`.

## Converting and indexing

`scripts/data_processing/convert_droid.py` converts the raw DROID 1.0.1 release and
`scripts/data_processing/convert_agibot.py` the AgiBot World Alpha and Beta releases; both write `dataset_index.csv`
next to the episodes (see the README's Reproducibility section). Converters for the other datasets will follow.

## Dataset configuration

Each dataset gets a block under `data.train`:

```yaml
data:
  s3_prefix: ${SLURM_TMPDIR}/          # dataset root: S3 URI or local path; env vars are expanded
  train:
    Droid:
      embodiment_name: franka          # selects the action/state normalisation and embedding
      s3_prefix: droid                 # relative to the root above
      raw_fps: 15                      # frame rate of the stored videos
      fps: 1.875                       # world-model step rate
      cameras: [left|right|wrist]      # regexes over camera names
```

Datasets are mixed with per-dataset sampling weights. The full pretraining mixture (`configs/scaling/wm_*.yaml`)
covers DROID, Roboset, RoboMind (Franka and AgileX), LeRobot SO-100/101, SOAR, RoboCasa, Open X-Embodiment subsets
(RT-1, Bridge v1/v2, QT-Opt, MT-Opt, RoboNet, Language Table, BC-Z, FMB), 1X and AgiBot.

**DROID videos are 15 fps.** Older configs that say `raw_fps: 60` are wrong: divide every DROID rate in them by 4
(`raw_fps: 15, fps: 1.875, intermediate_frames_fps: 7.5`).

**AgiBot** has no `cartesian_position` / `gripper_position`. Its config lists `state_keys` / `gripper_keys` (paths under
`trajectory/observation/robot_state`), which the loader concatenates into a 25-D pose and a 2-D gripper state (see
`configs/scaling/wm_100m_local_2k.yaml`).

## Data on local disk (no object store)

On clusters without S3, datasets are stored as tar packs of episode directories, which are extracted to the
node-local SSD at job start:

```bash
mkdir -p $SLURM_TMPDIR/droid
for p in /data/droid/droid_*.tar; do tar -xf "$p" -C $SLURM_TMPDIR/droid & done; wait
cp /data/droid/dataset_index.csv $SLURM_TMPDIR/droid/   # the prebuilt index, next to the packs
```

Run it once per node before training, and point `data.s3_prefix` at `${SLURM_TMPDIR}/droid/`.
`scripts/data_processing/prepare_data.sh <droid|lerobot|agibot>` does this (packs under `$DATA_ROOT/<name>/`); the
local-disk configs call it from `slurm_setup`, so export `DATA_ROOT` before submitting them.

Most LeRobot videos are AV1. The environment ships `decord2`, a drop-in `decord` with ffmpeg and dav1d; the loader
passes `num_threads=0` (with `-1`, dav1d fails with ENOMEM).
