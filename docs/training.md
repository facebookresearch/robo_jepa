# Training the world model

## Model

- **Encoder (frozen).** V-JEPA 2.1 ViT-G/384: 1664-dim features from 16x16 patches, so a 240x320 frame becomes a 15x20
  token grid. Each frame and each camera view is encoded separately.
- **Predictor.** A ViT over the token sequence `(T, V, H, W)`, with RoPE, grouped-query attention, local temporal
  attention windows and a KV-cache for step-by-step rollouts. Actions and proprioception enter as extra tokens through
  per-embodiment projections. Camera identity enters through strong view conditioning, a learned per-view embedding
  added to the queries and keys.
- **Objective.** Teacher-forced next-step prediction, plus an **autoregressive rollout loss**: from a random context
  prefix, the model unrolls `rollout_steps` (K) steps on its own predictions. Both losses are computed in feature space
  against the encoder's targets.

## Sizes

| Config | Width | Depth | Heads (KV) | Default layout | Sharding |
|---|---|---|---|---|---|
| `configs/scaling/wm_22m.yaml` | 512 | 7 | 4 (4) | 16 nodes x 8 GPUs | DDP |
| `configs/scaling/wm_50m.yaml` | 640 | 10 | 5 (5) | 16 x 8 | DDP |
| `configs/scaling/wm_100m.yaml` | 768 | 14 | 6 (6) | 16 x 8 | DDP |
| `configs/scaling/wm_300m.yaml` | 1024 | 24 | 8 (8) | 16 x 8 | DDP |
| `configs/scaling/wm_1b.yaml` | 1536 | 40 | 12 (4) | 32 x 8 | DDP |
| `configs/scaling/wm_2b.yaml` | 2048 | 40 | 16 (8) | 32 x 8 | DDP |
| `configs/scaling/wm_4b.yaml` | 2560 | 52 | 20 (10) | 32 x 8 | FSDP2 |
| `configs/scaling/wm_8b.yaml` | 3072 | 72 | 24 (12) | 64 x 8 | FSDP2 |

## Recipe

1. **Pretrain** with `configs/scaling/wm_<size>.yaml`: K = 2 on the full dataset mixture.
2. **Cool down** from a pretraining checkpoint, with constant LR 5e-5:
   - `configs/scaling/low_compute_cooldown/<size>.yaml`: K = 2;
   - `configs/scaling/high_compute_cooldown/<size>.yaml`: K = 10. This is the long-horizon recipe; the paper finds it
     necessary for long-horizon planning.

   Set `meta.read_checkpoint` (written `<INPUT_CHECKPOINT_PATH>` in the configs) to the pretraining checkpoint.

## Launching

`app.main` is the entry point. It writes the resolved config to `<folder>/params-pretrain.yaml` and starts one process
per device. (`train.py` has no `__main__`.)

```bash
# one node, inside a GPU allocation
WANDB_MODE=disabled python -u -m app.main --fname configs/scaling/wm_100m_local_2k.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3
python -m app.main --fname <config> --debug          # a single process on cuda:0, no spawning

# batch job through submitit
python -m app.main_distributed --fname configs/scaling/wm_100m_local_2k.yaml --time 180
```

The submitit launcher (`app/main_distributed.py`) reads these config keys: `nodes`, `tasks_per_node`, `cpus_per_task`,
`gpu_type` (e.g. `h100`, which becomes `--gpus-per-node=h100:N`), `mem_per_gpu`, and `slurm_setup` (shell lines run
before training, e.g. data staging). On a timeout or preemption, submitit requeues the job and training resumes from the
latest checkpoint in `folder`.

**Smoke test:** set `num_epochs: 1`, `iterations_per_epoch: 5` and `wandb.off: true`.

## Checkpoints and logging

- **DDP:** rank 0 writes `<folder>/world_model-latest.pth.tar` atomically, with a `.prev` backup, plus
  `world_model-e<N>.pth.tar` every `meta.save_every_freq` epochs.
- **FSDP2:** a sharded DCP directory `<folder>/world_model-latest/`. `save_checkpoint` must run on every rank.
- **Resume:** `meta.load_checkpoint: true` and `meta.read_checkpoint: <path>`. A DCP directory or a `.pth.tar` is
  detected automatically.
- **Logging:** Weights & Biases through the `wandb:` block. Disable it with `wandb.off: true` or `WANDB_MODE=disabled`.

## Multi-view groups and mixed resolutions

One model can train on batches with different numbers of camera views and different resolutions in the same job.
`data.train.view_groups` partitions the ranks into groups, each with its own view count, datasets and weights. Each
group can also have named resolution subgroups (`augmentations`) with their own crop size and `grad_accum_steps`:

```yaml
data:
  train:
    view_groups:
      two_view:
        num_views: 2
        rank_range: [0, 128]
        datasets: [...]
        augmentations:
          hi: { rank_range: [0, 64],   crop_height: 240, crop_width: 320, grad_accum_steps: 1 }
          lo: { rank_range: [64, 128], crop_height: 144, crop_width: 192, grad_accum_steps: 2 }
      one_view:
        num_views: 1
        rank_range: [128, 256]
        datasets: [...]
```

The predictor must know every runtime view count and grid size in advance. It pre-builds the attention masks, so no
rank allocates one mid-forward, which would desynchronise NCCL. It reads them from
`model.transition_model.config.view_groups_v` and `view_groups_hw`; `train.py` fills in `view_groups_hw`
automatically. `view_map` gives each view slot a camera identity for strong view conditioning. Subgroups must
partition their group's `rank_range` exactly, and crop sizes must be multiples of 16.

## FSDP2 / HSDP

The 4B and 8B configs shard with FSDP2 (`meta.fsdp.enabled: true`). The default `mesh_layout: hsdp` shards within a
node and replicates across nodes; `shard_only` and `replicate_only` are also available. Each predictor block is
wrapped separately, and the order matters: activation checkpointing around each block, then `fully_shard`, then the
optimizer, then `torch.compile`. Checkpoints are DCP directories written by the first replica's ranks only. A legacy
`.pth.tar` can warm-start an FSDP2 model (weights only, no optimizer state).

## Tensor and sequence parallelism

`meta.tp` enables tensor and sequence parallelism (`app/robo_jepa/parallel_utils.py`) for high-resolution finetuning.
See `configs/finetuning/droid_8b_3view_720p_2k_from300k.yaml`: 8B, DROID with 3 views at 720x1280, TP size 4.
