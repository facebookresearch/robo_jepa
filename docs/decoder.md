# Training the decoder

The decoder (`WorldModelDiTCosmosHead`, `app/robo_jepa/world_model_heads.py`) turns V-JEPA features back into video.
It is optional: training and planning do not need it.

- A frozen **Cosmos-0.1-Tokenizer-CV4x8x8** compresses RGB video 8x spatially and 4x temporally into 16-channel
  latents. A 240x320 frame becomes a 30x40 latent.
- A **DiT** (~800M parameters: 32 layers, width 1280, 20 heads / 10 KV heads) denoises those latents, conditioned on
  the V-JEPA features of the same clip.
- Training is DDPM with **x0 prediction**, the `cosine_simple_diffusion` noise schedule (1000 steps, zero terminal SNR)
  and **diffusion forcing**: each frame gets its own noise level, so generation can be conditioned on a clean prefix.
- Attention is **local in time** (`local_attn_window: [4, -1, -1]`), with a KV-cache for frame-by-frame generation.

The decoder trains **only on frozen encoder features** (`main_optimizer: image_head`, `train_heads_on_predictor:
false`). It learns to invert the encoder, independently of any predictor, so one decoder serves world models of every
size. [decoder_architecture.md](decoder_architecture.md) describes the architecture in detail.

## Running

```bash
# full run: DROID, 1 node x 8 GPUs (data.s3_prefix: the converted DROID root)
WANDB_MODE=offline python -m app.main --fname configs/decoder/video_dit_240.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
# smoke test: LeRobot, 1 node x 2 GPUs, 100 steps, through submitit
python -m app.main_distributed --fname configs/decoder/video_dit_240_lerobot_smoke.yaml --time 60
```

## Notes

- Point `model.heads.image_head.config.encoder_path` / `decoder_path` at the Cosmos `.jit` files.
- Use `attn_impl: sdpa` wherever `local_attn_window` is set. The `fmha` (xformers) path does not support the local
  attention mask.
- `video_dit_240.yaml` needs a full 80 GB H100. On a 40 GB MIG slice it fits only with `batch_size: 1` and
  `use_activation_checkpointing: true`.
- While only the head trains, the console `loss:` is hard-coded to 0.0. The real loss (`encoder_image_loss`) is logged
  to wandb.
- Long rollouts decoded with the full KV-cache can collapse into green/magenta noise in later frames. Dropping old
  cache entries fixes this.
