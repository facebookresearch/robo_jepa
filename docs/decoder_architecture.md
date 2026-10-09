# DiT Diffusion Decoder for V-JEPA World Model

The code in `app/robo_jepa/robo_jepa/diffusion/` implements a Diffusion Transformer (DiT) decoder that generates RGB video from V-JEPA representation-space features. It is used as an optional "image head" on top of the world model, primarily for visualization and interpretability of the learned latent dynamics.

## End-to-End Pipeline

```
                    ┌──────────────────────┐
                    │   RGB Video          │
                    │ (B, T_raw, V, H,W,3) │
                    └──────────┬───────────┘
                               │
                     temporal subsampling
                   (see robot_video_dataset.py)
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
  ┌───────────────────────┐        ┌──────────────────────┐
  │ prenorm_buffer        │        │ buffer               │
  │ ALL intermediate      │        │ KEY frames only      │
  │ frames at i_fps       │        │ (every mul-th frame) │
  │ (B, fpc*mul+1, V,     │        │ (B, fpc+1, V,        │
  │     H, W, 3)          │        │     H, W, 3)         │
  │ e.g. 33 frames@30fps  │        │ e.g. 9 frames@7.5fps │
  └──────────┬────────────┘        └──────────┬───────────┘
             │                                │
             ▼                                ▼
    ┌────────────────┐               ┌────────────────┐
    │ Cosmos 4x8x8   │               │  V-JEPA 2.1    │
    │ Tokenizer      │               │  Encoder       │
    │ (frozen)       │               │  (frozen)      │
    └───────┬────────┘               └───────┬────────┘
            │                                │
            ▼                                │
 Cosmos Latent Tokens                        │
 (B, V, 16, fpc+1, H/8, W/8)                 │
 e.g. (B, V, 16, 9, 30, 40)                  │
            │                                │
            ▼                                │
    ┌────────────────┐                       │
    │   Forward      │                       │
    │   Diffusion    │                       │
    │ q(x_t | x_0)   │                       │
    │ t ~ U[0, 999]  │                       │
    └───────┬────────┘                       │
            │                                │
            ▼                                ▼
     ┌──────────────────────────────────────────┐
     │           SimpleDiT                      │
     │          (denoiser)                      │
     │         ~800M params                     │
     │                                          │
     │  inputs:                                 │
     │    - noisy Cosmos tokens x_t             │
     │    - V-JEPA features (condition)         │
     │    - timestep t                          │
     │  output: predicted x_0                   │
     └────────────────┬─────────────────────────┘
                      │
                      ▼
             ┌────────────────┐
             │ Cosmos Decoder │
             │   (frozen)     │
             └───────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Reconstructed   │
            │ RGB Video       │
            │ (B,T,V,H,W,3)   │
            └─────────────────┘
```

## Frame Subsampling

The decoder operates at a **higher temporal resolution** than the V-JEPA encoder/world
model. To support this, the dataloader (`robot_video_dataset.py`) loads video at two
frame rates simultaneously and produces two tensors:

- **`buffer`** (low fps) — sparse key frames fed to the V-JEPA encoder / world model.
- **`prenorm_buffer`** (high fps) — all intermediate frames used as the ground-truth
  reconstruction target for the diffusion decoder.

The multiplier `mul` is chosen to match the **temporal compression ratio of the Cosmos
tokenizer** (4× for CV4x8x8), so that each `buffer` key frame corresponds to exactly
one Cosmos latent token.

### How it works

1. All frames are loaded at `intermediate_frames_fps` spacing. The raw-frame step is
   `i_fstp = ceil(raw_fps / intermediate_frames_fps)`, producing `fpc * mul + 1` frames
   (the `+1` is the extra prediction-target tail frame).

2. Both `buffer` and `prenorm_buffer` start as the same tensor of all loaded frames.

3. `buffer = buffer[:, :, ::mul]` keeps only every `mul`-th frame → `fpc + 1` key
   frames. `prenorm_buffer` is **not** subsampled — it retains all `fpc * mul + 1`
   frames.

4. Because `mul` equals the tokenizer's temporal compression, the Cosmos encoder
   compresses `fpc * mul + 1` prenorm frames into `fpc + 1` latent tokens — exactly one
   per buffer key frame.

### Visual layout

```
Time ──────────────────────────────────────────────────────────────────────────▶

prenorm_buffer (all 33 frames at 30 fps):

 pos: 0  1  2  3  4  5  6  7  8  9 10 11 12  ...  28 29 30 31 32
      [■][□][□][□][■][□][□][□][■][□][□][□][■] ... [■][□][□][□][■]
       ↑           ↑           ↑                    ↑           ↑
       └───── buffer (9 key frames = every 4th position) ───────┘

 ■ = key frame  (in BOTH buffer and prenorm_buffer)       ← "buffer" in the diagram
 □ = intermediate frame (ONLY in prenorm_buffer)          ← "prenorm buffer" in the diagram
```

### Shape invariant (checked at runtime)

```
buffer.shape[T]  == prenorm_buffer.shape[T] // mul + 1
(fpc + 1)        == (fpc * mul + 1)         // mul + 1
```

The `+1` tail frame exists in both tensors as the final prediction target. Actions and
states have length `fpc` (8), aligning with the first 8 key frames in `buffer` (the 9th
frame is the prediction target, not paired with an action). The tokenizer compresses the
33 prenorm frames into 9 latent tokens — one per buffer key frame — so the encoder and
the decoder operate in the same temporal grid.

## Architecture Details

### SimpleDiT (`model.py`)

The core denoiser is a PixArt-style Diffusion Transformer with adaptive layer-norm conditioning:

```
    noisy Cosmos tokens x_t          V-JEPA features
    (B, T*H*W, 16)                   (B, T*H'*W', 1664)
            │                               │
            ▼                               │
    ┌───────────────┐                       │
    │  PatchEmbed   │                       │
    │  (Conv3D)     │                       │
    └──────┬────────┘                       │
           │                                │
           ▼                                ▼
    ┌─────────────────────────────────────────┐
    │ cond_mixing = Linear(D + 1664, D)       │
    │ x = cond_mixing(cat[x, interpolate(c)]) │
    └───────────────────┬─────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │ TimestepEmbed │
                │ sin/cos + MLP │
                │ -> (shift,    │
                │  scale, gate) │
                │  x2 per block │
                └──────┬────────┘
                       │
                       ▼
              CondLayerNormBlock x 32
              (see detail below)
                       │
                       ▼
              ┌─────────────────┐
              │   FinalLinear   │
              │ adaptive LN +   │
              │ linear proj     │
              └────────┬────────┘
                       │
                       ▼
                rearrange to
             (B, T, C, H_px, W_px)
```

Each `CondLayerNormBlock` follows this flow:

```
  x ──▶ LayerNorm ──▶ modulate(shift, scale) ──▶ Self-Attention ──▶ gate ──▶ + residual
                                                      │
                                                 (RoPE or standard,
                                                  local temporal window,
                                                  optional KV-cache)
                                                      │
  x ──▶ LayerNorm ──▶ modulate(shift, scale) ──▶ FFN (SwiGLU/GELU) ──▶ gate ──▶ + residual
```

The timestep embedding produces 6 vectors per block: `(shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)`. With **diffusion forcing** (per-frame timesteps), these have shape `(B, T, D)` and the modulation is applied per-frame before flattening back to `(B, T·N, D)` for attention.

**Current config** (`configs/decoder/video_dit_240.yaml`):

| Parameter | Value |
|-----------|-------|
| `depth` | 32 |
| `decoder_embed_dim` | 1280 |
| `num_heads` | 20 |
| `num_kv_heads` | 10 (GQA) |
| `mlp_ratio` | 4.0 |
| `condition_dim` | 1664 (V-JEPA 2.1 ViT-G) |
| `in_chans` | 16 (Cosmos latent channels) |
| `patch_size` | 1 |
| `tubelet_size` | 1 |
| `use_rope` | true |
| `condition_dropout` | 0.1 |
| Total params | ~800M |

### Cosmos Video Tokenizer

The decoder operates in the latent space of [Cosmos-0.1-Tokenizer-CV4x8x8](https://github.com/NVIDIA/Cosmos-Tokenizer) (NVIDIA), a causal video tokenizer that compresses video into continuous latent tokens:

- **Spatial compression**: 8× (e.g. 240×320 → 30×40)
- **Temporal compression**: 4× (e.g. 8 frames → 2 latents), but due to the **causal** design the tokenizer always produces `T/4 + 1` temporal tokens (an extra "initial" token)
- **Channels**: 16 latent channels
- **Both encoder and decoder are frozen** — only the DiT denoiser is trained

Example dimensions for the 240p config (Droid: `raw_fps=60, fps=7.5,
intermediate_frames_fps=30, frames_per_clip=8`):
```
fstp   = 60 / 7.5 = 8   (raw-frame step for key frames)
i_fstp = 60 / 30  = 2   (raw-frame step for intermediate frames)
mul    = 8 / 2     = 4   (= Cosmos temporal compression ratio)

prenorm_buffer:  (B, 33, V, 3, 240, 320)   # fpc*mul+1 = 8*4+1 = 33 frames at 30fps
buffer:          (B,  9, V, 3, 240, 320)   # fpc+1 = 9 key frames at 7.5fps (every 4th)

Cosmos enc (prenorm_buffer → latent):
                 (B, V, 16, 9, 30, 40)     # 9 = 33//4 + 1 temporal tokens (causal +1)
V-JEPA enc (buffer key frames → features):
                 (B, 9, V, 15, 20, 1664)   # one feature map per key frame
DiT operates on: (B, 9*30*40, 1280)        # = 10800 tokens per sample
```

## Training (DDPM)

### Objective

The decoder is trained with a standard DDPM (Denoising Diffusion Probabilistic Model) objective.

The model predicts **x_0 directly** (`predict_xstart: true` in config), meaning:
- Given noisy input `x_t` and timestep `t`, the model outputs `x̂_0` (the predicted clean Cosmos tokens)
- This is controlled by `ModelMeanType.START_X` in `gaussian_diffusion.py`
- Alternative parameterizations (ε-prediction via `ModelMeanType.EPSILON`, velocity-prediction via `ModelMeanType.VELOCITY`) are supported but not used

The loss is MSE between the predicted `x̂_0` and the true `x_0`:
```python
# In gaussian_diffusion.py training_losses():
target = x_start  # when model_mean_type == START_X
loss = (target - model_output) ** 2
```

### Encoder-Only Training

**The decoder is trained only on frozen V-JEPA encoder outputs, not on world-model predictor outputs.** This is enforced by the config setting `train_heads_on_predictor: false`.

This design decision is critical: it means the decoder learns to faithfully invert the V-JEPA representation without being coupled to the predictor's error distribution. At inference time, the same decoder can be applied to either encoder features (for ground-truth visualization) or predictor features (for imagination/rollout visualization) — any artifacts in the latter case are attributable to the predictor, not the decoder.

### Variance

With `learn_sigma: false` and `sigma_small: true`, the model uses **fixed small variance** (`ModelVarType.FIXED_SMALL`). The posterior variance is set to `β̃_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)`, and the model only needs to predict the mean.

## Noise Schedule

The configs use the **`cosine_simple_diffusion`** schedule from [Simple Diffusion (Hoogeboom et al., 2023)](https://arxiv.org/abs/2301.11093) with 1000 diffusion steps.

```python
# noise_schedule.py
def cosine_simple_diffusion_schedule(timesteps, logsnr_min=-15.0, logsnr_max=15.0, shifted=1.0):
    t = linspace(0, 1, timesteps)
    logsnr = -2 * log(tan(t_min + t * (t_max - t_min)))
    alphas_cumprod = 1 / (1 + exp(-logsnr))
```

Key config parameters:

| Parameter | Value | Why it matters |
|-----------|-------|----------------|
| `noise_schedule` | `cosine_simple_diffusion` | Parameterized via log-SNR — smoother than classic cosine, better for latent spaces |
| `diffusion_steps` | 1000 | Full DDPM schedule; inference uses 128-step DDIM |
| `shift` | 1.0 | No SNR shift (1.0 = identity). For higher-resolution latents, increase shift to add more noise at low timesteps |
| `zero_terminal_snr` | true | Enforces SNR → 0 at t=T, preventing information leakage from the terminal step |
| `clip_min` | 1e-9 | Numerical stability floor for betas |

**Why `cosine_simple_diffusion` matters**: Unlike the original DDPM linear schedule or the IDDPM cosine schedule, Simple Diffusion's schedule is parameterized directly in log-SNR space. This gives finer control over the noise levels, especially important when diffusing in a learned latent space (Cosmos tokens) rather than raw pixels. The `shifted` parameter (not used currently at 1.0) would scale the log-SNR to compensate for lower-resolution latents needing less total noise — this becomes relevant if you change the Cosmos tokenizer's compression ratio.

**Zero terminal SNR** (`enforce_zero_terminal_snr` in `noise_schedule.py`) rescales `√ᾱ_t` so that `ᾱ_T = 0` exactly. Without this, the model would never see pure noise during training, creating a train-test mismatch at the start of the reverse process.

## Diffusion Forcing

When `diffusion_forcing=True` (the default in `GaussianDiffusion`), each temporal frame receives an **independently sampled** diffusion timestep during training.

```python
# model.py compute_loss():
t = torch.randint(0, num_timesteps, (B, T), device=x.device)  # per-frame t
#                                    ^^^^
#                              shape (batch, time) — NOT (batch,)
```

This means at any training step, the model sees a video where:
- Frame 0 might be at noise level t=200 (mostly clean)
- Frame 1 might be at noise level t=800 (mostly noise)
- Frame 2 might be at noise level t=50 (very clean)
- etc.

The timestep embedding accordingly has shape `(B, T, 6·D)` and is applied per-frame in the CondLayerNormBlock:
```python
# In CondLayerNormBlock.forward():
if len(t.shape) == 3:  # diffusion forcing: t is (B, T, 6D)
    shift, scale, gate = (scale_shift_table + t.reshape(B, T, 6, -1)).chunk(6, dim=2)
    x = modulate(norm(x.reshape(B, T, -1, C)), shift, scale).flatten(1, 2)
```

**At inference** (DDIM sampling), all frames start at `t = T-1` (maximum noise) and are denoised in lockstep. But diffusion-forcing training makes the model robust to mixed noise levels, which is beneficial for:
- **Prefix-conditioned generation**: providing a few clean frames as context alongside noisy frames to denoise
- **Robustness**: the model doesn't rely on all frames being at the same noise level

## Local Temporal Attention Window

The `SimpleDiT` uses a **local temporal attention window** configured via `local_attn_window`:

```yaml
# decoder config
local_attn_window: [4, -1, -1]  # [temporal, height, width]
#                   ▲   ▲   ▲
#                   │   │   └── -1 = full spatial width
#                   │   └────── -1 = full spatial height
#                   └────────── 4 = attend to 4 temporal neighbors
```

This means each token can attend to all spatial positions but only within a **4-frame temporal neighborhood**. The attention mask is constructed by `build_local_attention_mask()` and converted to block-sparse format (`mask_to_blockwise`) for the `fmha` backend:

```
  Time →    t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7    t=8
  t=0       ████   ████   ████   ████   ░░░░   ░░░░   ░░░░   ░░░░   ░░░░
  t=1       ████   ████   ████   ████   ████   ░░░░   ░░░░   ░░░░   ░░░░
  t=2       ████   ████   ████   ████   ████   ████   ░░░░   ░░░░   ░░░░
  t=3       ████   ████   ████   ████   ████   ████   ████   ░░░░   ░░░░
  t=4       ░░░░   ████   ████   ████   ████   ████   ████   ████   ░░░░
  ...
  (████ = can attend, ░░░░ = masked)
```

Benefits:
- **Memory**: O(window·H·W) per token instead of O(T·H·W)
- **Compute**: Linear in video length with fixed window
- **Quality**: Encourages local temporal coherence without long-range temporal shortcuts

The attention is **bidirectional within the window** (not strictly causal). For strictly causal generation, `is_causal=True` can be set.

## KV-Cache for Efficient Inference

The decoder supports KV-caching for autoregressive frame-by-frame generation, which is critical for real-time MPC planning and interactive visualization.

### Cache Architecture

```
                    KeyValueCache
                    ─────────────
                    shape per layer:
                    keys:   (B, num_kv_heads, cache_len, frame_tokens, head_dim)
                    values: (B, num_kv_heads, cache_len, frame_tokens, head_dim)
                    pointer: int (next write position / FIFO head)

                    cache_len = attn_depth (temporal window size, e.g. 4)
                    frame_tokens = H × W (e.g. 30 × 40 = 1200)
```

### Streaming Decode Step-by-Step

Each autoregressive step takes **one V-JEPA feature** (one temporal position),
denoises **one Cosmos latent token** via a multi-step DDIM loop, writes the KV
activations into the cache on the **last denoising step only**, then feeds the
latent into the Cosmos decoder's sliding window to produce pixel frames.

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEP 0  (first frame, empty cache)                                          │
  │                                                                             │
  │  V-JEPA feature[0] ──┐                                                      │
  │                       ▼                                                     │
  │              ┌─────────────────┐      DDIM denoising loop (e.g. 4 iters)    │
  │              │  z ~ N(0,1)     │──▶  iter 3 → 2 → 1: update_cache=False     │
  │              │  (1 latent tok) │      iter 0 (last):  update_cache=True ─┐  │
  │              └─────────────────┘                                         │  │
  │                       │                                                  │  │
  │                       ▼ denoised latent x̂₀                               │  │
  │              ┌─────────────────┐                                         │  │
  │              │ Cosmos Decoder  │ last_output = [x̂₀]  (1 latent)          │  │
  │              │ (sliding window)│ decode → take last 4 pixel frames       │  │
  │              └────────┬────────┘                                         │  │
  │                       │                                                  │  │
  │                       ▼                                                  ▼  │
  │              RGB frame(s) out                                            │  │
  │                                                                          │  │
  │  KV Cache (cache_len=4):  [frame0][     ][     ][     ]                  │  │
  │                             ▲ ptr=1                          ◀───────────┘  │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEP 1  (cache has 1 entry)                                                 │
  │                                                                             │
  │  V-JEPA feature[1] ──┐                                                      │
  │                       ▼                                                     │
  │              ┌─────────────────┐      DDIM denoising loop                   │
  │              │  z ~ N(0,1)     │──▶  iter 3 → 2 → 1: read cache (frame0)    │
  │              │  (1 latent tok) │      iter 0 (last):  write frame1 ─────┐   │
  │              └─────────────────┘                                        │   │
  │                       │                                                 │   │
  │                       ▼                                                 │   │
  │              ┌─────────────────┐                                        │   │
  │              │ Cosmos Decoder  │ last_output = [x̂₀, x̂₁]  (2 latents)    │   │
  │              │ (sliding window)│ decode → take last 4 pixel frames      │   │
  │              └────────┬────────┘                                        │   │
  │                       ▼                                                 ▼   │
  │              RGB frame(s) out                                           │   │
  │                                                                         │   │
  │  KV Cache:  [frame0][frame1][      ][      ]                            │   │
  │                       ▲ ptr=2                           ◀───────────────┘   │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEPS 2–3  (cache fills up)                                                 │
  │                                                                             │
  │  Same as above — each step denoises 1 new latent, writes KV on last iter    │
  │                                                                             │
  │  After step 2:  [frame0][frame1][frame2][      ]   ptr=3                    │
  │  After step 3:  [frame0][frame1][frame2][frame3]   ptr=4 (full)             │
  │                                                                             │
  │  Cosmos sliding window: last_output keeps last 8 latents, decodes last 4    │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEP 4+  (cache full — FIFO eviction)                                       │
  │                                                                             │
  │  V-JEPA feature[4] ──┐                                                      │
  │                       ▼                                                     │
  │              ┌─────────────────┐      DDIM denoising loop                   │
  │              │  z ~ N(0,1)     │──▶  iters read cache (frames 1,2,3)        │
  │              │  (1 latent tok) │      last iter: evict oldest, append ──┐   │
  │              └─────────────────┘                                        │   │
  │                       │                                                 │   │
  │                       ▼                                                 │   │
  │              ┌─────────────────┐                                        │   │
  │              │ Cosmos Decoder  │ last_output = [..., x̂₃, x̂₄] (last 8)   │   │
  │              │ (sliding window)│ decode → take last 4 pixel frames      │   │
  │              └────────┬────────┘                                        │   │
  │                       ▼                                                 ▼   │
  │              RGB frame(s) out                                           │   │
  │                                                                         │   │
  │  Before: [frame0][frame1][frame2][frame3]    (full, 4 slots)            │   │
  │  Shift:  [frame1][frame2][frame3][      ]    (evict frame0, shift left) │   │
  │  Write:  [frame1][frame2][frame3][frame4]    (append new at the end)    │   │
  │                                                         ◀───────────────┘   │
  │                                                                             │
  │  Next: step 5 → [frame2][frame3][frame4][frame5]  (evict frame1)            │
  │        step 6 → [frame3][frame4][frame5][frame6]  (evict frame2)            │
  │        ...  always the last cache_len frames in chronological order         │
  └─────────────────────────────────────────────────────────────────────────────┘
```

**Key details:**

- **Cache writes only on the last DDIM step.** During the multi-step DDIM loop
  (`ddim_sample_loop`), all iterations except the final one (`i == 0`) use
  `update_cache=False` — they read from the cache for temporal context but do
  not modify it. Only the last iteration sets `update_cache=True`, writing the
  now-clean activations into the cache. This avoids polluting the cache with
  noisy intermediate activations.

- **FIFO eviction.** The `KeyValueCache` is a sliding window that always
  keeps entries in chronological order. While the cache is filling up,
  new entries are appended at `pointer` and the pointer advances. Once
  full, each new write shifts all entries left by one (evicting the oldest
  from slot 0) and appends the new entry at the end. The cache always
  contains the most recent `cache_len` frames' KV activations in order.

- **Cosmos decoder sliding window.** `postprocess_rgb(online=True)` accumulates
  Cosmos latent tokens in `self.last_output`, keeps only the **last 8**, then
  decodes and extracts the **last 4 pixel frames**. This avoids re-decoding the
  entire history at every step.

```python
# postprocess_rgb streaming logic:
self.last_output = cat([self.last_output, new_latent], dim=2)
self.last_output = self.last_output[:, :, -8:]  # keep last 8 latent frames
decoded = self.tokenizer.decode(self.last_output)[:, :, -4:]  # last 4 pixel frames
```

- **Per-step complexity** drops from **O(T · H · W)** to **O(H · W)** in the
  transformer (only 1 new temporal position per step, reading up to
  `cache_len` cached positions).

## Sampling / Inference

At inference time, the decoder uses **DDIM sampling** (deterministic, fewer steps):

```python
# model.py sample():
samples, kv_cache = self.gaussian_diffusion.ddim_sample_loop(
    func, z.shape, z, clip_denoised=False, model_kwargs=kwargs,
    t_init=t, next_t_func=next_t,
)
```

- Training uses 1000 diffusion steps
- Inference uses 128 DDIM steps (configurable via `eval_denoising_steps` in config)
- `SpacedDiffusion` (`respace.py`) handles the timestep mapping from the reduced step count back to the original 1000-step schedule
- Classifier-free guidance is supported via `cfg_scale` (doubles the batch for conditional + unconditional forward passes)

## File Structure

```
app/robo_jepa/robo_jepa/diffusion/
├── __init__.py                ← create_diffusion() factory: builds SpacedDiffusion from config
├── model.py                   ← SimpleDiT (denoiser), CondLayerNormBlock, TimestepEmbedder, FinalLinear
├── gaussian_diffusion.py      ← GaussianDiffusion: forward process, loss, p_sample, ddim_sample, diffusion forcing
├── noise_schedule.py          ← Beta schedule functions: cosine_simple_diffusion, cosine, sigmoid, linear, sd
├── respace.py                 ← SpacedDiffusion: timestep subsampling for faster inference
└── diffusion_utils.py         ← Misc diffusion utilities

app/robo_jepa/
├── world_model_heads.py       ← WorldModelDiTCosmosHead: wraps SimpleDiT + Cosmos tokenizer + EMA
└── robo_jepa/models/
    └── utils/modules.py       ← KeyValueCache, RoPEAttention, Attention, Block, etc.

configs/decoder/
├── video_dit_240.yaml               ← 240×320 decoder config (DROID)
└── video_dit_240_lerobot_smoke.yaml ← 100-step smoke test (LeRobot)
```
