# AOT Compilation for RoboJEPA World Models

## What this does

Compiles ahead-of-time (AOT) inference programs for a RoboJEPA world model checkpoint using PyTorch Inductor. This produces lightweight `.pt2` files (~20-35 MB each, code-only) that run 2-12x faster than eager PyTorch depending on batch size. The compiled programs are stored inside the checkpoint directory and loaded transparently via `load_compiled=True`.

## Quick start

### 1. Compile a checkpoint (submit a SLURM job)

```bash
# Default (bf16 precision):
sbatch --gpus-per-node=4 --cpus-per-task=64 --mem=512G --time=06:00:00 \
    --wrap="cd <repo_root> && \
    python -m app.robo_jepa.inference.compile_checkpoint \
        /path/to/checkpoint --num-gpus 4"

# Mixed precision (fp32 weights + autocast bf16):
sbatch --gpus-per-node=4 --cpus-per-task=64 --mem=512G --time=06:00:00 \
    --wrap="cd <repo_root> && \
    python -m app.robo_jepa.inference.compile_checkpoint \
        /path/to/checkpoint --num-gpus 4 --precision mp-bf16"
```

This auto-detects resolution, num_views, action_dim, etc. from `config.json` — no model-specific flags needed. Compiles 54 programs (9 steps × 6 batch sizes) in ~40-60 min on 4 H200 GPUs.

### 2. Use the compiled model

```python
from app.robo_jepa.world_model import WorldModel

# Standard load (unchanged):
wm = WorldModel.load_pretrained("/path/to/checkpoint")

# Load with AOT dispatcher (bf16, default):
wm = WorldModel.load_pretrained("/path/to/checkpoint", load_compiled=True)

# Load with mixed-precision AOT dispatcher (fp32 weights + autocast bf16):
wm = WorldModel.load_pretrained("/path/to/checkpoint", load_compiled="mp-bf16")

# Same API — step() is now dispatched to compiled programs:
out, state = wm.step(action, state, use_cache=True)

# Encoder still works normally:
features = wm.encode(frames, dtype=torch.bfloat16, device="cuda")
```

## CLI reference

```
python -m app.robo_jepa.inference.compile_checkpoint CHECKPOINT [OPTIONS]

Arguments:
  CHECKPOINT              Path to model checkpoint directory
                          (must contain config.json + world_model.pth.tar)

Options:
  --num-gpus N            Number of GPUs for parallel compilation (default: 1)
  --batch-sizes 1,2,4...  Batch sizes to compile for (default: 1,2,4,8,16,32)
  --steps 0,1,2...        Step indices to compile for (default: 0,1,2,3,4,5,6,7,8)
  --output-dir DIR        Output directory (default: <checkpoint>/compiled/)
  --force                 Recompile even if .pt2 files already exist
```

## Examples

```bash
# Single GPU, all defaults (slower, ~2-3 hours):
python -m app.robo_jepa.inference.compile_checkpoint /path/to/checkpoint

# 4 GPUs parallel (~40-60 min):
python -m app.robo_jepa.inference.compile_checkpoint /path/to/checkpoint --num-gpus 4

# Only compile batch sizes you need:
python -m app.robo_jepa.inference.compile_checkpoint /path/to/checkpoint --batch-sizes 1,16

# Recompile everything from scratch:
python -m app.robo_jepa.inference.compile_checkpoint /path/to/checkpoint --num-gpus 4 --force
```

## What gets created

After compilation, the checkpoint directory looks like:

```
your-model-checkpoint/
    config.json              # (existing)
    encoder.pth.tar          # (existing)
    world_model.pth.tar      # (existing) — single source of truth for weights
    compiled/                # (new) — ~600 MB to ~1.8 GB depending on model
        manifest.json        # metadata + variant listing
        step0_b1.pt2         # code-only compiled programs
        step0_b2.pt2
        ...
        step8_b32.pt2        # 54 programs total
```

No `weights.pt` is stored — weights are extracted from the existing `world_model.pth.tar` at load time and converted to bf16 automatically.

## Already-compiled checkpoints

| Checkpoint | Views | Resolution | Status |
|-----------|-------|-----------|--------|
| `wm-r2-droid-agibot_opt2-vj25-1B-240p` | 1 | 240p | ✅ compiled |
| `wm-r2-droid-2-views-simple-240p-vj25-1B-opt2` | 2 | 240p | ✅ compiled |

## Performance (H200, PyTorch 2.9.0)

### 1-view model (240p, 1B)

| Batch Size | Eager | Dispatcher | Speedup | ms/step |
|-----------|-------|-----------|---------|---------|
| B=1 | 1565 ms | 127 ms | **12.3x** | 7.9 |
| B=4 | 1593 ms | 263 ms | **6.1x** | 16.4 |
| B=16 | 2119 ms | 770 ms | **2.8x** | 48.1 |
| B=32 | 3119 ms | 1504 ms | **2.1x** | 94.0 |

### 2-view model (240p, 1B)

| Batch Size | Eager | Dispatcher | Speedup | ms/step |
|-----------|-------|-----------|---------|---------|
| B=1 | 1858 ms | 201 ms | **9.3x** | 12.5 |
| B=4 | 1957 ms | 482 ms | **4.1x** | 30.1 |
| B=16 | 3971 ms | 1670 ms | **2.4x** | 104.4 |
| B=32 | 6619 ms | 3248 ms | **2.0x** | 203.0 |

(16-step rollout, 5 trials each. Total time = sum of all 16 steps.)

---

# AOT Compilation for DiT Diffusion Decoder

## What this does

Compiles AOT inference programs for the DiT diffusion decoder (`SimpleDiT`) that generates RGB frames from V-JEPA features via DDIM denoising. Produces 60 `.pt2` programs (5 cache steps × 2 update modes × 6 batch sizes) with shared weights injected at load time.

The decoder runs a multi-step DDIM loop per frame. Each denoising step except the last uses `update_cache=False` (read-only KV cache); the last step uses `update_cache=True` (writes new frame KV into cache with rolling window of 4).

## Quick start

### 1. Compile decoder programs

```bash
# 4 GPUs, ~36 minutes for full 60-program matrix:
sbatch --gpus-per-node=4 --cpus-per-task=40 --mem=512G --time=03:00:00 \
    --wrap="cd <repo_root> && \
    python -m app.robo_jepa.inference.compile_decoder_checkpoint \
        /path/to/checkpoint --num-gpus 4"
```

### 2. Use the compiled decoder

The compiled decoder integrates seamlessly with `WorldModel.load_pretrained()`. When `load_compiled_decoder=True`, the compiled decoder dispatcher is installed as `heads["image_head"]`, so all existing code that calls `head.decode()` automatically uses the compiled path — no API changes needed.

```python
from app.robo_jepa.world_model import WorldModel

# Load compiled WM + compiled decoder (independent flags):
dispatcher = WorldModel.load_pretrained(
    "/path/to/checkpoint",
    load_compiled=True,           # compiled WM dispatcher
    load_compiled_decoder=True,   # compiled decoder at heads["image_head"]
)

# Or load only compiled decoder with eager WM:
wm = WorldModel.load_pretrained(
    "/path/to/checkpoint",
    load_compiled_decoder=True,   # works independently of load_compiled
)

# Decode features → RGB via heads["image_head"] (compiled decoder)
head = wm.heads["image_head"]    # AOTDecoderDispatcher (drop-in for eager head)
rgb, cache = head.decode(
    features,                     # (B, T, V, H, W, C) V-JEPA features
    cfg_scale=1.0,
    denoising_steps=4,
    use_cache=True,
    cache=cache,
)

# Original eager head available at heads["image_head_eager"]:
eager_head = wm.heads["image_head_eager"]
```

#### Online (autoregressive) decoding with KV cache

```python
head = wm.heads["image_head"]
head.last_output = None
kv_cache = None

for step in range(n_steps):
    rgb, kv_cache = head.decode(
        features[:, step:step+1],
        cfg_scale=1.0,
        denoising_steps=4,
        use_cache=True,
        cache=kv_cache,
    )
    # rgb: (B, T_pixel, V, H, W, 3) uint8
```

#### Direct construction (advanced)

```python
from app.robo_jepa.inference import AOTDecoderDispatcher

# From an already-loaded WorldModel (avoids double-loading):
decoder = AOTDecoderDispatcher.from_world_model(
    wm,
    artifact_dir="/path/to/checkpoint/compiled/decoder-bf16",
    batch_sizes=[1],   # optional: only load B=1 programs
)

# Install manually:
wm.heads["image_head_eager"] = wm.heads["image_head"]
wm.heads["image_head"] = decoder

# Create empty cache tensors:
cache_keys, cache_values = decoder.make_empty_cache(batch_size=1)
```

## Decoder CLI reference

```
python -m app.robo_jepa.inference.compile_decoder_checkpoint CHECKPOINT [OPTIONS]

Arguments:
  CHECKPOINT              Path to model checkpoint directory

Options:
  --num-gpus N            Number of GPUs for parallel compilation (default: 1)
  --batch-sizes 1 2 4...  Batch sizes to compile (default: 1 2 4 8 16 32)
  --cache-steps 0 1 2...  Cache steps to compile (default: 0 1 2 3 4)
  --output-dir DIR        Output directory (default: <checkpoint>/compiled/decoder-bf16)
  --force                 Recompile even if .pt2 files exist
```

## Decoder artifact layout

```
checkpoint/compiled/decoder-bf16/          # 502 MB .pt2 + 1.2 GB weights
├── manifest.json                          # metadata + program map
├── decoder_weights.pt                     # shared bf16 state_dict
├── decoder_step0_update0_b1.pt2           # ~3-26 MB each (code-only)
├── decoder_step0_update1_b1.pt2
├── ...
└── decoder_step4_update1_b32.pt2          # 60 programs total
```

## Decoder performance (H200, 240p, bf16)

### Per-step latency (ms)

| Program | B=1 | B=8 | B=16 |
|---------|-----|-----|------|
| step4, denoise (update=F) | 11.2 | 63.7 | 121.9 |
| step4, update (update=T)  | 11.7 | 65.5 | 125.1 |

### Full frame (4-step DDIM, ms)

| Batch | Cache Step 0 | Cache Step 4 |
|-------|-------------|-------------|
| 1     | 29.3        | 46.8        |
| 8     | 152.3       | 256.6       |
| 16    | 295.6       | 493.2       |

### AOT vs Eager speedup

| Batch | Speedup |
|-------|---------|
| B=1   | **6.2×** |
| B=16  | **1.6×** |

## Already-compiled decoder checkpoints

| Checkpoint | Resolution | Programs | Status |
|-----------|-----------|----------|--------|
| `wm-r2-droid_opt2-vj25-240p` | 240p | 60 (5×2×6) | ✅ compiled |

---

## FP8 + torch.compile Inference

### What this does

Static FP8 quantization with `torch.compile` provides an alternative compiled inference path for both the **WM transition model** and the **decoder DiT**. Uses pre-calibrated activation scales (no runtime overhead) and `torch.compile(fullgraph=True)` with Inductor cache for instant startup after initial compilation.

**Why torch.compile instead of AOTI?** AOTI (`.pt2` programs) has a PyTorch 2.9 bug where loading 7+ FP8 programs simultaneously causes `CUDA illegal memory access`. `torch.compile` with Inductor cache is the workaround — same performance, no multi-program crash.

### Quick start: Pre-warm FP8 Inductor cache (CLI)

```bash
# WM only (B=1):
python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint --component wm --batch-sizes 1

# Decoder only:
python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint --component decoder --batch-sizes 1

# Both WM + decoder:
python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint --component both --batch-sizes 1,4

# With custom Inductor cache dir:
TORCHINDUCTOR_CACHE_DIR=/path/to/cache \
python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint --batch-sizes 1

# Save FP8 scales for reuse (skip calibration next time):
python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint \
    --save-scales /path/to/checkpoint/fp8_scales.pt

# Load pre-calibrated scales (skip calibration):
python -m app.robo_jepa.inference.compile_fp8 /path/to/checkpoint \
    --load-scales /path/to/checkpoint/fp8_scales.pt
```

### FP8 CLI reference

```
python -m app.robo_jepa.inference.compile_fp8 CHECKPOINT [OPTIONS]

Arguments:
  CHECKPOINT              Path to model checkpoint directory

Options:
  --component {wm,decoder,both}  Which component to compile (default: both)
  --batch-sizes 1,4,...          Comma-separated batch sizes (default: 1)
  --cache-dir DIR                Inductor cache directory (default: TORCHINDUCTOR_CACHE_DIR)
  --save-scales PATH             Save calibrated FP8 scales to .pt file
  --load-scales PATH             Load pre-calibrated FP8 scales (skip calibration)
```

The script:
1. Loads the model
2. Calibrates FP8 activation scales (or loads pre-calibrated ones)
3. Converts all `nn.Linear` → `StaticFloat8Linear`
4. Runs `torch.compile(fullgraph=True)` for every `(step × batch_size)` variant
5. Saves compiled `.so` files to `TORCHINDUCTOR_CACHE_DIR` — subsequent runs skip compilation entirely

### FP8 decoder usage (Python API)

```python
from app.robo_jepa.world_model import WorldModel
from app.robo_jepa.inference.decoder_dispatcher import AOTDecoderDispatcher
from app.robo_jepa.inference.static_fp8 import calibrate_and_quantize

# 1. Load model
wm = WorldModel.load_pretrained("/path/to/checkpoint", device="cuda", load_decoder=True)
wm.eval()

# 2. Calibrate and quantize decoder's linear layers
dit = wm.heads["image_head"].ema_model

def calibration_fn(model):
    from app.robo_jepa.inference.decoder_aot_compile import (
        DecoderStepModule, build_decoder_example_inputs, get_decoder_config,
    )
    cfg = get_decoder_config(wm)
    for cs in [0, 2, 4]:
        for uc in [False, True]:
            inputs = build_decoder_example_inputs(cfg, cs, 1, "cuda")
            mod = DecoderStepModule(model, cache_size=cs, update_cache=uc)
            mod.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mod(*inputs)

dit, n_fp8 = calibrate_and_quantize(dit, calibration_fn, output_dtype=torch.bfloat16)
print(f"Converted {n_fp8} layers to FP8")

# 3. Create torch.compile dispatcher (lazy compilation on first use)
decoder = AOTDecoderDispatcher.from_world_model(
    wm,
    artifact_dir="",  # no AOTI artifacts needed
    device="cuda",
    use_torch_compile=True,
    compile_cache_dir="/path/to/inductor_cache",
)

# 4. Use same API as eager/AOTI decoder
decoder.last_output = None
cache = None
rgb, cache = decoder.decode(
    features, cfg_scale=1.0, denoising_steps=4,
    use_cache=True, cache=cache,
)
```

### FP8 WM usage (Python API)

```python
from app.robo_jepa.inference.dispatcher import AOTWorldModelDispatcher
from app.robo_jepa.inference.static_fp8 import calibrate_and_quantize
from app.robo_jepa.inference.aot_compile import FunctionalStepModule, build_example_inputs, get_model_config

# 1. Load and calibrate
wm = WorldModel.load_pretrained("/path/to/checkpoint", device="cuda")
wm.eval()
model_config = get_model_config(wm)

def calibration_fn(model):
    for step in range(9):
        inputs = build_example_inputs(model_config, step, 1, model_config["cache_len"], "cuda")
        inputs = tuple(t.to(torch.bfloat16) if t.is_floating_point() else t for t in inputs)
        mod = FunctionalStepModule(model, cache_size=min(step, model_config["cache_len"]), input_position=step)
        mod.eval()
        with torch.no_grad():
            mod(*inputs)

wm.transition_model, n = calibrate_and_quantize(wm.transition_model, calibration_fn, output_dtype=torch.bfloat16)

# 2. Create torch.compile dispatcher
dispatcher = AOTWorldModelDispatcher(wm, artifact_dir="", device="cuda", use_torch_compile=True)

# 3. Pre-warm all variants (or let them compile lazily)
dispatcher.pre_warm_compile_cache(batch_sizes=[1])

# 4. Use same step() API
out, state = dispatcher.step(action, state, use_cache=True)
```

### Pre-calibrated FP8 scales

If you have pre-calibrated scales saved to a `.pt` file:

```python
from app.robo_jepa.inference.static_fp8 import apply_static_fp8

scales = torch.load("/path/to/fp8_scales.pt", map_location="cuda")
dit, n_fp8 = apply_static_fp8(dit, scales, output_dtype=torch.bfloat16)
```

### FP8 decoder performance (H200, 240p, B=1)

| Config | bf16 Eager | bf16 AOT | FP8 torch.compile | FP8 vs Eager |
|--------|-----------|----------|-------------------|--------------|
| step=0 | 70.4 ms | 6.9 ms | 10.2 ms | **6.9×** |
| step=4 | 71.6 ms | 11.3 ms | 12.5 ms | **5.7×** |

At B=1 the decoder is dispatch-bound, so FP8 is ~10% slower than bf16 AOT. At larger batch sizes (B≥8) where the model becomes matmul-bound, FP8's advantage grows.

---

## Troubleshooting

**Compilation fails with `v_mask_q` error on multi-view models:**
Fixed in the codebase. Make sure you have the latest `robo_jepa/models/utils/modules.py`.

**OOM during compilation:**
Reduce `--num-gpus` (each worker loads its own model copy). 4 GPUs works well on H200.

**OOM during inference with `load_compiled=True`:**
Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before running.
