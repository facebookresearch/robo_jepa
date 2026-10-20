# Evals

## DROID holdout forward prediction

`eval_droid_holdout.py` computes the per-step rollout L1 (steps 1–9) on the DROID holdout (REAL lab, "setup room")
for every checkpoint of one model size. It uses the same fixed set of clips for all checkpoints and sizes.

You need:
- **checkpoints:** from the main README table, laid out as `<models-dir>/wm-<size>/config.json` and
  `<models-dir>/wm-<size>/cooldown_<size>_<step>k.pth.tar`;
- **DROID:** converted with `scripts/data_processing/convert_droid.py`;
- **encoder:** the V-JEPA 2.1 ViT-G/384 weights (`vjepa2_1_vitG_384.pt`).

Run it from the repo root, on one GPU:

```bash
PYTHONPATH=$PWD python scripts/evals/eval_droid_holdout.py --size 50m \
    --models-dir /data/robojepa --data /data/droid --encoder /data/vjepa2_1_vitG_384.pt \
    --out-dir results/wm-50m
```

Options:
- `--steps 100,150` evaluates only those checkpoints (training steps in k).
- `--kv-cache` rolls out through the KV cache. It's faster and matches the uncached numbers up to bf16 noise.
- `--n-clips` (default 1024), `--seed` and `--batch-size` set the clip count, seed and batch size.

Output:
- One JSON per checkpoint goes in `--out-dir`, with `rollout_l1_per_step`, `rollout_l1_mean_1_9` and
  `step9_rollout_l1`.
- Checkpoints that already have a JSON are skipped, so an interrupted run resumes.
- The script adds the metadata of every episode missing from `<--data>/aggregated_metadata.json`, creating the file
  if needed; existing entries are kept.
