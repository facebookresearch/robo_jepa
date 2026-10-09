# Evaluation

Training only trains: there are no in-training evaluations.

## Forward prediction (imagination error)

The per-step L1 error of autoregressive latent rollouts on held-out data is the y-axis of the paper's scaling laws.
`scripts/evals/eval_droid_holdout.py` computes it on the DROID holdout for every checkpoint of a model size; see
[scripts/evals/README.md](../scripts/evals/README.md) and, for the scaling-law plot, the README's "DROID scaling law".
