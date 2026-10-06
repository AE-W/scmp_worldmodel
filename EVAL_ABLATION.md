# Evaluation ablation

## Hypothesis

At the same 96-cycle average budget, assigning stochastic stream lengths per
`(activation row, 128-channel input chunk)` improves paired Bridge PSNR over
the historical per-row assignment because input chunks expose useful
within-row sensitivity variation.

## Controls

- Same ten keys: `results/diverse_10.json`
- Same sampler: PNDM, 50 steps
- Same SC precision, halving, SmoothQuant scales, and skip recipe
- Same five-level ladder and calibrated fractions
- Same mean stream-length budget
- Same generation seed for every arm (`EVAL_PAIR_SEED=3407` by default)

The four arms are historical per-row uniform-96, historical per-row MP,
per-group uniform-96, and per-group MP. This separates the effect of chunked
quantization from the effect of mixed allocation. The per-group path is opt-in
through `SC_MP_GROUP_CHUNK_D`; existing reproduction paths remain unchanged.

## Run

```bash
bash RUN_EVAL_ABLATION.sh row-uniform
bash RUN_EVAL_ABLATION.sh row-mixed
bash RUN_EVAL_ABLATION.sh group-uniform
bash RUN_EVAL_ABLATION.sh group-mixed
bash RUN_EVAL_ABLATION.sh compare
```

`BRIDGE_ROOT`, `EVAL_OUT_ROOT`, and `CUDA_VISIBLE_DEVICES` can be supplied by
the environment. No machine-specific paths are embedded in the implementation.

## Decision rule

Advance to the paired n=100 test only if per-group MP improves mean PSNR over
both per-group uniform and historical per-row MP on n=10 without regressing
mean SSIM or increasing mean latent L2. Treat n=10 as a gate, not as final
evidence.
