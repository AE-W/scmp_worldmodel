# Per-group n=10 pilot

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

The three arms are uniform-96, historical per-row MP, and per-group MP. The
per-group path is opt-in through `SC_MP_GROUP_CHUNK_D`; existing reproduction
paths remain unchanged.

## Run

```bash
bash RUN_PER_GROUP_N10.sh uniform
bash RUN_PER_GROUP_N10.sh row
bash RUN_PER_GROUP_N10.sh group
bash RUN_PER_GROUP_N10.sh compare
```

`BRIDGE_ROOT`, `EVAL_OUT_ROOT`, and `CUDA_VISIBLE_DEVICES` can be supplied by
the environment. No machine-specific paths are embedded in the implementation.

## Decision rule

Advance to the paired n=100 test only if per-group MP improves mean PSNR over
both uniform and per-row MP on n=10 without regressing mean SSIM or increasing
mean latent L2. Treat n=10 as a gate, not as final evidence.
