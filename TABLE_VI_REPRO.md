# Table VI n=10 reproduction

The Table VI runs use the same ten Bridge trajectories from
`results/diverse_10.json`, PNDM with 50 inference steps, and the deployed skip
recipe in `results/final_sc_recipe.json`.

The original result summaries were recovered from the private Hugging Face
backup `BDXXN/scmp-worldmodel-progress`:

| Table row | Run tag | Runtime schedule | PSNR | SSIM | Latent L2 |
|---|---|---|---:|---:|---:|
| PaYN SC-MP 7.58 bit | `scr10_stepWt` | `step_sched_Wt.json` | 23.712 | 0.791 | 0.2544 |
| PaYN SC-MP 6.58 bit | `scr10_sw658` | `step_sched_B48.json` | 22.800 | 0.760 | 0.2710 |
| PaYN SC-MP 6.32 bit | `scr10_sw632` | `step_sched_B40.json` | 22.508 | 0.741 | 0.2864 |
| Uniform SC 6.58 bit | `scr10_u658` | fixed 48 | 22.208 | 0.744 | 0.2761 |
| Uniform SC 6.32 bit | `scr10_u632` | fixed 40 | 22.058 | 0.723 | 0.2920 |

Run the mixed rows with:

```bash
bash RUN_MP_QUEUE.sh stepWt
bash RUN_MP_QUEUE.sh sw658
bash RUN_MP_QUEUE.sh sw632
```

The result uploads were created on 2026-08-01 after Git commit `bfb0fc2`. The
saved evaluation script and configuration hashes are consistent with that
commit; the timestep runtime support was introduced in its ancestor `0406f16`.
The budget-specific schedule files were present in the HF backup but were
missing from Git, so they are included here.

HF provenance:

- `scr10_stepWt`: commit `c2602856fac3e4a2f017f6d33568055c14b4e4fa`
- `scr10_sw658` and `scr10_sw632`: commit
  `b3d9e3707b09bbd86c78c3a4c0142b7c26729453`
