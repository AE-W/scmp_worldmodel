# E3 n=100 replication

This reproduces the paired experiment recorded in `MP_INVESTIGATION_LOG.md` as
`+0.006 dB (52/100)`: five-level mixed precision (`sc_avg192_n5`) versus
uniform stream length 96 on the same 100 Bridge trajectories.

The original HF result tags are:

- uniform: `scr100_UNI96`
- mixed: `scr100m_sc_avg192_n5`

The original calibration file, `results/mp_fractions_sc_avg192_n5.json`, was
recovered from the private HF backup. The run predates commit `b57cc51` and used
the earlier raw-activation row ranking. `RUN_N100_REPLICATION.sh` enables that
historical behavior explicitly with `SC_MP_LEGACY_RAW_AMAX=1`; the default
deployment behavior is unchanged.

Run one of ten shards per job:

```bash
bash RUN_N100_REPLICATION.sh uniform "$SLURM_ARRAY_TASK_ID"
bash RUN_N100_REPLICATION.sh mixed "$SLURM_ARRAY_TASK_ID"
```

After all shards finish:

```bash
bash RUN_N100_REPLICATION.sh compare
```

For a serial run of both arms and the comparison:

```bash
bash RUN_N100_REPLICATION.sh all
```

HF provenance:

- calibration upload: `85261c8922f0ea5949871d7c6ffa3acf8fc95f7a`
- result summaries: `86fa087aac5d51e09137f7d8368f117fb3137cdc`
