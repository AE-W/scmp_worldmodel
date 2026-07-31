# Running the world-model (IRASim) SC experiments on a Slurm cluster

Companion to `EXPERIMENT_PLAN.md`. That file describes *what* the experiments
are; this one describes *how* to actually get them through a shared,
contended GPU queue without wasting weeks. Written against PSC Bridges-2
(ROBO partition, 6 nodes x 8 H100-80GB), but the mechanics transfer.

--------------------------------------------------------------------------------
## 0. TL;DR

```bash
# one-time
git clone --recurse-submodules <repo> && cd scmp_worldmodel
git checkout reorg/sc-kernel-submodule       # kernels submodule pinned fd0982e
bash scripts/mkenv.sh                        # conda env, torch cu121, -e ./kernels
ln -s /path/to/data/robotdata robotdata      # dataset_dir is relative to CWD

# every experiment is one array job of SHORT shards
sbatch -a 0-329 --export=ALL,CFG=sc_avg192,NSHARD=330 jobs/mp_full.sh
```

The single most important operational fact: **short jobs get scheduled, long
jobs do not.** Everything below follows from that.

--------------------------------------------------------------------------------
## 1. Shard for backfill, not for throughput

A contended queue schedules by priority, but *backfill* runs any job that fits
in the gap before the next high-priority reservation. A 4-hour job almost never
fits; a 1-hour job usually does.

Measured on this cluster: an unsharded 4-hour array sat `PENDING` for **10
hours with zero progress**. Re-cut into 1-hour shards, **5 shards were running
within 2 hours** and 36 finished in ~70 minutes of wall-clock.

Sizing rule:

```
samples_per_shard = (time_limit - model_load) / seconds_per_sample
```

with `model_load ~= 3.5 min` (10.87 GB checkpoint off Lustre) and a 1:15 limit:

| config | s/sample (H100) | samples/shard | shards for 2946 |
|---|---:|---:|---:|
| naive INT (any width) | 15 | 200 | 15 |
| SC uniform L=256 | 662 | 6 | 490 |
| SC uniform L=192 | 324 | 13 | 240 |
| SC-MP avg192 (96 cyc) | 356 | 12 | 250 |
| SC-MP avg96 (48 cyc) | 315 | 13 | 230 |

**Do not derive `s/sample` from the cycle budget.** SC has a large fixed
per-matmul overhead: avg96 (48 avg cycles) is only 12% faster than avg192
(96 avg cycles), not 2x. Sizing avg96 shards on the 2x assumption timed out
81 of 165 shards. Measure one shard first, then size the array.

Per-sample resume is built in, so a timed-out shard loses only the sample in
flight — but it still burns a GPU slot, so get the sizing right.

--------------------------------------------------------------------------------
## 2. Ask for the smallest allocation that works

- **CPUs**: `eval_local_n_samples.py` has no DataLoader — it `torch.load`s
  latents directly. The `num_workers: 11` in the eval YAML is dead config for
  this script. Request **4 cores**, not 12. On a node with 8 free CPUs and a
  free GPU, a 12-core request cannot land and a 4-core one can.
- **GPUs**: always `--gres=gpu:h100:1`. Whole-node requests queue for days.
- **Time**: use `scontrol update JobId=<id> TimeLimit=<shorter>` to shrink a
  *pending* job without losing its queue position. Shrinking `sc_verify` from
  2h to 40min moved its estimated start from 07-28 to 07-27.

--------------------------------------------------------------------------------
## 3. Two-phase: n=300 first, then the rest

`results/diverse_300.json` is a 300-sample subset (episode-dedup + farthest-
point). Run it first: it is 1/10 the cost and gives a config-vs-config
comparison good enough to decide whether the full run is worth it.

Then run the complement (`results/rest_2646.json`) as a separate array so the
two phases never recompute the same key.

**diverse_300 is systematically harder than the full set** — full-2946 PSNR
runs **+1.0 to +1.3 dB above** the diverse_300 subset, and that offset
reproduced across 12 configs. So:

- comparing configs on diverse_300: fine, the offset cancels
- quoting a diverse_300 number next to a 2946 number: **wrong**

--------------------------------------------------------------------------------
## 4. Sample-set and metric hazards

**Always pass `--keys_file`.** Without it the script takes the first N keys in
sorted order, which is a *different and easier* subset. A previously reported
`naive_int8_300` baseline was run this way; it shares only 20 of 300 samples
with `diverse_300` and reads 1.04 dB better. Any comparison against it was
cross-sample-set.

**FID has a severe small-sample bias.** Same videos, only n changes:

| n | frames | FID |
|---:|---:|---:|
| 2946 | 47136 | 7.29 |
| 300 | 4800 | 21.46 |

That is a +14 shift with zero quality difference. Never put an n=300 FID next
to an n=2946 FID. FVD moves the same way (97.1 -> 143.0).

**FVD**: use the official StyleGAN-V script for anything reported. The in-repo
streaming implementation is reproducible and rank-consistent but reads
2.2-4.8 low. `compute_fvd` sets `discard_short_videos=True`, so any clip that
failed to decode is dropped *silently* — verify every clip has exactly 16
frames before trusting the number.

--------------------------------------------------------------------------------
## 5. Mixed precision: pass the config correctly

`SC_MP_CONFIG` is read at **module import time** in
`models/sc_integration/sc_linear.py`, so it must be in the environment before
python starts. Use `export`, never `env VAR=...`:

```bash
# CORRECT
export SC_MP_CONFIG='{"stoc_len_levels":[128,96,64],"level_fractions":[0.2387,0.5226,0.2387]}'
python evaluate/eval_local_n_samples.py ...

# WRONG — the shell word-splits the JSON and the run silently uses no MP
env SC_MP_CONFIG={"stoc_len_levels":[128,96,64],...} python ...
```

Guard it in the job script so a malformed value fails loudly:

```bash
case "$SC_MP_CONFIG" in
  '{"stoc_len_levels":['*) : ;;
  *) echo "!!! SC_MP_CONFIG malformed"; exit 1 ;;
esac
```

`SC_MP_PER_MODULE=<calibration.json>` (optional) switches from one global
fraction triple to per-(operator, block) fractions taken from
`per_module_fractions`. Unset, behaviour is unchanged.

### Bit-width naming

Level values are **halved** cycle counts; nominal stream length `L = 2 x
avg_cycles`; effective bits `= log2(L)`.

| config | avg cycles | L | bits | type |
|---|---:|---:|---:|---|
| sc_int8 | 128 | 256 | 8.00 | uniform |
| sc_avg192 | 96 | 192 | 7.58 | MP |
| sc_int7 | 64 | 128 | 7.00 | uniform |
| sc_avg96 | 48 | 96 | 6.58 | MP |
| sc_avg80 | 40 | 80 | 6.32 | MP |
| sc_int6 | 32 | 64 | 6.00 | uniform |

MP tiers are `0.75 x` the reference uniform L, so the sequence is
192 / 96 / 48 — **there is no `avg64`**.

`SC_UNIFORM_STOC_LEN` accepts **any** positive integer (no power-of-two
constraint), so uniform SC at 7.58 or 6.58 bits is directly runnable and is
the correct same-budget control for the MP tiers. Comparing an MP tier against
the *next uniform tier up* compares across budgets and is not an MP ablation.

A calibration whose target equals the minimum of its level set degenerates to
uniform — `sc_int6` with levels `[64,48,32]` and budget 32 solves to
`[0, 0, 1]`. Choose levels that bracket the target.

--------------------------------------------------------------------------------
## 6. Environment traps

- **Editable installs break when the repo moves.** `pip install -e ./kernels`
  records an absolute path. After relocating the repo, re-run both editable
  installs or every job dies with `ModuleNotFoundError: scmp_kernels`. Put an
  import check at the top of each job so this costs 5 seconds, not 3 minutes:
  ```bash
  python -c "import scmp_kernels; from models.sc_integration import reconfigure" \
    || { echo "env broken"; exit 1; }
  ```
- **`nvcc` is not required.** `scmp_kernels` is pure Triton; a CUDA module is
  never needed, only the torch wheel's bundled runtime.
- **Detector weights**: `evaluation_model/` belongs at the `dataset_dir` root,
  not under `bridge/`. Pre-seed the I3D torchscript into
  `$DNNLIB_CACHE_DIR/downloads/<md5-of-url>_i3d_torchscript.pt` so GPU jobs
  never reach for the network.
- **The SDXL VAE is not in the repo or the dataset tars.** Fetch
  `stabilityai/stable-diffusion-xl-base-1.0` `vae/` separately or
  `AutoencoderKL.from_pretrained` fails at startup.
- **Some sources are text files with CRLF.** Patch them in binary mode or a
  one-line edit shows up as a whole-file diff.

--------------------------------------------------------------------------------
## 7. Pre-flight without a GPU

Everything below runs on a login node and catches most failures before a job
ever queues:

```bash
# every path the eval will touch
PYTHONPATH=. python - <<'PY'
import os
from omegaconf import OmegaConf
from util import update_paths
a = OmegaConf.merge(OmegaConf.load("configs/base/data.yaml"),
                    OmegaConf.load("configs/base/diffusion.yaml"),
                    OmegaConf.load("configs/evaluation/bridge/frame_ada_sc_full.yaml"))
update_paths(a)
for n in ("evaluate_checkpoint","scheduler_path","test_annotation_path",
          "fid_model_path","fvd_model_path","fid_cache_path",
          "true_sample_latent_videos_dir","true_sample_videos_dir"):
    p = getattr(a, n); print(("OK  " if os.path.exists(p) else "MISS"), n, p)
PY

# every key you intend to evaluate actually has a GT latent and mp4
```

--------------------------------------------------------------------------------
## 8. Keeping results

Long campaigns outlive any single session. Run the backup as its own
CPU-partition job (1 core, 48 h) rather than a login-node background process,
which gets reaped:

```bash
sbatch jobs/hf_sweeper.sh     # loops: upload -> verify on HF -> prune local
```

It re-lists the remote after uploading and deletes only confirmed-present
files, keeps the newest 30 mp4s per directory, and never prunes a tag whose
FID/FVD has not been computed. Set `SCMP_HF_TOKEN_FILE`, `SCMP_ROOT` and
`SCMP_RESULTS`; the defaults point at the original author's machine.

Note `hf_backup.py` ignores `*.pt` wholesale, which is how a calibration file
once went missing. Small, GPU-expensive artifacts are uploaded explicitly.

--------------------------------------------------------------------------------
## 9. Cost reference (H100, PNDM-50, 2946 samples)

| run | GPU-hours |
|---|---:|
| one integer-quantization cell | 12 |
| SC uniform L=192 or an MP tier | 250-300 |
| SC uniform L=256 (deployed recipe) | 540 |
| FID + official FVD for one line | ~0.2 |

Calibration (`calibrate_mp_fractions.py`, `calibrate_smoothquant.py`) is 2
samples x 10 steps — under 2 minutes. Calibrate before committing to a 250
GPU-hour evaluation, and check the calibrated fractions are non-degenerate
first.
