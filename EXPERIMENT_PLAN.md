# SCMP World Model (IRASim) — Experiment Plan & Runbook

Everything needed to reproduce / continue the SC quantization study on a fresh
machine. Written 2026-07-24. Code lives on `AE-W/scmp_worldmodel`, branch
`reorg/sc-kernel-submodule` (HEAD `e578d11`), kernels submodule pinned `fd0982e`.

--------------------------------------------------------------------------------
## 0. What this is

Task: run the IRASim video-DiT world model with **every matmul simulated in
stochastic computing (SC)**, and measure the quality/precision tradeoff against
FP and integer-quantization baselines. Model = IRASim-XL/2 (28-block video DiT,
hidden 1152, alternating spatial/temporal attention, frame-level adaLN action
conditioning), one checkpoint per dataset (bridge / RT-1 / Language-Table).

--------------------------------------------------------------------------------
## 1. Environment setup (new machine)

```bash
# clone with the kernels submodule
git clone --recurse-submodules https://github.com/AE-W/scmp_worldmodel.git
cd scmp_worldmodel
git checkout reorg/sc-kernel-submodule

# conda env (py3.10, cu121)
conda create -n scmp python=3.10 -y && conda activate scmp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm "diffusers[torch]==0.24.0" einops transformers scikit-image \
  pandas imageio imageio-ffmpeg omegaconf "huggingface_hub==0.25.2" \
  opencv-python-headless rotary_embedding_torch einops_exts accelerate decord \
  wandb tensorboard scipy matplotlib
pip install -e ./kernels          # scmp_kernels (Triton, needs CUDA GPU)
pip install -e ./pytorch-fid      # for FID (or: pip install pytorch-fid)
```

Requires: CUDA GPU (A100-80G used; ~15-20 GB per eval task, batch=1).

### Data (bridge, from ByteDance IRASim release)

The eval subset (per dataset) needs: `evaluation_latent_videos/test_sample_latent_videos/*.pt`,
`evaluation_videos/test_sample_videos/*.mp4`, `annotation/test/*.json`,
`checkpoints/frame_ada/0300000.pt`, `evaluation_cache/test_fid_cache.npz`.

```bash
BASE=https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1
mkdir -p robotdata/opensource_robotdata && cd robotdata/opensource_robotdata
# checkpoints tar is UNCOMPRESSED tar w/ HTTP range -> can parallel-fetch one file
# eval tar is gzip (stream + extract subset). annotation is in the TRAIN tar.
curl -sS "$BASE/bridge_checkpoints_data.tar.gz" | tar -x  --wildcards '*checkpoints/frame_ada/*'
curl -sS "$BASE/bridge_evaluation_data.tar.gz"  | tar -xz --wildcards \
  '*test_sample_latent_videos*' '*test_sample_videos*' '*evaluation_cache*'
curl -sS "$BASE/bridge_train_data.tar.gz"       | tar -xz --wildcards '*annotation/test*'
# FID/FVD detector models (NOT in the tars — standard weights):
mkdir -p evaluation_model
curl -sL "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth" \
  -o evaluation_model/pt_inception-2015-12-05-6726825d.pth
curl -sL "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1" \
  -o evaluation_model/i3d_torchscript.pt
```

Then set once per shell (adjust paths):
```bash
export BRIDGE_ROOT=$PWD/robotdata/opensource_robotdata/bridge
export EVAL_OUT_ROOT=$PWD/results/local_n_eval
export PYTHONPATH=.
```

--------------------------------------------------------------------------------
## 2. The SC recipe (the "deployed" config)

Full-coverage SC: **100% of diffusion steps × all 28 blocks × all 6 matmul
types** (qkv / qk / av / proj / mlp_fc1 / mlp_fc2). Quality recovered by four
optimizations, all env-gated (measured deltas, DPM10 n=8, full SC):

| optimization | env | delta |
|---|---|---|
| per-row quantization | `SC_LINEAR_GRANULARITY=per_row` | 14.16 → 20.21 dB |
| uSystolic stream halving | `SC_HALVE=1` | → 23.93 dB |
| SmoothQuant (α=0.5) | `SC_SMOOTH_SCALES=<file>` | → 24.43 dB |
| sensitivity top-17 skip | `--skip "<set>"` | keeps it (per mandate) |

**skip set** (10.1% of 168 operators, from leave-one-in sensitivity, n=300):
```
mlp_fc1=4,25,26,27;mlp_fc2=0,2,4,5,6,7,10,13;qkv=3,6,7,25
```
(NOTE: the value actually deployed on the full-test SC line was the "provisional
top-17", which adds qkv=27 and swaps two fc2 blocks — see
results/final_sc_recipe.json for the byte-exact string in use.)

SmoothQuant scales are pre-calibrated at `results/smoothquant_scales.pt`
(α=0.5, 112 linears). Regenerate with:
```bash
python evaluate/calibrate_smoothquant.py \
  --config configs/evaluation/bridge/frame_ada_sc_full.yaml \
  --keys_file results/diverse_300.json --num_samples 8 --alpha 0.5 \
  --out results/smoothquant_scales.pt
```

### SC spec (group convention, HPCA)
`sc_prec ≡ 8` always. "int7/int6" is the ISO-equivalent INT width for a cycle
budget, NOT a lowered grid. Level values ARE halved cycle counts.
"int series = uniform, avg series = mixed-precision". Full table in
`configs/sc_spec.yaml`:

| config | type | levels / stoc_len | budget | avg cycles | iso-int |
|---|---|---|---|---|---|
| sc_int8   | uniform | 128 | — | 128 | int8 |
| sc_int7   | uniform | 64  | — | 64  | int7 |
| sc_int6   | uniform | 32  | — | 32  | int6 |
| sc_avg192 | MP | [128,96,64] | 0.75×128 | 96 | int8 |
| sc_avg96  | MP | [64,48,32]  | 0.75×64  | 48 | int7 |

Scrambling: bitrev, 64 masks (kernel default, `SC_OWEN_MODE=bitrev`,
`SC_SCRAMBLE_MASKS=64` — no env needed). Symmetric + bipolar.

--------------------------------------------------------------------------------
## 3. Experiment matrix & status (bridge)

Legend: ✅ done · 🟢 running · ⏳ todo

### Baselines (11 cells)
- ✅ FP16 full (2946): **PSNR 25.30 / SSIM 0.834 / L2 0.1945 / FID 4.99 / FVD*
  (FVD needs official recompute)**
- ✅ W8A8_symm (naive int8) full (2946): 300-clip snapshot **24.81 / 0.815 / FID 21.59**
  (full aggregate to be recomputed)
- ⏳ W8A8_asymm, W7A7_{symm,asymm}, W6A6_{symm,asymm}, W5A5_{symm,asymm},
  W4A4_{symm,asymm} — 9 cells not run. These are cheap (plain int quant, fast).

### SC Uniform (n=300 diverse; ladder, same sample set)
- ✅ sc_int8 (128): **PSNR 23.27 / SSIM 0.793 / L2 0.259**
- ✅ sc_int7 (64):  **22.73 / 0.766 / 0.257**
- ✅ sc_int6 (32):  **21.25 / 0.684 / 0.320**

### SC Mixed precision (n=300 diverse)
- ✅ calibrated fractions: avg192 = [0.239,0.523,0.239]@[128,96,64];
  avg96 = [0.243,0.514,0.243]@[64,48,32]
- 🟢 sc_avg192, sc_avg96 evaluation running (`scripts/run_mp_ladder.sh`)

### SC full-test deployed recipe (2946, PNDM50)
- ✅ **PSNR 24.43 / SSIM 0.807 / L2 0.2082 / FID 10.85** — SC FID is HALF of
  naive int8's 21.59 at the same bit width (quantization cost, not SC noise)

### Other datasets
- ⏳ Language-Table: data ready, needs eval config (288×512, 2-D action) +
  action adapter (per dataset_2D.py). MUST re-run the 4-piece sensitivity
  (diverse sampling → LOI sweep → skip verify → SmoothQuant recalib) — weights
  are independently trained, sensitivity does NOT transfer.
- ⏳ RT-1: data incomplete (disk was full). Same 4-piece re-run rule applies.

--------------------------------------------------------------------------------
## 4. How to run each experiment (exact commands)

### 4a. One SC config, n samples (the primitive)
```bash
# uniform (e.g. sc_int7, stream length 64):
CUDA_VISIBLE_DEVICES=0 SC_LINEAR_GRANULARITY=per_row SC_HALVE=1 SC_MP_FIXED_PREC=1 \
  SC_PREC=8 SC_UNIFORM_STOC_LEN=64 \
  SC_SMOOTH_SCALES=results/smoothquant_scales.pt \
  python evaluate/eval_local_n_samples.py \
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml \
    --skip "mlp_fc1=4,25,26,27;mlp_fc2=0,2,4,5,6,7,10,13;qkv=3,6,7,25" \
    --tag sc_int7 --keys_file results/diverse_300.json --num_samples 300 \
    --shard 0 --num_shards 1 --inference_steps 50 --scheduler PNDM

# mixed precision (e.g. sc_avg192) — pass compact JSON via EXPORT (never `env VAR=`,
# it word-splits the JSON!):
export SC_MP_CONFIG='{"stoc_len_levels":[128,96,64],"level_fractions":[0.2387,0.5226,0.2387]}'
CUDA_VISIBLE_DEVICES=0 SC_LINEAR_GRANULARITY=per_row SC_HALVE=1 SC_MP_FIXED_PREC=1 \
  SC_PREC=8 SC_SMOOTH_SCALES=results/smoothquant_scales.pt \
  python evaluate/eval_local_n_samples.py \
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml \
    --skip "..." --tag sc_avg192 --keys_file results/diverse_300.json \
    --num_samples 300 --shard 0 --num_shards 1 --inference_steps 50 --scheduler PNDM
```
Key eval flags: `--inference_steps 50 --scheduler PNDM` = paper protocol (slow,
~13 min/sample with SC). `--inference_steps 10 --scheduler DPM` = fast probe
(~2 min/sample) for calibration/sanity only. `--naive_int8` = plain int8
(no SC) baseline. `--keys_file results/diverse_300.json` selects the 300
diverse samples (episode-dedup + farthest-point sampling; see
evaluate/select_diverse_samples.py). Per-sample metrics resume automatically.

### 4b. Full 5-config ladder (GPU auto-grab, resume)
```bash
# uniform int8/int7/int6 (edit EXCLUDE gpus inside if needed):
setsid nohup scripts/run_sc_ladder.sh   > sc_ladder.log 2>&1 < /dev/null & disown
# MP avg192/avg96 (+ any uniform backfill) — uses explicit export:
setsid nohup scripts/run_mp_ladder.sh   > mp_ladder.log 2>&1 < /dev/null & disown
```
Both slice each config into 12 shards, auto-grab GPUs 2-6 (free>20 GB, util<45%),
skip contended ones, and resume from per-sample metric files.

### 4c. FID / FVD
```bash
# FID (standard pytorch-fid, reliable):
PYTHONPATH=.:pytorch-fid/src CUDA_VISIBLE_DEVICES=0 \
  python evaluate/compute_fid_fvd_stream.py \
    --pred_dir results/local_n_eval/<tag>/videos --out results/fidfvd_<tag>.json
# ⚠️ FVD: our streaming impl is NOT yet cross-checked against official
# StyleGAN-V (rescale bug fixed but unverified). For paper numbers, use
# stylegan-v/src/scripts/calc_metrics_for_dataset.py. DO NOT report our FVD
# until cross-checked.
```

### 4d. MP fraction calibration (per config, per dataset)
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate/calibrate_mp_fractions.py \
  --config_name sc_avg192 --out results/mp_fractions_sc_avg192.json
# Lagrangian budget allocation over measured per-row error curves.
# Repeat for sc_avg96. (int8/int7/int6 are uniform — no calibration.)
```

### 4e. Sensitivity re-run (REQUIRED per new dataset/checkpoint)
```bash
# 1) diverse 300 samples on THIS dataset's latents:
python evaluate/select_diverse_samples.py --n 300 --out results/diverse_<ds>.json
# 2) leave-one-in sweep (fast, ~all-FP): 169 configs × 300, GPU auto-grab:
setsid nohup python evaluate/gpu_scheduler_loi.py > loi.log 2>&1 < /dev/null & disown
#    -> rank operators, take top-17 (10%) as skip set
# 3) verify skip set beats no-skip on new baseline (probe, n=8)
# 4) recalibrate SmoothQuant on this dataset (activation stats differ)
```

--------------------------------------------------------------------------------
## 5. Backups & housekeeping

- Metrics/json/code: auto-backed-up hourly to HF private dataset
  `BDXXN/scmp-worldmodel-progress` (`evaluate/hf_backup.py`, cron :23).
- Videos: `evaluate/hf_backup_videos.py` (cron every 10 min) uploads new mp4s,
  verifies them on HF, then prunes local copies (keeps newest 30/dir; never
  touches a line whose FID/FVD isn't computed yet). Keeps disk flat.
- Kernel daily check: `scripts/check_kernel_update.sh` (cron 8:17) fetches
  scmp_kernels; log at ~/scmp_kernel_update.log.
- Disk on the current machine is shared & chronically full — guard writes.

--------------------------------------------------------------------------------
## 6. Immediate TODO (priority order)

1. Finish MP ladder (avg192/avg96 n=300) → complete the 5-config table.
2. Official StyleGAN-V FVD recompute for FP / SC / naive int8 (our FVD suspect).
3. Aggregate naive_int8 FULL (2946) FID/FVD for a fair 3-row headline table.
4. Fill the 9 missing integer baselines (W8A8_asymm … W4A4, symm+asymm) — cheap.
5. Language-Table: eval config + 2-D action adapter → 4-piece sensitivity → run.
6. RT-1: free disk, stream data, → 4-piece sensitivity → run.
7. QwT (optional) — kernel-side placeholder still.

--------------------------------------------------------------------------------
## 7. Key files

| file | purpose |
|---|---|
| `configs/sc_spec.yaml` | authoritative SC spec + HPCA config table |
| `models/sc_integration/sc_linear.py` | per_row / halve / SmoothQuant / MP dispatch |
| `evaluate/eval_local_n_samples.py` | the eval primitive (--keys_file, --skip, resume) |
| `evaluate/calibrate_mp_fractions.py` | MP fraction solver (Lagrangian) |
| `evaluate/calibrate_smoothquant.py` | SmoothQuant scale calibration |
| `evaluate/select_diverse_samples.py` | diverse-300 selection (episode-dedup + FPS) |
| `evaluate/gpu_scheduler_loi.py` | leave-one-in sensitivity sweep scheduler |
| `evaluate/compute_fid_fvd_stream.py` | streaming FID (FVD pending official) |
| `scripts/run_sc_ladder.sh` / `run_mp_ladder.sh` | 5-config ladder runners |
| `results/final_sc_recipe.json` | byte-exact deployed SC recipe |
| `results/diverse_300.json` | the 300 diverse sample keys |
| `results/mp_fractions_*.json` | calibrated MP fractions |
