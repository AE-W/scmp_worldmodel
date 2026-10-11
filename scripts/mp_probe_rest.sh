#!/bin/bash
# Probe the remaining MP configs (int7, int6) at n=8, then print the 5-config table.
cd /home/qiuyid/scmp_worldmodel
export BRIDGE_ROOT=/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge
export EVAL_OUT_ROOT=/home/qiuyid/scmp_worldmodel/results/line_eval
export PYTHONPATH=.
PY=/home/qiuyid/.conda/envs/scmp/bin/python
SKIP="mlp_fc1=4,25,26,27;mlp_fc2=0,2,4,5,6,7,10,13;qkv=3,6,7,25,27"
SQ=/home/qiuyid/scmp_worldmodel/results/smoothquant_scales.pt

for cfg in sc_int7 sc_int6; do
  [ -f results/line_eval/mp_${cfg}/summary_shard_0.json ] && { echo "$cfg done"; continue; }
  MPJ=$($PY -c "import json;d=json.load(open('results/mp_fractions_${cfg}.json'));print(json.dumps({'stoc_len_levels':d['stoc_len_levels'],'level_fractions':d['level_fractions']}))")
  echo "[probe] $cfg mp=$MPJ"
  CUDA_VISIBLE_DEVICES=7 SC_LINEAR_GRANULARITY=per_row SC_HALVE=1 SC_MP_FIXED_PREC=1 \
    SC_PREC=8 SC_SMOOTH_SCALES=$SQ SC_MP_CONFIG="$MPJ" \
    $PY evaluate/eval_local_n_samples.py \
      --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
      --tag mp_${cfg} --num_samples 8 --inference_steps 10 --scheduler DPM 2>&1 | grep -E "DONE" | tail -1
done

$PY - <<'PYEOF'
import json, os
rows=[("sc_int8","(uniform 128)"),("sc_avg192","96 cyc"),("sc_int7","64 cyc"),("sc_avg96","48 cyc"),("sc_int6","32 cyc")]
print(f"\n{'config':<12}{'note':<14}{'PSNR':>7}{'SSIM':>7}{'L2':>8}")
for cfg,note in rows:
    f=f"results/line_eval/mp_{cfg}/summary_shard_0.json"
    # sc_int8 uniform baseline reuses q_perrow_halve
    if cfg=="sc_int8": f="results/line_eval/q_perrow_halve/summary_shard_0.json"
    if os.path.exists(f):
        d=json.load(open(f)); print(f"{cfg:<12}{note:<14}{d['mean_psnr']:>7.2f}{d['mean_ssim']:>7.3f}{d['mean_l2']:>8.3f}")
    else: print(f"{cfg:<12}{note:<14}{'--pending--':>22}")
PYEOF
echo "MP_PROBE_REST_DONE"
