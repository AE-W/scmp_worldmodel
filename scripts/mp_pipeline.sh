#!/bin/bash
# MP (heterogeneous SC) pipeline on bridge:
#   1) calibrate fractions for the remaining HPCA configs
#   2) probe each config (n=8, DPM10) to sanity-check quality + measure speed
# Full-test runs are launched separately once probes look sane.
cd /home/qiuyid/scmp_worldmodel
export BRIDGE_ROOT=/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge
export EVAL_OUT_ROOT=/home/qiuyid/scmp_worldmodel/results/line_eval
export PYTHONPATH=.
PY=/home/qiuyid/.conda/envs/scmp/bin/python
SKIP="mlp_fc1=4,25,26,27;mlp_fc2=0,2,4,5,6,7,10,13;qkv=3,6,7,25,27"
SQ=/home/qiuyid/scmp_worldmodel/results/smoothquant_scales.pt

# ---- 1) calibrate the three remaining configs (GPU 7, sequential) ----
for cfg in sc_int7 sc_avg96 sc_int6; do
  f=results/mp_fractions_${cfg}.json
  [ -f "$f" ] && { echo "[calib] $cfg already done"; continue; }
  echo "[calib] $cfg ..."
  CUDA_VISIBLE_DEVICES=7 $PY evaluate/calibrate_mp_fractions.py \
    --config_name $cfg --out $f 2>&1 | grep -E "achieved_avg|level_fractions|Error" | tail -3
done

# ---- 2) probe every calibrated config (n=8) ----
for cfg in sc_avg192 sc_int7 sc_avg96 sc_int6; do
  f=results/mp_fractions_${cfg}.json
  [ -f "$f" ] || { echo "[probe] $cfg: no calibration, skip"; continue; }
  [ -f results/line_eval/mp_${cfg}/summary_shard_0.json ] && { echo "[probe] $cfg done"; continue; }
  MPJ=$($PY -c "import json;d=json.load(open('$f'));print(json.dumps({'stoc_len_levels':d['stoc_len_levels'],'level_fractions':d['level_fractions']}))")
  PREC=$($PY -c "import json;print(json.load(open('$f'))['sc_prec'])")
  echo "[probe] $cfg prec=$PREC mp=$MPJ"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=7 SC_LINEAR_GRANULARITY=per_row SC_HALVE=1 SC_MP_FIXED_PREC=1 \
    SC_PREC=$PREC SC_SMOOTH_SCALES=$SQ SC_MP_CONFIG="$MPJ" \
    $PY evaluate/eval_local_n_samples.py \
      --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
      --tag mp_${cfg} --num_samples 8 --inference_steps 10 --scheduler DPM 2>&1 | grep -E "DONE" | tail -1
  echo "  elapsed $(( $(date +%s) - t0 ))s"
done
echo "MP_PIPELINE_DONE"
