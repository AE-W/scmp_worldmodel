#!/bin/bash
# Reproduce MP_INVESTIGATION_LOG.md E3: n5+smoothed calibration versus
# uniform-96 on the same 100 Bridge trajectories (10 shards x 10 samples).
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH=.
export BRIDGE_ROOT=${BRIDGE_ROOT:-$PWD/robotdata/opensource_robotdata/bridge}
export EVAL_OUT_ROOT=${EVAL_OUT_ROOT:-$PWD/results/local_n_eval}

CAL=$PWD/results/mp_fractions_sc_avg192_n5.json
KEYS=$PWD/results/diverse_100.json
SCALES=$PWD/results/smoothquant_scales.pt
SKIP=$(python -c "import json;print(json.load(open('results/final_sc_recipe.json'))['skip'])")
COMMON=(
  SC_LINEAR_GRANULARITY=per_row
  SC_HALVE=1
  SC_MP_FIXED_PREC=1
  SC_PREC=8
  SC_SMOOTH_SCALES="$SCALES"
)

run_uniform() {
  local shard=$1
  env "${COMMON[@]}" SC_UNIFORM_STOC_LEN=96 \
    python evaluate/eval_local_n_samples.py \
      --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
      --tag scr100_UNI96 --keys_file "$KEYS" --num_samples 100 \
      --shard "$shard" --num_shards 10 --inference_steps 50 --scheduler PNDM
}

run_mixed() {
  local shard=$1
  export SC_MP_CONFIG
  SC_MP_CONFIG=$(python -c "import json;d=json.load(open('$CAL'));print(json.dumps({'stoc_len_levels':d['stoc_len_levels'],'level_fractions':d['level_fractions']},separators=(',',':')))")
  env "${COMMON[@]}" \
    SC_MP_PER_MODULE="$CAL" \
    SC_MP_LEGACY_RAW_AMAX=1 \
    python evaluate/eval_local_n_samples.py \
      --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
      --tag scr100m_sc_avg192_n5 --keys_file "$KEYS" --num_samples 100 \
      --shard "$shard" --num_shards 10 --inference_steps 50 --scheduler PNDM
  unset SC_MP_CONFIG
}

compare_results() {
  python - <<'PY'
import glob, json, os

root = os.environ["EVAL_OUT_ROOT"]
tags = {"uniform": "scr100_UNI96", "mixed": "scr100m_sc_avg192_n5"}
rows = {}
for label, tag in tags.items():
    by_key = {}
    for path in glob.glob(os.path.join(root, tag, "metrics", "*.json")):
        with open(path) as handle:
            item = json.load(handle)
        key = item.get("key") or os.path.splitext(os.path.basename(path))[0]
        by_key[key] = item
    rows[label] = by_key

keys = sorted(set(rows["uniform"]) & set(rows["mixed"]))
if not keys:
    raise SystemExit("No paired metrics found; run all uniform and mixed shards first.")
deltas = [rows["mixed"][k]["psnr"] - rows["uniform"][k]["psnr"] for k in keys]
print(f"paired={len(keys)} wins={sum(x > 0 for x in deltas)}/{len(keys)} "
      f"mean_psnr_delta={sum(deltas) / len(deltas):+.6f} dB")
PY
}

mode=${1:-help}
shard=${2:-${SLURM_ARRAY_TASK_ID:-0}}
case "$mode" in
  uniform|mixed)
    [[ "$shard" =~ ^[0-9]+$ ]] && (( shard >= 0 && shard < 10 )) || {
      echo "shard must be in [0, 9]" >&2; exit 2;
    }
    "run_$mode" "$shard"
    ;;
  all)
    for i in {0..9}; do run_uniform "$i"; done
    for i in {0..9}; do run_mixed "$i"; done
    compare_results
    ;;
  compare) compare_results;;
  *) echo "usage: $0 {uniform|mixed} [shard] | all | compare";;
esac
