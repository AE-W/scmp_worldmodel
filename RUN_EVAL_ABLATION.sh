#!/bin/bash
# Paired n=10 test of per-(row, input-chunk) MP at the 7.58-bit / 96-cycle
# budget. All paths and device choices are supplied through environment vars.
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH=.
export BRIDGE_ROOT=${BRIDGE_ROOT:-$PWD/robotdata/opensource_robotdata/bridge}
export EVAL_OUT_ROOT=${EVAL_OUT_ROOT:-$PWD/results/local_n_eval}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export EVAL_PAIR_SEED=${EVAL_PAIR_SEED:-3407}

CAL=$PWD/results/mp_fractions_sc_avg192_n5.json
KEYS=${KEYS_FILE:-$PWD/results/diverse_10.json}
NUM_SAMPLES=${NUM_SAMPLES:-10}
TAG_SUFFIX=${TAG_SUFFIX:-}
SCALES=$PWD/results/smoothquant_scales.pt
SKIP=$(python -c "import json;print(json.load(open('results/final_sc_recipe.json'))['skip'])")
COMMON=(
  SC_LINEAR_GRANULARITY=per_row
  SC_HALVE=1
  SC_MP_FIXED_PREC=1
  SC_PREC=8
  SC_SMOOTH_SCALES="$SCALES"
)

run_eval() {
  local mode=$1 tag=$2
  shift 2
  tag="${tag}${TAG_SUFFIX}"
  env "${COMMON[@]}" "$@" \
    python evaluate/eval_local_n_samples.py \
      --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
      --tag "$tag" --keys_file "$KEYS" --num_samples "$NUM_SAMPLES" \
      --shard 0 --num_shards 1 --inference_steps 50 --scheduler PNDM
}

set_mp_config() {
  export SC_MP_CONFIG
  SC_MP_CONFIG=$(python -c "import json;d=json.load(open('$CAL'));print(json.dumps({'stoc_len_levels':d['stoc_len_levels'],'level_fractions':d['level_fractions']},separators=(',',':')))")
}

compare_results() {
  python - <<'PY'
import glob, json, os

root = os.environ["EVAL_OUT_ROOT"]
tags = {
    "row_uniform": "eval_a",
    "row_mixed": "eval_b",
    "group_uniform": "eval_c",
    "group_mixed": "eval_d",
}
rows = {}
for label, tag in tags.items():
    by_key = {}
    for path in glob.glob(os.path.join(root, tag, "metrics", "*.json")):
        with open(path) as handle:
            item = json.load(handle)
        key = item.get("key") or os.path.splitext(os.path.basename(path))[0]
        by_key[key] = item
    rows[label] = by_key

common = sorted(set.intersection(*(set(v) for v in rows.values())))
if not common:
    raise SystemExit("No four-way paired metrics found; run all arms first.")
pairs = (
    ("row_mixed", "row_uniform"),
    ("group_uniform", "row_uniform"),
    ("group_mixed", "group_uniform"),
    ("group_mixed", "row_mixed"),
)
for label, baseline in pairs:
    print(f"{label} vs {baseline}: paired={len(common)}")
    for metric, higher, unit in (
        ("psnr", True, " dB"),
        ("ssim", True, ""),
        ("latent_l2", False, ""),
    ):
        delta = [rows[label][k][metric] - rows[baseline][k][metric]
                 for k in common]
        wins = sum((x > 0) if higher else (x < 0) for x in delta)
        print(f"  {metric}: wins={wins}/{len(common)} "
              f"mean_delta={sum(delta) / len(delta):+.6f}{unit}")
PY
}

case "${1:-help}" in
  row-uniform)
    run_eval row-uniform eval_a \
      SC_MP_CONFIG= SC_MP_PER_MODULE= SC_MP_GROUP_CHUNK_D= \
      SC_UNIFORM_STOC_LEN=96
    ;;
  row-mixed)
    set_mp_config
    run_eval row-mixed eval_b \
      SC_MP_PER_MODULE="$CAL" SC_MP_GROUP_CHUNK_D= \
      SC_MP_LEGACY_RAW_AMAX=1
    ;;
  group-uniform)
    export SC_MP_CONFIG='{"stoc_len_levels":[96],"level_fractions":[1.0]}'
    run_eval group-uniform eval_c \
      SC_MP_PER_MODULE= SC_MP_GROUP_CHUNK_D=128
    ;;
  group-mixed)
    set_mp_config
    run_eval group-mixed eval_d \
      SC_MP_PER_MODULE="$CAL" SC_MP_GROUP_CHUNK_D=128
    ;;
  all)
    "$0" row-uniform
    "$0" row-mixed
    "$0" group-uniform
    "$0" group-mixed
    "$0" compare
    ;;
  compare) compare_results;;
  *) echo "usage: $0 {row-uniform|row-mixed|group-uniform|group-mixed|all|compare}";;
esac
