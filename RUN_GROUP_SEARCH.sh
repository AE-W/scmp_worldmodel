#!/bin/bash
# Recalibrate against the deployed kernel before testing the untouched n=10.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
export BRIDGE_ROOT=${BRIDGE_ROOT:-$PWD/robotdata/opensource_robotdata/bridge}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export EVAL_PAIR_SEED=3407
export SC_LINEAR_GRANULARITY=per_row SC_HALVE=1 SC_MP_FIXED_PREC=1 SC_PREC=8
export SC_SMOOTH_SCALES=$PWD/results/smoothquant_scales.pt
unset SC_STEP_SCHEDULE SC_MP_INVERT SC_UNIFORM_STOC_LEN SC_MP_LEGACY_RAW_AMAX
export SC_MP_CONFIG= SC_MP_PER_MODULE= SC_MP_GROUP_CHUNK_D=
OUT=${GROUP_SEARCH_ROOT:-$PWD/results/group_search_v1}
mkdir -p "$OUT"
SKIP=$(python -c "import json; print(json.load(open('results/final_sc_recipe.json'))['skip'])")
CAL=$OUT/selected.json
python -m unittest evaluate.test_group_mp_policy
python evaluate/calibrate_group_mp.py \
  --calibration_keys results/eval_calibration_keys.json \
  --holdout_keys results/eval_holdout_keys.json --test_keys results/diverse_10.json \
  --skip "$SKIP" --scales "$SC_SMOOTH_SCALES" --out "$CAL"
export SC_MP_GROUP_CHUNK_D=128 EVAL_OUT_ROOT=$OUT/evaluation
for arm in uniform selected; do
  if [ "$arm" = uniform ]; then
    export SC_MP_CONFIG='{"stoc_len_levels":[96],"level_fractions":[1]}' SC_MP_PER_MODULE=
  else
    export SC_MP_CONFIG=$(python -c "import json; d=json.load(open('$CAL')); print(json.dumps({k:d[k] for k in ('stoc_len_levels','level_fractions')}))")
    export SC_MP_PER_MODULE=$CAL
  fi
  python evaluate/eval_local_n_samples.py \
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
    --tag "$arm" --keys_file results/diverse_10.json --num_samples 10 \
    --inference_steps 50 --scheduler PNDM
done
python - "$EVAL_OUT_ROOT" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
keys = json.load(open('results/diverse_10.json'))
rows = {}
for arm in ('uniform', 'selected'):
    rows[arm] = [json.load(open(root / arm / 'metrics' / (k + '.json'))) for k in keys]
    assert all(r.get('generation_seed') == 3407 for r in rows[arm])
    assert all(math.isfinite(r[m]) for r in rows[arm] for m in ('psnr','ssim','latent_l2'))
report = {'paired_count': len(keys)}
for metric in ('psnr','ssim','latent_l2'):
    delta = [a[metric] - b[metric] for a,b in zip(rows['selected'],rows['uniform'])]
    report[metric] = {'mean_delta':sum(delta)/len(delta),
                      'wins':sum(d < 0 if metric == 'latent_l2' else d > 0 for d in delta)}
(root / 'comparison.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY
