#!/bin/bash
# Explicit search and confirmation stages.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
export BRIDGE_ROOT=${BRIDGE_ROOT:-$PWD/robotdata/opensource_robotdata/bridge}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SC_HALVE=1 SC_PREC=8 SC_MP_FIXED_PREC=1 SC_LINEAR_GRANULARITY=per_row
export SC_OWEN_MODE=bitrev SC_SCRAMBLE_MASKS=64
unset SC_MP_CONFIG SC_MP_PER_MODULE SC_STEP_SCHEDULE SC_MP_INVERT
unset SC_UNIFORM_STOC_LEN SC_MP_GROUP_CHUNK_D SC_MP_LEGACY_RAW_AMAX
OUT=${JOINT_SEARCH_ROOT:-$PWD/results/joint_v11}
SKIP=$(python -c "import json; print(json.load(open('results/final_sc_recipe.json'))['skip'])")
STAGE=${1:-help}
if [ "$#" -gt 0 ]; then shift; fi
case "$STAGE" in
  smoke|search|noise-validate)
    python evaluate/joint_mp_search.py --stage "$STAGE" --out "$OUT" --skip "$SKIP" "$@"
    ;;
  video-validate)
    python evaluate/joint_mp_video_confirm.py --out "$OUT" "$@"
    ;;
  *) echo 'usage: bash RUN_JOINT_SEARCH.sh {smoke|search|noise-validate|video-validate}'; exit 2;;
esac
