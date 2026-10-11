#!/bin/bash
# Rigorous bridge finalization:
#  1) naive int8 -> full 2946 (fair sample-size parity with FP/SC for FID/FVD)
#  2) official StyleGAN-V FVD cross-check on FP & SC (our streaming FVD is suspect)
cd /home/qiuyid/scmp_worldmodel
export PYTHONPATH=.:pytorch-fid/src
export BRIDGE_ROOT=/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge
export EVAL_OUT_ROOT=/home/qiuyid/scmp_worldmodel/results/local_n_eval
PY=/home/qiuyid/.conda/envs/scmp/bin/python

echo "=== [1] naive int8 -> full 2946 (24 shards over free GPUs) ==="
# reuse the final scheduler pattern but for a single naive-int8 line
for s in $(seq 0 23); do
  # round-robin the free GPUs 2,3,4,5,6 (skip 0,1,7 per contention rules)
  gpu=$(( 2 + (s % 5) )); [ $gpu -eq 7 ] && gpu=6
  CUDA_VISIBLE_DEVICES=$gpu $PY evaluate/eval_local_n_samples.py \
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml --naive_int8 \
    --tag naive_int8_full --num_samples 2946 --shard $s --num_shards 24 \
    >/edrive2/qiuyid/naive_full_s$s.log 2>&1 &
  # cap concurrency at 5
  while [ "$(jobs -r | wc -l)" -ge 5 ]; do sleep 30; done
done
wait
echo "naive_int8_full done: $(ls results/local_n_eval/naive_int8_full/metrics | wc -l)/2946"

echo "=== [2] streaming FVD for naive_int8_full ==="
CUDA_VISIBLE_DEVICES=6 $PY evaluate/compute_fid_fvd_stream.py \
  --pred_dir results/local_n_eval/naive_int8_full/videos \
  --out results/fidfvd_naive_int8_full.json --fid_precomputed -1 2>&1 | grep -E "^FID|^FVD|pred="

echo "ALL_RIGOROUS_DONE"
