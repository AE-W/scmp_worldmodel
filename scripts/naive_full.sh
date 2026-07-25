#!/bin/bash
cd /home/qiuyid/scmp_worldmodel
export BRIDGE_ROOT=/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge
export EVAL_OUT_ROOT=/home/qiuyid/scmp_worldmodel/results/local_n_eval PYTHONPATH=.
PY=/home/qiuyid/.conda/envs/scmp/bin/python
# 24 shard, 抢空闲卡(排除 0/1/7),每卡一个,断点续传
declare -A running
NS=24
while :; do
  done_all=1
  for s in $(seq 0 $((NS-1))); do
    # 该 shard 是否完成: 粗略用 metrics 数近似(精确判断靠脚本内 resume)
    [ -f /edrive2/qiuyid/naivef_s${s}.done ] && continue
    done_all=0
    # 找空闲卡
    for g in 2 3 4 5 6; do
      [ -n "${running[$g]}" ] && kill -0 ${running[$g]} 2>/dev/null && continue
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
      if [ "$free" -ge 20000 ] && [ "$util" -le 40 ]; then
        ( CUDA_VISIBLE_DEVICES=$g $PY evaluate/eval_local_n_samples.py \
            --config configs/evaluation/bridge/frame_ada_sc_full.yaml --naive_int8 \
            --tag naive_int8_full --num_samples 2946 --shard $s --num_shards $NS \
            >/edrive2/qiuyid/naivef_s${s}.log 2>&1 && touch /edrive2/qiuyid/naivef_s${s}.done ) &
        running[$g]=$!
        break
      fi
    done
  done
  [ $done_all -eq 1 ] && break
  sleep 60
done
echo "NAIVE_FULL_DONE: $(ls results/local_n_eval/naive_int8_full/metrics 2>/dev/null | wc -l)/2946"
