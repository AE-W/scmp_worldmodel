#!/bin/bash
# Run the two MP configs (avg192, avg96) + backfill any uniform gaps.
# Uses explicit `export` (not `env VAR=...`) so the compact MP JSON is passed
# as a single argument and never word-split.
cd /home/qiuyid/scmp_worldmodel
export BRIDGE_ROOT=/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge
export EVAL_OUT_ROOT=/home/qiuyid/scmp_worldmodel/results/local_n_eval
export PYTHONPATH=.
PY=/home/qiuyid/.conda/envs/scmp/bin/python
SKIP="mlp_fc1=4,25,26,27;mlp_fc2=0,2,4,5,6,7,10,13;qkv=3,6,7,25,27"
export SC_SMOOTH_SCALES=/home/qiuyid/scmp_worldmodel/results/smoothquant_scales.pt
export SC_LINEAR_GRANULARITY=per_row SC_HALVE=1 SC_MP_FIXED_PREC=1 SC_PREC=8
KEYS=results/diverse_300.json
N=$($PY -c "import json;print(len(json.load(open('$KEYS'))))")
NS=12

run_shard() {  # cfg gpu shard  (env for SC path already exported by caller)
  local cfg=$1 g=$2 s=$3
  CUDA_VISIBLE_DEVICES=$g $PY evaluate/eval_local_n_samples.py \
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
    --tag ladder_${cfg} --keys_file $KEYS --num_samples $N \
    --shard $s --num_shards $NS >/edrive2/qiuyid/ladder_${cfg}_s${s}.log 2>&1 \
    && touch /edrive2/qiuyid/ladder_${cfg}_s${s}.done
}

declare -A pid
for cfg in sc_int7 sc_avg192 sc_avg96; do
  # set the SC-path env for this config in the current shell (exported → children inherit)
  unset SC_UNIFORM_STOC_LEN SC_MP_CONFIG
  if [ "$cfg" = "sc_int7" ]; then
    export SC_UNIFORM_STOC_LEN=64
  else
    export SC_MP_CONFIG=$($PY -c "import json;d=json.load(open('results/mp_fractions_${cfg}.json'));print(json.dumps({'stoc_len_levels':d['stoc_len_levels'],'level_fractions':d['level_fractions']},separators=(',',':')))")
  fi
  echo "[$cfg] SC_UNIFORM_STOC_LEN=${SC_UNIFORM_STOC_LEN:-unset} SC_MP_CONFIG=${SC_MP_CONFIG:-unset}"
  for s in $(seq 0 $((NS-1))); do
    [ -f /edrive2/qiuyid/ladder_${cfg}_s${s}.done ] && continue
    # wait for a free allowed GPU (2,3,4,5,6)
    while :; do
      launched=0
      for g in 2 3 4 5 6; do
        [ -n "${pid[$g]}" ] && kill -0 ${pid[$g]} 2>/dev/null && continue
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
        if [ "$free" -ge 20000 ] && [ "$util" -le 45 ]; then
          run_shard $cfg $g $s & pid[$g]=$!; launched=1; break
        fi
      done
      [ $launched -eq 1 ] && break
      sleep 45
    done
  done
  wait   # finish this config before switching env for the next
done
echo "MP_LADDER_ALL_DONE"
