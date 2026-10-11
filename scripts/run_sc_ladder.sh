#!/bin/bash
# The 5-config SC precision ladder on bridge, n=300 diverse samples, PNDM50.
# int8/int7/int6 = uniform (fixed stoc_len); avg192/avg96 = MP (calibrated).
# All share the deployed recipe: per_row + halve + SmoothQuant + top-17 skip.
# GPU auto-grab over 2,3,4,5,6 (skip 0,1,7 per contention rules); per-sample resume.
cd /home/qiuyid/scmp_worldmodel
export BRIDGE_ROOT=/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge
export EVAL_OUT_ROOT=/home/qiuyid/scmp_worldmodel/results/local_n_eval
export PYTHONPATH=.
PY=/home/qiuyid/.conda/envs/scmp/bin/python
SKIP="mlp_fc1=4,25,26,27;mlp_fc2=0,2,4,5,6,7,10,13;qkv=3,6,7,25,27"
SQ=/home/qiuyid/scmp_worldmodel/results/smoothquant_scales.pt
KEYS=results/diverse_300.json
N=$($PY -c "import json;print(len(json.load(open('$KEYS'))))")

# config -> env for the SC path (uniform via SC_UNIFORM_STOC_LEN, MP via SC_MP_CONFIG)
env_for() {
  case "$1" in
    sc_int8) echo "SC_UNIFORM_STOC_LEN=128" ;;
    sc_int7) echo "SC_UNIFORM_STOC_LEN=64"  ;;
    sc_int6) echo "SC_UNIFORM_STOC_LEN=32"  ;;
    sc_avg192|sc_avg96)
      MPJ=$($PY -c "import json;d=json.load(open('results/mp_fractions_$1.json'));print(json.dumps({'stoc_len_levels':d['stoc_len_levels'],'level_fractions':d['level_fractions']},separators=(',',':')))")
      echo "SC_MP_CONFIG=$MPJ" ;;
  esac
}

CONFIGS="sc_int8 sc_int7 sc_int6 sc_avg192 sc_avg96"
NS=12   # shards per config
declare -A pid
for cfg in $CONFIGS; do
  extra=$(env_for $cfg)
  for s in $(seq 0 $((NS-1))); do
    flag=/edrive2/qiuyid/ladder_${cfg}_s${s}.done
    [ -f "$flag" ] && continue
    # wait for a free allowed GPU
    while :; do
      launched=0
      for g in 2 3 4 5 6; do
        [ -n "${pid[$g]}" ] && kill -0 ${pid[$g]} 2>/dev/null && continue
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
        if [ "$free" -ge 20000 ] && [ "$util" -le 45 ]; then
          ( CUDA_VISIBLE_DEVICES=$g SC_LINEAR_GRANULARITY=per_row SC_HALVE=1 SC_MP_FIXED_PREC=1 SC_PREC=8 \
              SC_SMOOTH_SCALES=$SQ env $extra \
              $PY evaluate/eval_local_n_samples.py \
                --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
                --tag ladder_${cfg} --keys_file $KEYS --num_samples $N \
                --shard $s --num_shards $NS >/edrive2/qiuyid/ladder_${cfg}_s${s}.log 2>&1 \
            && touch $flag ) &
          pid[$g]=$!; launched=1; break
        fi
      done
      [ $launched -eq 1 ] && break
      sleep 45
    done
  done
done
wait
echo "SC_LADDER_ALL_DONE"
