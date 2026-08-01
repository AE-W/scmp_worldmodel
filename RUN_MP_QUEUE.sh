#!/bin/bash
# 换机器跑 MP 四栏表 + 三条备选方案的自包含脚本。
# 前置: conda env scmp 已建 (见 RUNNING_ON_SLURM.md §0), 仓库已 clone,
#   robotdata 软链已建, results/ 下已有:
#     mp_error_grid_sq2.npz  block_gamma.npz  smoothquant_scales.pt
#     diverse_10.json  diverse_100.json  final_sc_recipe.json
#   以及 mp_fractions_sc_{g758,g658,g632,g600}.json (Gamma块级配置)
#     step_sched_Wt.json (时间步W_t调度)
#   —— 这些小产物(~5MB)已直接提交进本仓库 results/, git pull 即得,无需另拷。
#   数据集 robotdata(133GB, bridge eval 子集): 若本机没有, 见 RUNNING_ON_SLURM.md §0.3
#   下载, 或从已有机器 rsync。HF token 放 ~/hf_token.txt 用于回传。
#
# 用法: 每个函数是一个独立作业, 挑没跑完的投。SC ~350s/样本, n=10 单档 ~1h。
set -e
cd "$(dirname "$0")"
export PYTHONPATH=. BRIDGE_ROOT=$PWD/robotdata/opensource_robotdata/bridge
export EVAL_OUT_ROOT=$PWD/results/local_n_eval
SKIP=$(python -c "import json;print(json.load(open('results/final_sc_recipe.json'))['skip'])")
COMMON="SC_LINEAR_GRANULARITY=per_row SC_HALVE=1 SC_MP_FIXED_PREC=1 SC_PREC=8 SC_SMOOTH_SCALES=$PWD/results/smoothquant_scales.pt"

# ---- 四栏 gblock MP (块间 Γ), n=10, 同批 diverse_10 ----
run_mp() {   # $1=tag(g758/g658/g632/g600)
  CAL=results/mp_fractions_sc_$1.json
  export SC_MP_CONFIG=$(python -c "import json;d=json.load(open('$CAL'));print(json.dumps({'stoc_len_levels':d['stoc_len_levels'],'level_fractions':d['level_fractions']},separators=(',',':')))")
  export SC_MP_PER_MODULE=$PWD/$CAL
  env $COMMON python evaluate/eval_local_n_samples.py \
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
    --tag scr10_$1 --keys_file results/diverse_10.json --num_samples 10 \
    --shard 0 --num_shards 1 --inference_steps 50 --scheduler PNDM
  unset SC_MP_CONFIG SC_MP_PER_MODULE
}
# ---- 四栏对应 uniform ----
run_uni() { # $1=tag(u758/u658/u632/u600) $2=折半流长(96/48/40/32)
  env $COMMON SC_UNIFORM_STOC_LEN=$2 python evaluate/eval_local_n_samples.py \
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
    --tag scr10_$1 --keys_file results/diverse_10.json --num_samples 10 \
    --shard 0 --num_shards 1 --inference_steps 50 --scheduler PNDM
}
# ---- 备选1: 时间步 W_t 调度 (7.58位, 平均96) ----
run_stepWt() {
  export SC_STEP_SCHEDULE=$(cat results/step_sched_Wt.json)
  env $COMMON python evaluate/eval_local_n_samples.py \
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
    --tag scr10_stepWt --keys_file results/diverse_10.json --num_samples 10 \
    --shard 0 --num_shards 1 --inference_steps 50 --scheduler PNDM
  unset SC_STEP_SCHEDULE
}
# ---- 备选2: 完整机制(阈值+保护通道) 先校准再评测 ----
run_fullmp() {
  python evaluate/calibrate_full_mp.py --phase both   # 产 mp_fractions_sc_avg192_full.json
  run_mp_generic sc_avg192_full scr10_fullmp
}
run_mp_generic() { # $1=配置名 $2=tag
  CAL=results/mp_fractions_$1.json
  export SC_MP_CONFIG=$(python -c "import json;d=json.load(open('$CAL'));print(json.dumps({'stoc_len_levels':d['stoc_len_levels'],'level_fractions':d['level_fractions']},separators=(',',':')))")
  export SC_MP_PER_MODULE=$PWD/$CAL
  env $COMMON python evaluate/eval_local_n_samples.py \
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml --skip "$SKIP" \
    --tag $2 --keys_file results/diverse_10.json --num_samples 10 \
    --shard 0 --num_shards 1 --inference_steps 50 --scheduler PNDM
  unset SC_MP_CONFIG SC_MP_PER_MODULE
}

# 挑一个跑, 或全跑(串行, 每个~1h):
case "${1:-help}" in
  g758) run_mp g758;;  g658) run_mp g658;;  g632) run_mp g632;;  g600) run_mp g600;;
  u758) run_uni u758 96;; u658) run_uni u658 48;; u632) run_uni u632 40;; u600) run_uni u600 32;;
  stepWt) run_stepWt;;  fullmp) run_fullmp;;
  upload)   # 把所有 scr10_* 结果 json 回传 HF (BDXXN/scmp-worldmodel-progress)
    export SCMP_HF_TOKEN_FILE=${SCMP_HF_TOKEN_FILE:-$HOME/hf_token.txt}
    export SCMP_ROOT=$PWD SCMP_RESULTS=$PWD/results
    python evaluate/hf_backup.py ;;
  all)      # 串行跑所有未完成档 + 回传 (每档~1h, SC 350s/样本)
    for t in g758 g658 g632 g600; do [ -d results/local_n_eval/scr10_$t/metrics ] && [ $(ls results/local_n_eval/scr10_$t/metrics|wc -l) -ge 10 ] || run_mp $t; done
    run_uni u758 96; run_uni u658 48; run_uni u632 40; run_uni u600 32
    run_stepWt; run_fullmp
    $0 upload ;;
  *) echo "用法: $0 {g758|g658|g632|g600|u758|u658|u632|u600|stepWt|fullmp|upload|all}";;
esac
