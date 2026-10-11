#!/bin/bash
# Bridge finalization: FVD for both full lines (FID already computed), plus a
# naive-int8 300-clip run to attribute how much of the FID gap is SC noise vs
# plain int8 quantization. Rebuilds results/BRIDGE_FINAL_TABLE.json at the end.
cd /home/qiuyid/scmp_worldmodel
export PYTHONPATH=.:pytorch-fid/src
export BRIDGE_ROOT=/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge
export EVAL_OUT_ROOT=/home/qiuyid/scmp_worldmodel/results/local_n_eval
PY=/home/qiuyid/.conda/envs/scmp/bin/python

echo "[1/4] FVD: FP line"
CUDA_VISIBLE_DEVICES=6 $PY evaluate/compute_fid_fvd_stream.py \
  --pred_dir results/local_n_eval/final_fp_full/videos \
  --out results/fidfvd_final_fp_full.json --fid_precomputed 4.985 2>&1 | grep -E "^FID|^FVD|skipped|pred="

echo "[2/4] FVD: SC line"
CUDA_VISIBLE_DEVICES=6 $PY evaluate/compute_fid_fvd_stream.py \
  --pred_dir results/local_n_eval/final_sc_full/videos \
  --out results/fidfvd_final_sc_full.json --fid_precomputed 10.852 2>&1 | grep -E "^FID|^FVD|skipped|pred="

echo "[3/4] naive int8 x300 (FID attribution)"
CUDA_VISIBLE_DEVICES=6 $PY evaluate/eval_local_n_samples.py \
  --config configs/evaluation/bridge/frame_ada_sc_full.yaml --naive_int8 \
  --tag naive_int8_300 --num_samples 300 2>&1 | grep -E "DONE" | tail -1

echo "[4/4] FID/FVD for naive int8 + rebuild table"
CUDA_VISIBLE_DEVICES=6 $PY evaluate/compute_fid_fvd_stream.py \
  --pred_dir results/local_n_eval/naive_int8_300/videos \
  --out results/fidfvd_naive300.json 2>&1 | grep -E "^FID|^FVD|pred="

$PY - <<'PYEOF'
import json, glob, os
def agg(tag):
    ms=[json.load(open(f)) for f in glob.glob(f"results/local_n_eval/{tag}/metrics/*.json")]
    ok=[m for m in ms if m.get("psnr")]; n=len(ok)
    r={"n":n,"latent_l2":round(sum(m['latent_l2'] for m in ok)/n,4),
       "psnr":round(sum(m['psnr'] for m in ok)/n,2),"ssim":round(sum(m['ssim'] for m in ok)/n,3)}
    f=f"results/fidfvd_{tag}.json"
    if os.path.exists(f):
        d=json.load(open(f)); r["fid"]=d.get("fid"); r["fvd"]=d.get("fvd")
    return r
rec=json.load(open("results/final_sc_recipe.json"))
t={"dataset":"bridge test 2946, PNDM50",
   "FP_baseline":agg("final_fp_full"),
   "SC_final_recipe":agg("final_sc_full"),
   "recipe":rec["env"]|{"skip":rec["skip"]}}
if os.path.exists("results/local_n_eval/naive_int8_300/metrics"):
    t["naive_int8_300_attribution"]=agg("naive_int8_300")
json.dump(t,open("results/BRIDGE_FINAL_TABLE.json","w"),indent=2,ensure_ascii=False)
print(json.dumps(t,indent=2,ensure_ascii=False))
PYEOF
echo "BRIDGE_FINISH_ALL_DONE"
