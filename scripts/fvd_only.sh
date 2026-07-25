#!/bin/bash
# FVD for both full lines + naive-int8 attribution run (FID reused from json).
cd /home/qiuyid/scmp_worldmodel
export PYTHONPATH=.:pytorch-fid/src
export BRIDGE_ROOT=/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge
PY=/home/qiuyid/.conda/envs/scmp/bin/python

run() {  # tag  fid
  echo "=== FVD: $1 ==="
  CUDA_VISIBLE_DEVICES=6 $PY evaluate/compute_fid_fvd_stream.py \
    --pred_dir results/local_n_eval/$1/videos \
    --out results/fidfvd_$1.json --fid_precomputed $2 2>&1 \
    | grep -E "^FID|^FVD|skipped|pred=|\[gt\]|\[pred\]|Error|Traceback"
}
run final_fp_full 4.985
run final_sc_full 10.852
run naive_int8_300 21.586

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
   "naive_int8_300_attribution":agg("naive_int8_300"),
   "recipe":rec["env"]|{"skip":rec["skip"]}}
json.dump(t,open("results/BRIDGE_FINAL_TABLE.json","w"),indent=2,ensure_ascii=False)
print(json.dumps(t,indent=2,ensure_ascii=False))
PYEOF
echo "FVD_ALL_DONE"
