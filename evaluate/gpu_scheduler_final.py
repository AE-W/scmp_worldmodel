"""GPU auto-grab scheduler for the FINAL full-test two lines (bridge, 2946):

  line 1: FP baseline           (frame_ada.yaml, math)      — dispatches immediately
  line 2: SC final recipe       (frame_ada_sc_full.yaml)    — GATED on recipe file

The SC line waits until results/final_sc_recipe.json exists, so the recipe
(env + skip string) can be finalized (LOI final top-17 + SmoothQuant/chunk_d
probe verdicts) without holding up the FP line. Recipe file schema:
    {"env": {"SC_LINEAR_GRANULARITY": "per_row", "SC_HALVE": "1", ...},
     "skip": "mlp_fc1=...;mlp_fc2=...;qkv=..."}

Both lines run PNDM 50-step over all test clips, sliced into NUM_SHARDS
shard-tasks (eval_local_n_samples --shard/--num_shards). Per-sample metric
files enable resume, so tasks can be re-dispatched freely.

  cd /home/qiuyid/scmp_worldmodel && PYTHONPATH=. nohup \
    /home/qiuyid/.conda/envs/scmp/bin/python evaluate/gpu_scheduler_final.py \
    > /edrive2/qiuyid/final_sched.log 2>&1 & disown
"""
import json, os, subprocess, time

ROOT = "/home/qiuyid/scmp_worldmodel"
PY = "/home/qiuyid/.conda/envs/scmp/bin/python"
BRIDGE = f"{ROOT}/robotdata/opensource_robotdata/bridge"
OUT_ROOT = f"{ROOT}/results/local_n_eval"
RECIPE_FILE = f"{ROOT}/results/final_sc_recipe.json"
GT_LATENT_DIR = f"{BRIDGE}/evaluation_latent_videos/test_sample_latent_videos"
NUM_SHARDS = 24
FREE_MIN_MIB = 22000
UTIL_MAX = 40
POLL_SEC = 45
# 用户指定(2026-07-17, 按算力空闲修订): 排除别人算力占用最高的 GPU1/GPU7
# (各被实跑 ~50% SM)。GPU0 上 vggt 只囤显存不占算力,拿回来用。
EXCLUDE_GPUS = {1, 7}

os.chdir(ROOT)
ALL_FILES = sorted(f for f in os.listdir(GT_LATENT_DIR) if f.endswith(".pt"))
N = len(ALL_FILES)

LINES = [
    {"tag": "final_fp_full", "config": "configs/evaluation/bridge/frame_ada.yaml",
     "env": {}, "skip": "", "gated": False},
    {"tag": "final_sc_full", "config": "configs/evaluation/bridge/frame_ada_sc_full.yaml",
     "env": None, "skip": None, "gated": True},   # filled from RECIPE_FILE
]


def shard_keys(shard):
    return [f[:-3] for f in ALL_FILES[shard::NUM_SHARDS]]


def task_done(line, shard):
    met = os.path.join(OUT_ROOT, line["tag"], "metrics")
    return all(os.path.exists(os.path.join(met, f"{k}.json")) for k in shard_keys(shard))


def gpu_status():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], timeout=30).decode()
    except Exception:
        return {}
    st = {}
    for ln in out.strip().splitlines():
        i, free, util = [x.strip() for x in ln.split(",")]
        st[int(i)] = (int(free), int(util))
    return st


def load_recipe(line):
    if not line["gated"] or line["env"] is not None:
        return True
    if not os.path.exists(RECIPE_FILE):
        return False
    r = json.load(open(RECIPE_FILE))
    line["env"] = r.get("env", {})
    line["skip"] = r.get("skip", "")
    print(f"[recipe] SC line armed: env={line['env']} skip='{line['skip']}'", flush=True)
    return True


def launch(gpu, line, shard):
    log = open(f"/edrive2/qiuyid/final_{line['tag']}_s{shard}.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), BRIDGE_ROOT=BRIDGE,
               EVAL_OUT_ROOT=OUT_ROOT, PYTHONPATH=".", **line["env"])
    cmd = [PY, "evaluate/eval_local_n_samples.py", "--config", line["config"],
           "--tag", line["tag"], "--num_samples", str(N),
           "--shard", str(shard), "--num_shards", str(NUM_SHARDS)]
    if line["skip"]:
        cmd += ["--skip", line["skip"]]
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    running = {}
    print(f"final scheduler: {N} clips x {len(LINES)} lines, {NUM_SHARDS} shards each", flush=True)
    while True:
        for g in list(running):
            if running[g][0].poll() is not None:
                running.pop(g)
        pending = []
        for li, line in enumerate(LINES):
            if line["gated"] and not load_recipe(line):
                continue
            for s in range(NUM_SHARDS):
                if not task_done(line, s):
                    pending.append((li, s))
        if not pending and not running:
            print("ALL FINAL TASKS DONE", flush=True)
            break
        busy = {t for _, t in running.values()}
        for g, (free, util) in sorted(gpu_status().items()):
            if g in EXCLUDE_GPUS or g in running:
                continue
            if free >= FREE_MIN_MIB and util <= UTIL_MAX:
                nxt = next((t for t in pending if t not in busy), None)
                if nxt is None:
                    break
                li, s = nxt
                running[g] = (launch(g, LINES[li], s), nxt)
                busy.add(nxt)
                print(f"[{time.strftime('%m-%d %H:%M:%S')}] GPU{g} <- {LINES[li]['tag']} shard{s} "
                      f"running={len(running)} pending={len(pending)-1}", flush=True)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
