"""GPU auto-grab scheduler for the leave-one-IN op-level sensitivity sweep
(aligned with the repo's original sensitivity_sweep.py method).

Configs: fp reference + 6 ops x 28 blocks = 169. Each runs the 300 diverse
samples. Network is ~all-FP per config (~30-40s/sample), so chunks are large.
Per-sample result files make everything resume-safe.

Run under nohup:
  cd /home/qiuyid/scmp_worldmodel && PYTHONPATH=. nohup \
    /home/qiuyid/.conda/envs/scmp/bin/python evaluate/gpu_scheduler_loi.py \
    > /edrive2/qiuyid/sens300_loi_sched.log 2>&1 & disown
"""
import json, os, subprocess, time

ROOT = "/home/qiuyid/scmp_worldmodel"
PY = "/home/qiuyid/.conda/envs/scmp/bin/python"
CONFIG = "configs/evaluation/bridge/frame_ada_sc_full.yaml"
KEYS_FILE = f"{ROOT}/results/diverse_300.json"
OUT_DIR = "/edrive2/qiuyid/sens300_loi"
BRIDGE = "/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge"
DEPTH = 28
OPS = ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2")
CHUNK = 100                                   # ~35s/sample -> ~1h/task
FREE_MIN_MIB = 22000
UTIL_MAX = 40
POLL_SEC = 30

os.chdir(ROOT)
os.makedirs(OUT_DIR, exist_ok=True)
keys = json.load(open(KEYS_FILE))
N = len(keys)

CONFIGS = [("fp", -1)] + [(op, b) for op in OPS for b in range(DEPTH)]
TASKS = []
for op, b in CONFIGS:
    for s in range(0, N, CHUNK):
        TASKS.append((op, b, s, min(s + CHUNK, N)))


def tag(op, b):
    return "fp_ref" if op == "fp" else f"{op}_{b}"


def task_done(t):
    op, b, s, e = t
    d = os.path.join(OUT_DIR, f"loi_{tag(op, b)}")
    return all(os.path.exists(os.path.join(d, f"{keys[i]}.json")) for i in range(s, e))


def gpu_status():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], timeout=30).decode()
    except Exception:
        return {}
    st = {}
    for line in out.strip().splitlines():
        i, free, util = [x.strip() for x in line.split(",")]
        st[int(i)] = (int(free), int(util))
    return st


def launch(gpu, task):
    op, b, s, e = task
    log = open(os.path.join(OUT_DIR, f"sched_g{gpu}_{tag(op, b)}_{s}.log"), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), BRIDGE_ROOT=BRIDGE, PYTHONPATH=".")
    return subprocess.Popen(
        [PY, "evaluate/sensitivity_shard_loi.py", "--config", CONFIG,
         "--op", op, "--block", str(b), "--keys_file", KEYS_FILE,
         "--key_start", str(s), "--key_end", str(e),
         "--inference_steps", "50", "--scheduler", "PNDM", "--out_dir", OUT_DIR],
        env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    running = {}
    print(f"LOI scheduler: {len(TASKS)} tasks ({len(CONFIGS)} configs x {N}/{CHUNK})", flush=True)
    while True:
        for g in list(running):
            if running[g][0].poll() is not None:
                running.pop(g)
        pending = [t for t in TASKS if not task_done(t)]
        if not pending and not running:
            print("ALL LOI TASKS DONE", flush=True)
            break
        busy = {t for _, t in running.values()}
        for g, (free, util) in sorted(gpu_status().items()):
            if g in running:
                continue
            if free >= FREE_MIN_MIB and util <= UTIL_MAX:
                nxt = next((t for t in pending if t not in busy), None)
                if nxt is None:
                    break
                running[g] = (launch(g, nxt), nxt)
                busy.add(nxt)
                op, b, s, e = nxt
                print(f"[{time.strftime('%m-%d %H:%M:%S')}] GPU{g} <- {tag(op,b)} [{s}:{e}] "
                      f"(free={free} util={util})  running={len(running)} pending={len(pending)-1}",
                      flush=True)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
