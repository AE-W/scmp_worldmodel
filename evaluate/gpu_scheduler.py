"""GPU auto-grab scheduler for the n=300 per-block sensitivity sweep.

Watches every GPU; whenever one is free (enough free memory + low util and not
already ours), it launches the next pending task on it. Tasks are (block,
key-slice) units; per-sample result files make everything resume-safe, so the
scheduler can die/restart or re-dispatch freely without recomputation.

Blocks: -1 (reference, no skip) + 0..27  => 29 configs.
Each config runs over the 300 diverse samples, sliced into CHUNK-sized tasks.

Run under nohup so it survives the conversation:
  cd /home/qiuyid/scmp_worldmodel && nohup PYTHONPATH=. \
    /home/qiuyid/.conda/envs/scmp/bin/python evaluate/gpu_scheduler.py \
    > /edrive2/qiuyid/sens300_sched.log 2>&1 & disown
"""
import json, os, subprocess, time

ROOT = "/home/qiuyid/scmp_worldmodel"
PY = "/home/qiuyid/.conda/envs/scmp/bin/python"
CONFIG = "configs/evaluation/bridge/frame_ada_sc_full.yaml"
KEYS_FILE = f"{ROOT}/results/diverse_300.json"
OUT_DIR = "/edrive2/qiuyid/sens300"          # persistent (edrive2)
BRIDGE = "/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge"
DEPTH = 28
CHUNK = 15                                    # samples per task (~3h each @PNDM50)
FREE_MIN_MIB = 22000                          # need ~20G for eval + margin
UTIL_MAX = 40                                 # don't steal a GPU someone is actively using
POLL_SEC = 30

os.chdir(ROOT)
os.makedirs(OUT_DIR, exist_ok=True)
keys = json.load(open(KEYS_FILE))
N = len(keys)
BLOCKS = [-1] + list(range(DEPTH))            # -1 = reference

TASKS = []                                    # (block, start, end)
for b in BLOCKS:
    for s in range(0, N, CHUNK):
        TASKS.append((b, s, min(s + CHUNK, N)))


def task_done(t):
    b, s, e = t
    bdir = os.path.join(OUT_DIR, f"block_{b}")
    return all(os.path.exists(os.path.join(bdir, f"{keys[i]}.json")) for i in range(s, e))


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
    b, s, e = task
    log = open(os.path.join(OUT_DIR, f"sched_g{gpu}_b{b}_{s}.log"), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), BRIDGE_ROOT=BRIDGE, PYTHONPATH=".")
    proc = subprocess.Popen(
        [PY, "evaluate/sensitivity_shard.py", "--config", CONFIG,
         "--block", str(b), "--keys_file", KEYS_FILE,
         "--key_start", str(s), "--key_end", str(e),
         "--inference_steps", "50", "--scheduler", "PNDM", "--out_dir", OUT_DIR],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc


def main():
    running = {}   # gpu -> (proc, task)
    print(f"scheduler start: {len(TASKS)} tasks ({len(BLOCKS)} blocks x {N} samples / {CHUNK})", flush=True)
    while True:
        # reap finished
        for g in list(running):
            proc, _ = running[g]
            if proc.poll() is not None:
                running.pop(g)
        pending = [t for t in TASKS if not task_done(t)]
        if not pending and not running:
            print("ALL TASKS DONE", flush=True)
            break
        busy_tasks = {t for _, t in running.values()}
        st = gpu_status()
        for g, (free, util) in sorted(st.items()):
            if g in running:
                continue
            if free >= FREE_MIN_MIB and util <= UTIL_MAX:
                nxt = next((t for t in pending if t not in busy_tasks), None)
                if nxt is None:
                    break
                proc = launch(g, nxt)
                running[g] = (proc, nxt)
                busy_tasks.add(nxt)
                b, s, e = nxt
                print(f"[{time.strftime('%m-%d %H:%M:%S')}] GPU{g} <- block{b} keys[{s}:{e}]  "
                      f"(free={free}MiB util={util}%)  running={len(running)} pending={len(pending)-1}",
                      flush=True)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
