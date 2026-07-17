"""Final SC recipe decider: combine every optimization that adds value.

Waits for: results/skip_winner.json (skip arbitration) and the nb_sq probe
(SmoothQuant on per_row+halve). If SQ is positive, runs one combined probe
(SQ + winning skip) so the exact final recipe is validated end-to-end before
committing the multi-day full run. Writes results/final_sc_recipe.json —
the gated gpu_scheduler_final.py picks it up and launches the SC line.

Run:  cd worldmodel && PYTHONPATH=. nohup python evaluate/final_recipe_decider.py &
"""
import json, os, subprocess, time

ROOT = "/home/qiuyid/scmp_worldmodel"
PY = "/home/qiuyid/.conda/envs/scmp/bin/python"
BRIDGE = f"{ROOT}/robotdata/opensource_robotdata/bridge"
SQ_FILE = f"{ROOT}/results/smoothquant_scales.pt"
os.chdir(ROOT)


def mean_of(tag):
    f = f"results/line_eval/{tag}/summary_shard_0.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return d["mean_psnr"], -d["mean_l2"]


def wait_for(path, proc_pat, label):
    while not os.path.exists(path):
        r = subprocess.run(["pgrep", "-f", proc_pat], capture_output=True)
        if r.returncode != 0:
            print(f"[decider] {label}: process gone without result", flush=True)
            return os.path.exists(path)
        time.sleep(60)
    return True


def run_probe(tag, skip, use_sq):
    env = dict(os.environ, BRIDGE_ROOT=BRIDGE, EVAL_OUT_ROOT=f"{ROOT}/results/line_eval",
               PYTHONPATH=".", CUDA_VISIBLE_DEVICES="3",
               SC_LINEAR_GRANULARITY="per_row", SC_HALVE="1")
    if use_sq:
        env["SC_SMOOTH_SCALES"] = SQ_FILE
    cmd = [PY, "evaluate/eval_local_n_samples.py",
           "--config", "configs/evaluation/bridge/frame_ada_sc_full.yaml",
           "--tag", tag, "--num_samples", "8", "--inference_steps", "10", "--scheduler", "DPM"]
    if skip:
        cmd += ["--skip", skip]
    print(f"[decider] probing {tag} ...", flush=True)
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mean_of(tag)


def main():
    wait_for("results/skip_winner.json", "tag nb_skip17_final", "skip winner")
    wait_for("results/line_eval/nb_sq/summary_shard_0.json", "tag nb_sq", "sq probe")

    sw = json.load(open("results/skip_winner.json")) if os.path.exists("results/skip_winner.json") \
        else {"winner": "q_perrow_halve", "skip": ""}
    base = mean_of("q_perrow_halve")
    sq = mean_of("nb_sq")
    skip_str = sw["skip"]
    use_sq = bool(sq and base and sq[0] > base[0])
    print(f"[decider] base={base} sq={sq} use_sq={use_sq} skip_winner={sw['winner']}", flush=True)

    # 用户指令: full 线必须带 skip。候选只在"带 skip"的变体里选,
    # 唯一自由度是 SmoothQuant 开/关(组合探针实测决定)。
    cands = {}
    cands[(False, skip_str)] = mean_of(sw["winner"])
    if use_sq:
        cands[(True, skip_str)] = run_probe("nb_sq_skip", skip_str, True)

    best = max((v[0], v[1], k) for k, v in cands.items() if v)
    use_sq_f, skip_f = best[2]
    env = {"SC_LINEAR_GRANULARITY": "per_row", "SC_HALVE": "1"}
    if use_sq_f:
        env["SC_SMOOTH_SCALES"] = SQ_FILE
    recipe = {"env": env, "skip": skip_f,
              "decision_table": {f"sq={k[0]},skip={bool(k[1])}": v for k, v in cands.items() if v},
              "chosen": {"sq": use_sq_f, "skip": bool(skip_f), "psnr": best[0]}}
    json.dump(recipe, open("results/final_sc_recipe.json", "w"), indent=2)
    print(f"[decider] RECIPE WRITTEN: sq={use_sq_f} skip='{skip_f}' psnr={best[0]}", flush=True)


if __name__ == "__main__":
    main()
