"""Search MP stoc_len level sets instead of hand-picking them.

The 3-level {ref, 0.75ref, 0.5ref} shape used by the HPCA table spans only 2x,
which bounds how much budget mixed precision can move: at a 96-cycle budget at
most 50% of rows can reach 128. The scmp_llm ladder instead uses 5-8 irregular
levels per (model, target) - values like [111, 85, 49, 48, 33] that are clearly
searched, not derived from a rule.

This does that search. Phase 1 measures the real per-row SC error on a DENSE
grid of candidate stream lengths (one GPU pass). Phase 2 is pure CPU: for a
given budget and level count k, every candidate subset is scored by running the
same Lagrangian assignment the calibrator uses, and the subset with the lowest
assigned error wins. Scoring a subset costs a few ms, so the subsets can be
enumerated exhaustively.

    # phase 1 (GPU, ~20 min)
    python evaluate/search_mp_levels.py measure --out results/mp_error_grid.npz

    # phase 2 (CPU, seconds)
    python evaluate/search_mp_levels.py search --grid results/mp_error_grid.npz \
        --budget 96 --k 3,5,7 --out results/mp_levels_search_b96.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

# Dense candidate grid, in HALVED cycle counts (see the HPCA note in
# calibrate_mp_fractions.py: a level value IS the cycle count). 128 is the
# ceiling - the uniform int8 rung - so nothing above it is searched.
DEFAULT_GRID = [16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
# Every integer stream length in [16, 128]. Measuring this costs one longer
# GPU pass but lets the refinement land on off-grid values like 111 or 49.
DENSE_GRID = list(range(16, 129))


def measure(cli):
    """Phase 1: per-row SC error at every candidate stream length."""
    import torch
    from diffusers.models import AutoencoderKL
    from scmp_kernels.sc import sc_matmul
    from scmp_kernels.sc.config_helpers import make_sobol_simple_config
    from models.sc_integration import reconfigure, get_config
    from evaluate.calibrate_mp_fractions import (
        TARGET_SUFFIXES, build_args, load_model, make_pipe,
        GT_LATENT_DIR, ANNOT_DIR, C_ACT_SCALER, SEQUENCE_LENGTH,
        compute_actions_for_slice,
    )

    grid = [int(x) for x in cli.grid.split(",")] if cli.grid else DEFAULT_GRID
    args = build_args(cli.eval_config, cli.inference_steps)
    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)
    reconfigure(args.attention_mode)
    cfg = get_config()
    for f in ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2"):
        setattr(cfg, f"enable_{f}", False)          # teacher pass stays FP

    caps: dict[str, list] = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES):
            def mk(n):
                def hook(_m, inp, _out):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                    keep = caps.setdefault(n, [])
                    if sum(t.shape[0] for t in keep) < 512:
                        idx = torch.randperm(x.shape[0], device=x.device)[:64]
                        keep.append(x[idx].cpu())
                return hook
            hooks.append(mod.register_forward_hook(mk(name)))

    pipe = make_pipe(args, vae, model, "DPM")
    for k in json.load(open(cli.keys_file))[: cli.num_samples]:
        parts = k.split("_")
        eid, start = "_".join(parts[:-2]), int(parts[-1])
        ann = json.load(open(os.path.join(ANNOT_DIR, f"{eid}.json")))
        if start + SEQUENCE_LENGTH > len(ann["state"]):
            continue
        gt = torch.load(os.path.join(GT_LATENT_DIR, f"{k}.pt"),
                        weights_only=False, map_location=device)
        arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
        grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
        act = torch.from_numpy(
            compute_actions_for_slice(arm, grip) * C_ACT_SCALER).float().unsqueeze(0)
        with torch.no_grad():
            pipe(act.to(device).float(), mask_x=gt[0:1].unsqueeze(0).to(device).float(),
                 video_length=args.num_frames, height=args.video_size[0],
                 width=args.video_size[1], num_inference_steps=cli.inference_steps,
                 guidance_scale=args.guidance_scale, device=device,
                 return_dict=False, output_type="latent_only")
        print(f"  teacher pass {k}", flush=True)
    for h in hooks:
        h.remove()

    named = dict(model.named_modules())
    all_err, all_module, order = [], [], []
    for name, chunks in caps.items():
        X = torch.cat(chunks, 0)[: cli.max_rows].to(device)
        W = named[name].weight.detach().float()
        ref_out = X @ W.t()
        errs = []
        for L in grid:
            with torch.no_grad():
                y = sc_matmul(X, W, granularity="per_row", mode="bipolar",
                              sc_prec=8, stoc_len=L,
                              config=make_sobol_simple_config(X.shape[-1], X.shape[-1], 8),
                              halve_bipolar_stoc_len=True)
            errs.append(((y - ref_out) ** 2).mean(dim=-1).sqrt().cpu().numpy())
        all_err.append(np.stack(errs, 1))
        all_module.append(np.full(X.shape[0], len(order)))
        order.append(name)
        print(f"  error curve {name}: rows={X.shape[0]} levels={len(grid)}", flush=True)

    E = np.concatenate(all_err, 0)
    M = np.concatenate(all_module, 0)
    np.savez_compressed(cli.out, errors=E, module_idx=M,
                        grid=np.array(grid), modules=np.array(order))
    print(f"wrote {cli.out}: E{E.shape} over grid {grid}", flush=True)


def _assign(errors, costs, budget_total):
    """Lagrangian per-row level assignment (same solver as the calibrator)."""
    n = errors.shape[0]
    if budget_total <= n * costs[-1]:
        return np.full(n, len(costs) - 1, dtype=np.int64)
    if budget_total >= n * costs[0]:
        return np.zeros(n, dtype=np.int64)

    def solve(lmbd):
        a = (errors + lmbd * costs[None, :]).argmin(axis=1)
        return a, float(costs[a].sum())

    lo, hi = 0.0, 1.0
    _, c = solve(hi)
    while c > budget_total and hi < 1e6:
        hi *= 2.0
        _, c = solve(hi)
    best = np.zeros(n, dtype=np.int64)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        best, c = solve(mid)
        if c > budget_total:
            lo = mid
        else:
            hi = mid
    return best


def score(E_sub, costs, budget_per_row):
    """Mean assigned error for one candidate level set. Lower is better."""
    n = E_sub.shape[0]
    a = _assign(E_sub, np.asarray(costs, dtype=np.float64), budget_per_row * n)
    return float(E_sub[np.arange(n), a].mean()), a


def search(cli):
    """Phase 2: exhaustive subset search over the measured grid."""
    d = np.load(cli.grid_file, allow_pickle=True)
    E_full, grid = d["errors"], [int(x) for x in d["grid"]]
    # Scoring a subset costs one Lagrangian solve over every row, and there are
    # up to ~18k subsets per (budget, k). Subsampling rows keeps the ranking
    # intact — the score is a mean over rows — while making the exhaustive
    # search minutes instead of hours. Sampling is deterministic.
    if cli.max_search_rows and E_full.shape[0] > cli.max_search_rows:
        rs = np.random.RandomState(0)
        idx = rs.choice(E_full.shape[0], cli.max_search_rows, replace=False)
        E_full = E_full[np.sort(idx)]
        print(f"  subsampled {cli.max_search_rows} of {d['errors'].shape[0]} rows "
              f"for the subset search", flush=True)
    budget = cli.budget
    ks = [int(x) for x in cli.k.split(",")]

    # A level set must bracket the budget, otherwise the assignment collapses
    # onto one level and the schedule degenerates to uniform.
    feasible = [i for i, g in enumerate(grid)]
    out = {"budget": budget, "grid": grid, "n_rows": int(E_full.shape[0]), "results": {}}

    for k in ks:
        best = None
        n_tried = 0
        for combo in itertools.combinations(feasible, k):
            lv = [grid[i] for i in combo][::-1]          # descending
            if not (min(lv) < budget < max(lv)):
                continue
            n_tried += 1
            s, _ = score(E_full[:, list(combo)][:, ::-1], lv, budget)
            if best is None or s < best[0]:
                best = (s, lv)
        if best is None:
            print(f"k={k}: no feasible set brackets budget {budget}")
            continue
        s, lv = best
        a = score(E_full[:, [grid.index(x) for x in lv]], lv, budget)[1]
        counts = np.bincount(a, minlength=k)
        fr = (counts / counts.sum()).tolist()
        # Rounding each fraction independently makes the sum drift off 1.0 by
        # ~1e-4, and MPConfig rejects anything outside its tolerance — with more
        # levels the drift grows, so the k=6/7/8 sets failed at import time and
        # the job exited 0 before loading the model (Slurm reported COMPLETED).
        # Absorb the residual into the largest fraction.
        fr_r = [round(f, 4) for f in fr]
        j = max(range(len(fr_r)), key=lambda i: fr_r[i])
        fr_r[j] = round(fr_r[j] + (1.0 - sum(fr_r)), 10)
        out["results"][str(k)] = {
            "levels": lv,
            "level_fractions": fr_r,
            "achieved_avg_cycles": round(float(np.dot(fr, lv)), 2),
            "mean_assigned_error": round(s, 6),
            "n_subsets_tried": n_tried,
        }
        print(f"k={k}: levels={lv} fractions={[round(f,3) for f in fr]} "
              f"err={s:.6f} (searched {n_tried} subsets)", flush=True)

    # Reference points: what the current hand-picked sets score on the same data.
    for name, lv in (cli.compare or {}).items():
        if all(x in grid for x in lv) and min(lv) < budget < max(lv):
            s, a = score(E_full[:, [grid.index(x) for x in lv]], lv, budget)
            out.setdefault("reference", {})[name] = {
                "levels": lv, "mean_assigned_error": round(s, 6)}
            print(f"  reference {name}: levels={lv} err={s:.6f}", flush=True)

    json.dump(out, open(cli.out, "w"), indent=2)
    print(f"wrote {cli.out}", flush=True)


def refine(cli):
    """Coordinate-descent refinement of a level set on a DENSE measured grid.

    The subset search picks the best k values out of a coarse grid, so its
    answer is quantised to that grid: it can return 112 but never 111. The
    scmp_llm ladder is full of off-grid values (111, 85, 49, 47, 46, 45, 30,
    19), which is what a search over the values themselves produces. Exhaustive
    subset search over every integer stream length is infeasible - C(113, 4) is
    6.8M - so this starts from the coarse-grid optimum and moves one level at a
    time to its best nearby value, in the same spirit as the boundary search in
    scmp_kernels.mp.auto_calibrator.
    """
    d = np.load(cli.grid_file, allow_pickle=True)
    E_full, grid = d["errors"], [int(x) for x in d["grid"]]
    if cli.max_search_rows and E_full.shape[0] > cli.max_search_rows:
        rs = np.random.RandomState(0)
        E_full = E_full[np.sort(rs.choice(E_full.shape[0], cli.max_search_rows, replace=False))]
    gi = {g: i for i, g in enumerate(grid)}
    budget = cli.budget
    levels = sorted((int(x) for x in cli.init.split(",")), reverse=True)
    for x in levels:
        if x not in gi:
            raise SystemExit(f"init level {x} not measured; grid is {grid}")

    def sc(lv):
        if not (min(lv) < budget < max(lv)):
            return float("inf")
        return score(E_full[:, [gi[x] for x in lv]], lv, budget)[0]

    best = sc(levels)
    print(f"  init {levels} err={best:.6f}", flush=True)
    for sweep in range(cli.sweeps):
        moved = False
        for pos in range(len(levels)):
            cur = levels[pos]
            # candidates: every measured value not already in the set
            for cand in grid:
                if cand in levels:
                    continue
                trial = sorted(levels[:pos] + [cand] + levels[pos + 1:], reverse=True)
                if len(set(trial)) != len(trial):
                    continue
                s = sc(trial)
                if s < best - 1e-12:
                    best, levels, moved = s, trial, True
            if levels[pos if pos < len(levels) else -1] != cur:
                pass
        print(f"  sweep {sweep}: {levels} err={best:.6f}", flush=True)
        if not moved:
            break

    a = score(E_full[:, [gi[x] for x in levels]], levels, budget)[1]
    fr = (np.bincount(a, minlength=len(levels)) / len(a)).tolist()
    out = {"budget": budget, "levels": levels,
           "level_fractions": [round(f, 4) for f in fr],
           "achieved_avg_cycles": round(float(np.dot(fr, levels)), 2),
           "mean_assigned_error": round(best, 6),
           "init": [int(x) for x in cli.init.split(",")]}
    json.dump(out, open(cli.out, "w"), indent=2)
    print(f"wrote {cli.out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure")
    m.add_argument("--eval_config", default="configs/evaluation/bridge/frame_ada_sc_full.yaml")
    m.add_argument("--keys_file", default="results/diverse_300.json")
    m.add_argument("--num_samples", type=int, default=2)
    m.add_argument("--inference_steps", type=int, default=10)
    m.add_argument("--max_rows", type=int, default=4096)
    m.add_argument("--grid", default=None, help="comma-separated halved cycle counts")
    m.add_argument("--out", required=True)
    m.set_defaults(func=measure)

    s = sub.add_parser("search")
    s.add_argument("--grid_file", "--grid", dest="grid_file", required=True)
    s.add_argument("--budget", type=float, required=True)
    s.add_argument("--k", default="3,5,7")
    s.add_argument("--max_search_rows", type=int, default=8192,
                   help="rows used to score candidate subsets (0 = all)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=search, compare=None)

    r = sub.add_parser("refine")
    r.add_argument("--grid_file", "--grid", dest="grid_file", required=True)
    r.add_argument("--budget", type=float, required=True)
    r.add_argument("--init", required=True, help="comma-separated starting levels")
    r.add_argument("--sweeps", type=int, default=6)
    r.add_argument("--max_search_rows", type=int, default=8192)
    r.add_argument("--out", required=True)
    r.set_defaults(func=refine)

    cli = p.parse_args()
    cli.func(cli)


if __name__ == "__main__":
    main()
