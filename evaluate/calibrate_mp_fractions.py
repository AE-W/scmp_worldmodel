"""Calibrate MP level fractions for a HPCA SC config on this world model.

Port of the allocation core in scmp_llm's calibrate_mp_thresholds.py, reduced
to what we need: given mp_levels and a cycle budget (budget_ratio * ref),
decide what fraction of rows runs at each stream length.

Method
  1. Teacher pass: run the model with SC disabled, hook the SC'd linears, and
     capture real activation rows (per operator).
  2. Error curves: for each captured row, measure |SC(row) - FP(row)| at every
     candidate stoc_len (the real kernel, not a surrogate).
  3. Allocation: Lagrangian relaxation — each row picks argmin(error + λ·cost),
     binary-search λ so mean cost hits the budget. Fractions = level counts.

Rows are ranked at runtime by |x|.amax(-1) (same metric as the dispatcher), so
the calibrated fractions translate directly into MPConfig.level_fractions.

Usage:
  PYTHONPATH=. BRIDGE_ROOT=... CUDA_VISIBLE_DEVICES=N python3 \
    evaluate/calibrate_mp_fractions.py --config_name sc_avg192 \
    --out results/mp_fractions_sc_avg192.json
"""
import argparse, json, os
import numpy as np
import torch
from diffusers.models import AutoencoderKL

from scmp_kernels import sc_matmul
from scmp_kernels.sc.config_helpers import make_sobol_simple_config
from models.sc_integration import reconfigure, get_config
from evaluate.eval_local_n_samples import (
    GT_LATENT_DIR, ANNOT_DIR, C_ACT_SCALER, SEQUENCE_LENGTH,
    build_args, load_model, make_pipe, compute_actions_for_slice,
)

# HPCA config table (scmp_llm hpca script; see configs/sc_spec.yaml).
# KEY: the level values ARE halved cycle counts (hpca line 13: "sc_prec=8,
# halve ON, tag value IS the cycle count"). So kernel sc_prec stays 8 for every
# config — "int7/int6" is the ISO-EQUIVALENT int width for that cycle budget
# (line 320: halved 128/96/64/48/32 -> int 8/8/7/7/6), NOT a lowered grid.
HPCA = {
    "sc_int8":   dict(sc_prec=8, levels=[128, 96, 64], ratio=1.00, ref=128),
    "sc_avg192": dict(sc_prec=8, levels=[128, 96, 64], ratio=0.75, ref=128),
    "sc_int7":   dict(sc_prec=8, levels=[128, 64, 32], ratio=0.50, ref=128),
    "sc_avg96":  dict(sc_prec=8, levels=[64, 48, 32],  ratio=0.75, ref=64),
    "sc_int6":   dict(sc_prec=8, levels=[64, 48, 32],  ratio=0.50, ref=64),
    # 6.32-bit MP tier (nominal L = 80 = 2 x 40 halved cycles). Levels are
    # shifted down so the 40-cycle budget lands INSIDE [48, 32] — with the
    # sc_avg96 level set [64,48,32] a 40 budget would still be interior, but
    # anchoring at 48 keeps the same 3-level shape as the other MP tiers and
    # avoids reusing a coarser grid than the budget needs.
    "sc_avg80":  dict(sc_prec=8, levels=[48, 40, 32],  ratio=0.625, ref=64),

    # ---- level-set ablation ------------------------------------------------
    # The 3-level {ref, 0.75ref, 0.5ref} shape above spans only 2x, which caps
    # how much budget MP can move: at target 96 at most 50% of rows can reach
    # 128. The scmp_llm ladder uses 5-8 levels spanning ~4x for the same
    # targets (its target-96 set, [128,96,64,48,32], is identical across 4B /
    # 8B / 14B / 30B). These entries vary LEVEL COUNT and SPAN independently so
    # the two can be told apart:
    #   _w5 = scmp_llm's set verbatim   (5 levels, 4x span)
    #   _n5 = 5 levels at the OLD 2x span (isolates level count)
    #   _w7 = 7 levels, 8x span          (pushes span further)
    "sc_avg192_w5": dict(sc_prec=8, levels=[128, 96, 64, 48, 32],         ratio=0.75, ref=128),
    "sc_avg192_n5": dict(sc_prec=8, levels=[128, 112, 96, 80, 64],        ratio=0.75, ref=128),
    "sc_avg192_w7": dict(sc_prec=8, levels=[128, 96, 64, 48, 32, 24, 16], ratio=0.75, ref=128),
    "sc_avg96_w5":  dict(sc_prec=8, levels=[64, 48, 32, 24, 16],          ratio=0.75, ref=64),
    "sc_avg96_n5":  dict(sc_prec=8, levels=[64, 56, 48, 40, 32],          ratio=0.75, ref=64),
    "sc_avg96_w7":  dict(sc_prec=8, levels=[128, 96, 64, 48, 32, 24, 16], ratio=0.375, ref=128),
}
TARGET_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def cost_assignments(errors, costs, budget_total):
    """errors [n_units, n_levels], costs [n_levels] -> level index per unit."""
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
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        best, c = solve(mid)
        if c > budget_total:
            lo = mid
        else:
            hi = mid
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_name", required=True, choices=list(HPCA))
    p.add_argument("--eval_config", default="configs/evaluation/bridge/frame_ada_sc_full.yaml")
    p.add_argument("--keys_file", default="results/diverse_300.json")
    p.add_argument("--num_samples", type=int, default=2)
    p.add_argument("--inference_steps", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=4096, help="rows sampled per operator")
    p.add_argument("--out", required=True)
    cli = p.parse_args()
    spec = HPCA[cli.config_name]
    levels = spec["levels"]; sc_prec = spec["sc_prec"]
    budget_per_row = spec["ratio"] * spec["ref"]

    args = build_args(cli.eval_config, cli.inference_steps)
    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)
    reconfigure(args.attention_mode)
    cfg = get_config()
    for f in ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2"):
        setattr(cfg, f"enable_{f}", False)      # teacher pass: pure FP

    # ---- capture activation rows from the SC'd linears ----
    caps = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES):
            def mk(n, m):
                def hook(_m, inp, _out):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                    keep = caps.setdefault(n, [])
                    if sum(t.shape[0] for t in keep) < 512:
                        idx = torch.randperm(x.shape[0], device=x.device)[:64]
                        keep.append(x[idx].cpu())
                return hook
            hooks.append(mod.register_forward_hook(mk(name, mod)))

    pipe = make_pipe(args, vae, model, "DPM")
    keys = json.load(open(cli.keys_file))[: cli.num_samples]
    for k in keys:
        parts = k.split("_"); eid = "_".join(parts[:-2]); start = int(parts[-1])
        ann = json.load(open(os.path.join(ANNOT_DIR, f"{eid}.json")))
        if start + SEQUENCE_LENGTH > len(ann["state"]):
            continue
        gt = torch.load(os.path.join(GT_LATENT_DIR, f"{k}.pt"), weights_only=False, map_location=device)
        arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
        grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
        act = torch.from_numpy(compute_actions_for_slice(arm, grip) * C_ACT_SCALER).float().unsqueeze(0)
        with torch.no_grad():
            pipe(act.to(device).float(), mask_x=gt[0:1].unsqueeze(0).to(device).float(),
                 video_length=args.num_frames, height=args.video_size[0], width=args.video_size[1],
                 num_inference_steps=cli.inference_steps, guidance_scale=args.guidance_scale,
                 device=device, return_dict=False, output_type="latent_only")
        print(f"  teacher pass {k}", flush=True)
    for h in hooks:
        h.remove()

    # ---- per-row error at each level (real SC kernel) ----
    named = dict(model.named_modules())
    all_err, all_metric = [], []
    for name, chunks in caps.items():
        X = torch.cat(chunks, 0)[: cli.max_rows].to(device)
        W = named[name].weight.detach().float()
        ref_out = X @ W.t()
        errs = []
        for L in levels:
            with torch.no_grad():
                y = sc_matmul(X, W, granularity="per_row", mode="bipolar",
                              sc_prec=sc_prec, stoc_len=L,
                              config=make_sobol_simple_config(X.shape[-1], X.shape[-1], sc_prec),
                              halve_bipolar_stoc_len=True)
            errs.append(((y - ref_out) ** 2).mean(dim=-1).sqrt().cpu().numpy())
        all_err.append(np.stack(errs, 1))            # [rows, n_levels]
        all_metric.append(X.abs().amax(-1).cpu().numpy())
        print(f"  error curve {name}: rows={X.shape[0]}", flush=True)

    E = np.concatenate(all_err, 0)
    costs = np.array(levels, dtype=np.float64)
    assign = cost_assignments(E, costs, budget_per_row * E.shape[0])
    counts = np.bincount(assign, minlength=len(levels))
    fractions = (counts / counts.sum()).tolist()
    achieved = float(np.dot(fractions, levels))

    # The Lagrangian above already lets budget FLOW ACROSS modules: a module
    # with steeper error curves takes more of its rows at the high levels.
    # Collapsing `assign` to one global fraction triple throws that away and
    # spends an identical average on every (operator, block) — which is why a
    # global-fraction MP schedule measured no better than uniform at matched
    # budget. Keep the per-module split the solver actually produced.
    per_module, off = {}, 0
    for (name, _chunks), err in zip(caps.items(), all_err):
        n_rows = err.shape[0]
        c = np.bincount(assign[off:off + n_rows], minlength=len(levels))
        f = (c / max(c.sum(), 1)).tolist()
        per_module[name] = {"level_fractions": [round(v, 4) for v in f],
                            "avg_cycles": round(float(np.dot(f, levels)), 2),
                            "n_rows": int(n_rows)}
        off += n_rows
    assert off == E.shape[0], f"row bookkeeping mismatch: {off} vs {E.shape[0]}"

    out = {"config_name": cli.config_name, "sc_prec": sc_prec, "stoc_len_levels": levels,
           "level_fractions": [round(f, 4) for f in fractions],
           "target_avg_cycles": budget_per_row, "achieved_avg_cycles": round(achieved, 2),
           "n_rows_calibrated": int(E.shape[0]),
           "per_module_fractions": per_module}
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    json.dump(out, open(cli.out, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
