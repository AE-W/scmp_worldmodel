"""Full scmp_llm-style MP calibration for the world model.

Produces the three ingredients the LLM ladder actually uses (and which the
naive quantile port lacked):

  1. salient-channel protection: per (op, block), the top channels by
         E[x_j^2] * sum_o W[o,j]^2 * E[(dL/dy_o)^2]
     (activation energy x gradient-weighted weight-column energy — the
     gradient-weighted selector from scmp_llm calibrate_mp_thresholds), run at
     a fixed high stream length, with row dispatch on the residual columns;
  2. per-module THRESHOLDS on the per-call min-max-normalised metric
     (distribution-adaptive fractions, not fixed quantiles);
  3. budget bookkeeping that charges the protected columns' extra cycles to
     the same global average so the uniform comparison stays fair.

Phase 1 (GPU): teacher forward + pixel-space Hutchinson probes accumulating
per-input-channel x^2 and per-output-channel grad^2 for every target linear.
Phase 2 (CPU): combine with the SmoothQuant-consistent error grid to solve the
row thresholds under the residual budget and emit the deployment JSON.
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import torch

from evaluate.calibrate_mp_fractions import (
    TARGET_SUFFIXES, build_args, load_model,
    GT_LATENT_DIR, ANNOT_DIR, C_ACT_SCALER, SEQUENCE_LENGTH,
    compute_actions_for_slice,
)
from models.sc_integration import reconfigure, get_config


def gpu_phase(cli, device):
    args = build_args(cli.eval_config, 50)
    model = load_model(args, device)
    reconfigure(args.attention_mode)
    cfg = get_config()
    for f in ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2"):
        setattr(cfg, f"enable_{f}", False)
    from diffusers.models import AutoencoderKL
    from diffusers.schedulers import PNDMScheduler
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    sched = PNDMScheduler.from_pretrained(
        args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end,
        beta_schedule=args.beta_schedule, variance_type=args.variance_type)
    sched.set_timesteps(50, device=device)
    probe_ts = [sched.timesteps[int(f * 49)] for f in (0.2, 0.5, 0.8)]

    x2, g2, W2 = {}, {}, {}
    hold = {}
    hooks = []

    def mk(name, mod):
        def hook(_m, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            x2[name] = x2.get(name, 0) + x.pow(2).sum(0).cpu()
            if out.requires_grad:
                out.retain_grad()
                hold[name] = out
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES):
            hooks.append(mod.register_forward_hook(mk(name, mod)))
            W2[name] = mod.weight.detach().float().pow(2).cpu()

    gen = torch.Generator(device="cpu").manual_seed(0)
    for key in json.load(open(cli.keys_file))[: cli.num_samples]:
        parts = key.split("_"); eid, start = "_".join(parts[:-2]), int(parts[-1])
        ann = json.load(open(os.path.join(ANNOT_DIR, f"{eid}.json")))
        if start + SEQUENCE_LENGTH > len(ann["state"]):
            continue
        x0 = torch.load(os.path.join(GT_LATENT_DIR, f"{key}.pt"),
                        weights_only=False, map_location=device).float().unsqueeze(0)
        arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
        grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
        act = torch.from_numpy(
            compute_actions_for_slice(arm, grip) * C_ACT_SCALER).float().unsqueeze(0).to(device)
        for t in probe_ts:
            noise = torch.randn_like(x0)
            x_t = sched.add_noise(x0, noise, t.reshape(1)); x_t[:, :1] = x0[:, :1]
            abar = sched.alphas_cumprod.to(device)[t.long()].float()
            for _ in range(cli.probes):
                hold.clear(); model.zero_grad(set_to_none=True)
                out = model(x_t.clone(), actions=act, t=t.reshape(1).to(device),
                            mask_frame_num=1, use_fp16=False)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                x0_hat = (x_t - torch.sqrt(1 - abar) * out) / torch.sqrt(abar)
                fr = torch.randperm(x0_hat.shape[1], generator=gen)[:2]
                pix = vae.decode(x0_hat[0, fr].float() / vae.config.scaling_factor).sample
                (pix * torch.randn_like(pix)).sum().backward()
                for name, o in hold.items():
                    if o.grad is not None:
                        g = o.grad.detach().reshape(-1, o.shape[-1]).float()
                        g2[name] = g2.get(name, 0) + g.pow(2).sum(0).cpu()
            print(f"  {key} t={int(t)} ok", flush=True)
    for h in hooks:
        h.remove()
    np.savez_compressed(cli.stats_out,
        names=np.array(sorted(x2)),
        **{f"x2::{n}": x2[n].numpy() for n in x2},
        **{f"g2::{n}": g2[n].numpy() for n in g2},
        **{f"w2sum::{n}": W2[n].numpy().sum(0) for n in W2})   # fallback (unweighted)
    # gradient-weighted column energy: sum_o W[o,j]^2 g2[o]
    np.savez_compressed(cli.stats_out.replace(".npz", "_sal.npz"),
        names=np.array(sorted(x2)),
        **{f"sal::{n}": (x2[n] * (W2[n] * g2[n][:, None]).sum(0)).numpy()
           for n in x2 if n in g2})
    print("stats written", flush=True)


def assemble(cli):
    g = np.load(cli.grid, allow_pickle=True)
    E, grid_asc = g["errors"].astype(np.float64), [int(x) for x in g["grid"]]
    mod, met, ref = g["module_idx"], g["row_metric"], g["ref_norm"]
    gmods = [str(m) for m in g["modules"]]
    od = np.argsort(grid_asc)[::-1]
    grid = [grid_asc[i] for i in od]
    sig = (E / np.maximum(ref[:, None], 1e-8))[:, od]
    cur = np.maximum(sig - sig[:, :1], 0.0) ** 2          # delta_sigma2

    sal = np.load(cli.grid.replace("mp_error_grid_sq2.npz", "")
                  + os.path.basename(cli.stats_out).replace(".npz", "_sal.npz"),
                  allow_pickle=True) if False else np.load(
                      cli.stats_out.replace(".npz", "_sal.npz"), allow_pickle=True)

    # protected channels + residual budget
    protected = {}
    p_frac_tot, n_mod = 0.0, 0
    for name in gmods:
        key = f"sal::{name}"
        if key not in sal:
            continue
        v = sal[key]
        k = max(1, int(round(cli.protect_frac * v.shape[0])))
        protected[name] = np.argsort(-v)[:k].tolist()
        p_frac_tot += k / v.shape[0]; n_mod += 1
    p = p_frac_tot / max(n_mod, 1)
    resid_budget = (cli.budget - p * cli.protect_sl) / max(1 - p, 1e-6)
    print(f"protected frac/module ~{p:.4f}, protect_sl={cli.protect_sl}, "
          f"residual budget {resid_budget:.2f} (global {cli.budget})")

    import sys
    sys.path.insert(0, ".")
    from evaluate.search_mp_levels import _assign
    a = _assign(cur, np.array(grid, float), resid_budget * cur.shape[0])

    pm = {}
    for mi, name in enumerate(gmods):
        rows = np.where(mod == mi)[0]
        if not len(rows):
            continue
        m = met[rows]
        mn = (m - m.min()) / max(m.max() - m.min(), 1e-8)
        lv_rows = a[rows]
        ths = []
        for li in range(len(grid) - 1):     # threshold between level li and li+1
            hi = mn[lv_rows <= li]
            lo = mn[lv_rows > li]
            if len(hi) == 0:
                ths.append(1.0)
            elif len(lo) == 0:
                ths.append(0.0)
            else:
                ths.append(float((hi.min() + lo.max()) / 2))
        ths = sorted(ths, reverse=True)
        pm[name] = {"thresholds": ths,
                    "protected": protected.get(name, []),
                    "protect_sl": cli.protect_sl,
                    "avg_cycles": float(np.mean([grid[i] for i in lv_rows])),
                    "invert": False}
    cnt = np.bincount(a, minlength=len(grid))
    fr = (cnt / cnt.sum()).round(4)
    j = int(np.argmax(fr)); fr[j] = round(fr[j] + (1.0 - fr.sum()), 10)
    out = {"config_name": cli.name, "sc_prec": 8, "stoc_len_levels": grid,
           "level_fractions": fr.tolist(), "target_avg_cycles": cli.budget,
           "achieved_avg_cycles": round(float(np.dot(fr, grid)) * (1 - p)
                                        + p * cli.protect_sl, 2),
           "source": "full scmp_llm mechanism: thresholds + protected channels "
                     "+ delta_sigma2 relative currency",
           "per_module_fractions": pm}
    json.dump(out, open(cli.out, "w"), indent=2)
    print(f"wrote {cli.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gpu", "assemble", "both"], default="both")
    ap.add_argument("--eval_config", default="configs/evaluation/bridge/frame_ada_sc_full.yaml")
    ap.add_argument("--keys_file", default="results/diverse_300.json")
    ap.add_argument("--num_samples", type=int, default=2)
    ap.add_argument("--probes", type=int, default=2)
    ap.add_argument("--grid", default="results/mp_error_grid_sq2.npz")
    ap.add_argument("--stats_out", default="results/channel_stats.npz")
    ap.add_argument("--budget", type=float, default=96.0)
    ap.add_argument("--protect_frac", type=float, default=0.01)
    ap.add_argument("--protect_sl", type=int, default=128)
    ap.add_argument("--name", default="sc_avg192_full")
    ap.add_argument("--out", default="results/mp_fractions_sc_avg192_full.json")
    cli = ap.parse_args()
    if cli.phase in ("gpu", "both"):
        gpu_phase(cli, torch.device("cuda:0"))
    if cli.phase in ("assemble", "both"):
        assemble(cli)


if __name__ == "__main__":
    main()
