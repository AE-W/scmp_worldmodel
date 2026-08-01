"""Per-row downstream sensitivity for MP allocation (the un-simplified loss).

First-order, the end-to-end damage of SC noise injected at row i of a linear's
output is

    dQ  ~=  sum_i  w_i * sigma_i(L_i)^2,      w_i = || d(model out) / d(y_i) ||^2

sigma_i(L) we already measure exactly with the real kernels (the error grid).
What was missing is w_i: every objective tried so far hard-codes it (absolute
RMSE: w=1; relative L2: w ~ 1/||y_i||^2) instead of measuring it. This script
measures it the way scmp_diffusion's Method 3 does — Hutchinson probes
backpropagated from the model output, grad norms read off each target linear's
output — but per ROW rather than per (op, block) bucket, and it records the
runtime ranking metric (smoothed row amax) alongside so the search can learn
the metric -> weight mapping and, crucially, its SIGN.

    python evaluate/measure_row_sensitivity.py --out results/row_sensitivity.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from evaluate.calibrate_mp_fractions import (
    TARGET_SUFFIXES, build_args, load_model,
    GT_LATENT_DIR, ANNOT_DIR, C_ACT_SCALER, SEQUENCE_LENGTH,
    compute_actions_for_slice,
)
from models.sc_integration import reconfigure, get_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_config", default="configs/evaluation/bridge/frame_ada_sc_full.yaml")
    p.add_argument("--keys_file", default="results/diverse_300.json")
    p.add_argument("--num_samples", type=int, default=2)
    p.add_argument("--timestep_fracs", default="0.2,0.5,0.8",
                   help="positions in the 50-step schedule to probe")
    p.add_argument("--probes", type=int, default=2, help="Hutchinson probes per state")
    p.add_argument("--pixel_frames", type=int, default=2,
                   help="frames decoded per pixel-space probe (0 disables)")
    p.add_argument("--rows_per_call", type=int, default=512)
    p.add_argument("--smooth_scales", default="results/smoothquant_scales.pt")
    p.add_argument("--out", required=True)
    cli = p.parse_args()

    args = build_args(cli.eval_config, 50)
    device = torch.device("cuda:0")
    model = load_model(args, device)
    # Pixel-space probes need the decoder: PSNR lives after the VAE, whose
    # Jacobian reweights token rows. latent-L2 probes do not (x0_hat is affine
    # in eps, so eps-probes already rank rows correctly for it).
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    reconfigure(args.attention_mode)
    cfg = get_config()
    for f in ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2"):
        setattr(cfg, f"enable_{f}", False)          # FP teacher; plain autograd

    smooth = torch.load(cli.smooth_scales, map_location=device,
                        weights_only=False)["scales"]

    from diffusers.schedulers import PNDMScheduler
    sched = PNDMScheduler.from_pretrained(
        args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end,
        beta_schedule=args.beta_schedule, variance_type=args.variance_type)
    sched.set_timesteps(50, device=device)
    fracs = [float(x) for x in cli.timestep_fracs.split(",")]
    probe_ts = [sched.timesteps[int(f * (len(sched.timesteps) - 1))] for f in fracs]

    # ---- hooks: keep each target linear's output grad + the runtime metric of
    #      the SAME sampled rows (row i of input maps to row i of output) ----
    captured: dict[str, dict] = {}
    hooks = []
    gen = torch.Generator(device="cpu").manual_seed(0)

    def mk(name, mod):
        s = smooth.get(name)
        s_dev = s.to(device).float() if s is not None else None

        def hook(_m, inp, out):
            if not out.requires_grad:
                return
            out.retain_grad()
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            n = x.shape[0]
            k = min(cli.rows_per_call, n)
            idx = torch.randperm(n, generator=gen)[:k].to(device)
            xs = x[idx] / s_dev if s_dev is not None else x[idx]
            amax = xs.abs().amax(-1)
            l2 = xs.norm(dim=-1)
            # candidate dispatch metrics, mirroring scmp_llm's "auto" mode:
            # the calibrator picks the best of amax / l2 / crest per module by
            # rank correlation with the true weight, instead of assuming amax.
            captured[name] = {"out": out, "idx": idx,
                              "metric": amax.cpu(), "l2": l2.cpu(),
                              "crest": (amax / (l2 + 1e-12)).cpu()}
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES):
            hooks.append(mod.register_forward_hook(mk(name, mod)))

    recs: dict[str, list] = {}
    keys = json.load(open(cli.keys_file))[: cli.num_samples]
    for key in keys:
        parts = key.split("_")
        eid, start = "_".join(parts[:-2]), int(parts[-1])
        ann = json.load(open(os.path.join(ANNOT_DIR, f"{eid}.json")))
        if start + SEQUENCE_LENGTH > len(ann["state"]):
            continue
        x0 = torch.load(os.path.join(GT_LATENT_DIR, f"{key}.pt"),
                        weights_only=False, map_location=device).float().unsqueeze(0)
        arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
        grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
        act = torch.from_numpy(
            compute_actions_for_slice(arm, grip) * C_ACT_SCALER
        ).float().unsqueeze(0).to(device)

        for t in probe_ts:
            noise = torch.randn_like(x0)
            x_t = sched.add_noise(x0, noise, t.reshape(1))
            x_t[:, :1] = x0[:, :1]                      # conditioning frame stays clean
            abar = sched.alphas_cumprod.to(device)[t.long()].float()
            for _pr in range(cli.probes):
                for kind in (("eps",) if cli.pixel_frames == 0 else ("eps", "pix")):
                    captured.clear()
                    model.zero_grad(set_to_none=True)
                    out = model(x_t.clone(), actions=act,
                                t=t.reshape(1).to(device),
                                mask_frame_num=1, use_fp16=False)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if kind == "eps":
                        # Gauss-Newton weight for latent-space MSE: x0_hat is
                        # affine in eps, so probing eps ranks rows identically.
                        v = torch.randn_like(out)
                        (out * v).sum().backward()
                    else:
                        # PSNR weight: push the probe through the VAE decoder
                        # so its Jacobian reweights the rows.
                        x0_hat = (x_t - torch.sqrt(1 - abar) * out) / torch.sqrt(abar)
                        fr = torch.randperm(x0_hat.shape[1], generator=gen)[: cli.pixel_frames]
                        pix = vae.decode(
                            x0_hat[0, fr].float() / vae.config.scaling_factor).sample
                        u = torch.randn_like(pix)
                        (pix * u).sum().backward()
                    for name, c in captured.items():
                        g = c["out"].grad
                        if g is None:
                            continue
                        g2 = g.detach().reshape(-1, g.shape[-1]).float()[c["idx"]]
                        g2 = g2.pow(2).sum(-1).cpu().numpy()
                        recs.setdefault((name, kind), []).append(
                            np.stack([c["metric"].numpy(), c["l2"].numpy(),
                                      c["crest"].numpy(), g2], 1))
            print(f"  {key} t={int(t)}: {len(recs)} modules recorded", flush=True)

    for h in hooks:
        h.remove()

    names = sorted(recs)
    mats = [np.concatenate(recs[n], 0) for n in names]
    lens = np.array([m.shape[0] for m in mats])
    np.savez_compressed(cli.out, data=np.concatenate(mats, 0), lengths=lens,
                        modules=np.array([f"{n}|{k}" for n, k in names]))
    print(f"wrote {cli.out}: {int(lens.sum())} (metric, grad^2) pairs "
          f"over {len(names)} module-kind pairs", flush=True)

    # quick readout: does the runtime metric rank the true weight, and which way?
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        return float((ra * rb).sum() / d) if d > 0 else float("nan")

    for want in ("eps", "pix"):
        for ci, mname in ((0, "amax"), (1, "l2"), (2, "crest")):
            rhos = [spearman(m[:, ci], m[:, 3])
                    for (n, k), m in zip(names, mats) if k == want]
            if rhos:
                ab = [abs(r) for r in rhos]
                print(f"Spearman({mname:5}, w_{want}): mean {np.mean(rhos):+.3f}  "
                      f"mean|rho| {np.mean(ab):.3f}  P90|rho| {np.percentile(ab,90):.3f}")
    print("  >0: long streams belong to HIGH-metric rows (current direction)")
    print("  <0: long streams belong to LOW-metric rows (inverted direction)")


if __name__ == "__main__":
    main()
