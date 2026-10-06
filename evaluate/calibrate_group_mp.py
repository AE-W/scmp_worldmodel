"""Search deployable group policies with the actual chunked kernel and SQ.

This is a bounded PTQ search, not an oracle assignment of error-ranked groups.
Each module selects one/two lengths from a grid and a runtime absmax quantile
policy. Uniform is a candidate. A disjoint activation holdout decides whether
to keep the complete policy or fall back to uniform. Image quality is tested
after selection; reconstruction-error fallback does not guarantee test PSNR.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from scmp_kernels import sc_matmul
from scmp_kernels.mp import classify_rows_by_metric
from evaluate.group_mp_policy import group_counts, mean_cycles, policies

from models.sc_integration import get_config, reconfigure
from models.sc_integration.sc_controller import get_current_step
from evaluate.eval_local_n_samples import (
    ANNOT_DIR, C_ACT_SCALER, GT_LATENT_DIR, SEQUENCE_LENGTH, build_args,
    compute_actions_for_slice, load_model, make_pipe, parse_skip,
)

SUFFIX = {"attn.qkv": "qkv", "attn.proj": "proj",
          "mlp.fc1": "mlp_fc1", "mlp.fc2": "mlp_fc2"}


def rung(metric, levels, policy):
    if policy["invert"]:
        metric = -metric
    return classify_rows_by_metric(
        metric.flatten(), levels, policy["level_fractions"]
    ).row_levels.reshape(metric.shape).to(torch.int32)


def sampled_rung(captures, policy, device):
    """Retain ranks from each full original call, not from pooled samples."""
    tables = []
    for _x, ranks, n_groups in captures:
        counts = group_counts(n_groups, policy["level_fractions"])
        boundaries = torch.tensor(np.cumsum(counts)[:-1], device=device)
        rank = ranks[int(policy["invert"])].to(device)
        tables.append(torch.bucketize(rank.contiguous(), boundaries, right=True).to(torch.int32))
    return torch.cat(tables)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/evaluation/bridge/frame_ada_sc_full.yaml")
    p.add_argument("--calibration_keys", required=True)
    p.add_argument("--holdout_keys", required=True)
    p.add_argument("--test_keys", default="results/diverse_10.json")
    p.add_argument("--scales", default="results/smoothquant_scales.pt")
    p.add_argument("--skip", required=True)
    p.add_argument("--levels", default="128,112,96,64,32")
    p.add_argument("--budget", type=int, default=96)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--steps", default="0,25,49")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--out", required=True)
    cli = p.parse_args()
    levels = sorted(set(map(int, cli.levels.split(","))), reverse=True)
    if cli.budget not in levels or min(levels) < 1 or max(levels) > 128:
        raise ValueError("Include the uniform budget in a grid inside [1,128]")
    splits = [json.load(open(path)) for path in
              (cli.calibration_keys, cli.holdout_keys, cli.test_keys)]
    episodes = [{"_".join(k.split("_")[:-2]) for k in keys} for keys in splits]
    if any(episodes[i] & episodes[j] for i in range(3) for j in range(i)):
        raise ValueError("Calibration, holdout and test episodes must be disjoint")
    device = torch.device("cuda:0")
    args = build_args(cli.config, 50)
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    model = load_model(args, device)
    named = dict(model.named_modules())
    scales = torch.load(cli.scales, map_location=device, weights_only=False)["scales"]
    reconfigure(args.attention_mode)
    cfg = get_config()
    skip = parse_skip(cli.skip)
    active = {}
    for name, mod in named.items():
        m = re.search(r"blocks\.(\d+)\.", name)
        if not isinstance(mod, torch.nn.Linear) or m is None:
            continue
        for suffix, op in SUFFIX.items():
            if name.endswith(suffix) and getattr(cfg, "enable_" + op):
                if int(m.group(1)) not in skip.get(op, []):
                    if mod.in_features % 128 or name not in scales:
                        raise ValueError(f"Missing SQ or incomplete input group: {name}")
                    active[name] = mod
    for op in ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2"):
        setattr(cfg, "enable_" + op, False)
    steps = set(map(int, cli.steps.split(",")))
    captures = [{name: [] for name in active} for _ in range(2)]
    full_shapes = {name: set() for name in active}
    phase = 0
    hooks = []
    sampler = torch.Generator(device=device).manual_seed(cli.seed + 1)
    for name, mod in active.items():
        def hook(_m, inp, _out, name=name):
            step, _ = get_current_step()
            if step not in steps:
                return
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            full_shapes[name].add(int(x.shape[0]))
            indices = torch.randperm(x.shape[0], device=device, generator=sampler)[:cli.rows]
            metric = x.abs().view(x.shape[0], -1, 128).amax(-1)
            ranks = []
            for direction in (metric, -metric):
                order = direction.flatten().argsort(descending=True)
                position = torch.empty_like(order)
                position[order] = torch.arange(order.numel(), device=device)
                ranks.append(position.view(metric.shape)[indices].cpu())
            captures[phase][name].append((x[indices].cpu(), ranks, metric.numel()))
        hooks.append(mod.register_forward_hook(hook))
    pipe = make_pipe(args, vae, model, "PNDM")
    for phase in range(2):
        for key in splits[phase]:
            eid = "_".join(key.split("_")[:-2])
            start = int(key.split("_")[-1])
            ann = json.load(open(os.path.join(ANNOT_DIR, eid + ".json")))
            gt = torch.load(os.path.join(GT_LATENT_DIR, key + ".pt"), map_location=device, weights_only=False)
            arm = np.asarray(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
            grip = np.asarray(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
            if len(arm) != SEQUENCE_LENGTH:
                raise ValueError(f"Incomplete calibration trajectory: {key}")
            actions = torch.from_numpy(compute_actions_for_slice(arm, grip) * C_ACT_SCALER).float().unsqueeze(0)
            with torch.no_grad():
                pipe(actions.to(device), mask_x=gt[:1].unsqueeze(0).float(),
                     video_length=args.num_frames, height=args.video_size[0], width=args.video_size[1],
                     num_inference_steps=50, guidance_scale=args.guidance_scale, device=device,
                     return_dict=False, output_type="latent_only",
                     generator=torch.Generator(device=device).manual_seed(cli.seed))
            print(f"teacher phase={phase} key={key}", flush=True)
    for handle in hooks:
        handle.remove()
    del pipe, vae
    if not active or any(not c[name] for c in captures for name in active):
        raise RuntimeError("No teacher captures for one or more active modules")
    candidates = policies(levels)
    uniform_index = next(i for i, d in enumerate(candidates)
                         if d["level_fractions"][levels.index(cli.budget)] == 1)
    chosen, measured = {}, {}
    train_total = val_total = train_uniform = val_uniform = total_weight = 0.0
    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, mod in active.items():
        xs = [torch.cat([item[0] for item in c[name]]).to(device) for c in captures]
        weight = mod.weight.detach().float()
        refs = [x @ weight.T for x in xs]
        # Evaluate the full matmul: preserve correlations and cancellation
        # between chunks. No sum of isolated chunk-error surrogates.
        errors, costs = [], []
        for candidate in candidates:
            phase_errors = []
            for phase_index, (x, ref) in enumerate(zip(xs, refs)):
                with torch.no_grad():
                    y = sc_matmul(x, weight, granularity="per_row", mode="bipolar",
                                  sc_prec=8, stoc_len=max(levels), chunk_d=128,
                                  halve_bipolar_stoc_len=True, smooth_scales=scales[name].float(),
                                  rung_table=sampled_rung(captures[phase_index][name], candidate, device), level_lens=levels)
                phase_errors.append(float((y - ref).square().mean() / ref.square().mean().clamp_min(1e-12)))
            # Check realized rounded dispatch, both sampled and deployment sizes.
            sizes = full_shapes[name]
            cost = max(mean_cycles(n * (mod.in_features // 128), levels,
                                   candidate["level_fractions"]) for n in sizes)
            errors.append(phase_errors)
            costs.append(cost)
        feasible = [i for i, cost in enumerate(costs) if cost <= cli.budget + 1e-6]
        best = min(feasible, key=lambda i: (errors[i][0], costs[i]))
        if errors[best][0] >= errors[uniform_index][0]:
            best = uniform_index
        chosen[name] = dict(candidates[best], avg_cycles=costs[best])
        measured[name] = dict(errors=errors, avg_cycles=costs, selected_index=best,
                              full_rows=sorted(full_shapes[name]))
        w = sum(full_shapes[name]) * mod.out_features
        total_weight += w
        train_total += w * errors[best][0]
        val_total += w * errors[best][1]
        train_uniform += w * errors[uniform_index][0]
        val_uniform += w * errors[uniform_index][1]
        print(f"profile {name}: train={errors[best][0]:.6g} holdout={errors[best][1]:.6g} cycles={costs[best]:.4f}", flush=True)
        # Persist progress before the next expensive module.
        Path(str(out) + ".profiles.json").write_text(json.dumps(measured, indent=2))
    fallback = val_total >= val_uniform
    if fallback:
        chosen = {name: dict(candidates[uniform_index], avg_cycles=cli.budget) for name in active}
    payload = dict(stoc_len_levels=levels, level_fractions=candidates[uniform_index]["level_fractions"],
                   per_module_fractions=chosen, chunk_d=128, sc_prec=8,
                   target_avg_cycles=cli.budget, uniform_fallback=fallback,
                   objective="output relative MSE, weighted by deployment output elements",
                   search="one/two rungs, quantiles 0.25/0.5/0.75, both rank directions; per-module budget",
                   calibration_keys=splits[0], holdout_keys=splits[1], test_keys=splits[2],
                   scheduler="PNDM", inference_steps=50, seed=cli.seed, captured_steps=sorted(steps),
                   skip=cli.skip, scales_sha256=digest(cli.scales),
                   code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   kernel_commit=subprocess.check_output(["git", "-C", "kernels", "rev-parse", "HEAD"], text=True).strip(),
                   proposal_train_error=train_total / total_weight,
                   proposal_holdout_error=val_total / total_weight,
                   selected_holdout_error=(val_uniform if fallback else val_total) / total_weight,
                   uniform_train_error=train_uniform / total_weight,
                   uniform_holdout_error=val_uniform / total_weight)
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved {out}; uniform_fallback={fallback}", flush=True)


if __name__ == "__main__":
    main()
