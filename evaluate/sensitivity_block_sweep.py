"""Block-level SC sensitivity sweep using per-clip GT latents (local data only).

Unlike sensitivity_sweep.py this does NOT need the per-episode latent dir
(dataset get_dataset); it iterates the same per-clip GT latents as
eval_local_n_samples.py, so it runs off the partially-extracted bridge data.

For each block i in 0..depth-1, skip ALL SC ops in block i (leave-one-out,
i.e. "escape" the whole block back to FP) and record mean latent L2 vs GT.
A reference run with no skips (SC uniform) is recorded as "none".

Usage:
    PYTHONPATH=. BRIDGE_ROOT=... CUDA_VISIBLE_DEVICES=N \
      python3 evaluate/sensitivity_block_sweep.py \
        --config configs/evaluation/bridge/frame_ada_sc_full.yaml \
        --num_samples 3 --inference_steps 50 \
        --out_json results/sensitivity_block_sc_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL

from models.sc_integration import reconfigure, set_skip_blocks, clear_skip_blocks
from evaluate.eval_local_n_samples import (
    BRIDGE_ROOT, GT_LATENT_DIR, ANNOT_DIR, C_ACT_SCALER, SEQUENCE_LENGTH,
    build_args, load_model, make_pipe, compute_actions_for_slice, parse_key,
)

ALL_OPS = ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--depth", type=int, default=28)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--scheduler", choices=["PNDM", "DPM"], default="PNDM")
    p.add_argument("--out_json", required=True)
    p.add_argument("--block_start", type=int, default=0)
    p.add_argument("--block_end", type=int, default=None)
    p.add_argument("--skip_reference", action="store_true",
                   help="skip the no-skip SC-uniform reference run (when chunking)")
    cli = p.parse_args()

    args = build_args(cli.config, cli.inference_steps)
    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)
    reconfigure(args.attention_mode)
    clear_skip_blocks()
    pipe = make_pipe(args, vae, model, cli.scheduler)

    # Cache samples (same ordering as eval_local_n_samples).
    files = sorted(f for f in os.listdir(GT_LATENT_DIR) if f.endswith(".pt"))[: cli.num_samples]
    samples = []
    for fn in files:
        eid, cam, start = parse_key(fn)
        key = f"{eid}_{cam}_{start}"
        ann_path = os.path.join(ANNOT_DIR, f"{eid}.json")
        if not os.path.exists(ann_path):
            continue
        gt_lat = torch.load(os.path.join(GT_LATENT_DIR, fn), weights_only=False, map_location=device)
        with open(ann_path) as f:
            ann = json.load(f)
        if start + SEQUENCE_LENGTH > len(ann["state"]):
            continue
        arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
        grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
        actions = torch.from_numpy(
            compute_actions_for_slice(arm, grip) * C_ACT_SCALER).float().unsqueeze(0)
        samples.append({"key": key, "mask_x": gt_lat[0:1].unsqueeze(0),
                        "actions": actions, "gt_lat": gt_lat})
    print(f"cached {len(samples)} samples; variant={args.attention_mode} "
          f"steps={cli.inference_steps} sched={cli.scheduler}", flush=True)

    def eval_setting(label):
        l2s = {}
        for s in samples:
            with torch.no_grad():
                _, lat = pipe(
                    s["actions"].to(device).float(),
                    mask_x=s["mask_x"].to(device).float(),
                    video_length=args.num_frames,
                    height=args.video_size[0], width=args.video_size[1],
                    num_inference_steps=args.infer_num_sampling_steps,
                    guidance_scale=args.guidance_scale,
                    device=device, return_dict=False, output_type="latent_only",
                )
            lat = lat.squeeze(0)
            l2s[s["key"]] = F.mse_loss(lat[1:], s["gt_lat"][1:]).item()
        mean = sum(l2s.values()) / len(l2s)
        print(f"[{time.strftime('%H:%M:%S')}] {label}: mean L2={mean:.4f} "
              f"{ {k: round(v,4) for k,v in l2s.items()} }", flush=True)
        return {"mean_l2": mean, "per_sample": l2s}

    out = {"config": cli.config, "attention_mode": args.attention_mode,
           "depth": cli.depth, "steps": cli.inference_steps,
           "samples": [s["key"] for s in samples], "results": {}}
    os.makedirs(os.path.dirname(cli.out_json) or ".", exist_ok=True)

    def save():
        with open(cli.out_json, "w") as f:
            json.dump(out, f, indent=2)

    if not cli.skip_reference:
        clear_skip_blocks()
        out["results"]["none"] = eval_setting("SC uniform (no skip)")
        save()

    end = cli.block_end if cli.block_end is not None else cli.depth
    for i in range(cli.block_start, end):
        for op in ALL_OPS:
            set_skip_blocks(op, [i])
        out["results"][str(i)] = eval_setting(f"skip block {i} (all ops)")
        save()
    clear_skip_blocks()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
