"""Per-(block, op) SC sensitivity sweep — leave-one-out at operator granularity.

For each (block b, op o) with o in the 6 SC ops, keep every other op on SC
(sc_int8_full) and revert ONLY (b, o) back to FP; measure latent L2. Lower L2
(bigger improvement over the no-skip SC-uniform baseline) = that single
operator is a bigger SC-noise source. Ranking all 168 (b,o) lets us fix the
"most-sensitive 10%" (~17 ops) as a permanent skip set.

Loads model/VAE/samples ONCE and loops all configs (cheap per-config: just
set_skip_blocks + a short eval). Shard across GPUs by block range.

Usage:
    PYTHONPATH=. BRIDGE_ROOT=... CUDA_VISIBLE_DEVICES=N \
      python3 evaluate/sensitivity_op_block_sweep.py \
        --config configs/evaluation/bridge/frame_ada_sc_full.yaml \
        --num_samples 3 --inference_steps 10 --scheduler DPM \
        --block_start 0 --block_end 4 \
        --out_json results/opblock/sens_0_3.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

from models.sc_integration import reconfigure, set_skip_blocks, clear_skip_blocks
from evaluate.eval_local_n_samples import (
    GT_LATENT_DIR, ANNOT_DIR, C_ACT_SCALER, SEQUENCE_LENGTH,
    build_args, load_model, make_pipe, compute_actions_for_slice, parse_key,
)

ALL_OPS = ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--depth", type=int, default=28)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--inference_steps", type=int, default=10)
    p.add_argument("--scheduler", choices=["PNDM", "DPM"], default="DPM")
    p.add_argument("--out_json", required=True)
    p.add_argument("--block_start", type=int, default=0)
    p.add_argument("--block_end", type=int, default=None)
    p.add_argument("--skip_reference", action="store_true")
    cli = p.parse_args()

    args = build_args(cli.config, cli.inference_steps)
    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)
    reconfigure(args.attention_mode)
    clear_skip_blocks()
    pipe = make_pipe(args, vae, model, cli.scheduler)

    files = sorted(f for f in os.listdir(GT_LATENT_DIR) if f.endswith(".pt"))[: cli.num_samples]
    samples = []
    for fn in files:
        eid, cam, start = parse_key(fn)
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
        samples.append({"key": f"{eid}_{cam}_{start}", "mask_x": gt_lat[0:1].unsqueeze(0),
                        "actions": actions, "gt_lat": gt_lat})
    print(f"cached {len(samples)} samples; variant={args.attention_mode} "
          f"blocks[{cli.block_start},{cli.block_end}) steps={cli.inference_steps}", flush=True)

    def eval_mean_l2():
        l2s = []
        for s in samples:
            with torch.no_grad():
                _, lat = pipe(
                    s["actions"].to(device).float(), mask_x=s["mask_x"].to(device).float(),
                    video_length=args.num_frames, height=args.video_size[0], width=args.video_size[1],
                    num_inference_steps=args.infer_num_sampling_steps, guidance_scale=args.guidance_scale,
                    device=device, return_dict=False, output_type="latent_only",
                )
            l2s.append(F.mse_loss(lat.squeeze(0)[1:], s["gt_lat"][1:]).item())
        return sum(l2s) / len(l2s)

    out = {"config": cli.config, "attention_mode": args.attention_mode,
           "depth": cli.depth, "steps": cli.inference_steps,
           "samples": [s["key"] for s in samples], "results": {}}
    os.makedirs(os.path.dirname(cli.out_json) or ".", exist_ok=True)

    def save():
        with open(cli.out_json, "w") as f:
            json.dump(out, f, indent=2)

    if not cli.skip_reference:
        clear_skip_blocks()
        ref = eval_mean_l2()
        out["reference_no_skip"] = ref
        print(f"[{time.strftime('%H:%M:%S')}] SC-uniform baseline L2={ref:.4f}", flush=True)
        save()

    end = cli.block_end if cli.block_end is not None else cli.depth
    for b in range(cli.block_start, end):
        for op in ALL_OPS:
            clear_skip_blocks()
            set_skip_blocks(op, [b])
            l2 = eval_mean_l2()
            out["results"][f"{b}:{op}"] = l2
            print(f"[{time.strftime('%H:%M:%S')}] skip ({b},{op}) L2={l2:.4f}", flush=True)
            save()
    clear_skip_blocks()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
