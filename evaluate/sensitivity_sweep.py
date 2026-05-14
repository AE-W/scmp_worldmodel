"""Per-block sensitivity sweep for one SC op (default: mlp_fc1).

Usage:
    PYTHONPATH=. python3 evaluate/sensitivity_sweep.py \
        --config configs/evaluation/bridge/frame_ada_sc_qk_av_proj_fc1.yaml \
        --op mlp_fc1 --depth 28 \
        --out_json results/sensitivity_fc1.json

Method (leave-one-in):
    - Model loaded once with attention_mode = mode in config (e.g.
      sc_int8_qk_av_proj_fc1 — only fc1 is the new op).
    - All other SC ops (qk/av/proj) are *disabled* via cfg overrides so the
      sweep isolates `op` alone.
    - For each block_idx i in 0..depth-1, set skip_blocks[op] = all-blocks-
      except-i, run short eval on the same val_dataset (max_eval_samples
      episodes), and record mean latent_L2 vs GT short-sample latents.

Output JSON has per-block mean L2 and a baseline run with op fully off (FP).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL

from dataset import get_dataset
from models import get_models
from models.sc_integration import (
    get_config, reconfigure, set_skip_blocks, clear_skip_blocks,
)
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from diffusers.schedulers import PNDMScheduler, DDPMScheduler
from util import update_paths


def build_args(config_path: str):
    data_config = OmegaConf.load("configs/base/data.yaml")
    diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
    config = OmegaConf.load(config_path)
    args = OmegaConf.merge(data_config, diffusion_config, config)
    update_paths(args)
    args.latent_size = [t // 8 for t in args.video_size]
    return args


def load_model(args, device):
    model = get_models(args)
    if args.evaluate_checkpoint:
        ckpt = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        if "ema" in ckpt:
            ckpt = ckpt["ema"]
        msd = model.state_dict()
        loaded = 0
        for k, v in ckpt.items():
            if k in msd:
                msd[k] = v
                loaded += 1
        model.load_state_dict(msd)
        print(f"loaded {loaded}/{len(ckpt)} keys from {args.evaluate_checkpoint}")
    model.to(device).eval()
    return model


def encode_video_to_latent(video, vae, device, batch_size=16):
    b, f, _, _, _ = video.shape
    video = rearrange(video, "b f c h w -> (b f) c h w").contiguous().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, video.size(0), batch_size):
            chunk = vae.encode(video[i:i+batch_size]).latent_dist.sample().mul_(vae.config.scaling_factor)
            out.append(chunk)
    enc = torch.cat(out, 0)
    return rearrange(enc, "(b f) c h w -> b f c h w", b=b, f=f)


def run_one_episode(args, mask_x, actions, vae, model, device):
    if args.sample_method == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(
            args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end,
            beta_schedule=args.beta_schedule, variance_type=args.variance_type,
        )
    elif args.sample_method == "DDPM":
        scheduler = DDPMScheduler.from_pretrained(
            args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end,
            beta_schedule=args.beta_schedule, variance_type=args.variance_type,
        )
    pipe = Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)
    with torch.no_grad():
        videos, latents = pipe(
            actions.to(device).to(torch.float32),
            mask_x=mask_x.to(device).to(torch.float32),
            video_length=args.num_frames,
            height=args.video_size[0], width=args.video_size[1],
            num_inference_steps=args.infer_num_sampling_steps,
            guidance_scale=args.guidance_scale,
            device=device, return_dict=False,
            output_type="latent_only",
        )
    return latents  # (1, 16, 4, 32, 40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--op", default="mlp_fc1",
                        choices=["qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2"])
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--gt_lat_dir", default=None)
    parser.add_argument("--block_start", type=int, default=0,
                        help="First block index (inclusive) to sweep this run.")
    parser.add_argument("--block_end", type=int, default=None,
                        help="Last block index (exclusive). Default: depth.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit eval to first N samples from val_dataloader.")
    parser.add_argument("--skip_fp_baseline", action="store_true",
                        help="Skip the FP-only baseline (use when chunking).")
    cli = parser.parse_args()

    args = build_args(cli.config)
    if cli.gt_lat_dir is None:
        cli.gt_lat_dir = args.true_sample_latent_videos_dir

    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)

    model = load_model(args, device)

    _, val_dataset = get_dataset(args)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Cache batches and GT latents in memory.
    samples = []
    for batch in val_dataloader:
        if args.pre_encode:
            x = batch["latent"].to(device)
        else:
            x = encode_video_to_latent(batch["video"], vae, device)
        mask_x = x[:, 0:1]
        actions = batch["action"]
        episode_id = batch["video_name"]["episode_id"][0]
        cam_id = batch["video_name"]["cam_id"][0]
        start_frame_id = batch["video_name"]["start_frame_id"][0]
        key = f"{episode_id}_{cam_id}_{start_frame_id}"
        gt_lat_path = os.path.join(cli.gt_lat_dir, f"{key}.pt")
        gt_lat = torch.load(gt_lat_path, map_location=device)
        samples.append({"key": key, "mask_x": mask_x, "actions": actions, "gt_lat": gt_lat})
    if cli.num_samples is not None:
        samples = samples[:cli.num_samples]
    print(f"cached {len(samples)} eval samples")

    cfg = get_config()
    op = cli.op

    def eval_one_setting(label):
        l2s, keys = [], []
        for s in samples:
            pred = run_one_episode(args, s["mask_x"], s["actions"], vae, model, device)
            pred = pred.squeeze(0)  # (16, 4, 32, 40)
            l2 = F.mse_loss(pred[1:], s["gt_lat"][1:]).item()
            l2s.append(l2); keys.append(s["key"])
        mean = sum(l2s) / len(l2s)
        print(f"  [{label}] mean L2 = {mean:.4f}    per-sample = {dict(zip(keys, [round(v,4) for v in l2s]))}")
        return {"mean_l2": mean, "per_sample": dict(zip(keys, l2s))}

    out = {
        "config": cli.config,
        "op": op,
        "attention_mode": args.attention_mode,
        "depth": cli.depth,
        "samples": [s["key"] for s in samples],
        "results": {},
    }

    # Force re-apply preset, then disable everything except `op`. We keep
    # only `op` enabled so each measurement isolates `op`.
    reconfigure(args.attention_mode)
    for f in ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2"):
        setattr(cfg, f"enable_{f}", (f == op))
    clear_skip_blocks()

    # Baseline: op fully off (pure FP) — establishes "no-SC" L2 floor.
    if not cli.skip_fp_baseline:
        setattr(cfg, f"enable_{op}", False)
        print(f"[{time.strftime('%H:%M:%S')}] FP-only baseline (op disabled):")
        out["results"]["fp_only"] = eval_one_setting("FP-only")

    # Re-enable op, then sweep block-by-block.
    setattr(cfg, f"enable_{op}", True)
    end = cli.block_end if cli.block_end is not None else cli.depth
    for i in range(cli.block_start, end):
        skip = [b for b in range(cli.depth) if b != i]
        set_skip_blocks(op, skip)
        print(f"[{time.strftime('%H:%M:%S')}] block {i:2d} only (others FP):")
        out["results"][f"block_{i}"] = eval_one_setting(f"block_{i}")
        # Save partial after each block in case of interruption.
        with open(cli.out_json, "w") as fp:
            json.dump(out, fp, indent=2)

    print(f"wrote {cli.out_json}")


if __name__ == "__main__":
    main()
