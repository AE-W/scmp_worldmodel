"""Local-only N-sample short eval with a given SC variant + skip schedule.

Adapted from eval_full_short_hfupload.py but:
- no HF upload (everything stays local)
- configurable variant / skip / inference steps via CLI
- supports sharding across GPUs (--shard / --num_shards)

Iterates over per-clip GT latents in
  bridge/evaluation_latent_videos/test_sample_latent_videos/<ep>_<cam>_<start>.pt
so it doesn't depend on the (partially-extracted) per-episode latent dir.

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=N \\
      python3 evaluate/eval_local_n_samples.py \\
        --config configs/evaluation/bridge/frame_ada_sc_qk_av_proj_fc1.yaml \\
        --skip 'mlp_fc1=25,27,4,26,13,16' \\
        --tag fc1var_skip6_n20 \\
        --num_samples 20 \\
        --shard 0 --num_shards 4 \\
        --inference_steps 50
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio
from einops import rearrange
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler, DPMSolverMultistepScheduler

from dataset.dataset_util import euler2rotm, rotm2euler
from models import get_models
from models.sc_integration import reconfigure, set_skip_blocks, clear_skip_blocks
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from evaluate.compute_psnr_ssim import process_video_psnr_ssim
from util import update_paths


BRIDGE_ROOT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge"
GT_LATENT_DIR = f"{BRIDGE_ROOT}/evaluation_latent_videos/test_sample_latent_videos"
GT_VIDEO_DIR = f"{BRIDGE_ROOT}/evaluation_videos/test_sample_videos"
ANNOT_DIR = f"{BRIDGE_ROOT}/annotation/test"

C_ACT_SCALER = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0], dtype=float)
SEQUENCE_LENGTH = 16
ACTION_DIM = 7


def parse_skip(skip_str: str) -> dict:
    if not skip_str:
        return {}
    out = {}
    for clause in skip_str.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        op, blocks = clause.split("=")
        out[op.strip()] = [int(x) for x in blocks.split(",") if x.strip()]
    return out


def build_args(config_path: str, infer_steps: int):
    args = OmegaConf.merge(
        OmegaConf.load("configs/base/data.yaml"),
        OmegaConf.load("configs/base/diffusion.yaml"),
        OmegaConf.load(config_path),
    )
    update_paths(args)
    args.latent_size = [t // 8 for t in args.video_size]
    args.infer_num_sampling_steps = infer_steps
    return args


def load_model(args, device):
    model = get_models(args)
    if args.evaluate_checkpoint:
        ckpt = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        if "ema" in ckpt:
            ckpt = ckpt["ema"]
        msd = model.state_dict()
        for k, v in ckpt.items():
            if k in msd:
                msd[k] = v
        model.load_state_dict(msd)
    model.to(device).eval()
    return model


def make_pipe(args, vae, model, scheduler_kind: str):
    if scheduler_kind == "DPM":
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=args.beta_start, beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            algorithm_type="dpmsolver++", solver_order=2,
        )
    else:  # PNDM (default IRASim sampler)
        scheduler = PNDMScheduler.from_pretrained(
            args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end,
            beta_schedule=args.beta_schedule, variance_type=args.variance_type,
        )
    return Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)


def gen_one(pipe, mask_x, actions, args, device):
    with torch.no_grad():
        videos, latents = pipe(
            actions.to(device).to(torch.float32),
            mask_x=mask_x.to(device).to(torch.float32),
            video_length=args.num_frames,
            height=args.video_size[0], width=args.video_size[1],
            num_inference_steps=args.infer_num_sampling_steps,
            guidance_scale=args.guidance_scale,
            device=device, return_dict=False, output_type="both",
        )
    return videos, latents


def write_mp4(video_tensor, path):
    t = ((video_tensor / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu()
    t = rearrange(t, "f c h w -> f h w c").numpy()
    w = imageio.get_writer(path, fps=4)
    for fr in t:
        w.append_data(fr)
    w.close()


def compute_actions_for_slice(arm_states, gripper_states):
    n = arm_states.shape[0]
    action_num = n - 1
    action = np.zeros((action_num, ACTION_DIM))
    for k in range(1, action_num + 1):
        prev_xyz = arm_states[k - 1, 0:3]
        prev_rpy = arm_states[k - 1, 3:6]
        prev_rotm = euler2rotm(prev_rpy)
        curr_xyz = arm_states[k, 0:3]
        curr_rpy = arm_states[k, 3:6]
        curr_gripper = gripper_states[k]
        curr_rotm = euler2rotm(curr_rpy)
        rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ curr_rotm
        rel_rpy = rotm2euler(rel_rotm)
        action[k - 1, 0:3] = rel_xyz
        action[k - 1, 3:6] = rel_rpy
        action[k - 1, 6] = curr_gripper
    return action


def parse_key(filename):
    base = os.path.basename(filename)[:-3]
    parts = base.split("_")
    eid = "_".join(parts[:-2])
    cam = int(parts[-2])
    start = int(parts[-1])
    return eid, cam, start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--skip", default="")
    p.add_argument("--tag", required=True)
    p.add_argument("--num_samples", type=int, required=True)
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--scheduler", choices=["PNDM", "DPM"], default="PNDM")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--out_root", default="/home/dingqy/Bench/IRASim/results/local_n_eval")
    cli = p.parse_args()

    args = build_args(cli.config, cli.inference_steps)
    skip_map = parse_skip(cli.skip)

    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)
    reconfigure(args.attention_mode)
    clear_skip_blocks()
    for op, blocks in skip_map.items():
        set_skip_blocks(op, blocks)
    pipe = make_pipe(args, vae, model, cli.scheduler)
    print(f"[shard {cli.shard}/{cli.num_shards}] variant={args.attention_mode} skip={skip_map} "
          f"steps={args.infer_num_sampling_steps} sched={cli.scheduler}", flush=True)

    # Decide which samples to handle this shard.
    all_files = sorted(f for f in os.listdir(GT_LATENT_DIR) if f.endswith(".pt"))[: cli.num_samples]
    shard_files = all_files[cli.shard::cli.num_shards]
    print(f"[shard {cli.shard}] {len(shard_files)}/{len(all_files)} samples", flush=True)

    out_dir = os.path.join(cli.out_root, cli.tag)
    vid_dir = os.path.join(out_dir, "videos")
    met_dir = os.path.join(out_dir, "metrics")
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    metrics = []
    t_start = time.time()

    for fn in shard_files:
        eid, cam, start = parse_key(fn)
        key = f"{eid}_{cam}_{start}"
        ann_path = os.path.join(ANNOT_DIR, f"{eid}.json")
        if not os.path.exists(ann_path):
            print(f"  [{key}] skip — no annotation", flush=True)
            continue
        try:
            gt_lat = torch.load(os.path.join(GT_LATENT_DIR, fn), weights_only=False, map_location=device)
            with open(ann_path) as f:
                ann = json.load(f)
            n_total = len(ann["state"])
            if start + SEQUENCE_LENGTH > n_total:
                print(f"  [{key}] skip — oob (start+16>{n_total})", flush=True)
                continue
            arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
            grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
            actions_np = compute_actions_for_slice(arm, grip) * C_ACT_SCALER
            actions = torch.from_numpy(actions_np).float().unsqueeze(0)
            mask_x = gt_lat[0:1].unsqueeze(0)

            t0 = time.time()
            videos, pred_lat = gen_one(pipe, mask_x, actions, args, device)
            dt = time.time() - t0
            pred_lat = pred_lat.squeeze(0)
            l2 = F.mse_loss(pred_lat[1:], gt_lat[1:]).item()

            mp4_path = os.path.join(vid_dir, f"{key}.mp4")
            write_mp4(videos.squeeze(0), mp4_path)
            _, ps = process_video_psnr_ssim(f"{key}.mp4", vid_dir, GT_VIDEO_DIR)

            metric = {
                "key": key, "latent_l2": round(l2, 4),
                "psnr": round(float(ps["PSNR"]), 3),
                "ssim": round(float(ps["SSIM"]), 3),
                "infer_seconds": round(dt, 2),
            }
            with open(os.path.join(met_dir, f"{key}.json"), "w") as f:
                json.dump(metric, f)
            metrics.append(metric)
            elapsed = time.time() - t_start
            rate = len(metrics) / elapsed if elapsed > 0 else 0
            eta = (len(shard_files) - len(metrics)) / rate / 60 if rate > 0 else 0
            print(f"[shard {cli.shard}] {key}  L2={l2:.4f} PSNR={ps['PSNR']:.2f} SSIM={ps['SSIM']:.3f}  "
                  f"({len(metrics)}/{len(shard_files)} done, {dt:.1f}s; eta {eta:.1f}m)", flush=True)
        except Exception:
            print(f"  [{key}] FAILED:", flush=True)
            traceback.print_exc()
            continue

    # Aggregate
    if metrics:
        mean_l2 = sum(m["latent_l2"] for m in metrics) / len(metrics)
        mean_psnr = sum(m["psnr"] for m in metrics) / len(metrics)
        mean_ssim = sum(m["ssim"] for m in metrics) / len(metrics)
        summary = {
            "tag": cli.tag, "shard": cli.shard, "num_shards": cli.num_shards,
            "variant": args.attention_mode, "skip": {k: list(v) for k, v in skip_map.items()},
            "steps": args.infer_num_sampling_steps, "scheduler": cli.scheduler,
            "n_samples_done": len(metrics),
            "mean_l2": round(mean_l2, 4), "mean_psnr": round(mean_psnr, 3), "mean_ssim": round(mean_ssim, 3),
            "per_sample": metrics,
        }
        with open(os.path.join(out_dir, f"summary_shard_{cli.shard}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[shard {cli.shard}] DONE — mean L2={mean_l2:.4f} PSNR={mean_psnr:.2f} SSIM={mean_ssim:.3f}", flush=True)


if __name__ == "__main__":
    main()
