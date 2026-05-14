"""Quick test: batch=1 vs batch=4 timing + per-sample PSNR for sc_int8_full+top6, 50 steps.

Picks 4 samples from shard 0 (including 0_0_0 for reference), runs:
  - batch=1: 4 sequential calls
  - batch=4: 1 stacked call

Reports per-sample PSNR, L2 to verify batch=4 doesn't degrade quality.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import imageio
from einops import rearrange
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler

sys.path.insert(0, "/home/dingqy/Bench/IRASim")
os.chdir("/home/dingqy/Bench/IRASim")

from dataset.dataset_util import euler2rotm, rotm2euler
from models import get_models
from models.sc_integration import reconfigure, set_skip_blocks, clear_skip_blocks
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from evaluate.compute_psnr_ssim import process_video_psnr_ssim
from util import update_paths

CONFIG = "configs/evaluation/bridge/frame_ada_sc_full.yaml"
SKIP = {"mlp_fc1": [25, 27, 4, 26, 13, 16],
        "mlp_fc2": [27, 0, 25, 4, 26, 2],
        "qkv":     [25, 7, 8, 6, 14, 26]}
NUM_INFER_STEPS = 50
BRIDGE_ROOT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge"
GT_LATENT_DIR = f"{BRIDGE_ROOT}/evaluation_latent_videos/test_sample_latent_videos"
GT_VIDEO_DIR = f"{BRIDGE_ROOT}/evaluation_videos/test_sample_videos"
ANNOT_DIR = f"{BRIDGE_ROOT}/annotation/test"
LOCAL_OUT = "/tmp/batch_test"
SEQUENCE_LENGTH = 16
ACTION_DIM = 7
C_ACT_SCALER = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0], dtype=float)


def build_args():
    args = OmegaConf.merge(
        OmegaConf.load("configs/base/data.yaml"),
        OmegaConf.load("configs/base/diffusion.yaml"),
        OmegaConf.load(CONFIG),
    )
    update_paths(args)
    args.latent_size = [t // 8 for t in args.video_size]
    args.infer_num_sampling_steps = NUM_INFER_STEPS
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


def make_pipe(args, vae, model):
    scheduler = PNDMScheduler.from_pretrained(
        args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end,
        beta_schedule=args.beta_schedule, variance_type=args.variance_type,
    )
    return Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)


def compute_actions(arm, grip):
    n = arm.shape[0]; out = np.zeros((n - 1, ACTION_DIM))
    for k in range(1, n):
        prev_xyz, prev_rpy = arm[k - 1, 0:3], arm[k - 1, 3:6]
        curr_xyz, curr_rpy = arm[k, 0:3], arm[k, 3:6]
        prev_rotm = euler2rotm(prev_rpy); curr_rotm = euler2rotm(curr_rpy)
        rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ curr_rotm
        rel_rpy = rotm2euler(rel_rotm)
        out[k - 1, 0:3] = rel_xyz
        out[k - 1, 3:6] = rel_rpy
        out[k - 1, 6] = grip[k]
    return out


def parse_key(filename):
    base = os.path.basename(filename)[:-3]
    parts = base.split("_")
    return "_".join(parts[:-2]), int(parts[-2]), int(parts[-1])


def load_one_sample(fn, device):
    eid, cam, start = parse_key(fn)
    key = f"{eid}_{cam}_{start}"
    gt_lat = torch.load(os.path.join(GT_LATENT_DIR, fn), weights_only=False, map_location=device)
    with open(os.path.join(ANNOT_DIR, f"{eid}.json")) as f:
        ann = json.load(f)
    arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
    grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
    actions_np = compute_actions(arm, grip) * C_ACT_SCALER
    actions = torch.from_numpy(actions_np).float()
    return key, gt_lat, actions


def write_mp4(video_tensor, path):
    t = ((video_tensor / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu()
    t = rearrange(t, "f c h w -> f h w c").numpy()
    w = imageio.get_writer(path, fps=4)
    for fr in t: w.append_data(fr)
    w.close()


def gen(pipe, mask_x, actions, args, device):
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


def run_metrics(key, pred_lat, gt_lat, pred_video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mp4 = os.path.join(out_dir, f"{key}.mp4")
    write_mp4(pred_video, mp4)
    _, ps = process_video_psnr_ssim(f"{key}.mp4", out_dir, GT_VIDEO_DIR)
    l2 = F.mse_loss(pred_lat[1:], gt_lat[1:]).item()
    return {"L2": round(l2, 4), "PSNR": round(float(ps["PSNR"]), 3),
            "SSIM": round(float(ps["SSIM"]), 3)}


def main():
    os.makedirs(LOCAL_OUT, exist_ok=True)
    args = build_args()
    device = torch.device("cuda:0")

    print("Loading model...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)
    reconfigure(args.attention_mode); clear_skip_blocks()
    for op, blocks in SKIP.items(): set_skip_blocks(op, blocks)
    pipe = make_pipe(args, vae, model)
    print(f"mode={args.attention_mode} steps={args.infer_num_sampling_steps}", flush=True)

    # Pick 4 samples (shard 0 first 4)
    files = sorted(f for f in os.listdir(GT_LATENT_DIR) if f.endswith(".pt"))
    test_files = files[0:4:1]  # 4 contiguous samples
    # Filter to those with annotations available
    test_files = [f for f in test_files
                  if os.path.exists(os.path.join(ANNOT_DIR, f"{parse_key(f)[0]}.json"))][:4]
    if len(test_files) < 4:
        # fall back: pick from shard 0 samples (every 4th)
        test_files = [f for f in files
                      if os.path.exists(os.path.join(ANNOT_DIR, f"{parse_key(f)[0]}.json"))][:4]
    print(f"test files: {test_files}", flush=True)

    samples = [load_one_sample(f, device) for f in test_files]

    # ---- batch=1 baseline ----
    print("\n=== batch=1 ===", flush=True)
    b1_metrics = {}
    t0 = time.perf_counter()
    for key, gt_lat, actions in samples:
        mask_x = gt_lat[0:1].unsqueeze(0)
        actions_b = actions.unsqueeze(0)
        videos, pred_lat = gen(pipe, mask_x, actions_b, args, device)
        m = run_metrics(key, pred_lat.squeeze(0), gt_lat,
                        videos.squeeze(0), os.path.join(LOCAL_OUT, "b1"))
        b1_metrics[key] = m
        print(f"  b1 {key}: {m}", flush=True)
    t_b1 = time.perf_counter() - t0
    print(f"b1 total: {t_b1:.1f}s for {len(samples)} samples ({t_b1/len(samples):.1f}s/sample)", flush=True)

    # ---- batch=4 ----
    print("\n=== batch=4 ===", flush=True)
    keys = [s[0] for s in samples]
    gt_lats = torch.stack([s[1] for s in samples], dim=0)  # (4, 16, 4, 32, 40)
    actions_batch = torch.stack([s[2] for s in samples], dim=0)  # (4, 15, 7)
    mask_batch = gt_lats[:, 0:1]  # (4, 1, 4, 32, 40)
    t0 = time.perf_counter()
    videos, pred_lats = gen(pipe, mask_batch, actions_batch, args, device)
    t_b4 = time.perf_counter() - t0
    print(f"b4 forward: {t_b4:.1f}s for {len(samples)} samples ({t_b4/len(samples):.1f}s/sample)", flush=True)
    b4_metrics = {}
    for i, key in enumerate(keys):
        m = run_metrics(key, pred_lats[i], gt_lats[i],
                        videos[i], os.path.join(LOCAL_OUT, "b4"))
        b4_metrics[key] = m
        print(f"  b4 {key}: {m}", flush=True)

    # ---- compare ----
    print("\n=== compare ===", flush=True)
    for key in keys:
        b1 = b1_metrics[key]; b4 = b4_metrics[key]
        dpsnr = b4["PSNR"] - b1["PSNR"]
        dl2 = b4["L2"] - b1["L2"]
        print(f"  {key}: b1 PSNR={b1['PSNR']:.2f} L2={b1['L2']:.4f}  "
              f"|  b4 PSNR={b4['PSNR']:.2f} L2={b4['L2']:.4f}  "
              f"|  ΔPSNR={dpsnr:+.3f} ΔL2={dl2:+.4f}", flush=True)
    speedup = t_b1 / t_b4
    print(f"\nSpeedup b4 vs b1: {speedup:.2f}× ({t_b1:.1f}s -> {t_b4:.1f}s for 4 samples)", flush=True)


if __name__ == "__main__":
    main()
