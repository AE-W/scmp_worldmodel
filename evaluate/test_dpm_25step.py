"""Test DPM-Solver++ at 25 steps vs PNDM 50 steps for sc_int8_full+top6.

Runs 1 sample (0_0_0) with DPMSolverMultistepScheduler at 25 steps.
Compares to known PNDM 50-step reference (PSNR=16.51, L2=0.34).
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
from diffusers.schedulers import DPMSolverMultistepScheduler

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
NUM_INFER_STEPS = 25
BRIDGE_ROOT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge"
GT_LATENT_DIR = f"{BRIDGE_ROOT}/evaluation_latent_videos/test_sample_latent_videos"
GT_VIDEO_DIR = f"{BRIDGE_ROOT}/evaluation_videos/test_sample_videos"
ANNOT_DIR = f"{BRIDGE_ROOT}/annotation/test"
LOCAL_OUT = "/tmp/dpm_test"
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


def write_mp4(video_tensor, path):
    t = ((video_tensor / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu()
    t = rearrange(t, "f c h w -> f h w c").numpy()
    w = imageio.get_writer(path, fps=4)
    for fr in t: w.append_data(fr)
    w.close()


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

    # Use DPM-Solver++ instead of PNDM
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=args.beta_start, beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        algorithm_type="dpmsolver++",
        solver_order=2,
    )
    pipe = Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)
    print(f"sampler=DPMSolverMultistep++ steps={args.infer_num_sampling_steps}", flush=True)

    # Pick 0_0_0
    fn = "0_0_0.pt"
    eid, cam, start = "0", 0, 0
    key = f"{eid}_{cam}_{start}"
    gt_lat = torch.load(os.path.join(GT_LATENT_DIR, fn), weights_only=False, map_location=device)
    with open(os.path.join(ANNOT_DIR, f"{eid}.json")) as f:
        ann = json.load(f)
    arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
    grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
    actions = torch.from_numpy(compute_actions(arm, grip) * C_ACT_SCALER).float().unsqueeze(0)
    mask_x = gt_lat[0:1].unsqueeze(0)

    t0 = time.perf_counter()
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
    dt = time.perf_counter() - t0
    pred_lat = latents.squeeze(0)
    pred_video = videos.squeeze(0)

    mp4 = os.path.join(LOCAL_OUT, f"{key}.mp4")
    write_mp4(pred_video, mp4)
    _, ps = process_video_psnr_ssim(f"{key}.mp4", LOCAL_OUT, GT_VIDEO_DIR)
    l2 = F.mse_loss(pred_lat[1:], gt_lat[1:]).item()

    print(f"\nDPM 25-step result for {key}:", flush=True)
    print(f"  L2={l2:.4f} PSNR={ps['PSNR']:.3f} SSIM={ps['SSIM']:.3f}  in {dt:.1f}s", flush=True)
    print(f"\nReference (PNDM 50-step):", flush=True)
    print(f"  L2=0.3397 PSNR=16.514 SSIM=0.469  in 612.8s", flush=True)
    print(f"\nDelta:", flush=True)
    print(f"  ΔPSNR={ps['PSNR']-16.514:+.3f}  ΔL2={l2-0.3397:+.4f}  Speedup={612.8/dt:.2f}×", flush=True)


if __name__ == "__main__":
    main()
