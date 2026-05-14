import os
import sys
import torch
import imageio
import numpy as np
from diffusers.models import AutoencoderKL
from einops import rearrange
from evaluate.compute_psnr_ssim import compute_psnr_ssim

DATA_ROOT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge"
GT_LATENT_DIR = f"{DATA_ROOT}/evaluation_latent_videos/test_sample_latent_videos"
GT_VIDEO_DIR = f"{DATA_ROOT}/evaluation_videos/test_sample_videos"
PRED_VIDEO_DIR = "/home/dingqy/Bench/IRASim/results/04/24/test_bridge_frame_ada-debug/checkpoints/0300000/test_sample_videos"
VAE_PATH = "/home/dingqy/Bench/IRASim/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0"

os.makedirs(GT_VIDEO_DIR, exist_ok=True)

device = torch.device("cuda:0")
vae = AutoencoderKL.from_pretrained(VAE_PATH, subfolder="vae").to(device)
vae.eval()
sf = vae.config.scaling_factor
print(f"VAE loaded, scaling_factor={sf}")

for fname in sorted(os.listdir(GT_LATENT_DIR)):
    if not fname.endswith(".pt"):
        continue
    lat = torch.load(os.path.join(GT_LATENT_DIR, fname), map_location=device).to(device)
    with torch.no_grad():
        dec = vae.decode(lat / sf).sample
    frames = ((dec / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu()
    frames = rearrange(frames, "f c h w -> f h w c").numpy()
    out_path = os.path.join(GT_VIDEO_DIR, fname.replace(".pt", ".mp4"))
    writer = imageio.get_writer(out_path, fps=4)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print(f"decoded {fname} -> {out_path}  shape={frames.shape}")

print("\n--- Running compute_psnr_ssim ---")
psnr, ssim_val = compute_psnr_ssim(PRED_VIDEO_DIR, GT_VIDEO_DIR)
print(f"PSNR: {psnr:.3f}")
print(f"SSIM: {ssim_val:.3f}")
