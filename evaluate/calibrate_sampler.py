"""5-sample FP calibration: PNDM 50 vs DPM-Solver++ 25.

If ΔPSNR < 0.5 dB across 5 samples, we conclude the SC eval (DPM 25) is
fairly comparable to the IRASim paper's reported PNDM 50 baseline numbers.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from einops import rearrange
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler, DPMSolverMultistepScheduler

sys.path.insert(0, "/home/dingqy/Bench/IRASim")
os.chdir("/home/dingqy/Bench/IRASim")

from dataset.dataset_util import euler2rotm, rotm2euler
from models import get_models
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from evaluate.compute_psnr_ssim import process_video_psnr_ssim
from util import update_paths

# Use FP config (no SC)
CONFIG = "configs/evaluation/bridge/frame_ada.yaml"
BRIDGE_ROOT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge"
GT_LATENT_DIR = f"{BRIDGE_ROOT}/evaluation_latent_videos/test_sample_latent_videos"
GT_VIDEO_DIR = f"{BRIDGE_ROOT}/evaluation_videos/test_sample_videos"
ANNOT_DIR = f"{BRIDGE_ROOT}/annotation/test"
LOCAL_OUT = "/tmp/sampler_calib"
SEQUENCE_LENGTH = 16
ACTION_DIM = 7
C_ACT_SCALER = np.array([20.0]*6 + [1.0], dtype=float)
N_CALIB = 5


def build_args(steps):
    args = OmegaConf.merge(
        OmegaConf.load("configs/base/data.yaml"),
        OmegaConf.load("configs/base/diffusion.yaml"),
        OmegaConf.load(CONFIG),
    )
    update_paths(args)
    args.latent_size = [t // 8 for t in args.video_size]
    args.infer_num_sampling_steps = steps
    return args


def load_model(args, device):
    model = get_models(args)
    if args.evaluate_checkpoint:
        ckpt = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        if "ema" in ckpt: ckpt = ckpt["ema"]
        msd = model.state_dict()
        for k, v in ckpt.items():
            if k in msd: msd[k] = v
        model.load_state_dict(msd)
    model.to(device).eval()
    return model


def compute_actions(arm, grip):
    n = arm.shape[0]; out = np.zeros((n - 1, ACTION_DIM))
    for k in range(1, n):
        prev_xyz, prev_rpy = arm[k-1, 0:3], arm[k-1, 3:6]
        curr_xyz, curr_rpy = arm[k, 0:3], arm[k, 3:6]
        prev_rotm = euler2rotm(prev_rpy); curr_rotm = euler2rotm(curr_rpy)
        rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ curr_rotm
        rel_rpy = rotm2euler(rel_rotm)
        out[k-1, 0:3] = rel_xyz
        out[k-1, 3:6] = rel_rpy
        out[k-1, 6] = grip[k]
    return out


def parse_key(filename):
    base = os.path.basename(filename)[:-3]
    parts = base.split("_")
    return "_".join(parts[:-2]), int(parts[-2]), int(parts[-1])


def write_mp4(video_tensor, path):
    t = ((video_tensor / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu()
    t = rearrange(t, "f c h w -> f h w c").numpy()
    w = imageio.get_writer(path, fps=4)
    for fr in t: w.append_data(fr)
    w.close()


def gen(pipe, mask_x, actions, args, device):
    with torch.no_grad():
        videos, latents = pipe(
            actions.to(device).float(), mask_x=mask_x.to(device).float(),
            video_length=args.num_frames,
            height=args.video_size[0], width=args.video_size[1],
            num_inference_steps=args.infer_num_sampling_steps,
            guidance_scale=args.guidance_scale,
            device=device, return_dict=False, output_type="both",
        )
    return videos, latents


def metrics_for(key, pred_lat, pred_video, gt_lat, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mp4 = os.path.join(out_dir, f"{key}.mp4")
    write_mp4(pred_video, mp4)
    _, ps = process_video_psnr_ssim(f"{key}.mp4", out_dir, GT_VIDEO_DIR)
    l2 = F.mse_loss(pred_lat[1:], gt_lat[1:]).item()
    return {"L2": round(l2, 4), "PSNR": round(float(ps["PSNR"]), 3),
            "SSIM": round(float(ps["SSIM"]), 3)}


def run_pass(name, scheduler_kind, steps, samples, vae, model, device):
    args = build_args(steps)
    if scheduler_kind == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(
            args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end,
            beta_schedule=args.beta_schedule, variance_type=args.variance_type)
    else:
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000, beta_start=args.beta_start, beta_end=args.beta_end,
            beta_schedule=args.beta_schedule, algorithm_type="dpmsolver++", solver_order=2)
    pipe = Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)

    out = {}
    for key, gt_lat, actions in samples:
        mask_x = gt_lat[0:1].unsqueeze(0)
        actions_b = actions.unsqueeze(0)
        t0 = time.perf_counter()
        videos, pred_lat = gen(pipe, mask_x, actions_b, args, device)
        dt = time.perf_counter() - t0
        m = metrics_for(key, pred_lat.squeeze(0), videos.squeeze(0),
                        gt_lat, os.path.join(LOCAL_OUT, name))
        out[key] = {**m, "t": round(dt, 1)}
        print(f"  {name} {key}: {out[key]}", flush=True)
    return out


def main():
    os.makedirs(LOCAL_OUT, exist_ok=True)
    device = torch.device("cuda:0")

    # Load model once (FP path, attention_mode='math')
    args = build_args(50)
    print(f"FP attention_mode={args.attention_mode}", flush=True)
    print("Loading model...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)
    print("Loaded.", flush=True)

    # Pick first 5 valid samples
    files = sorted(f for f in os.listdir(GT_LATENT_DIR) if f.endswith(".pt"))
    samples = []
    for f in files:
        eid, cam, start = parse_key(f)
        if not os.path.exists(os.path.join(ANNOT_DIR, f"{eid}.json")): continue
        gt_lat = torch.load(os.path.join(GT_LATENT_DIR, f), weights_only=False, map_location=device)
        with open(os.path.join(ANNOT_DIR, f"{eid}.json")) as fp:
            ann = json.load(fp)
        if start + SEQUENCE_LENGTH > len(ann["state"]): continue
        arm = np.array(ann["state"])[start:start+SEQUENCE_LENGTH, :6]
        grip = np.array(ann["continuous_gripper_state"])[start:start+SEQUENCE_LENGTH]
        actions = torch.from_numpy(compute_actions(arm, grip) * C_ACT_SCALER).float()
        key = f"{eid}_{cam}_{start}"
        samples.append((key, gt_lat, actions))
        if len(samples) == N_CALIB: break
    keys = [s[0] for s in samples]
    print(f"calibration set: {keys}", flush=True)

    print("\n=== FP / PNDM 50 (paper config) ===", flush=True)
    pndm = run_pass("pndm50", "PNDM", 50, samples, vae, model, device)

    print("\n=== FP / DPM-Solver++ 25 (our SC config) ===", flush=True)
    dpm = run_pass("dpm25", "DPM", 25, samples, vae, model, device)

    print("\n=== compare ===", flush=True)
    dp_acc, ds_acc, dl_acc = [], [], []
    for k in keys:
        dpsnr = dpm[k]["PSNR"] - pndm[k]["PSNR"]
        dssim = dpm[k]["SSIM"] - pndm[k]["SSIM"]
        dl2   = dpm[k]["L2"] - pndm[k]["L2"]
        dp_acc.append(dpsnr); ds_acc.append(dssim); dl_acc.append(dl2)
        print(f"  {k}: PNDM50 PSNR={pndm[k]['PSNR']:.2f} L2={pndm[k]['L2']:.4f} SSIM={pndm[k]['SSIM']:.3f} ({pndm[k]['t']}s)  "
              f"|  DPM25 PSNR={dpm[k]['PSNR']:.2f} L2={dpm[k]['L2']:.4f} SSIM={dpm[k]['SSIM']:.3f} ({dpm[k]['t']}s)  "
              f"|  ΔPSNR={dpsnr:+.3f} ΔSSIM={dssim:+.3f} ΔL2={dl2:+.4f}", flush=True)
    n = len(keys)
    print(f"\nMEAN ΔPSNR={sum(dp_acc)/n:+.3f}dB  ΔSSIM={sum(ds_acc)/n:+.3f}  ΔL2={sum(dl_acc)/n:+.4f}", flush=True)
    pndm_t = sum(pndm[k]["t"] for k in keys)/n
    dpm_t  = sum(dpm[k]["t"] for k in keys)/n
    print(f"speed: PNDM50 ~{pndm_t:.1f}s/sample, DPM25 ~{dpm_t:.1f}s/sample (speedup {pndm_t/dpm_t:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
