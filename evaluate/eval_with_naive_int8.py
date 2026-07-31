"""Run short eval with NAIVE int8 quantization (per-tensor symmetric, fake-quant)
applied to the same op categories as a chosen SC config.

This is the standard PTQ baseline to compare against SC int8.

Usage:
    PYTHONPATH=. python3 evaluate/eval_with_naive_int8.py \
        --config configs/evaluation/bridge/frame_ada_sc_full.yaml \
        --tag naive_int8_full \
        --mode short

Quantization scheme: y = (round(a/sa).clamp(-127,127)*sa) @ (round(b/sb).clamp(-127,127)*sb)
where sa = max(|a|)/127, sb = max(|b|)/127. For Linear, weight scale is per-tensor.
"""
from __future__ import annotations

import argparse
import json
import os
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler

from dataset import get_dataset
from models import get_models
from models.sc_integration import reconfigure, clear_skip_blocks
import models.sc_integration.sc_attention as sc_attention
import models.sc_integration.sc_linear as sc_linear
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from evaluate.compute_psnr_ssim import process_video_psnr_ssim
from util import update_paths
import imageio


# Bit width / symmetry for the integer baseline. Set by install_naive_int8_patches();
# the defaults reproduce the original per-tensor symmetric int8 path exactly.
_QUANT_BITS = 8
_QUANT_ASYMM = False


def _q_int8_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor fake-quant of x at _QUANT_BITS. Returns dequantized fp32 tensor.

    symmetric:  q = round(x/s).clamp(-qmax, qmax),  s = max|x| / qmax,  qmax = 2^(b-1)-1
    asymmetric: q = (round(x/s)+z).clamp(0, qmax),  s = (max-min) / qmax, qmax = 2^b - 1
    """
    if x.numel() == 0:
        return x.float()
    xf = x.float()
    if _QUANT_ASYMM:
        qmax = float(2 ** _QUANT_BITS - 1)
        xmin = xf.detach().amin()
        xmax = xf.detach().amax()
        scale = ((xmax - xmin) / qmax).clamp_min(1e-8)
        zp = (-xmin / scale).round()
        q = ((xf / scale).round() + zp).clamp(0, qmax)
        return (q - zp) * scale
    qmax = float(2 ** (_QUANT_BITS - 1) - 1)
    amax = xf.detach().abs().amax().clamp_min(1e-8)
    scale = amax / qmax
    q = (xf / scale).round().clamp(-qmax, qmax)
    return q * scale


def naive_int8_qk(q_scaled: torch.Tensor, k: torch.Tensor, sc_prec: int = 8, stoc_len=None) -> torch.Tensor:
    qd = _q_int8_per_tensor(q_scaled)
    kd = _q_int8_per_tensor(k)
    return torch.matmul(qd, kd.transpose(-1, -2)).to(q_scaled.dtype)


def naive_int8_av(attn: torch.Tensor, v: torch.Tensor, sc_prec: int = 8, stoc_len=None) -> torch.Tensor:
    ad = _q_int8_per_tensor(attn)
    vd = _q_int8_per_tensor(v)
    return torch.matmul(ad, vd).to(attn.dtype)


def naive_int8_linear_forward(x: torch.Tensor, linear, sc_prec: int = 8, stoc_len=None) -> torch.Tensor:
    """y = (q(x) @ q(W)^T) + bias"""
    xd = _q_int8_per_tensor(x)
    wd = _q_int8_per_tensor(linear.weight)
    y = F.linear(xd, wd, linear.bias.float() if linear.bias is not None else None)
    return y.to(x.dtype)


def install_naive_int8_patches(bits: int = 8, asymm: bool = False):
    """Swap the SC kernels for plain WxAx fake-quant matmuls (x = `bits`)."""
    global _QUANT_BITS, _QUANT_ASYMM
    if not 2 <= bits <= 8:
        raise ValueError(f"naive quant bits must be in [2,8], got {bits}")
    _QUANT_BITS, _QUANT_ASYMM = bits, asymm
    sc_attention.sc_qk_matmul = naive_int8_qk
    sc_attention.sc_av_matmul = naive_int8_av
    sc_linear.sc_linear_forward = naive_int8_linear_forward
    # Update reference in attention/linear/mlp modules that imported the symbol.
    import models.sc_integration as _root
    _root.sc_qk_matmul = naive_int8_qk
    _root.sc_av_matmul = naive_int8_av
    _root.sc_linear_forward = naive_int8_linear_forward
    # Update sc_mlp.py which captured sc_linear_forward at import time
    import models.sc_integration.sc_mlp as sc_mlp
    sc_mlp.sc_linear_forward = naive_int8_linear_forward
    # Update irasim.py which also imported sc_linear_forward
    import models.irasim as irasim_mod
    if hasattr(irasim_mod, "sc_linear_forward"):
        irasim_mod.sc_linear_forward = naive_int8_linear_forward
    if hasattr(irasim_mod, "sc_qk_matmul"):
        irasim_mod.sc_qk_matmul = naive_int8_qk
    if hasattr(irasim_mod, "sc_av_matmul"):
        irasim_mod.sc_av_matmul = naive_int8_av


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
        for k, v in ckpt.items():
            if k in msd:
                msd[k] = v
        model.load_state_dict(msd)
    model.to(device).eval()
    return model


def run_pipeline(args, mask_x, actions, vae, model, device):
    scheduler = PNDMScheduler.from_pretrained(
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
            output_type="both",
        )
    return videos, latents


def run_short_eval(args, val_dataloader, vae, model, device, gt_lat_dir, gt_video_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for batch in val_dataloader:
        if args.pre_encode:
            x = batch["latent"].to(device)
        else:
            video = batch["video"].to(device)
            b, f, _, _, _ = video.shape
            video = rearrange(video, "b f c h w -> (b f) c h w").contiguous()
            enc = []
            for i in range(0, video.size(0), 16):
                enc.append(vae.encode(video[i:i+16]).latent_dist.sample().mul_(vae.config.scaling_factor))
            x = rearrange(torch.cat(enc, 0), "(b f) c h w -> b f c h w", b=b, f=f)
        mask_x = x[:, 0:1]
        actions = batch["action"]
        eid = batch["video_name"]["episode_id"][0]
        cam = batch["video_name"]["cam_id"][0]
        sf = batch["video_name"]["start_frame_id"][0]
        key = f"{eid}_{cam}_{sf}"

        gt_lat = torch.load(os.path.join(gt_lat_dir, f"{key}.pt"), map_location=device)
        videos, pred_lat = run_pipeline(args, mask_x, actions, vae, model, device)
        pred_lat = pred_lat.squeeze(0)
        l2 = F.mse_loss(pred_lat[1:], gt_lat[1:]).item()

        pred_video = videos.squeeze(0).permute(0, 2, 3, 1)
        pred_video = ((pred_video / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        pred_mp4 = os.path.join(out_dir, f"{key}.mp4")
        w = imageio.get_writer(pred_mp4, fps=4)
        for fr in pred_video:
            w.append_data(fr)
        w.close()

        # PSNR/SSIM vs GT mp4
        _, ps = process_video_psnr_ssim(f"{key}.mp4", out_dir, gt_video_dir)

        results[key] = {
            "latent_l2": round(l2, 4),
            "psnr": round(float(ps["PSNR"]), 3),
            "ssim": round(float(ps["SSIM"]), 3),
        }
        print(f"  short {key}: L2={l2:.4f}, PSNR={ps['PSNR']:.3f}, SSIM={ps['SSIM']:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--mode", choices=["short"], default="short")
    parser.add_argument("--out_root", default="/home/dingqy/Bench/IRASim/results/naive_int8_eval")
    cli = parser.parse_args()

    args = build_args(cli.config)
    device = torch.device("cuda:0")

    # Patch SC kernels to naive int8 BEFORE model construction (so any captured
    # references in TransformerBlock / Attention / Mlp use the patched versions).
    install_naive_int8_patches()
    print("INSTALLED naive int8 patches over sc_* kernels")

    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)

    reconfigure(args.attention_mode)
    clear_skip_blocks()
    print(f"variant={args.attention_mode} (with naive int8 kernels in place of SC int8)")

    out_dir = os.path.join(cli.out_root, f"{cli.tag}_short")
    payload = {"tag": cli.tag, "config": cli.config, "attention_mode": args.attention_mode,
               "kernel": "naive_int8_per_tensor_sym"}

    _, val_dataset = get_dataset(args)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    gt_lat_dir = args.true_sample_latent_videos_dir
    gt_video_dir = args.true_sample_videos_dir
    print(f"=== short eval [{cli.tag}] ===")
    payload["short"] = run_short_eval(args, val_dataloader, vae, model, device,
                                       gt_lat_dir, gt_video_dir, out_dir)

    # Aggregate
    s = payload["short"]
    n = len(s)
    if n:
        payload["short_mean"] = {
            "latent_l2": round(sum(v["latent_l2"] for v in s.values()) / n, 4),
            "psnr": round(sum(v["psnr"] for v in s.values()) / n, 3),
            "ssim": round(sum(v["ssim"] for v in s.values()) / n, 3),
        }
        print(f"\nMEAN over {n} samples: {payload['short_mean']}")

    out_json = os.path.join(cli.out_root, f"{cli.tag}.json")
    os.makedirs(cli.out_root, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
