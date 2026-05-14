"""Run short + long eval with a per-op block-skip schedule applied to SC.

Usage:
    PYTHONPATH=. python3 evaluate/eval_with_skip.py \
        --config configs/evaluation/bridge/frame_ada_sc_qk_av_proj_fc1.yaml \
        --skip 'mlp_fc1=25,27' \
        --tag fc1_skip2 \
        --mode short    # or 'long' or 'both'

Skip syntax: comma-separated `op=blocks` clauses, blocks are dash-separated
ints (e.g. "mlp_fc1=25,27;mlp_fc2=25,27"). Multiple ops separated by ";".
"""
from __future__ import annotations

import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler, DDPMScheduler

from dataset import get_dataset
from models import get_models
from models.sc_integration import (
    reconfigure, set_skip_blocks, clear_skip_blocks,
)
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from evaluate.compute_psnr_ssim import process_video_psnr
from util import update_paths
import imageio
import json as _json


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
    if args.sample_method == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(
            args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end,
            beta_schedule=args.beta_schedule, variance_type=args.variance_type,
        )
    else:
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
            output_type="both",
        )
    return videos, latents


def run_short_eval(args, val_dataloader, vae, model, device, gt_lat_dir, out_dir):
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
        # PSNR vs VAE-decoded GT latent (matches existing per_video_metrics)
        from evaluate.compute_latent_l2 import process_video as _l2_dummy  # noqa: F401
        # Decode pred to mp4
        pred_video = videos.squeeze(0).permute(0, 2, 3, 1)
        pred_video = ((pred_video / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        pred_mp4 = os.path.join(out_dir, f"{key}.mp4")
        w = imageio.get_writer(pred_mp4, fps=4)
        for fr in pred_video:
            w.append_data(fr)
        w.close()
        results[key] = {"latent_l2": round(l2, 4)}
        print(f"  short {key}: L2={l2:.4f}, mp4={pred_mp4}")
    return results


def run_long_eval(args, vae, model, device, gt_lat_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    val_dataset_train, val_dataset = _get_dataset(args)
    results = {}
    for sample_idx, ann_file in enumerate(val_dataset.ann_files):
        if sample_idx >= getattr(args, "max_eval_samples", 999):
            break
        with open(ann_file, "rb") as f:
            ann = _json.load(f)
        ann_id = ann_file.split("/")[-1].split(".")[0]
        latent_video_path = os.path.join(args.video_path, ann["latent_videos"][0]["latent_video_path"])
        latent_video = torch.load(latent_video_path)
        total_frame = latent_video.size(0)
        frame_ids = list(range(total_frame))
        arm_states, gripper_states = val_dataset._get_all_robot_states(ann, frame_ids)
        action = val_dataset._get_all_actions(arm_states, gripper_states, args.accumulate_action) * val_dataset.c_act_scaler

        seg_video_list = []
        latent_list = [latent_video[0:1].to(device)]
        current_frame = 0
        start_image = latent_video[0]
        while current_frame + args.num_frames - 1 < total_frame:
            seg_action = action[current_frame:current_frame + args.num_frames - 1]
            si = start_image.unsqueeze(0).unsqueeze(0)
            sa = seg_action.unsqueeze(0)
            videos, latents = run_pipeline(args, si, sa, vae, model, device)
            seg_video = videos.squeeze(0)
            seg_latents = latents.squeeze(0)
            start_image = seg_latents[-1].clone()
            latent_list.append(seg_latents[1:])
            seg_video_list.append(seg_video[1:])
            current_frame += args.num_frames - 1
        seg_action = action[current_frame:]
        true_action = seg_action.size(0)
        if true_action != 0:
            false_action = args.num_frames - true_action - 1
            seg_false_action = repeat(seg_action[0], "d -> f d", f=false_action)
            com_action = torch.cat([seg_action, seg_false_action], dim=0)
            si = start_image.unsqueeze(0).unsqueeze(0)
            sa = com_action.unsqueeze(0)
            videos, latents = run_pipeline(args, si, sa, vae, model, device)
            seg_video = videos.squeeze(0)
            seg_latents = latents.squeeze(0)
            seg_video_list.append(seg_video[1:true_action + 1])
            latent_list.append(seg_latents[1:true_action + 1])

        com_video = torch.cat(seg_video_list, dim=0).cpu()
        com_latent = torch.cat(latent_list, dim=0).cpu()
        # Save mp4 + compute metrics vs GT
        t = ((com_video / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        t = rearrange(t, "f c h w -> f h w c").numpy()
        out_mp4 = os.path.join(out_dir, f"{ann_id}.mp4")
        w = imageio.get_writer(out_mp4, fps=4)
        for fr in t:
            w.append_data(fr)
        w.close()

        gt_lat = torch.load(os.path.join(gt_lat_root, ann_id, "0.pt"), map_location=com_latent.device)[1:]
        # Align lengths
        n = min(com_latent.size(0), gt_lat.size(0))
        l2 = F.mse_loss(com_latent[:n].float(), gt_lat[:n].float()).item()
        # PSNR vs decoded GT mp4
        gt_mp4_dir = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge/videos/test"
        _, ps = process_video_psnr(f"{ann_id}.mp4", out_dir, gt_mp4_dir)
        results[ann_id] = {
            "num_frames": int(t.shape[0]),
            "latent_l2": round(l2, 4),
            "psnr": round(float(ps["PSNR"]), 3),
        }
        print(f"  long {ann_id}: frames={t.shape[0]}, L2={l2:.4f}, PSNR={ps['PSNR']:.3f}")
    return results


def _get_dataset(args):
    return get_dataset(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip", default="", help="op=b1,b2;op2=...")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--mode", choices=["short", "long", "both"], default="short")
    parser.add_argument("--out_root", default="/home/dingqy/Bench/IRASim/results/skip_eval")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Override args.max_eval_samples to evaluate more (or fewer) episodes.")
    parser.add_argument("--shard", type=int, default=0,
                        help="Shard index (0-based) when sharding across GPUs.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards. Samples are split by index mod num_shards.")
    cli = parser.parse_args()

    args = build_args(cli.config)
    if cli.max_samples is not None:
        args.max_eval_samples = cli.max_samples
    device = torch.device("cuda:0")

    # Apply skip schedule (must be done after model construction reads attention_mode,
    # but BEFORE first forward; we set up here and set_skip_blocks again inside eval).
    skip_map = parse_skip(cli.skip)

    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)

    # Force preset & install skip schedule.
    reconfigure(args.attention_mode)
    clear_skip_blocks()
    for op, blocks in skip_map.items():
        set_skip_blocks(op, blocks)
    print(f"variant={args.attention_mode} skip={skip_map}")

    out_dir_short = os.path.join(cli.out_root, f"{cli.tag}_short")
    out_dir_long = os.path.join(cli.out_root, f"{cli.tag}_long")

    payload = {
        "tag": cli.tag,
        "config": cli.config,
        "attention_mode": args.attention_mode,
        "skip": {k: list(v) for k, v in skip_map.items()},
    }

    if cli.mode in ("short", "both"):
        _, val_dataset = get_dataset(args)
        if cli.num_shards > 1:
            val_dataset.samples = [s for i, s in enumerate(val_dataset.samples) if i % cli.num_shards == cli.shard]
            print(f"[shard {cli.shard}/{cli.num_shards}] running {len(val_dataset.samples)} samples")
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        gt_lat_dir = args.true_sample_latent_videos_dir
        print(f"=== short eval [{cli.tag}] ===")
        payload["short"] = run_short_eval(args, val_dataloader, vae, model, device, gt_lat_dir, out_dir_short)

    if cli.mode in ("long", "both"):
        gt_lat_root = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge/latent_videos/test"
        print(f"=== long eval [{cli.tag}] ===")
        payload["long"] = run_long_eval(args, vae, model, device, gt_lat_root, out_dir_long)

    out_json = os.path.join(cli.out_root, f"{cli.tag}.json")
    os.makedirs(cli.out_root, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
