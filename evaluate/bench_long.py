"""Wall-clock benchmark for long-video autoregressive generation.

Times only the diffusion segments (warmup excluded). Compares fp32 baseline vs
SC + skip schedule on the 3 bridge test episodes (0, 1000, 1001).

Usage:
    PYTHONPATH=. python3 evaluate/bench_long.py \
        --config configs/evaluation/bridge/frame_ada.yaml \
        --tag fp32

    PYTHONPATH=. python3 evaluate/bench_long.py \
        --config configs/evaluation/bridge/frame_ada_sc_qk_av_proj_fc1.yaml \
        --skip 'mlp_fc1=25,27' --tag sc_qk_av_proj_fc1_skip2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from einops import repeat
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler

from dataset import get_dataset
from models import get_models
from models.sc_integration import reconfigure, set_skip_blocks, clear_skip_blocks
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from util import update_paths
import json as _json


def build_args(config_path: str):
    args = OmegaConf.merge(
        OmegaConf.load("configs/base/data.yaml"),
        OmegaConf.load("configs/base/diffusion.yaml"),
        OmegaConf.load(config_path),
    )
    update_paths(args)
    args.latent_size = [t // 8 for t in args.video_size]
    return args


def parse_skip(skip_str):
    if not skip_str:
        return {}
    out = {}
    for clause in skip_str.split(";"):
        if not clause.strip():
            continue
        op, blocks = clause.split("=")
        out[op.strip()] = [int(x) for x in blocks.split(",") if x.strip()]
    return out


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


def run_one_segment(pipe, mask_x, actions, args, device):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
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
    torch.cuda.synchronize()
    return time.perf_counter() - t0, videos, latents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--skip", default="", help="op=b1,b2;...")
    p.add_argument("--tag", required=True)
    p.add_argument("--out_root", default="/home/dingqy/Bench/IRASim/results/bench_long")
    p.add_argument("--episodes", default="0,1000,1001")
    cli = p.parse_args()

    args = build_args(cli.config)
    device = torch.device("cuda:0")

    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)
    print(f"[{cli.tag}] model+vae loaded; attention_mode={args.attention_mode}")

    if args.attention_mode != "math":
        reconfigure(args.attention_mode)
        clear_skip_blocks()
        skip_map = parse_skip(cli.skip)
        for op, blocks in skip_map.items():
            set_skip_blocks(op, blocks)
        print(f"[{cli.tag}] SC config={args.attention_mode} skip={skip_map}")
    else:
        skip_map = {}
        print(f"[{cli.tag}] FP32 baseline (no SC)")

    scheduler = PNDMScheduler.from_pretrained(
        args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end,
        beta_schedule=args.beta_schedule, variance_type=args.variance_type,
    )
    pipe = Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)

    _, val_dataset = get_dataset(args)
    ann_lookup = {ann_file.split("/")[-1].split(".")[0]: ann_file for ann_file in val_dataset.ann_files}
    eps = [e.strip() for e in cli.episodes.split(",") if e.strip()]

    # ------- Warmup (1 segment, not counted) -------
    print(f"[{cli.tag}] warmup ...")
    warm_ann = ann_lookup[eps[0]]
    with open(warm_ann, "rb") as f:
        ann = _json.load(f)
    latent_video_path = os.path.join(args.video_path, ann["latent_videos"][0]["latent_video_path"])
    latent_video = torch.load(latent_video_path)
    arm, grip = val_dataset._get_all_robot_states(ann, list(range(latent_video.size(0))))
    action = val_dataset._get_all_actions(arm, grip, args.accumulate_action) * val_dataset.c_act_scaler
    if not torch.is_tensor(action):
        action = torch.from_numpy(action).float()
    else:
        action = action.float()
    si = latent_video[0].unsqueeze(0).unsqueeze(0)
    sa = action[: args.num_frames - 1].unsqueeze(0)
    _ = run_one_segment(pipe, si, sa, args, device)

    # ------- Benchmark -------
    results = {}
    for ep_id in eps:
        ann_file = ann_lookup[ep_id]
        with open(ann_file, "rb") as f:
            ann = _json.load(f)
        latent_video = torch.load(os.path.join(args.video_path, ann["latent_videos"][0]["latent_video_path"]))
        total_frame = latent_video.size(0)
        arm, grip = val_dataset._get_all_robot_states(ann, list(range(total_frame)))
        action = val_dataset._get_all_actions(arm, grip, args.accumulate_action) * val_dataset.c_act_scaler
        if not torch.is_tensor(action):
            action = torch.from_numpy(action).float()
        else:
            action = action.float()

        seg_times = []
        cur = 0
        start_image = latent_video[0]
        while cur + args.num_frames - 1 < total_frame:
            seg_action = action[cur:cur + args.num_frames - 1]
            si = start_image.unsqueeze(0).unsqueeze(0)
            sa = seg_action.unsqueeze(0)
            dt, _, latents = run_one_segment(pipe, si, sa, args, device)
            seg_times.append(dt)
            start_image = latents.squeeze(0)[-1].clone()
            cur += args.num_frames - 1
        # tail (mirrors eval_with_skip.py logic)
        sa_tail = action[cur:]
        true_tail = sa_tail.size(0)
        if true_tail != 0:
            pad_n = args.num_frames - 1 - true_tail
            if pad_n > 0:
                pad = repeat(sa_tail[0], "d -> f d", f=pad_n)
                full_action = torch.cat([sa_tail, pad], dim=0)
            else:
                full_action = sa_tail[: args.num_frames - 1]
            si = start_image.unsqueeze(0).unsqueeze(0)
            sa = full_action.unsqueeze(0)
            dt, _, _ = run_one_segment(pipe, si, sa, args, device)
            seg_times.append(dt)

        results[ep_id] = {
            "num_frames": int(total_frame),
            "num_segments": len(seg_times),
            "seg_times_s": [round(t, 3) for t in seg_times],
            "total_s": round(sum(seg_times), 3),
            "mean_seg_s": round(sum(seg_times) / len(seg_times), 3),
        }
        print(f"[{cli.tag}] ep{ep_id}: frames={total_frame}, segs={len(seg_times)}, "
              f"per_seg={[round(t,2) for t in seg_times]}s, total={results[ep_id]['total_s']}s")

    # aggregate
    total = sum(r["total_s"] for r in results.values())
    nseg = sum(r["num_segments"] for r in results.values())
    payload = {
        "tag": cli.tag,
        "config": cli.config,
        "attention_mode": args.attention_mode,
        "skip": skip_map,
        "infer_num_sampling_steps": args.infer_num_sampling_steps,
        "per_episode": results,
        "summary": {
            "total_video_time_s": round(total, 3),
            "total_segments": nseg,
            "mean_seg_s": round(total / nseg, 3),
        },
    }
    print(f"\n[{cli.tag}] SUMMARY: total={total:.2f}s across {nseg} segments → mean {total/nseg:.2f}s/segment")

    os.makedirs(cli.out_root, exist_ok=True)
    out_json = os.path.join(cli.out_root, f"{cli.tag}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
