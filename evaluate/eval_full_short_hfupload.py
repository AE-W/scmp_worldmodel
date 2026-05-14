"""Full Bridge short eval with sc_int8_full + top6 skip + 25 inference steps.

Iterates over the per-sample latents in:
  bridge/evaluation_latent_videos/test_sample_latent_videos/<ep>_<cam>_<start>.pt

For each sample:
  - load 16-frame GT latent
  - parse (episode_id, cam, start_frame) from filename
  - open bridge/annotation/test/<episode_id>.json -> compute 15 actions
  - generate predicted video via IRASim
  - compute latent_l2 + PSNR + SSIM vs GT mp4
  - upload mp4 + latent + metric to HF, delete locals

Resumable per shard (progress/shard_<id>.json).

Usage:
    HF_TOKEN=... PYTHONPATH=. CUDA_VISIBLE_DEVICES=N \\
      python3 evaluate/eval_full_short_hfupload.py --shard <i> --num_shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import imageio
from einops import rearrange
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DPMSolverMultistepScheduler

from dataset.dataset_util import euler2rotm, rotm2euler
from models import get_models
from models.sc_integration import reconfigure, set_skip_blocks, clear_skip_blocks
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from evaluate.compute_psnr_ssim import process_video_psnr_ssim
from util import update_paths

from huggingface_hub import HfApi, CommitOperationAdd

# ---- config (constant) ----
REPO_ID = "AE-W/irasim-sc-int8-full-top6-bridge-short"
CONFIG = "configs/evaluation/bridge/frame_ada_sc_full.yaml"
SKIP = {
    "mlp_fc1": [25, 27, 4, 26, 13, 16],
    "mlp_fc2": [27, 0, 25, 4, 26, 2],
    "qkv":     [25, 7, 8, 6, 14, 26],
}
NUM_INFER_STEPS = 25  # DPM-Solver++ at 25 steps matches PNDM 50-step quality (PSNR=16.75 vs 16.51)

BRIDGE_ROOT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge"
GT_LATENT_DIR = f"{BRIDGE_ROOT}/evaluation_latent_videos/test_sample_latent_videos"
GT_VIDEO_DIR = f"{BRIDGE_ROOT}/evaluation_videos/test_sample_videos"
ANNOT_DIR = f"{BRIDGE_ROOT}/annotation/test"
LOCAL_OUT = "/home/dingqy/Bench/IRASim/results_full_short"

C_ACT_SCALER = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0], dtype=float)
SEQUENCE_LENGTH = 16
ACTION_DIM = 7


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
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=args.beta_start, beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        algorithm_type="dpmsolver++",
        solver_order=2,
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
    """Mirror Dataset_3D._get_all_actions logic: ee-frame relative pose deltas."""
    # arm_states: (N, 6), gripper_states: (N,)
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
    return action  # (N-1, 7)


def parse_key(filename):
    """0_0_16.pt -> ('0', '0', 16)."""
    base = os.path.basename(filename)[:-3]  # strip .pt
    parts = base.split("_")
    # episode_id may contain underscores but in Bridge it's just an int. cam is single int. start is single int.
    eid = "_".join(parts[:-2])
    cam = int(parts[-2])
    start = int(parts[-1])
    return eid, cam, start


def list_samples():
    files = sorted(os.listdir(GT_LATENT_DIR))
    files = [f for f in files if f.endswith(".pt")]
    return files


def progress_path(shard_id):
    return os.path.join(LOCAL_OUT, "progress", f"shard_{shard_id}.json")


def load_progress(shard_id):
    p = progress_path(shard_id)
    if not os.path.exists(p):
        return set()
    with open(p) as f:
        return set(json.load(f).get("done", []))


def save_progress(shard_id, done_set):
    p = progress_path(shard_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"done": sorted(list(done_set))}, f)


def hf_create_commit(api, hf_token, files_with_paths, message):
    operations = [
        CommitOperationAdd(path_in_repo=p, path_or_fileobj=local) for local, p in files_with_paths
    ]
    if not operations:
        return
    for attempt in range(4):
        try:
            api.create_commit(
                repo_id=REPO_ID, repo_type="dataset",
                operations=operations, commit_message=message, token=hf_token,
            )
            return
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  hf commit retry {attempt+1}/4: {e}; waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"hf commit failed after 4 attempts ({message})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num_shards", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--max_samples", type=int, default=None, help="cap shard size for testing")
    cli = p.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "HF_TOKEN env var required"

    args = build_args()
    device = torch.device("cuda:0")

    print(f"[shard {cli.shard}/{cli.num_shards}] loading model...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)
    reconfigure(args.attention_mode)
    clear_skip_blocks()
    for op, blocks in SKIP.items():
        set_skip_blocks(op, blocks)
    pipe = make_pipe(args, vae, model)
    print(f"[shard {cli.shard}] mode={args.attention_mode} skip={SKIP} steps={args.infer_num_sampling_steps}", flush=True)

    api = HfApi(token=hf_token)

    all_files = list_samples()
    shard_files = all_files[cli.shard::cli.num_shards]
    if cli.max_samples is not None:
        shard_files = shard_files[: cli.max_samples]
    done_keys = load_progress(cli.shard)
    print(f"[shard {cli.shard}] all={len(all_files)} this shard={len(shard_files)} done={len(done_keys)}", flush=True)

    os.makedirs(os.path.join(LOCAL_OUT, "videos"), exist_ok=True)
    os.makedirs(os.path.join(LOCAL_OUT, "latents"), exist_ok=True)
    os.makedirs(os.path.join(LOCAL_OUT, "metrics"), exist_ok=True)

    batch_files = []
    batch_keys = []
    counter = 0
    skipped_no_annot = 0
    t_start = time.time()

    for fn in shard_files:
        eid, cam, start = parse_key(fn)
        key = f"{eid}_{cam}_{start}"
        if key in done_keys:
            continue

        ann_path = os.path.join(ANNOT_DIR, f"{eid}.json")
        if not os.path.exists(ann_path):
            skipped_no_annot += 1
            continue

        try:
            # load gt latent
            gt_lat_path = os.path.join(GT_LATENT_DIR, fn)
            gt_lat = torch.load(gt_lat_path, weights_only=False, map_location=device)  # (16, 4, 32, 40)

            # load annotation, slice arm/gripper states for this 16-frame window
            with open(ann_path) as f:
                ann = json.load(f)
            n_total = len(ann["state"])
            if start + SEQUENCE_LENGTH > n_total:
                skipped_no_annot += 1  # stale sample, skip
                continue
            arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
            grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
            actions_np = compute_actions_for_slice(arm, grip) * C_ACT_SCALER  # (15, 7)
            actions = torch.from_numpy(actions_np).float().unsqueeze(0)

            mask_x = gt_lat[0:1].unsqueeze(0)  # (1, 1, 4, 32, 40)
            t0 = time.time()
            videos, pred_lat = gen_one(pipe, mask_x, actions, args, device)
            dt = time.time() - t0
            pred_lat = pred_lat.squeeze(0)

            l2 = F.mse_loss(pred_lat[1:], gt_lat[1:]).item()

            mp4_path = os.path.join(LOCAL_OUT, "videos", f"{key}.mp4")
            write_mp4(videos.squeeze(0), mp4_path)

            _, ps = process_video_psnr_ssim(f"{key}.mp4", os.path.join(LOCAL_OUT, "videos"), GT_VIDEO_DIR)

            lat_path = os.path.join(LOCAL_OUT, "latents", f"{key}.pt")
            torch.save(pred_lat.cpu(), lat_path)

            metric = {
                "key": key, "latent_l2": round(l2, 4),
                "psnr": round(float(ps["PSNR"]), 3),
                "ssim": round(float(ps["SSIM"]), 3),
                "infer_seconds": round(dt, 2),
            }
            met_path = os.path.join(LOCAL_OUT, "metrics", f"{key}.json")
            with open(met_path, "w") as f:
                json.dump(metric, f)

            batch_files.append((mp4_path, f"videos/{key}.mp4"))
            batch_files.append((lat_path, f"latents/{key}.pt"))
            batch_files.append((met_path, f"metrics/{key}.json"))
            batch_keys.append(key)
            counter += 1

            elapsed = time.time() - t_start
            rate = counter / elapsed if elapsed > 0 else 0
            remaining = (len(shard_files) - len(done_keys) - counter) / rate / 3600 if rate > 0 else 0
            print(f"[shard {cli.shard}] {key}  L2={l2:.4f} PSNR={ps['PSNR']:.2f} SSIM={ps['SSIM']:.3f}  "
                  f"({counter} done, {dt:.1f}s; eta {remaining:.1f}h)", flush=True)

            if counter % cli.batch_size == 0:
                hf_create_commit(api, hf_token, batch_files,
                                 f"shard {cli.shard}: batch {batch_keys[0]}..{batch_keys[-1]} ({len(batch_keys)} samples)")
                for local, _ in batch_files:
                    try: os.remove(local)
                    except FileNotFoundError: pass
                done_keys.update(batch_keys)
                save_progress(cli.shard, done_keys)
                hf_create_commit(api, hf_token, [(progress_path(cli.shard), f"progress/shard_{cli.shard}.json")],
                                 f"progress shard {cli.shard}: {len(done_keys)} done")
                batch_files, batch_keys = [], []
        except Exception:
            print(f"[shard {cli.shard}] sample {key} failed:", flush=True)
            traceback.print_exc()
            continue

    if batch_files:
        hf_create_commit(api, hf_token, batch_files,
                         f"shard {cli.shard}: final batch ({len(batch_keys)} samples)")
        for local, _ in batch_files:
            try: os.remove(local)
            except FileNotFoundError: pass
        done_keys.update(batch_keys)
        save_progress(cli.shard, done_keys)
        hf_create_commit(api, hf_token, [(progress_path(cli.shard), f"progress/shard_{cli.shard}.json")],
                         f"progress shard {cli.shard}: FINAL {len(done_keys)} done")

    print(f"[shard {cli.shard}] DONE. {len(done_keys)} committed; {skipped_no_annot} skipped (no annot/oob).", flush=True)


if __name__ == "__main__":
    main()
