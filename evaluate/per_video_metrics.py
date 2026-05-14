import os
import subprocess
import torch
import torch.nn.functional as F
from evaluate.compute_psnr_ssim import process_video_psnr_ssim

PRED_LAT = "/home/dingqy/Bench/IRASim/results/04/24/test_bridge_frame_ada-debug/checkpoints/0300000/test_sample_latents"
GT_LAT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge/evaluation_latent_videos/test_sample_latent_videos"
PRED_VID = "/home/dingqy/Bench/IRASim/results/04/24/test_bridge_frame_ada-debug/checkpoints/0300000/test_sample_videos"
GT_VID = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge/evaluation_videos/test_sample_videos"
FRAMES_DIR = "/home/dingqy/Bench/IRASim/results/04/24/test_bridge_frame_ada-debug/checkpoints/0300000/test_sample_frames"
FID_CACHE = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge/evaluation_cache/test_fid_cache.npz"
FID_SCRIPT = "/home/dingqy/Bench/IRASim/pytorch-fid/src/pytorch_fid/fid_score.py"
FID_SRC = "/home/dingqy/Bench/IRASim/pytorch-fid/src"
FLAT_TMP = "/tmp/fid_flat"

files = sorted(f for f in os.listdir(PRED_LAT) if f.endswith(".pt"))
results = {}

for f in files:
    key = f.replace(".pt", "")
    pl = torch.load(os.path.join(PRED_LAT, f), map_location="cpu")
    tl = torch.load(os.path.join(GT_LAT, f), map_location="cpu")
    l2 = F.mse_loss(pl[1:], tl[1:]).item()

    mp4 = f.replace(".pt", ".mp4")
    _, ps = process_video_psnr_ssim(mp4, PRED_VID, GT_VID)

    sub_frames = os.path.join(FRAMES_DIR, key)
    flat = os.path.join(FLAT_TMP, key)
    os.makedirs(flat, exist_ok=True)
    for old in os.listdir(flat):
        os.remove(os.path.join(flat, old))
    for png in os.listdir(sub_frames):
        os.symlink(os.path.join(sub_frames, png), os.path.join(flat, png))

    env = os.environ.copy()
    env["PYTHONPATH"] = FID_SRC
    proc = subprocess.run(
        ["/home/dingqy/miniconda3/envs/irasim/bin/python3", FID_SCRIPT, "--device", "cuda", flat, FID_CACHE],
        capture_output=True, text=True, env=env,
    )
    fid_line = [l for l in proc.stdout.splitlines() if "FID" in l]
    fid_val = float(fid_line[-1].split("FID:")[-1].strip()) if fid_line else None

    results[key] = {
        "latent_l2": round(l2, 4),
        "psnr": round(ps["PSNR"], 3),
        "ssim": round(ps["SSIM"], 3),
        "fid": round(fid_val, 2) if fid_val is not None else None,
    }
    print(f"{key}: {results[key]}")

import json
print("\n" + json.dumps(results, indent=2))
with open("/tmp/per_video_metrics.json", "w") as out:
    json.dump(results, out, indent=2)
