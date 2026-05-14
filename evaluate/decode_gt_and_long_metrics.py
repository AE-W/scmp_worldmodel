import os
import json
import torch
import imageio
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from einops import rearrange
from evaluate.compute_latent_l2 import process_video as process_latent_l2
from evaluate.compute_psnr_ssim import process_video_psnr

DATA_ROOT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge"
GT_LAT_ROOT = f"{DATA_ROOT}/latent_videos/test"
GT_VID_ROOT = f"{DATA_ROOT}/videos/test"
PRED_ROOT = "/home/dingqy/Bench/IRASim/results/04/24/full_test_bridge_frame_ada-debug/checkpoints/0300000"
PRED_LAT = f"{PRED_ROOT}/test_episode_latent_videos"
PRED_VID = f"{PRED_ROOT}/test_episode_videos"
VAE_PATH = "/home/dingqy/Bench/IRASim/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0"

device = torch.device("cuda:0")
vae = AutoencoderKL.from_pretrained(VAE_PATH, subfolder="vae").to(device)
vae.eval()
sf = vae.config.scaling_factor
print(f"VAE scaling_factor={sf}")

for eid_dir in sorted(os.listdir(GT_LAT_ROOT)):
    lat = torch.load(os.path.join(GT_LAT_ROOT, eid_dir, "0.pt"), map_location=device)
    with torch.no_grad():
        dec = vae.decode(lat / sf).sample
    frames = ((dec / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu()
    frames = rearrange(frames, "f c h w -> f h w c").numpy()
    gt_dir = os.path.join(GT_VID_ROOT, eid_dir)
    os.makedirs(gt_dir, exist_ok=True)
    out = os.path.join(gt_dir, "rgb.mp4")
    writer = imageio.get_writer(out, fps=4)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print(f"GT mp4: {out}  frames={frames.shape[0]}")

results = {}
for fname in sorted(os.listdir(PRED_LAT)):
    eid = fname.replace(".pt", "")

    l2 = process_latent_l2(fname, PRED_LAT, GT_LAT_ROOT)
    mp4 = eid + ".mp4"
    _, ps = process_video_psnr(mp4, PRED_VID, GT_VID_ROOT)
    results[eid] = {
        "num_frames_pred": None,
        "latent_l2": round(l2, 4),
        "psnr": round(float(ps["PSNR"]), 3),
    }

for eid in results:
    import cv2
    cap = cv2.VideoCapture(os.path.join(PRED_VID, eid + ".mp4"))
    results[eid]["num_frames_pred"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

print(json.dumps(results, indent=2))

out_json = os.path.join(PRED_ROOT, "metrics.json")
payload = {
    "model": "frame_ada",
    "dataset": "bridge",
    "checkpoint": "bridge/checkpoints/frame_ada/0300000.pt",
    "eval_date": "2026-04-24",
    "setting": "long_trajectory",
    "autoregressive_segment_length": 16,
    "per_video": {
        eid: {
            "episode_id": int(eid),
            "num_frames": results[eid]["num_frames_pred"],
            "duration_s_at_4fps": results[eid]["num_frames_pred"] / 4,
            "latent_l2": results[eid]["latent_l2"],
            "psnr": results[eid]["psnr"],
        }
        for eid in results
    },
    "sources": {
        "latent_l2_gt": "latent_videos/test/{eid}/0.pt (full episode)",
        "psnr_gt": "videos/test/{eid}/rgb.mp4 (VAE-decoded full latent; not original mp4)",
    },
    "caveats": [
        "PSNR uses VAE-decoded GT latent as GT video instead of the original mp4; values are slightly higher than the paper definition.",
        "3 test episodes only — not statistically comparable to paper results.",
    ],
}
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)
print(f"wrote {out_json}")
