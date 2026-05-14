import os
import sys
import json
import cv2
from evaluate.compute_latent_l2 import process_video as process_latent_l2
from evaluate.compute_psnr_ssim import process_video_psnr

VARIANT = sys.argv[1]  # e.g. "sc_int8", "sc_int8_full", "sc_int8_qk_av", "sc_int8_qk_av_proj", "sc_int8_qk_av_proj_fc1"
RUN_DIR = f"test_bridge_frame_ada_{VARIANT}-debug" if VARIANT else "full_test_bridge_frame_ada-debug"
PRED_ROOT = f"/home/dingqy/Bench/IRASim/results/04/24/{RUN_DIR}/checkpoints/0300000"
PRED_LAT = f"{PRED_ROOT}/test_episode_latent_videos"
PRED_VID = f"{PRED_ROOT}/test_episode_videos"
GT_LAT_ROOT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge/latent_videos/test"
GT_VID_ROOT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge/videos/test"

results = {}
for fname in sorted(os.listdir(PRED_LAT)):
    eid = fname.replace(".pt", "")
    l2 = process_latent_l2(fname, PRED_LAT, GT_LAT_ROOT)
    mp4 = eid + ".mp4"
    _, ps = process_video_psnr(mp4, PRED_VID, GT_VID_ROOT)
    cap = cv2.VideoCapture(os.path.join(PRED_VID, mp4))
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    results[eid] = {
        "episode_id": int(eid),
        "num_frames": nf,
        "duration_s_at_4fps": nf / 4,
        "latent_l2": round(l2, 4),
        "psnr": round(float(ps["PSNR"]), 3),
    }

payload = {
    "model": "frame_ada",
    "dataset": "bridge",
    "variant": VARIANT,
    "checkpoint": "bridge/checkpoints/frame_ada/0300000.pt",
    "setting": "long_trajectory",
    "autoregressive_segment_length": 16,
    "per_video": results,
}
out_json = os.path.join(PRED_ROOT, "metrics.json")
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)
print(f"[{VARIANT}] wrote {out_json}")
print(json.dumps(results, indent=2))
