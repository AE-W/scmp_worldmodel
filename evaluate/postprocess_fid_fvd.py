"""Post-process: compute global FID + FVD for the full SC eval run.

Pulls all generated mp4s from HF dataset, extracts frames, then:
  - FID:  pytorch-fid against bridge evaluation_cache/test_fid_cache.npz
  - FVD:  stylegan-v calc_metrics_for_dataset.py against test_sample_videos

Local working dir is wiped after completion. Final FID/FVD numbers written to
/home/dingqy/Bench/IRASim/results_full_short/global_metrics.json.

Usage:
    HF_TOKEN=... python3 evaluate/postprocess_fid_fvd.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "AE-W/irasim-sc-int8-full-top6-bridge-short"
BRIDGE_ROOT = "/home/dingqy/Bench/IRASim/robotdata/opensource_robotdata/bridge"
GT_VIDEO_DIR = f"{BRIDGE_ROOT}/evaluation_videos/test_sample_videos"
FID_CACHE = f"{BRIDGE_ROOT}/evaluation_cache/test_fid_cache.npz"
FID_SCRIPT = "/home/dingqy/Bench/IRASim/pytorch-fid/src/pytorch_fid/fid_score.py"
FID_SRC = "/home/dingqy/Bench/IRASim/pytorch-fid/src"
FVD_PROJECT = "/home/dingqy/Bench/IRASim/stylegan-v"
FVD_SCRIPT = "src/scripts/calc_metrics_for_dataset.py"

WORK_DIR = "/home/dingqy/Bench/IRASim/results_full_short/postproc"
PRED_MP4_DIR = f"{WORK_DIR}/videos"  # hf_hub_download places mp4s under WORK_DIR/videos/<key>.mp4
PRED_FRAMES_DIR = f"{WORK_DIR}/pred_frames"
FINAL_OUT = "/home/dingqy/Bench/IRASim/results_full_short/global_metrics.json"


def download_all_mp4s(token, limit=None):
    api = HfApi(token=token)
    files = sorted([f for f in api.list_repo_files(REPO_ID, repo_type="dataset")
                    if f.startswith("videos/") and f.endswith(".mp4")])
    if limit is not None:
        files = files[:limit]
    print(f"Downloading {len(files)} mp4s from HF...", flush=True)
    os.makedirs(PRED_MP4_DIR, exist_ok=True)

    def _dl(repo_path):
        # Download as actual file (not symlink) into WORK_DIR/<repo_path>
        local = hf_hub_download(REPO_ID, repo_path, repo_type="dataset", token=token,
                                local_dir=WORK_DIR)
        return local

    with ThreadPoolExecutor(4) as pool:  # 4 workers to stay within HF rate limit
        futs = {pool.submit(_dl, f): f for f in files}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 200 == 0 or done == len(files):
                print(f"  downloaded {done}/{len(files)}", flush=True)
            try:
                fut.result()
            except Exception as e:
                print(f"  download failed for {futs[fut]}: {e}", flush=True)
    return [os.path.join(WORK_DIR, f) for f in files]


def extract_frames(mp4_paths):
    print(f"Extracting frames from {len(mp4_paths)} mp4s...", flush=True)
    os.makedirs(PRED_FRAMES_DIR, exist_ok=True)

    def _extract(mp4_path):
        key = os.path.basename(mp4_path)[:-4]
        out_subdir = os.path.join(PRED_FRAMES_DIR, key)
        os.makedirs(out_subdir, exist_ok=True)
        try:
            reader = imageio.get_reader(mp4_path)
            for i, frame in enumerate(reader):
                imageio.imwrite(os.path.join(out_subdir, f"{i:06d}.png"), frame)
            reader.close()
        except Exception as e:
            return f"err {key}: {e}"
        return None

    with ThreadPoolExecutor(8) as pool:
        futs = {pool.submit(_extract, p): p for p in mp4_paths}
        done = 0
        errs = []
        for fut in as_completed(futs):
            r = fut.result()
            if r: errs.append(r)
            done += 1
            if done % 500 == 0 or done == len(mp4_paths):
                print(f"  frames {done}/{len(mp4_paths)}", flush=True)
        if errs:
            print(f"  {len(errs)} extraction errors (first 3): {errs[:3]}", flush=True)


def run_fid():
    print("Running pytorch-fid...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = FID_SRC
    cmd = [sys.executable, FID_SCRIPT, "--device", "cuda", PRED_FRAMES_DIR, FID_CACHE]
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
    fid_lines = [l for l in r.stdout.splitlines() if "FID" in l]
    if not fid_lines:
        return None
    return float(fid_lines[-1].split("FID:")[-1].strip())


def run_fvd():
    print("Running stylegan-v FVD...", flush=True)
    cmd = (
        f"cd {FVD_PROJECT} && python3 {FVD_SCRIPT} "
        f"--real_data_path {GT_VIDEO_DIR} "
        f"--fake_data_path {PRED_MP4_DIR} "
        f"--mirror 1 --gpus 1 --resolution 256 --metrics fvd2048_16f "
        f"--verbose 0 --use_cache 0"
    )
    print(f"  cmd: {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
    m = re.search(r'\{"results":.*?\}\}', r.stdout)
    if not m:
        m = re.search(r'\{"results":.*?\}', r.stdout)
    if not m:
        return None
    obj = json.loads(m.group())
    return obj.get("results", {}).get("fvd_15f")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of mp4s for testing (default: all)")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--skip_fvd", action="store_true")
    p.add_argument("--cleanup", action="store_true",
                   help="rm -rf working dirs after done")
    cli = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN env var required"

    os.makedirs(WORK_DIR, exist_ok=True)

    t0 = time.time()
    mp4_paths = []
    if not cli.skip_download:
        mp4_paths = download_all_mp4s(token, cli.limit)
        print(f"Download took {time.time()-t0:.1f}s", flush=True)
    else:
        print("Skipping download (using existing mp4s in PRED_MP4_DIR)", flush=True)
        mp4_paths = sorted(os.path.join(PRED_MP4_DIR, f)
                          for f in os.listdir(PRED_MP4_DIR) if f.endswith(".mp4"))

    if not cli.skip_fvd:
        t1 = time.time()
        # need frames extracted only for FID, but FVD reads mp4 directly
    else:
        pass

    if not cli.skip_fid:
        t1 = time.time()
        extract_frames(mp4_paths)
        print(f"Frame extraction took {time.time()-t1:.1f}s", flush=True)
        t2 = time.time()
        fid = run_fid()
        print(f"FID took {time.time()-t2:.1f}s -> FID={fid}", flush=True)
    else:
        fid = None

    if not cli.skip_fvd:
        t3 = time.time()
        fvd = run_fvd()
        print(f"FVD took {time.time()-t3:.1f}s -> FVD={fvd}", flush=True)
    else:
        fvd = None

    out = {
        "n_videos": len(mp4_paths),
        "fid": fid,
        "fvd": fvd,
        "config": "sc_int8_full + top6 skip + DPM-Solver++ 25 steps",
        "fid_cache": FID_CACHE,
        "gt_video_dir": GT_VIDEO_DIR,
    }
    os.makedirs(os.path.dirname(FINAL_OUT), exist_ok=True)
    with open(FINAL_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFinal: {out}", flush=True)
    print(f"Wrote {FINAL_OUT}", flush=True)

    if cli.cleanup:
        print("Cleaning up working dirs...", flush=True)
        shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
