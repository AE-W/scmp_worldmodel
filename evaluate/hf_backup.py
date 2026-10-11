"""Back up experiment progress to a private HuggingFace dataset (anti-local-loss).

Reads the token from a file (never prints it). Uploads results (json/log/figures)
+ my scripts, skipping big regenerable blobs (mp4/latents/checkpoints).

Usage:
  python3 evaluate/hf_backup.py [--once]
"""
import argparse, os, sys
from huggingface_hub import HfApi

# Overridable so the same script works on any machine (PSC, workstation, ...).
TOKEN_FILE = os.environ.get("SCMP_HF_TOKEN_FILE", "/home/qiuyid/huggingface_api_cmu.txt")
ROOT = os.environ.get("SCMP_ROOT", "/home/qiuyid/scmp_worldmodel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_name", default="scmp-worldmodel-progress")
    ap.parse_args()

    token = open(TOKEN_FILE).read().strip()
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/scmp-worldmodel-progress"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"dataset: https://huggingface.co/datasets/{repo_id} (private)")

    # results: json/log/figures/csv — skip regenerable big blobs
    if os.path.isdir(f"{ROOT}/results"):
        api.upload_folder(
            folder_path=f"{ROOT}/results", path_in_repo="results",
            repo_id=repo_id, repo_type="dataset",
            ignore_patterns=["*.mp4", "*.pt", "*.npz", "*.avi", "*.png.tmp"],
        )
    # my scripts (reproducibility)
    api.upload_folder(
        folder_path=f"{ROOT}/evaluate", path_in_repo="evaluate",
        repo_id=repo_id, repo_type="dataset", allow_patterns=["*.py"],
    )
    # eval configs
    if os.path.isdir(f"{ROOT}/configs/evaluation/bridge"):
        api.upload_folder(
            folder_path=f"{ROOT}/configs/evaluation/bridge", path_in_repo="configs_bridge",
            repo_id=repo_id, repo_type="dataset", allow_patterns=["*.yaml"],
        )
    # Calibration artifacts are small but need a GPU to regenerate, and the
    # blanket *.pt ignore above drops them (this is how smoothquant_scales.pt
    # went missing). Upload them explicitly, size-guarded.
    for name in ("smoothquant_scales.pt",):
        f = f"{ROOT}/results/{name}"
        if os.path.isfile(f) and os.path.getsize(f) < 512 * 2**20:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"results/{name}",
                            repo_id=repo_id, repo_type="dataset")
            print(f"uploaded {name} ({os.path.getsize(f)/2**20:.1f} MB)")

    print("backup done")


if __name__ == "__main__":
    main()
