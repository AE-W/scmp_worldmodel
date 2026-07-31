"""Daily video backup to the HF progress dataset.

Kept separate from hf_backup.py (hourly, json/log only) because uploading a few
thousand mp4s in one go burns the 1000-req/5min API quota. This runs once a day
and uploads only what changed, in batches with pauses between them.

Usage:  python3 evaluate/hf_backup_videos.py [--batch 400] [--pause 90]
"""
import argparse, os, time
from huggingface_hub import HfApi

# Overridable so the same script works on any machine (PSC, workstation, ...).
TOKEN_FILE = os.environ.get("SCMP_HF_TOKEN_FILE", "/home/qiuyid/huggingface_api_cmu.txt")
ROOT = os.environ.get("SCMP_RESULTS", "/home/qiuyid/scmp_worldmodel/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=400, help="files per upload call")
    ap.add_argument("--pause", type=int, default=90, help="seconds between batches")
    ap.add_argument("--prune", action="store_true",
                    help="delete local mp4s that are confirmed present on HF "
                         "(keeps disk flat; videos stay retrievable from the dataset)")
    ap.add_argument("--keep-min", type=int, default=30,
                    help="with --prune: keep this many newest mp4s per directory "
                         "(FID/FVD and spot-checks need recent files on disk)")
    cli = ap.parse_args()

    token = open(TOKEN_FILE).read().strip()
    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/scmp-worldmodel-progress"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
    local = []
    for dirpath, _dirs, files in os.walk(ROOT):
        for f in files:
            if not f.endswith(".mp4"):
                continue
            full = os.path.join(dirpath, f)
            rel = "results/" + os.path.relpath(full, ROOT)
            if rel not in remote:
                local.append((full, rel))
    print(f"{len(local)} new videos to upload (remote has {sum(1 for r in remote if r.endswith('.mp4'))})",
          flush=True)
    if not local and not cli.prune:
        return

    for i in range(0, len(local), cli.batch):
        chunk = local[i:i + cli.batch]
        # upload_folder with allow_patterns is the cheapest batched path: one
        # commit per batch instead of one per file.
        api.upload_folder(
            folder_path=ROOT, path_in_repo="results", repo_id=repo_id, repo_type="dataset",
            allow_patterns=[rel[len("results/"):] for _full, rel in chunk],
            commit_message=f"videos batch {i//cli.batch + 1} ({len(chunk)} files)",
        )
        print(f"  batch {i//cli.batch + 1}: {len(chunk)} files", flush=True)
        if i + cli.batch < len(local):
            time.sleep(cli.pause)   # stay under the 1000-req / 5-min quota
    print("VIDEO_BACKUP_DONE", flush=True)

    if cli.prune:
        prune_local(api, repo_id, cli.keep_min)


def prune_local(api, repo_id, keep_min):
    """Delete local mp4s that are verifiably on HF, keeping the newest few.

    Re-lists the remote AFTER uploading so we only ever delete files the
    dataset actually has. Keeps keep_min newest per directory because
    FID/FVD and visual spot-checks read videos off local disk.
    """
    remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
    freed = 0
    for dirpath, _dirs, files in os.walk(ROOT):
        mp4s = [os.path.join(dirpath, f) for f in files if f.endswith(".mp4")]
        if not mp4s:
            continue
        # Never prune a line whose FID/FVD hasn't been computed yet — those
        # metrics need every video of the run present on local disk.
        tag = os.path.basename(os.path.dirname(dirpath))     # .../<tag>/videos
        if not os.path.exists(f"{ROOT}/fidfvd_{tag}.json"):
            continue
        mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for full in mp4s[keep_min:]:                     # keep newest keep_min
            rel = "results/" + os.path.relpath(full, ROOT)
            if rel in remote:                            # confirmed backed up
                sz = os.path.getsize(full)
                os.remove(full)
                freed += sz
    print(f"PRUNED {freed/2**30:.2f} GB of backed-up local videos", flush=True)


if __name__ == "__main__":
    main()
