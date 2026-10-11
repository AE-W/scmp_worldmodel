"""Select N diverse test clips from bridge test set (anti-同质化).

Two-stage:
  1) De-dup by episode: one clip per episode (same-episode clips are near-duplicates).
  2) Farthest-point sampling (FPS) on latent features to pick the N most spread-out.

Feature = per-(frame,channel) spatial-mean of the latent, z-normalized.
Output: JSON list of N sample keys (episodeid_cam_start), used by the sensitivity sweep.

Usage:
  PYTHONPATH=. BRIDGE_ROOT=... python3 evaluate/select_diverse_samples.py --n 300 \
      --out results/diverse_300.json
"""
import argparse, json, os
import numpy as np
import torch

BRIDGE_ROOT = os.environ.get(
    "BRIDGE_ROOT", "/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge")
LAT = f"{BRIDGE_ROOT}/evaluation_latent_videos/test_sample_latent_videos"


def parse_key(fn):
    parts = fn[:-3].split("_")
    return "_".join(parts[:-2]), parts[-2], parts[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()

    files = sorted(f for f in os.listdir(LAT) if f.endswith(".pt"))
    by_ep = {}
    for f in files:
        eid, cam, start = parse_key(f)
        by_ep.setdefault(eid, f)
    reps = sorted(by_ep.values())
    print(f"{len(files)} clips -> {len(reps)} unique-episode reps", flush=True)

    if cli.n >= len(reps):
        keys = [f[:-3] for f in reps]
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        json.dump(keys, open(cli.out, "w"), indent=2)
        print(f"n>=reps, took all {len(keys)}"); return

    feats = []
    for i, f in enumerate(reps):
        lat = torch.load(os.path.join(LAT, f), map_location="cpu", weights_only=False)
        feats.append(lat.float().mean(dim=(2, 3)).flatten().numpy())
        if (i + 1) % 300 == 0:
            print(f"  loaded {i+1}/{len(reps)} features", flush=True)
    X = np.stack(feats).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # farthest-point sampling, deterministic start = point nearest the centroid
    start = int(((X - X.mean(0)) ** 2).sum(1).argmin())
    sel = [start]
    d = np.full(len(X), np.inf)
    for _ in range(cli.n - 1):
        d = np.minimum(d, ((X - X[sel[-1]]) ** 2).sum(1))
        sel.append(int(d.argmax()))
    keys = [reps[i][:-3] for i in sel]

    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    json.dump(keys, open(cli.out, "w"), indent=2)
    # diversity report: min pairwise distance among selected (higher = more spread)
    S = X[sel]; dmin = np.inf
    for i in range(len(S) - 1):
        dd = ((S[i + 1:] - S[i]) ** 2).sum(1)
        if len(dd):
            dmin = min(dmin, float(dd.min()) ** 0.5)
    print(f"wrote {cli.out} ({len(keys)} keys); FPS min-pairwise-dist = {dmin:.3f}")


if __name__ == "__main__":
    main()
