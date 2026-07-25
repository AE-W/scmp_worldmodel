"""Streaming FID + FVD for a directory of generated mp4s (no frame dumps).

FID:  frames -> vendored pytorch-fid InceptionV3 (local pt_inception weights)
      -> accumulate mu/sigma -> Frechet vs bridge test_fid_cache.npz.
FVD:  16-frame clip per mp4 -> i3d torchscript (StyleGAN-V) features
      -> mu/sigma vs GT clip features (GT stats cached to disk for reuse).

Everything is streamed; nothing is written except the output json (and the
GT FVD stats cache). Safe on a full disk.

Usage:
  PYTHONPATH=.:pytorch-fid/src CUDA_VISIBLE_DEVICES=N python3 evaluate/compute_fid_fvd_stream.py \
    --pred_dir results/local_n_eval/final_sc_full/videos \
    --out results/fidfvd_final_sc_full.json [--limit 64]
"""
import argparse, json, os
import numpy as np
import torch
import imageio.v2 as imageio
from scipy import linalg

BRIDGE = os.environ.get("BRIDGE_ROOT", "/home/qiuyid/scmp_worldmodel/robotdata/opensource_robotdata/bridge")
GT_DIR = f"{BRIDGE}/evaluation_videos/test_sample_videos"
FID_CACHE = f"{BRIDGE}/evaluation_cache/test_fid_cache.npz"
EM = "/edrive2/qiuyid/robotdata/opensource_robotdata/opensource_robotdata/evaluation_model"
INCEPTION_PTH = f"{EM}/pt_inception-2015-12-05-6726825d.pth"
I3D_PT = f"{EM}/i3d_torchscript.pt"
GT_FVD_STATS = "/edrive2/qiuyid/gt_fvd_stats_bridge.npz"


def frechet(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((s1 + eps * np.eye(s1.shape[0])).dot(s2 + eps * np.eye(s2.shape[0])))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def load_inception(device):
    import pytorch_fid.inception as pi
    import torch.hub
    orig = torch.hub.load_state_dict_from_url
    torch.hub.load_state_dict_from_url = lambda *a, **k: torch.load(INCEPTION_PTH, map_location="cpu", weights_only=False)
    try:
        model = pi.InceptionV3([pi.InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    finally:
        torch.hub.load_state_dict_from_url = orig
    return model


def video_frames(path):
    r = imageio.get_reader(path)
    for fr in r:
        yield fr
    r.close()


def fid_stats_for_dir(mp4s, model, device, bs=64):
    feats = []
    buf = []
    with torch.no_grad():
        for p in mp4s:
            for fr in video_frames(p):
                buf.append(torch.from_numpy(fr.copy()).permute(2, 0, 1).float() / 255.0)
                if len(buf) == bs:
                    x = torch.stack(buf).to(device); buf = []
                    feats.append(model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        if buf:
            x = torch.stack(buf).to(device)
            feats.append(model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    f = np.concatenate(feats, 0)
    return f.mean(0), np.cov(f, rowvar=False), len(f)


def fvd_feats_for_dir(mp4s, i3d, device, bs=16, n_frames=16):
    feats, buf, skipped = [], [], 0
    # Official StyleGAN-V FVD: rescale=True means i3d normalizes internally from
    # raw [0,255] pixels — so we must feed uint8-range floats, NOT pre-scaled.
    kw = dict(rescale=True, resize=True, return_features=True)
    with torch.no_grad():
        for p in mp4s:
            try:
                frs = [torch.from_numpy(fr.copy()).permute(2, 0, 1) for fr in video_frames(p)]
            except Exception as e:
                print(f"  [fvd] skip unreadable {os.path.basename(p)}: {e}", flush=True)
                skipped += 1; continue
            if len(frs) < n_frames:   # uniform clip length is required to batch
                skipped += 1; continue
            v = torch.stack(frs[:n_frames], 1).float()  # (C,16,H,W) in [0,255]
            buf.append(v)
            if len(buf) == bs:
                feats.append(i3d(torch.stack(buf).to(device), **kw).cpu().numpy()); buf = []
        if buf:
            feats.append(i3d(torch.stack(buf).to(device), **kw).cpu().numpy())
    if skipped:
        print(f"  [fvd] skipped {skipped}/{len(mp4s)} videos (short/unreadable)", flush=True)
    f = np.concatenate(feats, 0)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fid_precomputed", type=float, default=None,
                    help="reuse an already-computed FID (skip the inception pass)")
    cli = ap.parse_args()
    device = torch.device("cuda:0")

    preds = sorted(f"{cli.pred_dir}/{f}" for f in os.listdir(cli.pred_dir) if f.endswith(".mp4"))
    if cli.limit:
        preds = preds[: cli.limit]
    keys = [os.path.basename(p) for p in preds]
    gts = [f"{GT_DIR}/{k}" for k in keys if os.path.exists(f"{GT_DIR}/{k}")]
    print(f"pred={len(preds)} gt_matched={len(gts)}", flush=True)

    out = {"pred_dir": cli.pred_dir, "n_videos": len(preds)}

    # ---- FID ----
    if cli.fid_precomputed is not None:
        out["fid"] = cli.fid_precomputed
        print(f"FID = {out['fid']} (precomputed, reused)", flush=True)
    else:
        inc = load_inception(device)
        mu_p, s_p, nf = fid_stats_for_dir(preds, inc, device)
        cache = np.load(FID_CACHE)
        mu_r, s_r = cache["mu"], cache["sigma"]
        out["fid"] = round(frechet(mu_p, s_p, mu_r, s_r), 3)
        out["fid_frames"] = nf
        print(f"FID = {out['fid']} ({nf} frames)", flush=True)
        del inc; torch.cuda.empty_cache()
    json.dump(out, open(cli.out, "w"), indent=2)  # incremental save (FID survives an FVD crash)

    # ---- FVD ----
    # feature extraction is chunked so a full 2946-video run never holds more
    # than CHUNK videos' activations at once (the un-chunked version OOM'd).
    CHUNK = 300
    i3d = torch.jit.load(I3D_PT).to(device).eval()

    def stats_chunked(paths, label):
        parts = []
        for i in range(0, len(paths), CHUNK):
            f = fvd_feats_for_dir(paths[i:i + CHUNK], i3d, device)
            parts.append(f)
            print(f"  [{label}] {min(i+CHUNK, len(paths))}/{len(paths)}", flush=True)
        allf = np.concatenate(parts, 0)
        return allf.mean(0), np.cov(allf, rowvar=False)

    if os.path.exists(GT_FVD_STATS) and cli.limit is None:
        g = np.load(GT_FVD_STATS); mu_g, s_g = g["mu"], g["sigma"]
        print("GT FVD stats: cached", flush=True)
    else:
        mu_g, s_g = stats_chunked(gts, "gt")
        if cli.limit is None:
            np.savez(GT_FVD_STATS, mu=mu_g, sigma=s_g)
    mu_p2, s_p2 = stats_chunked(preds, "pred")
    out["fvd"] = round(frechet(mu_p2, s_p2, mu_g, s_g), 3)
    print(f"FVD = {out['fvd']}", flush=True)

    json.dump(out, open(cli.out, "w"), indent=2)
    print("saved", cli.out, flush=True)


if __name__ == "__main__":
    main()
