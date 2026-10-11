"""Validation gate with uniform fallback.

Uniform is a feasible point of the MP search space, so a correctly selected
schedule can never deploy worse than uniform — enforce that by construction:
every candidate is compared PAIRED against uniform on the same keys, on the
TRUE metrics, and the deployment is the argmax with uniform as the floor.

    python evaluate/select_mp_or_uniform.py --uniform scr10_UNI96 \
        --candidates scr10_sc_avg192,scr10w_sc_avg192_sens --margin 0.0
"""
import argparse, glob, json, os

P = "results/local_n_eval"


def load(tag):
    return {os.path.basename(f)[:-5]: json.load(open(f))
            for f in glob.glob(f"{P}/{tag}/metrics/*.json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", required=True)
    ap.add_argument("--candidates", required=True, help="comma-separated tags")
    ap.add_argument("--margin", type=float, default=0.0, help="required paired dPSNR")
    ap.add_argument("--ssim_guard", type=float, default=0.005)
    ap.add_argument("--l2_guard", type=float, default=0.005)
    a = ap.parse_args()

    U = load(a.uniform)
    best, rows = None, []
    for tag in a.candidates.split(","):
        M = load(tag)
        ks = sorted(set(M) & set(U))
        if not ks:
            rows.append((tag, None)); continue
        dp = sum(M[k]["psnr"] - U[k]["psnr"] for k in ks) / len(ks)
        ds = sum(M[k]["ssim"] - U[k]["ssim"] for k in ks) / len(ks)
        dl = sum(M[k]["latent_l2"] - U[k]["latent_l2"] for k in ks) / len(ks)
        ok = dp > a.margin and ds >= -a.ssim_guard and dl <= a.l2_guard
        rows.append((tag, (len(ks), dp, ds, dl, ok)))
        if ok and (best is None or dp > best[1]):
            best = (tag, dp)

    print(f"{'candidate':34} {'n':>3} {'dPSNR':>8} {'dSSIM':>8} {'dL2':>8}  gate")
    for tag, r in rows:
        if r is None:
            print(f"{tag:34}   no paired data"); continue
        n, dp, ds, dl, ok = r
        print(f"{tag:34} {n:3} {dp:+8.3f} {ds:+8.4f} {dl:+8.4f}  {'PASS' if ok else 'fail'}")
    print(f"\nDEPLOY: {best[0] if best else a.uniform + '  (uniform fallback — degenerate solution)'}")


if __name__ == "__main__":
    main()
