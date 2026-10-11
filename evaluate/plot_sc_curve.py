"""SC progressive-replacement line chart (facet by metric).

x-axis = how many op-types run on SC (FP → +qk → +av → +proj → +fc1 → full).
One subplot per metric (latent L2 / PSNR / SSIM). Overlaid references:
    - naive int8 uniform  (horizontal dashed line): int8 but NOT stochastic
    - SC full + skip 0,4,18 (star): the sensitivity-anchor recovery point

All points share one self-consistent setting: DPM 10-step, n=8.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "results/line_eval"
C_SEQ, C_NAIVE, C_SKIP = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, MUTED, GRID = "#0f1319", "#555b66", "#868b95", "#e6e9ee"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 200, "savefig.dpi": 200,
})

SEQ = [("line_v0_fp", "FP\n(math)"), ("line_v1_qk", "+qk"), ("line_v2_qkav", "+av"),
       ("line_v3_qkavproj", "+proj"), ("line_v4_fc1", "+fc1"), ("line_v5_full", "full\n(6 ops)")]


def load(tag):
    with open(f"{ROOT}/{tag}/summary_shard_0.json") as f:
        return json.load(f)


data = [load(t) for t, _ in SEQ]
naive = load("line_naive")
skip = load("line_skip")
xs = list(range(len(SEQ)))
xlabels = [l for _, l in SEQ]

METRICS = [
    ("mean_l2",   "latent L2   (↓ lower better)",  "{:.3f}"),
    ("mean_psnr", "PSNR / dB   (↑ higher better)", "{:.2f}"),
    ("mean_ssim", "SSIM   (↑ higher better)",      "{:.3f}"),
]


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1)


fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1))
for ax, (key, title, fmt) in zip(axes, METRICS):
    ys = [d[key] for d in data]
    # SC progressive main line
    ax.plot(xs, ys, "-o", color=C_SEQ, lw=2.2, ms=7, zorder=4,
            markerfacecolor=C_SEQ, markeredgecolor="white", markeredgewidth=1.2)
    # value labels on FP (start) and full (end)
    ax.annotate(fmt.format(ys[0]), (xs[0], ys[0]), textcoords="offset points",
                xytext=(2, 9), fontsize=9, color=INK, fontweight="700")
    ax.annotate(fmt.format(ys[-1]), (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(-4, -15), fontsize=9, color=INK, fontweight="700", ha="center")
    # naive int8 reference (horizontal)
    ax.axhline(naive[key], ls="--", color=C_NAIVE, lw=1.8, zorder=2)
    # skip anchor point (same x as full, better y), with recovery arrow
    ax.plot([xs[-1]], [skip[key]], marker="*", ms=17, color=C_SKIP,
            markeredgecolor="white", markeredgewidth=1, ls="", zorder=5)
    ax.annotate("", xy=(xs[-1], skip[key]), xytext=(xs[-1], ys[-1]),
                arrowprops=dict(arrowstyle="->", color=C_SKIP, lw=1.6))

    ax.set_title(title, fontsize=11.5, color=INK, pad=9, loc="left")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlim(-0.35, len(SEQ) - 0.4)
    _clean(ax)
    # headroom for labels
    lo = min(min(ys), naive[key], skip[key])
    hi = max(max(ys), naive[key], skip[key])
    pad = (hi - lo) * 0.16 + 1e-6
    ax.set_ylim(lo - pad, hi + pad)

# shared legend (proxy handles)
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=C_SEQ, lw=2.2, marker="o", markerfacecolor=C_SEQ,
           markeredgecolor="white", label="SC progressive (FP → full)"),
    Line2D([0], [0], color=C_NAIVE, lw=1.8, ls="--", label="naive int8 uniform (non-SC)"),
    Line2D([0], [0], color=C_SKIP, lw=0, marker="*", ms=13, markeredgecolor="white",
           label="SC full + skip blocks 0,4,18"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, 1.005))
fig.suptitle("How SC degrades the world model — quality vs. how many matmul types run on stochastic int8",
             fontsize=13, color=INK, fontweight="700", x=0.012, ha="left", y=1.10)
fig.text(0.012, 1.005,
         "Each matmul type is moved onto SC left→right. naive int8 (same ops, uniform quant) stays flat near FP → "
         "the collapse is SC noise, not low precision. The star shows skipping the 3 most sensitive blocks pulls quality back.",
         fontsize=9.3, color=INK2, ha="left")
fig.text(0.5, -0.02, "operator types moved to SC  (cumulative)",
         fontsize=10.5, color=INK2, ha="center")

os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/fig_sc_curve.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", os.path.abspath("results/figures/fig_sc_curve.png"))

# also dump the numbers used
tbl = {"setting": "DPM 10-step, n=8",
       "sequence": {l.replace(chr(10), " "): {k: data[i][k] for k in ("mean_l2", "mean_psnr", "mean_ssim")}
                    for i, (t, l) in enumerate(SEQ)},
       "naive_int8": {k: naive[k] for k in ("mean_l2", "mean_psnr", "mean_ssim")},
       "skip_0_4_18": {k: skip[k] for k in ("mean_l2", "mean_psnr", "mean_ssim")}}
json.dump(tbl, open("results/figures/sc_curve_data.json", "w"), indent=2)
print("wrote results/figures/sc_curve_data.json")
