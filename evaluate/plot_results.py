"""Static PNG figures for the SC-quantization world-model study.

Outputs (results/figures/):
    fig_comparison.png   4 configs x 3 metrics (latent L2 / PSNR / SSIM)
    fig_sensitivity.png  per-block leave-one-out sensitivity (Δ latent L2)
    fig_overview.png     both panels stacked, one-glance summary
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---- validated palette (light surface) ----
C_FP, C_NAIVE, C_SCU, C_SKIP = "#2a78d6", "#1baf7a", "#e34948", "#eb6834"
C_NEG, C_POS = "#2a78d6", "#e34948"          # improve / worsen (diverging)
INK, INK2, MUTED, GRID = "#0f1319", "#555b66", "#868b95", "#e6e9ee"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 200, "savefig.dpi": 200,
})

METHODS = [
    ("FP baseline\n(math)",       0.1702, 25.75, 0.834, C_FP),
    ("Naive int8\n(uniform)",     0.1755, 25.08, 0.816, C_NAIVE),
    ("SC uniform\n(sc_int8_full)",0.5637, 13.61, 0.408, C_SCU),
    ("SC + skip\n0,4,18",         0.3521, 16.53, 0.472, C_SKIP),
]
REF = 0.5585
SENS = {0:0.4843,1:6.2325,2:0.5361,3:0.5798,4:0.5117,5:0.5329,6:0.5315,7:0.5312,8:0.5323,9:0.5331,
10:0.5473,11:0.5407,12:0.5376,13:0.5472,14:0.5311,15:0.5468,16:0.5432,17:0.5333,18:0.5142,19:0.5447,
20:0.5612,21:0.5460,22:0.5417,23:0.5611,24:0.5352,25:0.5262-0.001,26:0.5262,27:0.5278}
SENS[25]=0.5252
TOP3 = {0,4,18}

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_comparison(axes):
    names   = [m[0] for m in METHODS]
    colors  = [m[4] for m in METHODS]
    y = range(len(METHODS))
    specs = [
        (1, "latent L2  (↓ lower better)", 0.62, "{:.3f}"),
        (2, "PSNR / dB  (↑ higher better)", 29,  "{:.2f}"),
        (3, "SSIM  (↑ higher better)",      1.02,"{:.3f}"),
    ]
    for ax, (idx, title, xmax, fmt) in zip(axes, specs):
        vals = [m[idx] for m in METHODS]
        bars = ax.barh(y, vals, height=0.62, color=colors, zorder=3)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(METHODS)-0.4)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")
        ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        _clean(ax)
        ax.tick_params(length=0)
        for b, v in zip(bars, vals):
            ax.text(v + xmax*0.02, b.get_y()+b.get_height()/2, fmt.format(v),
                    va="center", ha="left", fontsize=10, color=INK, fontweight="600")
        axes[0].set_yticks(list(y)); axes[0].set_yticklabels(names, fontsize=9.5, color=INK)
    for ax in axes[1:]:
        ax.set_yticks(list(y)); ax.set_yticklabels([])


def draw_sensitivity(ax):
    ymin, ymax = -0.085, 0.042
    blocks = list(range(28))
    for b in blocks:
        d = SENS[b] - REF
        clipped = d > ymax
        dd = ymax if clipped else d
        color = C_NEG if d < 0 else C_POS
        ax.bar(b, dd, width=0.72, color=color, zorder=3,
               edgecolor=INK if b in TOP3 else "none", linewidth=1.1 if b in TOP3 else 0)
        if clipped:
            # break marks near the clipped top, then value above the bar
            ax.plot([b-0.36, b+0.36], [ymax*0.60]*2, color="white", lw=1.3, zorder=5)
            ax.plot([b-0.36, b+0.36], [ymax*0.74]*2, color="white", lw=1.3, zorder=5)
            ax.text(b, ymax+0.003, "L2=6.23", ha="center", va="bottom",
                    fontsize=8.5, color=C_POS, fontweight="700")
        if b in TOP3:
            ax.text(b, d-0.004, f"{d:.3f}", ha="center", va="top",
                    fontsize=8.5, color=INK, fontweight="700")
    ax.axhline(0, color="#c3c2b7", lw=1.2, zorder=2)
    ax.set_xlim(-0.8, 27.8)
    ax.set_ylim(ymin, 0.052)
    ax.set_xticks(blocks)
    ax.set_xticklabels([str(b) for b in blocks], fontsize=8)
    for t, b in zip(ax.get_xticklabels(), blocks):
        t.set_color(INK if b in TOP3 else MUTED)
        if b in TOP3: t.set_fontweight("700")
    ax.yaxis.set_major_locator(MultipleLocator(0.025))
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    _clean(ax)
    ax.tick_params(length=0)
    ax.set_xlabel("transformer block index (0–27)", fontsize=10, color=INK2)
    ax.set_ylabel("Δ latent L2  vs  SC uniform baseline", fontsize=10, color=INK2)
    # direction legend, placed in the empty middle band (avoids all bars)
    ax.text(14, 0.036, "red ↑   skip → worsens", ha="center",
            fontsize=9, color=C_POS, fontweight="600")
    ax.text(14, -0.080, "blue ↓   noise source, skip → improves", ha="center",
            fontsize=9, color=C_NEG, fontweight="600")


# ---- fig 1: comparison ----
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
draw_comparison(axes)
fig.suptitle("SC quantization vs baselines  ·  bridge test, n=20, PNDM 50-step",
             fontsize=12.5, color=INK, fontweight="700", x=0.02, ha="left", y=1.04)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(f"{OUT}/fig_comparison.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- fig 2: sensitivity ----
fig, ax = plt.subplots(figsize=(11, 3.6))
draw_sensitivity(ax)
fig.suptitle("Per-block leave-one-out sensitivity  ·  DPM 10-step, n=2  ·  top-3 = blocks 0, 4, 18",
             fontsize=12.5, color=INK, fontweight="700", x=0.02, ha="left", y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_sensitivity.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- fig 3: overview (both) ----
fig = plt.figure(figsize=(11, 7.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.55, wspace=0.12)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
draw_comparison(axes)
axb = fig.add_subplot(gs[1, :])
draw_sensitivity(axb)
fig.suptitle("Stochastic-computing int8 in the IRASim world model — full comparison",
             fontsize=14, color=INK, fontweight="700", x=0.02, ha="left", y=0.99)
fig.text(0.02, 0.945,
         "SC full-replace collapses quality (PSNR 25.75→13.61); naive int8 is near-lossless "
         "→ damage is SC noise, not low precision. Skipping the 3 most sensitive blocks recovers 53.8% of the L2 gap.",
         fontsize=9.5, color=INK2, ha="left")
fig.savefig(f"{OUT}/fig_overview.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("wrote:")
for f in ("fig_comparison.png", "fig_sensitivity.png", "fig_overview.png"):
    print(" ", os.path.abspath(f"{OUT}/{f}"))
