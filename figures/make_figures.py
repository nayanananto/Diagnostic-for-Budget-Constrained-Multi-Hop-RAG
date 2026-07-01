"""Generate the two data figures for the paper (reader ladder + AiC incremental validity).
Run: python make_figures.py   -> writes reader_ladder.png and aic_validity.png (300 dpi)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 300,
})

C_SUBFOC = "#1b6ca8"   # submod - focused
C_SUBPAK = "#c44e52"   # submod - packed
C_CTRL   = "#5aa0d0"   # 7B-4bit control

# ------------------------------------------------------------------
# Figure 2: reader-scale ladder
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.35, 2.75))

# x positions for 3B, 7B, 14B (evenly spaced, categorical)
x = [0, 1, 2]
labels = ["3B", "7B", "14B"]

# submod - focused: 3B +0.022, 7B-fp16 -0.010, 14B -0.029   (CIs)
sf_y   = [0.022, -0.010, -0.029]
sf_lo  = [0.002, -0.035, -0.052]
sf_hi  = [0.041,  0.015, -0.006]
sf_err = [[sf_y[i]-sf_lo[i] for i in range(3)], [sf_hi[i]-sf_y[i] for i in range(3)]]

# 7B-4bit precision control at x=1 (slight offset), -0.008 [-0.032,+0.017]
ctrl_x, ctrl_y = 1.18, -0.008
ctrl_err = [[ctrl_y-(-0.032)], [0.017-ctrl_y]]

# submod - packed: 3B +0.054, 7B +0.054, 14B +0.044 (all p<=0.001)
sp_y = [0.054, 0.054, 0.044]

# zero line + region shading
ax.axhline(0, color="0.45", lw=0.9, ls="--", zorder=1)
ax.axhspan(0, 0.075, color="#e8f1e8", alpha=0.6, zorder=0)
ax.axhspan(-0.065, 0, color="#fbeaea", alpha=0.6, zorder=0)

# submod - packed line (robust positive)
ax.plot(x, sp_y, "-s", color=C_SUBPAK, lw=1.6, ms=5, zorder=4,
        label=r"submod $-$ naive packed")
# submod - focused line (erodes + reverses)
ax.errorbar(x, sf_y, yerr=sf_err, fmt="-o", color=C_SUBFOC, lw=1.6, ms=5,
            capsize=2.5, elinewidth=1.0, zorder=5, label=r"submod $-$ focused")
# 7B-4bit control point
ax.errorbar([ctrl_x], [ctrl_y], yerr=ctrl_err, fmt="D", color=C_CTRL, ms=4.5,
            mfc="white", mec=C_CTRL, capsize=2.5, elinewidth=1.0, zorder=6,
            label="7B 4-bit (control)")

# significance stars
ax.annotate("*", (x[0], sf_hi[0]+0.004), ha="center", fontsize=11, color=C_SUBFOC)
ax.annotate("n.s.", (x[1]-0.02, sf_hi[1]+0.006), ha="center", fontsize=6.5, color="0.4")
ax.annotate("*", (x[2], sf_lo[2]-0.012), ha="center", fontsize=11, color=C_SUBFOC)

# region text
ax.text(0.04, 0.066, "packer wins", fontsize=7, color="#2f6f3f", style="italic")
ax.text(1.40, -0.057, "heuristic wins", fontsize=7, color="#9b3030", style="italic",
        ha="right")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlim(-0.3, 2.45)
ax.set_ylim(-0.065, 0.078)
ax.set_xlabel("Reader scale (Qwen2.5-Instruct)")
ax.set_ylabel(r"$\Delta$ F1 (paired bootstrap)")
ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderpad=0.2,
          labelspacing=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig("reader_ladder.png", bbox_inches="tight")
plt.close(fig)
print("wrote reader_ladder.png")

# ------------------------------------------------------------------
# Figure 3: AiC incremental validity (retrieval held perfect)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.35, 2.6))
groups = ["F1", "EM"]
aic1 = [0.61, 0.50]
aic0 = [0.20, 0.11]
xpos = np.arange(len(groups))
w = 0.34
b1 = ax.bar(xpos - w/2, aic1, w, color=C_SUBFOC, label="answer in context", zorder=3)
b0 = ax.bar(xpos + w/2, aic0, w, color="#d9a441", label="answer dropped", zorder=3)
for b in list(b1) + list(b0):
    ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()+0.012),
                ha="center", fontsize=7.5)
ax.set_xticks(xpos)
ax.set_xticklabels(groups)
ax.set_ylim(0, 0.74)
ax.set_ylabel("Answer quality")
ax.set_title("Among questions with all gold retrieved", fontsize=8.2)
ax.legend(loc="upper right", frameon=False, handlelength=1.3, labelspacing=0.25,
          bbox_to_anchor=(1.0, 0.92))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
# annotate the 27% drop in the empty central band between the two groups
ax.text(0.5, 0.36, "27% of\nretrieval-perfect\nquestions drop\nthe answer",
        ha="center", va="center", fontsize=6.8, color="0.3",
        bbox=dict(boxstyle="round,pad=0.3", fc="0.96", ec="0.8", lw=0.5))
fig.tight_layout(pad=0.3)
fig.savefig("aic_validity.png", bbox_inches="tight")
plt.close(fig)
print("wrote aic_validity.png")

# ------------------------------------------------------------------
# Figure 4: budget sweep (inverted-U of submod - focused vs. budget)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.35, 2.6))

budgets = [96, 128, 160, 224]
sf      = [0.004, 0.014, 0.035, 0.017]   # submod - focused
pvals   = [0.81, 0.45, 0.04, 0.26]

# submod - naive packed: robustly positive band (+0.044 to +0.055, all p<=0.022)
ax.axhspan(0.044, 0.055, color=C_SUBPAK, alpha=0.13, zorder=0)
ax.text(160, 0.0495, r"submod $-$ naive packed (all $p\!\leq\!.02$)",
        fontsize=6.3, color="#9b3030", ha="center", va="center", style="italic")

ax.axhline(0, color="0.45", lw=0.9, ls="--", zorder=1)

# the inverted-U: hollow markers (n.s.) + filled significant peak
ax.plot(budgets, sf, "-o", color=C_SUBFOC, lw=1.6, ms=5.5, zorder=3,
        mfc="white", mec=C_SUBFOC)
ax.plot(160, 0.035, "o", color=C_SUBFOC, ms=6.5, zorder=5)          # sig peak filled
ax.annotate("*", (160, 0.0385), ha="center", va="bottom", fontsize=12, color=C_SUBFOC)
ax.annotate(r"$p{=}.04$", (167, 0.0335), ha="left", va="center",
            fontsize=6.6, color=C_SUBFOC)
ax.annotate(r"submod $-$ focused", (118, 0.027), ha="left", va="center",
            fontsize=6.6, color=C_SUBFOC, style="italic")

# mechanism notes at the two ends
ax.annotate("only 2-3\nsnippets fit", (96, 0.004), xytext=(99, -0.012),
            ha="center", va="top", fontsize=6.0, color="0.45")
ax.annotate("heuristic\ncatches up", (224, 0.017), xytext=(216, 0.030),
            ha="center", va="center", fontsize=6.0, color="0.45")

ax.set_xticks(budgets)
ax.set_xticklabels([str(b) for b in budgets])
ax.set_xlim(82, 240)
ax.set_ylim(-0.022, 0.062)
ax.set_xlabel("Reader-token budget $B$")
ax.set_ylabel(r"$\Delta$ F1 (paired bootstrap)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig("budget_sweep.png", bbox_inches="tight")
plt.close(fig)
print("wrote budget_sweep.png")
