"""Generate a polished hero banner for the README (retrieval vs. answer-in-context).
Run: python make_hero.py  ->  hero.png (high-res, ~2600px wide).

v2: soft drop shadows, document-style icons, a 'squeeze' funnel for the budget,
refined palette and typography.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
})

# ---- palette ----
GOLD_F, GOLD_E, GOLD_FOLD = "#cdeccf", "#2e8b57", "#a9d8b3"
DIST_F, DIST_E, DIST_FOLD = "#ededed", "#a2a2a2", "#dcdcdc"
PROC_F, PROC_E = "#e9f1fb", "#3b6fb0"
SET_E         = "#93a7bd"
FUNNEL_F, FUNNEL_E = "#fdf1c9", "#e0b100"
RED           = "#c0392b"
INK, MUTE     = "#212121", "#6f6f6f"

SHADOW = [pe.withSimplePatchShadow(offset=(1.6, -1.6), shadow_rgbFace="#9aa5b1", alpha=0.28)]

fig, ax = plt.subplots(figsize=(13, 4.7))
ax.set_xlim(0, 122); ax.set_ylim(0, 47); ax.axis("off")

def rbox(x, y, w, h, fc, ec, lw=1.7, label="", fs=11.5, bold=False, shadow=True, fcolor=INK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.7",
                       fc=fc, ec=ec, lw=lw, zorder=4)
    if shadow: p.set_path_effects(SHADOW)
    ax.add_patch(p)
    if label:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, color=fcolor, zorder=6,
                fontweight="bold" if bold else "normal")

def dashbox(x, y, w, h, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.7",
                 fc="#fbfcfe", ec=ec, lw=1.5, ls=(0, (5, 3)), zorder=3))

def paper(cx, cy, kind="gold", w=3.4, h=4.4, shadow=True):
    fc, ec, fold = (GOLD_F, GOLD_E, GOLD_FOLD) if kind == "gold" else (DIST_F, DIST_E, DIST_FOLD)
    fc2 = w * 0.34  # fold size
    body = Polygon([(cx-w/2, cy-h/2), (cx-w/2, cy+h/2), (cx+w/2-fc2, cy+h/2),
                    (cx+w/2, cy+h/2-fc2), (cx+w/2, cy-h/2)],
                   closed=True, fc=fc, ec=ec, lw=1.4, zorder=6, joinstyle="round")
    if shadow: body.set_path_effects(SHADOW)
    ax.add_patch(body)
    ax.add_patch(Polygon([(cx+w/2-fc2, cy+h/2), (cx+w/2-fc2, cy+h/2-fc2), (cx+w/2, cy+h/2-fc2)],
                 closed=True, fc=fold, ec=ec, lw=1.1, zorder=7))
    for i, dy in enumerate((0.8, -0.2, -1.2)):
        ax.plot([cx-w/2+0.7, cx+w/2-0.7], [cy+dy, cy+dy], color=ec, lw=0.8, alpha=0.55, zorder=8)
    if kind == "gold":  # small marker = answer-bearing
        ax.scatter([cx], [cy+h/2-0.9], s=16, marker="*", color=GOLD_E, zorder=9)

def arrow(x1, y1, x2, y2, color=INK, lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                 lw=lw, color=color, ls=ls, shrinkA=2, shrinkB=2, zorder=5))

# ---- query -> retriever ----
rbox(2, 26.5, 12, 7.5, "white", INK, label="query", fs=11)
arrow(14.5, 30.2, 20, 30.2)
rbox(20, 26, 16, 8.5, PROC_F, PROC_E, label="Retriever", fs=12, bold=True)

# ---- retrieved set (recall here) ----
dashbox(42, 23, 25, 15, SET_E)
for i, k in enumerate(["gold", "gold", "dist", "dist"]):
    paper(46.8 + i*4.8, 30.5, k)
ax.text(54.5, 40.0, "retrieved set", ha="center", fontsize=11.5, color=INK, fontweight="bold")
ax.text(54.5, 20.4, "recall@k  scored here", ha="center", fontsize=9.5, color=MUTE, style="italic")
arrow(36, 30.2, 41.5, 30.2)

# ---- budget squeeze funnel ----
arrow(67, 30.2, 69.5, 30.2, lw=2.2)
fn = Polygon([(70, 38), (81, 34.4), (81, 26.6), (70, 23)], closed=True,
             fc=FUNNEL_F, ec=FUNNEL_E, lw=1.6, zorder=3, joinstyle="round")
fn.set_path_effects(SHADOW); ax.add_patch(fn)
ax.text(75.5, 41.2, "budget  B ≤ 160 tokens", ha="center", fontsize=10,
        color="#8a6d00", fontweight="bold")

# ---- packed context (AiC here) ----
dashbox(83, 25, 14.5, 11.5, SET_E)
paper(87.4, 30.6, "gold"); paper(92.6, 30.6, "dist")
ax.text(90.2, 38.6, "packed context", ha="center", fontsize=11.5, color=INK, fontweight="bold")
ax.text(90.2, 22.6, "answer-in-context  scored here", ha="center", fontsize=9.5,
        color=MUTE, style="italic")
arrow(81.2, 30.2, 82.8, 30.2, lw=2.2)

# ---- dropped gold -> answer lost ----
paper(75.5, 11.5, "gold", shadow=False)
ax.add_patch(FancyBboxPatch((73.4, 8.9), 4.2, 5.2, boxstyle="round,pad=0.02,rounding_size=0.8",
             fc="none", ec=RED, lw=1.8, ls=(0, (2, 2)), zorder=9))
arrow(74.5, 24.5, 75.5, 14.6, color=RED, lw=1.7, ls=(0, (3, 2)))
ax.text(79.5, 11.5, "gold #2 dropped  →  answer lost", ha="left", va="center",
        fontsize=10.5, color=RED, fontweight="bold")

# ---- reader ----
arrow(97.5, 30.2, 101, 30.2)
rbox(101, 26, 16, 8.5, PROC_F, PROC_E, label="Reader", fs=12, bold=True)

# ---- strap line ----
ax.text(61, 3.4,
        "Recall counts what the retriever finds.  Under a budget, answer quality depends on "
        "what survives into the reader’s context.",
        ha="center", fontsize=11.5, color=INK)

fig.tight_layout(pad=0.4)
fig.savefig("hero.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote hero.png")
