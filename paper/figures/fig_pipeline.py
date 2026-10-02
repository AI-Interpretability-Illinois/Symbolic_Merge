"""Figure 1: the SAE-IDF pipeline, left to right, with one worked view."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import style as S

S.setup()
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "cm"})
HERE = Path(__file__).resolve().parent
W, H = 1000, 360
fig = plt.figure(figsize=(S.TEXT_IN, S.TEXT_IN * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
BLUE, BLUE_T, GREY_T, EDGE = S.BLUE, "#eaf1fc", "#f0efec", S.AXIS
FS = 7.2


def box(x, y, w, h, fc, ec, lw=0.8, r=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, color=S.INK2, lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=lw, color=color, shrinkA=0, shrinkB=0))


def head(x, n, text):
    ax.text(x, H - 14, f"{n}  {text}", fontsize=FS + 0.8, color=S.INK, fontweight="bold", va="center")


# --- panel 1: a native view, evidence then symbol ------------------------------------------
head(12, "1", "Native view")
tokens = [("Instance:", 0), ("Earth", 0), ("mass.", 0), ("Class:", 0), ("Mass", 1)]
x, y0, th, gap = 14, 262, 28, 6
spans = []
for word, sym in tokens:
    w = 10 + 7.4 * len(word)
    box(x, y0, w, th, BLUE if sym else GREY_T, BLUE if sym else EDGE, 0.7, 4)
    ax.text(x + w / 2, y0 + th / 2, word, fontsize=FS, color="white" if sym else S.INK, ha="center", va="center",
            fontweight="bold" if sym else "normal")
    spans.append((x, w, sym)); x += w + gap
ev_l, ev_r = spans[0][0], spans[-2][0] + spans[-2][1]
sx, sw, _ = spans[-1]
ax.text(14, y0 + th + 12, r"evidence $e_j(s)$ from the symbol's own system, then the symbol $s$", fontsize=FS - 0.6, color=S.INK2, ha="left", va="center")
# LM
lm_y, lm_h = 170, 50
box(14, lm_y, x - gap - 14, lm_h, "white", S.INK, 0.8)
ax.text((14 + x - gap) / 2, lm_y + lm_h / 2 + 8, "LM reads the whole view", fontsize=FS, color=S.INK, ha="center", va="center", fontweight="bold")
ax.text((14 + x - gap) / 2, lm_y + lm_h / 2 - 9, r"residual stream at layer $\ell$", fontsize=FS - 0.4, color=S.INK2, ha="center", va="center")
for (tx, tw, sym) in spans:
    arrow(tx + tw / 2, y0 - 2, tx + tw / 2, lm_y + lm_h + 1, BLUE if sym else S.AXIS, 1.0 if sym else 0.7)
cx = sx + sw / 2
arrow(cx, lm_y - 2, cx, 112, BLUE)
ax.text(cx + 6, 140, r"keep $h_t$,", fontsize=FS - 0.4, color=BLUE, ha="left", va="center")
ax.text(cx + 6, 126, r"$t \in T_j(s)$", fontsize=FS - 0.4, color=BLUE, ha="left", va="center")
ax.text((ev_l + ev_r) / 2 - 30, 133, "evidence states\nare discarded", fontsize=FS - 1, color=S.INK2, ha="center", va="center", style="italic")
ax.text(14, 88, "A program, a formal statement or a table\ncolumn is its own single view: every token\nis pooled.", fontsize=FS - 1, color=S.INK2, va="top")
ax.text(14, 40, "Ontology / KG terms: one view per instance,\naxiom, definition or parent (up to 16).", fontsize=FS - 1, color=S.INK2, va="top")

# --- panel 2: sparse code ---------------------------------------------------------------------
P2 = 330
head(P2, "2", "Sparse code")
ax.text(P2, 300, r"$a(s) = \frac{1}{m}\sum_{j}\frac{1}{|T_j(s)|}\sum_{t \in T_j(s)} E(h_t)\in\mathbb{R}^{F}$", fontsize=FS + 1.2, color=S.INK, va="center")
ax.text(P2, 274, "encode each token, then average\nover tokens and views", fontsize=FS - 0.8, color=S.INK2, va="center", linespacing=1.1)
# a sparse bar sketch: F features on the x axis, a few active
import numpy as np
rng = np.random.default_rng(3)
bx, by, bw, bh = P2 + 36, 135, 270, 95
ax.plot([bx, bx + bw], [by, by], color=S.AXIS, lw=0.6)
feats = {18: (0.95, "notation", True), 61: (0.55, "numbers", True), 130: (0.35, None, True),
         203: (0.7, "“mass”", False), 240: (0.45, "“weight”", False), 275: (0.25, None, False)}
for xi, (hgt, lab, generic) in feats.items():
    col = S.MUTED if generic else BLUE
    ax.plot([bx + xi, bx + xi], [by, by + hgt * bh], color=col, lw=2.2, solid_capstyle="round")
    if lab:
        ax.text(bx + xi, by + hgt * bh + 6, lab, fontsize=FS - 1.2, color=col, ha="center", va="bottom")
ax.text(bx + bw / 2, by - 12, r"features $f = 1,\ldots,F$  ($F$ = 131,072)", fontsize=FS - 1, color=S.INK2, ha="center", va="center")
ax.text(P2, 86, "grey: features that fire on almost every symbol\n(notation, numbers, licence text)", fontsize=FS - 1, color=S.MUTED, va="top")
ax.text(P2, 46, "blue: rare features specific to this symbol", fontsize=FS - 1, color=BLUE, va="top")

# --- panel 3: rarity weighting + match ---------------------------------------------------------
P3 = 665
head(P3, "3", "Rarity weighting and matching")
ax.text(P3, 300, r"$w_f(s) = a_f(s)\cdot \mathrm{idf}_f,\quad \mathrm{idf}_f=\log\frac{N}{1+\mathrm{df}(f)}$", fontsize=FS + 1.2, color=S.INK, va="center")
ax.text(P3, 274, "df$(f)$: symbols in the reference\ncollection on which $f$ fires", fontsize=FS - 0.8, color=S.INK2, va="center", linespacing=1.1)
bx3 = P3
ax.plot([bx3, bx3 + bw], [by, by], color=S.AXIS, lw=0.6)
weighted = {18: 0.02, 61: 0.04, 130: 0.08, 203: 0.95, 240: 0.6, 275: 0.4}
for xi, hgt in weighted.items():
    generic = feats[xi][2]
    ax.plot([bx3 + xi, bx3 + xi], [by, by + hgt * bh], color=S.MUTED if generic else BLUE, lw=2.2, solid_capstyle="round")
ax.annotate("", xy=(bx3 + 40, by + 0.3 * bh), xytext=(bx3 + 90, by + 0.75 * bh), arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=0.8, mutation_scale=8))
ax.text(bx3 + 95, by + 0.78 * bh, "ubiquitous features → 0", fontsize=FS - 1, color=S.MUTED, va="bottom")
ax.text(bx3 + bw / 2, by - 12, r"$w(s) = a(s)\odot\mathrm{idf} \in \mathbb{R}^{F}$", fontsize=FS - 0.4, color=S.INK2, ha="center", va="center")
ax.text(P3, 92, r"$\hat t(s) = \arg\max_{t\in C(s)} \cos(w(s),\, w(t))$", fontsize=FS + 1.2, color=S.INK, va="top")
ax.text(P3, 52, "the cosine is a sum over shared features, so every\nmatch reads as a list of named features", fontsize=FS - 1, color=S.INK2, va="top")

# arrows between panels
for xa in (P2 - 22, P3 - 22):
    arrow(xa - 8, 190, xa + 12, 190, S.INK2, 1.2)
fig.savefig(HERE / "fig_pipeline.pdf"); fig.savefig(HERE / "fig_pipeline.png", dpi=150)
print("wrote fig_pipeline.pdf")
