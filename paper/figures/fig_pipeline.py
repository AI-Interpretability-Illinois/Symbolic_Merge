"""Figure 1: SAE-IDF in four steps on one real query (Common-KG NELL -> DBpedia; the example is chosen by fig_pipeline_data.py).

1  Symbol in context: one view of the NELL query "trainstation" ("Instance: rochester city. Class:
   trainstation") goes through the LLM as in any inference pass; the residual stream at layer l is
   read at the symbol's tokens only.
2  SAE encoder: a = JumpReLU(W_enc h + b_enc) maps the d-dimensional state to F = 131,072 features,
   of which about 114 are active.
3  Sparse code -> rarity weighting: the real largest coordinates of a(s) for "trainstation", before
   and after multiplying by idf (features firing on every symbol of the collection drop to zero),
   each with its automatic explanation (its meaning).
4  Matching: the query against DBpedia's "RailwayStation" (gold) and "AutombileEngine" (the candidate
   the dense state prefers): the cosine decomposes into per-feature contributions, carried by rare,
   named features.

Numbers come from data/fig_pipeline_example.json (fig_pipeline_data.py, from the stored run).
Units: 1 = 0.01 in on a 650-unit-wide canvas.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import style as S

S.setup()
plt.rcParams.update({"mathtext.fontset": "stix"})
HERE = Path(__file__).resolve().parent
D = json.loads((HERE.parent / "data" / "fig_pipeline_example.json").read_text())
W, H, Y0 = 650, 247, 27
fig = plt.figure(figsize=(W / 100, (H - Y0) / 100))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(Y0, H); ax.axis("off")

BLUE, GREY, INK, INK2, EDGE, ORANGE = S.BLUE, S.MUTED, S.INK, S.INK2, S.AXIS, S.ORANGE
CHIP_GREY, CHIP_EDGE, LLM_FILL, LLM_EDGE, SAE_FILL, SAE_EDGE = "#f1f0ec", "#cfcec6", "#f2f6fc", "#b9c8e0", "#e9f6ef", "#8fc9ad"
FS = 6.6
DATA = {"family": "DejaVu Sans Mono"}          # data samples: tokens, labels, view text
EXPL = {"family": "DejaVu Sans", "style": "italic"}   # automatic feature explanations
HEAD_Y = 240
P1, P2, P3, P4 = (4, 172), (186, 300), (318, 466), (484, 646)


def box(x, y, w, h, fc="white", ec=INK, lw=0.7, r=4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(x0, y0, x1, y1, color=INK2, lw=0.8, scale=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color, shrinkA=0, shrinkB=0))


def header(p, n, text):
    ax.text(p[0], HEAD_Y, f"{n}  {text}", fontsize=FS + 1.4, fontweight="bold", color=INK, va="center")


def fit_text(x, y, text, max_x, suffix="", **kw):
    """Draw `text` (an explanation) truncated at a word boundary so that it ends before max_x; suffix is kept."""
    renderer = fig.canvas.get_renderer()
    n = len(text)
    while True:
        t = ax.text(x, y, f"“{short(text, n)}”{suffix}", **kw)
        x1 = ax.transData.inverted().transform(t.get_window_extent(renderer=renderer))[1][0]
        if x1 <= max_x or n <= 8:
            return t
        t.remove()
        n = min(n - 1, int(n * (max_x - x) / max(x1 - x, 1e-6)))


def short(text, n):
    """Truncate an explanation to n characters at a word boundary, marking the cut with an ellipsis."""
    text = text.replace("“", "").replace("”", "").strip().rstrip(".")
    if len(text) <= n:
        return text
    cut = text[: n - 1]
    if " " in cut[n // 2:]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:") + "…"


# ====================================================================== 1  symbol in context (LLM inference)
header(P1, 1, "Symbol in context")
VIEW_PREF = "rochester"                                        # the view shown as tokens (NELL's instance lists are noisy)
view = next((v for v in D["symbol"]["views"] if VIEW_PREF in v.lower()), min(D["symbol"]["views"], key=len))
words = view.split()
tokens = [(wd, i == len(words) - 1) for i, wd in enumerate(words)]
x, cy, ch, gap = 6, 44, 15, 1.5
nchar = sum(len(w) for w, _ in tokens)
tfs = FS - 2.4 if nchar > 34 else FS - 1.6                      # monospace: ~0.6 em per character
cw = 0.6 * tfs * 100 / 72
spans = []
for word, is_sym in tokens:
    w = 2 + cw * len(word)
    box(x, cy - ch / 2, w, ch, fc=BLUE if is_sym else CHIP_GREY, ec=BLUE if is_sym else CHIP_EDGE, lw=0.5, r=2.5)
    ax.text(x + w / 2, cy, word, fontsize=tfs, color="white" if is_sym else INK, ha="center", va="center", fontweight="bold" if is_sym else "normal", **DATA)
    spans.append((x, w, is_sym)); x += w + gap
sym_x = spans[-1][0] + spans[-1][1] / 2
ax.text((spans[0][0] + spans[-2][0] + spans[-2][1]) / 2, cy - ch / 2 - 6, "context", fontsize=FS - 1.0, color=INK2, ha="center", va="center")
ax.text(sym_x, cy - ch / 2 - 6, "symbol", fontsize=FS - 1.0, color=BLUE, ha="center", va="center", fontweight="bold")

LX0, LX1, LY0, LY1 = 8, 168, 62, 228
box(LX0, LY0, LX1 - LX0, LY1 - LY0, fc=LLM_FILL, ec=LLM_EDGE, lw=0.7, r=6)
for tx, tw, is_sym in spans:
    arrow(tx + tw / 2, cy + ch / 2 + 1, tx + tw / 2, LY0 - 1, color=BLUE if is_sym else EDGE, lw=0.8 if is_sym else 0.5, scale=5)
BX0, BX1 = 22, 154


def layer(y, h, label, fc="white", ec=EDGE, color=INK, ls="-", fs=FS - 1.2):
    box(BX0, y, BX1 - BX0, h, fc=fc, ec=ec, lw=0.55, r=2, ls=ls)
    ax.text((BX0 + BX1) / 2, y + h / 2, label, fontsize=fs, color=color, ha="center", va="center")


layer(68, 10, "token embedding")
layer(83, 10, "transformer block 1")
ax.text((BX0 + BX1) / 2, 98.5, r"$\vdots$", fontsize=FS, color=INK2, ha="center", va="center")
layer(105, 10, r"transformer block $\ell = 20$")
RY = 122                                     # the residual stream after block l, one cell per token
for tx, tw, is_sym in spans:
    ax.add_patch(Rectangle((tx + tw / 2 - 5, RY), 10, 10, fc=BLUE if is_sym else "#d9d8d2", ec="white", lw=0.5))
layer(147, 10, r"blocks $\ell\!+\!1,\ldots,42$", fc="none", ec=EDGE, color=GREY, ls=(0, (2, 2)))
layer(162, 10, "unembedding", fc="none", ec=EDGE, color=GREY, ls=(0, (2, 2)))
layer(177, 10, "next-token logits (unused)", fc="none", ec=EDGE, color=GREY, ls=(0, (2, 2)))
ax.text((LX0 + LX1) / 2, LY1 - 9, "LLM (Gemma-2-9B)", fontsize=FS + 0.6, color=INK, ha="center", va="center", fontweight="bold")

# ====================================================================== 2  SAE encoder
header(P2, 2, "SAE encoder")
SX0, SX1, SY0, SY1 = P2[0], P2[1], 62, 228
box(SX0, SY0, SX1 - SX0, SY1 - SY0, fc=SAE_FILL, ec=SAE_EDGE, lw=0.7, r=6)
arrow(sym_x + 5, RY + 5, SX0 - 1, RY + 5, color=BLUE, lw=1.0)             # h_t of the symbol token enters the SAE
# input: a dense vector, all coordinates non-zero
rng = np.random.default_rng(3)
IX0, IY = SX0 + 10, RY
dense_cells = 22
for k in range(dense_cells):
    v = 0.35 + 0.65 * rng.random()
    ax.add_patch(Rectangle((IX0 + k * 4.2, IY), 3.8, 10, fc=plt.cm.Blues(0.25 + 0.6 * v), ec="white", lw=0.3))
ax.text(IX0, IY - 7, r"$h_t\in\mathbb{R}^{3584}$ (dense)", fontsize=FS - 1.2, color=INK2, va="center")
# the computation, beside an arrow from the dense input up to the sparse output
arrow(SX0 + 6, IY + 12, SX0 + 6, IY + 50, color=INK2, lw=0.8)
ax.text(SX0 + 9, IY + 31, r"$E(h_t) = \mathrm{JumpReLU}\,(W_{\!enc}\,h_t + b_{enc})$", fontsize=FS - 1.3, color=INK, va="center")
# output: a long sparse vector
OY = IY + 52
n_out = 60
active_cells = {7, 19, 31, 44, 52}
for k in range(n_out):
    xk = SX0 + 8 + k * (SX1 - SX0 - 16) / n_out
    ax.add_patch(Rectangle((xk, OY), (SX1 - SX0 - 16) / n_out - 0.4, 10, fc=BLUE if k in active_cells else "#f7f7f4", ec="#d2d1ca" if k not in active_cells else BLUE, lw=0.3))
ax.text(SX0 + 8, OY + 16, r"$E(h_t)\in\mathbb{R}^{131{,}072}$, $\approx$114 non-zero", fontsize=FS - 1.2, color=INK2, va="center")
ax.text((SX0 + SX1) / 2, SY1 - 9, "Gemma Scope SAE", fontsize=FS + 0.6, color=INK, ha="center", va="center", fontweight="bold")
ax.text((SX0 + SX1) / 2, SY1 - 19, "(Layer 20)", fontsize=FS + 0.2, color=INK, ha="center", va="center", fontweight="bold")
arrow(SX1 + 1, OY + 5, P3[0] - 1, OY + 5, color=BLUE, lw=1.0)

# ====================================================================== 3  sparse code -> rarity weighting (real values)
header(P3, 3, "Sparse code, rarity-weighted")
sym = D["symbol"]
N = D["n_docs_reference"]
# representative features: the largest unweighted coordinates (incl. the ubiquitous ones) and the largest weighted ones
picked = []
for r in sym["top_unweighted"][:6] + sym["top_weighted"][:4]:
    if r["feature"] not in [p["feature"] for p in picked]:
        picked.append(r)
picked = sorted(picked, key=lambda r: -r["a"])[:6]
amax, wmax = max(r["a"] for r in picked), max(r["w"] for r in picked)
col_a, col_w, BW = P3[0] + 2, P3[0] + 76, 46       # two bar columns: a(s) and w(s) = a(s)·idf
rowh, y_top = 23, 210
ax.text(P3[0] + 2, y_top + 19, r"$a(s)$ = mean of $E(h_t)$ over symbol tokens and contexts", fontsize=FS - 1.6, color=INK2, va="center")
ax.text(col_a + BW / 2, y_top + 9, r"$a_f(s)$", fontsize=FS - 0.4, color=INK, ha="center", va="center")
ax.text(P3[1], y_top + 9, r"$w_f(s)=a_f(s)\cdot\mathrm{idf}_f$", fontsize=FS - 0.4, color=INK, ha="right", va="center")   # right-aligned above the weighted bars
for i, r in enumerate(picked):
    y = y_top - i * rowh
    col = GREY if r["w"] < r["a"] else BLUE           # grey: damped by idf (fires on many symbols)
    ax.add_patch(Rectangle((col_a, y - 5), BW * r["a"] / amax, 7, fc=col, ec="none"))
    ax.text(col_a + BW * r["a"] / amax + 2, y - 1.5, f"{r['a']:.2f}", fontsize=FS - 1.6, color=INK2, va="center")
    ax.add_patch(Rectangle((col_w, y - 5), max(BW * r["w"] / wmax, 0.6), 7, fc=col, ec="none"))
    ax.text(col_w + max(BW * r["w"] / wmax, 0.6) + 2, y - 1.5, f"{r['w']:.2f}", fontsize=FS - 1.6, color=INK2, va="center")
    ax.text(P3[1], y - 11.5, f"df {r['df']}/{N}", fontsize=FS - 1.8, color=col, va="center", ha="right")
    fit_text(col_a, y - 11.5, r["explanation"], P3[1] - 30, fontsize=FS - 2.0, color=col, va="center", **EXPL)
ax.text(P3[0] + 2, y_top - 5 * rowh - 24, "“…”: feature meaning (automatic explanation)", fontsize=FS - 1.7, color=INK2, va="center", **EXPL)
arrow(P3[1] + 2, 206, P4[0] - 2, 206, color=BLUE, lw=1.0, scale=8)

# ====================================================================== 4  matching (real values)
header(P4, 4, "Matching")
m = D["match"]
q, g, dis = m["query"], m["gold"], m["distractor"]
SRC, TGT = D.get("source_system", "source"), D.get("target_system", "target")


def text_end(t):
    """Right edge (data units) of a drawn text object."""
    return ax.transData.inverted().transform(t.get_window_extent(renderer=fig.canvas.get_renderer()))[1][0]


def cand_block(y0, cand, title_color, tag, dfs=0.0, rowgap=12, bh=6, prefix=""):
    """One method's top-1 candidate: '<tag> <label>', its cosine with the query, and the four largest
    shares of that cosine. dfs shifts every font size (the dense comparison is drawn smaller)."""
    t = ax.text(P4[0], y0, tag, fontsize=FS - 0.6 + dfs, color=title_color, va="center", fontweight="bold")
    ax.text(text_end(t) + 3, y0, cand["label"], fontsize=FS - 1.0 + dfs, color=title_color, va="center", fontweight="bold", **DATA)
    t = ax.text(P4[0], y0 - 9, f"({prefix}cos sim.(query, top 1) = {cand['cos_idf']:.2f}, rank = {cand['rank_idf']})", fontsize=FS - 1.0 + dfs, color=title_color, va="center")
    print("cosine line ends at", round(text_end(t), 1), "of", P4[1])
    for i, c in enumerate(cand["contributions"][:4]):
        y = y0 - 19 - i * rowgap
        share = c["c"] / cand["cos_idf"]                      # the feature's share of this cosine
        bw = max(40 * share / 0.6, 0.6)                       # 40 units = 60 %
        ax.add_patch(Rectangle((P4[0], y - bh / 2), bw, bh, fc=title_color, ec="none"))
        ax.text(P4[0] + bw + 2, y, f"{100 * share:.0f}%", fontsize=FS - 1.7 + dfs, color=INK2, va="center")
        fit_text(P4[0] + bw + 15, y, c["explanation"], P4[1] - 1, fontsize=FS - 2.0 + dfs, color=INK2, va="center", **EXPL)


for yy, tag, label in ((224, f"Query ({SRC}):", q["label"]), (213, f"Gold ({TGT}):", g["label"])):
    t_tag = ax.text(P4[0], yy, tag, fontsize=FS - 0.6, color=INK, va="center", fontweight="bold")
    ax.text(text_end(t_tag) + 3, yy, label, fontsize=FS - 1.0, color=INK, va="center", fontweight="bold", **DATA)
cand_block(199, g, BLUE, "SAE-IDF top 1:", rowgap=13)
# the dense hidden state's top 1: a comparison, not part of the pipeline, so boxed and smaller
DX0, DY0, DX1, DY1 = P4[0] - 4, 64, P4[1] + 2, 133
box(DX0, DY0, DX1 - DX0, DY1 - DY0, fc="#fbfbf9", ec=CHIP_EDGE, lw=0.6, r=3, ls=(0, (3, 2)))
cand_block(126, dis, GREY, "Dense top 1:", dfs=-0.5, rowgap=12, bh=5, prefix="SAE-IDF: ")

fig.savefig(HERE / "fig_pipeline.pdf"); fig.savefig(HERE / "fig_pipeline.png", dpi=170)
print("wrote fig_pipeline.pdf")
