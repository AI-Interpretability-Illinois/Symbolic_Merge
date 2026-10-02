"""Figure 4: share of each correct match's cosine carried by common features
(active on >1% of Pile tokens), unweighted SAE code vs SAE-IDF, per task."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import style as S

S.setup()
HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE.parent / "data" / "interpretability_summary.csv")))
order = ["miniF2F Lean 4 → Isabelle", "miniF2F Lean 4 → Metamath", "miniF2F Lean 4 → HOL Light",
         "XLCoST, Java to 6 languages", "Valentine, whole column", "Common-KG NELL → DBpedia",
         "Common-KG YAGO → Wikidata", "MultiFarm zh → en", "MultiFarm ru → en", "MultiFarm ar → en"]
short = {"miniF2F Lean 4 → Isabelle": "Lean 4 → Isabelle", "miniF2F Lean 4 → Metamath": "Lean 4 → Metamath",
         "miniF2F Lean 4 → HOL Light": "Lean 4 → HOL Light", "XLCoST, Java to 6 languages": "XLCoST Java → 6 langs",
         "Valentine, whole column": "Valentine columns", "Common-KG NELL → DBpedia": "NELL → DBpedia",
         "Common-KG YAGO → Wikidata": "YAGO → Wikidata", "MultiFarm zh → en": "MultiFarm zh → en",
         "MultiFarm ru → en": "MultiFarm ru → en", "MultiFarm ar → en": "MultiFarm ar → en"}
D = {(r["task"], r["method"]): float(r["common"]) for r in rows}

fig, ax = plt.subplots(figsize=(S.COLUMN_IN, 2.1))
ys = list(range(len(order)))[::-1]
for y, t in zip(ys, order):
    a, b = D[(t, "sae_mean")], D[(t, "sae_idf")]
    ax.plot([b, a], [y, y], color=S.AXIS, lw=1.2, zorder=1, solid_capstyle="round")
    ax.plot(a, y, zorder=2, **S.marker("sae_mean", markersize=4.5))
    ax.plot(b, y, zorder=3, **S.marker("sae_idf", markersize=5))
    ax.annotate(f"{a:.0%}", (a, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color=S.INK2)
ax.set_yticks(ys); ax.set_yticklabels([short[t] for t in order])
ax.set_xlim(0, 1.0); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Share of match score from features active on >1% of Pile tokens", fontsize=7.3)
ax.grid(axis="y", visible=False)
handles = [plt.Line2D([], [], **S.marker(m, markersize=4.5), label=S.SERIES[m]["label"]) for m in ("sae_mean", "sae_idf")]
fig.legend(handles=handles, loc="lower center", ncol=2, columnspacing=1.2, handletextpad=0.3, bbox_to_anchor=(0.55, -0.005))
fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1))
fig.savefig(HERE / "fig_interp.pdf"); fig.savefig(HERE / "fig_interp.png", dpi=150)
print("wrote fig_interp.pdf")
