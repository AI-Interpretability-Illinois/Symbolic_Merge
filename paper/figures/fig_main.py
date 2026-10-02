"""Figure 2: main results as a dot plot, one row per task, four methods."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import style as S

S.setup()
HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE.parent / "data" / "all_results.csv")))
val = lambda r, k: float(r[k]) if r.get(k) else None

tasks = [  # (label, selector)
    ("Lean 4 → Isabelle", lambda r: r["task"] == "miniF2F Lean4->Isabelle"),
    ("Lean 4 → Metamath", lambda r: r["task"] == "miniF2F Lean4->Metamath"),
    ("Lean 4 → HOL Light", lambda r: r["task"] == "miniF2F Lean4->HOL Light"),
    ("XLCoST, 7 languages (mean)", None),
    ("Valentine columns", lambda r: r["task"].startswith("Valentine") and r["metric"] == "MeanReciprocalRank" and r["group"] == "main"),
    ("NELL → DBpedia", lambda r: r["task"] == "Common-KG NELL->DBpedia" and r["group"] == "main"),
    ("YAGO → Wikidata", lambda r: r["task"] == "Common-KG YAGO->Wikidata" and r["group"] == "main"),
    ("MultiFarm zh → en", lambda r: r["task"] == "MultiFarm Chinese->English" and r["group"] == "main"),
    ("MultiFarm ru → en", lambda r: r["task"] == "MultiFarm Russian->English" and r["group"] == "main"),
    ("MultiFarm ar → en", lambda r: r["task"] == "MultiFarm Arabic->English" and r["group"] == "main"),
]
methods = ["strings", "dense", "dense_pc1", "sae_idf"]
data = []
for label, sel in tasks:
    if sel is None:
        xs = [r for r in rows if r["task"].startswith("XLCoST") and r["metric"] == "MRR"]
        data.append((label, {m: sum(val(r, m) for r in xs) / len(xs) for m in methods}))
    else:
        r = next(r for r in rows if sel(r))
        data.append((label, {m: val(r, m) for m in methods}))

fig, ax = plt.subplots(figsize=(S.COLUMN_IN, 3.0))
ys = list(range(len(data)))[::-1]
for y, (label, d) in zip(ys, data):
    ax.plot([d["dense"], d["sae_idf"]], [y, y], color=S.AXIS, lw=1.2, zorder=1, solid_capstyle="round")
    for m in methods:
        if d[m] is None:
            continue
        ax.plot(d[m], y, zorder=3 if m == "sae_idf" else 2, **S.marker(m, markersize=5 if m == "sae_idf" else 4.5))
ax.set_yticks(ys); ax.set_yticklabels([t for t, _ in data])
ax.set_xlim(0, 1.0); ax.set_xlabel("MRR (XLCoST: official MRR)")
ax.grid(axis="y", visible=False)
for y in (6.5, 4.5, 2.5):   # separators between kinds of system
    ax.axhline(y, color=S.GRID, lw=0.6)
handles = [plt.Line2D([], [], **S.marker(m, markersize=4.5), label=S.SERIES[m]["label"]) for m in methods]
fig.legend(handles=handles, loc="lower center", ncol=2, columnspacing=1.2, handletextpad=0.3, bbox_to_anchor=(0.55, -0.005))
fig.tight_layout(pad=0.3, rect=(0, 0.09, 1, 1))
fig.savefig(HERE / "fig_main.pdf"); fig.savefig(HERE / "fig_main.png", dpi=150)
print("wrote fig_main.pdf")
