"""Appendix figure: layer sweep (top) and SAE-width sweep (bottom) on Common-KG and MultiFarm,
Gemma-2-9B with native views; same SAE files as the miniF2F sweep."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import style as S

S.setup()
HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE.parent / "data" / "sweep_kg.csv")))
R = {(r["task"], r["config"]): r for r in rows}
layers = [5, 9, 10, 15, 20, 25, 30, 31, 35, 40]
widths = [("16k", 16384, "L20_w16k"), ("32k", 32768, "L20_w32k"), ("65k", 65536, "L20_w65k"),
          ("131k", 131072, "L20_w131k"), ("524k", 524288, "L20_w524k"), ("1M", 1048576, "L20_w1m")]
methods = ["dense", "dense_pc1", "sae_mean", "sae_idf"]
tasks = [("NELL->DBpedia", "NELL → DBpedia (129 classes)"), ("YAGO->Wikidata", "YAGO → Wikidata (304 classes)"),
         ("MultiFarm", "MultiFarm zh/ru/ar → en (mean of 3)")]

fig, axes = plt.subplots(2, 3, figsize=(S.TEXT_IN, 3.6), sharey="row")
for col, (task, title) in enumerate(tasks):
    ax = axes[0, col]
    for m in methods:
        ax.plot(layers, [float(R[(task, f"L{l}_w131k")][m]) for l in layers], **S.style(m))
    ax.set_title(title, color=S.INK, loc="left")
    ax.set_xlabel("Gemma-2-9B layer (131k-feature SAE)"); ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
    ax.axvline(20, color=S.GRID, lw=0.8, zorder=0)
    ax = axes[1, col]
    xs = [w for _, w, _ in widths]
    for m in methods:
        ax.plot(xs, [float(R[(task, c)][m]) for _, _, c in widths], **S.style(m))
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([n for n, _, _ in widths]); ax.minorticks_off()
    ax.set_xlabel("SAE width at layer 20 (canonical $L_0$)")
for ax in axes[0]:
    ax.set_ylim(0.1, 1.0)
for ax in axes[1]:
    ax.set_ylim(0.1, 1.0)
for ax in axes[:, 0]:
    ax.set_ylabel("MRR")
for ax in axes.flat:
    ax.grid(axis="x", visible=False)
handles = [plt.Line2D([], [], **S.style(m), label=S.SERIES[m]["label"]) for m in methods]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), columnspacing=1.5, handlelength=2.2)
fig.tight_layout(pad=0.3, rect=(0, 0.06, 1, 1))
fig.savefig(HERE / "fig_sweep_kg.pdf"); fig.savefig(HERE / "fig_sweep_kg.png", dpi=150)
print("wrote fig_sweep_kg.pdf")
