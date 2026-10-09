"""Figure 2: layer sweep (top) and SAE-width sweep at matched sparsity (bottom),
for miniF2F Lean 4 -> HOL Light and -> Metamath. The x-axis of each row names the ablated factor;
the settings held fixed are in the caption."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import style as S

S.setup()
HERE = Path(__file__).resolve().parent
A = {(r["task"], r["config"]): r for r in csv.DictReader(open(HERE.parent / "data" / "ablation_minif2f.csv"))}
layers = [5, 9, 10, 15, 20, 25, 30, 31, 35, 40]
widths = [("16k", 16384, "L20_w16k_l0_58"), ("32k", 32768, "L20_w32k"), ("65k", 65536, "L20_w65k"),
          ("131k", 131072, "L20_w131k_l0_53"), ("262k", 262144, "L20_w262k_l0_50"),
          ("524k", 524288, "L20_w524k"), ("1M", 1048576, "L20_w1m_l0_57")]
methods = ["dense", "dense_pc1", "sae_mean", "sae_idf"]
tasks = [("HOLLight", "Lean 4 → HOL Light (317 problems)"), ("Metamath", "Lean 4 → Metamath (482 problems)")]

fig, axes = plt.subplots(2, 2, figsize=(S.TEXT_IN, 2.62), sharey=True)
for col, (task, title) in enumerate(tasks):
    ax = axes[0, col]
    for m in methods:
        ax.plot(layers, [float(A[(task, f"L{l}_w131k")][m]) for l in layers], **S.style(m))
    ax.set_title(title, color=S.INK, loc="left")
    ax.set_xlabel(r"Ablated factor: layer $\ell$"); ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])   # data at 9 and 31 too
    ax.axvline(20, color=S.GRID, lw=0.8, zorder=0)
    ax = axes[1, col]
    xs = [w for _, w, _ in widths]
    for m in methods:
        ax.plot(xs, [float(A[(task, c)][m]) for _, _, c in widths], **S.style(m))
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([n for n, _, _ in widths]); ax.minorticks_off()
    ax.set_xlabel(r"Ablated factor: SAE width $F$ (features)")
for ax in axes[:, 0]:
    ax.set_ylabel("MRR")
for ax in axes.flat:
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.grid(axis="x", visible=False)
# direct labels on the right-most points of the top-left panel, legend once
handles = [plt.Line2D([], [], **S.style(m), label=S.SERIES[m]["label"]) for m in methods]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.5, handlelength=2.2, frameon=False)
fig.tight_layout(pad=0.3, h_pad=0.9, w_pad=1.2, rect=(0, 0.075, 1, 1))
fig.savefig(HERE / "fig_ablation.pdf"); fig.savefig(HERE / "fig_ablation.png", dpi=150)
print("wrote fig_ablation.pdf")
