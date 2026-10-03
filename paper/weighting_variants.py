#!/usr/bin/env python3
"""Which feature weighting? Collect, from the existing evaluation outputs, the alternatives
that every evaluator already computes next to sae_idf:
  sae_mean            unweighted sparse code
  sae_idf             idf over the matched collection (the paper's method)
  sae_idf_universal   idf from Neuronpedia's Pile activation densities (no collection needed)
  sae_info            information weighting (log of in-symbol vs. background firing rate)
  sae_logodds         log-odds weighting
Writes data/weightings.csv and tables/tab_weightings.tex."""
import csv
import json
from pathlib import Path

M = Path("/projects/biro/xiaocong/main_table")
HERE = Path(__file__).resolve().parent
W = ["sae_mean", "sae_idf", "sae_idf_universal", "sae_info", "sae_logodds"]


def xl(path, key):
    m = json.load(open(path))["metrics"]
    return {w: m[w][key] for w in W if w in m}


def onto(path):
    d = {r["method"]: float(r["MRR"]) for r in csv.DictReader(open(path))}
    return {w: d[w] for w in W if w in d}


def val(path, key):
    a = json.load(open(path))["macro_average"]
    return {w: a[w][key] for w in W if w in a}


rows = [
    ("Lean 4 $\\to$ Isabelle", "MRR", xl(M / "minif2f/lean4_Isabelle/eval/scores.json", "MRR_first_hit")),
    ("Lean 4 $\\to$ Metamath", "MRR", xl(M / "minif2f/lean4_Metamath/eval/scores.json", "MRR_first_hit")),
    ("Lean 4 $\\to$ HOL Light", "MRR", xl(M / "minif2f/lean4_HOLLight/eval/scores.json", "MRR_first_hit")),
]
xls = [xl(M / f"xlcost/{L}_program/eval/scores.json", "MRR") for L in ("Java", "Cpp", "Python", "Csharp", "Javascript", "PHP", "C")]
rows.append(("XLCoST, 7 languages", "MRR", {w: sum(x[w] for x in xls) / len(xls) for w in W if all(w in x for x in xls)}))
rows += [
    ("Valentine columns", "F1", val(M / "valentine_column/eval/scores.json", "F1Score")),
    ("Valentine columns", "MRR", val(M / "valentine_column/eval/scores.json", "MeanReciprocalRank")),
    ("NELL $\\to$ DBpedia", "MRR", onto(M / "commonkg/native_scored_nell_dbpedia/L20_w131k_l0_114/analysis/ranking_summary.csv")),
    ("YAGO $\\to$ Wikidata", "MRR", onto(M / "commonkg/native_scored_yago_wikidata/L20_w131k_l0_114/analysis/ranking_summary.csv")),
    ("MultiFarm zh $\\to$ en", "MRR", onto(M / "multifarm/native_scored/L20_w131k_l0_114/analysis_views_cn-en/ranking_summary.csv")),
    ("MultiFarm ru $\\to$ en", "MRR", onto(M / "multifarm/native_scored/L20_w131k_l0_114/analysis_views_en-ru/ranking_summary.csv")),
    ("MultiFarm ar $\\to$ en", "MRR", onto(M / "multifarm/native_scored/L20_w131k_l0_114/analysis_views_ar-en/ranking_summary.csv")),
    ("Bio-ML valid", "MRR", onto(M / "bioml/native_scored/L20_w131k_l0_114/analysis_views_valid/ranking_summary.csv")),
    ("Anatomy", "MRR", onto(M / "anatomy/native_scored/L20_w131k_l0_114/analysis/ranking_summary.csv")),
]
with open(HERE / "data" / "weightings.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["task", "metric"] + W)
    for t, m, d in rows:
        w.writerow([t, m] + [f"{d[k]:.4f}" if k in d else "" for k in W])
lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Task & Metric & Unweighted & \textbf{idf (collection)} & idf (Pile density) & Information & Log-odds \\", r"\midrule"]
wins = {k: 0 for k in W}; n = 0
for t, m, d in rows:
    best = max((k for k in W if k in d), key=lambda k: d[k]); wins[best] += 1; n += 1
    cells = [(r"\textbf{" + f"{d[k]:.3f}" + "}") if k == best else (f"{d[k]:.3f}" if k in d else "--") for k in W]
    lines.append(f"{t} & {m} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "tables" / "tab_weightings.tex").write_text("\n".join(lines) + "\n")
print("rows", n, "best counts", wins)
for t, m, d in rows:
    print(f"{t:24s} {m:4s} " + " ".join(f"{k}={d[k]:.3f}" for k in W if k in d))
