#!/usr/bin/env python3
"""Paired bootstraps for the pre-registered extras (paper/PREREG_extras_2026-10-04.md).

For every main-table row, the per-query score the metric averages (reciprocal rank of the gold
counterpart; Valentine: per-pair F1 and MRR) under SAE-IDF is compared with
  S1  the random 131k dictionary + idf (and the random dictionary unweighted),
  S2  each stronger dense control (top-36 PCs removed, z-score, ZCA, dense-idf, sparse dense-idf),
by resampling queries 10,000 times: mean difference SAE-IDF minus the comparison, 95% interval,
one-sided bootstrap p-value P(diff <= 0), sign counts. XLCoST is bootstrapped per language.
Rows whose inputs are not on disk yet are skipped, so the script can run while jobs finish.

Writes data/significance_extras.csv and tables/tab_significance_extras.tex, and prints the
pre-registered counts (S1: rows where SAE-IDF leads random+idf by >= 0.01 with p < 0.01;
S2: rows where a control is within noise of, or ahead of, SAE-IDF).
"""
import csv
import json
from pathlib import Path

import numpy as np

M = Path("/projects/biro/xiaocong/main_table")
X = M / "extras"
RD = M / "random_dict/seed0"
HERE = Path(__file__).resolve().parent
B, SEED = 10000, 0
rng = np.random.default_rng(SEED)
CTRLS = ["pc36", "zscore", "zca", "dense_idf", "dense_topk_idf"]
LANGS = ["Java", "Cpp", "Python", "Csharp", "Javascript", "PHP", "C"]


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    idx = rng.integers(0, len(d), size=(B, len(d)))
    boot = d[idx].mean(axis=1)
    return dict(n=len(d), mean_a=a.mean(), mean_b=b.mean(), diff=d.mean(), ci_lo=np.percentile(boot, 2.5),
                ci_hi=np.percentile(boot, 97.5), p_boot=float((boot <= 0).mean()),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()))


# ----------------------------------------------------------------------------- per-query readers
def official_rr(ranked, gold_prefix):
    hits = [1.0 / r for r, a in enumerate(ranked, 1) if a.split("/")[0] == gold_prefix]
    return float(np.mean(hits)) if hits else 0.0


def xl_per_query(run_dir, method):
    """url -> official per-query reciprocal rank of one method of an XLCoST-format run."""
    pred = run_dir / "eval" / "predictions" / f"predictions_{method}.jsonl"
    cfg_file = run_dir / "run_config.json"
    if not pred.is_file() or not cfg_file.is_file():
        return None
    dataset = json.loads(cfg_file.read_text())["dataset_file"].replace("/srv/local/xy51/symbolic_merge/data", "/projects/biro/xiaocong/data")
    answers = {r["url"]: r["idx"] for r in (json.loads(l) for l in open(dataset, encoding="utf-8") if l.strip())}
    out = {}
    for l in open(pred):
        js = json.loads(l)
        out[js["url"]] = official_rr(js["answers"], answers[js["url"]].split("/")[0])
    return out


def onto_per_query(csv_path, col="sae_idf_rank"):
    p = Path(csv_path)
    if not p.is_file():
        return None
    return [1.0 / float(r[col]) for r in csv.DictReader(open(p))]


def valentine_per_pair(eval_dir, method, metric):
    p = Path(eval_dir) / "per_pair_metrics.csv"
    if not p.is_file():
        return None
    return {r["pair"]: float(r[f"{method}__{metric}"]) for r in csv.DictReader(open(p))}


def xl_first_hit(run_dir):
    """url -> 1/first-hit rank of SAE-IDF, from its predictions file."""
    pred = run_dir / "eval" / "predictions" / "predictions_sae_idf.jsonl"
    dataset = json.loads((run_dir / "run_config.json").read_text())["dataset_file"].replace("/srv/local/xy51/symbolic_merge/data", "/projects/biro/xiaocong/data")
    answers = {r["url"]: r["idx"] for r in (json.loads(l) for l in open(dataset, encoding="utf-8") if l.strip())}
    out = {}
    for l in open(pred):
        js = json.loads(l)
        g = answers[js["url"]].split("/")[0]
        hit = next((r for r, a in enumerate(js["answers"], 1) if a.split("/")[0] == g), None)
        out[js["url"]] = 1.0 / hit if hit else 0.0
    return out



def control_json(name):
    p = X / "dense_controls_v2" / f"{name}.json"
    if not p.is_file():
        p = X / "dense_controls" / f"{name}.json"
    return json.load(open(p)) if p.is_file() else None


# ----------------------------------------------------------------------------- rows
XL_ROWS = [("Lean 4 $\\to$ Isabelle", M / "minif2f/lean4_Isabelle", RD / "minif2f/lean4_Isabelle", "minif2f_isabelle")]
XL_ROWS += [("Lean 4 $\\to$ Metamath", M / "minif2f/lean4_Metamath", RD / "minif2f/lean4_Metamath", "minif2f_metamath"),
            ("Lean 4 $\\to$ HOL Light", M / "minif2f/lean4_HOLLight", RD / "minif2f/lean4_HOLLight", "minif2f_hollight")]
XL_ROWS += [(f"XLCoST {L}", M / f"xlcost/{L}_program", RD / f"xlcost/{L}_program", f"xlcost_{L}") for L in LANGS]
ONTO_ROWS = [
    ("NELL $\\to$ DBpedia", M / "commonkg/native_scored_nell_dbpedia/L20_w131k_l0_114/analysis/per_query_ranks.csv",
     M / "commonkg/native_scored_nell_dbpedia/random_seed0/analysis_all/per_query_ranks.csv", "commonkg_nell_dbpedia"),
    ("YAGO $\\to$ Wikidata", M / "commonkg/native_scored_yago_wikidata/L20_w131k_l0_114/analysis/per_query_ranks.csv",
     M / "commonkg/native_scored_yago_wikidata/random_seed0/analysis_all/per_query_ranks.csv", "commonkg_yago_wikidata"),
    ("MultiFarm zh $\\to$ en", M / "multifarm/native_scored/L20_w131k_l0_114/analysis_views_cn-en/per_query_ranks.csv",
     M / "multifarm/native_scored/random_seed0/analysis_views_cn-en/per_query_ranks.csv", "multifarm_zh"),
    ("MultiFarm ru $\\to$ en", M / "multifarm/native_scored/L20_w131k_l0_114/analysis_views_en-ru/per_query_ranks.csv",
     M / "multifarm/native_scored/random_seed0/analysis_views_en-ru/per_query_ranks.csv", "multifarm_ru"),
    ("MultiFarm ar $\\to$ en", M / "multifarm/native_scored/L20_w131k_l0_114/analysis_views_ar-en/per_query_ranks.csv",
     M / "multifarm/native_scored/random_seed0/analysis_views_ar-en/per_query_ranks.csv", "multifarm_ar"),
    # biomedical ontologies (S1 run on 2026-10-05 from the cached layer-20 states; no S2 controls for these)
    ("Bio-ML valid", M / "bioml/native_scored/L20_w131k_l0_114/analysis_views_valid/per_query_ranks.csv",
     M / "bioml/native_scored/random_seed0/analysis_views_valid/per_query_ranks.csv", "bioml_valid"),
    ("Bio-ML train", M / "bioml/native_scored/L20_w131k_l0_114/analysis_views_train/per_query_ranks.csv",
     M / "bioml/native_scored/random_seed0/analysis_views_train/per_query_ranks.csv", "bioml_train"),
    ("Anatomy", M / "anatomy/native_scored/L20_w131k_l0_114/analysis/per_query_ranks.csv",
     M / "anatomy/native_scored/random_seed0/analysis/per_query_ranks.csv", "anatomy"),
]

results = []   # (section, row, comparison, stats)


def add(section, row, comp, a, b):
    if a is None or b is None:
        return
    if isinstance(a, dict):
        keys = [k for k in a if k in b]
        a, b = [a[k] for k in keys], [b[k] for k in keys]
    if len(a) != len(b):
        print(f"  skip {row} / {comp}: {len(a)} vs {len(b)} queries"); return
    results.append((section, row, comp, paired(a, b)))


for label, main_run, rand_run, name in XL_ROWS:
    sae = xl_per_query(main_run, "sae_idf")
    add("S1", label, "random + idf", sae, xl_per_query(rand_run, "sae_idf"))
    add("S1", label, "random, unweighted", sae, xl_per_query(rand_run, "sae_mean"))
    j = control_json(name)
    if j and "per_query_rank" in j["metrics"]["pc36"] and sae is not None:
        for c in CTRLS:
            pq = j["metrics"][c]["per_query_rank"]
            # first-hit reciprocal rank of the control, compared with SAE-IDF's official per-query RR
            # (identical for miniF2F; for XLCoST the official RR averages over all relevant hits, so
            #  SAE-IDF is also re-read as first-hit here to keep the comparison like for like)
            sae_fh = xl_first_hit(main_run) if label.startswith("XLCoST") else sae
            add("S2", label, c, sae_fh, {u: (1.0 / r if r else 0.0) for u, r in pq.items()})

for label, main_csv, rand_csv, name in ONTO_ROWS:
    sae = onto_per_query(main_csv)
    add("S1", label, "random + idf", sae, onto_per_query(rand_csv, "sae_idf_rank"))
    add("S1", label, "random, unweighted", sae, onto_per_query(rand_csv, "sae_mean_rank"))
    j = control_json(name)
    if j and "per_query_rank" in j["metrics"]["pc36"]:
        for c in CTRLS:
            add("S2", label, c, sae, [1.0 / r for r in j["metrics"][c]["per_query_rank"]])

for metric, mlabel in (("F1Score", "F1"), ("MeanReciprocalRank", "MRR")):
    sae = valentine_per_pair(M / "valentine_column/eval", "sae_idf", metric)
    add("S1", f"Valentine ({mlabel})", "random + idf", sae, valentine_per_pair(RD / "valentine_column/eval", "sae_idf", metric))
    add("S1", f"Valentine ({mlabel})", "random, unweighted", sae, valentine_per_pair(RD / "valentine_column/eval", "sae_mean", metric))
    for c in CTRLS:
        add("S2", f"Valentine ({mlabel})", c, sae, valentine_per_pair(X / f"dense_controls_valentine/valentine_{c}/eval", "dense", metric))


# ----------------------------------------------------------------------------- outputs
with open(HERE / "data/significance_extras.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["section", "row", "comparison", "n", "sae_idf", "other", "diff", "ci_lo", "ci_hi", "p_boot", "wins", "losses", "ties"])
    for sec, row, comp, s in results:
        w.writerow([sec, row, comp, s["n"], f"{s['mean_a']:.4f}", f"{s['mean_b']:.4f}", f"{s['diff']:.4f}", f"{s['ci_lo']:.4f}", f"{s['ci_hi']:.4f}", f"{s['p_boot']:.4f}", s["wins"], s["losses"], s["ties"]])

names = {"random + idf": "Random + idf", "random, unweighted": "Random, unw.", "pc36": "$-$36 PCs", "zscore": "z-score", "zca": "ZCA", "dense_idf": "dense-idf", "dense_topk_idf": "sparse dense-idf"}
CAP = (r"\caption{Paired bootstrap for the post-hoc controls of Appendix~\ref{app:controls}: \method minus the comparison, "
       r"per row (XLCoST per language), with the 95\% interval and one-sided $p$.}\label{tab:significanceextras}\\")
HEAD = r"Row & Comparison & $n$ & \method & other & $\Delta$ [95\% CI] & $p$ \\"
# a longtable: about 120 rows, more than one page even in \scriptsize (the caption and label live here, not in main.tex)
lines = [r"\begin{longtable}{@{}lllrrrr@{}}", CAP, r"\toprule", HEAD, r"\midrule", r"\endfirsthead",
         r"\multicolumn{7}{@{}l}{\small\emph{Table~\ref{tab:significanceextras}, continued}} \\", r"\toprule", HEAD, r"\midrule", r"\endhead",
         r"\midrule", r"\multicolumn{7}{r@{}}{\small\emph{continued on the next page}} \\", r"\endfoot",
         r"\bottomrule", r"\endlastfoot"]
for sec in ("S1", "S2"):
    sel = [r for r in results if r[0] == sec]
    if not sel:
        continue
    lines.append(r"\multicolumn{7}{@{}l}{\emph{" + ("S1: random dictionary of the same width and sparsity" if sec == "S1" else "S2: stronger dense controls") + r"}} \\")
    for _, row, comp, s in sel:
        p = "$<$0.001" if s["p_boot"] < 0.001 else f"{s['p_boot']:.3f}"
        lines.append(f"{row} & {names[comp]} & {s['n']} & {s['mean_a']:.3f} & {s['mean_b']:.3f} & {s['diff']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] & {p} \\\\")
lines += [r"\end{longtable}"]
(HERE / "tables/tab_significance_extras.tex").write_text("\n".join(lines) + "\n")

s1 = [r for r in results if r[0] == "S1" and r[2] == "random + idf"]
if s1:
    lead = sum(1 for r in s1 if r[3]["diff"] >= 0.01 and r[3]["p_boot"] < 0.01)
    noise = sum(1 for r in s1 if r[3]["ci_lo"] <= 0 or r[3]["diff"] < 0.01)
    print(f"S1: {len(s1)} rows; SAE-IDF significantly ahead of random+idf (>=0.01, p<0.01) on {lead}; random within noise or ahead on {noise}")
for c in CTRLS:
    sel = [r for r in results if r[0] == "S2" and r[2] == c]
    if sel:
        within = sum(1 for r in sel if r[3]["ci_lo"] <= 0)
        print(f"S2 {c:15s}: {len(sel)} rows; within noise of or ahead of SAE-IDF on {within}")
print(f"{len(results)} comparisons written")
