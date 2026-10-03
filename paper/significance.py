#!/usr/bin/env python3
"""Paired bootstrap over queries: is SAE-IDF's gain over each baseline more than noise?

For every main-table task we take the per-query score that the task's metric averages
(reciprocal rank of the gold counterpart; for Valentine the per-pair F1 and MRR) under
SAE-IDF and under a baseline, resample queries with replacement 10,000 times, and report
the mean difference, its 95% interval and the one-sided bootstrap p-value P(diff <= 0),
plus the sign-test counts. Writes data/significance.csv and tables/tab_significance.tex.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

M = Path("/projects/biro/xiaocong/main_table")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/u/xiaocong/Symbolic_Merge/tools")
B, SEED = 10000, 0
rng = np.random.default_rng(SEED)


def paired(a, b):
    """a, b: per-query scores (method, baseline). Returns stats of mean(a-b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    boot = d[idx].mean(axis=1)
    return dict(n=n, mean_a=a.mean(), mean_b=b.mean(), diff=d.mean(),
                ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                p_boot=float((boot <= 0).mean()), wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                ties=int((d == 0).sum()))


# ------------------------------------------------------------------ XLCoST-format tasks
def read_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def official_rr(ranked, gold_prefix):
    """XLCoST's per-query MRR: mean of 1/rank over ALL relevant hits in the (top-100) list, 0 if none.
    For miniF2F each problem has exactly one relevant row, so this is the gold's reciprocal rank."""
    hits = [1.0 / r for r, a in enumerate(ranked, 1) if a.split("/")[0] == gold_prefix]
    return float(np.mean(hits)) if hits else 0.0


def rr_from_predictions(pred_file, answers):
    out = {}
    for l in open(pred_file):
        js = json.loads(l)
        out[js["url"]] = official_rr(js["answers"], answers[js["url"]].split("/")[0])
    return out


def lexical_rr(rows, method, topk=100):
    """Per-query official metric of the string baselines, scored exactly as tools/lexical_baseline.py does."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import lexical_baseline as LB
    q = [" ".join(r["docstring_tokens"]) for r in rows]
    c = [" ".join(r.get("code_tokens") or r["function_tokens"]) for r in rows]
    if method == "bm25_word":
        tok = lambda s: re.findall(r"[^\s()\[\]{},:;\"`]+", s)
        S = LB.bm25_scores([tok(s) for s in q], [tok(s) for s in c])
    else:
        vec = (TfidfVectorizer(token_pattern=r"[^\s()\[\]{},:;\"`]+", sublinear_tf=True) if method == "tfidf_word"
               else TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        vec.fit(q + c)
        S = (vec.transform(q) @ vec.transform(c).T).toarray()
    idx = [r["idx"] for r in rows]
    out = {}
    for i, r in enumerate(rows):
        order = np.argsort(-S[i], kind="stable")[:topk]
        out[r["url"]] = official_rr([idx[j] for j in order], r["idx"].split("/")[0])
    return out


def xlcost_like(run_dir, lex_method):
    cfg = json.loads((run_dir / "run_config.json").read_text())
    dataset = cfg["dataset_file"].replace("/srv/local/xy51/symbolic_merge/data", "/projects/biro/xiaocong/data")
    rows = read_rows(dataset)
    answers = {r["url"]: r["idx"] for r in rows}
    urls = [r["url"] for r in rows]
    per = {m: rr_from_predictions(run_dir / "eval" / "predictions" / f"predictions_{m}.jsonl", answers)
           for m in ("dense", "dense_pc1", "sae_mean", "sae_idf")}
    per["strings"] = lexical_rr(rows, lex_method)
    return {m: [per[m][u] for u in urls] for m in per}


# ------------------------------------------------------------------ ontology / KG tasks
def ranks_csv(path):
    rows = list(csv.DictReader(open(path)))
    cols = {"dense": "dense_mean_rank", "dense_pc1": "dense_pc1_rank", "sae_mean": "sae_mean_rank", "sae_idf": "sae_idf_rank"}
    per = {m: [1.0 / float(r[c]) for r in rows] for m, c in cols.items()}
    lex = {c: [1.0 / float(r[c]) for r in rows] for c in ("lex_label_rank", "lex_context_rank")}
    per["strings"] = max(lex.values(), key=lambda v: np.mean(v))   # the better string scorer, as in the main table
    return per


# ------------------------------------------------------------------ Valentine
def valentine(run_dir, metric):
    rows = list(csv.DictReader(open(run_dir / "eval" / "per_pair_metrics.csv")))
    return {m: [float(r[f"{m}__{metric}"]) for r in rows] for m in ("dense", "dense_pc1", "sae_mean", "sae_idf")}


def tasks_for(runs):
    if runs == "gemma":
        return [
            ("Lean 4 $\\to$ Isabelle", "MRR", lambda: xlcost_like(M / "minif2f/lean4_Isabelle", "bm25_word")),
            ("Lean 4 $\\to$ Metamath", "MRR", lambda: xlcost_like(M / "minif2f/lean4_Metamath", "tfidf_word")),
            ("Lean 4 $\\to$ HOL Light", "MRR", lambda: xlcost_like(M / "minif2f/lean4_HOLLight", "tfidf_char3-5")),
        ] + [
            (f"XLCoST {name}", "MRR", (lambda L=L: xlcost_like(M / f"xlcost/{L}_program", "tfidf_word")))
            for L, name in (("Java", "Java"), ("Cpp", "C++"), ("Python", "Python"), ("Csharp", "C\\#"), ("Javascript", "JavaScript"), ("PHP", "PHP"), ("C", "C"))
        ] + [
            ("Valentine (F1)", "F1", lambda: valentine(M / "valentine_column", "F1Score")),
            ("Valentine (MRR)", "MRR", lambda: valentine(M / "valentine_column", "MeanReciprocalRank")),
            ("NELL $\\to$ DBpedia", "MRR", lambda: ranks_csv(M / "commonkg/native_scored_nell_dbpedia/L20_w131k_l0_114/analysis/per_query_ranks.csv")),
            ("YAGO $\\to$ Wikidata", "MRR", lambda: ranks_csv(M / "commonkg/native_scored_yago_wikidata/L20_w131k_l0_114/analysis/per_query_ranks.csv")),
            ("MultiFarm zh $\\to$ en", "MRR", lambda: ranks_csv(M / "multifarm/native_scored/L20_w131k_l0_114/analysis_views_cn-en/per_query_ranks.csv")),
            ("MultiFarm ru $\\to$ en", "MRR", lambda: ranks_csv(M / "multifarm/native_scored/L20_w131k_l0_114/analysis_views_en-ru/per_query_ranks.csv")),
            ("MultiFarm ar $\\to$ en", "MRR", lambda: ranks_csv(M / "multifarm/native_scored/L20_w131k_l0_114/analysis_views_ar-en/per_query_ranks.csv")),
            ("Bio-ML valid", "MRR", lambda: ranks_csv(M / "bioml/native_scored/L20_w131k_l0_114/analysis_views_valid/per_query_ranks.csv")),
            ("Bio-ML train", "MRR", lambda: ranks_csv(M / "bioml/native_scored/L20_w131k_l0_114/analysis_views_train/per_query_ranks.csv")),
            ("Anatomy", "MRR", lambda: ranks_csv(M / "anatomy/native_scored/L20_w131k_l0_114/analysis/per_query_ranks.csv")),
        ]
    # Llama-3.1-8B, layer 15, Llama Scope 131k
    return [
        ("Lean 4 $\\to$ Isabelle", "MRR", lambda: xlcost_like(M / "ablation/Isabelle/llama_L15_w131k", "bm25_word")),
        ("Lean 4 $\\to$ Metamath", "MRR", lambda: xlcost_like(M / "ablation/Metamath/llama_L15_w131k", "tfidf_word")),
        ("Lean 4 $\\to$ HOL Light", "MRR", lambda: xlcost_like(M / "ablation/HOLLight/llama_L15_w131k", "tfidf_char3-5")),
    ] + [
        (f"XLCoST {name}", "MRR", (lambda L=L: xlcost_like(M / f"xlcost_llama/{L}_program", "tfidf_word")))
        for L, name in (("Java", "Java"), ("Cpp", "C++"), ("Python", "Python"), ("Csharp", "C\\#"), ("Javascript", "JavaScript"), ("PHP", "PHP"), ("C", "C"))
    ] + [
        ("Valentine (F1)", "F1", lambda: valentine(M / "valentine_column_llama", "F1Score")),
        ("Valentine (MRR)", "MRR", lambda: valentine(M / "valentine_column_llama", "MeanReciprocalRank")),
        ("NELL $\\to$ DBpedia", "MRR", lambda: ranks_csv(M / "commonkg/native_scored_nell_dbpedia/llama_L15_w131k/analysis_all/per_query_ranks.csv")),
        ("YAGO $\\to$ Wikidata", "MRR", lambda: ranks_csv(M / "commonkg/native_scored_yago_wikidata/llama_L15_w131k/analysis_all/per_query_ranks.csv")),
        ("MultiFarm zh $\\to$ en", "MRR", lambda: ranks_csv(M / "multifarm/native_scored/llama_L15_w131k/analysis_views_cn-en/per_query_ranks.csv")),
        ("MultiFarm ru $\\to$ en", "MRR", lambda: ranks_csv(M / "multifarm/native_scored/llama_L15_w131k/analysis_views_en-ru/per_query_ranks.csv")),
        ("MultiFarm ar $\\to$ en", "MRR", lambda: ranks_csv(M / "multifarm/native_scored/llama_L15_w131k/analysis_views_ar-en/per_query_ranks.csv")),
        ("Bio-ML valid", "MRR", lambda: ranks_csv(M / "bioml/native_scored/llama_L15_w131k/analysis_views_valid/per_query_ranks.csv")),
        ("Bio-ML train", "MRR", lambda: ranks_csv(M / "bioml/native_scored/llama_L15_w131k/analysis_views_train/per_query_ranks.csv")),
        ("Anatomy", "MRR", lambda: ranks_csv(M / "anatomy/native_scored/llama_L15_w131k/analysis_all/per_query_ranks.csv")),
    ]


RUNS = sys.argv[1] if len(sys.argv) > 1 else "gemma"
SUFFIX = "" if RUNS == "gemma" else "_" + RUNS
TASKS = tasks_for(RUNS)

out_rows = []
for task, metric, load in TASKS:
    per = load()
    for base in ("dense", "dense_pc1", "sae_mean", "strings"):
        if base not in per:
            continue
        st = paired(per["sae_idf"], per[base])
        out_rows.append(dict(task=task, metric=metric, baseline=base, **st))
        print(f"{task:24s} vs {base:9s} n={st['n']:5d} {st['mean_a']:.3f}-{st['mean_b']:.3f} = {st['diff']:+.3f} "
              f"[{st['ci_lo']:+.3f},{st['ci_hi']:+.3f}] p={st['p_boot']:.4f} W/L/T={st['wins']}/{st['losses']}/{st['ties']}")

(HERE / "data").mkdir(exist_ok=True)
with open(HERE / "data" / f"significance{SUFFIX}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)


def cell(task, base):
    r = next((r for r in out_rows if r["task"] == task and r["baseline"] == base), None)
    if r is None:
        return "--"
    p = "$<$.001" if r["p_boot"] < 0.001 else ("$<$.01" if r["p_boot"] < 0.01 else f"{r['p_boot']:.2f}")
    return f"{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] & {p}"


lines = [r"\begin{tabular}{@{}lrllllll@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{vs.\ Dense} & \multicolumn{2}{c}{vs.\ Dense$_{-\mathrm{PC1}}$} & \multicolumn{2}{c}{vs.\ Strings} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Task & $n$ & $\Delta$ [95\% CI] & $p$ & $\Delta$ [95\% CI] & $p$ & $\Delta$ [95\% CI] & $p$ \\", r"\midrule"]
for task, metric, _ in TASKS:
    n = next(r for r in out_rows if r["task"] == task)["n"]
    lines.append(f"{task} & {n:,} & {cell(task, 'dense')} & {cell(task, 'dense_pc1')} & {cell(task, 'strings')} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "tables" / f"tab_significance{SUFFIX}.tex").write_text("\n".join(lines) + "\n")
print("wrote", HERE / "tables" / f"tab_significance{SUFFIX}.tex")
