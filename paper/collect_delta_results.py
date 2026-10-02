#!/usr/bin/env python3
"""Collect the 2026-10-01/02 Delta runs into paper tables and data files:
  * Llama-3.1-8B second-model-family main table      -> data/llama_main.csv, tables/tab_llama_main.tex
  * Gemma-2-9B layer/width sweep on Common-KG, MultiFarm -> data/sweep_kg.csv (figure: figures/fig_sweep_kg.py)
  * View-sampling seed study (Common-KG, Valentine)   -> data/seeds.csv, tables/tab_seeds.tex
Runs that have not finished are skipped, so the script can be re-run as results arrive."""
import csv
import json
from pathlib import Path

M = Path("/projects/biro/xiaocong/main_table")
HERE = Path(__file__).resolve().parent
(HERE / "data").mkdir(exist_ok=True); (HERE / "tables").mkdir(exist_ok=True)
METHODS = ("dense", "dense_pc1", "sae_mean", "sae_idf")


def rank_summary(path):
    if not path.exists():
        return None
    d = {r["method"]: float(r["MRR"]) for r in csv.DictReader(open(path))}
    return {"dense": d["dense_mean"], "dense_pc1": d["dense_pc1"], "sae_mean": d["sae_mean"], "sae_idf": d["sae_idf"],
            "strings": max(d.get("lex_label", 0), d.get("lex_context", 0)), "n": int(next(csv.DictReader(open(path)))["n"])}


def xl_scores(path, key="MRR"):
    if not path.exists():
        return None
    m = json.load(open(path))["metrics"]
    k = "MRR_first_hit" if "MRR_first_hit" in m["dense"] and key == "MRR" and "minif2f" in str(path) else key
    return {x: m[x][k] for x in METHODS}


def valentine_scores(path, key):
    if not path.exists():
        return None
    a = json.load(open(path))["macro_average"]
    return {x: a[x][key] for x in METHODS}


gemma = {r["task"] + "|" + r["metric"]: r for r in csv.DictReader(open(HERE / "data" / "all_results.csv")) if r["group"] == "main"}
F = lambda r, k: float(r[k]) if r.get(k) else None

# ------------------------------------------------------------------ Llama main table
rows = []
def add(task, metric, gkey, llama):
    if llama is None:
        return
    g = gemma[gkey]
    rows.append(dict(task=task, metric=metric, strings=F(g, "strings"), gemma_dense=F(g, "dense"), gemma_idf=F(g, "sae_idf"),
                     llama_dense=llama["dense"], llama_pc1=llama["dense_pc1"], llama_mean=llama["sae_mean"], llama_idf=llama["sae_idf"]))

for t, name in (("Isabelle", "Lean 4 $\\to$ Isabelle"), ("Metamath", "Lean 4 $\\to$ Metamath"), ("HOLLight", "Lean 4 $\\to$ HOL Light")):
    gk = {"Isabelle": "miniF2F Lean4->Isabelle", "Metamath": "miniF2F Lean4->Metamath", "HOLLight": "miniF2F Lean4->HOL Light"}[t]
    add(name, "MRR", gk + "|MRR", xl_scores(M / f"ablation/{t}/llama_L15_w131k/eval/scores.json"))
xl = []
for L in ("Java", "Cpp", "Python", "Csharp", "Javascript", "PHP", "C"):
    s = xl_scores(M / f"xlcost_llama/{L}_program/eval/scores.json")
    if s:
        xl.append((L, s, xl_scores(M / f"xlcost_llama/{L}_program/eval/scores.json", "Precision@6")))
if len(xl) == 7:
    g = [gemma[f"XLCoST {L}|MRR"] for L, _, _ in xl]
    mean = lambda key, i: sum(float(x[i][key]) for x in xl) / 7
    rows.append(dict(task="XLCoST, 7 languages", metric="MRR", strings=sum(F(r, "strings") for r in g) / 7, gemma_dense=sum(F(r, "dense") for r in g) / 7,
                     gemma_idf=sum(F(r, "sae_idf") for r in g) / 7, llama_dense=mean("dense", 1), llama_pc1=mean("dense_pc1", 1),
                     llama_mean=mean("sae_mean", 1), llama_idf=mean("sae_idf", 1)))
    g6 = [gemma[f"XLCoST {L}|Precision@6"] for L, _, _ in xl]
    rows.append(dict(task="XLCoST, 7 languages", metric="P@6", strings=sum(F(r, "strings") for r in g6) / 7, gemma_dense=sum(F(r, "dense") for r in g6) / 7,
                     gemma_idf=sum(F(r, "sae_idf") for r in g6) / 7, llama_dense=mean("dense", 2), llama_pc1=mean("dense_pc1", 2),
                     llama_mean=mean("sae_mean", 2), llama_idf=mean("sae_idf", 2)))
else:
    print(f"XLCoST Llama: {len(xl)} of 7 languages done")
add("Valentine columns", "F1", "Valentine (551 pairs)|F1Score", valentine_scores(M / "valentine_column_llama/eval/scores.json", "F1Score"))
add("Valentine columns", "MRR", "Valentine (551 pairs)|MeanReciprocalRank", valentine_scores(M / "valentine_column_llama/eval/scores.json", "MeanReciprocalRank"))
add("NELL $\\to$ DBpedia", "MRR", "Common-KG NELL->DBpedia|MRR", rank_summary(M / "commonkg/native_scored_nell_dbpedia/llama_L15_w131k/analysis_all/ranking_summary.csv"))
add("YAGO $\\to$ Wikidata", "MRR", "Common-KG YAGO->Wikidata|MRR", rank_summary(M / "commonkg/native_scored_yago_wikidata/llama_L15_w131k/analysis_all/ranking_summary.csv"))
for p, name, gk in (("cn-en", "MultiFarm zh $\\to$ en", "Chinese"), ("en-ru", "MultiFarm ru $\\to$ en", "Russian"), ("ar-en", "MultiFarm ar $\\to$ en", "Arabic")):
    add(name, "MRR", f"MultiFarm {gk}->English|MRR", rank_summary(M / f"multifarm/native_scored/llama_L15_w131k/analysis_views_{p}/ranking_summary.csv"))
appendix = []
for name, path, gk in (("Bio-ML valid", "bioml/native_scored/llama_L15_w131k/analysis_views_valid", "Bio-ML NCIT->DOID valid"),
                       ("Bio-ML train", "bioml/native_scored/llama_L15_w131k/analysis_views_train", "Bio-ML NCIT->DOID train"),
                       ("Anatomy", "anatomy/native_scored/llama_L15_w131k/analysis_all", "Anatomy mouse->human")):
    s = rank_summary(M / path / "ranking_summary.csv")
    if s:
        appendix.append((name, s))

if rows:
    with open(HERE / "data" / "llama_main.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    fmt = lambda v: f"{v:.3f}" if v is not None else "--"
    lines = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Gemma-2-9B, L20} & \multicolumn{4}{c}{Llama-3.1-8B, L15} \\", r"\cmidrule(lr){4-5}\cmidrule(lr){6-9}",
             r"Task & Metric & Strings & Dense & \textbf{SAE-IDF} & Dense & $-$PC1 & SAE unw. & \textbf{SAE-IDF} \\", r"\midrule"]
    for r in rows:
        ll = {k: r[k] for k in ("llama_dense", "llama_pc1", "llama_mean", "llama_idf")}
        best = max(ll, key=ll.get)
        cells = [(r"\textbf{" + fmt(v) + "}") if k == best else fmt(v) for k, v in ll.items()]
        lines.append(f"{r['task']} & {r['metric']} & {fmt(r['strings'])} & {fmt(r['gemma_dense'])} & {fmt(r['gemma_idf'])} & " + " & ".join(cells) + r" \\")
    if appendix:
        lines.append(r"\midrule")
        for name, s in appendix:
            ll = {k: s[k] for k in METHODS}; best = max(ll, key=ll.get)
            cells = [(r"\textbf{" + fmt(v) + "}") if k == best else fmt(v) for k, v in ll.items()]
            gk = {"Bio-ML valid": "Bio-ML NCIT->DOID valid", "Bio-ML train": "Bio-ML NCIT->DOID train", "Anatomy": "Anatomy mouse->human"}[name]
            ga = next(r for r in csv.DictReader(open(HERE / "data" / "all_results.csv")) if r["task"] == gk)
            lines.append(f"{name} (appendix task) & MRR & {fmt(F(ga, 'strings'))} & {fmt(F(ga, 'dense'))} & {fmt(F(ga, 'sae_idf'))} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "tab_llama_main.tex").write_text("\n".join(lines) + "\n")
    n_beat = sum(r["llama_idf"] > r["llama_dense"] for r in rows); n_pc1 = sum(r["llama_idf"] > r["llama_pc1"] for r in rows)
    print(f"Llama main table: {len(rows)} rows; SAE-IDF > dense in {n_beat}, > PC1 in {n_pc1}")
    for r in rows:
        print(f"  {r['task']:24s} {r['metric']:4s} llama dense {r['llama_dense']:.3f} pc1 {r['llama_pc1']:.3f} mean {r['llama_mean']:.3f} idf {r['llama_idf']:.3f} | gemma idf {r['gemma_idf']:.3f}")

# ------------------------------------------------------------------ KG / MultiFarm sweep
sweep = []
for store, name, specs in (("commonkg/native_scored_nell_dbpedia", "NELL->DBpedia", ["all"]),
                           ("commonkg/native_scored_yago_wikidata", "YAGO->Wikidata", ["all"]),
                           ("multifarm/native_scored", "MultiFarm", ["views_cn-en", "views_en-ru", "views_ar-en"])):
    for cfg_dir in sorted((M / store / "sweep").glob("*")) if (M / store / "sweep").exists() else []:
        vals = [rank_summary(cfg_dir / f"analysis_{sp}" / "ranking_summary.csv") for sp in specs]
        if any(v is None for v in vals):
            continue
        avg = {k: sum(v[k] for v in vals) / len(vals) for k in METHODS}
        cfg = cfg_dir.name  # L5_w131k, L20_w16k, ...
        layer = int(cfg[1:cfg.index("_")]); width = cfg.split("_w")[1]
        sweep.append(dict(task=name, config=cfg, layer=layer, width=width, **avg))
if sweep:
    with open(HERE / "data" / "sweep_kg.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)
    print(f"sweep: {len(sweep)} (task, config) results; SAE-IDF > dense in {sum(r['sae_idf'] > r['dense'] for r in sweep)}, > PC1 in {sum(r['sae_idf'] > r['dense_pc1'] for r in sweep)}")

# ------------------------------------------------------------------ seeds
seeds = []
for c, name in (("nell_dbpedia", "NELL->DBpedia"), ("yago_wikidata", "YAGO->Wikidata")):
    base = rank_summary(M / f"commonkg/native_scored_{c}/L20_w131k_l0_114/analysis/ranking_summary.csv")
    seeds.append(dict(task=name, seed=0, **{k: base[k] for k in METHODS}))
    for s in (1, 2, 3):
        r = rank_summary(M / f"commonkg/seed_study/seed{s}_{c}/analysis/ranking_summary.csv")
        if r:
            seeds.append(dict(task=name, seed=s, **{k: r[k] for k in METHODS}))
for metric, key in (("F1", "F1Score"), ("MRR", "MeanReciprocalRank")):
    base = valentine_scores(M / "valentine_column/eval/scores.json", key)
    seeds.append(dict(task=f"Valentine ({metric})", seed=42, **base))
    for s in (7, 13):
        r = valentine_scores(M / f"valentine_column_seed{s}/eval/scores.json", key)
        if r:
            seeds.append(dict(task=f"Valentine ({metric})", seed=s, **r))
with open(HERE / "data" / "seeds.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(seeds[0].keys())); w.writeheader(); w.writerows(seeds)
import statistics as st
lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule", r"Task & Seeds & Dense & $-$PC1 & SAE unw. & \textbf{SAE-IDF} \\", r"\midrule"]
for task in dict.fromkeys(r["task"] for r in seeds):
    rs = [r for r in seeds if r["task"] == task]
    if len(rs) < 2:
        continue
    cell = lambda k: f"{st.mean(r[k] for r in rs):.3f} $\\pm$ {(max(r[k] for r in rs) - min(r[k] for r in rs)) / 2:.3f}"
    tname = task.replace("->", " $\\to$ ")
    lines.append(f"{tname} & {len(rs)} & {cell('dense')} & {cell('dense_pc1')} & {cell('sae_mean')} & {cell('sae_idf')} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "tables" / "tab_seeds.tex").write_text("\n".join(lines) + "\n")
print("seeds:", len(seeds), "rows")
