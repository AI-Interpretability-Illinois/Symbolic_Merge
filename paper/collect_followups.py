#!/usr/bin/env python3
"""Collect the 2026-10-02/03 follow-up analyses into paper tables (each skipped if its inputs are missing):
  * XLCoST exclude-self protocol (no free hit)           -> tables/tab_exclude_self.tex
  * embedding-model baseline (bge-m3) vs SAE-IDF         -> tables/tab_embed.tex, data/embed_baseline.csv
  * Llama-3.1-8B layer sweep on Common-KG / MultiFarm    -> data/sweep_kg_llama.csv, tables/tab_llama_kg.tex
  * four-prover merge / transitivity test                -> tables/tab_prover_merge.tex
  * abstention curves summary                            -> tables/tab_abstention.tex
  * explanation-type labeler agreement                   -> data/type_agreement.json (numbers quoted in the text)
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
sys.path.insert(0, "/u/xiaocong/Symbolic_Merge/xlcost")
(HERE / "tables").mkdir(exist_ok=True); (HERE / "data").mkdir(exist_ok=True)
fmt = lambda v, nd=3: f"{v:.{nd}f}" if v is not None else "--"
LANGS = (("Java", "Java"), ("Cpp", "C++"), ("Python", "Python"), ("Csharp", "C\\#"), ("Javascript", "JavaScript"), ("PHP", "PHP"), ("C", "C"))


def bold_best(d, keys):
    present = {k: d[k] for k in keys if d.get(k) is not None}
    best = max(present, key=present.get) if present else None
    return [(r"\textbf{" + fmt(d[k]) + "}") if k == best else fmt(d.get(k)) for k in keys]


# ---------------------------------------------------------------- XLCoST exclude-self
def strings_exclude_self(dataset_file, topk=100):
    """Official per-query MRR / P@6 of word TF-IDF with the query's own row removed from the pool."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from eval_xlcost_sae import calculate_scores, precison_atk
    rows = [json.loads(l) for l in open(dataset_file, encoding="utf-8") if l.strip()]
    q = [" ".join(r["docstring_tokens"]) for r in rows]; c = [" ".join(r.get("code_tokens") or r["function_tokens"]) for r in rows]
    vec = TfidfVectorizer(token_pattern=r"[^\s()\[\]{},:;\"`]+", sublinear_tf=True).fit(q + c)
    S = (vec.transform(q) @ vec.transform(c).T).toarray()
    np.fill_diagonal(S, -np.inf)
    answers = {r["url"]: r["idx"] for r in rows}
    preds = {r["url"]: [rows[j]["idx"] for j in np.argsort(-S[i], kind="stable")[:topk]] for i, r in enumerate(rows)}
    return {**calculate_scores(answers, preds), **precison_atk(answers, preds, 6)}


ex_rows = []
for L, name in LANGS:
    f = M / f"xlcost/{L}_program/eval_exclude_self/scores.json"
    if not f.exists():
        continue
    m = json.load(open(f))["metrics"]
    cfg = json.loads((M / f"xlcost/{L}_program/run_config.json").read_text())
    ds = cfg["dataset_file"].replace("/srv/local/xy51/symbolic_merge/data", "/projects/biro/xiaocong/data")
    cache = HERE / "data" / f"strings_exclude_self_{L}.json"
    if cache.exists():
        st = json.loads(cache.read_text())
    else:
        st = strings_exclude_self(ds); cache.write_text(json.dumps(st))
    ex_rows.append((name, int(json.load(open(f))["queries"]), {"strings": st, **{k: m[k] for k in ("dense", "dense_pc1", "sae_mean", "sae_idf")}}))
if ex_rows:
    lines = [r"\begin{tabular}{@{}lrlrrrrr@{}}", r"\toprule",
             r"Query language & Programs & Metric & Strings & Dense & $-$PC1 & SAE unw. & \textbf{SAE-IDF} \\", r"\midrule"]
    for name, n, d in ex_rows:
        for metric, label in (("MRR", "MRR"), ("Precision@6", "P@6")):
            vals = {k: d[k][metric] for k in d}
            lines.append((f"{name} & {n:,}" if metric == "MRR" else " & ") + f" & {label} & " + " & ".join(bold_best(vals, ["strings", "dense", "dense_pc1", "sae_mean", "sae_idf"])) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "tab_exclude_self.tex").write_text("\n".join(lines) + "\n")
    wins = sum(d["sae_idf"]["MRR"] > d["dense"]["MRR"] for _, _, d in ex_rows); wins_s = sum(d["sae_idf"]["MRR"] > d["strings"]["MRR"] for _, _, d in ex_rows)
    print(f"exclude-self: {len(ex_rows)} languages; SAE-IDF > dense MRR in {wins}, > strings in {wins_s}")
    for name, n, d in ex_rows:
        print(f"  {name:11s} MRR strings {d['strings']['MRR']:.3f} dense {d['dense']['MRR']:.3f} pc1 {d['dense_pc1']['MRR']:.3f} idf {d['sae_idf']['MRR']:.3f} | P@6 strings {d['strings']['Precision@6']:.3f} dense {d['dense']['Precision@6']:.3f} idf {d['sae_idf']['Precision@6']:.3f}")

# ---------------------------------------------------------------- embedding baseline
E = M / "embed_baseline/bge-m3"
gemma = {r["task"] + "|" + r["metric"]: r for r in csv.DictReader(open(HERE / "data" / "all_results.csv")) if r["group"] in ("main", "appendix")}
F = lambda r, k: float(r[k]) if r.get(k) else None
emb_rows = []
def emb(task, label, gkey, metric_key, gmetric="MRR"):
    f = E / f"{task}.json"
    if not f.exists():
        return
    res = json.load(open(f))["metrics"]
    g = gemma.get(gkey)
    emb_rows.append(dict(task=label, metric=gmetric, strings=F(g, "strings") if g else None, gemma_dense=F(g, "dense") if g else None, gemma_idf=F(g, "sae_idf") if g else None,
                         emb=res["dense"][metric_key], emb_centered=res["dense_centered"][metric_key], emb_pc1=res["dense_pc1"][metric_key]))
emb("minif2f_isabelle", "Lean 4 $\\to$ Isabelle", "miniF2F Lean4->Isabelle|MRR", "MRR")
emb("minif2f_metamath", "Lean 4 $\\to$ Metamath", "miniF2F Lean4->Metamath|MRR", "MRR")
emb("minif2f_hollight", "Lean 4 $\\to$ HOL Light", "miniF2F Lean4->HOL Light|MRR", "MRR")
xl = [json.load(open(E / f"xlcost_{L}.json"))["metrics"] for L, _ in LANGS if (E / f"xlcost_{L}.json").exists()]
if len(xl) == 7:
    for mk, gm in (("MRR", "MRR"), ("Precision@6", "Precision@6")):
        gs = [gemma[f"XLCoST {L}|{gm}"] for L, _ in LANGS]
        emb_rows.append(dict(task="XLCoST, 7 languages", metric="MRR" if mk == "MRR" else "P@6", strings=np.mean([F(r, "strings") for r in gs]), gemma_dense=np.mean([F(r, "dense") for r in gs]),
                             gemma_idf=np.mean([F(r, "sae_idf") for r in gs]), emb=np.mean([x["dense"][mk] for x in xl]), emb_centered=np.mean([x["dense_centered"][mk] for x in xl]), emb_pc1=np.mean([x["dense_pc1"][mk] for x in xl])))
if (E / "valentine.json").exists():
    res = json.load(open(E / "valentine.json"))["metrics"]
    for mk, gm, lab in (("F1Score", "F1Score", "F1"), ("MeanReciprocalRank", "MeanReciprocalRank", "MRR")):
        g = gemma["Valentine (551 pairs)|" + gm]
        emb_rows.append(dict(task="Valentine columns", metric=lab, strings=None, gemma_dense=F(g, "dense"), gemma_idf=F(g, "sae_idf"), emb=res["dense"][mk], emb_centered=res["dense_centered"][mk], emb_pc1=res["dense_pc1"][mk]))
emb("commonkg_nell_dbpedia", "NELL $\\to$ DBpedia", "Common-KG NELL->DBpedia|MRR", "MRR")
emb("commonkg_yago_wikidata", "YAGO $\\to$ Wikidata", "Common-KG YAGO->Wikidata|MRR", "MRR")
emb("multifarm_zh", "MultiFarm zh $\\to$ en", "MultiFarm Chinese->English|MRR", "MRR")
emb("multifarm_ru", "MultiFarm ru $\\to$ en", "MultiFarm Russian->English|MRR", "MRR")
emb("multifarm_ar", "MultiFarm ar $\\to$ en", "MultiFarm Arabic->English|MRR", "MRR")
emb("bioml_valid", "Bio-ML valid", "Bio-ML NCIT->DOID valid|MRR", "MRR")
emb("bioml_train", "Bio-ML train", "Bio-ML NCIT->DOID train|MRR", "MRR")
emb("anatomy", "Anatomy", "Anatomy mouse->human|MRR", "MRR")
if emb_rows:
    with open(HERE / "data" / "embed_baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(emb_rows[0].keys())); w.writeheader(); w.writerows(emb_rows)
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Gemma-2-9B} & \multicolumn{3}{c}{BGE-M3 embeddings} \\", r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"Task & Metric & Strings & Dense & \textbf{SAE-IDF} & plain & centred & $-$PC1 \\", r"\midrule"]
    for r in emb_rows:
        cells = bold_best(r, ["gemma_dense", "gemma_idf", "emb", "emb_centered", "emb_pc1"])
        lines.append(f"{r['task']} & {r['metric']} & {fmt(r['strings'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "tab_embed.tex").write_text("\n".join(lines) + "\n")
    best_emb = lambda r: max(r["emb"], r["emb_centered"], r["emb_pc1"])
    print(f"embedding baseline: {len(emb_rows)} rows; SAE-IDF > best BGE-M3 variant in {sum(r['gemma_idf'] > best_emb(r) for r in emb_rows)}; > plain BGE-M3 in {sum(r['gemma_idf'] > r['emb'] for r in emb_rows)}")
    for r in emb_rows:
        print(f"  {r['task']:24s} {r['metric']:4s} idf {r['gemma_idf']:.3f} | bge {r['emb']:.3f} centred {r['emb_centered']:.3f} pc1 {r['emb_pc1']:.3f}")

# ---------------------------------------------------------------- Llama KG / MultiFarm sweep
def rank_summary(path):
    if not path.exists():
        return None
    d = {r["method"]: float(r["MRR"]) for r in csv.DictReader(open(path))}
    return {"dense": d["dense_mean"], "dense_pc1": d["dense_pc1"], "sae_mean": d["sae_mean"], "sae_idf": d["sae_idf"]}
lk = []
for store, name, specs in (("commonkg/native_scored_nell_dbpedia", "NELL->DBpedia", ["all"]), ("commonkg/native_scored_yago_wikidata", "YAGO->Wikidata", ["all"]),
                           ("multifarm/native_scored", "MultiFarm", ["views_cn-en", "views_en-ru", "views_ar-en"])):
    for cfg_dir in sorted((M / store / "sweep").glob("llama_*")) if (M / store / "sweep").exists() else []:
        vals = [rank_summary(cfg_dir / f"analysis_{sp}" / "ranking_summary.csv") for sp in specs]
        if any(v is None for v in vals):
            continue
        avg = {k: sum(v[k] for v in vals) / len(vals) for k in ("dense", "dense_pc1", "sae_mean", "sae_idf")}
        m = re.match(r"llama_L(\d+)_w(\w+)", cfg_dir.name)
        lk.append(dict(task=name, config=cfg_dir.name, layer=int(m.group(1)), width=m.group(2), **avg))
if lk:
    with open(HERE / "data" / "sweep_kg_llama.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lk[0].keys())); w.writeheader(); w.writerows(lk)
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Task & Llama Scope SAE & Dense & $-$PC1 & SAE unw. & \textbf{SAE-IDF} \\", r"\midrule"]
    for task in ("NELL->DBpedia", "YAGO->Wikidata", "MultiFarm"):
        rs = sorted([r for r in lk if r["task"] == task], key=lambda r: (r["layer"], r["width"]))
        for i, r in enumerate(rs):
            lines.append((task.replace("->", " $\\to$ ") if i == 0 else "") + f" & layer {r['layer']}, {r['width']} & " + " & ".join(bold_best(r, ["dense", "dense_pc1", "sae_mean", "sae_idf"])) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "tab_llama_kg.tex").write_text("\n".join(lines) + "\n")
    print(f"llama kg sweep: {len(lk)} (task, config); SAE-IDF > dense in {sum(r['sae_idf'] > r['dense'] for r in lk)}, > PC1 in {sum(r['sae_idf'] > r['dense_pc1'] for r in lk)}")
    for r in lk:
        print(f"  {r['task']:15s} {r['config']:16s} dense {r['dense']:.3f} pc1 {r['dense_pc1']:.3f} unw {r['sae_mean']:.3f} idf {r['sae_idf']:.3f}")

# ---------------------------------------------------------------- four-prover merge
mg = M / "interpretability/minif2f_merge/merge_transitivity.json"
mg_fix = M / "interpretability/minif2f_merge_pc1fix/merge_transitivity.json"   # Dense-PC1 recomputed after the 2026-10-06 fix
if mg.exists():
    rep = json.load(open(mg))
    if mg_fix.exists():                         # global_merge_transitivity.py used plain dense for dense_pc1 before the fix
        rep["metrics"]["dense_pc1"] = json.load(open(mg_fix))["metrics"]["dense_pc1"]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Method & \makecell{Mutual-top-1\\precision} & Coverage & \makecell{Transitivity\\(triples)} & \makecell{Merge\\purity} & ARI & \makecell{Components\\(489 true)} \\", r"\midrule"]
    for m, lab in (("dense", "Dense"), ("dense_pc1", "Dense$_{-\\mathrm{PC1}}$"), ("sae_mean", "SAE unweighted"), ("sae_idf", "\\textbf{SAE-IDF}")):
        r = rep["metrics"].get(m)
        if not r:
            continue
        lines.append(f"{lab} & {fmt(r.get('mean_pair_precision'))} & {fmt(r.get('mean_pair_coverage'))} & {fmt(r.get('transitivity'))} ({r.get('transitivity_triples'):,}) & {fmt(r.get('merge_purity'))} & {fmt(r.get('merge_ari'))} & {r.get('merge_components'):,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "tab_prover_merge.tex").write_text("\n".join(lines) + "\n")
    print("prover merge:", json.dumps({m: {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('purity', 'ARI', 'ari', 'agreement', 'components')}) for k, v in r.items() if k != 'pairwise'} for m, r in rep["metrics"].items()}, default=str)[:900])

# ---------------------------------------------------------------- abstention summary
ab = M / "interpretability/abstention/summary.csv"
if ab.exists():
    rows = [r for r in csv.DictReader(open(ab)) if not r['task'].startswith('XLCoST')]   # the official pool holds the query's own row
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Dense} & \multicolumn{3}{c}{\textbf{SAE-IDF}} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Task & $n$ & Top-1 & Cov.@90\% & Cov.@95\% & Top-1 & Cov.@90\% & Cov.@95\% \\", r"\midrule"]
    for r in rows:
        tname = r['task'].replace('→', '$\\to$')
        lines.append(f"{tname} & {int(r['n']):,} & {float(r['dense_acc']):.3f} & {float(r['dense_cov90']):.2f} & {float(r['dense_cov95']):.2f} & {float(r['sae_idf_acc']):.3f} & {float(r['sae_idf_cov90']):.2f} & {float(r['sae_idf_cov95']):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "tab_abstention.tex").write_text("\n".join(lines) + "\n")
    print("abstention:", len(rows), "tasks")

# ---------------------------------------------------------------- labeler agreement
ag = M / "interpretability/matches/explanation_types_agreement.json"
if ag.exists():
    rep = json.load(open(ag)); (HERE / "data" / "type_agreement.json").write_text(json.dumps(rep, indent=2))
    print(f"labeler agreement: n={rep['n']} agreement={rep['agreement']:.3f} kappa={rep['kappa']:.3f}")
