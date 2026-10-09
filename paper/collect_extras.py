#!/usr/bin/env python3
"""Tables for the pre-registered extras (paper/PREREG_extras_2026-10-04.md):

  tab_random_dict.tex     S1  random 131k dictionary + idf vs SAE-IDF, all main-table rows
  tab_dense_controls.tex  S2  stronger dense controls (top-36 PCs, z-score, ZCA, dense-idf) vs SAE-IDF
  tab_embed_qwen.tex      S4  Qwen3-Embedding-8B next to BGE-M3 and SAE-IDF

Reads the run outputs where they are (Delta stores, plus the copies under main_table/extras/),
and the paper's own numbers from data/all_results.csv / data/embed_baseline.csv. Missing runs
leave "--" cells, so the script can be run while jobs are still going. Also writes data/*.csv
snapshots next to the tables.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = Path("/projects/biro/xiaocong/main_table")
X = M / "extras"
RD = M / "random_dict/seed0"

ROWS = [  # (label, metric label, key in all_results.csv, metric in all_results.csv, S1 scores path, S2 json, S4 json)
    ("Lean 4 $\\to$ Isabelle", "MRR", "miniF2F Lean4->Isabelle", "MRR", RD / "minif2f/lean4_Isabelle/eval/scores.json", "minif2f_isabelle", "minif2f_isabelle"),
    ("Lean 4 $\\to$ Metamath", "MRR", "miniF2F Lean4->Metamath", "MRR", RD / "minif2f/lean4_Metamath/eval/scores.json", "minif2f_metamath", "minif2f_metamath"),
    ("Lean 4 $\\to$ HOL Light", "MRR", "miniF2F Lean4->HOL Light", "MRR", RD / "minif2f/lean4_HOLLight/eval/scores.json", "minif2f_hollight", "minif2f_hollight"),
    ("XLCoST, 7 languages", "MRR", "XLCoST", "MRR", "xlcost", "xlcost", "xlcost"),
    ("XLCoST, 7 languages", "P@6", "XLCoST", "Precision@6", "xlcost", "xlcost", "xlcost"),
    ("Valentine columns", "F1", "Valentine", "F1Score", RD / "valentine_column/eval/scores.json", "valentine", "valentine"),
    ("Valentine columns", "MRR", "Valentine", "MeanReciprocalRank", RD / "valentine_column/eval/scores.json", "valentine", "valentine"),
    ("NELL $\\to$ DBpedia", "MRR", "Common-KG NELL->DBpedia", "MRR", M / "commonkg/native_scored_nell_dbpedia/random_seed0/analysis_all/ranking_summary.csv", "commonkg_nell_dbpedia", "commonkg_nell_dbpedia"),
    ("YAGO $\\to$ Wikidata", "MRR", "Common-KG YAGO->Wikidata", "MRR", M / "commonkg/native_scored_yago_wikidata/random_seed0/analysis_all/ranking_summary.csv", "commonkg_yago_wikidata", "commonkg_yago_wikidata"),
    ("MultiFarm zh $\\to$ en", "MRR", "MultiFarm Chinese->English", "MRR", M / "multifarm/native_scored/random_seed0/analysis_views_cn-en/ranking_summary.csv", "multifarm_zh", "multifarm_zh"),
    ("MultiFarm ru $\\to$ en", "MRR", "MultiFarm Russian->English", "MRR", M / "multifarm/native_scored/random_seed0/analysis_views_en-ru/ranking_summary.csv", "multifarm_ru", "multifarm_ru"),
    ("MultiFarm ar $\\to$ en", "MRR", "MultiFarm Arabic->English", "MRR", M / "multifarm/native_scored/random_seed0/analysis_views_ar-en/ranking_summary.csv", "multifarm_ar", "multifarm_ar"),
]
LANGS = ["Java", "Cpp", "Python", "Csharp", "Javascript", "PHP", "C"]
XL_FILES = {"Java": "XLCoST Java", "Cpp": "XLCoST Cpp", "Python": "XLCoST Python", "Csharp": "XLCoST Csharp", "Javascript": "XLCoST Javascript", "PHP": "XLCoST PHP", "C": "XLCoST C"}


def fmt(v, best=False):
    if v is None:
        return "--"
    s = f"{v:.3f}"
    return r"\textbf{" + s + "}" if best else s


def load_json(p):
    p = Path(p)
    return json.load(open(p)) if p.is_file() else None


# ----------------------------------------------------------------------------- the paper's numbers
main = list(csv.DictReader(open(HERE / "data/all_results.csv")))


def paper(task_key, metric, method):
    if task_key == "XLCoST":
        vals = [float(r[method]) for r in main if r["group"] == "main" and r["task"] in XL_FILES.values() and r["metric"] == metric]
        return sum(vals) / len(vals) if vals else None
    for r in main:
        if r["group"] == "main" and r["task"].startswith(task_key) and r["metric"] == metric:
            return float(r[method]) if r[method] else None
    return None


# ----------------------------------------------------------------------------- S1: random dictionary
def s1_value(spec, metric_name):
    if spec == "xlcost":
        vals = []
        for L in LANGS:
            s = load_json(RD / f"xlcost/{L}_program/eval/scores.json")
            if s is None:
                return None, None
            vals.append((s["metrics"]["sae_idf"][metric_name], s["metrics"]["sae_mean"][metric_name]))
        return sum(v[0] for v in vals) / 7, sum(v[1] for v in vals) / 7
    p = Path(spec)
    if not p.is_file():
        return None, None
    if p.suffix == ".json":
        s = json.load(open(p))
        if "macro_average" in s:                       # Valentine
            return s["macro_average"]["sae_idf"][metric_name], s["macro_average"]["sae_mean"][metric_name]
        key = "MRR_first_hit" if "minif2f" in str(p) else metric_name
        return s["metrics"]["sae_idf"][key], s["metrics"]["sae_mean"][key]
    d = {r["method"]: float(r["MRR"]) for r in csv.DictReader(open(p))}   # ontology ranking_summary.csv
    return d.get("sae_idf"), d.get("sae_mean")


lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Task & Metric & Dense & Random, unw. & Random + idf & SAE unw. & \textbf{SAE-IDF} \\", r"\midrule"]
s1_rows = []
for label, mlabel, key, metric, spec, _, _ in ROWS:
    r_idf, r_unw = s1_value(spec, metric)
    dense, sae_unw, sae = paper(key, metric, "dense"), paper(key, metric, "sae_mean"), paper(key, metric, "sae_idf")
    cells = {"dense": dense, "r_unw": r_unw, "r_idf": r_idf, "s_unw": sae_unw, "sae": sae}
    have = {k: v for k, v in cells.items() if v is not None}
    best = max(have, key=have.get) if have else None
    lines.append(f"{label} & {mlabel} & " + " & ".join(fmt(cells[k], k == best) for k in ("dense", "r_unw", "r_idf", "s_unw", "sae")) + r" \\")
    s1_rows.append([label, mlabel] + [cells[k] for k in ("dense", "r_unw", "r_idf", "s_unw", "sae")])
# biomedical ontologies (S1 run on 2026-10-05 from the cached layer-20 states)
BIO_S1 = [("Bio-ML valid", "Bio-ML NCIT->DOID valid", M / "bioml/native_scored/random_seed0/analysis_views_valid/ranking_summary.csv"),
          ("Bio-ML train", "Bio-ML NCIT->DOID train", M / "bioml/native_scored/random_seed0/analysis_views_train/ranking_summary.csv"),
          ("Anatomy", "Anatomy mouse->human", M / "anatomy/native_scored/random_seed0/analysis/ranking_summary.csv")]
if any(Path(spec).is_file() for _, _, spec in BIO_S1):
    lines.append(r"\midrule")
for label, key, spec in BIO_S1:
    if not Path(spec).is_file():
        continue
    r_idf, r_unw = s1_value(spec, "MRR")
    a = next(r for r in main if r["group"] == "appendix" and r["task"] == key and r["metric"] == "MRR")
    cells = {"dense": float(a["dense"]), "r_unw": r_unw, "r_idf": r_idf, "s_unw": float(a["sae_mean"]), "sae": float(a["sae_idf"])}
    best = max(cells, key=cells.get)
    lines.append(f"{label} & MRR & " + " & ".join(fmt(cells[k], k == best) for k in ("dense", "r_unw", "r_idf", "s_unw", "sae")) + r" \\")
    s1_rows.append([label, "MRR"] + [cells[k] for k in ("dense", "r_unw", "r_idf", "s_unw", "sae")])
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "tables/tab_random_dict.tex").write_text("\n".join(lines) + "\n")
with open(HERE / "data/random_dict.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["task", "metric", "dense", "random_unweighted", "random_idf", "sae_unweighted", "sae_idf"]); w.writerows(s1_rows)
done1 = sum(1 for r in s1_rows if r[4] is not None)
print(f"S1 rows with results: {done1}/{len(s1_rows)}")
if done1:
    wins = sum(1 for r in s1_rows if r[4] is not None and r[6] is not None and r[6] - r[4] >= 0.01)
    print(f"  SAE-IDF ahead of random+idf by >=0.01 on {wins} of {done1}; random+idf > dense on {sum(1 for r in s1_rows if r[4] is not None and r[2] is not None and r[4] > r[2])}")

# ----------------------------------------------------------------------------- S2: dense controls
CTRLS = [("pc36", "$-$36 PCs"), ("zscore", "z-score"), ("zca", "ZCA"), ("dense_idf", "dense-idf"), ("dense_topk_idf", "sparse dense-idf")]


def s2_value(name, metric):
    if name == "xlcost":
        vals = {c: [] for c, _ in CTRLS}
        for L in LANGS:
            j = load_json(X / f"dense_controls/xlcost_{L}.json")
            if j is None:
                return {c: None for c, _ in CTRLS}
            for c, _ in CTRLS:
                vals[c].append(j["metrics"][c][metric])
        return {c: sum(v) / 7 for c, v in vals.items()}
    j = load_json(X / f"dense_controls/{name}.json")
    if j is None:
        return {c: None for c, _ in CTRLS}
    key = {"MRR": "MRR", "F1Score": "F1Score", "MeanReciprocalRank": "MeanReciprocalRank"}[metric]
    return {c: j["metrics"][c].get(key) for c, _ in CTRLS}


lines = [r"\begin{tabular}{@{}llrr" + "r" * len(CTRLS) + r"r@{}}", r"\toprule",
         r"Task & Metric & Dense & $-$PC1 & " + " & ".join(n for _, n in CTRLS) + r" & \textbf{SAE-IDF} \\", r"\midrule"]
s2_rows = []
for label, mlabel, key, metric, _, s2name, _ in ROWS:
    v = s2_value(s2name, metric)
    cells = {"dense": paper(key, metric, "dense"), "pc1": paper(key, metric, "dense_pc1"), **v, "sae": paper(key, metric, "sae_idf")}
    have = {k: x for k, x in cells.items() if x is not None}
    best = max(have, key=have.get) if have else None
    order = ["dense", "pc1"] + [c for c, _ in CTRLS] + ["sae"]
    lines.append(f"{label} & {mlabel} & " + " & ".join(fmt(cells[k], k == best) for k in order) + r" \\")
    s2_rows.append([label, mlabel] + [cells[k] for k in order])
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "tables/tab_dense_controls.tex").write_text("\n".join(lines) + "\n")
with open(HERE / "data/dense_controls.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["task", "metric", "dense", "pc1"] + [c for c, _ in CTRLS] + ["sae_idf"]); w.writerows(s2_rows)
done2 = sum(1 for r in s2_rows if r[4] is not None)
best_ctrl_wins = {c: sum(1 for r in s2_rows if r[4 + i] is not None and r[-1] is not None and r[4 + i] >= r[-1]) for i, (c, _) in enumerate(CTRLS)}
print(f"S2 rows with results: {done2}/{len(s2_rows)}; rows where a control >= SAE-IDF: {best_ctrl_wins}")

# ----------------------------------------------------------------------------- S4: Qwen3-Embedding-8B
emb = {(r["task"], r["metric"]): r for r in csv.DictReader(open(HERE / "data/embed_baseline.csv"))}
Q = X / "embed_qwen3"


def s4_value(name, metric):
    best_key = {"MRR": "MRR", "Precision@6": "Precision@6", "F1Score": "F1Score", "MeanReciprocalRank": "MeanReciprocalRank"}[metric]
    if name == "xlcost":
        vals = {"dense": [], "dense_pc1": []}
        for L in LANGS:
            j = load_json(Q / f"xlcost_{L}.json")
            if j is None:
                return None, None
            for k in vals:
                vals[k].append(j["metrics"][k][best_key])
        return sum(vals["dense"]) / 7, sum(vals["dense_pc1"]) / 7
    j = load_json(Q / f"{name}.json")
    if j is None:
        return None, None
    return j["metrics"]["dense"][best_key], j["metrics"]["dense_pc1"][best_key]


lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Task & Metric & BGE-M3 & BGE-M3 $-$PC1 & Qwen3-Emb-8B & Qwen3-Emb-8B $-$PC1 & \textbf{SAE-IDF} \\", r"\midrule"]
s4_rows = []
for label, mlabel, key, metric, _, _, s4name in ROWS:
    e = emb.get((label.replace("XLCoST, 7 languages", "XLCoST, 7 languages"), {"MRR": "MRR", "Precision@6": "P@6", "F1Score": "F1", "MeanReciprocalRank": "MRR"}[metric]))
    bge, bge_pc1 = (float(e["emb"]), float(e["emb_pc1"])) if e else (None, None)
    q, q_pc1 = s4_value(s4name, metric)
    sae = paper(key, metric, "sae_idf")
    cells = {"bge": bge, "bge_pc1": bge_pc1, "q": q, "q_pc1": q_pc1, "sae": sae}
    have = {k: x for k, x in cells.items() if x is not None}
    best = max(have, key=have.get) if have else None
    lines.append(f"{label} & {mlabel} & " + " & ".join(fmt(cells[k], k == best) for k in ("bge", "bge_pc1", "q", "q_pc1", "sae")) + r" \\")
    s4_rows.append([label, mlabel] + [cells[k] for k in ("bge", "bge_pc1", "q", "q_pc1", "sae")])
# the biomedical rows (Bio-ML valid is in Table 1; train split and Anatomy as in the BGE-M3 table)
lines.append(r"\midrule")
for label, key, qname in (("Bio-ML valid", "Bio-ML NCIT->DOID valid", "bioml_valid"), ("Bio-ML train", "Bio-ML NCIT->DOID train", "bioml_train"),
                          ("Anatomy", "Anatomy mouse->human", "anatomy")):
    e = emb.get((label, "MRR"))
    bge, bge_pc1 = (float(e["emb"]), float(e["emb_pc1"])) if e else (None, None)
    q, q_pc1 = s4_value(qname, "MRR")
    sae = next((float(r["sae_idf"]) for r in main if r["group"] == "appendix" and r["task"] == key and r["metric"] == "MRR"), None)
    cells = {"bge": bge, "bge_pc1": bge_pc1, "q": q, "q_pc1": q_pc1, "sae": sae}
    have = {k: x for k, x in cells.items() if x is not None}
    best = max(have, key=have.get) if have else None
    lines.append(f"{label} & MRR & " + " & ".join(fmt(cells[k], k == best) for k in ("bge", "bge_pc1", "q", "q_pc1", "sae")) + r" \\")
    s4_rows.append([label, "MRR"] + [cells[k] for k in ("bge", "bge_pc1", "q", "q_pc1", "sae")])
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE / "tables/tab_embed_qwen.tex").write_text("\n".join(lines) + "\n")
with open(HERE / "data/embed_qwen3.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["task", "metric", "bge_m3", "bge_m3_pc1", "qwen3_emb_8b", "qwen3_emb_8b_pc1", "sae_idf"]); w.writerows(s4_rows)
done4 = sum(1 for r in s4_rows if r[4] is not None)
print(f"S4 rows with results: {done4}/{len(s4_rows)}; Qwen3 > SAE-IDF on {sum(1 for r in s4_rows if r[4] is not None and r[6] is not None and r[4] > r[6])} of {done4}")
