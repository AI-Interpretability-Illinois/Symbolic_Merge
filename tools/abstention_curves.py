#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abstention_curves.py

Toward full alignments: if a matcher may abstain, how much coverage does it keep at a
given precision? For every query we take the cosine margin between the best and the
second-best candidate as the confidence, sort queries by it, and report precision
(top-1 correct) against coverage (fraction of queries answered), for SAE-IDF and for
the dense hidden state, on the main-table tasks whose representations are on disk.

Writes <output_dir>/<task>.json (per-query margin + correctness per method) and
<output_dir>/summary.csv (coverage at 90% and 95% precision, area under the
precision-coverage curve), from the same weights the evaluation used.

Example:
    python tools/abstention_curves.py --output_dir /projects/biro/xiaocong/main_table/interpretability/abstention
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for sub in ("xlcost", "bioml", "schema", "tools"):
    sys.path.insert(0, str(REPO / sub))
import interpret_matches as IM  # noqa: E402  (loaders' helpers: sparse_vec, cos, load_pt)

M = Path("/projects/biro/xiaocong/main_table")
EPS = IM.EPS
METHODS = ("dense", "sae_idf")


def record(scores, relevant):
    """(top-1 correct, margin between best and second-best candidate)."""
    order = np.argsort(-scores, kind="stable")
    top, second = order[0], order[1] if len(order) > 1 else order[0]
    return bool(top in set(relevant)), float(scores[top] - scores[second])


def xlcost_task(run):
    import eval_xlcost_sae as E
    cfg = json.loads((run / "run_config.json").read_text())
    q, c = E.load_side(run, "query"), E.load_side(run, "candidate")
    size = int(c[0]["sae"]["size"])
    corpus, _ = E.background(c, "token", size)
    Q, C = E.build_matrix(q, "sae_idf", corpus, size, EPS), E.build_matrix(c, "sae_idf", corpus, size, EPS)
    S = {"sae_idf": (Q @ C.T).toarray()}
    qd = np.stack([d["dense"].numpy().astype(np.float64) for d in q]); cd = np.stack([d["dense"].numpy().astype(np.float64) for d in c])
    S["dense"] = (qd / np.linalg.norm(qd, axis=1, keepdims=True)) @ (cd / np.linalg.norm(cd, axis=1, keepdims=True)).T
    prefix = defaultdict(list)
    for j, d in enumerate(c):
        prefix[d["idx"].split("/")[0]].append(j)
    out = {m: [] for m in METHODS}
    for i, d in enumerate(q):
        rel = prefix.get(d["idx"].split("/")[0])
        if not rel:
            continue
        for m in METHODS:
            out[m].append(record(S[m][i], rel))
    return out


def entity_task(sae_run, ctx, pair_bg=False):
    import analyze_bioml_idf as A
    src, tgt = A.load_rep_dirs([sae_run])
    qs = A.load_queries([ctx])
    reps = list(src.values()) + list(tgt.values())
    bg, bg_total, doc_freq, n_docs = A.build_background(reps)
    vec, dense = {}, {}
    def get(rep):
        k = rep["entity_iri"]
        if k not in vec:
            sig = A.build_signature(rep, bg, bg_total, doc_freq, n_docs)
            vec[k] = IM.sparse_vec(sig["sae_idf"]["idx"].numpy(), sig["sae_idf"]["val"].numpy())
            d = rep["aggregate"]["dense_mean"].detach().cpu().float().numpy().astype(np.float64)
            dense[k] = d / max(np.linalg.norm(d), 1e-12)
        return vec[k], dense[k]
    out = {m: [] for m in METHODS}
    for q in qs:
        if q["src"] not in src or q["gold"] not in tgt:
            continue
        cands = [c for c in q["candidates"] if c in tgt]
        if q["gold"] not in cands:
            cands.append(q["gold"])
        sv, sd = get(src[q["src"]])
        cv = [get(tgt[c]) for c in cands]
        s_idf = np.array([IM.cos(sv, v) for v, _ in cv]); s_den = np.array([sd @ d for _, d in cv])
        rel = [i for i, c in enumerate(cands) if c == q["gold"]]
        out["sae_idf"].append(record(s_idf, rel)); out["dense"].append(record(s_den, rel))
    return out


def valentine_task(run):
    import eval_valentine_sae as E
    out = {m: [] for m in METHODS}
    for f in sorted((run / "pairs").glob("*.pt")):
        p = IM.load_pt(f)
        scols, tcols = p["source"]["columns"], p["target"]["columns"]
        if not scols or len(tcols) < 2:
            continue
        size = int(tcols[0]["sae"]["size"]); corpus = E.corpus_stats(tcols, size)
        w = lambda col: IM.sparse_vec(*E.column_weights(col, "sae_idf", corpus, EPS))
        tvec = [w(t) for t in tcols]
        td = np.stack([t["dense"].numpy().astype(np.float64) for t in tcols]); td /= np.linalg.norm(td, axis=1, keepdims=True)
        tname = {t["column"]: j for j, t in enumerate(tcols)}
        gold = defaultdict(set)
        for a, b in p["ground_truth"]:
            if b in tname:
                gold[a].add(tname[b])
        for sc in scols:
            rel = sorted(gold.get(sc["column"], ()))
            if not rel:
                continue
            sv = w(sc); sd = sc["dense"].numpy().astype(np.float64); sd /= max(np.linalg.norm(sd), 1e-12)
            out["sae_idf"].append(record(np.array([IM.cos(sv, t) for t in tvec]), rel))
            out["dense"].append(record(td @ sd, rel))
    return out


def curve(recs):
    """Precision at each coverage level, queries ordered by decreasing margin."""
    recs = sorted(recs, key=lambda r: -r[1])
    correct = np.cumsum([r[0] for r in recs]); n = np.arange(1, len(recs) + 1)
    prec = correct / n; cov = n / len(recs)
    return cov, prec


def coverage_at(cov, prec, target):
    ok = np.nonzero(prec >= target)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0


TASKS = {
    "Lean 4 → Isabelle": lambda: xlcost_task(M / "minif2f/lean4_Isabelle"),
    "Lean 4 → Metamath": lambda: xlcost_task(M / "minif2f/lean4_Metamath"),
    "Lean 4 → HOL Light": lambda: xlcost_task(M / "minif2f/lean4_HOLLight"),
    "XLCoST Java → 6 languages": lambda: xlcost_task(M / "xlcost/Java_program"),
    "Valentine columns": lambda: valentine_task(M / "valentine_column"),
    "NELL → DBpedia": lambda: entity_task(M / "commonkg/native_scored_nell_dbpedia/L20_w131k_l0_114", M / "commonkg/native/nell_dbpedia"),
    "YAGO → Wikidata": lambda: entity_task(M / "commonkg/native_scored_yago_wikidata/L20_w131k_l0_114", M / "commonkg/native/yago_wikidata"),
    "MultiFarm zh → en": lambda: entity_task(M / "multifarm/native_scored/L20_w131k_l0_114", M / "multifarm/native/views/cn-en"),
    "MultiFarm ru → en": lambda: entity_task(M / "multifarm/native_scored/L20_w131k_l0_114", M / "multifarm/native/views/en-ru"),
    "MultiFarm ar → en": lambda: entity_task(M / "multifarm/native_scored/L20_w131k_l0_114", M / "multifarm/native/views/ar-en"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--tasks", default="all")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    rows = []
    for name in names:
        out = args.output_dir / (name.replace(" ", "_").replace("→", "to") + ".json")
        if out.exists():
            recs = json.loads(out.read_text())
        else:
            recs = TASKS[name]()
            out.write_text(json.dumps(recs))
        row = {"task": name, "n": len(recs["sae_idf"])}
        for m in METHODS:
            cov, prec = curve(recs[m])
            row[f"{m}_acc"] = float(prec[-1]); row[f"{m}_cov90"] = coverage_at(cov, prec, 0.90)
            row[f"{m}_cov95"] = coverage_at(cov, prec, 0.95); row[f"{m}_aupc"] = float(np.trapz(prec, cov))
        rows.append(row)
        print(f"{name:28s} n={row['n']:5d}  " + "  ".join(f"{m}: acc {row[f'{m}_acc']:.3f} cov@90 {row[f'{m}_cov90']:.2f} cov@95 {row[f'{m}_cov95']:.2f} AUPC {row[f'{m}_aupc']:.3f}" for m in METHODS), flush=True)
    with open(args.output_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
