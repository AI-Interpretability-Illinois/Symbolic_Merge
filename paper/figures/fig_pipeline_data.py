"""Real numbers for Figure 1, from the stored YAGO -> Wikidata run (Common-KG, native views).

The worked example used in the paper is the NELL class "trainstation" (context: its instances) and its DBpedia
gold counterpart "RailwayStation" (--task nell --query_label trainstation --gold_label RailwayStation). For the
symbol shown in steps 1-3 (the query by default, --side) we record its unweighted sparse code a(s) and the
idf-weighted code w(s) (largest coordinates, with each feature's idf, Pile density and Neuronpedia
explanation); for the matching step, the weighted codes of the query, the gold and the candidate that
the dense hidden state prefers, their cosines with the query, and the per-feature contributions.

Writes data/fig_pipeline_example.json. CPU, about a minute.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for sub in ("bioml", "tools"):
    sys.path.insert(0, str(REPO / sub))
import analyze_bioml_idf as A  # noqa: E402
import interpret_matches as IM  # noqa: E402

M = IM.M
STORES = {"yago": (M / "commonkg/native_scored_yago_wikidata/L20_w131k_l0_114", M / "commonkg/native/yago_wikidata", "Common-KG YAGO -> Wikidata", "YAGO", "Wikidata"),
          "nell": (M / "commonkg/native_scored_nell_dbpedia/L20_w131k_l0_114", M / "commonkg/native/nell_dbpedia", "Common-KG NELL -> DBpedia", "NELL", "DBpedia")}
TOP = 8


def sparse(sig_entry):
    return sig_entry["idx"].numpy().astype(int), sig_entry["val"].float().numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_label", default="Permit")
    ap.add_argument("--gold_label", default="license")
    ap.add_argument("--task", choices=list(STORES), default="yago")
    ap.add_argument("--side", choices=["query", "gold"], default="query", help="which symbol steps 1-3 show")
    ap.add_argument("--out", default=None, help="output json (default data/fig_pipeline_example.json)")
    args = ap.parse_args()
    QUERY_LABEL, GOLD_LABEL = args.query_label, args.gold_label
    SAE_RUN, CTX, TASK_NAME, SRC_SYS, TGT_SYS = STORES[args.task]
    src, tgt = A.load_rep_dirs([SAE_RUN])
    qs = A.load_queries([CTX])
    reps = list(src.values()) + list(tgt.values())
    bg, bg_total, doc_freq, n_docs = A.build_background(reps)
    expl, density = IM.load_metadata()
    idf_all = np.maximum(np.log(n_docs / (1.0 + doc_freq.numpy() if hasattr(doc_freq, "numpy") else doc_freq)), 0.0) if not isinstance(doc_freq, dict) else None

    q = next(q for q in qs if src[q["src"]]["label"] == QUERY_LABEL and tgt[q["gold"]]["label"] == GOLD_LABEL)
    q_rep, g_rep = src[q["src"]], tgt[q["gold"]]
    cands = [c for c in q["candidates"] if c in tgt]
    if q["gold"] not in cands:
        cands.append(q["gold"])

    def code(rep):
        sig = A.build_signature(rep, bg, bg_total, doc_freq, n_docs)
        a_idx, a_val = sparse(rep["aggregate"]["sae_mean"])
        w_idx, w_val = sparse(sig["sae_idf"])
        return dict(zip(a_idx, a_val)), dict(zip(w_idx, w_val))

    def dense(rep):
        d = rep["aggregate"]["dense_mean"].float().numpy().astype(np.float64)
        return d / max(np.linalg.norm(d), 1e-12)

    def cos(u, v):
        keys = set(u) & set(v)
        num = sum(u[k] * v[k] for k in keys)
        return num / (np.sqrt(sum(x * x for x in u.values())) * np.sqrt(sum(x * x for x in v.values())) + 1e-12)

    def feat(f, a=None, w=None):
        f = int(f)
        return {"feature": f, "a": None if a is None else float(a), "w": None if w is None else float(w),
                "idf": float(idf_all[f]) if idf_all is not None else None, "df": int(doc_freq[f]),
                "density": (None if not (f < len(density)) or density[f] != density[f] else float(density[f])),
                "explanation": expl.get(f)}

    # the symbol shown in steps 1-3: unweighted vs weighted code (the query by default, so that the
    # same symbol flows through the LLM, the SAE, the weighting and the matching)
    a_g, w_g = code(g_rep)
    a_q, w_q = code(q_rep)

    def describe(rep, a, w):
        top_a = sorted(a, key=lambda f: -a[f])[:TOP]
        top_w = sorted(w, key=lambda f: -w[f])[:TOP]
        return {"label": rep["label"], "n_views": len(rep["contexts"]), "views": [c["text"] for c in rep["contexts"]],
                "n_active": len(a), "top_unweighted": [feat(f, a[f], w.get(f, 0.0)) for f in top_a],
                "top_weighted": [feat(f, a.get(f, 0.0), w[f]) for f in top_w]}

    symbol = describe(q_rep if args.side == "query" else g_rep, *((a_q, w_q) if args.side == "query" else (a_g, w_g)))
    gold_symbol = describe(g_rep, a_g, w_g)

    # matching: the query against the gold and against the candidate the dense state prefers
    dq = dense(q_rep)
    dense_scores = {c: float(dq @ dense(tgt[c])) for c in cands}
    idf_scores = {c: float(cos(w_q, code(tgt[c])[1])) for c in cands}
    order_idf = sorted(cands, key=lambda c: -idf_scores[c]); order_dense = sorted(cands, key=lambda c: -dense_scores[c])
    distractor = next(c for c in order_dense if c != q["gold"])
    a_d, w_d = code(tgt[distractor])

    def contributions(u, v, k=6):
        nu, nv = np.sqrt(sum(x * x for x in u.values())), np.sqrt(sum(x * x for x in v.values()))
        rows = sorted(((f, u[f] * v[f] / (nu * nv + 1e-12)) for f in set(u) & set(v)), key=lambda t: -t[1])[:k]
        return [dict(feat(f, None, None), c=float(c), w_query=float(u[f]), w_cand=float(v[f])) for f, c in rows]

    match = {"query": {"label": q_rep["label"], "views": [c["text"] for c in q_rep["contexts"][:3]], "n_active": len(a_q),
                       "top_weighted": [feat(f, a_q.get(f, 0.0), w_q[f]) for f in sorted(w_q, key=lambda f: -w_q[f])[:TOP]]},
             "n_candidates": len(cands),
             "gold": {"label": g_rep["label"], "cos_idf": idf_scores[q["gold"]], "cos_dense": dense_scores[q["gold"]],
                      "rank_idf": order_idf.index(q["gold"]) + 1, "rank_dense": order_dense.index(q["gold"]) + 1,
                      "contributions": contributions(w_q, w_g)},
             "distractor": {"label": tgt[distractor]["label"], "views": [c["text"] for c in tgt[distractor]["contexts"][:3]],
                            "cos_idf": idf_scores[distractor], "cos_dense": dense_scores[distractor],
                            "rank_idf": order_idf.index(distractor) + 1, "rank_dense": order_dense.index(distractor) + 1,
                            "contributions": contributions(w_q, w_d),
                            "top_weighted": [feat(f, a_d.get(f, 0.0), w_d[f]) for f in sorted(w_d, key=lambda f: -w_d[f])[:TOP]]},
             "unweighted_contributions_gold": contributions(a_q, a_g)}
    out = {"task": TASK_NAME, "source_system": SRC_SYS, "target_system": TGT_SYS, "n_docs_reference": int(n_docs),
           "symbol": symbol, "gold_symbol": gold_symbol, "match": match}
    (Path(args.out) if args.out else REPO / "paper/data/fig_pipeline_example.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
