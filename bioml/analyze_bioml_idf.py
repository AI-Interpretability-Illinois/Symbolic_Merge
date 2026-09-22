#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_bioml_idf.py

Reproduces analyze_bioml_information_weighted_sae_target_bg.py and adds the
enrichment-only weighting that wins on XLCoST and Valentine:

    sae_idf = mean_activation * log(N / (1 + df))

where df is the number of TARGET entities in which the feature fires at least
once and N is the number of target entities. Unlike sae_info, there is no
p_entity multiplier, and unlike sae_logodds there is no logit transform of
view-level rates -- it is document-frequency weighting over the candidate
population, which is the same definition used by
xlcost/eval_xlcost_sae.py and schema/eval_valentine_sae.py.

The existing methods are recomputed here so the comparison is apples to apples
(same candidates, same background). Their values should match the published
ranking_summary.csv of the corresponding run; the script prints the comparison
when --reference is given.

Example:
    python bioml/analyze_bioml_idf.py \
        --sae_run_dirs  <run_q20_sae>,<run_q20_40_sae>,... \
        --context_run_dirs <run_q20>,<run_q20_40>,... \
        --output_dir <out> \
        --reference <run>/ranking_summary.csv
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

METHODS = ["dense_mean", "sae_mean", "sae_stable_0.80",
           "sae_info", "sae_info_reliability", "sae_logodds", "sae_idf"]


def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_rep_dirs(run_dirs):
    src, tgt = {}, {}
    for run in run_dirs:
        for side, store in (("src", src), ("tgt", tgt)):
            for p in sorted((run / "representations" / side).glob("*.pt")):
                x = load_pt(p)
                store.setdefault(x["entity_iri"], x)
    return src, tgt


def load_queries(context_dirs):
    out, seen = [], set()
    for d in context_dirs:
        for q in json.loads((d / "selected_queries.json").read_text(encoding="utf-8")):
            qid = int(q.get("query_id", q.get("qid")))
            if qid in seen:
                continue
            seen.add(qid)
            out.append({"qid": qid, "src": q["src"], "gold": q["gold"],
                        "candidates": q["candidates"]})
    return sorted(out, key=lambda x: x["qid"])


def dense_cos(a, b):
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    return float(torch.dot(a, b)) / (na * nb) if na and nb else 0.0


def norm_sparse(x):
    idx = x["idx"].detach().cpu().long()
    val = x["val"].detach().cpu().float()
    if idx.numel() > 1:
        o = torch.argsort(idx)
        idx, val = idx[o], val[o]
    return {"idx": idx, "val": val, "size": int(x["size"])}


def sparse_cos(a, b):
    ai, av, bi, bv = a["idx"], a["val"], b["idx"], b["val"]
    if ai.numel() == 0 or bi.numel() == 0:
        return 0.0
    if ai.numel() <= bi.numel():
        d = dict(zip(ai.tolist(), av.tolist()))
        dot = sum(d.get(i, 0.0) * v for i, v in zip(bi.tolist(), bv.tolist()))
    else:
        d = dict(zip(bi.tolist(), bv.tolist()))
        dot = sum(d.get(i, 0.0) * v for i, v in zip(ai.tolist(), av.tolist()))
    na = float(torch.linalg.vector_norm(av))
    nb = float(torch.linalg.vector_norm(bv))
    return dot / (na * nb) if na and nb else 0.0


def existing(rep, method):
    agg = rep["aggregate"]
    if method == "sae_mean":
        return norm_sparse(agg["sae_mean"])
    obj = agg["sae_stable"]["0.80"]
    return norm_sparse(obj["vector"] if "vector" in obj else obj)


def build_background(reps):
    """Per-context presence (as in the original) plus per-entity document frequency."""
    counts = Counter()
    doc_freq = Counter()
    total = 0
    for rep in reps:
        seen = set()
        for c in rep["contexts"]:
            idx = set(int(i) for i in c["sae"]["idx"].detach().cpu().tolist())
            counts.update(idx)
            seen |= idx
            total += 1
        doc_freq.update(seen)
    return counts, total, doc_freq, len(reps)


def build_signature(rep, bg, bg_total, doc_freq, n_docs, eps=1e-6):
    n = len(rep["contexts"])
    size = int(rep["contexts"][0]["sae"]["size"])
    vals = defaultdict(lambda: [0.0] * n)
    for ci, c in enumerate(rep["contexts"]):
        for i, v in zip(c["sae"]["idx"].detach().cpu().tolist(),
                        c["sae"]["val"].detach().cpu().float().tolist()):
            vals[int(i)][ci] = float(v)
    buckets = {k: ([], []) for k in ("sae_info", "sae_info_reliability",
                                     "sae_logodds", "sae_idf")}
    for j, xs in vals.items():
        x = torch.tensor(xs, dtype=torch.float32)
        pe = float((x > 0).sum()) / n
        pb = bg.get(j, 0) / bg_total
        mean = float(x.mean())
        std = float(x.std(unbiased=False))
        cv = std / (abs(mean) + eps)
        info = max(math.log((pe + eps) / (pb + eps)), 0.0)
        pe2 = min(max(pe, eps), 1 - eps)
        pb2 = min(max(pb, eps), 1 - eps)
        lod = max(math.log(pe2 / (1 - pe2)) - math.log(pb2 / (1 - pb2)), 0.0)
        idf = max(math.log(n_docs / (1.0 + doc_freq.get(j, 0))), 0.0)
        weights = {
            "sae_info": mean * pe * info,
            "sae_info_reliability": mean * pe * info / (1 + cv),
            "sae_logodds": mean * lod,
            "sae_idf": mean * idf,
        }
        for k, w in weights.items():
            if w > 0:
                buckets[k][0].append(j)
                buckets[k][1].append(w)
    return {k: {"idx": torch.tensor(i, dtype=torch.long),
                "val": torch.tensor(v, dtype=torch.float32), "size": size}
            for k, (i, v) in buckets.items()}


def metrics(ranks):
    n = len(ranks)
    return {"n": n, "MRR": sum(1 / r for r in ranks) / n,
            "H@1": sum(r <= 1 for r in ranks) / n,
            "H@5": sum(r <= 5 for r in ranks) / n,
            "H@10": sum(r <= 10 for r in ranks) / n,
            "MeanRank": sum(ranks) / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sae_run_dirs", required=True)
    ap.add_argument("--context_run_dirs", required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--background", choices=("target", "all"), default="all",
                    help="'all' = source+target reps, which is what the published "
                         "q120 numbers used (analyze_bioml_information_weighted_sae.py); "
                         "'target' = target only (the _target_bg variant)")
    ap.add_argument("--reference", type=Path, default=None,
                    help="published ranking_summary.csv to validate against")
    args = ap.parse_args()

    sae_dirs = [Path(x).expanduser().resolve() for x in args.sae_run_dirs.split(",") if x.strip()]
    ctx_dirs = [Path(x).expanduser().resolve() for x in args.context_run_dirs.split(",") if x.strip()]
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("BIO-ML: INFORMATION WEIGHTINGS + IDF  (CPU)")
    print("=" * 100)
    src, tgt = load_rep_dirs(sae_dirs)
    qs = load_queries(ctx_dirs)
    print(f"queries={len(qs)} source reps={len(src)} target reps={len(tgt)}")
    bg_reps = list(tgt.values()) if args.background == "target" \
        else list(src.values()) + list(tgt.values())
    bg, bg_total, doc_freq, n_docs = build_background(bg_reps)
    print(f"background: mode={args.background} contexts={bg_total} features={len(bg)} "
          f"documents(entities)={n_docs}")

    print("building signatures...", flush=True)
    ssig = {i: build_signature(r, bg, bg_total, doc_freq, n_docs) for i, r in src.items()}
    tsig = {i: build_signature(r, bg, bg_total, doc_freq, n_docs) for i, r in tgt.items()}

    ranks = defaultdict(list)
    rows = []
    for q in qs:
        s, g = q["src"], q["gold"]
        cands = [c for c in q["candidates"] if c in tgt]
        if s not in src or g not in cands:
            raise RuntimeError(f"missing representation for qid={q['qid']}")
        row = {"qid": q["qid"], "source_label": src[s]["label"], "gold_label": tgt[g]["label"]}
        for m in METHODS:
            scored = []
            for c in cands:
                if m == "dense_mean":
                    score = dense_cos(src[s]["aggregate"]["dense_mean"],
                                      tgt[c]["aggregate"]["dense_mean"])
                elif m in ("sae_mean", "sae_stable_0.80"):
                    score = sparse_cos(existing(src[s], m), existing(tgt[c], m))
                else:
                    score = sparse_cos(ssig[s][m], tsig[c][m])
                scored.append((c, score))
            scored.sort(key=lambda z: (-z[1], z[0]))
            r = next(i + 1 for i, (iri, _) in enumerate(scored) if iri == g)
            row[m + "_rank"] = r
            ranks[m].append(r)
        rows.append(row)

    write_csv(out / "per_query_ranks.csv", rows)
    summary = [{"method": m, **metrics(ranks[m])} for m in METHODS]
    write_csv(out / "ranking_summary.csv", summary)
    (out / "analysis.json").write_text(json.dumps(
        {"n_queries": len(qs), "background_contexts": bg_total,
         "background_documents": n_docs, "summary": summary}, indent=2))

    print("\nFINAL SUMMARY")
    for s_ in summary:
        print(f"[{s_['method']:<22}] MRR={s_['MRR']:.4f} H@1={s_['H@1']:.4f} "
              f"H@5={s_['H@5']:.4f} H@10={s_['H@10']:.4f} MeanRank={s_['MeanRank']:.2f}")

    if args.reference and args.reference.is_file():
        ref = {r["method"]: float(r["MRR"]) for r in
               csv.DictReader(open(args.reference, encoding="utf-8"))}
        print("\nVALIDATION against", args.reference)
        ok = True
        for s_ in summary:
            if s_["method"] in ref:
                d = s_["MRR"] - ref[s_["method"]]
                flag = "match" if abs(d) < 1e-6 else f"DIFFERS by {d:+.6f}"
                if abs(d) >= 1e-6:
                    ok = False
                print(f"  {s_['method']:<22} mine={s_['MRR']:.6f} "
                      f"published={ref[s_['method']]:.6f}  {flag}")
            else:
                print(f"  {s_['method']:<22} mine={s_['MRR']:.6f}  (new, not in reference)")
        print("  -> reproduction exact" if ok else "  -> MISMATCH: investigate before trusting sae_idf")
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
