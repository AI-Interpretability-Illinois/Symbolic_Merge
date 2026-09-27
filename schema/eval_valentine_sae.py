#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_valentine_sae.py

Schema matching evaluation on Valentine, scored with Valentine's own metrics.

Reads the column representations from run_valentine_sae.py, scores every
source-column / target-column pair under each weighting, and hands the result
to `valentine.algorithms.matcher_results.MatcherResults.get_metrics`, so the
numbers are produced by the benchmark's implementation (Hungarian one-to-one
assignment for Precision / Recall / F1, plus MRR, Precision@top-10%, and
Recall at the size of the ground truth).

Unlike the code-retrieval task, the view unit here is a sampled cell value, so
p_column = (views where the feature fires) / n_views is a conservation measure
over independent observations of the same symbol, matching Bio-ML.

Metrics (all cosine similarity over the weighted feature vectors):
    dense                  mean hidden state over views
    sae_mean               mean SAE activation over views
    sae_stable_<f>         mean, zeroed unless the feature fires in >= f of views
    sae_idf                mean * log(N/df) over the background columns
    sae_info               mean * p_column * information
    sae_info_reliability   mean * p_column * information * 1/(1+CV)
    sae_logodds            mean * positive log-odds vs the background

Example:
    python schema/eval_valentine_sae.py \
        --run_dir <out>/valentine --output_dir <out>/valentine/eval
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

DEFAULT_METRICS = ("dense,dense_centered,dense_pc1,sae_mean,sae_idf,"
                   "sae_idf_universal,sae_info,sae_info_reliability,sae_logodds")


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# Weights
# =============================================================================

def corpus_stats(columns, size):
    """Background over a set of columns: view-level rates and document frequency."""
    counts = np.zeros(size, dtype=np.float64)
    doc_freq = np.zeros(size, dtype=np.float64)
    total_views = 0
    for c in columns:
        idx = c["sae"]["idx"].numpy().astype(np.int64)
        counts[idx] += c["sae"]["count"].numpy().astype(np.float64)
        doc_freq[idx] += 1.0
        total_views += int(c["n_views"])
    n = max(1, len(columns))
    return {"p_bg": counts / max(1, total_views),
            "idf_universal": load_universal_idf(size),
            "idf": np.log(n / (1.0 + doc_freq)),
            "n_cols": n, "total_views": total_views}



UNIVERSAL_DENSITY_FILE = "/projects/biro/xiaocong/pile_density_l0_114.json"


def load_universal_idf(size, path=None, floor=1e-6):
    """idf_universal(f) = -log(Pile density of f), from Neuronpedia's feature dump.

    The corpus-derived idf is relative to whichever candidates happen to be in
    the pool. This one is a fixed property of the feature, so signatures from
    systems that never co-occur are still comparable -- the property a global
    coordinate system needs.
    """
    import json as _json
    p = path or UNIVERSAL_DENSITY_FILE
    try:
        raw = _json.load(open(p, encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"universal density file unreadable ({p}): {e}")
    out = np.zeros(size, dtype=np.float64)
    for k, v in raw.items():
        i = int(k)
        if i < size:
            out[i] = -math.log(max(float(v), floor))
    return np.maximum(out, 0.0)

def column_weights(col, metric, corpus, eps):
    sae = col["sae"]
    idx = sae["idx"].numpy().astype(np.int64)
    n = float(col["n_views"])
    mean = sae["sum"].numpy().astype(np.float64) / n
    if metric == "sae_mean":
        return idx, mean
    if metric == "sae_idf":
        return idx, mean * np.maximum(corpus["idf"][idx], 0.0)
    if metric == "sae_idf_universal":
        return idx, mean * corpus["idf_universal"][idx]

    p_col = sae["count"].numpy().astype(np.float64) / n
    if metric.startswith("sae_stable_"):
        return idx, np.where(p_col >= float(metric[len("sae_stable_"):]), mean, 0.0)

    pb = corpus["p_bg"][idx]
    if metric == "sae_logodds":
        pe2 = np.clip(p_col, eps, 1 - eps)
        pb2 = np.clip(pb, eps, 1 - eps)
        return idx, mean * np.maximum(np.log(pe2 / (1 - pe2)) - np.log(pb2 / (1 - pb2)), 0.0)

    info = np.maximum(np.log((p_col + eps) / (pb + eps)), 0.0)
    if metric == "sae_info":
        return idx, mean * p_col * info
    sq = sae["sumsq"].numpy().astype(np.float64)
    var = np.maximum(sq / n - mean * mean, 0.0)
    cv = np.sqrt(var) / (np.abs(mean) + eps)
    return idx, mean * p_col * info * (1.0 / (1.0 + cv))


def sparse_vec(col, metric, corpus, eps):
    idx, w = column_weights(col, metric, corpus, eps)
    keep = w > 0
    return dict(zip(idx[keep].tolist(), w[keep].tolist()))


def cos_sparse(a, b):
    if not a or not b:
        return 0.0
    small, other = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(v * other.get(k, 0.0) for k, v in small.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0



def _top_pc(x, iters=50):
    """First principal direction of x (rows = samples), by power iteration."""
    v = np.random.default_rng(0).normal(size=x.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        v = x.T @ (x @ v)
        n = np.linalg.norm(v)
        if n == 0:
            return None
        v /= n
    return v


def dense_variants(queries, candidates, metric):
    """dense, dense_centered (subtract the candidate-corpus mean), dense_pc1
    (also project out the corpus's first principal direction).

    The control for the claim that SAE weighting beats dense: pooled hidden
    states share a large common component (cos(h, mean h) ~ 0.9), which
    compresses every dense cosine. Centering removes it without any SAE.
    """
    q = np.stack([d["dense"].numpy().astype(np.float64) for d in queries])
    c = np.stack([d["dense"].numpy().astype(np.float64) for d in candidates])
    if metric != "dense":
        mu = c.mean(0)
        q = q - mu
        c = c - mu
        if metric == "dense_pc1":
            pc = _top_pc(c)
            if pc is not None:
                q = q - np.outer(q @ pc, pc)
                c = c - np.outer(c @ pc, pc)
    qn = np.linalg.norm(q, axis=1, keepdims=True); qn[qn == 0] = 1.0
    cn = np.linalg.norm(c, axis=1, keepdims=True); cn[cn == 0] = 1.0
    return (q / qn).astype(np.float32), (c / cn).astype(np.float32)

def cos_dense(a, b):
    x = a.numpy().astype(np.float64)
    y = b.numpy().astype(np.float64)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx and ny else 0.0


# =============================================================================
# Diagnostic: is the reliability term doing anything?
# =============================================================================

def cv_diagnostic(columns):
    """CV vs sqrt((1-p)/p). Near 1.0 means CV only restates sparsity."""
    ps, cvs = [], []
    for c in columns:
        n = float(c["n_views"])
        mean = c["sae"]["sum"].numpy().astype(np.float64) / n
        sq = c["sae"]["sumsq"].numpy().astype(np.float64)
        p = c["sae"]["count"].numpy().astype(np.float64) / n
        var = np.maximum(sq / n - mean * mean, 0.0)
        cv = np.sqrt(var) / (np.abs(mean) + 1e-6)
        ps.append(p)
        cvs.append(cv)
    p = np.concatenate(ps)
    cv = np.concatenate(cvs)
    m = (p > 0) & (p < 1) & (cv > 0)
    if m.sum() < 10:
        return {}
    cvm = cv[m]
    pred = np.sqrt((1 - p[m]) / p[m])
    return {"cv_vs_sparsity_corr": round(float(np.corrcoef(cvm, pred)[0, 1]), 4),
            "cv_over_prediction_median": round(float(np.median(cvm / pred)), 4),
            "p_view_median": round(float(np.median(p[m])), 4),
            "features_compared": int(m.sum())}


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--metrics", default=DEFAULT_METRICS)
    ap.add_argument("--stable_frequencies", default="0.50,0.75,1.00")
    ap.add_argument("--background", choices=("pair_target", "pair_both", "global_target"),
                    default="pair_target",
                    help="which columns form the background population")
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--text_only", action="store_true",
                    help="skip numeric-dtype columns on both sides")
    args = ap.parse_args()

    from valentine.algorithms.matcher_results import MatcherResults, ColumnPair

    run_dir = args.run_dir.expanduser().resolve()
    out = (args.output_dir or (run_dir / "eval")).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    pair_files = sorted((run_dir / "pairs").glob("*.pt"))
    if not pair_files:
        raise RuntimeError(f"no pair files in {run_dir/'pairs'}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for f in [x.strip() for x in args.stable_frequencies.split(",") if x.strip()]:
        metrics.append(f"sae_stable_{float(f):.2f}")

    pairs = [load_pt(p) for p in pair_files]
    numeric = lambda c: str(c.get("dtype", "")).startswith(("int", "float", "uint"))
    if args.text_only:
        for p in pairs:
            for side in ("source", "target"):
                p[side]["columns"] = [c for c in p[side]["columns"] if not numeric(c)]

    size = int(pairs[0]["source"]["columns"][0]["sae"]["size"])
    global_target = [c for p in pairs for c in p["target"]["columns"]]
    global_corpus = corpus_stats(global_target, size) if args.background == "global_target" else None

    print("=" * 100)
    print(f"VALENTINE SCHEMA MATCHING  (preset={config['preset']}, views={config['views']})")
    print("=" * 100)
    n_cols = sum(len(p["source"]["columns"]) + len(p["target"]["columns"]) for p in pairs)
    print(f"pairs={len(pairs)} columns={n_cols} background={args.background}"
          f"{' text_only' if args.text_only else ''}")
    diag = cv_diagnostic(global_target)
    print(f"reliability diagnostic (CV vs sqrt((1-p)/p)): corr={diag.get('cv_vs_sparsity_corr')} "
          f"median_ratio={diag.get('cv_over_prediction_median')} "
          f"median p_view={diag.get('p_view_median')}")
    print("  (corr near 1.0 means CV only restates sparsity and the reliability term is idle)")

    per_pair, agg = [], defaultdict(lambda: defaultdict(list))
    skipped = []
    for p, path in zip(pairs, pair_files):
        src_cols, tgt_cols = p["source"]["columns"], p["target"]["columns"]
        if not src_cols or not tgt_cols:
            skipped.append({"pair": p["pair"]["name"], "reason": "no columns"})
            continue
        names = {c["column"] for c in src_cols}
        tnames = {c["column"] for c in tgt_cols}
        gt = [(s, t) for s, t in p["ground_truth"] if s in names and t in tnames]
        if not gt:
            skipped.append({"pair": p["pair"]["name"], "reason": "no reachable ground truth"})
            continue
        if args.background == "pair_target":
            corpus = corpus_stats(tgt_cols, size)
        elif args.background == "pair_both":
            corpus = corpus_stats(src_cols + tgt_cols, size)
        else:
            corpus = global_corpus

        row = {"pair": p["pair"]["name"], "dataset": p["pair"]["dataset"],
               "relatedness": p["pair"]["relatedness"],
               "source_columns": len(src_cols), "target_columns": len(tgt_cols),
               "gold_pairs": len(gt)}
        for metric in metrics:
            if metric.startswith("dense"):
                dq, dc = dense_variants(src_cols, tgt_cols, metric)
                scores = {ColumnPair(p["source"]["table"], s["column"],
                                     p["target"]["table"], t["column"]):
                          float(dq[i] @ dc[j])
                          for i, s in enumerate(src_cols)
                          for j, t in enumerate(tgt_cols)}
            else:
                sv = {s["column"]: sparse_vec(s, metric, corpus, args.epsilon) for s in src_cols}
                tv = {t["column"]: sparse_vec(t, metric, corpus, args.epsilon) for t in tgt_cols}
                scores = {ColumnPair(p["source"]["table"], s, p["target"]["table"], t):
                          cos_sparse(sv[s], tv[t]) for s in sv for t in tv}
            ordered = dict(sorted(scores.items(), key=lambda kv: -kv[1]))
            # Default metric set = the one from the Valentine paper: Precision,
            # Recall and F1 under Hungarian one-to-one, plus MRR,
            # Precision@top-10% and Recall at the size of the ground truth.
            res = MatcherResults(ordered).get_metrics(gt)
            for k, v in res.items():
                agg[metric][k].append(float(v))
                row[f"{metric}__{k}"] = round(float(v), 4)
        per_pair.append(row)

    write_csv(out / "per_pair_metrics.csv", per_pair)
    summary = {m: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for m, d in agg.items()}
    by_dataset = {}
    for metric in metrics:
        for ds in sorted({r["dataset"] for r in per_pair}):
            vals = [r[f"{metric}__F1Score"] for r in per_pair if r["dataset"] == ds
                    and f"{metric}__F1Score" in r]
            if vals:
                by_dataset.setdefault(ds, {})[metric] = round(float(np.mean(vals)), 4)
    report = {"run_dir": str(run_dir), "config": config, "background": args.background,
              "text_only": args.text_only, "pairs_scored": len(per_pair),
              "pairs_skipped": skipped, "reliability_diagnostic": diag,
              "macro_average": summary, "f1_by_dataset": by_dataset}
    with open(out / "scores.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    keys = ["Precision", "Recall", "F1Score", "PrecisionTop10Percent",
            "RecallAtSizeofGroundTruth", "MeanReciprocalRank"]
    print(f"\nmacro-average over {len(per_pair)} pairs"
          f"{f' ({len(skipped)} skipped)' if skipped else ''}:")
    print("  %-22s %s" % ("metric", " ".join(f"{k[:9]:>10}" for k in keys)))
    for metric in sorted(summary, key=lambda m: -summary[m].get("F1Score", 0)):
        print("  %-22s %s" % (metric, " ".join(
            f"{summary[metric].get(k, float('nan')):>10.4f}" for k in keys)))
    print(f"\nF1 by dataset family:")
    for ds, d in by_dataset.items():
        best = max(d, key=lambda m: d[m])
        print(f"  {ds:<12} best={best} ({d[best]:.4f})  dense={d.get('dense', float('nan')):.4f}")
    print(f"\nwrote {out/'scores.json'} and {out/'per_pair_metrics.csv'}")


if __name__ == "__main__":
    main()
