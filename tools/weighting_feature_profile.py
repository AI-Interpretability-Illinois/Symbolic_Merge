#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weighting_feature_profile.py

What kind of features does each weighting actually select, across benchmarks?

For every symbol in a run we compute per-feature weights under several
weightings, keep each symbol's top features, pool them, and then describe that
pooled set using Neuronpedia metadata that is independent of our pipeline:

    frac_nonzero   how often the feature fires on the Pile (its density)
    description    the published auto-interp explanation
    activations    top-activating examples

The hypothesis this tests: rarity weighting (sae_idf) wins because it selects
low-density, content-bearing features, while the conservation weightings select
high-density boilerplate whose activation is pervasive but uninformative. That
predicts a large gap in median Pile density between the two selected sets, in
every domain.

Works on all three representation schemas by importing the benchmark's own
weight code, so the weights here are exactly the ones used for retrieval:

    bioml      per-entity, views = generated contexts
    xlcost     per document, views = tokens
    valentine  per column,   views = sampled cell values

Requires the run to use an SAE whose indices match the Neuronpedia source
(see tools/verify_sae_vs_neuronpedia.py).

Example:
    python tools/weighting_feature_profile.py \
        --benchmark valentine --run_dir <valentine run> \
        --output_dir <out> --cache_dir <shared np cache>
"""

import argparse
import csv
import json
import math
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
NP_BASE = "https://www.neuronpedia.org"
NP_MODEL = "gemma-2-9b"
NP_SAE = "20-gemmascope-res-131k"
WEIGHTINGS = ("sae_idf", "sae_info", "sae_info_reliability", "sae_mean")

STOPWORDS = {"the", "of", "and", "or", "in", "to", "a", "an", "with", "for", "by",
             "on", "at", "from", "as", "is", "are", "was", "were", "be", "type",
             "name", "value", "column", "table", "code", "data"}


def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


# =============================================================================
# Per-benchmark symbol loading; each yields (label, weights_by_metric)
# =============================================================================

def symbols_valentine(run_dir, limit):
    sys.path.insert(0, str(REPO / "schema"))
    import eval_valentine_sae as E
    cols, labels = [], []
    for f in sorted((run_dir / "pairs").glob("*.pt")):
        p = load_pt(f)
        for side in ("source", "target"):
            for c in p[side]["columns"]:
                cols.append(c)
                labels.append(f'{p[side]["table"]}.{c["column"]}')
        if limit and len(cols) >= limit:
            break
    size = int(cols[0]["sae"]["size"])
    corpus = E.corpus_stats(cols, size)
    for c, lab in zip(cols[:limit or len(cols)], labels):
        yield lab, {m: E.column_weights(c, m, corpus, 1e-6) for m in WEIGHTINGS}, size


def symbols_xlcost(run_dir, limit):
    sys.path.insert(0, str(REPO / "xlcost"))
    import eval_xlcost_sae as E
    docs = []
    for shard in sorted((run_dir / "representations" / "candidate").glob("shard-*.pt")):
        docs.extend(load_pt(shard)["rows"])
        if limit and len(docs) >= limit:
            break
    size = int(docs[0]["sae"]["size"])
    corpus, _ = E.background(docs, "token", size)
    for d in docs[:limit or len(docs)]:
        yield d["idx"], {m: E.doc_weights(d, m, corpus, 1e-6) for m in WEIGHTINGS}, size


def symbols_bioml(run_dirs, limit):
    sys.path.insert(0, str(REPO / "bioml"))
    import analyze_bioml_idf as A
    reps = []
    for run in run_dirs:
        for p in sorted((run / "representations" / "tgt").glob("*.pt")):
            reps.append(load_pt(p))
            if limit and len(reps) >= limit:
                break
        if limit and len(reps) >= limit:
            break
    bg, bg_total, doc_freq, n_docs = A.build_background(reps)
    size = int(reps[0]["contexts"][0]["sae"]["size"])
    for rep in reps:
        sig = A.build_signature(rep, bg, bg_total, doc_freq, n_docs)
        out = {}
        for m in WEIGHTINGS:
            if m == "sae_mean":
                agg = A.norm_sparse(rep["aggregate"]["sae_mean"])
                out[m] = (agg["idx"].numpy(), agg["val"].numpy())
            else:
                out[m] = (sig[m]["idx"].numpy(), sig[m]["val"].numpy())
        yield rep["label"], out, size


# =============================================================================
# Neuronpedia metadata, cached on disk
# =============================================================================

def np_meta(feature, cache_dir, rps=4.0, _state={"next": 0.0}):
    cache = cache_dir / f"{feature}.json"
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            data = None
    else:
        data = None
    if data is None:
        wait = _state["next"] - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _state["next"] = time.monotonic() + 1.0 / rps
        url = f"{NP_BASE}/api/feature/{NP_MODEL}/{NP_SAE}/{feature}"
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                data = json.load(r)
            cache.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            return None
    exps = data.get("explanations") or []
    return {"density": data.get("frac_nonzero"),
            "max_act": data.get("maxActApprox"),
            "explanation": (exps[0].get("description", "").strip() if exps else ""),
            "url": f"{NP_BASE}/{NP_MODEL}/{NP_SAE}/{feature}"}


def lexical(expl, labels):
    """Does the explanation restate the wording of the symbols that selected it?"""
    words = set(w for w in "".join(ch if ch.isalnum() else " " for ch in expl.lower()).split()
                if len(w) > 2 and w not in STOPWORDS)
    if not words:
        return None
    for lab in labels:
        lw = set(w for w in "".join(ch if ch.isalnum() else " " for ch in str(lab).lower()).split()
                 if len(w) > 2 and w not in STOPWORDS)
        if lw & words:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=("bioml", "xlcost", "valentine"))
    ap.add_argument("--run_dir", required=True, help="comma-separated for bioml")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--cache_dir", type=Path,
                    default=Path("/projects/biro/xiaocong/np_cache_shared"))
    ap.add_argument("--symbols", type=int, default=300, help="symbols to profile")
    ap.add_argument("--top_per_symbol", type=int, default=10)
    ap.add_argument("--features_per_weighting", type=int, default=120)
    args = ap.parse_args()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    runs = [Path(x).expanduser().resolve() for x in args.run_dir.split(",")]

    gen = {"valentine": lambda: symbols_valentine(runs[0], args.symbols),
           "xlcost": lambda: symbols_xlcost(runs[0], args.symbols),
           "bioml": lambda: symbols_bioml(runs, args.symbols)}[args.benchmark]()

    print("=" * 100)
    print(f"FEATURE PROFILE BY WEIGHTING -- {args.benchmark}")
    print("=" * 100)
    mass = {m: Counter() for m in WEIGHTINGS}
    who = {m: defaultdict(list) for m in WEIGHTINGS}
    n = 0
    for label, weights, size in gen:
        n += 1
        for m, (idx, w) in weights.items():
            w = np.asarray(w, dtype=np.float64)
            if not len(w) or w.max() <= 0:
                continue
            order = np.argsort(-w)[:args.top_per_symbol]
            norm = w[order].sum() or 1.0
            for j in order:
                if w[j] > 0:
                    mass[m][int(idx[j])] += float(w[j] / norm)
                    who[m][int(idx[j])].append(label)
    print(f"symbols profiled: {n}")

    selected = {m: [f for f, _ in mass[m].most_common(args.features_per_weighting)]
                for m in WEIGHTINGS}
    union = sorted({f for fs in selected.values() for f in fs})
    print(f"features to look up: {len(union)} (cache: {args.cache_dir})")
    meta = {}
    for i, f in enumerate(union, 1):
        meta[f] = np_meta(f, args.cache_dir)
        if i % 100 == 0:
            print(f"  {i}/{len(union)}", flush=True)

    rows, summary = [], {}
    for m in WEIGHTINGS:
        dens, lex, expl_n = [], [], 0
        for f in selected[m]:
            md = meta.get(f)
            if not md:
                continue
            if isinstance(md["density"], (int, float)):
                dens.append(float(md["density"]))
            if md["explanation"]:
                expl_n += 1
                v = lexical(md["explanation"], who[m][f][:20])
                if v is not None:
                    lex.append(v)
            rows.append({"weighting": m, "feature": f,
                         "pooled_weight": round(mass[m][f], 4),
                         "n_symbols": len(who[m][f]),
                         "pile_density": md["density"], "max_act": md["max_act"],
                         "explanation": md["explanation"], "url": md["url"]})
        summary[m] = {
            "features": len(selected[m]),
            "median_pile_density": float(np.median(dens)) if dens else None,
            "mean_pile_density": float(np.mean(dens)) if dens else None,
            "share_density_gt_1pct": float(np.mean([d > 0.01 for d in dens])) if dens else None,
            "share_density_lt_0.1pct": float(np.mean([d < 0.001 for d in dens])) if dens else None,
            "explanation_coverage": expl_n / max(1, len(selected[m])),
            "share_lexical": float(np.mean(lex)) if lex else None,
        }

    with (out / "feature_profile.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(
        {"benchmark": args.benchmark, "symbols": n, "summary": summary}, indent=2))

    print(f"\n{'weighting':<22} {'median density':>15} {'>1%':>7} {'<0.1%':>7} "
          f"{'expl':>6} {'lexical':>8}")
    for m in WEIGHTINGS:
        s = summary[m]
        f = lambda v, p=".3f": "n/a" if v is None else format(v, p)
        print(f"{m:<22} {f(s['median_pile_density'], '.2e'):>15} "
              f"{f(s['share_density_gt_1pct']):>7} {f(s['share_density_lt_0.1pct']):>7} "
              f"{f(s['explanation_coverage']):>6} {f(s['share_lexical']):>8}")
    print("\ntop features by pooled weight:")
    for m in WEIGHTINGS:
        print(f"\n  [{m}]")
        for f in selected[m][:5]:
            md = meta.get(f) or {}
            d = md.get("density")
            print(f"    {f:>7} w={mass[m][f]:7.2f} density={'n/a' if d is None else f'{d:.2e}'}"
                  f"  {(md.get('explanation') or '-')[:74]}")
    print(f"\nwrote {out/'feature_profile.csv'} and {out/'summary.json'}")


if __name__ == "__main__":
    main()
