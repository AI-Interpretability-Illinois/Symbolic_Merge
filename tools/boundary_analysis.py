#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boundary_analysis.py

Where does the neural signature help, and where should a symbolic rule take over?

Challenge 3 of the agenda asks for a principled boundary between neural and
symbolic components: some identity is governed by graded similarity ("family
resemblance"), some by exact rules, where a near miss is simply wrong. This
script turns that distinction into a measurement by splitting each benchmark's
symbols into classes and reporting accuracy per class.

valentine  classes come from the sampled cell values of the source column:
           numeric, date, identifier/code, single-token categorical, free text.
           Numbers, dates and identifiers are format-governed; free text and
           categories are graded. Reported: share of gold pairs ranked first.

bioml      classes come from the surface relation between the source and gold
           labels: identical, containment, shared tokens, or disjoint (the
           eponym case, e.g. "Infantile Cortical Hyperostosis" vs "Caffey
           disease"). Reported: MRR per class, read from per_query_ranks.csv.

Example:
    python tools/boundary_analysis.py --benchmark valentine \
        --run_dir <valentine run> --output_dir <out>
    python tools/boundary_analysis.py --benchmark bioml \
        --ranks <analysis>/per_query_ranks.csv --output_dir <out>
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent

DATE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2})?$")
NUMBER = re.compile(r"^[-+]?\d[\d,]*\.?\d*([eE][-+]?\d+)?$")
CODEY = re.compile(r"^(?=.*\d)[A-Za-z0-9._:+/-]{2,}$")


def classify_values(values):
    """Format-governed vs graded, from a column's sampled values."""
    vals = [str(v).strip() for v in values if str(v).strip()]
    if not vals:
        return "empty"
    frac = lambda p: sum(bool(p(v)) for v in vals) / len(vals)
    if frac(lambda v: NUMBER.match(v)) > 0.8:
        return "numeric"
    if frac(lambda v: DATE.match(v)) > 0.6:
        return "date"
    if frac(lambda v: CODEY.match(v) and " " not in v) > 0.6:
        return "identifier"
    if frac(lambda v: " " in v.strip()) > 0.5:
        return "free_text"
    return "categorical"


GRADED = {"free_text", "categorical"}
RULE_GOVERNED = {"numeric", "date", "identifier"}


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()


def label_relation(src, gold):
    a, b = norm(src), norm(gold)
    sa, sb = set(a), set(b)
    if " ".join(a) == " ".join(b):
        return "identical"
    if " ".join(a) in " ".join(b) or " ".join(b) in " ".join(a):
        return "containment"
    if sa & sb:
        return "shared_tokens"
    return "disjoint"


def run_valentine(args):
    sys.path.insert(0, str(REPO / "schema"))
    import eval_valentine_sae as E
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    per_class = defaultdict(lambda: defaultdict(list))
    rows = []
    for f in sorted((Path(args.run_dir) / "pairs").glob("*.pt")):
        p = E.load_pt(f)
        src, tgt = p["source"]["columns"], p["target"]["columns"]
        if not src or not tgt:
            continue
        size = int(src[0]["sae"]["size"])
        corpus = E.corpus_stats(tgt, size)
        by_name = {c["column"]: c for c in src}
        tgt_names = [c["column"] for c in tgt]
        for s_name, t_name in p["ground_truth"]:
            s = by_name.get(s_name)
            if s is None or t_name not in tgt_names:
                continue
            cls = classify_values(s.get("values", []))
            for m in metrics:
                if m.startswith("dense"):
                    dq, dc = E.dense_variants([s], tgt, m)
                    scores = (dq @ dc.T)[0]
                else:
                    sv = E.sparse_vec(s, m, corpus, 1e-6)
                    scores = np.array([E.cos_sparse(sv, E.sparse_vec(t, m, corpus, 1e-6))
                                       for t in tgt])
                top1 = tgt_names[int(np.argmax(scores))]
                per_class[cls][m].append(1.0 if top1 == t_name else 0.0)
            rows.append({"pair": p["pair"]["name"], "column": s_name, "class": cls})
    return per_class, rows, "top-1 accuracy"


def run_bioml(args):
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    per_class = defaultdict(lambda: defaultdict(list))
    rows = []
    for r in csv.DictReader(open(args.ranks, encoding="utf-8")):
        cls = label_relation(r["source_label"], r["gold_label"])
        for m in metrics:
            key = f"{m}_rank"
            if key in r:
                per_class[cls][m].append(1.0 / int(r[key]))
        rows.append({"qid": r["qid"], "source_label": r["source_label"],
                     "gold_label": r["gold_label"], "class": cls})
    return per_class, rows, "MRR"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=("valentine", "bioml"))
    ap.add_argument("--run_dir", default=None, help="valentine run dir")
    ap.add_argument("--ranks", default=None, help="bioml per_query_ranks.csv")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--metrics", default=None)
    args = ap.parse_args()
    if args.metrics is None:
        args.metrics = ("dense,sae_idf,sae_info_reliability" if args.benchmark == "valentine"
                        else "dense_mean,sae_idf,sae_info_reliability")

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    per_class, rows, unit = (run_valentine(args) if args.benchmark == "valentine"
                             else run_bioml(args))
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print("=" * 100)
    print(f"NEURAL / SYMBOLIC BOUNDARY -- {args.benchmark} ({unit})")
    print("=" * 100)
    header = f"{'class':<16} {'n':>6} " + " ".join(f"{m[:20]:>21}" for m in metrics)
    print(header)
    summary = {}
    order = ([c for c in ("numeric", "date", "identifier", "categorical", "free_text")
              if c in per_class] if args.benchmark == "valentine"
             else [c for c in ("identical", "containment", "shared_tokens", "disjoint")
                   if c in per_class])
    for cls in order:
        d = per_class[cls]
        n = max(len(v) for v in d.values()) if d else 0
        summary[cls] = {m: (float(np.mean(d[m])) if d.get(m) else None) for m in metrics}
        summary[cls]["n"] = n
        cells = " ".join(f"{summary[cls][m]:>21.4f}" if summary[cls][m] is not None
                         else f"{'n/a':>21}" for m in metrics)
        print(f"{cls:<16} {n:>6} {cells}")

    # gap between the graded and rule-governed regimes
    if args.benchmark == "valentine":
        for label, group in (("graded", GRADED), ("rule_governed", RULE_GOVERNED)):
            vals = {m: [x for c in group if c in per_class for x in per_class[c].get(m, [])]
                    for m in metrics}
            if any(vals.values()):
                summary[label] = {m: (float(np.mean(v)) if v else None) for m, v in vals.items()}
                summary[label]["n"] = max(len(v) for v in vals.values())
                cells = " ".join(f"{summary[label][m]:>21.4f}" if summary[label][m] is not None
                                 else f"{'n/a':>21}" for m in metrics)
                print(f"{label:<16} {summary[label]['n']:>6} {cells}")

    (out / f"boundary_{args.benchmark}.json").write_text(
        json.dumps({"unit": unit, "summary": summary}, indent=2))
    with (out / f"boundary_{args.benchmark}_items.csv").open("w", newline="",
                                                             encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}/boundary_{args.benchmark}.json")


if __name__ == "__main__":
    main()
