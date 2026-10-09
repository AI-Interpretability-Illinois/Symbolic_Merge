#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_merge_transitivity.py

Does pairwise matching compose into one consistent inventory?

Pairwise retrieval scores say nothing about whether alignments across three or
more symbol systems agree, yet a global coordinate system requires exactly
that: if A aligns to B and B to C, the induced A-to-C alignment must be the one
you get directly. This script measures that, and then performs the merge itself.

XLCoST is the natural testbed: one program exists in up to 7 languages, so each
program is a symbol realised in 7 systems with known ground truth. The Java
program-level run already contains all of them -- the query side holds the Java
version of each program and the candidate side holds one other-language version
per row, with idx "<prog>-Java/<prog>-<Lang>".

Reported:
  pairwise           mutual-top-1 accuracy for every ordered language pair
  transitivity       over language triples, how often match(A,B) then match(B,C)
                     agrees with match(A,C) directly
  merge              connected components over mutual-top-1 edges from all
                     pairs, scored against the true program identity
                     (purity, ARI, and component count against the true count)

Metrics compared: dense, sae_idf, sae_idf_universal (and any other weighting the
xlcost eval supports).

Example:
    python tools/global_merge_transitivity.py \
        --run_dir <xlcost java_program run> --output_dir <out> --device cuda
"""

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "xlcost"))


def load_symbols(run_dir, max_programs=0):
    """(program, language) -> representation, from both sides of the run."""
    import eval_xlcost_sae as E
    sides = {}
    for side in ("query", "candidate"):
        docs = []
        for shard in sorted((run_dir / "representations" / side).glob("shard-*.pt")):
            docs.extend(E.load_pt(shard)["rows"])
        sides[side] = docs
    symbols = {}
    for side, docs in sides.items():
        for d in docs:
            m = re.match(r"^(.*)-(\w[\w+#]*)/(.*)-(\w[\w+#]*)$", d["idx"])
            if not m:
                continue
            prog, src_lang, _, tgt_lang = m.groups()
            lang = src_lang if side == "query" else tgt_lang
            symbols.setdefault((prog, lang), d)
    if max_programs:
        keep = sorted({p for p, _ in symbols})[:max_programs]
        keep = set(keep)
        symbols = {k: v for k, v in symbols.items() if k[0] in keep}
    return symbols


def dense_pc1_stats(docs):
    """Mean and first principal direction of the dense vectors of all merged symbols."""
    x = np.stack([d["dense"].numpy().astype(np.float64) for d in docs])
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[0]


def weight_matrix(docs, metric, corpus, size, eps, device, pc1=None):
    """Row-normalised dense matrix of the weighted signatures.

    dense      the averaged hidden state
    dense_pc1  the same after subtracting the mean and removing the first principal direction
               of all merged symbols, (I - u u^T)(d - mu), as for the Dense-PC1 baseline
               (before 2026-10-06 this branch silently returned plain dense)"""
    import eval_xlcost_sae as E
    if metric in ("dense", "dense_pc1"):
        x = np.stack([d["dense"].numpy().astype(np.float64) for d in docs])
        if metric == "dense_pc1":
            mu, u = pc1
            x = x - mu
            x = x - np.outer(x @ u, u)
        rows = torch.from_numpy(x.astype(np.float32)).to(device)
        return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if metric.startswith("dense"):
        raise ValueError(f"unknown dense variant {metric}")
    rows = torch.zeros((len(docs), size), dtype=torch.float32, device=device)
    for i, d in enumerate(docs):
        idx, w = E.doc_weights(d, metric, corpus, eps)
        keep = w > 0
        if keep.any():
            rows[i, torch.from_numpy(idx[keep]).to(device)] = \
                torch.from_numpy(w[keep]).to(device=device, dtype=torch.float32)
    rows /= rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return rows


def mutual_top1(A, B):
    """Indices where each side ranks the other first."""
    s = A @ B.T
    a2b = s.argmax(dim=1)
    b2a = s.argmax(dim=0)
    pairs = {}
    for i in range(A.shape[0]):
        j = int(a2b[i])
        if int(b2a[j]) == i:
            pairs[i] = j
    return pairs


def purity_and_ari(labels_true, labels_pred):
    clusters = defaultdict(list)
    for i, c in enumerate(labels_pred):
        clusters[c].append(labels_true[i])
    n = len(labels_true)
    pur = sum(max(np.bincount(np.unique(v, return_inverse=True)[1])) for v in clusters.values()) / n
    # adjusted Rand index
    from collections import Counter
    ct = defaultdict(Counter)
    for t, p in zip(labels_true, labels_pred):
        ct[p][t] += 1
    comb = lambda x: x * (x - 1) / 2
    sum_ij = sum(comb(c) for row in ct.values() for c in row.values())
    a = Counter(labels_pred)
    b = Counter(labels_true)
    sum_a = sum(comb(v) for v in a.values())
    sum_b = sum(comb(v) for v in b.values())
    exp = sum_a * sum_b / comb(n)
    mx = (sum_a + sum_b) / 2
    ari = (sum_ij - exp) / (mx - exp) if mx != exp else 0.0
    return pur, ari


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--metrics", default="dense,sae_idf,sae_idf_universal,sae_info_reliability")
    ap.add_argument("--max_programs", type=int, default=0)
    ap.add_argument("--min_languages", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epsilon", type=float, default=1e-6)
    args = ap.parse_args()

    import eval_xlcost_sae as E
    run_dir = args.run_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    symbols = load_symbols(run_dir, args.max_programs)
    by_lang = defaultdict(dict)
    for (prog, lang), d in symbols.items():
        by_lang[lang][prog] = d
    langs = sorted(by_lang, key=lambda l: -len(by_lang[l]))
    size = int(next(iter(symbols.values()))["sae"]["size"])
    print("=" * 100)
    print("GLOBAL MERGE AND TRANSITIVITY (XLCoST, one program across languages)")
    print("=" * 100)
    print(f"symbols={len(symbols)} languages={len(langs)} device={args.device}")
    for l in langs:
        print(f"  {l:<12} {len(by_lang[l])} programs")

    # Background for the corpus-derived weightings: every symbol we have.
    all_docs = list(symbols.values())
    corpus, _ = E.background(all_docs, "token", size)
    pc1 = dense_pc1_stats(all_docs)

    report = {"symbols": len(symbols), "languages": {l: len(by_lang[l]) for l in langs},
              "metrics": {}}
    for metric in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        progs = {l: sorted(by_lang[l]) for l in langs}
        mats = {l: weight_matrix([by_lang[l][p] for p in progs[l]], metric, corpus,
                                 size, args.epsilon, args.device, pc1=pc1) for l in langs}
        index = {l: {p: i for i, p in enumerate(progs[l])} for l in langs}

        # pairwise mutual-top-1 accuracy, restricted to shared programs
        pair_acc, align = {}, {}
        for a, b in itertools.permutations(langs, 2):
            shared = sorted(set(progs[a]) & set(progs[b]))
            if len(shared) < 10:
                continue
            ia = torch.tensor([index[a][p] for p in shared], device=args.device)
            ib = torch.tensor([index[b][p] for p in shared], device=args.device)
            A, B = mats[a][ia], mats[b][ib]
            pairs = mutual_top1(A, B)
            correct = sum(1 for i, j in pairs.items() if shared[i] == shared[j])
            pair_acc[f"{a}->{b}"] = {"shared": len(shared), "matched": len(pairs),
                                     "correct": correct,
                                     "precision": correct / max(1, len(pairs)),
                                     "coverage": len(pairs) / len(shared)}
            align[(a, b)] = {shared[i]: shared[j] for i, j in pairs.items()}

        # transitivity over triples
        agree = total = 0
        for a, b, c in itertools.permutations(langs, 3):
            if (a, b) not in align or (b, c) not in align or (a, c) not in align:
                continue
            for p, q in align[(a, b)].items():
                r = align[(b, c)].get(q)
                direct = align[(a, c)].get(p)
                if r is None or direct is None:
                    continue
                total += 1
                agree += (r == direct)
        trans = agree / total if total else None

        # global merge: connected components over all mutual-top-1 edges
        nodes = {(l, p): k for k, (l, p) in enumerate((l, p) for l in langs for p in progs[l])}
        parent = list(range(len(nodes)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry: parent[rx] = ry
        for (a, b), mp in align.items():
            for p, q in mp.items():
                union(nodes[(a, p)], nodes[(b, q)])
        keys = list(nodes)
        labels_pred = [find(nodes[k]) for k in keys]
        labels_true = [k[1] for k in keys]
        pur, ari = purity_and_ari(labels_true, labels_pred)
        n_comp = len(set(labels_pred))
        n_true = len({k[1] for k in keys})

        report["metrics"][metric] = {
            "pairwise": pair_acc,
            "mean_pair_precision": float(np.mean([v["precision"] for v in pair_acc.values()])),
            "mean_pair_coverage": float(np.mean([v["coverage"] for v in pair_acc.values()])),
            "transitivity": trans, "transitivity_triples": total,
            "merge_purity": pur, "merge_ari": ari,
            "merge_components": n_comp, "true_symbols": n_true,
            "component_ratio": n_comp / max(1, n_true),
        }
        r = report["metrics"][metric]
        print(f"\n[{metric}]")
        print(f"  pairwise mutual-top-1: precision={r['mean_pair_precision']:.4f} "
              f"coverage={r['mean_pair_coverage']:.4f} over {len(pair_acc)} ordered pairs")
        print(f"  transitivity: {trans if trans is None else f'{trans:.4f}'} "
              f"over {total} composed triples")
        print(f"  merge: purity={pur:.4f} ARI={ari:.4f} components={n_comp} "
              f"(true symbols {n_true}, ratio {r['component_ratio']:.2f})")
        del mats
        if args.device == "cuda":
            torch.cuda.empty_cache()

    (out / "merge_transitivity.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out/'merge_transitivity.json'}")


if __name__ == "__main__":
    main()
