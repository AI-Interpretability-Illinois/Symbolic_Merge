#!/usr/bin/env python3
"""DNA-inspired information-weighted ranking for KG Gemma+SAE runs.

The SAE encoder remains frozen. Weighting is post hoc and follows the
research record:

  p_e(f)       = active contexts for entity e / contexts of e
  p_bg(f)      = active target-background contexts / total target contexts
  I(e,f)       = max(log((p_e(f)+eps)/(p_bg(f)+eps)), 0)
  R(e,f)       = 1 / (1 + CV(e,f))
  w_info       = mean_activation * p_e * I
  w_final      = mean_activation * p_e * I * R
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch


def safe(value):
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "entity")[:180]


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def cosine(a, b):
    a = torch.as_tensor(a, dtype=torch.float32).flatten()
    b = torch.as_tensor(b, dtype=torch.float32).flatten()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def sparse_cos(a, b):
    """Cosine for sparse {feature_id: value} signatures."""
    if not a or not b:
        return 0.0
    small, other = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(float(v) * float(other.get(i, 0.0)) for i, v in small.items())
    na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
    nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def get_vector(rep, metric):
    if metric == "dense":
        return rep.get("dense_mean")
    if metric == "sae":
        return rep.get("sae_mean")
    if metric.startswith("stable_"):
        return rep.get("sae_stable", {}).get(metric[len("stable_"):])
    return None


def load_queries(path):
    with open(path, encoding="utf-8") as f:
        queries = json.load(f)
    if isinstance(queries, dict):
        queries = queries.get("queries", queries.get("items", []))
    if not isinstance(queries, list) or not queries:
        raise RuntimeError("selected_queries.json contains no queries")
    for q in queries:
        if not q.get("candidates"):
            raise RuntimeError(f"query {q.get('query_id')} has no candidates")
    return queries


def load_representations(rep_dir, queries):
    reps = {}
    for fp in sorted(rep_dir.glob("*.pt")):
        rep = load_pt(fp)
        if isinstance(rep, dict) and rep.get("entity_iri") is not None:
            reps[str(rep["entity_iri"])] = rep

    # Compatibility fallback for old representation files without entity_iri.
    needed = set()
    for q in queries:
        needed.add(str(q["src"]))
        needed.update(str(x) for x in q["candidates"])
    for fp in sorted(rep_dir.glob("*.pt")):
        for entity_id in needed:
            if entity_id not in reps and safe(entity_id) == fp.stem:
                reps[entity_id] = load_pt(fp)
    return reps


def context_feature_values(rep):
    """Return feature -> per-context activation values, including zeros."""
    contexts = rep.get("contexts", [])
    if not contexts:
        return {}, 0
    values = defaultdict(lambda: [0.0] * len(contexts))
    for ci, context in enumerate(contexts):
        sae = context.get("sae", {})
        idx = torch.as_tensor(sae.get("idx", []), dtype=torch.long).flatten().tolist()
        val = torch.as_tensor(sae.get("val", []), dtype=torch.float32).flatten().tolist()
        for feature_id, activation in zip(idx, val):
            values[int(feature_id)][ci] = float(activation)
    return values, len(contexts)


def build_background(reps):
    """Count feature presence once per context, as defined in the record."""
    counts = Counter()
    total_contexts = 0
    for rep in reps:
        for context in rep.get("contexts", []):
            idx = torch.as_tensor(
                context.get("sae", {}).get("idx", []), dtype=torch.long
            ).flatten().tolist()
            counts.update(set(int(i) for i in idx))
            total_contexts += 1
    return counts, total_contexts


def build_dna_signatures(rep, background_counts, background_total, eps=1e-6):
    """Build information, reliability, and log-odds DNA signatures."""
    values, n_contexts = context_feature_values(rep)
    if not n_contexts or not background_total:
        return {"dna_info": {}, "dna_info_reliability": {}, "dna_logodds": {}}

    signatures = {"dna_info": {}, "dna_info_reliability": {}, "dna_logodds": {}}
    for feature_id, per_context in values.items():
        x = torch.tensor(per_context, dtype=torch.float32)
        p_entity = float((x > 0).sum()) / n_contexts
        p_background = background_counts.get(feature_id, 0) / background_total
        mean_activation = float(x.mean())
        std_activation = float(x.std(unbiased=False))
        cv = std_activation / (abs(mean_activation) + eps)

        # Positive-only enrichment: ubiquitous background features get zero weight.
        information = max(math.log((p_entity + eps) / (p_background + eps)), 0.0)
        reliability = 1.0 / (1.0 + cv)
        w_info = mean_activation * p_entity * information
        w_final = w_info * reliability

        p_entity_clip = min(max(p_entity, eps), 1.0 - eps)
        p_background_clip = min(max(p_background, eps), 1.0 - eps)
        logodds = max(
            math.log(p_entity_clip / (1.0 - p_entity_clip))
            - math.log(p_background_clip / (1.0 - p_background_clip)),
            0.0,
        )
        w_logodds = mean_activation * logodds

        if w_info > 0:
            signatures["dna_info"][feature_id] = w_info
        if w_final > 0:
            signatures["dna_info_reliability"][feature_id] = w_final
        if w_logodds > 0:
            signatures["dna_logodds"][feature_id] = w_logodds
    return signatures


def metric_score(src_rep, tgt_rep, metric, dna_signatures):
    if metric in ("dense", "sae") or metric.startswith("stable_"):
        src_vec = get_vector(src_rep, metric)
        tgt_vec = get_vector(tgt_rep, metric)
        if src_vec is None or tgt_vec is None:
            return None
        return cosine(src_vec, tgt_vec)
    return sparse_cos(dna_signatures[id(src_rep)][metric], dna_signatures[id(tgt_rep)][metric])


def summarize(ranks):
    if not ranks:
        return {"n": 0, "H@1": None, "H@5": None, "H@10": None, "MRR": None, "MeanRank": None}
    n = len(ranks)
    return {
        "n": n,
        "H@1": sum(r <= 1 for r in ranks) / n,
        "H@5": sum(r <= 5 for r in ranks) / n,
        "H@10": sum(r <= 10 for r in ranks) / n,
        "MRR": sum(1.0 / r for r in ranks) / n,
        "MeanRank": sum(ranks) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="SAE output directory containing representations/")
    ap.add_argument("--queries", default=None, help="selected_queries.json; defaults to ../selected_queries.json")
    ap.add_argument("--output", default=None, help="score report; defaults to <run_dir>/scores.json")
    ap.add_argument("--background", choices=("target", "all"), default="target",
                    help="DNA background; target is the documented KG default")
    ap.add_argument("--epsilon", type=float, default=1e-6)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    rep_dir = run_dir / "representations"
    queries_path = Path(args.queries).expanduser().resolve() if args.queries else run_dir.parent / "selected_queries.json"
    output_path = Path(args.output).expanduser().resolve() if args.output else run_dir / "scores.json"
    if not rep_dir.is_dir():
        raise RuntimeError(f"representation directory not found: {rep_dir}")
    if not queries_path.is_file():
        raise RuntimeError(f"query file not found: {queries_path}")

    queries = load_queries(queries_path)
    reps = load_representations(rep_dir, queries)
    source_ids = {str(q["src"]) for q in queries}
    target_ids = {str(c) for q in queries for c in q["candidates"]}
    source_reps = [reps[x] for x in source_ids if x in reps]
    target_reps = [reps[x] for x in target_ids if x in reps]
    background_reps = target_reps if args.background == "target" else source_reps + target_reps
    if not target_reps:
        raise RuntimeError("No target representations found for the query candidates")

    background_counts, background_total = build_background(background_reps)
    dna_signatures = {
        id(rep): build_dna_signatures(rep, background_counts, background_total, args.epsilon)
        for rep in reps.values()
    }

    metrics = [
        "dense", "sae",
        "stable_0.20", "stable_0.40", "stable_0.60", "stable_0.80", "stable_1.00",
        "dna_info", "dna_info_reliability", "dna_logodds",
    ]
    report = {
        "queries": str(queries_path),
        "representations": str(rep_dir),
        "background": args.background,
        "background_entities": len(background_reps),
        "background_contexts": background_total,
        "background_features": len(background_counts),
        "num_queries": len(queries),
        "candidate_count_min": min(len(q["candidates"]) for q in queries),
        "candidate_count_max": max(len(q["candidates"]) for q in queries),
        "num_representations": len(reps),
        "metrics": {},
    }

    for metric in metrics:
        ranks, errors, per_query = [], [], []
        for query in queries:
            source_id = str(query["src"])
            gold_id = str(query["gold"])
            candidates = [str(x) for x in query["candidates"]]
            source_rep = reps.get(source_id)
            if source_rep is None:
                errors.append({"query_id": query.get("query_id"), "error": "missing source representation"})
                continue
            scored = []
            for order, candidate_id in enumerate(candidates):
                target_rep = reps.get(candidate_id)
                score = float("-inf") if target_rep is None else metric_score(
                    source_rep, target_rep, metric, dna_signatures
                )
                scored.append((float("-inf") if score is None else score, order, candidate_id))
            scored.sort(key=lambda item: (-item[0], item[1]))
            ordered_ids = [x[2] for x in scored]
            if gold_id not in ordered_ids:
                errors.append({"query_id": query.get("query_id"), "error": "gold missing"})
                continue
            rank = ordered_ids.index(gold_id) + 1
            ranks.append(rank)
            per_query.append({
                "query_id": query.get("query_id"),
                "source": source_id,
                "gold": gold_id,
                "rank": rank,
                "top1": ordered_ids[0],
                "gold_score": next(score for score, _, eid in scored if eid == gold_id),
            })
        report["metrics"][metric] = {
            **summarize(ranks), "ranks": ranks, "errors": errors, "per_query": per_query
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"queries={len(queries)} representations={len(reps)}")
    print(f"candidates={report['candidate_count_min']}..{report['candidate_count_max']}")
    print(f"DNA background={args.background} entities={len(background_reps)} contexts={background_total} features={len(background_counts)}")
    print(f"SCORES: {output_path}")
    for metric, result in report["metrics"].items():
        print(f"{metric:>22}  n={result['n']:>3}  H@1={result['H@1']!s:<8}  H@5={result['H@5']!s:<8}  H@10={result['H@10']!s:<8}  MRR={result['MRR']!s:<8}  MeanRank={result['MeanRank']!s}")
    if report["candidate_count_max"] <= 1:
        print("WARNING: candidate_count=1; ranking metrics are trivial. Use candidate_count>=50.")
    if report["metrics"]["dna_info_reliability"]["n"] == 0:
        raise RuntimeError("DNA weighting produced no scorable queries; check raw SAE contexts")


if __name__ == "__main__":
    main()
