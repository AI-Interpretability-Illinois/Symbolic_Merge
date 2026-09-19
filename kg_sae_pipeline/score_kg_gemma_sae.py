#!/usr/bin/env python3
"""Rank KG entities with dense, stable-SAE, and DNA-weighted signatures."""

import argparse
import json
from pathlib import Path

from kg_sae_utils import (
    build_background,
    build_dna_signatures,
    load_queries,
    load_representations,
    score_representations,
    summarize_ranks,
)


METHODS = [
    "dense_mean",
    "sae_mean",
    "sae_stable_0.20",
    "sae_stable_0.40",
    "sae_stable_0.60",
    "sae_stable_0.80",
    "sae_stable_1.00",
    "sae_info",
    "sae_info_reliability",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output")
    parser.add_argument("--background", choices=("target", "all"), default="target")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    query_path = Path(args.queries).expanduser().resolve()
    representation_dir = run_dir / "representations"
    queries = load_queries(query_path)
    representations = load_representations(representation_dir, queries)

    source_ids = {str(query["src"]) for query in queries}
    target_ids = {str(candidate) for query in queries for candidate in query["candidates"]}
    required_ids = source_ids | target_ids
    missing = required_ids - representations.keys()
    if missing:
        raise RuntimeError(f"Missing {len(missing)} representations: {list(missing)[:5]}")

    source_records = [representations[entity_id] for entity_id in source_ids]
    target_records = [representations[entity_id] for entity_id in target_ids]
    background_records = target_records if args.background == "target" else source_records + target_records
    background_counts, background_contexts = build_background(background_records)
    dna_signatures = {
        id(record): build_dna_signatures(
            record, background_counts, background_contexts, args.epsilon
        )
        for record in representations.values()
    }

    report = {
        "background": args.background,
        "background_entities": len(background_records),
        "background_contexts": background_contexts,
        "num_queries": len(queries),
        "num_sources": len(source_ids),
        "num_targets": len(target_ids),
        "candidate_count_min": min(len(query["candidates"]) for query in queries),
        "candidate_count_max": max(len(query["candidates"]) for query in queries),
        "metrics": {},
    }

    for method in METHODS:
        ranks = []
        per_query = []
        for query_index, query in enumerate(queries):
            source_id = str(query["src"])
            gold_id = str(query["gold"])
            candidates = [str(candidate) for candidate in query["candidates"]]
            scored = [
                (
                    score_representations(
                        representations[source_id],
                        representations[target_id],
                        method,
                        dna_signatures,
                    ),
                    order,
                    target_id,
                )
                for order, target_id in enumerate(candidates)
            ]
            scored.sort(key=lambda item: (-item[0], item[1]))
            ranking = [item[2] for item in scored]
            gold_rank = ranking.index(gold_id) + 1
            ranks.append(gold_rank)
            per_query.append({
                "query_id": query.get("query_id", query_index),
                "source": source_id,
                "gold": gold_id,
                "gold_rank": gold_rank,
                "top1": ranking[0],
                "correct_top1": ranking[0] == gold_id,
                "ranking": [
                    {
                        "rank": rank,
                        "target": target_id,
                        "score": score,
                        "is_gold": target_id == gold_id,
                    }
                    for rank, (score, _, target_id) in enumerate(scored, 1)
                ],
            })
        report["metrics"][method] = {
            **summarize_ranks(ranks),
            "ranks": ranks,
            "per_query": per_query,
        }

    output_path = Path(args.output).expanduser().resolve() if args.output else run_dir / "kg_dna_scores.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 92)
    print(
        f"queries={len(queries)} sources={len(source_ids)} targets={len(target_ids)} "
        f"candidates={report['candidate_count_min']}..{report['candidate_count_max']} "
        f"{args.background}-bg-contexts={background_contexts}"
    )
    for method, result in report["metrics"].items():
        print(
            f"{method:>24}  H@1={result['H@1']:.4f}  H@5={result['H@5']:.4f} "
            f" H@10={result['H@10']:.4f}  MRR={result['MRR']:.4f} "
            f" MeanRank={result['MeanRank']:.4f}"
        )
    print("SAVED:", output_path)


if __name__ == "__main__":
    main()
