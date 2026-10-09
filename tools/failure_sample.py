#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
failure_sample.py

S3 of the pre-registered extras: a seeded, uniform random sample of SAE-IDF *failures* (gold not
ranked first) per main-table task, decomposed exactly like the wins in interpret_matches.py --
the features the query shares with the gold counterpart and with the wrong top-1 candidate, each
named by its Neuronpedia explanation -- plus one sheet of every feature involved, for a human check
of the auto-generated names.

Outputs in --output_dir:
  failures.json        every sampled failure with texts, ranks, margins and shared-feature lists
  feature_sheet.csv    one row per distinct feature: explanation, type, Pile density, Neuronpedia
                       link, where it appeared, and empty "judgement" / "note" columns to fill in
  failures.md          the same, readable: per task, each failure with its features
  summary.json         counts per task and the explanation-type mix of gold- vs runner-shared features

Example (on a host that holds the main-table runs):
    python tools/failure_sample.py --output_dir <main_table>/interpretability/failures --per_task 5 --seed 0
Set --root / --neuronpedia_root when the stores live elsewhere than on Delta.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None, help="main_table root (default: the Delta path in interpret_matches)")
    ap.add_argument("--neuronpedia_root", type=Path, default=None, help="dir holding neuronpedia/ and pile_density_l0_114.json")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--per_task", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_chars", type=int, default=700)
    ap.add_argument("--priority", type=int, default=120, help="features marked for the human check first, by total share")
    args = ap.parse_args()

    if args.neuronpedia_root:
        os.environ["SYMBOLIC_MERGE_PILE_DENSITY"] = str(args.neuronpedia_root / "pile_density_l0_114.json")
    import interpret_matches as IM
    if args.root:
        IM.M = args.root
    if args.neuronpedia_root:
        IM.NP_DIR = args.neuronpedia_root / "neuronpedia" / IM.NP_DIR.name
        IM.PILE_DENSITY = args.neuronpedia_root / "pile_density_l0_114.json"
    IM.subsample = lambda recs, n, rng, keep_all_cases: recs      # keep every record; failures are sampled here

    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    registry = IM.task_registry()
    names = list(registry) if args.tasks == "all" else args.tasks.split(",")
    expl, density = IM.load_metadata()
    types_file = IM.M / "interpretability/matches/explanation_types.json"
    types = {int(k): v for k, v in json.loads(types_file.read_text()).items()} if types_file.is_file() else {}
    loaders = {"xlcost": IM.load_xlcost, "entity": IM.load_entity, "valentine": IM.load_valentine}
    short = lambda s: s if len(s) <= args.max_chars else s[: args.max_chars - 1].rstrip() + "…"

    failures, summary = [], {}
    for name in names:
        task = registry[name]
        t0 = time.time(); print(f"[{name}] loading", flush=True)
        recs = loaders[task["kind"]](task, random.Random(args.seed), 10 ** 9)
        fails = [r for r in recs if r["rank"]["sae_idf"] > 1]
        rng = random.Random(args.seed)
        sample = rng.sample(fails, min(args.per_task, len(fails)))
        for r in sample:
            f, c = IM.contributions(*r["vec"]["sae_idf"])
            rec = {"task": name, "title": task["title"], "family": task["family"], "qid": r["qid"], "set": r["set"],
                   "query": short(r["q"]), "gold": short(r["g"]), "predicted": short(r["runner"]) if r["runner"] else "",
                   "rank": r["rank"], "margin": r["margin"], "cos_gold": float(c.sum()),
                   "shared_gold": IM.feature_rows(f, c, expl, density, args.top_k)}
            if r["runner_vec"] is not None:
                fw, cw = IM.contributions(r["vec"]["sae_idf"][0], r["runner_vec"])
                rec["cos_predicted"] = float(cw.sum())
                rec["shared_predicted"] = IM.feature_rows(fw, cw, expl, density, args.top_k)
            else:
                rec["shared_predicted"] = []
            for side in ("shared_gold", "shared_predicted"):
                for row in rec[side]:
                    row["type"] = types.get(int(row["feature"]))
            failures.append(rec)
        ranks = [r["rank"]["sae_idf"] for r in fails]
        summary[name] = {"queries": len(recs), "failures": len(fails), "sampled": len(sample),
                         "gold_rank_2_to_5": sum(1 for x in ranks if x <= 5) / max(1, len(ranks)),
                         "dense_right_when_we_fail": sum(1 for r in fails if r["rank"]["dense"] == 1) / max(1, len(fails))}
        print(f"[{name}] {len(recs)} queries, {len(fails)} failures, {len(sample)} sampled ({time.time() - t0:.0f}s)", flush=True)

    # the sheet: one row per distinct feature, sorted by its total share of evidence
    feats = defaultdict(lambda: {"total_share": 0.0, "n": 0, "where": []})
    for rec in failures:
        for side, cand in (("shared_gold", rec["gold"]), ("shared_predicted", rec["predicted"])):
            for row in rec[side]:
                e = feats[int(row["feature"])]
                e["total_share"] += float(row["share"]); e["n"] += 1
                e["where"].append((rec["title"], "gold" if side == "shared_gold" else "predicted", rec["query"][:120], cand[:120], float(row["share"])))
    order = sorted(feats, key=lambda k: -feats[k]["total_share"])
    with open(out / "feature_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["priority", "feature", "explanation", "type", "pile_density", "neuronpedia", "n_occurrences", "total_share",
                    "example_task", "example_side", "example_query", "example_candidate", "judgement (accurate / partly / wrong)", "note"])
        for i, k in enumerate(order):
            e = feats[k]; ex = max(e["where"], key=lambda t: t[4])
            dens = density[k] if k < len(density) else None
            w.writerow([1 if i < args.priority else 0, k, expl.get(k, ""), types.get(k, ""), "" if dens is None or dens != dens else f"{dens:.2e}",
                        IM.NP_URL.format(k), e["n"], f"{e['total_share']:.3f}", ex[0], ex[1], ex[2], ex[3], "", ""])

    # the readable version
    lines = ["# SAE-IDF failures, random sample", "",
             f"Seed {args.seed}; {args.per_task} failures sampled uniformly per task from every query whose gold counterpart "
             f"SAE-IDF did not rank first. Features are the {args.top_k} largest contributions to the cosine with the gold "
             "counterpart and with the wrong top-1 candidate (share of that cosine in parentheses; names are Neuronpedia's "
             "automatic explanations). To check the names, fill the `judgement` column of feature_sheet.csv "
             f"(rows with priority = 1 first, {args.priority} of {len(order)} features).", ""]
    for name in names:
        rows = [r for r in failures if r["task"] == name]
        if not rows:
            continue
        s = summary[name]
        lines += [f"## {rows[0]['title']}", "",
                  f"{s['failures']} of {s['queries']} queries fail; gold within the top 5 in {100 * s['gold_rank_2_to_5']:.0f}% of them; "
                  f"Dense is right on {100 * s['dense_right_when_we_fail']:.0f}% of our failures.", ""]
        for i, r in enumerate(rows, 1):
            lines += [f"**{i}. `{r['qid']}`** — gold rank: SAE-IDF {r['rank']['sae_idf']}, Dense {r['rank']['dense']}, unweighted {r['rank']['sae_mean']}; margin {r['margin']:.3f}", "",
                      f"- query: {r['query']}", f"- gold: {r['gold']}", f"- predicted: {r['predicted']}", "",
                      "- shared with gold: " + "; ".join(f"f{x['feature']} “{x['explanation']}” ({x['share']:.2f})" for x in r["shared_gold"]),
                      "- shared with predicted: " + "; ".join(f"f{x['feature']} “{x['explanation']}” ({x['share']:.2f})" for x in r["shared_predicted"]), ""]
    (out / "failures.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "failures.json").write_text(json.dumps(failures, indent=1, ensure_ascii=False), encoding="utf-8")

    # type mix of the features that carry failures, gold side vs predicted side
    mix = {side: Counter(row["type"] or "unlabelled" for rec in failures for row in rec[side]) for side in ("shared_gold", "shared_predicted")}
    summary["_feature_type_mix"] = {k: dict(v) for k, v in mix.items()}
    summary["_distinct_features"] = len(order)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
