#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_drug_names.py

Drug brand name -> nonproprietary (generic) name, from the FDA NDC product
directory: two naming systems for the same molecule that share no surface form
("Nesina" / "alogliptin", "Camzyos" / "mavacamten").

Pairs are human prescription products marketed under an NDA or BLA (so the
proprietary name is the originator's brand, not a generic labeller's), with one
active ingredient, where the two names share no word of 3+ letters and neither
contains the other, and that are one-to-one in both directions across the whole
directory. --n of them are sampled with a fixed seed; every sampled brand is
ranked against all sampled generic names.

Each name gets Claude-generated contexts through the Bio-ML max-diverse prompt,
given its naming system and the product's pharmacologic class. One rule is added
and enforced: a context may not name the drug any other way. A reply containing
the counterpart name (or any of its words of 4+ letters) is rejected and
regenerated, so the answer cannot leak through the text.

Output: <output_dir>/{selected_queries.json, contexts/{src,tgt}/*.json}, a
context_run_dir for the Bio-ML cache / encoder / analyzer.

Example:
    python medical/prepare_drug_names.py --product_file <fda>/product.txt \
        --output_dir <out> --n 500
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bioml"))
import generate_bioml_contexts_claude as G  # noqa: E402
from generate_bioml_contexts_bio8b_maxdiverse import (  # noqa: E402
    atomic_json_dump, build_prompt, complete_cache, entity_output_path)

NAMING = {"src": "brand (proprietary) drug name", "tgt": "nonproprietary (generic, INN/USAN) drug name"}

RULE = """
NAMING RULE: this entity is the {system} "{label}". Never write any other name for
this drug anywhere in the fragments: no brand names, no generic or nonproprietary
names, no chemical names, no abbreviations. Refer to it only through what it is
and does."""


def words(s):
    return set(re.findall(r"[a-z]{3,}", s.lower()))


def load_pairs(path):
    b2g, g2b, brand_case, pharm = defaultdict(set), defaultdict(set), {}, {}
    for r in csv.DictReader(open(path, encoding="latin-1"), delimiter="\t"):
        if (r["PRODUCTTYPENAME"] != "HUMAN PRESCRIPTION DRUG"
                or r["MARKETINGCATEGORYNAME"] not in ("NDA", "BLA")):
            continue
        sub = r["SUBSTANCENAME"]
        g, b = r["NONPROPRIETARYNAME"].strip().lower(), r["PROPRIETARYNAME"].strip()
        if not sub or ";" in sub or not g or not b or "," in g or " and " in g:
            continue
        if words(b) & words(g) or b.lower() in g or g in b.lower():
            continue
        b2g[b.lower()].add(g)
        g2b[g].add(b.lower())
        brand_case.setdefault(b.lower(), b)
        if r["PHARM_CLASSES"]:
            pharm.setdefault(g, r["PHARM_CLASSES"])
    pairs = []
    for b, gs in b2g.items():
        g = next(iter(gs))
        if len(gs) == 1 and len(g2b[g]) == 1:
            pairs.append((brand_case[b], g))
    return sorted(pairs), pharm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product_file", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop_unguardable", action="store_true",
                    help="after generation, exclude (and list) pairs where either name still "
                         "has no contexts free of its counterpart")
    G.add_api_args(ap)
    args = ap.parse_args()
    out = args.output_dir.expanduser().resolve()

    pairs, pharm = load_pairs(args.product_file)
    sample = sorted(random.Random(args.seed).sample(pairs, min(args.n, len(pairs))))
    print(f"one-to-one brand/generic pairs: {len(pairs)}; sampled {len(sample)}")

    def meta(name, side, generic):
        classes = [c.strip() for c in pharm.get(generic, "").split(",") if c.strip()]
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return {"entity_iri": f"urn:fda-ndc/{side}/{slug}", "ontology": NAMING[side],
                "preferred_label": name, "synonyms": [], "definitions": [],
                "parent_labels": classes}

    src = [meta(b, "src", g) for b, g in sample]
    tgt = [meta(g, "tgt", g) for _, g in sample]
    counterpart = {}
    for s, t in zip(src, tgt):
        counterpart[s["entity_iri"]] = t["preferred_label"]
        counterpart[t["entity_iri"]] = s["preferred_label"]
    targets = sorted(m["entity_iri"] for m in tgt)
    queries = [{"query_id": i, "src": s["entity_iri"], "gold": t["entity_iri"],
                "candidates": targets} for i, (s, t) in enumerate(zip(src, tgt))]
    atomic_json_dump(queries, out / "selected_queries.json")

    def prompt(m, n):
        system = m["ontology"]
        return build_prompt(m, n) + RULE.format(system=system, label=m["preferred_label"])

    def no_leak(job, contexts):
        other = counterpart[job["iri"]].lower()
        leaked = {w for w in re.findall(r"[a-z]{4,}", other)} | {other}
        for c in contexts:
            text = c["text"].lower()
            hit = next((w for w in leaked if w in text), None)
            if hit:
                return f"mentions counterpart '{hit}'"
        return None

    jobs = ([{"side": "src", "iri": m["entity_iri"], "meta": m} for m in src]
            + [{"side": "tgt", "iri": m["entity_iri"], "meta": m} for m in tgt])
    for j in jobs:
        j["path"] = entity_output_path(out, j["side"], j["iri"])
    pending = [j for j in jobs if not complete_cache(j["path"], j["iri"], args.contexts)]
    print(f"entities={len(jobs)} pending={len(pending)}", flush=True)

    tot, failures = G.generate_all(G.make_client(args), args, pending,
                                   prompt_fn=prompt, validate_fn=no_leak) if pending else ({}, [])

    excluded = []
    if args.drop_unguardable:
        # A name whose counterpart is an everyday word (insulin, oxygen, grass pollen,
        # immune globulin) cannot be described without it; after the retries such a
        # pair is excluded as a whole, and listed, rather than loosening the guard.
        missing = {j["iri"] for j in jobs if not complete_cache(j["path"], j["iri"], args.contexts)}
        kept = [(s, t) for s, t in zip(src, tgt)
                if s["entity_iri"] not in missing and t["entity_iri"] not in missing]
        excluded = [(s["preferred_label"], t["preferred_label"]) for s, t in zip(src, tgt)
                    if (s, t) not in kept]
        if len(kept) < 0.9 * len(src):
            raise SystemExit(f"{len(excluded)} of {len(src)} pairs could not be guarded; not dropping")
        targets = sorted(t["entity_iri"] for _, t in kept)
        queries = [{"query_id": i, "src": s["entity_iri"], "gold": t["entity_iri"],
                    "candidates": targets} for i, (s, t) in enumerate(kept)]
        atomic_json_dump(queries, out / "selected_queries.json")
        failures = []
        print(f"excluded {len(excluded)} pairs whose names cannot avoid the counterpart; "
              f"{len(kept)} pairs remain", flush=True)

    summary = {"entities": len(jobs), **tot, "model": args.model, "effort": args.effort,
               "protocol": "bio8b max-diverse prompt + no-other-name rule, generated by Claude",
               "pairs": len(queries), "excluded_pairs": excluded, "failures": failures}
    atomic_json_dump(summary, out / "generation_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("failures", "excluded_pairs")},
                     indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} entities failed; rerun to retry them, "
                         f"or pass --drop_unguardable to exclude their pairs")


if __name__ == "__main__":
    main()
