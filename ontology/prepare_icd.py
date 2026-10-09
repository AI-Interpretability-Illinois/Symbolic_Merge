#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_icd.py

ICD-9-CM -> ICD-10-CM diagnosis code matching through the 2018 CMS/NCHS General
Equivalence Mappings (GEMs). The symbols are opaque codes ("001.0" vs "A00.0"):
nothing about the string says which code in the other system is the same
diagnosis, so all meaning has to come from context.

Gold pairs are the GEM entries that are exact and one-to-one in BOTH directions
(forward I9gem and backward I10gem agree, approximate flag 0, no combination or
choice lists): 3,515 pairs, from which --n are sampled with a fixed seed. Every
sampled ICD-9 code is ranked against all sampled ICD-10 codes.

Each code is given to Claude with its system, official long description and the
description of its 3-character category, the ICD analogue of Bio-ML's ontology
metadata; the code, written in its standard dotted form, is the symbol inserted
into the contexts. The Bio-ML max-diverse prompt is used unchanged.

Inputs (--data_dir): 2018_I9gem.txt, 2018_I10gem.txt,
icd-9-cm-v32-master-descriptions/CMS32_DESC_LONG_DX.txt,
2018-icd-10-code-descriptions/icd10cm_codes_2018.txt.

Output: <output_dir>/{selected_queries.json, contexts/{src,tgt}/*.json}.

Example:
    python ontology/prepare_icd.py --data_dir <icd> --output_dir <out> --n 1000
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bioml"))
import generate_bioml_contexts_claude as G  # noqa: E402
from generate_bioml_contexts_bio8b_maxdiverse import (  # noqa: E402
    atomic_json_dump, complete_cache, entity_output_path)


def dot9(c):
    """ICD-9-CM diagnosis display form: E-codes split after 4 chars, others after 3."""
    k = 4 if c.startswith("E") else 3
    return c if len(c) <= k else f"{c[:k]}.{c[k:]}"


def dot10(c):
    return c if len(c) <= 3 else f"{c[:3]}.{c[3:]}"


def read_desc(path, encoding):
    out = {}
    for line in open(path, encoding=encoding):
        code, _, desc = line.rstrip("\n").partition(" ")
        if code:
            out[code] = desc.strip()
    return out


def exact_pairs(d):
    fwd, bwd, flag = defaultdict(set), defaultdict(set), {}
    for line in open(d / "2018_I9gem.txt"):
        a, b, f = line.split()
        if f[1] == "0":            # skip "no map"
            fwd[a].add(b)
            flag[a, b] = f
    for line in open(d / "2018_I10gem.txt"):
        b, a, f = line.split()
        if f[1] == "0":
            bwd[b].add(a)
    out = []
    for a, bs in fwd.items():
        if len(bs) != 1:
            continue
        b = next(iter(bs))
        # approximate=0, combination=0, and the reverse map is the same single code
        if flag[a, b][0] == "0" and flag[a, b][2] == "0" and bwd.get(b) == {a}:
            out.append((a, b))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    G.add_api_args(ap)
    args = ap.parse_args()
    d, out = args.data_dir, args.output_dir.expanduser().resolve()

    d9 = read_desc(d / "icd-9-cm-v32-master-descriptions" / "CMS32_DESC_LONG_DX.txt", "latin-1")
    d10 = read_desc(d / "2018-icd-10-code-descriptions" / "icd10cm_codes_2018.txt", "utf-8")
    pairs = [(a, b) for a, b in exact_pairs(d) if a in d9 and b in d10]
    sample = sorted(random.Random(args.seed).sample(pairs, min(args.n, len(pairs))))
    print(f"exact bidirectional 1:1 pairs with descriptions: {len(pairs)}; sampled {len(sample)}")

    def meta(code, system):
        if system == "ICD-9-CM":
            shown, desc, cat = dot9(code), d9[code], d9.get(code[:4 if code[0] == "E" else 3])
        else:
            shown, desc, cat = dot10(code), d10[code], d10.get(code[:3])
        iri = f"urn:{system.lower()}:{shown}"
        return {"entity_iri": iri, "ontology": system, "preferred_label": shown,
                "synonyms": [], "definitions": [desc],
                "parent_labels": [cat] if cat and cat != desc else []}

    src = [meta(a, "ICD-9-CM") for a, _ in sample]
    tgt = [meta(b, "ICD-10-CM") for _, b in sample]
    targets = sorted(m["entity_iri"] for m in tgt)
    queries = [{"query_id": i, "src": s["entity_iri"], "gold": t["entity_iri"],
                "candidates": targets} for i, (s, t) in enumerate(zip(src, tgt))]
    atomic_json_dump(queries, out / "selected_queries.json")

    jobs = ([{"side": "src", "iri": m["entity_iri"], "meta": m} for m in src]
            + [{"side": "tgt", "iri": m["entity_iri"], "meta": m} for m in tgt])
    for j in jobs:
        j["path"] = entity_output_path(out, j["side"], j["iri"])
    pending = [j for j in jobs if not complete_cache(j["path"], j["iri"], args.contexts)]
    print(f"entities={len(jobs)} pending={len(pending)}", flush=True)

    tot, failures = G.generate_all(G.make_client(args), args, pending)
    summary = {"entities": len(jobs), **tot, "model": args.model, "effort": args.effort,
               "protocol": "bio8b max-diverse prompt, generated by Claude",
               "failures": failures}
    atomic_json_dump(summary, out / "generation_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} entities failed; rerun to retry them")


if __name__ == "__main__":
    main()
