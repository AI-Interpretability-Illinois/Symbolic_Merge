#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_prover_pairs.py

Every ordered pair of miniF2F provers as an XLCoST-format retrieval file, from the
statements.json that prepare_minif2f.py wrote (one entry per prover and problem, with
everything identifying the problem already stripped). The Lean 4 -> X files it wrote
are the same format, so the runner and evaluator work unchanged; this adds the other
nine directions so that the merge / transitivity test can run over all four systems.

    <output_dir>/<Src>_<Tgt>/all.jsonl   url = problem id,
                                        idx = <id>-<Src>/<id>-<Tgt>,
                                        docstring_tokens = source statement, code_tokens = target

Example:
    python formal/make_prover_pairs.py --statements <minif2f_xsys>/statements.json --output_dir <minif2f_pairs>
"""
import argparse
import json
from itertools import permutations
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--statements", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--skip_source", default="Lean4",
                    help="source prover whose files already exist (comma-separated; '' = none)")
    args = ap.parse_args()
    systems = json.loads(args.statements.read_text(encoding="utf-8"))
    skip = {s for s in args.skip_source.split(",") if s}
    for src, tgt in permutations(sorted(systems), 2):
        if src in skip:
            continue
        shared = sorted(set(systems[src]) & set(systems[tgt]))
        out = args.output_dir / f"{src}_{tgt}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "all.jsonl", "w", encoding="utf-8") as f:
            for p in shared:
                f.write(json.dumps({"url": p, "idx": f"{p}-{src}/{p}-{tgt}",
                                    "docstring_tokens": systems[src][p]["text"].split(),
                                    "code_tokens": systems[tgt][p]["text"].split()}, ensure_ascii=False) + "\n")
        print(f"{src:>9} -> {tgt:<9} {len(shared)} problems -> {out / 'all.jsonl'}")


if __name__ == "__main__":
    main()
