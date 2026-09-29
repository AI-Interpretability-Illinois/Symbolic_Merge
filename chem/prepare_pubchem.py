#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_pubchem.py

Chemical notation matching: the same molecule written in three symbolic systems,
SMILES (a line notation over atoms and bonds), the IUPAC systematic name (a
natural-language-like nomenclature) and InChI (a layered canonical identifier).
Given a molecule in one notation, retrieve it among all molecules in another. The
strings themselves are the contexts, so no generation is involved.

Compounds are a seeded random sample of PubChem CIDs <= --max_cid, read from the
bulk FTP dumps (CID-SMILES, CID-IUPAC, CID-InChI-Key); a CID is kept only when it
has all three notations and is a single connected structure (no '.' in SMILES),
so salts and mixtures do not make the pairing ambiguous. Nothing identifying (CID,
synonyms, formula field) is shown to the model.

Writes, per ordered notation pair, <out>/<SRC>2<TGT>/all.jsonl in the XLCoST
format read by xlcost/run_xlcost_sae.py, eval_xlcost_sae.py and
tools/lexical_baseline.py, plus <out>/compounds.json.

Example:
    python chem/prepare_pubchem.py --output_dir <data>/pubchem_notation --n 1000
"""

import argparse
import gzip
import json
import random
import urllib.request
from pathlib import Path

FTP = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/{}"
FILES = {"SMILES": "CID-SMILES.gz", "IUPAC": "CID-IUPAC.gz", "InChI": "CID-InChI-Key.gz"}
PAIRS = [("SMILES", "IUPAC"), ("SMILES", "InChI"), ("IUPAC", "InChI")]


def stream(name, max_cid):
    """{cid: value} for cid <= max_cid, streamed from PubChem's CID-sorted bulk dump
    (the REST API rate-limits sampling at this scale)."""
    out = {}
    with urllib.request.urlopen(FTP.format(FILES[name]), timeout=120) as r:
        for line in gzip.open(r, "rt", encoding="utf-8"):
            cid, value = line.rstrip("\n").split("\t")[:2]
            cid = int(cid)
            if cid > max_cid:
                break
            out[cid] = value
    print(f"{name}: {len(out):,} CIDs <= {max_cid:,}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--max_cid", type=int, default=3_000_000,
                    help="sample among CIDs up to this (the dumps are streamed to here)")
    ap.add_argument("--max_chars", type=int, default=200,
                    help="drop compounds whose longest notation exceeds this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cols = {k: stream(k, args.max_cid) for k in FILES}
    kept = {}
    for cid in sorted(set.intersection(*(set(c) for c in cols.values()))):
        row = {k: cols[k][cid] for k in FILES}
        if "." in row["SMILES"] or max(map(len, row.values())) > args.max_chars:
            continue
        kept[cid] = row
    print(f"complete single-component compounds: {len(kept):,}")
    cids = sorted(random.Random(args.seed).sample(sorted(kept), args.n))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "compounds.json").write_text(json.dumps({c: kept[c] for c in cids}, indent=1))
    for a, b in PAIRS:
        d = out / f"{a}2{b}"
        d.mkdir(exist_ok=True)
        with open(d / "all.jsonl", "w") as f:
            for k, c in enumerate(cids):
                # Rows are keyed by position, not CID, so no identifier reaches the model.
                f.write(json.dumps({"url": f"m{k}", "idx": f"m{k}-{a}/m{k}-{b}",
                                    "docstring_tokens": [kept[c][a]],
                                    "code_tokens": [kept[c][b]]}) + "\n")
        print(f"{a} -> {b}: {len(cids)} molecules")


if __name__ == "__main__":
    main()
