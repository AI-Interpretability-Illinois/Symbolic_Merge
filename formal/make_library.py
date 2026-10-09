#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_library.py

Held-out same-system libraries for the idf (S5 of paper/PREREG_extras_2026-10-04.md): a seeded
sample of theorem statements from a prover's own library, written in the XLCoST row format that
xlcost/run_xlcost_sae.py reads, so the SAE codes of the library can serve as the reference
collection D for idf instead of the test candidate pool.

  Metamath   set.mm: every $p statement with the $e hypotheses of its enclosing ${ $} block, rendered
             exactly like the miniF2F Metamath statements ("h0 $e |- ... $. thm $p |- ... $= ? $.")
  HOL Light  the term of every `prove (`...`, ...)` in the distribution's .ml files

Example:
    python formal/make_library.py --set_mm <dir>/set.mm --hol_light_zip <dir>/hol-light.zip \
        --output_dir /projects/biro/xiaocong/data/formal/libraries --n 3000 --seed 0
"""
import argparse
import json
import random
import re
import zipfile
from pathlib import Path


def metamath_statements(path):
    """(hypotheses, assertion) for every $p statement, hypotheses from the enclosing ${ $} blocks."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\$\(.*?\$\)", " ", text, flags=re.S)          # comments
    tokens = text.split()
    out, stack, i = [], [[]], 0
    while i < len(tokens):
        t = tokens[i]
        if t == "${":
            stack.append([]); i += 1
        elif t == "$}":
            stack.pop(); i += 1
        elif i + 1 < len(tokens) and tokens[i + 1] in ("$e", "$p", "$a", "$f", "$d"):
            kind = tokens[i + 1]
            j = i + 2
            while j < len(tokens) and tokens[j] not in ("$.", "$="):
                j += 1
            body = tokens[i + 2:j]
            if kind == "$e":
                stack[-1].append(" ".join(body))
            elif kind == "$p":
                hyps = [h for frame in stack for h in frame]
                out.append((hyps, " ".join(body)))
            while j < len(tokens) and tokens[j] != "$.":     # skip the proof
                j += 1
            i = j + 1
        elif t == "$d":
            j = i + 1
            while j < len(tokens) and tokens[j] != "$.":
                j += 1
            i = j + 1
        else:
            i += 1
    return out


def render_metamath(hyps, assertion):
    parts = [f"h{k} $e {h} $." for k, h in enumerate(hyps)]
    parts.append(f"thm $p {assertion} $= ? $.")
    return " ".join(parts)


def hol_light_terms(zip_path):
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".ml") or "/Examples/" in name or "/Tutorial/" in name:
                continue
            src = z.read(name).decode("utf-8", errors="replace")
            for m in re.finditer(r"prove\s*\(\s*`([^`]*)`", src):
                term = " ".join(m.group(1).split())
                if 8 <= len(term) <= 600:
                    out.append(term)
    return out


def write_rows(statements, system, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for k, s in enumerate(statements):
            toks = s.split()
            f.write(json.dumps({"url": f"lib_{k}", "idx": f"lib_{k}-Library/lib_{k}-{system}",
                                "docstring_tokens": toks, "code_tokens": toks}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set_mm", type=Path, required=True)
    ap.add_argument("--hol_light_zip", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_chars", type=int, default=1500)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    mm = [render_metamath(h, a) for h, a in metamath_statements(args.set_mm)]
    mm = sorted({s for s in mm if len(s) <= args.max_chars})
    hol = sorted({t for t in hol_light_terms(args.hol_light_zip)})
    print(f"set.mm theorems: {len(mm)}; HOL Light terms: {len(hol)}")
    for system, pool in (("Metamath", mm), ("HOLLight", hol)):
        sample = rng.sample(pool, min(args.n, len(pool)))
        out = args.output_dir / system / "library.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_rows(sample, system, out)
        print(f"{system}: wrote {len(sample)} statements to {out}; e.g. {sample[0][:160]}")


if __name__ == "__main__":
    main()
