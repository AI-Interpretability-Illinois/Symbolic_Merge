#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_minif2f.py

Cross-prover statement matching on miniF2F: the same competition problem is
formalised independently in Lean 4, Isabelle, Metamath and HOL Light, so each
problem is one symbol realised in four formal systems with known identity.

Only the statement is kept. Everything that would identify the problem without
reading the mathematics is removed: the theorem name (it is the problem id in
every system), proofs, comments, author lines, imports and Metamath labels.
Each system keeps its own surface syntax (Isabelle's \\<le>, Metamath's
Polish-ish token stream, HOL Light's &-casts), which is the heterogeneity the
benchmark is about.

Writes, per target system, <out>/<Target>/<split>.jsonl in the XLCoST
code-to-code format that xlcost/run_xlcost_sae.py and eval_xlcost_sae.py read:
the query side (docstring_tokens) is the Lean 4 statement, the candidate side
(code_tokens) the target statement, url = problem id, idx =
"<pid>-Lean4/<pid>-<Target>" so the XLCoST prefix rule marks exactly one
relevant candidate. Also writes <out>/statements.json with every system.

Example:
    python formal/prepare_minif2f.py \
        --minif2f <data>/miniF2F --minif2f_lean4 <data>/miniF2F-lean4 \
        --output_dir <data>/minif2f_xsys
"""

import argparse
import json
import re
from pathlib import Path

SYSTEMS = ("Lean4", "Isabelle", "Metamath", "HOLLight")


def pid_of(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def lean4_statement(text):
    m = re.search(r"\btheorem\s+\S+(.*?):=\s*(by\b|sorry\b)", text, re.S)
    if not m:
        raise ValueError("no Lean 4 theorem")
    return "theorem" + m.group(1).rstrip()


ISA_STOP = re.compile(r"^\s*(proof|by|apply|sorry|oops|using|unfolding|including|"
                      r"supply|using|nitpick|sledgehammer)\b")


def isabelle_statement(text):
    text = re.sub(r"\(\*.*?\*\)", "", text, flags=re.S)
    m = re.search(r"^\s*(theorem|lemma)\s+[^\s:]+\s*:?", text, re.M)
    if not m:
        raise ValueError("no Isabelle theorem")
    lines = []
    for line in text[m.end():].splitlines():
        if ISA_STOP.match(line) or line.strip() == "end":
            break
        lines.append(line.rstrip())
    body = "\n".join(l for l in lines if l.strip())
    if not body:
        raise ValueError("empty Isabelle statement")
    return m.group(1) + "\n" + body


def metamath_statement(text):
    # Two layouts occur: proven files use $e/$p/$= with $( comments $), open ones
    # use the @e/@p/@= placeholder syntax with $@ terminators.
    text = "\n".join(l for l in text.splitlines() if "Contributed by" not in l)
    if "@{" in text:  # the whole open statement sits inside one $( ... $) block
        text = (text.replace("$(", " ").replace("$)", " ")
                    .replace("@e", "$e").replace("@p", "$p")
                    .replace("@=", "$=").replace("$@", "$."))
    text = re.sub(r"\$\(.*?\$\)", " ", text, flags=re.S)
    hyps = [f"h{i} $e {' '.join(m.split())} $."
            for i, m in enumerate(re.findall(r"\S+\s+\$e\s+(.*?)\s+\$\.", text, re.S))]
    goal = re.search(r"\S+\s+\$p\s+(.*?)\s+\$=", text, re.S)
    if goal is None:
        raise ValueError("no Metamath $p")
    return "\n".join(hyps + [f"thm $p {' '.join(goal.group(1).split())} $= ? $."])


def hollight_statement(text):
    m = re.search(r"`(.*?)`", text, re.S)
    if not m:
        raise ValueError("no HOL Light term")
    return m.group(1).strip()


def load_system(root, pattern, parse):
    out, bad = {}, []
    for p in sorted(root.glob(pattern)):
        split = p.parent.name.lower()
        try:
            out[pid_of(p.stem)] = {"split": split, "text": parse(p.read_text(encoding="utf-8"))}
        except ValueError as e:
            bad.append(f"{p.name}: {e}")
    return out, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minif2f", type=Path, required=True, help="openai/miniF2F checkout")
    ap.add_argument("--minif2f_lean4", type=Path, required=True,
                    help="yangky11/miniF2F-lean4 checkout")
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()

    systems = {}
    for name, root, pattern, parse in [
        ("Lean4", args.minif2f_lean4 / "MiniF2F", "*/*.lean", lean4_statement),
        ("Isabelle", args.minif2f / "isabelle", "*/*.thy", isabelle_statement),
        ("Metamath", args.minif2f / "metamath", "*/*.mm", metamath_statement),
        ("HOLLight", args.minif2f / "hollight", "*/*.ml", hollight_statement),
    ]:
        systems[name], bad = load_system(root, pattern, parse)
        print(f"{name:<9} {len(systems[name])} statements, {len(bad)} unparsed")
        for b in bad[:5]:
            print("   ", b)

    # Guard against the one leak that would make the task trivial.
    for name, stm in systems.items():
        leaks = [p for p, s in stm.items() if p in pid_of(s["text"])]
        if leaks:
            raise RuntimeError(f"{name}: problem id left in {len(leaks)} statements, "
                               f"e.g. {leaks[0]}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "statements.json").write_text(json.dumps(systems, indent=1, ensure_ascii=False))

    lean = systems["Lean4"]
    for tgt in SYSTEMS[1:]:
        pids = sorted(set(lean) & set(systems[tgt]))
        (out / tgt).mkdir(exist_ok=True)
        for split in ("test", "valid", "all"):
            rows = [p for p in pids if split == "all" or lean[p]["split"] == split]
            with open(out / tgt / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for p in rows:
                    f.write(json.dumps({
                        "url": p, "idx": f"{p}-Lean4/{p}-{tgt}",
                        "docstring_tokens": lean[p]["text"].split(),
                        "code_tokens": systems[tgt][p]["text"].split(),
                    }, ensure_ascii=False) + "\n")
            print(f"Lean4 -> {tgt:<9} {split:<5} {len(rows)} problems")


if __name__ == "__main__":
    main()
