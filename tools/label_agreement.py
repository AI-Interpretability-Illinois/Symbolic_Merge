#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_agreement.py

How reliable are the explanation-type labels (concept / surface / notation / generic)?
interpret_matches.py had one LLM assign a type to every Neuronpedia explanation that
appears in a decomposed match. This script has a second, unrelated model (GPT-5.6 through
the OpenAI Responses API) label the same explanations with the same prompt, and reports
raw agreement and Cohen's kappa, overall and per type.

Example:
    python tools/label_agreement.py --types <matches>/explanation_types.json \
        --output <matches>/explanation_types_second_labeler.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "bioml"))
import interpret_matches as IM  # noqa: E402  (TYPE_PROMPT, TYPES, load_metadata)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", type=Path, required=True, help="explanation_types.json from interpret_matches.py")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--key_file", default="~/.config/symbolic_merge/openai_key")
    ap.add_argument("--base_url", default="https://xiaocong-resource.services.ai.azure.com/openai/v1")
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    first = {int(k): v for k, v in json.loads(args.types.read_text()).items()}
    expl, _ = IM.load_metadata()
    from generate_bioml_contexts_claude import OpenAIClient
    client = OpenAIClient(args.key_file, args.base_url, effort="low")
    second = json.loads(args.output.read_text()) if args.output.is_file() else {}
    todo = [f for f in sorted(first) if str(f) not in second and f in expl]
    print(f"{len(first)} first-labeler types; {len(second)} second-labeler cached; {len(todo)} to label", flush=True)

    def label(batch):
        lines = "\n".join(f"{i + 1}. {expl[f]}" for i, f in enumerate(batch))
        for _ in range(3):
            msg = client.create(model=args.model, max_tokens=4000, system="You label text. Answer with JSON only.",
                                messages=[{"role": "user", "content": IM.TYPE_PROMPT + lines}])
            text = msg.content[0].text
            m = re.search(r"\{.*\}", text, re.S)
            try:
                types = json.loads(m.group(0))["types"]
            except Exception:
                continue
            if len(types) == len(batch) and all(t in IM.TYPES for t in types):
                return dict(zip(batch, types))
        return {}

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    with ThreadPoolExecutor(args.workers) as ex:
        for k, res in enumerate(ex.map(label, batches), 1):
            second.update({str(f): t for f, t in res.items()})
            args.output.write_text(json.dumps(second))
            if k % 10 == 0:
                print(f"  {k}/{len(batches)} batches", flush=True)

    both = [(first[f], second[str(f)]) for f in first if str(f) in second]
    n = len(both)
    agree = sum(a == b for a, b in both)
    pa, pb = Counter(a for a, _ in both), Counter(b for _, b in both)
    pe = sum(pa[t] * pb[t] for t in IM.TYPES) / (n * n)
    po = agree / n
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    per_type = {t: {"first": pa[t], "second": pb[t],
                    "agree_given_first": sum(1 for a, b in both if a == t and b == t) / max(1, pa[t])} for t in IM.TYPES}
    confusion = Counter(both)
    report = {"n": n, "agreement": po, "kappa": kappa, "per_type": per_type,
              "confusion": {f"{a}->{b}": c for (a, b), c in sorted(confusion.items())},
              "first_labeler": "claude-opus-5 (interpret_matches.py)", "second_labeler": args.model}
    (args.output.parent / "explanation_types_agreement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
