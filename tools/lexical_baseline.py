#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lexical_baseline.py

Surface-string baselines for any XLCoST-format jsonl (docstring_tokens = query,
code_tokens = candidate, relevance by idx prefix): TF-IDF cosine over word
tokens and over character n-grams, and BM25 over word tokens. sae_idf is TF-IDF
over SAE features, so TF-IDF over the strings themselves is the control that
says whether the features, rather than the weighting, carry the result.

Reports MRR (first relevant) and Hits@1/5 with the query's own row in the pool.

Example:
    python tools/lexical_baseline.py --dataset_file <minif2f_xsys>/Isabelle/all.jsonl
"""

import argparse
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def bm25_scores(q_docs, c_docs, k1=1.2, b=0.75):
    from collections import Counter
    c_tf = [Counter(d) for d in c_docs]
    lens = np.array([len(d) for d in c_docs], dtype=float)
    avg = lens.mean()
    df = Counter(t for d in c_docs for t in set(d))
    n = len(c_docs)
    idf = {t: np.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}
    out = np.zeros((len(q_docs), n))
    for i, q in enumerate(q_docs):
        for t in set(q):
            if t not in idf:
                continue
            for j, tf in enumerate(c_tf):
                f = tf.get(t, 0)
                if f:
                    out[i, j] += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[j] / avg))
    return out


def evaluate(S, q_ids, c_ids):
    ranks = []
    for i, q in enumerate(q_ids):
        order = np.argsort(-S[i], kind="stable")
        ranks.append(next(r + 1 for r, j in enumerate(order) if c_ids[j] == q))
    ranks = np.array(ranks)
    return {"MRR": float(np.mean(1 / ranks)), "Hits@1": float(np.mean(ranks <= 1)),
            "Hits@5": float(np.mean(ranks <= 5)), "n": len(ranks)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_file", required=True)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dataset_file, encoding="utf-8") if l.strip()]
    q = [" ".join(r["docstring_tokens"]) for r in rows]
    c = [" ".join(r.get("code_tokens") or r["function_tokens"]) for r in rows]
    ids = [r["idx"].split("/")[0] for r in rows]

    res = {}
    for name, vec in [
        ("tfidf_word", TfidfVectorizer(token_pattern=r"[^\s()\[\]{},:;\"`]+", sublinear_tf=True)),
        ("tfidf_char3-5", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ]:
        vec.fit(q + c)
        S = (vec.transform(q) @ vec.transform(c).T).toarray()
        res[name] = evaluate(S, ids, ids)
    tok = lambda s: re.findall(r"[^\s()\[\]{},:;\"`]+", s)
    res["bm25_word"] = evaluate(bm25_scores([tok(s) for s in q], [tok(s) for s in c]), ids, ids)
    for k, v in res.items():
        print(f"[{k:<14}] MRR={v['MRR']:.4f} H@1={v['Hits@1']:.4f} H@5={v['Hits@5']:.4f} n={v['n']}")
    if args.output_json:
        json.dump(res, open(args.output_json, "w"), indent=2)


if __name__ == "__main__":
    main()
