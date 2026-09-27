#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_xlcost_sae.py

XLCoST XL code search (code-to-code) retrieval + official scoring for the
SAE pipeline.

Takes the representations written by run_xlcost_sae.py, ranks every candidate
for every query under each scoring metric, writes predictions in the official
format, and scores them with XLCoST's own metric definitions.

Retrieval protocol, copied from XLCoST's code/codesearch/code/run.py test():
  * queries and candidates are the two sides of the same <split>.jsonl
  * the pool is every candidate row, including the query's own row (its match
    in the other language), which the official evaluator counts as relevant
  * predictions keep the top 100 candidates per query
  * a candidate is relevant when idx.split("/")[0] matches the query's own
    idx.split("/")[0]

Scoring metrics (all cosine similarity):
  dense                  mean hidden state over all tokens
  sae_mean               mean SAE activation over all tokens
  sae_stable_<f>         mean, zeroed unless the feature fires in >= f of tokens
  sae_info               mean * p_doc * information
  sae_info_reliability   mean * p_doc * information * 1/(1+CV)   <- our method
  sae_logodds            mean * positive log-odds vs the corpus
  sae_idf                mean * log(N/df): TF-IDF over SAE features

with tokens as the views of a document (p_doc = fraction of the document's
tokens where the feature fires) and the candidate corpus as the background,
mirroring the Bio-ML formulation in
bioml/analyze_bioml_information_weighted_sae_target_bg.py.

Example:
    python xlcost/eval_xlcost_sae.py \
        --run_dir <out>/xlcost_java_program \
        --dataset_file <code2codesearch>/dataset/program_level/Java/test.jsonl \
        --output_dir <out>/xlcost_java_program/eval \
        --official_evaluator <XLCoST>/code/codesearch/code2codesearch/evaluator/evaluator.py
"""

import argparse
import json
import logging
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


# =============================================================================
# Official metrics -- same logic as XLCoST evaluator/evaluator.py
# =============================================================================

def read_answers(filename):
    answers = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            answers[js["url"]] = js["idx"]
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            predictions[js["url"]] = js["answers"]
    return predictions


def calculate_scores(answers, predictions):
    """MRR as XLCoST defines it: mean over ALL relevant hits in the list."""
    scores = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for url {}.".format(key))
            sys.exit()
        query_scores = []
        flag = False
        for rank, idx in enumerate(predictions[key]):
            prediction_prefix = idx.split("/")[0]
            if prediction_prefix == answers[key].split("/")[0]:
                query_scores.append(1 / (rank + 1))
                flag = True
        if flag is False:
            scores.append(0)
        else:
            scores.append(round(np.mean(query_scores), 4))
    return {"MRR": round(float(np.mean(scores)), 4)}


def precison_atk(answers, predictions, k=6):
    scores = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for url {}.".format(key))
            sys.exit()
        precision_count = 0
        iterator_count = 0
        flag = False
        for rank, idx in enumerate(predictions[key]):
            prediction_prefix = idx.split("/")[0]
            if prediction_prefix == answers[key].split("/")[0]:
                precision_count += 1
                flag = True
            iterator_count += 1
            if iterator_count == k:
                break
        if flag is False:
            scores.append(0)
        else:
            scores.append(round(precision_count / k, 4))
    return {"Precision@" + str(k): round(float(np.mean(scores)), 4)}


def extra_diagnostics(answers, predictions):
    """Not part of the official metric; useful next to it."""
    first_rr, hit1, hit5, n_rel = [], [], [], []
    for key, gold in answers.items():
        prefix = gold.split("/")[0]
        hits = [r for r, idx in enumerate(predictions.get(key, []), 1)
                if idx.split("/")[0] == prefix]
        first_rr.append(1.0 / hits[0] if hits else 0.0)
        hit1.append(1.0 if hits and hits[0] == 1 else 0.0)
        hit5.append(1.0 if hits and hits[0] <= 5 else 0.0)
        n_rel.append(len(hits))
    n = max(1, len(first_rr))
    return {
        "MRR_first_hit": round(float(np.mean(first_rr)), 4),
        "Hits@1": round(float(np.mean(hit1)), 4),
        "Hits@5": round(float(np.mean(hit5)), 4),
        "mean_relevant_in_list": round(float(np.mean(n_rel)), 4),
        "queries": n,
    }


# =============================================================================
# Loading representations
# =============================================================================

def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_side(run_dir, side):
    folder = run_dir / "representations" / side
    if not folder.is_dir():
        raise FileNotFoundError(f"missing {folder}")
    docs = []
    for shard in sorted(folder.glob("shard-*.pt")):
        docs.extend(load_pt(shard)["rows"])
    if not docs:
        raise RuntimeError(f"no documents in {folder}")
    docs.sort(key=lambda d: int(d["row"]))
    return docs


def background(docs, unit, size):
    """Corpus statistics used by the information weighting.

    p_bg must be in the SAME unit as p_doc (a token fraction), otherwise the
    log ratio in the information term compares incommensurable quantities and
    clips to zero for most features. Document frequency is returned separately
    for the IDF metric, which needs no such comparison.
    """
    counts = np.zeros(size, dtype=np.float64)
    doc_freq = np.zeros(size, dtype=np.float64)
    total_tokens = 0
    for d in docs:
        idx = d["sae"]["idx"].numpy().astype(np.int64)
        doc_freq[idx] += 1.0
        if unit == "token":
            counts[idx] += d["sae"]["count"].numpy().astype(np.float64)
            total_tokens += int(d["n_tokens"])
        else:
            counts[idx] += 1.0
            total_tokens += 1
    n_docs = max(1, len(docs))
    corpus = {
        "p_bg": counts / max(1, total_tokens),
        "idf_universal": load_universal_idf(size),
        "idf": np.log(n_docs / (1.0 + doc_freq)),
        "doc_freq": doc_freq,
        "n_docs": n_docs,
    }
    return corpus, total_tokens


# =============================================================================
# Weights
# =============================================================================


UNIVERSAL_DENSITY_FILE = "/projects/biro/xiaocong/pile_density_l0_114.json"


def load_universal_idf(size, path=None, floor=1e-6):
    """idf_universal(f) = -log(Pile density of f), from Neuronpedia's feature dump.

    The corpus-derived idf is relative to whichever candidates happen to be in
    the pool. This one is a fixed property of the feature, so signatures from
    systems that never co-occur are still comparable -- the property a global
    coordinate system needs.
    """
    import json as _json
    p = path or UNIVERSAL_DENSITY_FILE
    try:
        raw = _json.load(open(p, encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"universal density file unreadable ({p}): {e}")
    out = np.zeros(size, dtype=np.float64)
    for k, v in raw.items():
        i = int(k)
        if i < size:
            out[i] = -math.log(max(float(v), floor))
    return np.maximum(out, 0.0)

def doc_weights(doc, metric, corpus, eps):
    """Per-feature weights of one document under one metric."""
    p_bg = corpus["p_bg"]
    sae = doc["sae"]
    idx = sae["idx"].numpy().astype(np.int64)
    n = float(doc["n_tokens"])
    s = sae["sum"].numpy().astype(np.float64)
    mean = s / n
    if metric == "sae_mean":
        return idx, mean

    if metric == "sae_idf_universal":
        return idx, mean * corpus["idf_universal"][idx]
    if metric == "sae_idf":
        # TF-IDF over SAE features: no p_doc factor, so nothing compares a
        # token fraction against a document fraction.
        return idx, mean * np.maximum(corpus["idf"][idx], 0.0)

    p_doc = sae["count"].numpy().astype(np.float64) / n
    if metric.startswith("sae_stable_"):
        thr = float(metric[len("sae_stable_"):])
        return idx, np.where(p_doc >= thr, mean, 0.0)

    pb = p_bg[idx]
    if metric == "sae_logodds":
        pe2 = np.clip(p_doc, eps, 1 - eps)
        pb2 = np.clip(pb, eps, 1 - eps)
        lod = np.maximum(np.log(pe2 / (1 - pe2)) - np.log(pb2 / (1 - pb2)), 0.0)
        return idx, mean * lod

    info = np.maximum(np.log((p_doc + eps) / (pb + eps)), 0.0)
    if metric == "sae_info":
        return idx, mean * p_doc * info

    sq = sae["sumsq"].numpy().astype(np.float64)
    var = np.maximum(sq / n - mean * mean, 0.0)
    cv = np.sqrt(var) / (np.abs(mean) + eps)
    return idx, mean * p_doc * info * (1.0 / (1.0 + cv))


def build_matrix(docs, metric, corpus, size, eps):
    from scipy import sparse

    data, indices, indptr = [], [], [0]
    for d in docs:
        idx, w = doc_weights(d, metric, corpus, eps)
        keep = w > 0
        indices.append(idx[keep])
        data.append(w[keep])
        indptr.append(indptr[-1] + int(keep.sum()))
    m = sparse.csr_matrix(
        (np.concatenate(data) if data else np.zeros(0),
         np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64),
         np.array(indptr)),
        shape=(len(docs), size), dtype=np.float64)
    norms = np.sqrt(m.multiply(m).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ m


def dense_matrix(docs):
    x = np.stack([d["dense"].numpy().astype(np.float32) for d in docs])
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms



def _top_pc(x, iters=50):
    """First principal direction of x (rows = samples), by power iteration."""
    v = np.random.default_rng(0).normal(size=x.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        v = x.T @ (x @ v)
        n = np.linalg.norm(v)
        if n == 0:
            return None
        v /= n
    return v


def dense_variants(queries, candidates, metric):
    """dense, dense_centered (subtract the candidate-corpus mean), dense_pc1
    (also project out the corpus's first principal direction).

    The control for the claim that SAE weighting beats dense: pooled hidden
    states share a large common component (cos(h, mean h) ~ 0.9), which
    compresses every dense cosine. Centering removes it without any SAE.
    """
    q = np.stack([d["dense"].numpy().astype(np.float64) for d in queries])
    c = np.stack([d["dense"].numpy().astype(np.float64) for d in candidates])
    if metric != "dense":
        mu = c.mean(0)
        q = q - mu
        c = c - mu
        if metric == "dense_pc1":
            pc = _top_pc(c)
            if pc is not None:
                q = q - np.outer(q @ pc, pc)
                c = c - np.outer(c @ pc, pc)
    qn = np.linalg.norm(q, axis=1, keepdims=True); qn[qn == 0] = 1.0
    cn = np.linalg.norm(c, axis=1, keepdims=True); cn[cn == 0] = 1.0
    return (q / qn).astype(np.float32), (c / cn).astype(np.float32)

def gpu_matrix(docs, metric, corpus, size, eps, device):
    """Same weights as build_matrix, materialised dense on the GPU.

    Sparse-sparse products are slow here because code-syntax features fire in
    nearly every document, so a few features dominate the cost. A dense block
    product is a rounding error on an A100 by comparison.
    """
    rows = torch.zeros((len(docs), size), dtype=torch.float32, device=device)
    for i, d in enumerate(docs):
        idx, w = doc_weights(d, metric, corpus, eps)
        keep = w > 0
        if keep.any():
            rows[i, torch.from_numpy(idx[keep]).to(device)] = \
                torch.from_numpy(w[keep]).to(device=device, dtype=torch.float32)
    rows /= rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return rows.to(torch.float16)


def score_gpu(qm, cm, block=512):
    out = np.zeros((qm.shape[0], cm.shape[0]), dtype=np.float32)
    for lo in range(0, qm.shape[0], block):
        hi = min(qm.shape[0], lo + block)
        out[lo:hi] = (qm[lo:hi] @ cm.T).float().cpu().numpy()
    return out


# =============================================================================
# Ranking
# =============================================================================

def write_predictions(path, scores, queries, candidates, topk, exclude_self):
    """Official prediction format: one line per query, top-k candidate idx."""
    cand_idx = [c["idx"] for c in candidates]
    cand_row = np.array([int(c["row"]) for c in candidates])
    all_rows = np.arange(len(candidates))
    with open(path, "w", encoding="utf-8") as f:
        for qi, q in enumerate(queries):
            row_scores = scores[qi]
            # Drop excluded candidates outright: masking their score would still
            # leave them in the list whenever topk >= pool size.
            allowed = all_rows[cand_row != int(q["row"])] if exclude_self else all_rows
            # Stable sort keeps the dataset order for ties; the official code
            # uses an unstable argsort, so exact tie order can differ.
            order = allowed[np.argsort(-row_scores[allowed], kind="stable")][:topk]
            f.write(json.dumps({"url": q["url"],
                                "answers": [cand_idx[int(j)] for j in order]}) + "\n")


def score_blocks(qm, cm, block=512):
    out = np.zeros((qm.shape[0], cm.shape[0]), dtype=np.float32)
    for lo in range(0, qm.shape[0], block):
        hi = min(qm.shape[0], lo + block)
        part = qm[lo:hi] @ cm.T
        out[lo:hi] = part.toarray() if hasattr(part, "toarray") else part
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True,
                    help="output_dir of run_xlcost_sae.py")
    ap.add_argument("--dataset_file", type=Path, default=None,
                    help="official <Lang>/<split>.jsonl used as the answers file "
                         "(defaults to the dataset_file recorded in run_config.json)")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--metrics",
                    default="dense,dense_centered,dense_pc1,sae_mean,sae_idf,"
                            "sae_idf_universal,sae_info,sae_info_reliability,sae_logodds")
    ap.add_argument("--stable_frequencies", default="0.05,0.10,0.25",
                    help="adds sae_stable_<f> metrics; views are TOKENS, so these "
                         "are much lower than the Bio-ML context thresholds")
    ap.add_argument("--background_unit", choices=("token", "document"), default="token",
                    help="token: p_bg = corpus tokens where the feature fires; "
                         "document: document frequency (IDF-like)")
    ap.add_argument("--topk", type=int, default=100,
                    help="answers per query; 100 matches the official run.py")
    ap.add_argument("--precision_at", type=int, default=6)
    ap.add_argument("--exclude_self", action="store_true",
                    help="drop the query's own row from its candidate pool "
                         "(the official protocol keeps it)")
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="cuda materialises the weight matrices densely on the GPU; "
                         "cpu uses scipy sparse products")
    ap.add_argument("--official_evaluator", type=Path, default=None,
                    help="path to XLCoST evaluator.py; runs it as a cross-check")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    dataset_file = (args.dataset_file or Path(config["dataset_file"])).expanduser().resolve()
    if not dataset_file.is_file():
        raise FileNotFoundError(f"answers file not found: {dataset_file}")
    out = (args.output_dir or (run_dir / "eval")).expanduser().resolve()
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    queries = load_side(run_dir, "query")
    candidates = load_side(run_dir, "candidate")
    size = int(candidates[0]["sae"]["size"])

    print("=" * 100)
    print(f"XLCoST XL CODE SEARCH EVAL  ({config['lang']}, {config['level']} level, "
          f"{config['split']})")
    print("=" * 100)
    print(f"queries={len(queries)} candidates={len(candidates)} sae_width={size}")
    print(f"pooling={config['pooling']}")
    trunc = sum(1 for d in queries + candidates if d.get("truncated"))
    if trunc:
        print(f"[warn] {trunc} documents hit --max_length={config['max_length']} "
              f"and were truncated")
    dup = [u for u, c in Counter(q["url"] for q in queries).items() if c > 1]
    if dup:
        print(f"[warn] {len(dup)} duplicate query urls; the official evaluator keys on "
              f"url, so duplicates collapse (first {dup[:3]})")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"scoring backend: {device}")

    corpus, bg_total = background(candidates, args.background_unit, size)
    print(f"background: unit={args.background_unit} total={bg_total} "
          f"features_seen={int((corpus['p_bg'] > 0).sum())}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for f in [x.strip() for x in args.stable_frequencies.split(",") if x.strip()]:
        metrics.append(f"sae_stable_{float(f):.2f}")

    answers = read_answers(dataset_file)
    report = {
        "run_dir": str(run_dir),
        "dataset_file": str(dataset_file),
        "lang": config["lang"], "level": config["level"], "split": config["split"],
        "pooling": config["pooling"], "sae_path": config["sae_path"],
        "layer": config["layer"], "max_length": config["max_length"],
        "queries": len(queries), "candidates": len(candidates),
        "answers_in_file": len(answers),
        "background_unit": args.background_unit, "topk": args.topk,
        "scoring_backend": device,
        "exclude_self": args.exclude_self, "truncated_documents": trunc,
        "metrics": {},
    }

    dense_q = dense_c = None
    for metric in metrics:
        if metric.startswith("dense"):
            dq, dc = dense_variants(queries, candidates, metric)
            scores = dq @ dc.T
        elif device == "cuda":
            qm = gpu_matrix(queries, metric, corpus, size, args.epsilon, device)
            cm = gpu_matrix(candidates, metric, corpus, size, args.epsilon, device)
            scores = score_gpu(qm, cm)
            del qm, cm
            torch.cuda.empty_cache()
        else:
            qm = build_matrix(queries, metric, corpus, size, args.epsilon)
            cm = build_matrix(candidates, metric, corpus, size, args.epsilon)
            scores = score_blocks(qm, cm)
            del qm, cm

        pred_path = out / "predictions" / f"predictions_{metric}.jsonl"
        write_predictions(pred_path, scores, queries, candidates,
                          args.topk, args.exclude_self)
        del scores

        predictions = read_predictions(pred_path)
        result = {}
        result.update(calculate_scores(answers, predictions))
        result.update(precison_atk(answers, predictions, args.precision_at))
        result.update(extra_diagnostics(answers, predictions))
        result["predictions_file"] = str(pred_path)

        if args.official_evaluator:
            proc = subprocess.run(
                [sys.executable, str(args.official_evaluator.expanduser()),
                 "-a", str(dataset_file), "-p", str(pred_path)],
                capture_output=True, text=True)
            result["official_evaluator_stdout"] = proc.stdout.strip()
            if proc.returncode != 0:
                result["official_evaluator_stderr"] = proc.stderr.strip()[-2000:]

        report["metrics"][metric] = result
        print(f"  {metric:<24} MRR={result['MRR']:.4f} "
              f"P@{args.precision_at}={result['Precision@' + str(args.precision_at)]:.4f} "
              f"Hits@1={result['Hits@1']:.4f} "
              f"rel/query={result['mean_relevant_in_list']:.2f}", flush=True)
        if result.get("official_evaluator_stdout"):
            print(f"  {'':<24} official: {result['official_evaluator_stdout']}")

    with open(out / "scores.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("-" * 100)
    best = max(report["metrics"].items(), key=lambda kv: kv[1]["MRR"])
    print(f"best MRR: {best[0]} = {best[1]['MRR']:.4f}")
    print(f"wrote {out/'scores.json'} and {out/'predictions'}/")


if __name__ == "__main__":
    main()
