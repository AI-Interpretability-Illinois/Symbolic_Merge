#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpret_matches.py

Why does sae_idf match what it matches? A cosine decomposes exactly over the
SAE features two symbols share,

    cos(w(s), w(t)) = sum_f c_f,    c_f = w_f(s) w_f(t) / (|w(s)| |w(t)|),

so every match reads as a short list of named features, each with its share of
the score. This script does that for every main-table task, on exactly the
weights the evaluation used, and describes each feature with metadata that is
independent of our pipeline: Neuronpedia's published explanation and the
feature's activation density on the Pile.

Per task, over a seeded sample of queries whose gold counterpart sae_idf ranks
first (the same pairs are scored with sae_mean, i.e. without idf, as control):
  n50 / n80      shared features needed to reach 50% / 80% of the cosine
  rarity         share of the cosine from features by Pile density band
  feature type   share of the top-k contributions by the type of the feature's
                 explanation (concept, surface, notation, generic), labelled by
                 an LLM from the explanation text alone (--classify)
  top features   the features that carry the most evidence across the task

Case studies are picked by rule, not by hand:
  wins      sae_idf ranks the gold first while dense ranks it >= --min_dense_rank;
            shortest texts first among those with a clear sae_idf margin
  failures  dense ranks the gold first and sae_idf does not; shows the shared
            features that pulled the wrong candidate ahead

All tasks use Gemma Scope layer_20/width_131k/average_l0_114, the SAE whose
feature indices Neuronpedia's explanations refer to.

Example:
    python tools/interpret_matches.py --output_dir <out> --classify
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for sub in ("xlcost", "bioml", "schema"):
    sys.path.insert(0, str(REPO / sub))

M = Path("/projects/biro/xiaocong/main_table")
NP_DIR = Path("/projects/biro/xiaocong/neuronpedia/gemma-2-9b_20-gemmascope-res-131k")
PILE_DENSITY = Path("/projects/biro/xiaocong/pile_density_l0_114.json")
os.environ.setdefault("SYMBOLIC_MERGE_PILE_DENSITY", str(PILE_DENSITY))
NP_URL = "https://www.neuronpedia.org/gemma-2-9b/20-gemmascope-res-131k/{}"
METHODS = ("sae_mean", "sae_idf")
TYPES = ("concept", "surface", "notation", "generic")
EPS = 1e-6


def task_registry():
    def entity(title, family, sae, ctx, ranks):
        return dict(kind="entity", title=title, family=family, sae=sae, ctx=ctx, ranks=ranks)
    ck = lambda c: entity({"nell_dbpedia": "Common-KG NELL → DBpedia",
                           "yago_wikidata": "Common-KG YAGO → Wikidata"}[c], "knowledge graph",
                          M / f"commonkg/native_scored_{c}/L20_w131k_l0_114",
                          M / f"commonkg/native/{c}",
                          M / f"commonkg/native_scored_{c}/L20_w131k_l0_114/analysis/per_query_ranks.csv")
    mf = lambda p, lang: entity(f"MultiFarm {lang} → en", "multilingual ontology",
                                M / "multifarm/native_scored/L20_w131k_l0_114",
                                M / f"multifarm/native/views/{p}",
                                M / f"multifarm/native_scored/L20_w131k_l0_114/analysis_views_{p}/per_query_ranks.csv")
    tasks = {
        f"minif2f_{s.lower()}": dict(kind="xlcost", title=f"miniF2F Lean 4 → {t}", family="formal math",
                                     runs=[M / f"minif2f/lean4_{s}"])
        for s, t in (("Isabelle", "Isabelle"), ("Metamath", "Metamath"), ("HOLLight", "HOL Light"))}
    # The Java run's candidates are the other six languages' versions of each program,
    # so it covers every language; the other runs keep no representations on Delta.
    tasks["xlcost"] = dict(kind="xlcost", title="XLCoST, Java to 6 languages", family="code",
                           runs=[M / "xlcost/Java_program"])
    tasks["commonkg_nell_dbpedia"] = ck("nell_dbpedia")
    tasks["commonkg_yago_wikidata"] = ck("yago_wikidata")
    tasks["multifarm_zh"] = mf("cn-en", "zh")
    tasks["multifarm_ru"] = mf("en-ru", "ru")
    tasks["multifarm_ar"] = mf("ar-en", "ar")
    tasks["valentine"] = dict(kind="valentine", title="Valentine, whole column", family="table",
                              run=M / "valentine_column")
    return tasks


def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def sparse_vec(idx, val):
    idx = np.asarray(idx, dtype=np.int64)
    val = np.asarray(val, dtype=np.float64)
    keep = val > 0
    idx, val = idx[keep], val[keep]
    o = np.argsort(idx)
    return idx[o], val[o]


def cos(a, b):
    _, x, y = np.intersect1d(a[0], b[0], assume_unique=True, return_indices=True)
    na, nb = np.linalg.norm(a[1]), np.linalg.norm(b[1])
    return float(a[1][x] @ b[1][y] / (na * nb)) if na and nb else 0.0


def contributions(a, b):
    """Shared features and their additive share of cos(a, b), largest first."""
    f, x, y = np.intersect1d(a[0], b[0], assume_unique=True, return_indices=True)
    na, nb = np.linalg.norm(a[1]), np.linalg.norm(b[1])
    if not len(f) or not na or not nb:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    c = a[1][x] * b[1][y] / (na * nb)
    o = np.argsort(-c)
    return f[o], c[o]


def first_hit_rank(scores, relevant):
    return 1 + int((scores > scores[relevant].max()).sum())


def short(text, n=240):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= n else text[:n - 1] + "…"


# =============================================================================
# Loaders: each returns a list of query records
#   {qid, q, g, rank{dense,sae_mean,sae_idf}, vec{m: (q, g)}, runner, runner_vec,
#    margin}
# =============================================================================

def load_xlcost(task, rng, max_pairs):
    import eval_xlcost_sae as E
    import run_xlcost_sae as RX
    out = []
    per_run = max(1, max_pairs // len(task["runs"]))
    for run in task["runs"]:
        cfg = json.loads((run / "run_config.json").read_text())
        dataset = Path(cfg["dataset_file"])
        if not dataset.is_file():  # runs made on a timan host record its local data path
            dataset = Path(str(dataset).replace("/srv/local/xy51/symbolic_merge/data",
                                                "/projects/biro/xiaocong/data"))
        rows = [json.loads(l) for l in open(dataset, encoding="utf-8")]
        q, c = E.load_side(run, "query"), E.load_side(run, "candidate")
        size = int(c[0]["sae"]["size"])
        corpus, _ = E.background(c, "token", size)
        mats = {m: (E.build_matrix(q, m, corpus, size, EPS), E.build_matrix(c, m, corpus, size, EPS))
                for m in METHODS}
        qd = np.stack([d["dense"].numpy().astype(np.float64) for d in q])
        cd = np.stack([d["dense"].numpy().astype(np.float64) for d in c])
        qd /= np.linalg.norm(qd, axis=1, keepdims=True)
        cd /= np.linalg.norm(cd, axis=1, keepdims=True)
        S = {"dense": qd @ cd.T}
        for m, (Q, C) in mats.items():
            S[m] = (Q @ C.T).toarray()
        prefix = defaultdict(list)
        for j, d in enumerate(c):
            prefix[d["idx"].split("/")[0]].append(j)
        recs = []
        for i, d in enumerate(q):
            rel = prefix.get(d["idx"].split("/")[0])
            if not rel:
                continue
            rel = np.array(rel)
            ranks = {m: first_hit_rank(S[m][i], rel) for m in S}
            g = int(rel[np.argmax(S["sae_idf"][i][rel])])
            masked = S["sae_idf"][i].copy()
            masked[rel] = -np.inf
            r = int(np.argmax(masked))
            row = lambda M_, k: sparse_vec(M_[k].indices, M_[k].data)
            recs.append({
                "qid": f"{run.name}:{d['url']}", "set": run.name,
                "q": RX.side_text(rows[int(d["row"])], "query"),
                "g": RX.side_text(rows[int(c[g]["row"])], "candidate"),
                "runner": RX.side_text(rows[int(c[r]["row"])], "candidate"),
                "rank": ranks,
                "vec": {m: (row(mats[m][0], i), row(mats[m][1], g)) for m in METHODS},
                "runner_vec": row(mats["sae_idf"][1], r),
                "margin": float(S["sae_idf"][i][g] - S["sae_idf"][i][r]),
            })
        mrr = np.mean([1 / x["rank"]["sae_idf"] for x in recs])
        ref = json.loads((run / "eval/scores.json").read_text())["metrics"]["sae_idf"]["MRR_first_hit"]
        print(f"  {run.name}: {len(recs)} queries, sae_idf first-hit MRR {mrr:.4f} (eval {ref})", flush=True)
        out.extend(subsample(recs, per_run, rng, keep_all_cases=True))
    return out


def load_entity(task, rng, max_pairs):
    import analyze_bioml_idf as A
    src, tgt = A.load_rep_dirs([task["sae"]])
    qs = A.load_queries([task["ctx"]])
    reps = list(src.values()) + list(tgt.values())
    bg, bg_total, doc_freq, n_docs = A.build_background(reps)
    cache = {}

    def vecs(rep):
        key = rep["entity_iri"]
        if key not in cache:
            sig = A.build_signature(rep, bg, bg_total, doc_freq, n_docs)
            mean = rep["aggregate"]["sae_mean"]
            cache[key] = {"sae_idf": sparse_vec(sig["sae_idf"]["idx"].numpy(), sig["sae_idf"]["val"].numpy()),
                          "sae_mean": sparse_vec(mean["idx"].numpy(), mean["val"].float().numpy())}
        return cache[key]

    def text(rep, k=3):
        views = [c["text"] for c in rep["contexts"][:k]]
        return f"{rep['label']}  |  " + "  /  ".join(views)

    ranks = {int(r["qid"]): r for r in csv.DictReader(open(task["ranks"], encoding="utf-8"))}
    recs = []
    for q in qs:
        r = ranks.get(q["qid"])
        s, g = src.get(q["src"]), tgt.get(q["gold"])
        if r is None or s is None or g is None:
            continue
        sv = vecs(s)
        scored = [(cos(sv["sae_idf"], vecs(tgt[c])["sae_idf"]), c) for c in q["candidates"]
                  if c in tgt and c != q["gold"]]
        best_wrong = max(scored) if scored else (0.0, None)
        gold_cos = cos(sv["sae_idf"], vecs(g)["sae_idf"])
        recs.append({
            "qid": q["qid"], "set": task["title"],
            "q": text(s), "g": text(g),
            "runner": text(tgt[best_wrong[1]]) if best_wrong[1] else "",
            "rank": {"dense": int(r["dense_mean_rank"]), "sae_mean": int(r["sae_mean_rank"]),
                     "sae_idf": int(r["sae_idf_rank"])},
            "vec": {m: (sv[m], vecs(g)[m]) for m in METHODS},
            "runner_vec": vecs(tgt[best_wrong[1]])["sae_idf"] if best_wrong[1] else None,
            "margin": gold_cos - best_wrong[0],
        })
    mrr = np.mean([1 / x["rank"]["sae_idf"] for x in recs])
    print(f"  {task['title']}: {len(recs)} queries, sae_idf MRR {mrr:.4f}", flush=True)
    return subsample(recs, max_pairs, rng, keep_all_cases=True)


def load_valentine(task, rng, max_pairs):
    import eval_valentine_sae as E
    recs = []
    for f in sorted((task["run"] / "pairs").glob("*.pt")):
        p = load_pt(f)
        scols, tcols = p["source"]["columns"], p["target"]["columns"]
        if not scols or not tcols:
            continue
        size = int(tcols[0]["sae"]["size"])
        corpus = E.corpus_stats(tcols, size)
        w = lambda col, m: sparse_vec(*E.column_weights(col, m, corpus, EPS))
        tvec = {m: [w(t, m) for t in tcols] for m in METHODS}
        td = np.stack([t["dense"].numpy().astype(np.float64) for t in tcols])
        td /= np.linalg.norm(td, axis=1, keepdims=True)
        tname = {t["column"]: j for j, t in enumerate(tcols)}
        gold = defaultdict(set)
        for a, b in p["ground_truth"]:
            if b in tname:
                gold[a].add(tname[b])
        txt = lambda col, side: f"{p[side]['table']}.{col['column']}: " + "; ".join(col["values"][:6])
        for sc in scols:
            rel = sorted(gold.get(sc["column"], ()))
            if not rel:
                continue
            rel = np.array(rel)
            sv = {m: w(sc, m) for m in METHODS}
            sd = sc["dense"].numpy().astype(np.float64)
            S = {"dense": td @ (sd / np.linalg.norm(sd))}
            for m in METHODS:
                S[m] = np.array([cos(sv[m], t) for t in tvec[m]])
            g = int(rel[np.argmax(S["sae_idf"][rel])])
            masked = S["sae_idf"].copy()
            masked[rel] = -np.inf
            r = int(np.argmax(masked)) if len(tcols) > len(rel) else None
            recs.append({
                "qid": f"{f.stem}:{sc['column']}", "set": p["pair"]["dataset"],
                "q": txt(sc, "source"), "g": txt(tcols[g], "target"),
                "runner": txt(tcols[r], "target") if r is not None else "",
                "rank": {m: first_hit_rank(S[m], rel) for m in S},
                "vec": {m: (sv[m], tvec[m][g]) for m in METHODS},
                "runner_vec": tvec["sae_idf"][r] if r is not None else None,
                "margin": float(S["sae_idf"][g] - (S["sae_idf"][r] if r is not None else 0.0)),
            })
    mrr = np.mean([1 / x["rank"]["sae_idf"] for x in recs])
    print(f"  {task['title']}: {len(recs)} source columns with a gold match, "
          f"sae_idf first-hit MRR {mrr:.4f}", flush=True)
    return subsample(recs, max_pairs, rng, keep_all_cases=True)


def subsample(recs, n, rng, keep_all_cases):
    """A seeded sample of n correct matches for the statistics, plus every record
    that could become a case study (so case selection sees the whole task)."""
    correct = [r for r in recs if r["rank"]["sae_idf"] == 1]
    chosen = set(id(r) for r in (rng.sample(correct, n) if len(correct) > n else correct))
    for r in recs:
        r["in_sample"] = id(r) in chosen
    if keep_all_cases:
        return [r for r in recs if r["in_sample"] or is_win(r, 2) or is_failure(r)]
    return [r for r in recs if r["in_sample"]]


def is_win(r, min_dense_rank):
    return r["rank"]["sae_idf"] == 1 and r["rank"]["dense"] >= min_dense_rank


def is_failure(r):
    return r["rank"]["dense"] == 1 and r["rank"]["sae_idf"] > 1


# =============================================================================
# Feature metadata and explanation types
# =============================================================================

def load_metadata():
    expl = json.loads((NP_DIR / "explanations.json").read_text())
    dens = json.loads(PILE_DENSITY.read_text())
    density = np.full(131072, np.nan)
    for k, v in dens.items():
        if v is not None:
            density[int(k)] = float(v)
    return {int(k): v for k, v in expl.items()}, density


TYPE_PROMPT = """Each line below describes what one feature of a sparse autoencoder, trained on a language model's activations, responds to. Assign each description exactly one type:

concept: a meaning, topic, kind of entity, quantity or relation, independent of the words or symbols used to express it. Examples: "prime numbers and divisibility", "musical instruments", "regions of Europe", "logarithms and exponents", "employment and job roles", "web frameworks".
surface: a particular word, name, string, spelling, morpheme or token pattern, or text in a particular language or script. Examples: "the word 'actor'", "the token 'Ph' at the start of words", "Chinese characters", "Russian words".
notation: the syntax or formatting of code, mathematics, markup or data: operators, brackets, punctuation, keywords, identifiers, LaTeX, digits. Examples: "closing parentheses in code", "mathematical operators and symbols", "method declarations in Java".
generic: vague, positional or uninterpretable. Examples: "end of sentences", "various tokens in different contexts".

Answer with JSON only, {"types": [...]}, one type per numbered line, in order.

"""


def classify(features, expl, out_path, args):
    cache = json.loads(out_path.read_text()) if out_path.is_file() else {}
    todo = [f for f in sorted(features) if str(f) not in cache and f in expl]
    print(f"explanation types: {len(cache)} cached, {len(todo)} to label", flush=True)
    if not todo:
        return {int(k): v for k, v in cache.items()}
    from anthropic import AnthropicFoundry
    key = Path(os.path.expanduser(args.key_file)).read_text().strip()
    client = AnthropicFoundry(api_key=key, base_url=args.endpoint, max_retries=8, timeout=300)

    def label(batch):
        lines = "\n".join(f"{i + 1}. {expl[f]}" for i, f in enumerate(batch))
        for _ in range(3):
            msg = client.messages.create(model=args.model, max_tokens=4000,
                                         messages=[{"role": "user", "content": TYPE_PROMPT + lines}],
                                         output_config={"effort": "low"})
            text = "".join(b.text for b in msg.content if b.type == "text")
            m = re.search(r"\{.*\}", text, re.S)
            try:
                types = json.loads(m.group(0))["types"]
            except Exception:
                continue
            if len(types) == len(batch) and all(t in TYPES for t in types):
                return dict(zip(batch, types))
        return {}

    batches = [todo[i:i + args.type_batch] for i in range(0, len(todo), args.type_batch)]
    with ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(label, batches):
            cache.update({str(k): v for k, v in res.items()})
            out_path.write_text(json.dumps(cache))
    print(f"explanation types: {len(cache)} labelled", flush=True)
    return {int(k): v for k, v in cache.items()}


# =============================================================================
# Statistics and case studies
# =============================================================================

def pair_stats(f, c, density, types, k):
    total = c.sum()
    if total <= 0:
        return None
    cum = np.cumsum(c) / total
    d = density[f]
    st = {"cos": float(total),
          "n50": int(np.searchsorted(cum, 0.5) + 1), "n80": int(np.searchsorted(cum, 0.8) + 1),
          "rare": float(c[d < 1e-3].sum() / total),
          "mid": float(c[(d >= 1e-3) & (d < 1e-2)].sum() / total),
          "common": float(c[d >= 1e-2].sum() / total),
          "topk_share": float(c[:k].sum() / total)}
    if types:
        top = c[:k]
        for t in TYPES:
            st[t] = float(sum(x for ff, x in zip(f[:k], top) if types.get(int(ff)) == t) / top.sum())
    return st


def feature_rows(f, c, expl, density, n):
    total = c.sum()
    return [{"feature": int(ff), "share": float(x / total), "density": float(density[ff]),
             "explanation": expl.get(int(ff), "(no explanation)")} for ff, x in zip(f[:n], c[:n])]


def summarize(name, task, recs, expl, density, types, args):
    sample = [r for r in recs if r["in_sample"]]
    stats = {m: [] for m in METHODS}
    mass = {m: defaultdict(float) for m in METHODS}
    hits = {m: Counter() for m in METHODS}
    for r in sample:
        for m in METHODS:
            f, c = contributions(*r["vec"][m])
            st = pair_stats(f, c, density, types, args.top_k)
            if st:
                stats[m].append(st)
            if c.sum() > 0:
                for ff, x in zip(f[:args.top_k], c[:args.top_k]):
                    mass[m][int(ff)] += x / c.sum()
                    hits[m][int(ff)] += 1
    agg = {}
    for m, rows in stats.items():
        if not rows:
            continue
        agg[m] = {"pairs": len(rows),
                  "median_n50": float(np.median([s["n50"] for s in rows])),
                  "median_n80": float(np.median([s["n80"] for s in rows])),
                  **{k: float(np.mean([s[k] for s in rows]))
                     for k in ("rare", "mid", "common", "topk_share", *(TYPES if types else ()))}}
    top = {m: [{"feature": f, "evidence": mass[m][f] / max(1, len(sample)), "pairs_in_top_k": hits[m][f],
                "density": float(density[f]), "explanation": expl.get(f, "(no explanation)"),
                "type": types.get(f, "") if types else ""}
               for f, _ in sorted(mass[m].items(), key=lambda kv: -kv[1])[:args.top_features]]
           for m in METHODS}

    def case(r, kind):
        f, c = contributions(*r["vec"]["sae_idf"])
        rec = {"kind": kind, "qid": r["qid"], "set": r["set"], "query": short(r["q"]),
               "gold": short(r["g"]), "rank": r["rank"], "margin": r["margin"],
               "cos_gold": float(c.sum()), "shared_gold": feature_rows(f, c, expl, density, args.case_features)}
        fm, cm = contributions(*r["vec"]["sae_mean"])
        rec["shared_gold_mean"] = feature_rows(fm, cm, expl, density, 3)
        if kind == "failure" and r["runner_vec"] is not None:
            fw, cw = contributions(r["vec"]["sae_idf"][0], r["runner_vec"])
            rec.update(runner=short(r["runner"]), cos_runner=float(cw.sum()),
                       shared_runner=feature_rows(fw, cw, expl, density, args.case_features))
        return rec

    readable = lambda r: len(r["q"]) + len(r["g"]) <= args.max_case_chars
    wins = sorted([r for r in recs if is_win(r, args.min_dense_rank) and r["margin"] > 0],
                  key=lambda r: (not readable(r), -r["margin"]))
    fails = sorted([r for r in recs if is_failure(r)], key=lambda r: (not readable(r), len(r["q"]) + len(r["g"])))
    cases = [case(r, "win") for r in wins[:args.cases]] + [case(r, "failure") for r in fails[:args.failures]]
    n_all = {"wins_available": len([r for r in recs if is_win(r, args.min_dense_rank)]),
             "failures_available": len(fails)}
    return {"task": name, "title": task["title"], "family": task["family"], "stats": agg,
            "top_features": top["sae_idf"], "top_features_sae_mean": top["sae_mean"],
            "cases": cases, **n_all}


def write_outputs(results, out):
    rows = []
    for res in results:
        d = out / res["task"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        for m, s in res["stats"].items():
            rows.append({"task": res["title"], "method": m, **{k: round(v, 4) if isinstance(v, float) else v
                                                              for k, v in s.items()}})
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    md = ["# Case studies (selected by rule)", ""]
    for res in results:
        md += [f"## {res['title']}", "",
               f"Wins available (sae_idf rank 1, dense rank >= threshold): {res['wins_available']}; "
               f"failures available (dense rank 1, sae_idf not): {res['failures_available']}.", ""]
        for key, what in (("top_features", "sae_idf"), ("top_features_sae_mean", "sae_mean (no idf)")):
            md += [f"Features carrying the most {what} evidence across the same correct matches:", ""]
            for t in res[key][:8]:
                md.append(f"- f{t['feature']} ({t['type'] or 'untyped'}, Pile density {t['density']:.1e}, "
                          f"in top-k of {t['pairs_in_top_k']} pairs): {t['explanation']}")
            md.append("")
        for c in res["cases"]:
            r = c["rank"]
            md += [f"### {c['kind']}: {c['qid']}",
                   f"- query: `{c['query']}`", f"- gold: `{c['gold']}`",
                   f"- rank of gold: dense {r['dense']}, sae_mean {r['sae_mean']}, sae_idf {r['sae_idf']}",
                   f"- sae_idf cosine to gold {c['cos_gold']:.3f}"
                   + (f", to the wrong top candidate {c['cos_runner']:.3f}" if "cos_runner" in c else
                      f", margin over runner-up {c['margin']:.3f}"), "",
                   "| shared feature | share of cosine | Pile density | explanation |", "|---|---|---|---|"]
            md += [f"| [f{x['feature']}]({NP_URL.format(x['feature'])}) | {x['share']:.2f} | {x['density']:.1e} | "
                   f"{x['explanation']} |" for x in c["shared_gold"]]
            md += ["", "same pair without idf (sae_mean), top shared features: " + "; ".join(
                f"f{x['feature']} {x['share']:.2f} ({x['density']:.0e}) {x['explanation']}" for x in c["shared_gold_mean"])]
            if "runner" in c:
                md += ["", f"wrong top candidate: `{c['runner']}`", "",
                       "| shared with wrong candidate | share | Pile density | explanation |", "|---|---|---|---|"]
                md += [f"| f{x['feature']} | {x['share']:.2f} | {x['density']:.1e} | {x['explanation']} |"
                       for x in c["shared_runner"]]
            md.append("")
    (out / "case_studies.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--tasks", default="all", help="comma list of task keys, or all")
    ap.add_argument("--max_pairs", type=int, default=300, help="correct matches sampled per task")
    ap.add_argument("--top_k", type=int, default=5, help="top features per pair for types and top lists")
    ap.add_argument("--top_features", type=int, default=20)
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--failures", type=int, default=1)
    ap.add_argument("--case_features", type=int, default=5)
    ap.add_argument("--min_dense_rank", type=int, default=5)
    ap.add_argument("--max_case_chars", type=int, default=420)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classify", action="store_true", help="label explanation types with Claude")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--endpoint", default="https://xiaocong-resource.services.ai.azure.com/anthropic")
    ap.add_argument("--key_file", default="~/.config/symbolic_merge/azure_anthropic_key")
    ap.add_argument("--type_batch", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    registry = task_registry()
    names = list(registry) if args.tasks == "all" else args.tasks.split(",")
    expl, density = load_metadata()
    loaders = {"xlcost": load_xlcost, "entity": load_entity, "valentine": load_valentine}
    loaded = {}
    for name in names:
        t0 = time.time()
        print(f"[{name}] loading", flush=True)
        loaded[name] = loaders[registry[name]["kind"]](registry[name], random.Random(args.seed), args.max_pairs)
        print(f"[{name}] {len(loaded[name])} records in {time.time() - t0:.0f}s", flush=True)

    types = {}
    if args.classify:
        need = set()
        for recs in loaded.values():
            for r in recs:
                vs = [r["vec"][m] for m in METHODS]
                if r["runner_vec"] is not None:
                    vs.append((r["vec"]["sae_idf"][0], r["runner_vec"]))
                for a, b in vs:
                    f, _ = contributions(a, b)
                    need.update(int(x) for x in f[:args.top_k])
        types = classify(need, expl, out / "explanation_types.json", args)

    results = [summarize(n, registry[n], loaded[n], expl, density, types, args) for n in names]
    write_outputs(results, out)
    print(f"wrote {out}/summary.csv, {out}/case_studies.md and per-task summary.json")


if __name__ == "__main__":
    main()
