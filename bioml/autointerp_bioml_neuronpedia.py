#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autointerp_bioml_neuronpedia.py

Auto-interpretability for the Bio-ML information-weighted SAE pipeline.

Pipeline:
    Bio-ML SAE representations (run_bioml_sae_from_bio8b_contexts.py)
      -> per-entity feature statistics (same formulas as the ranking analysis)
      -> CANDIDATE FILTER: keep only features with a real activation pattern
      -> Neuronpedia lookup (existing gpt-4o-mini explanations + top activations)
      -> per-feature / per-entity / per-pair interpretability reports

Design note: the SAE has 131,072 features, but only a few thousand carry any
signal for our entities, and only a few dozen per entity survive the
conservation / enrichment / reliability filter. We explain ONLY those. No LLM
calls are made by default: Neuronpedia already hosts auto-interp explanations
for every Gemma Scope feature, so the default path is a plain HTTP lookup of
the filtered candidates. Use --llm_explain_missing to auto-interp the leftovers
(features Neuronpedia has no explanation for) with the Azure Anthropic
deployment, built from Neuronpedia's own top-activating examples.

Feature indices must line up with Neuronpedia's source. For Bio-ML that is
    gemma-2-9b / 20-gemmascope-res-131k
    = google/gemma-scope-9b-pt-res : layer_20/width_131k/average_l0_114
Other average_l0_* variants are DIFFERENT SAEs with different feature indices,
so the script verifies the mapping against results.json and refuses to run on a
mismatch unless --force_sae_mapping is given.

Example:
    python bioml/autointerp_bioml_neuronpedia.py \
        --sae_run_dirs  /path/to/bioml_sae_run \
        --context_run_dirs /path/to/bioml_context_run \
        --output_dir    /path/to/autointerp_out
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch

NP_BASE = "https://www.neuronpedia.org"
NP_MODEL = "gemma-2-9b"
NP_SAE = "20-gemmascope-res-131k"
# Neuronpedia source id -> the Gemma Scope release it was built from.
# Neuronpedia publishes two L0 variants per (layer, site, width): the canonical
# one and a sparser "-l0_32plus" one. average_l0_53 is NOT published for layer 20
# residual 131k, so runs on that SAE cannot be interpreted against Neuronpedia.
NP_SAE_EXPECTED_FOLDER = {
    "20-gemmascope-res-131k": "layer_20/width_131k/average_l0_114",
    "20-gemmascope-res-131k-l0_32plus": "layer_20/width_131k/average_l0_34",
    "20-gemmascope-res-16k": "layer_20/width_16k/average_l0_68",
    "20-gemmascope-res-16k-l0_32plus": "layer_20/width_16k/average_l0_36",
}
METRICS = ("sae_idf", "sae_info_reliability", "sae_info", "sae_logodds", "sae_mean")

STOPWORDS = {
    "the", "of", "and", "or", "in", "to", "a", "an", "with", "for", "by", "on",
    "at", "from", "as", "is", "are", "was", "were", "be", "been", "not", "no",
    "other", "nos", "unspecified", "disease", "disorder", "syndrome",
}


# =============================================================================
# Small utilities (kept consistent with the other bioml scripts)
# =============================================================================

def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sparse_items(obj):
    """{'idx','val','size'} -> {feature_id: value}"""
    idx = obj["idx"].detach().cpu().long().tolist()
    val = obj["val"].detach().cpu().float().tolist()
    return {int(i): float(v) for i, v in zip(idx, val)}


def cos_dict(a, b):
    if not a or not b:
        return 0.0
    small, other = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(v * other.get(i, 0.0) for i, v in small.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def label_tokens(label):
    return {t for t in re.findall(r"[a-z0-9]+", str(label).lower())
            if len(t) > 2 and t not in STOPWORDS}


# =============================================================================
# Loading
# =============================================================================

def load_rep_dirs(run_dirs):
    src, tgt = {}, {}
    for run in run_dirs:
        for side, store in (("src", src), ("tgt", tgt)):
            folder = run / "representations" / side
            if not folder.is_dir():
                raise FileNotFoundError(f"missing representation folder: {folder}")
            for path in sorted(folder.glob("*.pt")):
                rep = load_pt(path)
                store.setdefault(str(rep["entity_iri"]), rep)
    if not src or not tgt:
        raise RuntimeError("no representations loaded")
    return src, tgt


def load_queries(context_dirs):
    out, seen = [], set()
    for d in context_dirs:
        rows = json.loads((d / "selected_queries.json").read_text(encoding="utf-8"))
        for q in rows:
            qid = int(q.get("query_id", q.get("qid")))
            if qid in seen:
                continue
            seen.add(qid)
            out.append({"qid": qid, "src": q["src"], "gold": q["gold"],
                        "candidates": list(q["candidates"])})
    if not out:
        raise RuntimeError("no queries loaded")
    return sorted(out, key=lambda x: x["qid"])


def verify_sae_mapping(run_dirs, reps, np_sae, force):
    """Feature indices are only comparable if we used Neuronpedia's exact SAE."""
    expected = NP_SAE_EXPECTED_FOLDER.get(np_sae)
    width = {int(rep["contexts"][0]["sae"]["size"]) for rep in reps}
    found_paths, notes = [], []
    for run in run_dirs:
        results = run / "results.json"
        if results.is_file():
            try:
                meta = json.loads(results.read_text(encoding="utf-8"))
                found_paths.append(str(meta.get("sae_path", "")))
                notes.append({"run": str(run), "sae_path": meta.get("sae_path"),
                              "layer": meta.get("layer"), "model_path": meta.get("model_path")})
            except Exception as e:  # pragma: no cover - defensive
                notes.append({"run": str(run), "error": f"unreadable results.json: {e}"})
        else:
            notes.append({"run": str(run), "error": "no results.json; SAE identity unverified"})

    problems = []
    if len(width) != 1:
        problems.append(f"representations mix SAE widths: {sorted(width)}")
    elif expected and "131k" in expected and next(iter(width)) != 131072:
        problems.append(f"SAE width {next(iter(width))} != 131072 expected by {np_sae}")
    if expected and found_paths:
        layer, wid, l0 = expected.split("/")
        for p in found_paths:
            norm = p.replace("\\", "/")
            if not (layer in norm and wid in norm and l0 in norm):
                problems.append(
                    f"sae_path {p!r} does not look like {expected}; feature indices "
                    f"would not match Neuronpedia source {np_sae}"
                )
    if problems and not force:
        raise RuntimeError(
            "SAE/Neuronpedia mapping check failed:\n  - " + "\n  - ".join(problems)
            + "\nPass --force_sae_mapping to override, or point --np_sae at the right source."
        )
    for p in problems:
        print(f"[warn] {p}", flush=True)
    return {"np_sae": np_sae, "expected_gemmascope_folder": expected,
            "sae_width": sorted(width), "runs": notes, "problems": problems}


# =============================================================================
# Feature statistics (same definitions as the ranking analysis)
# =============================================================================

def build_background(reps):
    """Feature presence per context, plus per-entity document frequency (for idf)."""
    counts = Counter()
    doc_freq = Counter()
    total = 0
    for rep in reps:
        seen = set()
        for c in rep["contexts"]:
            idx = set(int(i) for i in c["sae"]["idx"].detach().cpu().tolist())
            counts.update(idx)
            seen |= idx
            total += 1
        doc_freq.update(seen)
    if not total:
        raise RuntimeError("empty background")
    return counts, total, doc_freq, len(reps)


def feature_stats(rep, bg, bg_total, doc_freq=None, n_docs=0, eps=1e-6):
    """Per-feature activation pattern of one entity across its contexts."""
    n = len(rep["contexts"])
    values = defaultdict(lambda: [0.0] * n)
    for ci, c in enumerate(rep["contexts"]):
        for i, v in zip(c["sae"]["idx"].detach().cpu().tolist(),
                        c["sae"]["val"].detach().cpu().float().tolist()):
            values[int(i)][ci] = float(v)

    out = {}
    for f, xs in values.items():
        x = torch.tensor(xs, dtype=torch.float32)
        p_entity = float((x > 0).sum()) / n
        p_background = bg.get(f, 0) / bg_total
        mean = float(x.mean())
        std = float(x.std(unbiased=False))
        cv = std / (abs(mean) + eps)
        info = max(math.log((p_entity + eps) / (p_background + eps)), 0.0)
        reliability = 1.0 / (1.0 + cv)
        pe2 = min(max(p_entity, eps), 1 - eps)
        pb2 = min(max(p_background, eps), 1 - eps)
        logodds = max(math.log(pe2 / (1 - pe2)) - math.log(pb2 / (1 - pb2)), 0.0)
        idf = max(math.log(n_docs / (1.0 + doc_freq.get(f, 0))), 0.0) \
            if doc_freq is not None and n_docs else 0.0
        out[f] = {
            "n_contexts": n,
            "n_active": int((x > 0).sum()),
            "p_entity": p_entity,
            "p_background": p_background,
            "mean_activation": mean,
            "max_activation": float(x.max()),
            "cv": cv,
            "reliability": reliability,
            "information": info,
            "sae_info": mean * p_entity * info,
            "sae_info_reliability": mean * p_entity * info * reliability,
            "sae_logodds": mean * logodds,
            "sae_mean": mean,
            "sae_idf": mean * idf,
            "idf": idf,
            "doc_freq": doc_freq.get(f, 0) if doc_freq is not None else 0,
        }
    return out


def signature(stats, metric):
    return {f: s[metric] for f, s in stats.items() if s[metric] > 0}


# =============================================================================
# Candidate filter -- "features with a potential activation pattern"
# =============================================================================

def filter_entity_features(stats, args):
    """Conserved across contexts, enriched over background, stable in magnitude."""
    kept = []
    idf_only = args.metric == "sae_idf"
    for f, s in stats.items():
        if not idf_only and s["p_entity"] < args.min_p_entity:
            continue
        if s["p_background"] > args.max_p_background:
            continue
        if not idf_only and s["information"] <= args.min_information:
            continue
        if not idf_only and s["reliability"] < args.min_reliability:
            continue
        if s["mean_activation"] < args.min_mean_activation:
            continue
        kept.append((s[args.metric], f))
    kept.sort(reverse=True)
    return [f for _, f in kept[:args.top_per_entity]]


def pair_contributions(src_sig, tgt_sig, top_k):
    """Features that actually produce the src/candidate cosine."""
    na = math.sqrt(sum(v * v for v in src_sig.values()))
    nb = math.sqrt(sum(v * v for v in tgt_sig.values()))
    if not na or not nb:
        return []
    shared = []
    for f, v in src_sig.items():
        w = tgt_sig.get(f)
        if w:
            shared.append((v * w / (na * nb), f))
    shared.sort(reverse=True)
    return [(f, c) for c, f in shared[:top_k]]


def select_candidates(src_reps, tgt_reps, queries, stats_by_iri, sigs, args):
    """Union of per-entity top features and the drivers of each ranking decision."""
    roles = defaultdict(set)        # feature -> {"entity_top", "pair_driver", ...}
    entity_rows = []
    per_pair = []

    for iri, stats in stats_by_iri.items():
        for f in filter_entity_features(stats, args):
            roles[f].add("entity_top")

    for q in queries:
        src, gold = q["src"], q["gold"]
        if src not in sigs:
            continue
        cands = [c for c in q["candidates"] if c in sigs]
        if gold not in cands:
            continue
        scored = sorted(((cos_dict(sigs[src], sigs[c]), c) for c in cands),
                        key=lambda z: (-z[0], z[1]))
        order = [c for _, c in scored]
        rank = order.index(gold) + 1
        top1 = order[0]

        gold_drivers = pair_contributions(sigs[src], sigs[gold], args.top_per_pair)
        for f, _ in gold_drivers:
            roles[f].add("gold_driver")
        distractor_drivers = []
        if top1 != gold:
            distractor_drivers = pair_contributions(sigs[src], sigs[top1], args.top_per_pair)
            for f, _ in distractor_drivers:
                roles[f].add("distractor_driver")

        per_pair.append({
            "qid": q["qid"],
            "src": src,
            "src_label": src_reps[src]["label"],
            "gold": gold,
            "gold_label": tgt_reps[gold]["label"],
            "gold_rank": rank,
            "n_candidates": len(cands),
            "gold_score": next(s for s, c in scored if c == gold),
            "top1": top1,
            "top1_label": tgt_reps[top1]["label"],
            "top1_score": scored[0][0],
            "gold_drivers": gold_drivers,
            "distractor_drivers": distractor_drivers,
        })

    # Per-entity rows for every feature we are going to explain.
    for iri, stats in stats_by_iri.items():
        label = (src_reps.get(iri) or tgt_reps.get(iri))["label"]
        side = "src" if iri in src_reps else "tgt"
        for f in roles:
            if f in stats:
                s = stats[f]
                entity_rows.append({
                    "entity_iri": iri, "side": side, "label": label, "feature": f,
                    "roles": "|".join(sorted(roles[f])),
                    **{k: s[k] for k in ("n_active", "n_contexts", "p_entity",
                                         "p_background", "mean_activation",
                                         "max_activation", "cv", "reliability",
                                         "information", args.metric)},
                })

    features = sorted(roles)
    if args.max_features and len(features) > args.max_features:
        best = {}
        for row in entity_rows:
            f = row["feature"]
            best[f] = max(best.get(f, 0.0), float(row[args.metric]))
        prio = sorted(features, key=lambda f: (
            "gold_driver" not in roles[f] and "distractor_driver" not in roles[f],
            -best.get(f, 0.0),
        ))
        features = sorted(prio[:args.max_features])
        keep = set(features)
        entity_rows = [r for r in entity_rows if r["feature"] in keep]
    return features, {f: sorted(roles[f]) for f in features}, entity_rows, per_pair


# =============================================================================
# Neuronpedia (hijohnnylin/neuronpedia) lookup
# =============================================================================

class Neuronpedia:
    """Read-only client for the public feature endpoint, with an on-disk cache."""

    def __init__(self, base, model, sae, cache_dir, rps, api_key=None, timeout=30):
        self.base = base.rstrip("/")
        self.model = model
        self.sae = sae
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.headers = {"Accept": "application/json",
                        "User-Agent": "Symbolic_Merge-autointerp/1.0"}
        if api_key:
            self.headers["X-Api-Key"] = api_key
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self.timeout = timeout
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self.stats = Counter()

    def dashboard_url(self, feature):
        return f"{self.base}/{self.model}/{self.sae}/{feature}"

    def _throttle(self):
        if not self.min_interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait:
            time.sleep(wait)

    def fetch(self, feature, refresh=False):
        cache = self.cache_dir / f"{feature}.json"
        if cache.is_file() and not refresh:
            try:
                self.stats["cached"] += 1
                return json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                pass
        url = f"{self.base}/api/feature/{self.model}/{self.sae}/{feature}"
        delay = 1.0
        for attempt in range(4):
            self._throttle()
            try:
                r = self.session.get(url, headers=self.headers, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    cache.write_text(json.dumps(data), encoding="utf-8")
                    self.stats["fetched"] += 1
                    return data
                if r.status_code == 404:
                    self.stats["missing"] += 1
                    return None
                if r.status_code in (429, 500, 502, 503, 504):
                    self.stats[f"retry_{r.status_code}"] += 1
                    time.sleep(delay * (3 if r.status_code == 429 else 1))
                    delay *= 2
                    continue
                self.stats[f"http_{r.status_code}"] += 1
                return None
            except requests.RequestException as e:
                self.stats["exception"] += 1
                if attempt == 3:
                    print(f"[neuronpedia] feature {feature} failed: {e}", flush=True)
                    return None
                time.sleep(delay)
                delay *= 2
        return None

    def fetch_many(self, features, workers, refresh=False):
        out = {}
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for feature, data in zip(features, pool.map(
                    lambda f: self.fetch(f, refresh), features)):
                out[feature] = data
                done += 1
                if done % 100 == 0 or done == len(features):
                    print(f"[neuronpedia] {done}/{len(features)} "
                          f"(fetched={self.stats['fetched']} cached={self.stats['cached']})",
                          flush=True)
        return out


def summarize_np_feature(data):
    """Pull the fields we report out of a Neuronpedia feature payload."""
    if not data:
        return {"explanation": "", "explanation_model": "", "explanation_type": "",
                "n_explanations": 0, "np_density": None, "np_max_act": None,
                "top_tokens": "", "np_found": False}
    exps = data.get("explanations") or []
    best = exps[0] if exps else {}
    top_tokens = []
    for act in (data.get("activations") or [])[:3]:
        tokens = act.get("tokens") or []
        try:
            peak = int(act.get("maxValueTokenIndex"))
        except (TypeError, ValueError):
            peak = None
        if tokens and peak is not None and 0 <= peak < len(tokens):
            lo, hi = max(0, peak - 2), min(len(tokens), peak + 3)
            span = "".join(tokens[lo:hi]).replace("▁", " ").strip()
            top_tokens.append(f"{tokens[peak].replace(chr(9601), ' ').strip()} [{span}]")
    return {
        "explanation": str(best.get("description", "")).strip(),
        "explanation_model": best.get("explanationModelName", ""),
        "explanation_type": best.get("typeName", ""),
        "n_explanations": len(exps),
        "np_density": data.get("frac_nonzero"),
        "np_max_act": data.get("maxActApprox"),
        "top_tokens": " | ".join(top_tokens),
        "np_found": True,
    }


# =============================================================================
# Optional: auto-interp the features Neuronpedia has no explanation for
# =============================================================================

def llm_explain(features, np_data, args):
    """Token/activation-pair auto-interp using Neuronpedia's own top examples."""
    from anthropic import AnthropicFoundry

    api_key = os.environ.get(args.anthropic_api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key in ${args.anthropic_api_key_env}")
    client = AnthropicFoundry(api_key=api_key, base_url=args.anthropic_endpoint)
    out = {}
    for n, f in enumerate(features[:args.max_llm_features], 1):
        acts = (np_data.get(f) or {}).get("activations") or []
        blocks = []
        for act in acts[:args.llm_examples]:
            tokens = act.get("tokens") or []
            values = act.get("values") or []
            if not tokens or not values:
                continue
            pairs = [f"{t.replace(chr(9601), ' ')}\t{float(v):.2f}"
                     for t, v in zip(tokens, values) if float(v) > 0]
            if pairs:
                blocks.append("\n".join(pairs[:args.llm_tokens_per_example]))
        if not blocks:
            continue
        user = (
            "Below are text excerpts where one neuron of a language model activates. "
            "Each line is a token and its activation (only non-zero activations are shown).\n\n"
            + "\n---\n".join(blocks)
            + "\n\nIn one short phrase, state what the neuron detects. Return only the phrase."
        )
        try:
            msg = client.messages.create(
                model=args.anthropic_model,
                system="You explain neuron activation patterns concisely and literally.",
                messages=[{"role": "user", "content": user}],
                max_tokens=60,
            )
            parts = [b.text for b in getattr(msg, "content", [])
                     if getattr(b, "type", "text") == "text" and getattr(b, "text", None)]
            text = " ".join(parts).strip() or ""
        except Exception as e:
            print(f"[llm] feature {f} failed: {type(e).__name__}: {e}", flush=True)
            continue
        if text:
            out[f] = text
        if n % 10 == 0:
            print(f"[llm] explained {n}/{min(len(features), args.max_llm_features)}", flush=True)
    return out


# =============================================================================
# Reporting
# =============================================================================

def lexical_flag(explanation, top_tokens, labels):
    """Does the explanation/top token just restate the entity's own wording?"""
    if not explanation and not top_tokens:
        return ""
    text = f"{explanation} {top_tokens}".lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    for label in labels:
        if label_tokens(label) & words:
            return "lexical"
    return "semantic"


def write_pair_report(path, per_pair, feature_info, src_reps, tgt_reps):
    lines = ["# Which SAE features drove each Bio-ML ranking decision", ""]
    hits = [p for p in per_pair if p["gold_rank"] == 1]
    lines.append(f"- queries analysed: **{len(per_pair)}**")
    lines.append(f"- gold ranked first: **{len(hits)}/{len(per_pair)}**")
    lines.append("")
    for p in sorted(per_pair, key=lambda z: (z["gold_rank"], z["qid"])):
        lines.append(f"## q{p['qid']}  {p['src_label']}  ->  {p['gold_label']}"
                     f"   (gold rank {p['gold_rank']}/{p['n_candidates']})")
        lines.append("")
        lines.append(f"cosine(src, gold) = {p['gold_score']:.4f}")
        if p["gold_rank"] != 1:
            lines.append(f"top-1 was **{p['top1_label']}** at {p['top1_score']:.4f}")
        lines.append("")
        for title, drivers in (("Shared features with the gold entity", p["gold_drivers"]),
                               ("Shared features with the wrong top-1", p["distractor_drivers"])):
            if not drivers:
                continue
            lines.append(f"**{title}**")
            lines.append("")
            lines.append("| feature | share of cosine | explanation | kind | density |")
            lines.append("|---|---|---|---|---|")
            for f, contrib in drivers:
                info = feature_info.get(f, {})
                expl = (info.get("explanation") or "_no Neuronpedia explanation_").strip()
                dens = info.get("np_density")
                lines.append(
                    f"| [{f}]({info.get('dashboard_url', '')}) | {contrib:.3f} | {expl} | "
                    f"{info.get('kind', '')} | {'' if dens is None else f'{dens:.2e}'} |"
                )
            lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sae_run_dirs", required=True,
                    help="comma-separated run dirs containing representations/{src,tgt}")
    ap.add_argument("--context_run_dirs", required=True,
                    help="comma-separated dirs containing selected_queries.json")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--metric", default="sae_idf", choices=METRICS,
                    help="weighting used for ranking and for prioritising features")
    ap.add_argument("--background", choices=("target", "all"), default="target")

    # candidate filter
    ap.add_argument("--min_p_entity", type=float, default=0.6,
                    help="feature must fire in >= this fraction of the entity's contexts")
    ap.add_argument("--max_p_background", type=float, default=0.5,
                    help="drop features that fire in more than this fraction of background contexts")
    ap.add_argument("--min_information", type=float, default=0.0)
    ap.add_argument("--min_reliability", type=float, default=0.4,
                    help="1/(1+CV) floor; filters spiky, unstable features")
    ap.add_argument("--min_mean_activation", type=float, default=0.0)
    ap.add_argument("--top_per_entity", type=int, default=10)
    ap.add_argument("--top_per_pair", type=int, default=10)
    ap.add_argument("--max_features", type=int, default=1500,
                    help="global cap on features sent to Neuronpedia (0 = no cap)")
    ap.add_argument("--limit_entities", type=int, default=0, help="debug: cap entities")

    # neuronpedia
    ap.add_argument("--np_base", default=NP_BASE)
    ap.add_argument("--np_model", default=NP_MODEL)
    ap.add_argument("--np_sae", default=NP_SAE)
    ap.add_argument("--np_cache_dir", type=Path, default=None)
    ap.add_argument("--requests_per_second", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refresh_np", action="store_true", help="ignore the on-disk cache")
    ap.add_argument("--np_api_key_env", default="NEURONPEDIA_API_KEY")
    ap.add_argument("--skip_neuronpedia", action="store_true",
                    help="only run the candidate filter (no network)")
    ap.add_argument("--force_sae_mapping", action="store_true")

    # optional LLM fallback
    ap.add_argument("--llm_explain_missing", action="store_true")
    ap.add_argument("--max_llm_features", type=int, default=50)
    ap.add_argument("--llm_examples", type=int, default=6)
    ap.add_argument("--llm_tokens_per_example", type=int, default=40)
    ap.add_argument("--anthropic_model",
                    default=os.environ.get("ANTHROPIC_DEPLOYMENT_NAME", "claude-opus-5"))
    ap.add_argument("--anthropic_endpoint",
                    default=os.environ.get("ANTHROPIC_ENDPOINT",
                                           "https://xiaocong-resource.services.ai.azure.com/anthropic"))
    ap.add_argument("--anthropic_api_key_env", default="ANTHROPIC_API_KEY")
    args = ap.parse_args()

    sae_dirs = [Path(x).expanduser().resolve() for x in args.sae_run_dirs.split(",") if x.strip()]
    ctx_dirs = [Path(x).expanduser().resolve() for x in args.context_run_dirs.split(",") if x.strip()]
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.np_cache_dir or (out / "np_cache")).expanduser().resolve()

    print("=" * 100)
    print("BIO-ML SAE AUTO-INTERPRETABILITY (candidate-filtered, Neuronpedia-backed)")
    print("=" * 100)

    src_reps, tgt_reps = load_rep_dirs(sae_dirs)
    queries = load_queries(ctx_dirs)
    print(f"source reps={len(src_reps)} target reps={len(tgt_reps)} queries={len(queries)}")

    mapping = verify_sae_mapping(sae_dirs, list(src_reps.values())[:1] + list(tgt_reps.values())[:1],
                                 args.np_sae, args.force_sae_mapping)
    print(f"SAE mapping: {args.np_model}/{args.np_sae} "
          f"<- {mapping['expected_gemmascope_folder']} (width {mapping['sae_width']})")

    bg_source = list(tgt_reps.values()) if args.background == "target" \
        else list(src_reps.values()) + list(tgt_reps.values())
    bg, bg_total, doc_freq, n_docs = build_background(bg_source)
    print(f"background: {len(bg_source)} entities, {bg_total} contexts, {len(bg)} features seen")

    all_reps = {**src_reps, **tgt_reps}
    iris = sorted(all_reps)
    if args.limit_entities:
        keep = set(iris[:args.limit_entities]) | {q["src"] for q in queries} \
            | {q["gold"] for q in queries}
        iris = [i for i in iris if i in keep]
    stats_by_iri = {i: feature_stats(all_reps[i], bg, bg_total, doc_freq, n_docs)
                    for i in iris}
    sigs = {i: signature(s, args.metric) for i, s in stats_by_iri.items()}
    print(f"feature statistics built for {len(stats_by_iri)} entities "
          f"({sum(len(s) for s in stats_by_iri.values()):,} entity-feature pairs)")

    features, roles, entity_rows, per_pair = select_candidates(
        src_reps, tgt_reps, queries, stats_by_iri, sigs, args)
    total_seen = len({f for s in stats_by_iri.values() for f in s})
    print(f"candidates after filtering: {len(features):,} features "
          f"(of {total_seen:,} ever active, of {mapping['sae_width'][0]:,} in the SAE) "
          f"= {len(features) / max(1, mapping['sae_width'][0]):.2%} of the SAE")

    np_data, feature_info = {}, {}
    client = Neuronpedia(args.np_base, args.np_model, args.np_sae, cache_dir,
                         args.requests_per_second,
                         os.environ.get(args.np_api_key_env))
    if args.skip_neuronpedia:
        print("[neuronpedia] skipped (--skip_neuronpedia)")
    else:
        print(f"[neuronpedia] looking up {len(features):,} features "
              f"at <= {args.requests_per_second}/s (cache: {cache_dir})")
        np_data = client.fetch_many(features, args.workers, args.refresh_np)

    llm_extra = {}
    missing = [f for f in features if not (np_data.get(f) or {}).get("explanations")]
    if args.llm_explain_missing and missing:
        print(f"[llm] auto-interp for {min(len(missing), args.max_llm_features)} "
              f"of {len(missing)} features without a Neuronpedia explanation")
        llm_extra = llm_explain(missing, np_data, args)

    # per-feature aggregation
    labels_by_feature = defaultdict(set)
    weight_by_feature = defaultdict(float)
    entities_by_feature = defaultdict(set)
    for row in entity_rows:
        f = row["feature"]
        labels_by_feature[f].add(row["label"])
        weight_by_feature[f] = max(weight_by_feature[f], float(row[args.metric]))
        entities_by_feature[f].add(row["entity_iri"])

    feature_rows = []
    for f in features:
        s = summarize_np_feature(np_data.get(f))
        if not s["explanation"] and f in llm_extra:
            s.update(explanation=llm_extra[f], explanation_model=args.anthropic_model,
                     explanation_type="local_token-act-pair")
        kind = lexical_flag(s["explanation"], s["top_tokens"], labels_by_feature[f])
        info = {
            "feature": f,
            "dashboard_url": client.dashboard_url(f),
            "roles": "|".join(roles[f]),
            "n_entities": len(entities_by_feature[f]),
            f"max_{args.metric}": weight_by_feature[f],
            "kind": kind,
            **s,
        }
        feature_info[f] = info
        feature_rows.append(info)
    feature_rows.sort(key=lambda r: -r[f"max_{args.metric}"])

    write_csv(out / "candidate_features.csv", feature_rows)
    write_csv(out / "per_entity_features.csv", entity_rows)
    write_pair_report(out / "pair_explanations.md", per_pair, feature_info, src_reps, tgt_reps)
    atomic_json({
        "qid_ranks": {str(p["qid"]): p["gold_rank"] for p in per_pair},
        "pairs": [{k: v for k, v in p.items()
                   if k not in ("gold_drivers", "distractor_drivers")}
                  | {"gold_drivers": [[f, c] for f, c in p["gold_drivers"]],
                     "distractor_drivers": [[f, c] for f, c in p["distractor_drivers"]]}
                  for p in per_pair],
    }, out / "pair_drivers.json")

    explained = [r for r in feature_rows if r["explanation"]]
    kinds = Counter(r["kind"] for r in explained)
    driver_features = {f for f in features
                       if "gold_driver" in roles[f] or "distractor_driver" in roles[f]}
    driver_kinds = Counter(feature_info[f]["kind"] for f in driver_features
                           if feature_info[f]["explanation"])
    dens = [r["np_density"] for r in feature_rows if isinstance(r["np_density"], (int, float))]
    summary = {
        "sae_mapping": mapping,
        "metric": args.metric,
        "background": {"mode": args.background, "entities": len(bg_source),
                       "contexts": bg_total, "features_seen": len(bg)},
        "filter": {k: getattr(args, k) for k in
                   ("min_p_entity", "max_p_background", "min_information",
                    "min_reliability", "min_mean_activation", "top_per_entity",
                    "top_per_pair", "max_features")},
        "features": {
            "in_sae": mapping["sae_width"][0],
            "ever_active": total_seen,
            "candidates": len(features),
            "fraction_of_sae": len(features) / max(1, mapping["sae_width"][0]),
            "with_neuronpedia_explanation": sum(1 for r in feature_rows if r["np_found"]
                                                and r["n_explanations"]),
            "with_llm_explanation": len(llm_extra),
            "without_any_explanation": sum(1 for r in feature_rows if not r["explanation"]),
            "median_neuronpedia_density": (sorted(dens)[len(dens) // 2] if dens else None),
        },
        "kinds_all_candidates": dict(kinds),
        "kinds_ranking_drivers": dict(driver_kinds),
        "queries": {"analysed": len(per_pair),
                    "gold_rank_1": sum(1 for p in per_pair if p["gold_rank"] == 1)},
        "neuronpedia_requests": dict(client.stats),
    }
    atomic_json(summary, out / "autointerp_summary.json")

    print("-" * 100)
    print(f"explained by Neuronpedia : {summary['features']['with_neuronpedia_explanation']:,}"
          f" / {len(features):,}")
    if llm_extra:
        print(f"explained by local LLM   : {len(llm_extra):,}")
    print(f"no explanation           : {summary['features']['without_any_explanation']:,}")
    print(f"ranking drivers by kind  : {dict(driver_kinds)}")
    print(f"gold ranked first        : {summary['queries']['gold_rank_1']}/{len(per_pair)}")
    print("-" * 100)
    for r in feature_rows[:15]:
        print(f"  {r['feature']:>7}  {r[f'max_{args.metric}']:8.3f}  {r['kind']:<8} "
              f"{(r['explanation'] or '-')[:82]}")
    print(f"\nWrote:\n  {out/'candidate_features.csv'}\n  {out/'per_entity_features.csv'}"
          f"\n  {out/'pair_explanations.md'}\n  {out/'pair_drivers.json'}"
          f"\n  {out/'autointerp_summary.json'}")


if __name__ == "__main__":
    main()
