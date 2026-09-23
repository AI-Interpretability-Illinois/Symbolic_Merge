#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_bioml_sae_from_bio8b_contexts.py

Read Bio-8B-generated contexts and run the Bio-ML representation/ranking experiment:

Bio-8B contexts
    -> Gemma-2-9B layer 20 hidden states
    -> pool ONLY the preferred-label token span
    -> Gemma Scope SAE (width 131k)
    -> Dense Mean / SAE Mean / SAE Stable
    -> source-vs-100-candidate cosine ranking
    -> Gold Rank + MRR/Hits/MeanRank

Expected context run_dir layout:
    <context_run_dir>/
      selected_queries.json
      contexts/
        src/*.json
        tgt/*.json

Expected simple context JSON fields:
    entity_iri
    entity_metadata_given_to_model.preferred_label
    parsed_output.contexts[*].text
OR, if parsed_output is missing:
    raw_model_output

Important:
- No SAE_single or Dense_single in the main evaluation.
- Stable frequencies are swept over 0.2,0.4,0.6,0.8,1.0 by default.
- With 5 contexts/entity, these are the only distinct frequency thresholds.
"""

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Gemma Scope layer_20/width_131k/average_l0_114: the canonical release, and the
# only layer-20 residual 131k variant with full Neuronpedia auto-interp coverage
# (its source id is gemma-2-9b/20-gemmascope-res-131k). Keep runs on this SAE so
# feature indices stay comparable across benchmarks and remain interpretable.
DEFAULT_SAE_PATH = ("/projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/layer_20/width_131k/average_l0_114/params.npz")

# =============================================================================
# Utilities
# =============================================================================

def atomic_json_dump(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch_save(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def iri_tail(iri: str) -> str:
    s = str(iri).rstrip("/")
    return s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]


def cosine_dense(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    na = a.norm()
    nb = b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return -1.0
    return float(torch.dot(a, b) / (na * nb))


def sparse_pack(x: torch.Tensor) -> Dict[str, Any]:
    x = x.float().cpu()
    idx = torch.nonzero(x != 0, as_tuple=False).flatten().to(torch.int32)
    val = x[idx.long()].to(torch.float16)
    return {"idx": idx, "val": val, "size": int(x.numel())}


def sparse_unpack(obj: Dict[str, Any]) -> torch.Tensor:
    out = torch.zeros(int(obj["size"]), dtype=torch.float32)
    idx = obj["idx"].long()
    out[idx] = obj["val"].float()
    return out


def sparse_cosine(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ai = a["idx"].long()
    bi = b["idx"].long()
    av = a["val"].float()
    bv = b["val"].float()

    if ai.numel() == 0 or bi.numel() == 0:
        return -1.0

    i = j = 0
    dot = 0.0

    while i < ai.numel() and j < bi.numel():
        x = int(ai[i])
        y = int(bi[j])

        if x == y:
            dot += float(av[i] * bv[j])
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1

    na = float(torch.linalg.vector_norm(av))
    nb = float(torch.linalg.vector_norm(bv))

    if na == 0.0 or nb == 0.0:
        return -1.0

    return dot / (na * nb)


# =============================================================================
# Context loading
# =============================================================================

def extract_json_object(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None

    return None


def load_context_file(path: Path, expected_contexts: int) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    iri = obj["entity_iri"]

    label = None
    if isinstance(obj.get("entity_metadata_given_to_model"), dict):
        label = obj["entity_metadata_given_to_model"].get("preferred_label")

    if not label:
        label = obj.get("preferred_label") or obj.get("label")

    if not label:
        raise RuntimeError(f"No preferred label found in {path}")

    parsed = obj.get("parsed_output")

    if not isinstance(parsed, dict):
        raw = obj.get("raw_model_output", "")
        parsed = extract_json_object(raw)

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Could not parse generated JSON in {path}")

    contexts_raw = parsed.get("contexts")

    if not isinstance(contexts_raw, list):
        raise RuntimeError(f"No contexts list in {path}")

    texts = []

    for item in contexts_raw:
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = item

        if text:
            texts.append(re.sub(r"\s+", " ", str(text).strip()))

    if len(texts) < expected_contexts:
        raise RuntimeError(
            f"{iri_tail(iri)} has only {len(texts)} contexts; expected {expected_contexts}"
        )

    texts = texts[:expected_contexts]

    return {
        "entity_iri": iri,
        "label": label,
        "contexts": texts,
        "source_file": str(path),
    }


def load_all_contexts(context_run_dir: Path, expected_contexts: int):
    src = {}
    tgt = {}

    for side, store in [("src", src), ("tgt", tgt)]:
        folder = context_run_dir / "contexts" / side

        if not folder.exists():
            raise FileNotFoundError(f"Missing context folder: {folder}")

        for path in sorted(folder.glob("*.json")):
            data = load_context_file(path, expected_contexts)
            store[data["entity_iri"]] = data

    return src, tgt


# =============================================================================
# Symbol span -> token indices
# =============================================================================

def symbol_token_indices(tokenizer, text: str, label: str, max_length: int):
    start = text.find(label)

    if start < 0:
        # Fallback only for capitalization differences.
        low_text = text.casefold()
        low_label = label.casefold()
        start = low_text.find(low_label)

        if start < 0:
            raise RuntimeError(
                f"Could not find entity label in context.\n"
                f"label={label!r}\ntext={text!r}"
            )

    end = start + len(label)

    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    offsets = enc.pop("offset_mapping")[0].tolist()

    token_indices = []

    for i, (a, b) in enumerate(offsets):
        if a == 0 and b == 0:
            continue
        if a < end and b > start:
            token_indices.append(i)

    if not token_indices:
        raise RuntimeError(
            f"No token overlaps preferred-label span for:\n{text}"
        )

    return enc, token_indices, [start, end], offsets


# =============================================================================
# SAE
# =============================================================================

class GemmaScopeSAE:
    def __init__(self, params_path: Path, device: torch.device):
        npz = np.load(params_path)

        for key in ["W_enc", "b_enc", "threshold"]:
            if key not in npz.files:
                raise RuntimeError(f"Missing SAE array: {key}")

        self.W_enc = torch.from_numpy(npz["W_enc"]).to(
            device=device,
            dtype=torch.bfloat16,
        )
        self.b_enc = torch.from_numpy(npz["b_enc"]).to(
            device=device,
            dtype=torch.bfloat16,
        )
        self.threshold = torch.from_numpy(npz["threshold"]).to(
            device=device,
            dtype=torch.bfloat16,
        )

        self.d_model = int(self.W_enc.shape[0])
        self.n_features = int(self.W_enc.shape[1])

        print(
            f"[SAE] d_model={self.d_model}, "
            f"features={self.n_features:,}"
        )

    @torch.inference_mode()
    def encode(self, x: torch.Tensor):
        x = x.to(self.W_enc.device, dtype=torch.bfloat16)
        pre = x @ self.W_enc + self.b_enc
        return torch.where(
            pre > self.threshold,
            pre,
            torch.zeros_like(pre),
        )


# =============================================================================
# Representation extraction
# =============================================================================

class Extractor:
    def __init__(
        self,
        model_path: str,
        sae_path: Path,
        layer: int,
        device: str,
        max_length: int,
    ):
        self.device = torch.device(device)
        self.layer = layer
        self.max_length = max_length

        print(f"[Gemma] tokenizer: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )

        print(f"[Gemma] model: {model_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to(self.device)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to(self.device)

        self.model.eval()
        self.model.config.use_cache = False

        self.sae = GemmaScopeSAE(sae_path, self.device)

        if int(self.model.config.hidden_size) != self.sae.d_model:
            raise RuntimeError(
                f"Gemma hidden size {self.model.config.hidden_size} "
                f"!= SAE d_model {self.sae.d_model}"
            )

    @torch.inference_mode()
    def context_representation(self, text: str, label: str):
        enc, token_indices, char_span, offsets = symbol_token_indices(
            self.tokenizer,
            text,
            label,
            self.max_length,
        )

        inputs = {
            k: v.to(self.device)
            for k, v in enc.items()
        }

        out = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        # hidden_states[0] = embedding output
        # hidden_states[layer+1] = output of transformer layer `layer`
        hidden = out.hidden_states[self.layer + 1][0]

        idx = torch.tensor(
            token_indices,
            device=self.device,
            dtype=torch.long,
        )

        symbol_hidden_tokens = hidden.index_select(0, idx)

        # Pool ONLY preferred-label tokens.
        dense = symbol_hidden_tokens.float().mean(dim=0).cpu()

        # Encode each symbol token, then mean over symbol tokens.
        token_sae = self.sae.encode(symbol_hidden_tokens).float().cpu()
        sae = token_sae.mean(dim=0)

        token_strings = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0, token_indices].tolist()
        )

        return {
            "text": text,
            "char_span": char_span,
            "token_indices": token_indices,
            "symbol_tokens": token_strings,
            "dense": dense.to(torch.float16),
            "sae": sparse_pack(sae),
        }

    def entity_representation(
        self,
        entity: Dict[str, Any],
        stable_frequencies: List[float],
    ):
        context_results = []

        dense_rows = []
        sae_rows = []

        for i, text in enumerate(entity["contexts"], 1):
            r = self.context_representation(
                text,
                entity["label"],
            )

            context_results.append(r)
            dense_rows.append(r["dense"].float())
            sae_rows.append(sparse_unpack(r["sae"]))

            print(
                f"    context {i}: "
                f"tokens={r['symbol_tokens']}"
            )

        D = torch.stack(dense_rows)
        Z = torch.stack(sae_rows)

        dense_mean = D.mean(dim=0)
        sae_mean = Z.mean(dim=0)

        active_counts = (Z > 0).sum(dim=0)
        n = Z.shape[0]

        stable = {}

        for freq in stable_frequencies:
            required = int(math.ceil(freq * n - 1e-12))
            mask = active_counts >= required
            vec = sae_mean * mask.float()

            stable[f"{freq:.2f}"] = {
                "required_count": required,
                "vector": sparse_pack(vec),
                "nnz": int((vec != 0).sum()),
            }

        return {
            "entity_iri": entity["entity_iri"],
            "label": entity["label"],
            "n_contexts": n,
            "contexts": context_results,
            "aggregate": {
                "dense_mean": dense_mean.to(torch.float16),
                "sae_mean": sparse_pack(sae_mean),
                "sae_stable": stable,
            },
        }


# =============================================================================
# Ranking / metrics
# =============================================================================

def score_rep(method: str, src: Dict[str, Any], tgt: Dict[str, Any], freq_key=None):
    if method == "dense_mean":
        return cosine_dense(
            src["aggregate"]["dense_mean"],
            tgt["aggregate"]["dense_mean"],
        )

    if method == "sae_mean":
        return sparse_cosine(
            src["aggregate"]["sae_mean"],
            tgt["aggregate"]["sae_mean"],
        )

    if method == "sae_stable":
        return sparse_cosine(
            src["aggregate"]["sae_stable"][freq_key]["vector"],
            tgt["aggregate"]["sae_stable"][freq_key]["vector"],
        )

    raise ValueError(method)


def metrics(ranks: List[int]):
    return {
        "N": len(ranks),
        "MRR": float(np.mean([1.0 / r for r in ranks])),
        "H@1": float(np.mean([r <= 1 for r in ranks])),
        "H@5": float(np.mean([r <= 5 for r in ranks])),
        "H@10": float(np.mean([r <= 10 for r in ranks])),
        "MeanRank": float(np.mean(ranks)),
        "MedianRank": float(np.median(ranks)),
        "gold_ranks": ranks,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--context_run_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--sae_path", type=Path, default=Path(DEFAULT_SAE_PATH),
                    help="defaults to average_l0_114 (Neuronpedia-interpretable)")

    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--contexts", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument(
        "--stable_frequencies",
        default="0.2,0.4,0.6,0.8,1.0",
    )

    ap.add_argument("--overwrite_representations", action="store_true")

    args = ap.parse_args()

    context_run_dir = args.context_run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_frequencies = [
        float(x)
        for x in args.stable_frequencies.split(",")
        if x.strip()
    ]

    query_path = context_run_dir / "selected_queries.json"

    if not query_path.exists():
        raise FileNotFoundError(
            f"Missing selected_queries.json:\n{query_path}"
        )

    queries = json.loads(
        query_path.read_text(encoding="utf-8")
    )

    src_contexts, tgt_contexts = load_all_contexts(
        context_run_dir,
        args.contexts,
    )

    print("=" * 100)
    print("BIO-ML SAE FROM BIO-8B CONTEXTS")
    print("=" * 100)
    print(f"queries         : {len(queries)}")
    print(f"source entities : {len(src_contexts)}")
    print(f"target entities : {len(tgt_contexts)}")
    print(f"contexts/entity : {args.contexts}")
    print(f"layer           : {args.layer}")
    print(f"stable freqs    : {stable_frequencies}")

    rep_dir = output_dir / "representations"
    src_rep_dir = rep_dir / "src"
    tgt_rep_dir = rep_dir / "tgt"

    extractor = Extractor(
        model_path=str(Path(args.model_path).expanduser().resolve()),
        sae_path=args.sae_path.expanduser().resolve(),
        layer=args.layer,
        device=args.device,
        max_length=args.max_length,
    )

    src_reps = {}
    tgt_reps = {}

    def rep_path(side, iri):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", iri_tail(iri))
        return rep_dir / side / f"{safe}.pt"

    # Source representations.
    for i, (iri, entity) in enumerate(src_contexts.items(), 1):
        path = rep_path("src", iri)

        if path.exists() and not args.overwrite_representations:
            rep = load_torch(path)
            print(f"[src {i}/{len(src_contexts)}] cached {entity['label']}")
        else:
            print(f"[src {i}/{len(src_contexts)}] {entity['label']}")
            rep = extractor.entity_representation(
                entity,
                stable_frequencies,
            )
            atomic_torch_save(rep, path)

        src_reps[iri] = rep

    # Target representations.
    for i, (iri, entity) in enumerate(tgt_contexts.items(), 1):
        path = rep_path("tgt", iri)

        if path.exists() and not args.overwrite_representations:
            rep = load_torch(path)
            print(f"[tgt {i}/{len(tgt_contexts)}] cached {entity['label']}")
        else:
            print(f"[tgt {i}/{len(tgt_contexts)}] {entity['label']}")
            rep = extractor.entity_representation(
                entity,
                stable_frequencies,
            )
            atomic_torch_save(rep, path)

        tgt_reps[iri] = rep

    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Ranking methods.
    method_specs = [
        ("dense_mean", None),
        ("sae_mean", None),
    ] + [
        ("sae_stable", f"{freq:.2f}")
        for freq in stable_frequencies
    ]

    all_results = {}

    for method, freq_key in method_specs:
        display_name = (
            method
            if freq_key is None
            else f"sae_stable_{freq_key}"
        )

        print("\n" + "=" * 100)
        print(display_name)
        print("=" * 100)

        ranks = []
        query_results = []

        for q in queries:
            qid = q["query_id"]
            src_iri = q["src"]
            gold = q["gold"]

            if src_iri not in src_reps:
                raise RuntimeError(
                    f"Missing source representation for query {qid}: {src_iri}"
                )

            missing = [
                c for c in q["candidates"]
                if c not in tgt_reps
            ]

            if missing:
                raise RuntimeError(
                    f"Query {qid} missing {len(missing)} candidate representations."
                )

            scored = []

            for candidate in q["candidates"]:
                score = score_rep(
                    method,
                    src_reps[src_iri],
                    tgt_reps[candidate],
                    freq_key=freq_key,
                )
                scored.append((candidate, score))

            scored.sort(
                key=lambda x: (-x[1], x[0])
            )

            gold_rank = next(
                i + 1
                for i, (iri, _) in enumerate(scored)
                if iri == gold
            )

            ranks.append(gold_rank)

            top10 = []

            for rank, (iri, score) in enumerate(scored[:10], 1):
                top10.append({
                    "rank": rank,
                    "entity_iri": iri,
                    "label": tgt_reps[iri]["label"],
                    "score": score,
                    "is_gold": iri == gold,
                })

            print(
                f"qid={qid:>3} gold_rank={gold_rank:>3} "
                f"gold={tgt_reps[gold]['label']}"
            )

            query_results.append({
                "query_id": qid,
                "source_iri": src_iri,
                "source_label": src_reps[src_iri]["label"],
                "gold_iri": gold,
                "gold_label": tgt_reps[gold]["label"],
                "gold_rank": gold_rank,
                "candidate_count": len(scored),
                "top10": top10,
                "full_ranking": [
                    {
                        "rank": rank,
                        "entity_iri": iri,
                        "label": tgt_reps[iri]["label"],
                        "score": score,
                        "is_gold": iri == gold,
                    }
                    for rank, (iri, score) in enumerate(scored, 1)
                ],
            })

        summary = metrics(ranks)

        all_results[display_name] = {
            "metrics": summary,
            "queries": query_results,
        }

        print(
            f"MRR={summary['MRR']:.4f} "
            f"H@1={summary['H@1']:.4f} "
            f"H@5={summary['H@5']:.4f} "
            f"H@10={summary['H@10']:.4f} "
            f"MeanRank={summary['MeanRank']:.2f}"
        )

    atomic_json_dump(
        {
            "context_run_dir": str(context_run_dir),
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "sae_path": str(args.sae_path.expanduser().resolve()),
            "layer": args.layer,
            "contexts_per_entity": args.contexts,
            "stable_frequencies": stable_frequencies,
            "results": all_results,
        },
        output_dir / "results.json",
    )

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    for name, obj in all_results.items():
        m = obj["metrics"]
        print(
            f"[{name:16s}] "
            f"MRR={m['MRR']:.4f} "
            f"H@1={m['H@1']:.4f} "
            f"H@5={m['H@5']:.4f} "
            f"H@10={m['H@10']:.4f} "
            f"MeanRank={m['MeanRank']:.2f}"
        )

    print(f"\nSaved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
