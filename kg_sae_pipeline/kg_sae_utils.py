#!/usr/bin/env python3
"""Shared utilities for the OAEI KG context, SAE, and ranking programs.

The three pipeline entry points should keep orchestration here and import
these definitions instead of maintaining separate copies of the core logic.
"""

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import torch


def atomic_json(obj, path):
    path = Path(path)
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


def tail(value):
    text = unquote(str(value)).rstrip("/")
    text = text.rsplit("#", 1)[-1] if "#" in text else text.rsplit("/", 1)[-1]
    return text.replace("_", " ").strip()


def safe_id(iri):
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", tail(iri))[:70] or "entity"
    return base + "_" + hashlib.sha1(str(iri).encode()).hexdigest()[:10]


def safe_name(value):
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "entity")[:180]


def save_pt(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# Context schema and exact entity-span handling
ID_KEYS = ("entity_iri", "iri", "entity_id", "id", "uri", "focus_iri")
LABEL_KEYS = ("label", "symbol", "entity_label", "name", "focus_label")
TEXT_KEYS = ("text", "context", "sentence")
SPAN_KEYS = ("char_span", "symbol_char_span", "span")


def first(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def span2(value):
    if isinstance(value, dict) and "start" in value and "end" in value:
        return [int(value["start"]), int(value["end"])]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    return None


def norm_context(context, entity_id=None, label=None):
    text = first(context, TEXT_KEYS)
    if not isinstance(text, str):
        return None
    entity_id = first(context, ID_KEYS) or entity_id
    label = first(context, LABEL_KEYS) or label
    if isinstance(label, dict):
        label = first(label, LABEL_KEYS)
    if label is None:
        raise ValueError("context has no label/symbol")
    label = str(label)
    span = span2(first(context, SPAN_KEYS))
    if span is None:
        positions = [m.start() for m in re.finditer(re.escape(label), text)]
        if len(positions) != 1:
            raise ValueError(f"missing char_span; exact label occurs {len(positions)} times")
        span = [positions[0], positions[0] + len(label)]
    start, end = span
    if text[start:end] != label:
        raise ValueError(f"span mismatch: {text[start:end]!r} != {label!r}")
    result = dict(context)
    result.update(text=text, char_span=span, symbol=label)
    if entity_id is not None:
        result["entity_iri"] = str(entity_id)
    return result


def harvest_contexts(obj, source):
    out = []

    def add(record, entity_id=None, label=None):
        if not isinstance(record, dict):
            return
        entity_id = first(record, ID_KEYS) or entity_id
        label = first(record, LABEL_KEYS) or label
        if isinstance(label, dict):
            label = first(label, LABEL_KEYS)
        if isinstance(record.get("contexts"), list):
            for context in record["contexts"]:
                normalized = norm_context(context, entity_id, label)
                if normalized:
                    out.append((str(normalized.get("entity_iri", entity_id or source)), normalized["symbol"], normalized))
        elif first(record, TEXT_KEYS) is not None:
            normalized = norm_context(record, entity_id, label)
            out.append((str(normalized.get("entity_iri", entity_id or source)), normalized["symbol"], normalized))

    if isinstance(obj, list):
        for item in obj:
            add(item)
    elif isinstance(obj, dict):
        if "contexts" in obj or first(obj, TEXT_KEYS) is not None:
            add(obj)
        else:
            used = False
            for key in ("entities", "records", "items", "data"):
                if isinstance(obj.get(key), list):
                    used = True
                    for item in obj[key]:
                        add(item)
            if not used:
                for key, value in obj.items():
                    if isinstance(value, dict):
                        add(value, key)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                add(item, key)
    return out


def load_contexts(path):
    path = Path(path).expanduser()
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    grouped = defaultdict(list)
    labels = {}
    errors = []
    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                obj = json.load(f)
            for entity_id, label, context in harvest_contexts(obj, file_path.stem):
                grouped[entity_id].append(context)
                labels[entity_id] = label
        except Exception as exc:
            errors.append((str(file_path), str(exc)))
    if not grouped:
        raise RuntimeError("No usable contexts. Need text + label/symbol + char_span.\n" + str(errors[:10]))
    for entity_id in grouped:
        seen = set()
        unique = []
        for context in grouped[entity_id]:
            key = (context["text"], tuple(context["char_span"]), context["symbol"])
            if key not in seen:
                seen.add(key)
                unique.append(context)
        grouped[entity_id] = unique
    return grouped, labels, errors


# Gemma/S​​AE span extraction
class SAE(torch.nn.Module):
    def __init__(self, path, device):
        super().__init__()
        data = np.load(str(path))

        def get(*keys):
            for key in keys:
                if key in data:
                    return data[key]
            raise KeyError(f"missing {keys}; keys={list(data.keys())}")

        weights = get("W_enc", "w_enc")
        bias = get("b_enc")
        threshold = get("threshold", "thresholds")
        self.register_buffer("W", torch.from_numpy(weights).to(torch.bfloat16))
        self.register_buffer("b", torch.from_numpy(bias).to(torch.bfloat16))
        self.register_buffer("th", torch.from_numpy(threshold).to(torch.bfloat16))
        self.to(device)
        self.d_in = weights.shape[0]
        self.width = weights.shape[1]

    @torch.inference_mode()
    def encode(self, hidden_states):
        """Encode each position first, then mean in SAE feature space."""
        pre = hidden_states.to(self.W.dtype) @ self.W + self.b
        activations = torch.where(pre > self.th, pre, torch.zeros_like(pre))
        if activations.ndim == 1:
            indices = torch.nonzero(activations, as_tuple=False).flatten()
            return indices, activations[indices], None
        if activations.ndim != 2:
            raise ValueError(f"expected [d_in] or [tokens,d_in], got {tuple(activations.shape)}")
        pooled = activations.mean(dim=0)
        indices = torch.nonzero(pooled, as_tuple=False).flatten()
        token_nnz = (activations > 0).sum(dim=1).detach().to(torch.int32)
        return indices, pooled[indices], token_nnz


def token_ids(offsets, span):
    start, end = span
    ids = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not ids:
        raise ValueError(f"no tokens overlap {span}")
    return ids


def aggregate_context_sae(contexts, width):
    sums = torch.zeros(width)
    counts = torch.zeros(width, dtype=torch.int32)
    for context in contexts:
        indices = context["sae"]["idx"].long()
        values = context["sae"]["val"].float()
        sums[indices] += values
        counts[indices] += 1
    n = len(contexts)
    return sums / n, counts.float() / n


# DNA weighting and ranking
def cosine(a, b):
    if a is None or b is None:
        return None
    a = torch.as_tensor(a, dtype=torch.float32).flatten()
    b = torch.as_tensor(b, dtype=torch.float32).flatten()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return 0.0 if float(denominator) == 0.0 else float(torch.dot(a, b) / denominator)


def sparse_cosine(a, b):
    if not a or not b:
        return 0.0
    small, other = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(float(value) * float(other.get(index, 0.0)) for index, value in small.items())
    norm_a = math.sqrt(sum(float(value) ** 2 for value in a.values()))
    norm_b = math.sqrt(sum(float(value) ** 2 for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def load_queries(path):
    with open(path, encoding="utf-8") as f:
        query_list = json.load(f)
    if isinstance(query_list, dict):
        query_list = query_list.get("queries", query_list.get("items", []))
    if not isinstance(query_list, list) or not query_list:
        raise RuntimeError("No queries")
    for index, query in enumerate(query_list):
        if not all(key in query for key in ("src", "gold", "candidates")):
            raise RuntimeError(f"query {index} schema error")
        if str(query["gold"]) not in {str(x) for x in query["candidates"]}:
            raise RuntimeError(f"query {index}: gold not in candidates")
    return query_list


def load_representations(directory, query_list):
    directory = Path(directory)
    representations = {}
    for file_path in directory.glob("*.pt"):
        record = load_pt(file_path)
        if isinstance(record, dict) and record.get("entity_iri") is not None:
            representations[str(record["entity_iri"])] = record
    needed = {str(query["src"]) for query in query_list}
    needed.update(str(candidate) for query in query_list for candidate in query["candidates"])
    for entity_id in needed - representations.keys():
        file_path = directory / (safe_name(entity_id) + ".pt")
        if file_path.exists():
            representations[entity_id] = load_pt(file_path)
    return representations


def feature_values(record):
    contexts = record.get("contexts", [])
    values = defaultdict(lambda: [0.0] * len(contexts))
    for context_index, context in enumerate(contexts):
        sae = context.get("sae", {})
        indices = torch.as_tensor(sae.get("idx", []), dtype=torch.long).flatten().tolist()
        activations = torch.as_tensor(sae.get("val", []), dtype=torch.float32).flatten().tolist()
        if len(indices) != len(activations):
            raise RuntimeError("SAE idx/val mismatch")
        for index, activation in zip(indices, activations):
            values[int(index)][context_index] = float(activation)
    return values, len(contexts)


def build_background(records):
    counts = Counter()
    total_contexts = 0
    for record in records:
        for context in record.get("contexts", []):
            indices = torch.as_tensor(context.get("sae", {}).get("idx", []), dtype=torch.long).flatten().tolist()
            counts.update(set(map(int, indices)))
            total_contexts += 1
    if not total_contexts:
        raise RuntimeError("zero background contexts")
    return counts, total_contexts


def build_dna_signatures(record, background_counts, background_total, epsilon=1e-6):
    values, n_contexts = feature_values(record)
    info = {}
    reliability = {}
    for feature_id, per_context in values.items():
        activations = torch.tensor(per_context, dtype=torch.float32)
        p_entity = float((activations > 0).sum()) / n_contexts
        p_background = background_counts.get(feature_id, 0) / background_total
        mean = float(activations.mean())
        std = float(activations.std(unbiased=False))
        enrichment = max(math.log((p_entity + epsilon) / (p_background + epsilon)), 0.0)
        reliable = 1.0 / (1.0 + std / (abs(mean) + epsilon))
        info_weight = mean * p_entity * enrichment
        final_weight = info_weight * reliable
        if info_weight > 0:
            info[feature_id] = info_weight
        if final_weight > 0:
            reliability[feature_id] = final_weight
    return {"sae_info": info, "sae_info_reliability": reliability}


def representation_vector(record, method):
    if method == "dense_mean":
        return record.get("dense_mean")
    if method == "sae_mean":
        return record.get("sae_mean")
    if method.startswith("sae_stable_"):
        threshold = float(method.rsplit("_", 1)[-1])
        for key, value in record.get("sae_stable", {}).items():
            if abs(float(key) - threshold) < 1e-9:
                return value
    return None


def score_representations(source, target, method, dna_signatures):
    if method.startswith("sae_info"):
        return sparse_cosine(dna_signatures[id(source)][method], dna_signatures[id(target)][method])
    return cosine(representation_vector(source, method), representation_vector(target, method))


def summarize_ranks(ranks):
    n = len(ranks)
    return {
        "n": n,
        "H@1": sum(rank == 1 for rank in ranks) / n,
        "H@5": sum(rank <= 5 for rank in ranks) / n,
        "H@10": sum(rank <= 10 for rank in ranks) / n,
        "MRR": sum(1.0 / rank for rank in ranks) / n,
        "MeanRank": sum(ranks) / n,
    }
