#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bioml_sae_v3.py

Corrected Bio-ML NCIT->DOID pilot for comparing:
  1) dense_single
  2) dense_mean
  3) sae_single
  4) sae_mean
  5) sae_stable

IMPORTANT CORRECTNESS RULES
---------------------------
A. ONLY the tokens overlapping the exact symbol span are represented.
   Context tokens never enter Dense/SAE span pooling.

B. Gemma Scope 9B PT residual layer_20 uses:
      blocks.20.hook_resid_post
   through TransformerLens.

C. Official Gemma Scope 9B PT residual SAE config uses:
      apply_b_dec_to_input = False
   Therefore the encoder is:
      pre = x @ W_enc + b_enc
      z   = pre where pre > threshold, else 0
   DO NOT subtract b_dec.

D. Ontology labels are read only from trusted label predicates.
   In particular, OBO metadata such as:
      has_obo_namespace -> "disease_ontology"
   is NOT a label.

E. local.valid.cands.tsv is parsed row-by-row.
   TgtCandidates is a serialized Python/JSON list.
   Each row gets a query_id so evaluation is never keyed only by SrcEntity.

F. This v2 script writes to neurosymbolic_outputs_v3 so it NEVER reuses
   representations produced by the previous buggy script.
"""

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, SKOS
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_VERSION = "3.0.0"

METHODS = [
    "dense_single",
    "dense_mean",
    "sae_single",
    "sae_mean",
    "sae_stable",
]


# ============================================================
# General utilities
# ============================================================

def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def iri_tail(iri: str) -> str:
    s = str(iri).rstrip("/")
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    na = a.norm()
    nb = b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return -1.0
    v = torch.dot(a, b) / (na * nb)
    return float(v)


def dedupe_keep_order(xs):
    return list(dict.fromkeys(xs))


# ============================================================
# Bio-ML pool parser
# ============================================================

def parse_candidate_list(raw: str) -> List[str]:
    """
    Bio-ML TgtCandidates is commonly a Python-list string (single quotes),
    not strict JSON. Try ast.literal_eval first, then JSON.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("empty TgtCandidates")

    obj = None
    try:
        obj = ast.literal_eval(raw)
    except Exception:
        try:
            obj = json.loads(raw)
        except Exception as e:
            raise ValueError(
                f"Could not parse TgtCandidates as Python/JSON list: {raw[:300]!r}"
            ) from e

    if not isinstance(obj, (list, tuple)):
        raise TypeError(
            f"TgtCandidates parsed as {type(obj).__name__}, expected list/tuple"
        )

    out = [str(x).strip() for x in obj if str(x).strip()]
    return dedupe_keep_order(out)


def load_local_pool(
    path: Path,
    limit: Optional[int],
    require_100_candidates: bool = True,
) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames or []

        print(f"[pool] fields = {fields}")

        required = {"SrcEntity", "TgtEntity", "TgtCandidates"}
        missing = required - set(fields)
        if missing:
            raise RuntimeError(
                f"Missing required columns {sorted(missing)} in {path}\n"
                f"Found: {fields}"
            )

        queries = []
        for row_idx, row in enumerate(reader):
            src = row["SrcEntity"].strip()
            gold = row["TgtEntity"].strip()
            cands = parse_candidate_list(row["TgtCandidates"])

            if gold not in cands:
                raise RuntimeError(
                    f"Gold target is not in candidate pool at TSV line {row_idx + 2}\n"
                    f"SrcEntity={src}\nGold={gold}\nCandidates={len(cands)}"
                )

            if require_100_candidates and len(cands) != 100:
                raise RuntimeError(
                    f"Expected exactly 100 candidates at TSV line {row_idx + 2}, "
                    f"found {len(cands)}."
                )

            queries.append({
                "query_id": row_idx,
                "src": src,
                "gold": gold,
                "candidates": cands,
            })

            if limit is not None and len(queries) >= limit:
                break

    if not queries:
        raise RuntimeError(f"No queries loaded from {path}")

    sizes = [len(q["candidates"]) for q in queries]
    unique_src = len({q["src"] for q in queries})
    repeated = len(queries) - unique_src

    print(
        f"[pool] loaded {len(queries)} ranking rows; "
        f"candidate sizes min/mean/max="
        f"{min(sizes)}/{np.mean(sizes):.1f}/{max(sizes)}"
    )
    print(
        f"[pool] unique SrcEntity={unique_src}; repeated-source rows={repeated}"
    )
    if repeated:
        print(
            "[pool][note] repeated SrcEntity values exist in this slice. "
            "Internal metrics are therefore keyed by query_id, not SrcEntity."
        )

    return queries


# ============================================================
# Ontology parser
# ============================================================

# Exact trusted local predicate names.
# We deliberately DO NOT use substring matching such as "name" in predicate.
LABEL_LOCAL_EXACT = {
    "label",
    "preflabel",
    "preferred_name",
    "preferredname",
    "preferred name",
    "p108",               # NCIt Preferred_Name
}

SYNONYM_LOCAL_EXACT = {
    "altlabel",
    "hasexactsynonym",
    "hasrelatedsynonym",
    "hasbroadsynonym",
    "hasnarrowsynonym",
    "exactsynonym",
    "relatedsynonym",
    "broadsynonym",
    "narrowsynonym",
    "synonym",
    "full_syn",
    "fullsynonym",
    "p90",                # NCIt FULL_SYN
}

DEFINITION_LOCAL_EXACT = {
    "definition",
    "def",
    "description",
    "iao_0000115",        # OBO textual definition
    "p97",                # NCIt DEFINITION
    "comment",
}

BAD_GENERIC_LABELS = {
    "disease_ontology",
    "disease ontology",
    "ontology",
    "owl ontology",
}


def predicate_local(pred: URIRef) -> str:
    return iri_tail(str(pred)).strip().lower()


def classify_literal_predicate(pred: URIRef) -> Optional[str]:
    # Highest-confidence standard predicates first.
    if pred == RDFS.label or pred == SKOS.prefLabel:
        return "label"
    if pred == SKOS.altLabel:
        return "synonym"
    if pred == RDFS.comment:
        return "definition"

    tail = predicate_local(pred)

    if tail in LABEL_LOCAL_EXACT:
        return "label"
    if tail in SYNONYM_LOCAL_EXACT:
        return "synonym"
    if tail in DEFINITION_LOCAL_EXACT:
        return "definition"

    return None


class OntologyIndex:
    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name
        self.graph = Graph()

        print(f"[ontology] parsing {name}: {path}")
        self.graph.parse(str(path))
        print(f"[ontology] {name} triples = {len(self.graph):,}")

        self.info: Dict[str, Dict[str, Any]] = {}
        self._build()

    def _build(self):
        recs = defaultdict(lambda: {
            "labels_rdfs": [],
            "labels_other": [],
            "synonyms": [],
            "definitions": [],
            "parents": [],
        })

        # Text annotations + direct URI parents.
        for s, p, o in self.graph:
            if not isinstance(s, URIRef):
                continue
            si = str(s)

            if p == RDFS.subClassOf and isinstance(o, URIRef):
                recs[si]["parents"].append(str(o))

            if not isinstance(o, Literal):
                continue

            kind = classify_literal_predicate(p)
            if kind is None:
                continue

            text = norm_text(o)
            if not text:
                continue

            if kind == "label":
                if p == RDFS.label or p == SKOS.prefLabel:
                    recs[si]["labels_rdfs"].append(text)
                else:
                    recs[si]["labels_other"].append(text)
            elif kind == "synonym":
                recs[si]["synonyms"].append(text)
            elif kind == "definition":
                recs[si]["definitions"].append(text)

        # Make sure every explicit OWL class gets a record.
        for s in self.graph.subjects(RDF.type, OWL.Class):
            if isinstance(s, URIRef):
                recs[str(s)]

        # Dedupe raw lists.
        for d in recs.values():
            for k in (
                "labels_rdfs", "labels_other",
                "synonyms", "definitions", "parents"
            ):
                d[k] = dedupe_keep_order(d[k])

        # Preferred label selection.
        for iri, d in recs.items():
            if d["labels_rdfs"]:
                label = d["labels_rdfs"][0]
                source = "rdfs/skos-label"
            elif d["labels_other"]:
                label = d["labels_other"][0]
                source = "trusted-preferred-label"
            else:
                label = None
                source = None

            d["label"] = label
            d["label_source"] = source

        # Resolve parent labels after all preferred labels are chosen.
        for iri, d in recs.items():
            parent_labels = []
            for piri in d["parents"]:
                pd = recs.get(piri)
                if pd and pd.get("label"):
                    parent_labels.append(pd["label"])
            d["parent_labels"] = dedupe_keep_order(parent_labels)

            if d["label"]:
                lab_cf = norm_text(d["label"]).casefold()
                d["synonyms"] = [
                    s for s in d["synonyms"]
                    if norm_text(s).casefold() != lab_cf
                ]

        self.info = dict(recs)

    def get_required(self, iri: str) -> Dict[str, Any]:
        if iri not in self.info:
            raise KeyError(
                f"{self.name}: required entity is absent from ontology: {iri}"
            )
        d = self.info[iri]
        label = d.get("label")

        if not label:
            raise RuntimeError(
                f"{self.name}: no trusted lexical label for required entity:\n{iri}\n"
                "Refusing to fall back to the IRI because that would invalidate "
                "the symbol-representation experiment."
            )

        if norm_text(label).casefold() in BAD_GENERIC_LABELS:
            raise RuntimeError(
                f"{self.name}: suspicious generic label {label!r} for entity {iri}.\n"
                "This usually means ontology metadata was mistaken for a class label. "
                "The v3 parser should never accept this."
            )

        return d


def discover_ontology_files(ontology_dir: Path) -> Tuple[Path, Path]:
    files = list(ontology_dir.glob("*.owl")) + list(ontology_dir.glob("*.rdf"))

    ncit = [
        p for p in files
        if "ncit" in p.name.lower() or "thesaurus" in p.name.lower()
    ]
    doid = [p for p in files if "doid" in p.name.lower()]

    if len(ncit) != 1 or len(doid) != 1:
        raise RuntimeError(
            "Could not uniquely identify NCIT and DOID ontology files.\n"
            f"NCIT candidates: {ncit}\n"
            f"DOID candidates: {doid}\n"
            f"All ontology files: {files}"
        )
    return ncit[0], doid[0]


def write_entity_label_audit(
    path: Path,
    src_iris: List[str],
    tgt_iris: List[str],
    ncit: OntologyIndex,
    doid: OntologyIndex,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "Side", "IRI", "Label", "LabelSource",
            "NSynonyms", "NDefinitions", "NParents"
        ])

        for side, iris, ont in [
            ("src", src_iris, ncit),
            ("tgt", tgt_iris, doid),
        ]:
            for iri in iris:
                d = ont.get_required(iri)
                w.writerow([
                    side,
                    iri,
                    d["label"],
                    d["label_source"],
                    len(d["synonyms"]),
                    len(d["definitions"]),
                    len(d["parent_labels"]),
                ])


# ============================================================
# Controlled contexts
# ============================================================

def clip(text: str, n: int = 500) -> str:
    return norm_text(text)[:n]


def build_contexts(
    entity: Dict[str, Any],
    n: int,
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Each context contains the preferred label verbatim exactly at a tracked span.
    The model sees the whole context, BUT downstream representation pooling uses
    only token positions overlapping this tracked label span.
    """
    label = entity["label"]
    synonyms = entity.get("synonyms", [])
    definitions = entity.get("definitions", [])
    parents = entity.get("parent_labels", [])

    frames: List[Tuple[str, str]] = [
        (
            "The biomedical concept ",
            " is being described."
        ),
        (
            "In this biomedical ontology, ",
            " is a named concept."
        ),
    ]

    if definitions:
        frames.append((
            "The concept ",
            f" has the following ontology description: {clip(definitions[0])}"
        ))
    else:
        frames.append((
            "The biomedical concept ",
            " appears in clinical or biological terminology."
        ))

    if synonyms:
        syn = "; ".join(clip(x, 120) for x in synonyms[:3])
        frames.append((
            "The ontology entity ",
            f" has lexical alternatives including: {syn}."
        ))
    else:
        frames.append((
            "The ontology entity ",
            " is part of a biomedical vocabulary."
        ))

    if parents:
        par = "; ".join(clip(x, 120) for x in parents[:2])
        frames.append((
            "The biomedical entity ",
            f" occurs under broader ontology concepts including: {par}."
        ))
    else:
        frames.append((
            "Within the ontology hierarchy, ",
            " participates in biomedical conceptual structure."
        ))

    extras = [
        ("A biomedical terminology contains the concept ", "."),
        ("A domain expert may encounter the entity ", " in ontology data."),
        ("The named biomedical symbol ", " appears in this ontology."),
        ("For ontology matching, the concept ", " is represented here."),
    ]
    frames.extend(extras)

    if n > len(frames):
        raise ValueError(
            f"--contexts={n} requested, but only {len(frames)} deterministic "
            "context templates are defined."
        )

    out = []
    for prefix, suffix in frames[:n]:
        text = prefix + label + suffix
        start = len(prefix)
        end = start + len(label)
        if text[start:end] != label:
            raise AssertionError("Internal character-span construction error")
        out.append((text, (start, end)))

    return out


# ============================================================
# Exact symbol character span -> token indices
# ============================================================

def tokenize_symbol_span(
    tokenizer,
    text: str,
    char_span: Tuple[int, int],
    max_length: int,
):
    start, end = char_span

    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )

    offsets = enc.pop("offset_mapping")[0].tolist()

    token_indices = []
    for i, (a, b) in enumerate(offsets):
        # special tokens usually have (0, 0)
        if a == 0 and b == 0:
            continue
        # [a,b) overlaps [start,end)
        if a < end and b > start:
            token_indices.append(i)

    if not token_indices:
        raise RuntimeError(
            "No tokenizer token overlaps the symbol span.\n"
            f"text={text!r}\n"
            f"symbol={text[start:end]!r}\n"
            f"char_span={char_span}"
        )

    cov_start = min(offsets[i][0] for i in token_indices)
    cov_end = max(offsets[i][1] for i in token_indices)

    if cov_start > start or cov_end < end:
        raise RuntimeError(
            "Selected token offsets do not fully cover symbol span.\n"
            f"symbol={text[start:end]!r}\n"
            f"char_span={char_span}\n"
            f"token_offsets={[offsets[i] for i in token_indices]}"
        )

    return enc, token_indices, offsets


# ============================================================
# Gemma Scope JumpReLU SAE
# ============================================================

class GemmaScopeSAE:
    """
    Correct encoder for google/gemma-scope-9b-pt-res.

    Gemma Scope 9B PT residual SAE metadata has:
      apply_b_dec_to_input = False
      normalize_activations = None

    Therefore:
      pre = x @ W_enc + b_enc
      z = pre * 1[pre > threshold]

    b_dec is decoder-side and is NOT subtracted from x.
    """

    def __init__(
        self,
        params_path: Path,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        npz = np.load(params_path)
        print(f"[sae] keys = {list(npz.files)}")

        required = {"W_enc", "b_enc", "threshold"}
        missing = required - set(npz.files)
        if missing:
            raise RuntimeError(
                f"SAE NPZ missing required arrays {sorted(missing)}; "
                f"found {list(npz.files)}"
            )

        for k in npz.files:
            print(
                f"[sae] {k}: shape={npz[k].shape}, dtype={npz[k].dtype}"
            )

        self.W_enc = torch.from_numpy(npz["W_enc"]).to(
            device=device, dtype=dtype
        )
        self.b_enc = torch.from_numpy(npz["b_enc"]).to(
            device=device, dtype=dtype
        )
        self.threshold = torch.from_numpy(npz["threshold"]).to(
            device=device, dtype=dtype
        )

        self.d_model = int(self.W_enc.shape[0])
        self.n_features = int(self.W_enc.shape[1])
        self.dtype = dtype
        self.device = device

        print(
            f"[sae] d_model={self.d_model}, n_features={self.n_features}"
        )
        print(
            "[sae] IMPORTANT: apply_b_dec_to_input=False; "
            "b_dec is NOT subtracted."
        )

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"SAE expected [N,{self.d_model}], got {tuple(x.shape)}"
            )

        x = x.to(self.device, dtype=self.dtype)
        pre = x @ self.W_enc + self.b_enc
        return torch.where(
            pre > self.threshold,
            pre,
            torch.zeros_like(pre),
        )


# ============================================================
# Gemma + representation extraction
# ============================================================

class RepresentationExtractor:
    """
    HF-only Gemma extractor.

    Why HF-only:
      Hugging Face output_hidden_states returns the residual hidden state at
      the output of each decoder layer. TransformerLens hook_resid_post is
      likewise the residual stream at the output of a block.

    HF convention:
      hidden_states[0]      = embedding output
      hidden_states[L + 1]  = output of decoder layer L

    Therefore Gemma Scope layer_20 residual activation is taken as:
      hidden_states[21]

    This avoids constructing a second 9B TransformerLens copy of Gemma, which
    can exceed a 64 GB CPU-memory allocation during conversion.
    """

    def __init__(
        self,
        model_path: str,
        sae_path: Path,
        layer: int,
        max_length: int,
        device: str,
        max_reasonable_token_l0: int,
    ):
        self.device = torch.device(device)
        self.layer = layer
        self.max_length = max_length
        self.max_reasonable_token_l0 = max_reasonable_token_l0
        self.activation_name = f"HF hidden_states[{layer + 1}] (layer {layer} output / resid_post)"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                "A fast tokenizer is required for exact offset_mapping."
            )

        print(f"[model] loading local Gemma 2 9B in BF16: {model_path}")

        # Load directly onto GPU with low_cpu_mem_usage so there is only one
        # model copy. device_map is deliberately not used here because the
        # experiment is designed for a single allocated GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model.eval()
        self.model.config.use_cache = False

        n_layers = int(self.model.config.num_hidden_layers)
        d_model = int(self.model.config.hidden_size)

        if not (0 <= layer < n_layers):
            raise ValueError(
                f"Requested layer {layer}, but model has {n_layers} layers."
            )

        print(
            f"[model] n_layers={n_layers}, d_model={d_model}, "
            f"activation={self.activation_name}"
        )
        print(
            "[model] HF convention: hidden_states[0]=embedding; "
            f"hidden_states[{layer + 1}]=output after layer {layer}"
        )

        self.sae = GemmaScopeSAE(
            sae_path,
            device=self.device,
            dtype=torch.bfloat16,
        )

        if d_model != self.sae.d_model:
            raise RuntimeError(
                f"Model d_model={d_model} but SAE d_in={self.sae.d_model}"
            )

    @torch.inference_mode()
    def one_context(
        self,
        text: str,
        char_span: Tuple[int, int],
        debug: bool = False,
    ):
        enc, token_indices, offsets = tokenize_symbol_span(
            self.tokenizer,
            text,
            char_span,
            self.max_length,
        )

        model_inputs = {
            k: v.to(self.device)
            for k, v in enc.items()
        }

        out = self.model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hs_index = self.layer + 1

        if out.hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")

        if hs_index >= len(out.hidden_states):
            raise RuntimeError(
                f"Requested hidden_states[{hs_index}], but model returned "
                f"{len(out.hidden_states)} hidden-state tensors."
            )

        # [batch, seq, d_model] -> [seq, d_model]
        hidden = out.hidden_states[hs_index][0]

        idx = torch.tensor(
            token_indices,
            device=self.device,
            dtype=torch.long,
        )

        # CRITICAL: ONLY symbol-span tokens are selected.
        symbol_h = hidden.index_select(0, idx)

        # Dense representation uses exactly the same symbol token activations.
        dense_span = symbol_h.float().mean(dim=0)

        # SAE is applied independently to each symbol token.
        token_codes = self.sae.encode(symbol_h)

        token_l0 = (
            (token_codes > 0)
            .sum(dim=1)
            .detach()
            .cpu()
            .tolist()
        )
        token_l0_mean = float(np.mean(token_l0))
        token_l0_max = int(max(token_l0))

        # Strong sanity check. For an average_l0_53 SAE, individual tokens can
        # deviate substantially, but thousands of active features indicate an
        # activation/SAE mismatch.
        if token_l0_max > self.max_reasonable_token_l0:
            raise RuntimeError(
                "SAE L0 sanity check failed.\n"
                f"symbol={text[char_span[0]:char_span[1]]!r}\n"
                f"per-symbol-token L0={token_l0}\n"
                f"max allowed={self.max_reasonable_token_l0}\n"
                f"activation={self.activation_name}\n"
                "For average_l0_53, values in the thousands indicate an "
                "activation-point / SAE-encoder mismatch. Aborting."
            )

        # Span-level SAE representation:
        # average feature magnitudes across ONLY the symbol tokens.
        sae_span = token_codes.float().mean(dim=0)

        if debug:
            start, end = char_span
            toks = self.tokenizer.convert_ids_to_tokens(
                model_inputs["input_ids"][0, token_indices].tolist()
            )

            print("[debug-span]")
            print(f"  text          : {text}")
            print(f"  symbol        : {text[start:end]!r}")
            print(f"  token idx     : {token_indices}")
            print(f"  tokens        : {toks}")
            print(
                f"  offsets       : "
                f"{[offsets[i] for i in token_indices]}"
            )
            print(f"  activation    : {self.activation_name}")
            print(f"  token SAE L0  : {token_l0}")
            print(
                f"  span SAE nnz  : "
                f"{int((sae_span > 0).sum().item())}"
            )

        # Release references to all hidden layers immediately.
        del out
        del hidden
        del symbol_h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            dense_span.cpu(),
            sae_span.cpu(),
            token_l0_mean,
            token_l0_max,
        )

    def entity_representations(
        self,
        entity: Dict[str, Any],
        n_contexts: int,
        stable_frequency: float,
        debug: bool = False,
    ) -> Dict[str, Any]:
        contexts = build_contexts(entity, n_contexts)

        dense_all = []
        sae_all = []
        token_l0_means = []
        token_l0_maxima = []

        for ci, (text, span) in enumerate(contexts):
            d, z, l0mean, l0max = self.one_context(
                text,
                span,
                debug=(debug and ci == 0),
            )

            dense_all.append(d)
            sae_all.append(z)
            token_l0_means.append(l0mean)
            token_l0_maxima.append(l0max)

        D = torch.stack(dense_all, dim=0)
        Z = torch.stack(sae_all, dim=0)

        dense_single = D[0]
        dense_mean = D.mean(dim=0)

        sae_single = Z[0]
        sae_mean = Z.mean(dim=0)

        # Cross-context support persistence.
        frequency = (Z > 0).float().mean(dim=0)
        stable_mask = frequency >= stable_frequency
        sae_stable = sae_mean * stable_mask.float()

        context_nnz = (Z > 0).sum(dim=1).tolist()

        return {
            "dense_single": dense_single,
            "dense_mean": dense_mean,
            "sae_single": sae_single,
            "sae_mean": sae_mean,
            "sae_stable": sae_stable,
            "sae_frequency": frequency,
            "_diag_token_l0_mean": token_l0_means,
            "_diag_token_l0_max": token_l0_maxima,
            "_diag_context_span_nnz": context_nnz,
        }


# ============================================================
# Cache
# ============================================================

def save_cache(path: Path, cache: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    print(f"[cache] loading {path}")
    obj = torch.load(path, map_location="cpu")
    if obj.get("__script_version__") != SCRIPT_VERSION:
        raise RuntimeError(
            f"Cache version mismatch in {path}. "
            "Delete it or use the v3 output directory."
        )
    return obj


# ============================================================
# Ranking, metrics, output
# ============================================================

def compute_metrics(
    ranked_by_qid: Dict[int, List[str]],
    gold_by_qid: Dict[int, str],
):
    rr = []
    h1 = h5 = h10 = 0
    ranks = {}

    for qid, ranked in ranked_by_qid.items():
        gold = gold_by_qid[qid]
        rank = next(
            (i + 1 for i, tgt in enumerate(ranked) if tgt == gold),
            None,
        )
        if rank is None:
            raise RuntimeError(
                f"Gold target absent from ranking for query_id={qid}"
            )

        ranks[qid] = rank
        rr.append(1.0 / rank)
        h1 += int(rank <= 1)
        h5 += int(rank <= 5)
        h10 += int(rank <= 10)

    n = len(rr)
    return {
        "n": n,
        "MRR": float(np.mean(rr)),
        "Hits@1": h1 / n,
        "Hits@5": h5 / n,
        "Hits@10": h10 / n,
        "ranks": ranks,
    }


def write_method_outputs(
    outdir: Path,
    method: str,
    queries: List[Dict[str, Any]],
    scores_by_qid: Dict[int, List[Tuple[str, float]]],
    metrics: Dict[str, Any],
):
    # Detailed row-aware file (always unambiguous).
    detailed = outdir / f"{method}.detailed.tsv"
    with detailed.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "QueryID", "SrcEntity", "GoldEntity",
            "TgtCandidate", "Score", "Rank", "IsGold"
        ])

        qmap = {q["query_id"]: q for q in queries}

        for qid, scored in scores_by_qid.items():
            q = qmap[qid]
            ranked = sorted(scored, key=lambda x: (-x[1], x[0]))
            for rank, (tgt, score) in enumerate(ranked, start=1):
                w.writerow([
                    qid,
                    q["src"],
                    q["gold"],
                    tgt,
                    f"{score:.10f}",
                    rank,
                    int(tgt == q["gold"]),
                ])

    # Bio-ML BLOCK-like file.
    # Fully official-compatible only when each SrcEntity identifies one query.
    block = outdir / f"{method}.tsv"
    with block.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["SrcEntity", "TgtCandidate", "Score"])
        qmap = {q["query_id"]: q for q in queries}

        for qid, scored in scores_by_qid.items():
            q = qmap[qid]
            for tgt, score in scored:
                w.writerow([
                    q["src"],
                    tgt,
                    f"{score:.10f}",
                ])

    return detailed, block


def summarize_diagnostics(cache, src_iris, tgt_iris):
    def collect(prefix, iris):
        stable_nnz = []
        token_l0_mean = []
        token_l0_max = []
        context_span_nnz = []

        for iri in iris:
            d = cache[f"{prefix}|{iri}"]
            stable_nnz.append(
                int((d["sae_stable"] > 0).sum().item())
            )
            token_l0_mean.extend(d["_diag_token_l0_mean"])
            token_l0_max.extend(d["_diag_token_l0_max"])
            context_span_nnz.extend(d["_diag_context_span_nnz"])

        return {
            "stable_nnz_mean": float(np.mean(stable_nnz)),
            "stable_nnz_median": float(np.median(stable_nnz)),
            "stable_nnz_max": int(np.max(stable_nnz)),
            "token_l0_mean": float(np.mean(token_l0_mean)),
            "token_l0_median": float(np.median(token_l0_mean)),
            "token_l0_max_seen": int(np.max(token_l0_max)),
            "context_span_nnz_mean": float(np.mean(context_span_nnz)),
            "context_span_nnz_median": float(np.median(context_span_nnz)),
        }

    return {
        "source": collect("src", src_iris),
        "target": collect("tgt", tgt_iris),
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--repo_dir",
        type=Path,
        default=Path.home() / "bio_ml_exp/OAEI-Bio-ML",
    )
    ap.add_argument("--pair", default="NCIT-DOID")
    ap.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid"],
    )
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--contexts", type=int, default=5)
    ap.add_argument(
        "--stable_frequency",
        type=float,
        default=0.6,
    )

    ap.add_argument(
        "--model_path",
        type=str,
        default=str(Path.home() / "models/gemma-2-9b"),
    )
    ap.add_argument(
        "--sae_path",
        type=Path,
        default=(
            Path.home()
            / "sae/gemma-scope-9b-layer20-131k-l0_53/"
            / "layer_20/width_131k/average_l0_53/params.npz"
        ),
    )
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug_first", action="store_true")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument(
        "--max_reasonable_token_l0",
        type=int,
        default=500,
        help=(
            "Fail if any symbol token activates more than this many SAE "
            "features. average_l0_53 should not produce thousands/token."
        ),
    )
    ap.add_argument(
        "--allow_non100_candidates",
        action="store_true",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    if args.pair != "NCIT-DOID":
        raise NotImplementedError(
            "v2 pilot intentionally supports NCIT-DOID only."
        )
    if not (0 < args.stable_frequency <= 1):
        raise ValueError("--stable_frequency must be in (0,1]")

    repo = args.repo_dir.expanduser().resolve()
    sae_path = args.sae_path.expanduser().resolve()
    model_path = str(Path(args.model_path).expanduser().resolve())

    pool_path = (
        repo / "bio-ml" / args.pair
        / f"local.{args.split}.cands.tsv"
    )
    ontology_dir = repo / "bio-ml" / "ontologies"

    if not pool_path.exists():
        raise FileNotFoundError(pool_path)
    if not sae_path.exists():
        raise FileNotFoundError(sae_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)

    if args.output_dir is None:
        outdir = (
            repo
            / "neurosymbolic_outputs_v3"
            / (
                f"{args.pair}_{args.split}_n{args.limit}"
                f"_ctx{args.contexts}_L{args.layer}"
                f"_stable{args.stable_frequency:g}"
            )
        )
    else:
        outdir = args.output_dir.expanduser().resolve()

    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 94)
    print(f"Bio-ML SAE pilot v{SCRIPT_VERSION}")
    print(f"repo              = {repo}")
    print(f"pool              = {pool_path}")
    print(f"model             = {model_path}")
    print(f"sae               = {sae_path}")
    print(f"layer/hook        = {args.layer} / blocks.{args.layer}.hook_resid_post")
    print(f"contexts/entity   = {args.contexts}")
    print(f"stable_frequency  = {args.stable_frequency}")
    print(f"query limit       = {args.limit}")
    print(f"output            = {outdir}")
    print("=" * 94)

    # ---------- Pool ----------
    queries = load_local_pool(
        pool_path,
        limit=args.limit,
        require_100_candidates=not args.allow_non100_candidates,
    )

    # ---------- Ontologies ----------
    ncit_path, doid_path = discover_ontology_files(ontology_dir)
    ncit = OntologyIndex(ncit_path, "NCIT")
    doid = OntologyIndex(doid_path, "DOID")

    src_iris = dedupe_keep_order(q["src"] for q in queries)
    tgt_iris = dedupe_keep_order(
        tgt for q in queries for tgt in q["candidates"]
    )

    print(f"[entities] unique NCIT source entities = {len(src_iris)}")
    print(f"[entities] unique DOID target entities = {len(tgt_iris)}")

    # Fail-fast label QA BEFORE loading the 9B model.
    for iri in src_iris:
        ncit.get_required(iri)
    for iri in tgt_iris:
        doid.get_required(iri)

    audit_path = outdir / "entity_labels.tsv"
    write_entity_label_audit(
        audit_path,
        src_iris,
        tgt_iris,
        ncit,
        doid,
    )

    print("[label-audit] first NCIT labels:")
    for iri in src_iris[:5]:
        d = ncit.get_required(iri)
        print(
            f"  {iri_tail(iri)} :: {d['label']} "
            f"[{d['label_source']}]"
        )

    print("[label-audit] first DOID labels:")
    for iri in tgt_iris[:20]:
        d = doid.get_required(iri)
        print(
            f"  {iri_tail(iri)} :: {d['label']} "
            f"[{d['label_source']}]"
        )

    # Explicitly guarantee old broken cache cannot be reused.
    cache_path = outdir / "representations_v3.pt"
    if args.no_cache:
        cache = {"__script_version__": SCRIPT_VERSION}
    else:
        cache = load_cache(cache_path)
        if not cache:
            cache = {"__script_version__": SCRIPT_VERSION}

    # ---------- Model / SAE ----------
    extractor = RepresentationExtractor(
        model_path=model_path,
        sae_path=sae_path,
        layer=args.layer,
        max_length=args.max_length,
        device=args.device,
        max_reasonable_token_l0=args.max_reasonable_token_l0,
    )

    def get_or_compute(
        side: str,
        iri: str,
        entity_info: Dict[str, Any],
        debug: bool = False,
    ):
        key = f"{side}|{iri}"
        if key not in cache:
            print(
                f"[repr] {side}: {iri_tail(iri)} :: "
                f"{entity_info['label']}"
            )
            cache[key] = extractor.entity_representations(
                entity_info,
                n_contexts=args.contexts,
                stable_frequency=args.stable_frequency,
                debug=debug,
            )
            if not args.no_cache:
                n_repr = sum(
                    1 for k in cache
                    if k.startswith("src|") or k.startswith("tgt|")
                )
                if n_repr % 25 == 0:
                    save_cache(cache_path, cache)
        return cache[key]

    for i, iri in enumerate(src_iris):
        get_or_compute(
            "src",
            iri,
            ncit.get_required(iri),
            debug=(args.debug_first and i == 0),
        )

    for iri in tgt_iris:
        get_or_compute(
            "tgt",
            iri,
            doid.get_required(iri),
            debug=False,
        )

    if not args.no_cache:
        save_cache(cache_path, cache)

    # ---------- Ranking ----------
    summary = {}
    gold_by_qid = {
        q["query_id"]: q["gold"]
        for q in queries
    }

    all_query_src_unique = (
        len({q["src"] for q in queries}) == len(queries)
    )

    for method in METHODS:
        ranked_by_qid = {}
        scores_by_qid = {}

        for q in queries:
            svec = cache[f"src|{q['src']}"][method]
            scored = []

            for tgt in q["candidates"]:
                tvec = cache[f"tgt|{tgt}"][method]
                score = cosine(svec, tvec)
                if not math.isfinite(score):
                    score = -1.0
                scored.append((tgt, score))

            scored_sorted = sorted(
                scored,
                key=lambda x: (-x[1], x[0]),
            )

            scores_by_qid[q["query_id"]] = scored
            ranked_by_qid[q["query_id"]] = [
                tgt for tgt, _ in scored_sorted
            ]

        metrics = compute_metrics(
            ranked_by_qid,
            gold_by_qid,
        )
        ranks = metrics.pop("ranks")
        summary[method] = metrics

        write_method_outputs(
            outdir,
            method,
            queries,
            scores_by_qid,
            {**metrics, "ranks": ranks},
        )

        print(
            f"[{method:12s}] "
            f"MRR={metrics['MRR']:.4f} "
            f"H@1={metrics['Hits@1']:.4f} "
            f"H@5={metrics['Hits@5']:.4f} "
            f"H@10={metrics['Hits@10']:.4f}"
        )

    # ---------- Diagnostics ----------
    diagnostics = summarize_diagnostics(
        cache,
        src_iris,
        tgt_iris,
    )

    print("\n[SAE diagnostics]")
    for side in ("source", "target"):
        d = diagnostics[side]
        print(f"  {side}:")
        print(
            f"    token L0 mean/median/max = "
            f"{d['token_l0_mean']:.2f} / "
            f"{d['token_l0_median']:.2f} / "
            f"{d['token_l0_max_seen']}"
        )
        print(
            f"    context span nnz mean/median = "
            f"{d['context_span_nnz_mean']:.2f} / "
            f"{d['context_span_nnz_median']:.2f}"
        )
        print(
            f"    stable nnz mean/median/max = "
            f"{d['stable_nnz_mean']:.2f} / "
            f"{d['stable_nnz_median']:.2f} / "
            f"{d['stable_nnz_max']}"
        )

    result = {
        "script_version": SCRIPT_VERSION,
        "config": {
            "repo_dir": str(repo),
            "pair": args.pair,
            "split": args.split,
            "limit": args.limit,
            "contexts": args.contexts,
            "stable_frequency": args.stable_frequency,
            "model_path": model_path,
            "sae_path": str(sae_path),
            "layer": args.layer,
            "activation_name": f"HF hidden_states[{args.layer + 1}] (layer {args.layer} output / resid_post)",
            "model_dtype": "bfloat16",
            "sae_dtype": "bfloat16",
            "sae_apply_b_dec_to_input": False,
            "symbol_span_only": True,
            "sae_span_pool": "mean_over_symbol_tokens_only",
            "max_reasonable_token_l0": args.max_reasonable_token_l0,
        },
        "summary": summary,
        "diagnostics": diagnostics,
        "official_block_format_safe_for_this_slice": all_query_src_unique,
    }

    with (outdir / "summary_v3.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    print("\nOutputs:")
    print(f"  label audit : {audit_path}")
    print(f"  summary     : {outdir / 'summary_v3.json'}")
    print(f"  cache       : {cache_path}")
    for method in METHODS:
        print(f"  {method}:")
        print(f"    {outdir / (method + '.tsv')}")
        print(f"    {outdir / (method + '.detailed.tsv')}")

    if all_query_src_unique:
        print(
            "\n[official scorer] This slice has unique SrcEntity values. "
            "For a FULL validation run, the BLOCK file is compatible with "
            "the Bio-ML scorer:"
        )
        print(
            "python3 scoring_kit/score_local.py "
            f"{outdir / 'sae_stable.tsv'} "
            f"{pool_path}"
        )
    else:
        print(
            "\n[official scorer note] This limited slice contains repeated "
            "SrcEntity values. Use the script's row-aware internal metrics "
            "for this debug run; inspect the full validation serialization "
            "before official scoring."
        )


if __name__ == "__main__":
    main()
