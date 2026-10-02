#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_valentine_sae.py

Column representations for schema matching on the Valentine benchmark.

Why this benchmark: a column's cells are independent samples of the same
symbol, so the views the information weighting needs occur naturally -- no
LLM-generated contexts, no manufactured windows. This is the closest analogue
to the Bio-ML setup, where one entity is observed in 5 independent contexts:

    Bio-ML                          Valentine
    entity                    ->    column
    5 generated sentences     ->    k sampled cell values
    entity label span         ->    the cell value (or the column name)
    pool span tokens, then    ->    same
      aggregate over contexts         aggregate over views

Per view we encode every token, mean over the symbol span, and then accumulate
across views, storing per feature

    count = views where it fires -> p_column = count / n_views
    sum   = sum of activations   -> mean activation
    sumsq = sum of squares       -> CV, hence reliability

Presets for the view text (--preset):
    value        "Table {table}, column {column}. Example value: {value}"
                 span = the value; names are available as left context
    value_only   "Example value: {value}"
                 span = the value; ablation with no names at all, which is the
                 realistic case for cryptic column identifiers
    name         "A data table column contains the value {value}. This column
                  is named {column}."
                 span = the column name, with the value as left context, the
                 closest mirror of Bio-ML's label-in-a-sentence structure
    cell         "{column}\n{value}"
                 span = the whole text: name and value together are the symbol
    column       "{column}\n{value 1}\n...\n{value k}", ONE view per column
                 span = the whole text, the table analogue of a program in
                 XLCoST, whose symbol is its full text (use --max_length 1024)

Example:
    python schema/run_valentine_sae.py \
        --data_root <Valentine-datasets> \
        --output_dir <out>/valentine \
        --model_path /projects/biro/shared/models/gemma-2-9b \
        --sae_path <gemma-scope>/layer_20/width_131k/average_l0_114/params.npz \
        --views 16 --limit_pairs 20
"""

import argparse
import sys
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from lm_device import load_causal_lm  # noqa: E402

# Gemma Scope layer_20/width_131k/average_l0_114: the canonical release, and the
# only layer-20 residual 131k variant with full Neuronpedia auto-interp coverage
# (its source id is gemma-2-9b/20-gemmascope-res-131k). Keep runs on this SAE so
# feature indices stay comparable across benchmarks and remain interpretable.
DEFAULT_SAE_PATH = ("/projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/layer_20/width_131k/average_l0_114/params.npz")

PRESETS = {
    "value": ("Table {table}, column {column}. Example value: ", "{value}", ""),
    "value_only": ("Example value: ", "{value}", ""),
    "name": ("A data table column contains the value {value}. This column is named ",
             "{column}", "."),
    "cell": ("", "{column}\n{value}", ""),
}
# "column" has no per-value template: the whole column is one view,
# "<name>\n<value 1>\n...\n<value k>", pooled over every token.
WHOLE_COLUMN = "column"


# =============================================================================
# Valentine dataset discovery
# =============================================================================

def find_pairs(root):
    """Locate (source csv, target csv, mapping json) triples under the root.

    The archive nests these as <relatedness>/<dataset>/<pair>/... , but layouts
    have shifted between releases, so discover by the mapping file instead of
    hard-coding depth.
    """
    pairs = []
    for mapping in sorted(Path(root).rglob("*mapping.json")):
        folder = mapping.parent
        csvs = sorted(folder.glob("*.csv"))
        if len(csvs) < 2:
            continue
        src = next((c for c in csvs if "source" in c.name.lower()), None)
        tgt = next((c for c in csvs if "target" in c.name.lower()), None)
        if src is None or tgt is None:
            src, tgt = csvs[0], csvs[1]
        rel = folder.relative_to(root)
        known = {"joinable", "semantically-joinable", "unionable", "view-unionable"}
        relatedness = next((p for p in rel.parts if p.lower() in known), "unknown")
        pairs.append({"name": str(rel).replace(os.sep, "__"),
                      "relatedness": relatedness,
                      "dataset": rel.parts[0],
                      "source_csv": str(src), "target_csv": str(tgt),
                      "mapping": str(mapping)})
    return pairs


def read_ground_truth(mapping_path):
    """Return [(source_column, target_column), ...] from a Valentine mapping."""
    obj = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    rows = obj.get("matches", obj) if isinstance(obj, dict) else obj
    out = []
    for m in rows:
        if isinstance(m, dict):
            sc = m.get("source_column", m.get("source_col", m.get("source")))
            tc = m.get("target_column", m.get("target_col", m.get("target")))
            if sc is not None and tc is not None:
                out.append((str(sc), str(tc)))
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            out.append((str(m[0]), str(m[1])))
    if not out:
        raise RuntimeError(f"no correspondences parsed from {mapping_path}")
    return out


def read_table(path, max_rows):
    for kwargs in ({}, {"sep": ";"}, {"encoding": "latin-1"},
                   {"sep": ";", "encoding": "latin-1"}):
        try:
            df = pd.read_csv(path, nrows=max_rows, low_memory=False, **kwargs)
            if df.shape[1] > 1 or not kwargs:
                return df
        except Exception:
            continue
    raise RuntimeError(f"could not parse {path}")


def sample_values(series, k, seed, max_chars):
    """k distinct non-null values, deterministic."""
    vals = series.dropna()
    seen, out = set(), []
    for v in vals:
        s = re.sub(r"\s+", " ", str(v)).strip()[:max_chars]
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= k * 4:
            break
    random.Random(seed).shuffle(out)
    return out[:k]


# =============================================================================
# SAE
# =============================================================================

class JumpReluSAE:
    def __init__(self, path, device):
        npz = np.load(str(path))

        def get(*keys):
            for k in keys:
                if k in npz:
                    return npz[k]
            raise KeyError(f"missing {keys}; file has {list(npz.keys())}")

        self.W = torch.from_numpy(get("W_enc", "w_enc")).to(device=device, dtype=torch.bfloat16)
        self.b = torch.from_numpy(get("b_enc")).to(device=device, dtype=torch.bfloat16)
        self.th = torch.from_numpy(get("threshold", "thresholds")).to(device=device,
                                                                     dtype=torch.bfloat16)
        self.d_in = int(self.W.shape[0])
        self.width = int(self.W.shape[1])

    @torch.inference_mode()
    def encode(self, x):
        pre = x.to(self.W.dtype) @ self.W + self.b
        return torch.where(pre > self.th, pre, torch.zeros_like(pre))


def load_sae(path, device):
    """Gemma Scope params.npz, or a Llama Scope checkpoints/final.safetensors."""
    if str(path).endswith(".safetensors"):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bioml"))
        from encode_bioml_cached import LlamaScopeSAE
        sae = LlamaScopeSAE(path, device)
        sae.d_in, sae.width = sae.d_model, sae.n_features
        return sae
    return JumpReluSAE(path, device)


# =============================================================================
# Encoder
# =============================================================================

class Encoder:
    def __init__(self, args):
        self.args = args
        model_path = str(Path(args.model_path).expanduser())
        print(f"[model] {model_path}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                                 local_files_only=True)
        self.model, args.device = load_causal_lm(
            model_path, args.device, args.max_gpu_memory, local_files_only=True)
        self.sae = load_sae(Path(args.sae_path).expanduser(), args.device)
        if int(self.model.config.hidden_size) != self.sae.d_in:
            raise RuntimeError(f"hidden {self.model.config.hidden_size} != d_in {self.sae.d_in}")
        print(f"[sae] width={self.sae.width} d_in={self.sae.d_in} layer={args.layer}", flush=True)

    def render(self, table, column, value):
        """Return (text, char_span_of_symbol)."""
        pre, sym, post = PRESETS[self.args.preset]
        fields = {"table": table, "column": column, "value": value}
        left = pre.format(**fields)
        middle = sym.format(**fields)
        right = post.format(**fields)
        return left + middle + right, (len(left), len(left) + len(middle))

    @torch.inference_mode()
    def view(self, text, span):
        """Span-pooled dense vector and SAE code for one view."""
        args = self.args
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True,
                       truncation=True, max_length=args.max_length, add_special_tokens=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        s, e = span
        ids = [i for i, (a, b) in enumerate(offsets) if b > a and a < e and b > s]
        if not ids:
            return None, None
        inputs = {k: v.to(args.device) for k, v in enc.items()}
        out = self.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        h = out.hidden_states[args.layer + 1][0][ids].to(args.device)
        dense = h.float().mean(0)
        # Encode each span token, then mean over the span: a feature counts as
        # present in this view when it fires on any token of the symbol.
        code = self.sae.encode(h).float().mean(0)
        del out, h
        return dense, code

    def column_rep(self, table, column, values, dtype):
        width = self.sae.width
        total = torch.zeros(width, dtype=torch.float32, device=self.args.device)
        totalsq = torch.zeros(width, dtype=torch.float32, device=self.args.device)
        count = torch.zeros(width, dtype=torch.int32, device=self.args.device)
        dense_rows, used = [], []
        if self.args.preset == WHOLE_COLUMN:
            text = "\n".join([column, *values])
            views = [(text, (0, len(text)), list(values))]
        else:
            views = [(*self.render(table, column, v), [v]) for v in values]
        for text, span, vs in views:
            dense, code = self.view(text, span)
            if code is None:
                continue
            total += code
            totalsq += code * code
            count += (code > 0).to(torch.int32)
            dense_rows.append(dense)
            used.extend(vs)
        if not dense_rows:
            return None
        idx = torch.nonzero(count, as_tuple=False).flatten()
        if self.args.keep_top_features and idx.numel() > self.args.keep_top_features:
            order = torch.argsort(total[idx], descending=True)[:self.args.keep_top_features]
            idx = idx[order].sort().values
        return {
            "table": table, "column": column, "dtype": str(dtype),
            "n_views": len(dense_rows), "values": used,
            "dense": torch.stack(dense_rows).mean(0).to(torch.float16).cpu(),
            "sae": {"idx": idx.to(torch.int32).cpu(), "sum": total[idx].cpu(),
                    "sumsq": totalsq[idx].cpu(), "count": count[idx].to(torch.int32).cpu(),
                    "size": width},
        }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--model_path", default="/projects/biro/shared/models/gemma-2-9b")
    ap.add_argument("--sae_path", default=DEFAULT_SAE_PATH,
                    help="defaults to average_l0_114 (Neuronpedia-interpretable)")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--device", default="cuda",
                    help="cuda, cuda:N, or 'split' to spread the model over all visible GPUs")
    ap.add_argument("--max_gpu_memory", default="7GiB,12GiB",
                    help="per-GPU weight caps for --device split (last value repeats)")
    ap.add_argument("--preset", default="value", choices=(*PRESETS, WHOLE_COLUMN))
    ap.add_argument("--views", type=int, default=16, help="cell values sampled per column")
    ap.add_argument("--max_rows", type=int, default=2000, help="rows read per csv")
    ap.add_argument("--max_value_chars", type=int, default=120)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--keep_top_features", type=int, default=8192)
    ap.add_argument("--max_columns", type=int, default=0, help="cap columns per table (0 = all)")
    ap.add_argument("--limit_pairs", type=int, default=0)
    ap.add_argument("--relatedness", default="", help="substring filter on the pair path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.data_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    (out / "pairs").mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(root)
    if args.relatedness:
        pairs = [p for p in pairs if args.relatedness.lower() in p["name"].lower()]
    if args.limit_pairs:
        pairs = pairs[:args.limit_pairs]
    if not pairs:
        raise RuntimeError(f"no Valentine pairs found under {root}")

    print("=" * 100)
    print(f"VALENTINE COLUMN REPRESENTATIONS  (preset={args.preset}, views={args.views})")
    print("=" * 100)
    print(f"pairs={len(pairs)}")

    encoder = None
    done = 0
    t0 = time.time()
    for pi, pair in enumerate(pairs, 1):
        dest = out / "pairs" / f"{pair['name']}.pt"
        if dest.exists() and not args.overwrite:
            print(f"[{pi}/{len(pairs)}] skip {pair['name']}")
            continue
        try:
            gt = read_ground_truth(pair["mapping"])
            src_df = read_table(pair["source_csv"], args.max_rows)
            tgt_df = read_table(pair["target_csv"], args.max_rows)
        except Exception as e:
            print(f"[{pi}/{len(pairs)}] SKIP {pair['name']}: {type(e).__name__}: {e}", flush=True)
            continue
        if encoder is None:
            encoder = Encoder(args)

        sides = {}
        for side, df, csv in (("source", src_df, pair["source_csv"]),
                              ("target", tgt_df, pair["target_csv"])):
            table = Path(csv).stem
            cols = list(df.columns)[:args.max_columns] if args.max_columns else list(df.columns)
            reps = []
            for ci, col in enumerate(cols):
                vals = sample_values(df[col], args.views,
                                     args.seed + pi * 1009 + ci, args.max_value_chars)
                if not vals:
                    continue
                rep = encoder.column_rep(table, str(col), vals, df[col].dtype)
                if rep is not None:
                    reps.append(rep)
            sides[side] = {"table": table, "csv": csv, "columns": reps}

        payload = {"pair": pair, "ground_truth": gt,
                   "preset": args.preset, "views": args.views,
                   "source": sides["source"], "target": sides["target"]}
        tmp = dest.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, dest)
        done += 1
        print(f"[{pi}/{len(pairs)}] {pair['name']}: "
              f"{len(sides['source']['columns'])}x{len(sides['target']['columns'])} columns, "
              f"{len(gt)} gold pairs ({time.time() - t0:.0f}s elapsed)", flush=True)

    config = {"data_root": str(root), "pairs_found": len(pairs), "pairs_written": done,
              "model_path": str(Path(args.model_path).expanduser()),
              "sae_path": str(Path(args.sae_path).expanduser()), "layer": args.layer,
              "preset": args.preset, "views": args.views, "max_rows": args.max_rows,
              "keep_top_features": args.keep_top_features,
              "pooling": "per_token_sae_encode_then_mean_over_symbol_span",
              "max_length": args.max_length,
              "view_unit": "whole_column" if args.preset == WHOLE_COLUMN else "sampled_cell_value"}
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nDONE in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
