#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_xlcost_sae.py

Representations for XLCoST XL code search (code-to-code retrieval).

Input is the official dataset layout used by XLCoST's own
code/codesearch/code/run.py:

    <data_dir>/<Lang>/<split>.jsonl        Lang in C++ Java Python C# Javascript PHP C

with the official field names, read exactly as run.py reads them:

    docstring_tokens             -> QUERY side (the source-language code in the
                                    code2code task; XLCoST reuses the NL slot)
    code_tokens/function_tokens  -> CANDIDATE side (target-language code)
    url, idx                     -> identifiers used by evaluator.py

For each side of each row we embed the WHOLE snippet / program (the retrieval
level is just which dataset folder you point at):

    dense = mean over all real tokens of the layer-L hidden state
    SAE   = encode EVERY token, then mean over tokens

The encode-then-pool order matters. The SAE threshold applies per token, so
encoding an already-pooled vector silently drops features that fire on only a
few tokens. bioml/run_bioml_sae_from_bio8b_contexts.py uses this same order.

Tokens are the "views" of a document, so the Bio-ML information weighting
carries over unchanged. Per feature we store

    count = tokens where it fires  -> p_entity = count / n_tokens
    sum   = sum of activations     -> mean activation
    sumsq = sum of squares         -> CV, hence reliability

which is all eval_xlcost_sae.py needs, and is far smaller than the full
[n_tokens x n_features] code.

Example:
    python xlcost/run_xlcost_sae.py \
        --data_dir  <code2codesearch>/dataset/program_level \
        --lang Java --split test --level program \
        --model_path ~/models/gemma-2-9b \
        --sae_path <gemma-scope>/layer_20/width_131k/average_l0_114/params.npz \
        --output_dir <out>/xlcost_java_program
"""

import argparse
import sys
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from lm_device import load_causal_lm  # noqa: E402

# Gemma Scope layer_20/width_131k/average_l0_114: the canonical release, and the
# only layer-20 residual 131k variant with full Neuronpedia auto-interp coverage
# (its source id is gemma-2-9b/20-gemmascope-res-131k). Keep runs on this SAE so
# feature indices stay comparable across benchmarks and remain interpretable.
DEFAULT_SAE_PATH = ("/projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/layer_20/width_131k/average_l0_114/params.npz")
DEFAULT_MODEL_PATH = "/projects/biro/shared/models/gemma-2-9b"

LANGS = ("C++", "Java", "Python", "C#", "Javascript", "PHP", "C")
SIDES = ("query", "candidate")


# =============================================================================
# Data
# =============================================================================

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return rows


def side_text(js, side):
    """Join tokens exactly like XLCoST's convert_examples_to_features."""
    if side == "candidate":
        tokens = js["code_tokens"] if "code_tokens" in js else js["function_tokens"]
    else:
        tokens = js["docstring_tokens"]
    return " ".join(tokens)


# =============================================================================
# SAE
# =============================================================================

class JumpReluSAE:
    """Gemma Scope JumpReLU encoder; operates on [n_tokens, d_in]."""

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
# Representation for one text
# =============================================================================

class Encoder:
    def __init__(self, args):
        self.args = args
        model_path = str(Path(args.model_path).expanduser())
        print(f"[model] {model_path}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                                 local_files_only=args.local_files_only)
        self.model, args.device = load_causal_lm(
            model_path, args.device, args.max_gpu_memory, local_files_only=args.local_files_only)
        self.sae = load_sae(Path(args.sae_path).expanduser(), args.device)
        hidden = int(self.model.config.hidden_size)
        if hidden != self.sae.d_in:
            raise RuntimeError(f"hidden size {hidden} != SAE d_in {self.sae.d_in}")
        print(f"[sae] width={self.sae.width} d_in={self.sae.d_in} layer={args.layer}", flush=True)

    @torch.inference_mode()
    def __call__(self, text):
        args = self.args
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True,
                       truncation=True, max_length=args.max_length, add_special_tokens=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        # Real tokens only: BOS/pad have empty character spans.
        keep = [i for i, (a, b) in enumerate(offsets) if b > a]
        if not keep:
            return None
        inputs = {k: v.to(args.device) for k, v in enc.items()}
        out = self.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        # hidden_states[0] is the embedding output, so layer L is at L+1.
        h = out.hidden_states[args.layer + 1][0][keep].to(args.device)
        n_tokens = int(h.shape[0])

        dense = h.float().mean(0)

        width = self.sae.width
        total = torch.zeros(width, dtype=torch.float32, device=args.device)
        totalsq = torch.zeros(width, dtype=torch.float32, device=args.device)
        count = torch.zeros(width, dtype=torch.int32, device=args.device)
        for lo in range(0, n_tokens, args.token_chunk):
            z = self.sae.encode(h[lo:lo + args.token_chunk]).float()
            total += z.sum(0)
            totalsq += (z * z).sum(0)
            count += (z > 0).sum(0).to(torch.int32)
        del out, h

        idx = torch.nonzero(count, as_tuple=False).flatten()
        if args.keep_top_features and idx.numel() > args.keep_top_features:
            order = torch.argsort(total[idx], descending=True)[:args.keep_top_features]
            idx = idx[order].sort().values
        return {
            "n_tokens": n_tokens,
            "n_chars": len(text),
            "truncated": bool(len(offsets) >= args.max_length),
            "dense": dense.to(torch.float16).cpu(),
            "sae": {
                "idx": idx.to(torch.int32).cpu(),
                "sum": total[idx].cpu(),
                "sumsq": totalsq[idx].cpu(),
                "count": count[idx].to(torch.int32).cpu(),
                "size": width,
            },
        }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=None,
                    help="XLCoST code2codesearch dataset/{snippet_level,program_level}")
    ap.add_argument("--lang", default=None, choices=LANGS)
    ap.add_argument("--dataset_file", type=Path, default=None,
                    help="any jsonl in the same format (e.g. formal/prepare_minif2f.py "
                         "output); replaces --data_dir/--lang/--split")
    ap.add_argument("--split", default="test", choices=("test", "val", "train"))
    ap.add_argument("--level", default=None, choices=("snippet", "program"),
                    help="metadata only; inferred from --data_dir when omitted")
    ap.add_argument("--output_dir", type=Path, required=True)

    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--sae_path", default=DEFAULT_SAE_PATH,
                    help="defaults to average_l0_114 (Neuronpedia-interpretable)")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--device", default="cuda",
                    help="cuda, cuda:N, or 'split' to spread the model over all visible GPUs")
    ap.add_argument("--max_gpu_memory", default="7GiB,12GiB",
                    help="per-GPU weight caps for --device split (last value repeats)")
    ap.add_argument("--max_length", type=int, default=1024,
                    help="512 matches the official block_size; programs need more")
    ap.add_argument("--token_chunk", type=int, default=64,
                    help="tokens encoded per SAE call; caps peak memory")
    ap.add_argument("--keep_top_features", type=int, default=8192,
                    help="per document, keep this many features by total activation (0 = all)")
    ap.add_argument("--local_files_only", action="store_true", default=True)

    ap.add_argument("--sides", default="both", choices=("both", "query", "candidate"))
    ap.add_argument("--shard_size", type=int, default=256)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_dedupe_texts", dest="dedupe_texts", action="store_false",
                    help="encode every row even when the text repeats (query texts "
                         "repeat once per target language)")
    args = ap.parse_args()

    if args.dataset_file:
        path = args.dataset_file.expanduser().resolve()
        data_dir = path.parent
        args.lang = args.lang or path.parent.name
        args.split = path.stem
    else:
        if not (args.data_dir and args.lang):
            ap.error("--data_dir and --lang are required without --dataset_file")
        data_dir = args.data_dir.expanduser().resolve()
        path = data_dir / args.lang / f"{args.split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset file: {path}")
    level = args.level or ("snippet" if "snippet" in data_dir.name else "program")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(path)
    lo = args.start
    hi = len(rows) if not args.limit else min(len(rows), lo + args.limit)
    todo = list(range(lo, hi))
    sides = SIDES if args.sides == "both" else (args.sides,)

    print("=" * 100)
    print(f"XLCoST XL CODE SEARCH REPRESENTATIONS  ({args.lang}, {level} level, {args.split})")
    print("=" * 100)
    print(f"rows={len(rows)} processing={len(todo)} sides={sides} max_length={args.max_length}")

    encoder = None
    manifest = {s: [] for s in sides}
    text_cache, cache_hits = {}, [0]
    t0 = time.time()

    for side in sides:
        text_cache.clear()
        side_dir = out / "representations" / side
        side_dir.mkdir(parents=True, exist_ok=True)
        for shard_start in range(lo, hi, args.shard_size):
            shard_rows = [i for i in todo
                          if shard_start <= i < shard_start + args.shard_size]
            if not shard_rows:
                continue
            shard_path = side_dir / f"shard-{shard_start:07d}.pt"
            if shard_path.exists() and not args.overwrite:
                print(f"[{side}] skip existing {shard_path.name}", flush=True)
                manifest[side].append(str(shard_path))
                continue
            if encoder is None:
                encoder = Encoder(args)
            payload = []
            for i in shard_rows:
                js = rows[i]
                text = side_text(js, side)
                # The same query program is paired with every target language, so
                # query texts repeat ~4x; encode each distinct text once.
                key = hash(text) if args.dedupe_texts else None
                rep = text_cache.get(key) if key is not None else None
                if rep is None:
                    rep = encoder(text)
                    if key is not None and rep is not None:
                        text_cache[key] = rep
                else:
                    cache_hits[0] += 1
                rep = dict(rep) if rep is not None else None
                if rep is None:
                    print(f"[{side}] row {i} ({js.get('idx')}) has no usable tokens; skipped",
                          flush=True)
                    continue
                rep.update(row=i, url=js["url"], idx=js["idx"], side=side)
                payload.append(rep)
            tmp = shard_path.with_suffix(".pt.tmp")
            torch.save({"rows": payload, "lang": args.lang, "level": level,
                        "split": args.split, "side": side}, tmp)
            os.replace(tmp, shard_path)
            manifest[side].append(str(shard_path))
            done = shard_start + len(shard_rows) - lo
            rate = done / max(1e-9, time.time() - t0)
            print(f"[{side}] {shard_path.name}: {len(payload)} docs "
                  f"({done}/{len(todo)}, {rate:.1f} docs/s, "
                  f"cache_hits={cache_hits[0]})", flush=True)

    config = {
        "dataset_file": str(path),
        "lang": args.lang,
        "level": level,
        "split": args.split,
        "rows_in_file": len(rows),
        "rows_processed": [lo, hi],
        "sides": list(sides),
        "model_path": str(Path(args.model_path).expanduser()),
        "sae_path": str(Path(args.sae_path).expanduser()),
        "layer": args.layer,
        "hidden_states_index": args.layer + 1,
        "max_length": args.max_length,
        "keep_top_features": args.keep_top_features,
        "dedupe_texts": args.dedupe_texts,
        "encoder_cache_hits": cache_hits[0],
        "pooling": "per_token_sae_encode_then_mean_over_all_tokens",
        "dense_pooling": "mean_over_all_tokens",
        "shards": manifest,
    }
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nDONE in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
