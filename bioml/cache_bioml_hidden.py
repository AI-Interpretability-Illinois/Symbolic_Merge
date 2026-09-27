#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cache_bioml_hidden.py

One Gemma forward pass per Bio-ML context, keeping the preferred-label span's
residual stream at several layers, so that every SAE ablation (layer, width, L0,
SAE family on the same model) is an offline encode instead of a full rerun.

The span, tokenisation and hidden-state index are exactly those of
run_bioml_sae_from_bio8b_contexts.py (hidden_states[layer + 1] =
blocks.<layer>.hook_resid_post), so encoding the layer-20 cache with the default
SAE reproduces that script's representations.

Entities appearing in several context slices keep their first occurrence, in
the order the slices are given, which is what analyze_bioml_idf.py does.

Output: <output_dir>/shard-<k>.pt, each {"layers": [...], "entities": [
    {"side", "entity_iri", "label", "contexts": [
        {"text", "token_indices", "hidden": bf16 [n_layers, n_tokens, d_model]}]}]}

Example (one process per GPU):
    python bioml/cache_bioml_hidden.py --context_run_dirs <q20>,<q20_40>,... \
        --output_dir <cache> --layers 5,9,10,15,20,25,30,31,35,40 \
        --num_workers 4 --worker 0 --device cuda:0
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bioml_sae_from_bio8b_contexts import (  # noqa: E402
    atomic_torch_save, load_all_contexts, symbol_token_indices)


def collect_entities(context_dirs, n_contexts):
    seen, out = set(), []
    for d in context_dirs:
        src, tgt = load_all_contexts(d, n_contexts)
        for side, store in (("src", src), ("tgt", tgt)):
            for iri, ent in store.items():
                if (side, iri) in seen:
                    continue
                seen.add((side, iri))
                out.append((side, ent))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context_run_dirs", required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--model_path", default="/projects/biro/shared/models/gemma-2-9b")
    ap.add_argument("--layers", default="5,9,10,15,20,25,30,31,35,40")
    ap.add_argument("--contexts", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--shard_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    ctx_dirs = [Path(x).expanduser().resolve() for x in args.context_run_dirs.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entities = collect_entities(ctx_dirs, args.contexts)
    shards = [(k, entities[k:k + args.shard_size])
              for k in range(0, len(entities), args.shard_size)]
    mine = [s for i, s in enumerate(shards) if i % args.num_workers == args.worker]
    todo = [(k, ents) for k, ents in mine
            if not (args.output_dir / f"shard-{k:07d}.pt").exists()]
    print(f"entities={len(entities)} shards={len(shards)} worker={args.worker} "
          f"mine={len(mine)} todo={len(todo)} layers={layers}", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    model.eval()
    model.config.use_cache = False

    for n, (k, ents) in enumerate(todo, 1):
        rows = []
        for side, ent in ents:
            ctxs = []
            for text in ent["contexts"]:
                enc, token_indices, _, _ = symbol_token_indices(
                    tok, text, ent["label"], args.max_length)
                with torch.inference_mode():
                    out = model(**{k_: v.to(args.device) for k_, v in enc.items()},
                                output_hidden_states=True, use_cache=False, return_dict=True)
                idx = torch.tensor(token_indices, device=args.device, dtype=torch.long)
                hid = torch.stack([out.hidden_states[l + 1][0].index_select(0, idx)
                                   for l in layers]).to(torch.bfloat16).cpu()
                ctxs.append({"text": text, "token_indices": token_indices, "hidden": hid})
            rows.append({"side": side, "entity_iri": ent["entity_iri"],
                         "label": ent["label"], "contexts": ctxs})
        atomic_torch_save({"layers": layers, "entities": rows},
                          args.output_dir / f"shard-{k:07d}.pt")
        print(f"[worker {args.worker}] shard {k} done ({n}/{len(todo)})", flush=True)


if __name__ == "__main__":
    main()
