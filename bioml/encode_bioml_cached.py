#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encode_bioml_cached.py

Turn a cache_bioml_hidden.py cache into a Bio-ML run directory for one
(layer, SAE) pair, without touching the language model. The output has the same
representations/{src,tgt}/<entity>.pt layout and fields as
run_bioml_sae_from_bio8b_contexts.py, so analyze_bioml_idf.py reads it as is.

Each label token is SAE-encoded and then averaged over the span, as in the
original runner; dense is the span mean of the same layer's hidden state.

Example:
    python bioml/encode_bioml_cached.py --cache_dir <cache> --layer 20 \
        --sae_path <gemma-scope>/layer_20/width_16k/average_l0_68/params.npz \
        --output_dir <runs>/L20_w16k --device cuda:0
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bioml_sae_from_bio8b_contexts import (  # noqa: E402
    GemmaScopeSAE, atomic_json_dump, atomic_torch_save, iri_tail, sparse_pack,
    sparse_unpack)


class LlamaScopeSAE:
    """Llama Scope (OpenMOSS lm-saes) JumpReLU SAE from checkpoints/final.safetensors.

    Inputs are rescaled dataset-wise to norm sqrt(d_model), and the threshold
    applies to the pre-activation times the decoder column norm
    (sparsity_include_decoder_norm). Activations are returned in that scaled form,
    i.e. as if the decoder were unit-norm, which is Gemma Scope's convention.
    """

    def __init__(self, ckpt, device):
        import json
        from safetensors.torch import load_file
        hp = json.loads((Path(ckpt).parent.parent / "hyperparams.json").read_text())
        if hp["act_fn"] != "jumprelu" or hp["norm_activation"] != "dataset-wise":
            raise RuntimeError(f"unsupported Llama Scope config: {hp['act_fn']}, "
                               f"{hp['norm_activation']}")
        w = load_file(str(ckpt))
        self.W_enc = w["encoder.weight"].T.contiguous().to(device, torch.bfloat16)
        self.b_enc = w["encoder.bias"].to(device, torch.bfloat16)
        dec_norm = w["decoder.weight"].float().norm(dim=0)
        self.dec_norm = (dec_norm if hp.get("sparsity_include_decoder_norm")
                         else torch.ones_like(dec_norm)).to(device, torch.bfloat16)
        self.threshold = float(hp["jump_relu_threshold"])
        self.scale = math.sqrt(hp["d_model"]) / hp["dataset_average_activation_norm"]["in"]
        self.d_model, self.n_features = int(hp["d_model"]), int(hp["d_sae"])
        print(f"[LlamaScope] d_model={self.d_model} features={self.n_features:,} "
              f"threshold={self.threshold} input_scale={self.scale:.3f}")

    @torch.inference_mode()
    def encode(self, x):
        x = x.to(self.W_enc.device, torch.bfloat16) * self.scale
        pre = (x @ self.W_enc + self.b_enc) * self.dec_norm
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))


def load_sae(path, device):
    if str(path).endswith(".safetensors"):
        return LlamaScopeSAE(path, device)
    return GemmaScopeSAE(path, device)


def entity_rep(ent, li, sae, device, stable_frequencies):
    ctxs, dense_rows, sae_rows = [], [], []
    for c in ent["contexts"]:
        h = c["hidden"][li].to(device)
        dense = h.float().mean(0).cpu()
        z = sae.encode(h).float().mean(0).cpu()
        packed = sparse_pack(z)
        ctxs.append({"text": c["text"], "token_indices": c["token_indices"],
                     "dense": dense.to(torch.float16), "sae": packed})
        # Aggregate from the stored fp16 values, as the original runner does, so
        # sae_mean / sae_stable reproduce it exactly rather than to ~1e-3.
        dense_rows.append(dense.to(torch.float16).float())
        sae_rows.append(sparse_unpack(packed))
    D, Z = torch.stack(dense_rows), torch.stack(sae_rows)
    sae_mean = Z.mean(0)
    counts = (Z > 0).sum(0)
    n = Z.shape[0]
    stable = {}
    for f in stable_frequencies:
        req = int(math.ceil(f * n - 1e-12))
        vec = sae_mean * (counts >= req).float()
        stable[f"{f:.2f}"] = {"required_count": req, "vector": sparse_pack(vec),
                              "nnz": int((vec != 0).sum())}
    return {"entity_iri": ent["entity_iri"], "label": ent["label"], "n_contexts": n,
            "contexts": ctxs,
            "aggregate": {"dense_mean": D.mean(0).to(torch.float16),
                          "sae_mean": sparse_pack(sae_mean), "sae_stable": stable}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sae_path", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stable_frequencies", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads; the per-entity ops are tiny, and the "
                         "default (all cores) is ~20x slower on a shared node")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    freqs = [float(x) for x in args.stable_frequencies.split(",") if x.strip()]
    out = args.output_dir.expanduser().resolve()
    done = out / "results.json"
    if done.exists():
        print(f"already done: {done}")
        return
    sae = load_sae(args.sae_path, torch.device(args.device))

    n = 0
    for shard in sorted(args.cache_dir.glob("shard-*.pt")):
        obj = torch.load(shard, map_location="cpu", weights_only=False)
        li = obj["layers"].index(args.layer)
        for ent in obj["entities"]:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", iri_tail(ent["entity_iri"]))
            with torch.inference_mode():
                rep = entity_rep(ent, li, sae, args.device, freqs)
            atomic_torch_save(rep, out / "representations" / ent["side"] / f"{safe}.pt")
            n += 1
        print(f"{shard.name}: {n} entities", flush=True)

    atomic_json_dump({"cache_dir": str(args.cache_dir), "layer": args.layer,
                      "sae_path": str(args.sae_path), "n_features": sae.n_features,
                      "entities": n, "pooling": "per_token_sae_encode_then_mean_over_label_span"},
                     done)


if __name__ == "__main__":
    main()
