#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_sae_vs_neuronpedia.py

Check that a local Gemma Scope SAE, loaded and applied the way this repo applies
it, reproduces the activation values Neuronpedia publishes for the same feature.

A file hash proves the weights are the ones Neuronpedia indexed. It says nothing
about whether we read the right hook point, index the right hidden state, or
handle BOS the same way -- any of which would silently produce a different
dictionary of "features" under the same indices. This script closes that gap by
replaying Neuronpedia's own top-activating example through the local model and
SAE and comparing activation vectors token by token.

Neuronpedia's dashboards use prepend_bos=True on the residual stream after the
given layer (blocks.<L>.hook_resid_post), which in HF is
output_hidden_states[L + 1].

Exits non-zero if the median correlation falls below --min_correlation, so it
can gate a pipeline run.

Example:
    python tools/verify_sae_vs_neuronpedia.py --features 12345,100,7 \
        --model_path /projects/biro/shared/models/gemma-2-9b
"""

import argparse
import json
import sys
import urllib.request

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SAE_PATH = ("/projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/"
                    "layer_20/width_131k/average_l0_114/params.npz")
DEFAULT_MODEL_PATH = "/projects/biro/shared/models/gemma-2-9b"


def np_feature(base, model_id, sae_id, feature, timeout=40):
    url = f"{base}/api/feature/{model_id}/{sae_id}/{feature}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--sae_path", default=DEFAULT_SAE_PATH)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--np_base", default="https://www.neuronpedia.org")
    ap.add_argument("--np_model", default="gemma-2-9b")
    ap.add_argument("--np_sae", default="20-gemmascope-res-131k")
    ap.add_argument("--features", default="12345,100,7,60000",
                   help="comma-separated feature indices to check")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--min_correlation", type=float, default=0.95)
    args = ap.parse_args()

    features = [int(x) for x in args.features.split(",") if x.strip()]

    npz = np.load(args.sae_path)
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    W = torch.from_numpy(npz["W_enc"]).to(device=args.device, dtype=dt)
    b = torch.from_numpy(npz["b_enc"]).to(device=args.device, dtype=dt)
    th = torch.from_numpy(npz["threshold"]).to(device=args.device, dtype=dt)
    print(f"[sae] {args.sae_path}")
    print(f"[sae] d_in={W.shape[0]} d_sae={W.shape[1]} dtype={args.dtype}")

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dt, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    print(f"[gemma] {args.model_path} | comparing against "
          f"{args.np_model}/{args.np_sae} layer {args.layer} "
          f"(hidden_states[{args.layer + 1}])")

    rows = []
    for f in features:
        try:
            data = np_feature(args.np_base, args.np_model, args.np_sae, f)
        except Exception as e:
            print(f"  feature {f}: lookup failed ({e})")
            continue
        acts = data.get("activations") or []
        if not acts:
            print(f"  feature {f}: no published activations")
            continue
        a = acts[0]
        tokens = a.get("tokens") or []
        theirs = np.asarray([float(v) for v in (a.get("values") or [])], dtype=np.float64)
        if len(tokens) != len(theirs) or not len(tokens):
            print(f"  feature {f}: malformed example")
            continue

        ids = tok.convert_tokens_to_ids(tokens)
        if any(i is None or i == tok.unk_token_id for i in ids):
            print(f"  feature {f}: could not map tokens back to ids")
            continue
        prepend = tokens[0] != tok.bos_token
        if prepend:
            ids = [tok.bos_token_id] + ids
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=args.device),
                        output_hidden_states=True, use_cache=False, return_dict=True)
            h = out.hidden_states[args.layer + 1][0]
            pre = h.to(dt) @ W + b
            z = torch.where(pre > th, pre, torch.zeros_like(pre))[:, f].float().cpu().numpy()
        mine = z[1:] if prepend else z
        mine = mine[:len(theirs)].astype(np.float64)

        corr = float(np.corrcoef(mine, theirs)[0, 1]) if mine.std() and theirs.std() else float("nan")
        peak_ok = int(np.argmax(mine)) == int(np.argmax(theirs))
        scale = mine.max() / theirs.max() if theirs.max() else float("nan")
        rows.append((f, corr, scale, peak_ok))
        print(f"  feature {f:>7}: corr={corr:.4f}  max mine/theirs={mine.max():.3f}/"
              f"{theirs.max():.3f} (ratio {scale:.3f})  peak token "
              f"{'same' if peak_ok else 'DIFFERENT'}")

    if not rows:
        print("no features could be compared")
        return 2
    corrs = np.array([r[1] for r in rows])
    med = float(np.nanmedian(corrs))
    print(f"\nmedian correlation over {len(rows)} features: {med:.4f}")
    print(f"peak token matches: {sum(r[3] for r in rows)}/{len(rows)}")
    print(f"median scale ratio: {float(np.nanmedian([r[2] for r in rows])):.3f} "
          "(1.0 = identical scale; small deviations are bf16 vs float32)")
    if med < args.min_correlation:
        print(f"FAIL: below --min_correlation {args.min_correlation}. The weights may match "
              "but the hook point, layer index or BOS handling does not.")
        return 1
    print("PASS: local SAE reproduces Neuronpedia's published activations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
