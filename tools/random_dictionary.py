#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_dictionary.py

A random sparse dictionary with the width and sparsity of a Gemma Scope SAE, written in Gemma
Scope's params.npz layout (W_enc, b_enc, threshold) so that every runner in this repository
loads it unchanged through --sae_path. It is the control for the claim that the *learned* SAE
features matter: the same token activations, the same width, the same number of active features
per token, the same idf weighting, with a random encoder in place of the trained one.

Encoder W in R^{d x F}: i.i.d. N(0, 1) entries, columns normalised to unit length; b_enc = 0;
one global JumpReLU threshold chosen so that the mean number of active features per token on a
fixed calibration sample of symbol tokens equals that of the real SAE on the same tokens.

Calibration tokens come from hidden-state caches written by bioml/cache_bioml_hidden.py
(shard-*.pt: {"layers": [...], "entities": [{"contexts": [{"hidden": [n_layers, n_tokens, d]}]}]}).

Example:
    python tools/random_dictionary.py --out /projects/biro/xiaocong/sae/random/random_w131k_seed0.npz \
        --real_sae /projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/layer_20/width_131k/average_l0_114/params.npz \
        --calib_caches <store1>/hidden_sweep,<store2>/hidden_sweep --layer 20 --seed 0 --device cuda
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def sample_tokens(cache_dirs, layer, n_tokens, seed):
    """A seeded sample of symbol-token hidden states at one layer, from the caches' contexts."""
    rng = np.random.default_rng(seed)
    rows = []
    for cd in cache_dirs:
        for shard in sorted(Path(cd).glob("shard-*.pt")):
            d = torch.load(shard, map_location="cpu", weights_only=False)
            li = d["layers"].index(layer)
            for ent in d["entities"]:
                for c in ent["contexts"]:
                    rows.append(c["hidden"][li].float())
    H = torch.cat(rows)
    idx = np.sort(rng.choice(H.shape[0], size=min(n_tokens, H.shape[0]), replace=False))
    return H[torch.from_numpy(idx)]


@torch.inference_mode()
def active_counts(H, W, b, th, device, batch=512):
    """Number of active features per token for a JumpReLU encoder (pre > threshold)."""
    out = []
    for i in range(0, H.shape[0], batch):
        pre = H[i:i + batch].to(device, torch.bfloat16) @ W + b
        out.append((pre > th).sum(-1).float().cpu())
    return torch.cat(out)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--real_sae", required=True, help="params.npz of the SAE whose width and L0 are matched")
    ap.add_argument("--calib_caches", required=True, help="comma-separated hidden-state cache dirs")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_tokens", type=int, default=4000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()

    real = np.load(args.real_sae)
    d, F = real["W_enc"].shape
    H = sample_tokens(args.calib_caches.split(","), args.layer, args.n_tokens, args.seed)
    print(f"calibration tokens: {tuple(H.shape)} from {args.calib_caches} ({time.time() - t0:.0f}s)", flush=True)

    # the real SAE's sparsity on these tokens
    Wr = torch.from_numpy(real["W_enc"]).to(args.device, torch.bfloat16)
    br = torch.from_numpy(real["b_enc"]).to(args.device, torch.bfloat16)
    thr = torch.from_numpy(real["threshold"]).to(args.device, torch.bfloat16)
    l0_real = active_counts(H, Wr, br, thr, args.device)
    del Wr, br, thr
    target = float(l0_real.mean())
    print(f"real SAE: mean L0 {target:.1f} (median {float(l0_real.median()):.0f}) on the sample", flush=True)

    # the random dictionary
    g = torch.Generator(device=args.device).manual_seed(args.seed)
    W = torch.randn(d, F, generator=g, device=args.device, dtype=torch.float32)
    W /= W.norm(dim=0, keepdim=True)
    Wb = W.to(torch.bfloat16)
    pre = torch.cat([(H[i:i + 512].to(args.device, torch.bfloat16) @ Wb) for i in range(0, H.shape[0], 512)])  # [n, F] bf16
    lo, hi = 0.0, float(pre.float().max())
    for _ in range(60):                       # bisection on the global threshold: mean active count = target
        mid = (lo + hi) / 2
        if float((pre > mid).sum(-1).float().mean()) > target:
            lo = mid
        else:
            hi = mid
    theta = (lo + hi) / 2
    active = pre > theta
    l0_rand = active.sum(-1).float().cpu()
    density = active.float().mean(0).cpu()          # per-feature firing rate on the sample
    stats = {
        "seed": args.seed, "d_model": int(d), "width": int(F), "layer": args.layer, "threshold": theta,
        "n_calibration_tokens": int(H.shape[0]), "calibration_caches": args.calib_caches.split(","),
        "real_sae": args.real_sae,
        "l0_real": {"mean": target, "median": float(l0_real.median()), "p10": float(l0_real.quantile(0.1)), "p90": float(l0_real.quantile(0.9))},
        "l0_random": {"mean": float(l0_rand.mean()), "median": float(l0_rand.median()), "p10": float(l0_rand.quantile(0.1)), "p90": float(l0_rand.quantile(0.9))},
        "random_features_never_active_on_sample": float((density == 0).float().mean()),
        "random_feature_density_max": float(density.max()),
        "random_features_density_gt_1pct": int((density > 0.01).sum()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, W_enc=W.to(torch.float16).cpu().numpy(), b_enc=np.zeros(F, np.float32),
             threshold=np.full(F, theta, np.float32))
    args.out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
