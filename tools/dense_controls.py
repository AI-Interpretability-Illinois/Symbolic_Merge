#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dense_controls.py

Stronger controls on the stored dense hidden-state vectors of the main-table runs, all fitted on
the same reference collection D that SAE-IDF's idf uses (the candidate pool for miniF2F, XLCoST
and Valentine; all symbols of both systems for the ontology and knowledge-graph tasks):

  dense          the pooled hidden state, as in the paper
  centered       x - mu
  pc1            centred, first principal direction removed (the paper's Dense_-PC1; a check)
  pc36           centred, top-36 principal directions removed (36 = d / 100 for d = 3,584)
  zscore         per-dimension standardisation over D
  zca            ZCA whitening with Ledoit-Wolf shrinkage of D's covariance
  dense_idf      z-scored, every dimension weighted by an idf computed from "activity sets":
                 per symbol the 114 largest |z| dimensions count as active (114 = the SAE's L0)
  dense_topk_idf the same, but only the 114 active dimensions are kept (a sparse code of width d)

Every similarity is a cosine. Metrics are the benchmarks' own: official MRR / Precision@6 for the
XLCoST-format tasks (miniF2F, XLCoST), Valentine F1 and MRR through schema/eval_valentine_sae.py,
MRR over the candidate lists for the ontology tasks. CPU only; run as a batch job, not on a login node.

Example:
    python tools/dense_controls.py --root /srv/local/xy51/symbolic_merge/main_table_delta \
        --output_dir /srv/local/xy51/symbolic_merge/main_table_delta/dense_controls
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for sub in ("xlcost", "bioml", "schema"):
    sys.path.insert(0, str(REPO / sub))
from eval_xlcost_sae import calculate_scores, precison_atk, load_side  # noqa: E402

L0 = 114
K_PCS = 36
CONTROLS = ["dense", "centered", "pc1", "pc36", "zscore", "zca", "dense_idf", "dense_topk_idf"]


# ----------------------------------------------------------------------------- the transforms
class Controls:
    """Fit on the reference collection D (rows = symbols), then transform any matrix."""

    def __init__(self, D, k_pcs=K_PCS, l0=L0):
        D = np.asarray(D, np.float64)
        n, d = D.shape
        self.mu = D.mean(0)
        R = D - self.mu
        self.sd = R.std(0) + 1e-6
        k = min(k_pcs, max(1, n - 2))
        self.V = np.linalg.svd(R, full_matrices=False)[2][:k]          # top principal directions
        # Ledoit-Wolf shrinkage towards a scaled identity, then the inverse square root
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf(assume_centered=True).fit(R)
        evals, evecs = np.linalg.eigh(lw.covariance_)
        self.W_zca = (evecs / np.sqrt(np.maximum(evals, 1e-8))) @ evecs.T
        # activity sets on the standardised vectors -> document frequency -> idf
        Z = R / self.sd
        active = np.zeros_like(Z, dtype=bool)
        top = np.argpartition(-np.abs(Z), l0 - 1, axis=1)[:, :l0]
        np.put_along_axis(active, top, True, axis=1)
        df = active.sum(0)
        self.idf = np.maximum(np.log(n / (1.0 + df)), 0.0)
        self.l0, self.k = l0, k

    def transform(self, X, name):
        X = np.asarray(X, np.float64)
        if name == "dense":
            return X
        R = X - self.mu
        if name == "centered":
            return R
        if name == "pc1":
            return R - (R @ self.V[:1].T) @ self.V[:1]
        if name == "pc36":
            return R - (R @ self.V.T) @ self.V
        Z = R / self.sd
        if name == "zscore":
            return Z
        if name == "zca":
            return R @ self.W_zca.T
        if name == "dense_idf":
            return Z * self.idf
        if name == "dense_topk_idf":
            keep = np.zeros_like(Z, dtype=bool)
            top = np.argpartition(-np.abs(Z), self.l0 - 1, axis=1)[:, :self.l0]
            np.put_along_axis(keep, top, True, axis=1)
            return np.where(keep, Z * self.idf, 0.0)
        raise KeyError(name)


def cos_matrix(q, c):
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)
    return qn @ cn.T


def dense_of(docs):
    return np.stack([d["dense"].float().numpy() for d in docs]).astype(np.float64)


# ----------------------------------------------------------------------------- tasks
def xlcost_task(run, topk=100):
    q, c = load_side(run, "query"), load_side(run, "candidate")
    Q, C = dense_of(q), dense_of(c)
    ctl = Controls(C)
    answers = {d["url"]: d["idx"] for d in q}
    res = {}
    for name in CONTROLS:
        S = cos_matrix(ctl.transform(Q, name), ctl.transform(C, name))
        preds = {}
        for i, d in enumerate(q):
            order = np.argsort(-S[i], kind="stable")[:topk]
            preds[d["url"]] = [c[j]["idx"] for j in order]
        res[name] = {**calculate_scores(answers, preds), **precison_atk(answers, preds, 6)}
        # first-hit rank of the gold idx per query, for paired bootstraps against the other methods
        res[name]["per_query_rank"] = {u: (p.index(answers[u]) + 1 if answers[u] in p else None) for u, p in preds.items()}
    return {"n": len(q), "k_pcs": ctl.k, "metrics": res}


def ontology_task(sae_run, ctx):
    import analyze_bioml_idf as A
    src, tgt = A.load_rep_dirs([sae_run])
    qs = A.load_queries([ctx])
    vec = {k: r["aggregate"]["dense_mean"].float().numpy().astype(np.float64) for k, r in list(src.items()) + list(tgt.items())}
    iris = sorted(vec)
    ctl = Controls(np.stack([vec[i] for i in iris]))           # all symbols of both systems
    res = {}
    for name in CONTROLS:
        T = dict(zip(iris, ctl.transform(np.stack([vec[i] for i in iris]), name)))
        ranks = []
        for q in qs:
            if q["src"] not in T or q["gold"] not in T:
                continue
            cands = [c for c in q["candidates"] if c in T]
            if q["gold"] not in cands:
                cands.append(q["gold"])
            s = cos_matrix(T[q["src"]][None, :], np.stack([T[c] for c in cands]))[0]
            order = np.argsort(-s, kind="stable")
            ranks.append(int(np.nonzero(np.array(cands)[order] == q["gold"])[0][0]) + 1)
        ranks = np.array(ranks)
        res[name] = {"MRR": float(np.mean(1 / ranks)), "H@1": float(np.mean(ranks <= 1)), "H@5": float(np.mean(ranks <= 5)), "n": int(len(ranks)),
                     "per_query_rank": [int(r) for r in ranks]}
    return {"n": len(qs), "k_pcs": ctl.k, "metrics": res}


def valentine_task(run_dir, out_root, scratch=None):
    """Fit on all target columns of all pairs, rewrite each pair's column vectors, score with the evaluator."""
    pairs = sorted((run_dir / "pairs").glob("*.pt"))
    loaded = [torch.load(pf, map_location="cpu", weights_only=False) for pf in pairs]
    D = np.stack([c["dense"].float().numpy() for p in loaded for c in p["target"]["columns"]]).astype(np.float64)
    ctl = Controls(D)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    res = {}
    for name in CONTROLS:
        out = (Path(scratch) if scratch else out_root) / f"valentine_{name}"     # rewritten pair files are large; keep them off quota'd stores
        if not (out / "eval" / "scores.json").exists():
            (out / "pairs").mkdir(parents=True, exist_ok=True)
            cfg["dense_control"] = name
            (out / "run_config.json").write_text(json.dumps(cfg, indent=2))
            for pf, p in zip(pairs, loaded):
                cols = p["source"]["columns"] + p["target"]["columns"]
                X = ctl.transform(np.stack([c["dense"].float().numpy() for c in cols]), name)
                for c, v in zip(cols, X):
                    c["dense"] = torch.from_numpy(v.astype(np.float32))
                torch.save(p, out / "pairs" / pf.name)
            subprocess.run([sys.executable, str(REPO / "schema" / "eval_valentine_sae.py"), "--run_dir", str(out),
                            "--output_dir", str(out / "eval"), "--metrics", "dense"], check=True)
        s = json.load(open(out / "eval" / "scores.json"))
        keep = out_root / f"valentine_{name}_eval"; keep.mkdir(parents=True, exist_ok=True)
        (keep / "scores.json").write_text(json.dumps(s, indent=1))
        res[name] = {"F1Score": s["macro_average"]["dense"]["F1Score"], "MeanReciprocalRank": s["macro_average"]["dense"]["MeanReciprocalRank"]}
    return {"n": len(pairs), "k_pcs": ctl.k, "metrics": res}


def tasks(root, alt_root=None):
    M = Path(root)

    def with_reps(rel):   # the Delta store keeps representations only for some XLCoST runs; fall back to another store
        for base in (M, Path(alt_root) if alt_root else None):
            if base is not None and all((base / rel / "representations" / side).is_dir() for side in ("query", "candidate")):
                return base / rel
        return M / rel

    T = {f"minif2f_{t.lower()}": ("xlcost", with_reps(f"minif2f/lean4_{t}")) for t in ("Isabelle", "Metamath", "HOLLight")}
    T.update({f"xlcost_{L}": ("xlcost", with_reps(f"xlcost/{L}_program")) for L in ("C", "PHP", "Java", "Cpp", "Python", "Csharp", "Javascript")})
    T["commonkg_nell_dbpedia"] = ("onto", M / "commonkg/native_scored_nell_dbpedia/L20_w131k_l0_114", M / "commonkg/native/nell_dbpedia")
    T["commonkg_yago_wikidata"] = ("onto", M / "commonkg/native_scored_yago_wikidata/L20_w131k_l0_114", M / "commonkg/native/yago_wikidata")
    for n, v in (("zh", "cn-en"), ("ru", "en-ru"), ("ar", "ar-en")):
        T[f"multifarm_{n}"] = ("onto", M / "multifarm/native_scored/L20_w131k_l0_114", M / f"multifarm/native/views/{v}")
    T["valentine"] = ("valentine", M / "valentine_column")
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="main_table root holding the stored runs")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--alt_root", type=Path, default=None, help="second store to look in for runs that keep representations")
    ap.add_argument("--scratch", type=Path, default=None, help="where the rewritten Valentine pair files go (default: output_dir)")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    T = tasks(args.root, args.alt_root)
    names = list(T) if args.tasks == "all" else args.tasks.split(",")
    for name in names:
        out = args.output_dir / f"{name}.json"
        if out.exists():
            print(f"[{name}] exists, skipping"); continue
        t0 = time.time(); spec = T[name]
        if spec[0] == "xlcost":
            res = xlcost_task(spec[1])
        elif spec[0] == "onto":
            res = ontology_task(spec[1], spec[2])
        else:
            res = valentine_task(spec[1], args.output_dir, args.scratch)
        res.update(task=name, controls=CONTROLS, l0=L0, k_pcs_requested=K_PCS)
        out.write_text(json.dumps(res, indent=2))
        key = next(k for k in ("MRR", "MeanReciprocalRank") if k in res["metrics"]["dense"])
        print(f"[{name}] n={res['n']} " + " ".join(f"{m}={v[key]:.4f}" for m, v in res["metrics"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
