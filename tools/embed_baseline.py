#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed_baseline.py

An off-the-shelf embedding model as a baseline for symbolic matching, on exactly the
views, candidate pools and metrics of the main table. Each symbol's vector is the mean
of its (L2-normalised) view embeddings; candidates are ranked by cosine. Alongside the
plain embedding ("dense") we report the same two controls as for the LLM hidden state:
mean-centred and first-principal-component-removed vectors over the reference collection.

Default model: BAAI/bge-m3 (XLM-RoBERTa-large, CLS pooling, multilingual, 8k context).

Tasks (same inputs as the SAE-IDF runs):
  minif2f_*     Lean 4 -> Isabelle / Metamath / HOL Light statements (XLCoST-format files)
  xlcost_*      XLCoST program-level code-to-code search, official metrics (MRR, Precision@6)
  commonkg_*    NELL -> DBpedia, YAGO -> Wikidata classes, native views
  multifarm_*   zh / ru / ar -> en terms, native views
  bioml_*, anatomy  appendix tasks, native views
  valentine     whole-column layout: the pair files of the main run with the column vectors
                replaced, scored by schema/eval_valentine_sae.py

Example:
    python tools/embed_baseline.py --model_dir /projects/biro/xiaocong/models/bge-m3 \
        --output_dir /projects/biro/xiaocong/main_table/embed_baseline/bge-m3
"""
import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "xlcost"))
from eval_xlcost_sae import calculate_scores, precison_atk  # noqa: E402

M = Path("/projects/biro/xiaocong/main_table")
DATA = Path("/projects/biro/xiaocong/data")


# ----------------------------------------------------------------------------- embedding
class Embedder:
    def __init__(self, model_dir, device, max_length, batch_size):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16).to(device).eval()
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.cache = {}

    @torch.inference_mode()
    def __call__(self, texts):
        """L2-normalised CLS embeddings, cached by text."""
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        todo.sort(key=len)   # length-sorted batches waste less padding
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i:i + self.batch_size]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            out = self.model(**enc).last_hidden_state[:, 0]
            out = torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()
            for t, v in zip(chunk, out):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts])


def controls(q, c, ref):
    """dense / centred / PC1-removed variants of (query, candidate) matrices w.r.t. a reference set."""
    mu = ref.mean(axis=0)
    rc = ref - mu
    u1 = np.linalg.svd(rc, full_matrices=False)[2][0]
    out = {"dense": (q, c)}
    out["dense_centered"] = (q - mu, c - mu)
    proj = lambda x: (x - mu) - np.outer((x - mu) @ u1, u1)
    out["dense_pc1"] = (proj(q), proj(c))
    return out


def cos_matrix(q, c):
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)
    return qn @ cn.T


# ----------------------------------------------------------------------------- XLCoST-format tasks
def xlcost_task(emb, dataset_file, topk=100):
    rows = [json.loads(l) for l in open(dataset_file, encoding="utf-8") if l.strip()]
    q_text = [" ".join(r["docstring_tokens"]) for r in rows]
    c_text = [" ".join(r.get("code_tokens") or r["function_tokens"]) for r in rows]
    Q, C = emb(q_text), emb(c_text)
    answers = {r["url"]: r["idx"] for r in rows}
    res = {}
    for name, (q, c) in controls(Q, C, C).items():
        S = cos_matrix(q, c)
        preds = {}
        for i, r in enumerate(rows):
            order = np.argsort(-S[i], kind="stable")[:topk]
            preds[r["url"]] = [rows[j]["idx"] for j in order]
        res[name] = {**calculate_scores(answers, preds), **precison_atk(answers, preds, 6)}
    return {"n": len(rows), "metrics": res}


# ----------------------------------------------------------------------------- ontology / KG tasks
def load_store(store):
    """iri -> list of view texts, from contexts/{src,tgt}/*.json."""
    texts = {}
    for side in ("src", "tgt"):
        for f in sorted((store / "contexts" / side).glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            texts[d["entity_iri"]] = [c["text"] for c in d["contexts"]] or [f"Class: {d.get('preferred_label', '')}"]
    return texts


def ontology_task(emb, store, queries_file):
    texts = load_store(store)
    queries = json.loads(Path(queries_file).read_text(encoding="utf-8"))
    iris = sorted({q["src"] for q in queries} | {c for q in queries for c in q["candidates"]} | {q["gold"] for q in queries})
    iris = [i for i in iris if i in texts]
    flat = [t for i in iris for t in texts[i]]
    E = emb(flat)
    vec, k = {}, 0
    for i in iris:
        n = len(texts[i]); v = E[k:k + n].mean(axis=0); k += n
        vec[i] = v / max(np.linalg.norm(v), 1e-9)
    ref = np.stack([vec[i] for i in iris])    # all symbols of both systems, as for the LLM controls
    res = {}
    for name, (_, _) in controls(ref, ref, ref).items():
        mu = ref.mean(axis=0); u1 = np.linalg.svd(ref - mu, full_matrices=False)[2][0]
        def tf(x):
            if name == "dense":
                return x
            x = x - mu
            return x - (x @ u1) * u1 if name == "dense_pc1" else x
        ranks = []
        for q in queries:
            if q["src"] not in vec or q["gold"] not in vec:
                continue
            cands = [c for c in q["candidates"] if c in vec]
            if q["gold"] not in cands:
                cands.append(q["gold"])
            s = cos_matrix(tf(vec[q["src"]])[None, :], np.stack([tf(vec[c]) for c in cands]))[0]
            order = np.argsort(-s, kind="stable")
            ranks.append(int(np.nonzero(np.array(cands)[order] == q["gold"])[0][0]) + 1)
        ranks = np.array(ranks)
        res[name] = {"MRR": float(np.mean(1 / ranks)), "H@1": float(np.mean(ranks <= 1)), "H@5": float(np.mean(ranks <= 5)), "n": len(ranks)}
    return {"n": len(queries), "metrics": res}


# ----------------------------------------------------------------------------- Valentine
def valentine_task(emb, run_dir, out_dir, metrics="dense,dense_centered,dense_pc1"):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs").mkdir(exist_ok=True)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    cfg["model_path"] = "embedding baseline"; cfg["pooling"] = "embedding model, whole-column text"
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    for pf in sorted((run_dir / "pairs").glob("*.pt")):
        p = torch.load(pf, map_location="cpu", weights_only=False)
        cols = p["source"]["columns"] + p["target"]["columns"]
        texts = []
        for c in cols:
            vals = ast.literal_eval(c["values"]) if isinstance(c["values"], str) else list(c["values"])
            texts.append(f"{c['table']}.{c['column']}: " + "; ".join(str(v) for v in vals))
        E = emb(texts)
        for c, v in zip(cols, E):
            c["dense"] = torch.from_numpy(v.astype(np.float16))
        torch.save(p, out_dir / "pairs" / pf.name)
    subprocess.run([sys.executable, str(REPO / "schema" / "eval_valentine_sae.py"), "--run_dir", str(out_dir),
                    "--output_dir", str(out_dir / "eval"), "--metrics", metrics], check=True)
    s = json.load(open(out_dir / "eval" / "scores.json"))
    return {"n": s["pairs_scored"], "metrics": {m: s["macro_average"][m] for m in metrics.split(",")}}


TASKS = {
    "minif2f_isabelle": ("xlcost", DATA / "formal/minif2f_xsys/Isabelle/all.jsonl"),
    "minif2f_metamath": ("xlcost", DATA / "formal/minif2f_xsys/Metamath/all.jsonl"),
    "minif2f_hollight": ("xlcost", DATA / "formal/minif2f_xsys/HOLLight/all.jsonl"),
    **{f"xlcost_{n}": ("xlcost", DATA / f"XLCoST_data/retrieval/code2code_search/program_level/{L}/test.jsonl")
       for L, n in (("Java", "Java"), ("C++", "Cpp"), ("Python", "Python"), ("C#", "Csharp"), ("Javascript", "Javascript"), ("PHP", "PHP"), ("C", "C"))},
    "commonkg_nell_dbpedia": ("onto", M / "commonkg/native/nell_dbpedia", M / "commonkg/native/nell_dbpedia/selected_queries.json"),
    "commonkg_yago_wikidata": ("onto", M / "commonkg/native/yago_wikidata", M / "commonkg/native/yago_wikidata/selected_queries.json"),
    "multifarm_zh": ("onto", M / "multifarm/native", M / "multifarm/native/views/cn-en/selected_queries.json"),
    "multifarm_ru": ("onto", M / "multifarm/native", M / "multifarm/native/views/en-ru/selected_queries.json"),
    "multifarm_ar": ("onto", M / "multifarm/native", M / "multifarm/native/views/ar-en/selected_queries.json"),
    "bioml_valid": ("onto", M / "bioml/native", M / "bioml/native/views/valid/selected_queries.json"),
    "bioml_train": ("onto", M / "bioml/native", M / "bioml/native/views/train/selected_queries.json"),
    "anatomy": ("onto", M / "anatomy/native", M / "anatomy/native/selected_queries.json"),
    "valentine": ("valentine", M / "valentine_column"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--tasks", default="all", help="comma-separated subset of task names, or all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb = Embedder(args.model_dir, args.device, args.max_length, args.batch_size)
    names = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    for name in names:
        out = args.output_dir / f"{name}.json"
        if out.exists():
            print(f"[{name}] exists, skipping"); continue
        t0 = time.time(); spec = TASKS[name]
        if spec[0] == "xlcost":
            res = xlcost_task(emb, spec[1])
        elif spec[0] == "onto":
            res = ontology_task(emb, spec[1], spec[2])
        else:
            res = valentine_task(emb, spec[1], args.output_dir / "valentine_run")
        res.update(task=name, model=str(args.model_dir), max_length=args.max_length)
        out.write_text(json.dumps(res, indent=2))
        print(f"[{name}] n={res['n']} " + " ".join(f"{m}={v.get('MRR', v.get('MeanReciprocalRank', 0)):.4f}" for m, v in res["metrics"].items()) + f" ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
