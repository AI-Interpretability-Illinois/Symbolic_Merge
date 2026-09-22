# Symbolic_Merge

This repository investigates whether sparse autoencoder (SAE) features can serve as semantic signatures for symbols and support alignment across heterogeneous symbolic systems.

The central idea is to represent each symbol using neural activations obtained from multiple diverse contexts. Rather than treating every SAE feature equally, we identify features that are conserved across contexts for the same entity, enriched relative to a background population, and reliable in their activation strength.

The repository currently contains experiments on biomedical ontology alignment using OAEI Bio-ML and a generalization pipeline for knowledge-graph alignment.

---

## Repository Structure

```text
Symbolic_Merge/
├── bioml/
│   ├── generate_bioml_contexts_bio8b_maxdiverse.py
│   ├── run_bioml_sae_from_bio8b_contexts.py
│   ├── analyze_bioml_information_weighted_sae.py
│   ├── analyze_bioml_information_weighted_sae_target_bg.py
│   └── autointerp_bioml_neuronpedia.py
│
├── kg/
│   └── prepare_oaei_starwars_cached_transformers.py
│
├── xlcost/
│   ├── run_xlcost_sae.py
│   └── eval_xlcost_sae.py
│
├── run_bioml_sae_v3.py
├── score_cached_subset.py
└── README.md
```

---

## Auto-interpretability (Bio-ML)

`bioml/autointerp_bioml_neuronpedia.py` names the features our information weighting
relies on. It does not explain all 131,072 SAE features: it first keeps only features
with a real activation pattern (conserved across an entity's contexts, enriched over
the target background, stable in magnitude), plus the features that produce each
src/candidate cosine, and explains only those — typically a few hundred.

Explanations come from Neuronpedia (`hijohnnylin/neuronpedia`), which already hosts
auto-interp explanations for every Gemma Scope feature, so the default path makes no
LLM calls. `--llm_explain_missing` auto-interps the leftovers from Neuronpedia's own
top-activating examples.

Feature indices only line up with Neuronpedia's `gemma-2-9b/20-gemmascope-res-131k`
if the run used `google/gemma-scope-9b-pt-res : layer_20/width_131k/average_l0_114`.
The script checks this against `results.json` and refuses to run on a mismatch.

```bash
python bioml/autointerp_bioml_neuronpedia.py \
    --sae_run_dirs     <bioml sae run dir> \
    --context_run_dirs <bioml context run dir> \
    --output_dir       <autointerp out>
```

Outputs: `candidate_features.csv` (feature, explanation, our statistics, dashboard
link), `per_entity_features.csv`, `pair_explanations.md` (why each pair matched, and
for failures what the wrong top-1 shared), `pair_drivers.json`, and
`autointerp_summary.json`.

---

## XLCoST XL code search (code-to-code)

Third benchmark for the same representation idea: retrieve the implementation of the
same program in other languages. Get the dataset from the
[XLCoST repo](https://github.com/reddy-lab-code-research/XLCoST) and keep its layout,
`<code2codesearch>/dataset/{snippet_level,program_level}/<Lang>/<split>.jsonl`.
Field names are read exactly as XLCoST's own `run.py` reads them: `docstring_tokens`
is the query side (the source-language code), `code_tokens`/`function_tokens` the
candidate side, with `url` and `idx` as identifiers.

Step 1 embeds the whole snippet or program — the retrieval level is just which folder
you point at. Every token is encoded by the SAE and then pooled, never the reverse:
the JumpReLU threshold applies per token, so pooling first drops features that fire
on only part of the code. Tokens act as the views of a document, so the Bio-ML
information weighting carries over unchanged.

```bash
python xlcost/run_xlcost_sae.py \
    --data_dir <code2codesearch>/dataset/program_level \
    --lang Java --split test \
    --model_path ~/models/gemma-2-9b \
    --sae_path <gemma-scope>/layer_20/width_131k/average_l0_114/params.npz \
    --output_dir <out>/xlcost_java_program

python xlcost/eval_xlcost_sae.py \
    --run_dir <out>/xlcost_java_program \
    --official_evaluator <XLCoST>/code/codesearch/code2codesearch/evaluator/evaluator.py
```

Step 2 follows the official retrieval protocol: candidates are all rows of the same
file, the pool includes the query's own row (its match in the other language, which
the official evaluator counts as relevant, so the first hit is free), predictions keep
the top 100, and relevance is a prefix match on `idx.split("/")[0]`. It writes
`predictions_<metric>.jsonl` in the official format and scores them with XLCoST's own
MRR and Precision@6 definitions. `--official_evaluator` additionally runs their
script as a cross-check; the two agree exactly. `--exclude_self` reports the harder
variant without the free hit.