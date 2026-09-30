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
---

## miniF2F cross-prover statement matching (formal mathematics)

Fourth benchmark: the same competition problem formalised independently in Lean 4,
Isabelle, Metamath and HOL Light. Given a problem's Lean 4 statement, retrieve the
statement of the same problem in another prover. `formal/prepare_minif2f.py` keeps
only the statement and strips everything that identifies the problem without
reading the mathematics (theorem names, proofs, comments, Metamath labels). It
refuses to write a dataset if a problem id survives in any statement. The output is
in XLCoST format, so the XLCoST runner and evaluator are reused as is (`--dataset_file`).

```bash
python formal/prepare_minif2f.py --minif2f <openai/miniF2F> \
    --minif2f_lean4 <yangky11/miniF2F-lean4> --output_dir <minif2f_xsys>
python xlcost/run_xlcost_sae.py --dataset_file <minif2f_xsys>/Metamath/all.jsonl \
    --output_dir <out>/lean4_Metamath
python xlcost/eval_xlcost_sae.py --run_dir <out>/lean4_Metamath \
    --output_dir <out>/lean4_Metamath/eval
python tools/lexical_baseline.py --dataset_file <minif2f_xsys>/Metamath/all.jsonl
```

`tools/lexical_baseline.py` gives the surface-string controls (TF-IDF over words
and character n-grams, BM25). sae_idf is TF-IDF over SAE features, so TF-IDF over the
strings is the baseline that separates the features from the weighting.

---

## Bio-ML SAE ablation (layer, width, L0, SAE family)

`bioml/cache_bioml_hidden.py` runs the language model once and stores the
label-span residual stream at several layers. `bioml/encode_bioml_cached.py` then
turns that cache into a standard Bio-ML run directory for any (layer, SAE) pair
without the model: Gemma Scope `params.npz`, or Llama Scope `final.safetensors`.
Its output is bit-identical to `run_bioml_sae_from_bio8b_contexts.py` for the same
SAE, and `analyze_bioml_idf.py` reads it unchanged. Pass
`--universal_density none` to the analyzer for any SAE other than layer-20
`average_l0_114`, since the Pile density file indexes that dictionary only.

```bash
python bioml/cache_bioml_hidden.py --context_run_dirs <slices> --output_dir <cache> \
    --layers 5,9,10,15,20,25,30,31,35,40 --num_workers 4 --worker 0 --device cuda:0
python bioml/encode_bioml_cached.py --cache_dir <cache> --layer 20 \
    --sae_path <gemma-scope>/layer_20/width_16k/average_l0_68/params.npz --output_dir <run>
python bioml/analyze_bioml_idf.py --sae_run_dirs <run> --context_run_dirs <slices> \
    --universal_density none --output_dir <run>/analysis
```

---

## Native-context matching (main method)

Every task takes each symbol's contexts from its own system's data; nothing is
generated. Programs, formal statements and table cells are used directly (XLCoST,
miniF2F, Valentine). For KG classes and ontology terms,
`native/build_native_views.py` builds one view per piece of system-internal
evidence, evidence first and symbol last (the model reads left to right, so the
symbol's tokens then carry the evidence):

    Instance: Edward Thomas (poet). Class: Writer       # Common-KG: the class's instances
    domain 作者. Term: 撰写论文                           # MultiFarm: the term's own axioms
    Parent: Intestinal Atresia. Class: Duodenal Atresia  # Bio-ML / Anatomy: definitions, parents

Synonyms are never used (across systems a synonym is often the other side's label).
The views are scored with the same cache / encode / analyze pipeline:

```bash
python native/build_native_views.py commonkg --data_dir <commonkg> --output_dir <views>
python bioml/cache_bioml_hidden.py --context_run_dirs <views>/nell_dbpedia --output_dir <hidden> \
    --layers 20 --contexts 0 --label_last
python bioml/encode_bioml_cached.py --cache_dir <hidden> --layer 20 --sae_path <sae> --output_dir <run>
python bioml/analyze_bioml_idf.py --sae_run_dirs <run> --context_run_dirs <views>/nell_dbpedia \
    --background all --fast --lexical --output_dir <run>/analysis
```

`--contexts 0` keeps every view (their number varies per symbol) and `--label_last`
pools the symbol's final occurrence. `tools/lexical_baseline.py --xlcost_official`
scores string baselines with XLCoST's own MRR and Precision@6.

LLM-generated contexts (`bioml/generate_bioml_contexts_claude.py`, with Claude,
GPT-5.6 via `--backend openai`, or a local model via `--backend local`, one side
of each pair per run via `--sides`) are kept for the secondary analysis: with
them, dense vectors with their first principal component removed match sae_idf.
