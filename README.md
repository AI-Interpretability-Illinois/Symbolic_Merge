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
│   └── analyze_bioml_information_weighted_sae_target_bg.py
│
├── kg/
│   └── prepare_oaei_starwars_cached_transformers.py
│
├── run_bioml_sae_v3.py
├── score_cached_subset.py
└── README.md