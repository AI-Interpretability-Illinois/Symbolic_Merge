# Pre-registered analysis plan for the extra experiments (S1–S5)

Written 2026-10-04 00:55 CDT, before any of these runs started. Owner-approved list:
`/u/xiaocong/gpu_coord/naacl_extras_plan_2026-10-04.md`, section "symbolic-merge". All of
these are post-hoc additions to the main results and will be marked as such in the paper.

## S1 — random sparse dictionary + idf (decides the title claim)

**Setup.** A random dictionary with the width of the Gemma Scope SAE used in the main table
(F = 131,072) applied to the same layer-20 token activations of Gemma-2-9B: encoder
W ∈ R^{3584×131072} with i.i.d. N(0, 1) entries, columns normalised to unit length, b = 0, one
global JumpReLU threshold θ chosen so that the mean number of active features per token on a
fixed calibration sample of symbol tokens equals that of the real SAE on the same tokens
(nominal L0 114). Everything downstream is identical: native views, per-token encode then
mean, idf over the same reference collection, cosine. Seed 0 on all 16 main-table rows;
seeds 1 and 2 on the three miniF2F rows and the two Common-KG rows if time allows.

**Comparisons.** random+idf vs SAE-IDF; random+idf vs Dense; random (unweighted) vs SAE
(unweighted). Paired bootstrap over queries, 10,000 resamples, one-sided, as in Appendix K.
"Significantly ahead" = p < 0.01 and a difference of at least 0.01.

**Decision rules.**
- (a) SAE-IDF significantly ahead of random+idf on ≥ 12 of 16 rows → the learned dictionary
  matters beyond sparsity + rarity weighting. Title and abstract stand; random+idf is added
  as a baseline column and the gap SAE-IDF − random+idf is reported as the value of the
  learned features.
- (b) random+idf within noise of SAE-IDF, or ahead, on ≥ 8 of 16 rows → the contribution is
  rarity weighting of a *sparse code*, not of SAE features in particular. Reframe: title
  becomes about sparse rarity-weighted codes as the shared coordinate system; the abstract
  states that a random dictionary of the same width and sparsity matches the SAE on k of 16
  rows, and the SAE's value is that its axes are named (Section 6). The interpretability
  claims are kept; the "learned features" wording is removed.
- (c) 9–11 significant wins → report row by row; the claim is restricted to the kinds of
  system where the learned dictionary wins; the title is kept only if the formal-mathematics
  and code rows (where the gain over Dense is largest) are among the wins, otherwise (b).
- In every case: if random+idf beats Dense on most rows, Section 5's "the sparse basis alone
  does not help" is rewritten — sparsification plus idf of random projections already helps,
  and the SAE's additional contribution is quantified as SAE-IDF − random+idf.

## S2 — stronger dense controls (CPU, on the stored dense vectors)

**Controls**, each fitted on the same reference collection D that the idf uses:
1. top-k principal-component removal with k = 36 (≈ d/100 for d = 3,584), after centring;
2. per-dimension standardisation (z-score over D);
3. whitening: ZCA whitening with Ledoit–Wolf shrinkage of the covariance of D;
4. dense-idf: per symbol, mark the 114 largest |standardised coordinates| as active, compute
   idf over D from those activity sets, and score cos(x_std ⊙ idf, y_std ⊙ idf).

**Decision rule.** If any single control is within noise of SAE-IDF on ≥ 8 of 16 rows, the
sentence "SAE-IDF's advantage does not reduce to anisotropy removal" (Section 4) is dropped
and that control replaces Dense_−PC1 as the strongest dense baseline throughout. Otherwise
the controls go into the appendix with the best of them added to Table 1's comparison
counts. All results are reported either way.

## S3 — random-sample failure decompositions and a human check of feature names

A seeded (seed 0) uniform random sample of 5 failures (gold not ranked first by SAE-IDF) per
main-table task, decomposed exactly like the wins in Appendix I (features shared with the
wrong top-1 candidate and with the gold). Every distinct feature that appears in a top-5
contribution goes into one sheet with its Neuronpedia explanation and the two texts it fired
on; the owner marks each name as accurate / partly / wrong (≈ 2 h). The fraction judged
accurate is reported in Section 6; if it is below 80%, the explanation-type analysis is
re-weighted to the features judged accurate, and the paper says so.

## S4 — a 7–8B embedder (if time)

Same protocol as the BGE-M3 baseline (Appendix E), same controls. If it beats SAE-IDF on
most rows, the abstract's retriever sentence names it as the strongest trained retriever and
the counts are updated; the training-free claim is unchanged.

## S5 — idf from a held-out same-system library (if time)

idf for the miniF2F rows computed from a library of the target system (set.mm theorem
statements for Metamath; Mathlib declarations for Lean 4) instead of the test candidate
pool. If MRR drops by more than 0.05 on any row, the paper states that the idf should be
computed on the matched pool and lists this as a limitation; if not, it reports that a
library idf works, which removes the dependence on seeing the candidate pool.
