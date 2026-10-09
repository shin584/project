# SaCas9 XAI Comparison Pipeline

Deep-analysis and explainability comparison between the previous and current SaCas9 cleavage-efficiency prediction models, validating that Model B's metric gains reflect real biological mechanism learning rather than noise.

## Language

**Model A**:
The previous SaCas9 model: One-hot encoded 36bp sequence → CNN+RNN backbone → cleavage efficiency score. Implemented by `Model_A_Predictor` in `model_a_wrapper.py`, weights in `best_model_fold1.pth`.
_Avoid_: "old model", "baseline model"

**Model B**:
The current SaCas9 model: a hybrid of NT sequence embeddings + 4 physical/thermodynamic features fed into an XGBoost regressor. Implemented by `Model_B_Predictor` in `model_b_wrapper.py`. Total feature width is 8,964 (8,960 flattened NT embedding + 4 physical features).
_Avoid_: "new model", "current model" (ambiguous once ablation variants exist — see Model B-Embedding / Model B-Physical)

**NT regression checkpoint**:
The `NT_sacas9_fintuned_model` checkpoint: an `EsmForSequenceClassification` (`problem_type: regression`, `num_labels=1`) — a Nucleotide Transformer backbone with a single-output linear regression head, trained end-to-end (see `nt_pretrain.py`). This checkpoint serves two distinct purposes:
- As a **frozen embedding extractor** (only its encoder hidden states are read) — this is what feeds Model B's XGBoost.
- As a **standalone differentiable regressor** (its own regression head, used directly) — this is Integrated Gradients' target for Model B's sequence-level analysis (Step 2), and the source of "NT-only regression head performance."
_Avoid_: "the NT model" alone (ambiguous — specify which of the two roles)

**NT-only regression head performance**:
The native evaluation metrics (Spearman/Pearson/MAE/MSE) produced by the NT regression checkpoint predicting scores directly, with no XGBoost involved — computed in `nt_pretrain.py`'s own Phase 4 evaluation.
_Avoid_: conflating with "Model B-Embedding ablation" — methodologically different pipeline, kept as a separate number.

**Model B-Embedding / Model B-Physical / Model B-Full (ablation variants)**:
Three XGBoost regressors from Step 4①, each independently retrained on a feature subset of Model B's full feature set: embedding only (8,960d), physical only (4d), or full (8,964d). Used to isolate how much the physical features contribute via Paired Bootstrap comparison.
_Avoid_: "embedding_only" as a synonym for "NT-only regression head performance" — they are different pipelines producing different numbers.

**Testset**:
The fixed, isolated evaluation set: 514 SaCas9 sequences (10% of the 5,145-row dataset), carved out via a single `seed=42` random permutation (`torch.randperm`) in `test_dataset_verifier.py` — not K-fold cross-validation. `sample_id` ranges 0–513. Stored in `test_metadata.csv` with SHA256 checksums (`sequence_hash` per row, plus a whole-column dataset hash) to guard against data leakage.
_Avoid_: "515 samples" (a corrected documentation error — the real count is 514)

**PAM window**:
The 6bp window at 0-indexed positions 25-30 of the 36bp sequence (`PAM_REGION_START/END` in `mismatch_profiling.py`). In this dataset it follows `NNGRRN`: the canonical SaCas9 `NNGRRT` with position 30 not fixed — every Testset row has `NNGRR` at 25-29, but only 54 of 514 have `T` at 30 (issue #31). Typed-sequence validation therefore requires G at 27 and A/G at 28-29 only.
_Avoid_: "NNGRRT" as the description of this dataset's PAM window — requiring `T` at position 30 rejects about 90% of real Testset sequences.

**sample_id / original_id**:
`sample_id` is the Testset-internal fixed index (0–513, assigned by row order after isolation). `original_id` is the row's index in the full, pre-split SaCas9 dataset. Kept distinct to preserve traceability back to the source data without letting internal indices leak into cross-dataset joins.

**Concordant / Discordant case**:
Case Study classifications based on per-model absolute error quantiles on the Testset (`e_A`, `e_B`).
- **Concordant**: both models predict well (`e_A ≤ Q25(e_A)` and `e_B ≤ Q25(e_B)`).
- **Primary Discordant**: Model A predicts poorly, Model B predicts well (`e_A ≥ Q75(e_A)` and `e_B ≤ Q25(e_B)`) — the core mechanism-improvement signal.
- **Secondary Discordant**: a relaxed fallback used only to fill out Primary Discordant's top-5 when fewer than 5 samples qualify (`(e_A - e_B)/e_A ≥ 0.5`, with `e_A ≥ Median(e_A)`).
- **Reverse Primary Discordant** (issue #18): the mirror image of Primary Discordant — Model B regresses relative to Model A (`e_B ≥ Q75(e_B)` and `e_A ≤ Q25(e_A)`). Drawn from its own separate 5-slot budget, not a subset of the Primary/Secondary Discordant budget above, since the reverse population on the live Testset (11 samples) is comparably sized to the forward population and would starve one direction if the two shared a single 5-slot budget.
- **Reverse Secondary Discordant** (issue #18): a relaxed fallback used only to fill out Reverse Primary Discordant's top-5 when fewer than 5 samples qualify (`(e_B - e_A)/e_B ≥ 0.5`, with `e_B ≥ Median(e_B)`) — same relative-gap convention as Secondary Discordant, mirrored.

Case Study export is capped at 15 total (5 Primary/Secondary Discordant + 5 Reverse Primary/Secondary Discordant + 5 Concordant); `case_id` values follow `DISCORDANT_P0x`/`DISCORDANT_S0x` (forward), `DISCORDANT_R0x`/`DISCORDANT_RS0x` (reverse), `CONCORDANT_C0x`.

**Deterministic Projection Rule**:
The analysis assumption that a 6-mer token's Integrated Gradients attribution is divided evenly across the 6 nucleotides it spans, when mapping token-level attribution back to the 36bp sequence. Explicitly documented as an assumption, not evidence that the model perceives individual nucleotides independently.

**Information Dependency** (Attention Rollout result):
The recursively-rolled-out attention connectivity between input token pairs, computed as `0.5·A + 0.5·I` (applied before row-normalization) across NT's transformer layers with the `[CLS]` token removed and coordinates mapped back to nucleotide positions. Explicitly scoped as describing internal information dependency within the model, not a claim about physical DNA/protein binding.
