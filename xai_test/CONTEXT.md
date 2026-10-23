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

**Attribution Logo**:
A per-sample visualization of Integrated Gradients attribution over the 36bp sequence: at each position only the actual base's letter is drawn, its height set by the signed attribution (above the axis for positive, below for negative). For Model B the attribution is the Deterministic Projection Rule's projection of token-level IG, and the view must state that assumption.
_Avoid_: "Sequence Logo" (in bioinformatics that means an information-content (bits) logo over an alignment, which this is not)

**Token-grouped SHAP**:
Model B's per-sample SHAP summary in which the 8,960 embedding dimensions are collapsed into their 7 source tokens (signed sum of SHAP over each token's 1,280 dimensions, so each group shows whether it pushed the score up or down), shown alongside the 4 physical features individually — 11 groups in total. Of the 7 tokens, 6 are the 6-mers covering the 36bp sequence and one is `[CLS]`, which maps to no position and is labelled as global sequence context. Individual embedding dimension indices are never shown to users, since they carry no human-readable meaning.

## Language — SpCas9 multi-variant models

These terms describe a separate model family from Model A / Model B: nine SpCas9 variant heads over the Kim 2020 dataset, not SaCas9. They share no weights, no dataset and no split with the SaCas9 pipeline above.

**9-Hydra v1**:
The previous SpCas9 multi-variant model: one-hot encoded 30bp sequence → `Conv1D(128,3)` + `BatchNormalization` + `LSTM(64)` + `Dropout(0.3)` + `Dense(64)` shared trunk → 9 per-variant heads (`Dense(32)` → `Dense(1)`, sigmoid). Weights in `SpCas9/Multi-Cas_1IN_9Hydra_Divide_Testset.keras`, built by `models/Multi_Cas_1IN_9Hydra_Divide_Testset.ipynb`. Trained on a row-level `train_test_split(test_size=0.1, random_state=42)` over the all-9-heads subset, with Keras `validation_split=0.2`; its split roles rebuild deterministically as 21,097 train / 5,275 val / 2,931 test.
_Avoid_: "Model A" (that is the SaCas9 model), "the SpCas9 model" alone (ambiguous — Model A was pretrained on SpCas9 data too)

**9-Hydra v2**:
The current SpCas9 multi-variant model: 9-Hydra v1's architecture unchanged except a **linear** output head in place of sigmoid, trained on the 9-Hydra split with a masked per-head loss so each row contributes only to the variant heads it actually has measured. Labels are the published percentages divided by 100, a range that spans negatives (−7.5 to 95.4 raw) and which sigmoid could not represent. Built by `SpCas9/spcas9_train.py`, which reads the published split from disk rather than re-deriving it. Its functional model returns its nine outputs as a **dict** keyed by output layer name where v1 returned a list — same layers, same names, same weights; Keras 3 resolves a per-output `sample_weight` dict against the output structure, so a list-output model raises `KeyError: 0` as soon as the **head mask** arrives.
_Avoid_: "the new 9-Hydra", "the fixed model" (ambiguous once v3 exists — v3 is the architecture revision); counting the dict outputs as an architecture change (it is the container, not the graph)

**Head mask**:
The per-output Keras `sample_weight` dict that makes 9-Hydra v2's loss masked: keyed by output layer name, 0 where that head reads `n.a.` on that row — with the label filled to 0.0, since Keras needs a real number in the slot — and 1 everywhere a head is measured. It is passed to `weighted_metrics` as well as to the loss, which is why a per-head figure like `val_Output_SpCas9_NG_mae` is a mean over that head's measured rows and never an average over fabricated zeros. Built by `masked_labels_and_weights`.
_Avoid_: "the masking loss" / "the NaN loss" (no custom loss was written — a `sample_weight` dict was chosen precisely so the checkpoint reloads with no custom objects registered); "dropping the missing heads" (no row or head is dropped; the weight is what excludes it)

**9-Hydra v2 training log**:
`SpCas9/v2_train/training_log.json`, written beside the 9-Hydra v2 checkpoint by the same run that produces it: the seed, v1's hyperparameters as data, the per-epoch history, the **stopping epoch**, the TensorFlow/Keras/NumPy/pandas versions, the device (which is how the CPU-only run is evidenced — `gpus_visible` is 0 on the intended target), and the consumed **9-Hydra split**'s checksums. The split digests are recomputed from the CSVs on disk rather than copied out of `split_checksum.json` (which is itself hashed into the log), so the log pins the bytes that were trained on. `rows_fitted` is what the run actually fed the model, named apart from `split.rows` so a subsampled `--smoke` log cannot read as if the whole split went through it. Written LF-only, like the split's sidecar, since git tracks it. That is a fixed line ending, **not** byte-reproducibility: `duration_seconds` and `finished_at` record wall clock, so two runs of the same seed produce two different files on purpose — the sidecar's reproducibility rule does not carry over to this artifact. `checkpoint.sha256` likewise identifies one saved *file*, not the weights in it: seed 42 reproduces the numbers exactly (two runs agreed on `val_loss` to all 17 digits and stopped at the same epoch) while the digests differed, because a `.keras` file is a zip that embeds timestamps. The checkpoint beside it is not tracked, until there is a decision to keep one under `models/`.
_Avoid_: "the training log" unqualified (SaCas9 Model B writes its own `grouped_model_b/training_log.json` — a different family, a different schema); calling it a metrics report (it records the run, not the model's scored performance — that is #48's job)

**9-Hydra v2 scoring report**:
`SpCas9/v2_eval/results.json`, with `predictions.csv` and a `predictions_checksum.json` sidecar beside it, written by `SpCas9/spcas9_eval.py`: where 9-Hydra v2 stands on the 5,186-row **9-Hydra Testset**. Per head it reports n / Spearman / Pearson / MAE / MSE **over that head's measured rows only** — the rows where its variant was actually measured — so a headline figure is never an average over the rows the **head mask** excluded from the loss. Each head is additionally cut two ways, both read from the **9-Hydra split**'s own metadata columns rather than re-derived: `in_pam_library` (PAM-library vs non-library rows) and the **tRNA feature** (0 vs 1). A stratum holding no rows is named under `empty_strata` rather than left as an `n=0` entry — `SpCas9-NG` and `VRQR variant` have no tRNA=1 measurement at all, so for those two heads the tRNA cut carries no comparison. Seven of the nine heads score markedly worse on the PAM-library rows than off them (`HypaCas9` 0.426 against 0.843 Spearman), and every head that has tRNA=1 rows does worse on them; the tRNA cut is what says whether giving that feature to v3 is worth building. The predictions carry the split's `row_index`, which addresses the split and nothing else — the v1 membership rebuild numbers the all-9-heads frame, so its identically named column is a different count, and a v1 ↔ v2 comparison reaches v1's **Unseen target / PAM-variant overlap / Exact duplicate** categories through the `protospacer` instead. The sidecar pins the CSV's digest, the scored checkpoint's and the consumed split's, which is what lets the comparison cite these numbers without reloading TensorFlow. Metrics are on the scale v2 was trained on, the published percentage divided by 100. Scoring reads the checkpoint and the split from disk and neither retrains nor re-splits.
_Avoid_: reading a per-head `n` as the Testset size (only `SpCas9`'s 5,021, the ~5,170 of most heads and the 3,982 / 3,979 of the two censored heads are each that head's own coverage); "the eval" / "the metrics" unqualified (the SaCas9 family has its own, under `final_analysis_result/`); joining these predictions to v1's by `row_index` (the two columns count different frames); treating the tRNA stratum as a model input experiment (the feature is still metadata only)

**9-Hydra v1 ↔ v2 comparison**:
`SpCas9/v1_v2_compare/results.json`, with `basis.csv` and a `basis_checksum.json` sidecar beside it, written by `SpCas9/spcas9_compare.py`: what v2 bought, over the **1,740 rows of the 9-Hydra Testset that fall in neither of v1's train and val roles** (401 of v1's own test role plus 1,339 the all-9-heads frame never held). The two frames are related row by row through the position each row holds in the one normalized spreadsheet — `frame_row_map` — never by joining their identically named `row_index` columns. Results are reported **per category and never pooled**: 606 **Unseen target** and 1,134 **PAM-variant overlap**, cut at the protospacer against every row v1 *fitted*, train and val both, since val drove its early stopping. `pooling` in the report says why there is no combined figure. Each stratum carries a `composition` count, because 990 of the PAM-variant stratum's 1,134 rows repeat a 30-mer v1 fitted outright — **Exact duplicate**, not a new PAM on a known target. A head with fewer than `MIN_N_FOR_COMPARISON` (300) comparable **rows** in a stratum is named under `excluded` with its count instead of being scored; on the real data that is `SpCas9-NG` and `VRQR variant` in the PAM-variant stratum (190 / 189 rows), not in the Unseen-target one, where they hold 346 / 344. That 300 is a line drawn for this comparison, not an inherited threshold — `leakage_check.MIN_N_FOR_SPEARMAN` is 10, and the v1 audit's 41-row group was handled by reporting error metrics only. Every `n` is reported beside a `distinct_sequences` count, because **both models are a pure function of the 30-mer** and the strata hold replicate rows: `SpCas9`'s 466 Unseen-target rows are only 297 distinct 30-mers. `SpCas9` therefore carries two 95% CIs on its Spearman and MAE deltas — `delta_ci95` resampling rows, which #49 names, and `delta_ci95_by_sequence` resampling whole 30-mers (`paired_bootstrap.clustered_paired_bootstrap_delta_ci`). **Quote the by-sequence one**; the row-level interval is about a third narrower than the distinct targets support, and the gap between them is the size of that inflation. v1's predictions are read from `SpCas9/v1_audit/predictions.csv` by 30-mer — v1 is deterministic per sequence — and only the 319 basis rows holding a 30-mer the audit never scored go through v1's pinned checkpoint; the sidecar records that split and nothing in `v1_audit/` or `v2_eval/` is rewritten. The result: on Unseen target v2 is level or ahead on all nine heads, though `SpCas9`'s margin is not resolved (Spearman 0.815 → 0.825, by-sequence CI −0.019 to +0.042), while on PAM-variant overlap v1 leads (0.893 → 0.868, by-sequence CI −0.040 to −0.012). Per ADR 0004 that drop is the leak being removed, not a regression to tune away.
_Avoid_: quoting a pooled v1 ↔ v2 number (none is emitted, by design); reading the 1,134-row stratum as a generalization test (87% of it is memorization); "the 1,740 rows v1 never saw" (v1 saw 990 of their 30-mers in rows it fitted — it was never fitted on *these* rows); reading an `n` as a count of independent targets (`distinct_sequences` is that count); quoting `delta_ci95` as the interval (it resamples rows, and the rows are replicates)

**Stopping epoch**:
Which epoch's weights a 9-Hydra v2 checkpoint actually holds. With `EarlyStopping(restore_best_weights=True)` the saved weights are the lowest-`val_loss` epoch's, not the last one, so the **9-Hydra v2 training log** reports `best_epoch` (1-indexed, matching Keras' console numbering) separately from `epochs_run`. `val_loss` here is the protospacer-grouped validation carve-out's, passed as `validation_data` — never a Keras `validation_split`, which would slice the training rows and put a protospacer's siblings on both sides.
_Avoid_: reading `epochs_run` as the stopping epoch; "the last epoch's weights"

**9-Hydra split**:
The published train/val/test partition of the at-least-one-head frame, written to disk by `SpCas9/spcas9_split.py` as `SpCas9/published_split/{train,val,test}.csv` with a `split_checksum.json` sidecar. Train and test are `dataset/kim_2020_table_s8.xlsx`'s own `Use (Training or Test)` column **verbatim** — 32,810 / 5,186 rows, see `docs/adr/0004-honor-kim-2020-use-column.md`. Validation is the one part we carve ourselves: 10% of the 20,759 training protospacers, grouped so no protospacer spans train and val, seeded with `np.random.default_rng(42)`, giving 29,134 train / 3,676 val rows. Only the sidecar is version-controlled (`*.csv` is git-ignored); it pins the seed, the counts, each CSV's SHA-256 and `dataset_sha256`, the spreadsheet's own digest, so a re-export cannot silently redefine the split.
_Avoid_: "the grouped split" (that is the SaCas9 sequence-grouped split from issue #34 — a different dataset, and grouped by sequence rather than by protospacer); describing the train/test boundary as seeded (only the validation carve-out is)

**tRNA feature**:
The spreadsheet's `tRNA feature` column: 1 for a tRNA-N20 sgRNA, 0 for a (G/g)N19 sgRNA. Carried through the shared loader and the 9-Hydra split as `trna_feature` so later steps can stratify on it, but **metadata only** — never a model input to any 9-Hydra version.
_Avoid_: "the tRNA flag" as something the model sees

**9-Hydra Testset**:
The 5,186 rows of `dataset/kim_2020_table_s8.xlsx` whose `Use (Training or Test)` column reads `Test`, restricted to 30bp sequences with at least one measured variant head. It shares zero protospacers and zero exact 30-mers with the 32,810 training rows. Per-head label coverage is 5,021 for `SpCas9` and ~5,170 for most heads, but only 3,982 / 3,979 for `SpCas9-NG` / `VRQR variant`.
_Avoid_: "Testset" unqualified (that is the 514-row SaCas9 set defined above)

**9-Hydra v1 Testset (3,847 rows)**:
The 9-Hydra Testset narrowed to rows with **all nine** heads measured: the paper's `Use = Test` rows that fall inside the 29,303-row all-9-heads frame 9-Hydra v1 was trained on. This is the set the leakage audit scores v1 against, because v1's row-level split ignored the `Use` column entirely — of these 3,847 rows, 2,781 were trained on and 665 validated on, leaving 401 that v1 never saw. Only those 401 are comparable with the published Kim 2020 numbers. Rebuilt by `is_published_test` / `summarize`'s `published_test_by_role` in `SpCas9/`.
_Avoid_: treating it as interchangeable with the 5,186-row **9-Hydra Testset** (that one keeps rows with at least one head, and is what 9-Hydra v2 is evaluated on); "v1's test set" / "v1's Testset" as a name for it (v1's own test **role** is a different 2,931 rows, carved by `random_state=42` rather than by the `Use` column — the two sets meet only in the 401 clean rows)

**SpCas9 target window**:
The layout of the 30bp input sequence: left flank at 0-indexed positions 0-3, 20nt **protospacer** at 4-23, 3nt PAM at 24-26, right flank at 27-29.
_Avoid_: "PAM window" (that term is defined above for SaCas9's 36bp sequence at positions 25-30 — a different layout entirely)

**Unseen target / PAM-variant overlap / Exact duplicate**:
Three mutually exclusive, exhaustive categories for a row being evaluated, describing its relationship to a model's training rows. Defined at the protospacer level, because the dataset contains a PAM library of 30 protospacers each repeated across up to 268 rows that vary only at positions 24-29.
- **Unseen target**: the protospacer appears in no training row.
- **PAM-variant overlap**: the protospacer appears in training, but this exact 30-mer does not — a known target with a new PAM and a new 3' flank. A legitimate generalization test, *not* memorization.
- **Exact duplicate**: this exact 30-mer appears in a training row. The only category that is memorization.

Over 9-Hydra v1's 2,931 test rows the split is 2,191 / 699 / 41. The categories are reported separately and never pooled into a single "seen" figure, since 94% of the overlap mass is the benign kind.
_Avoid_: binary "seen / unseen"; "known target, new PAM" as the name for the middle category (the 3' flank varies too)


