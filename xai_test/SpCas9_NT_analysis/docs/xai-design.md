# 9-Hydra v3 XAI design

Explainability analysis of **9-Hydra v3**, the Nucleotide Transformer fine-tune of the SpCas9
multi-variant cleavage-efficiency model. This is a **single-model analysis**: it does not compare
v3 against 9-Hydra v1 or v2, and it reuses the SaCas9 pipeline's *techniques* only — not its
comparison frame, which was defined entirely by two models' error quantiles.

Scope decisions in this document were settled in a design interview and are not open for
re-derivation during implementation. Anything marked **Open** below is not yet settled.

## Objective and boundaries

The analysis answers three questions about v3:

1. **Where in the 30bp window does v3 read cleavage efficiency from?** Per-position sensitivity
   across the protospacer, PAM and flanks.
2. **Does the shared trunk use the PAM differently per variant head?** `SpCas9-NG` recognises NGA
   where `SpCas9` requires NGG; if their PAM sensitivity profiles do not separate, that is itself
   a finding about the shared trunk.
3. **Where does v3 fail, and does the explanation describe the failure?** Error-based and
   head-ordering-based cases, described rather than fixed.

Out of scope: retraining v3, changing its split, tuning it, and any claim about physical
DNA/protein binding. Attribution describes the model's internal information use, nothing else.

## Model and verified split facts

**9-Hydra v3** is `InstaDeepAI/nucleotide-transformer-500m-human-ref` fine-tuned as a 9-head
multi-task regressor (`AutoModelForSequenceClassification`, 9 outputs) on 30bp sequences,
`max_length=32`, per-head mean/std standardisation (`mu_sd.npy`), and a masked loss so a row
contributes only to the heads it has measured. Built by `SpCas9_NT_analysis/SpCas9_v3.ipynb`;
`config.json` confirms `EsmForSequenceClassification` with `problem_type: regression` and nine
`id2label` entries over a 24-layer, hidden-1280 backbone. The weights live on Drive and are read
there by the Colab drivers; the analysis consumes the checkpoint, never rebuilds it.

Verified against the checkpoint's **own** tokenizer in `result/`: `30bp -> ['<cls>'] + 5` 6-mer
tokens.

The **SpCas9 target window** layout (0-indexed over 30bp): left flank 0-3, protospacer 4-23,
PAM 24-26, right flank 27-29.

v3 trains on `SpCas9/published_split/`. Measured directly from those CSVs:

| | train | val | test |
|---|---|---|---|
| rows | 29,134 | 3,676 | 5,186 |
| distinct protospacers | 18,684 | 2,075 | 2,426 |

- test rows whose protospacer appears in train or val: **0 / 5,186**
- test 30-mers appearing in train or val: **0**

**Consequence:** every test row is an unseen target for v3. The `Unseen target` /
`PAM-variant overlap` / `Exact duplicate` trichotomy in `CONTEXT.md` is a **v1 artifact** — v1's
row-level random split ignored the `Use` column, which is what made those categories meaningful.
For v3 they collapse to a single category and **must not be used**; writing "restricted to unseen
targets" about a v3 population is a vacuous truth that reads as if a leakage control had been
applied.

## Analysis population and strata

The population is the 9-Hydra Testset, 5,186 rows. Its internal heterogeneity is not relationship
to training data but the dataset's own construction:

| stratum | rows | distinct protospacers |
|---|---|---|
| `in_pam_library = False` | 3,889 | 2,421 |
| `in_pam_library = True` | 1,297 | **5** |

NGG rows 2,008 / non-NGG 3,178. `trna_feature = 1` on 1,191 rows (metadata only, per ADR 0005).

The two strata are analysed **separately and never pooled**, because they answer different
questions:

- **Non-library stratum, restricted to NGG** — sequence generalisation: what the model reads from a
  protospacer it has never seen, *once the PAM shortcut is unavailable*. On the analysis
  intersection this is 895 rows over 818 distinct protospacers.
- **PAM library stratum, all PAMs** — 5 protospacers repeated across 1,297 rows that vary only at
  positions 24-29 (1,287 on the intersection). A natural experiment on the PAM with the target held
  fixed, and where 9-Hydra v2 collapsed (`HypaCas9` 0.426 Spearman against 0.843 off-library). The
  PAM contrast *is* the object of study here, so non-NGG rows are kept deliberately: restricting
  this stratum to NGG would leave 134 rows over 5 targets and delete the question.

### Why the non-library stratum is NGG-restricted

v3's own metrics report Spearman **0.9235 over all Testset rows** against **0.6054 within NGG**.
Measured on the `SpCas9` head's 5,021 labelled rows, the reason is in the labels, not the model:

| | n | mean efficiency | median |
|---|---|---|---|
| NGG | 1,907 | 47.5 | 49.6 |
| non-NGG | 3,114 | 7.6 | 1.9 |

The binary `is_NGG` **on its own** reaches Spearman 0.7965 against the label. A predictor that
emitted nothing but "does the PAM end in GG" would score 0.80 on the pooled set, so the pooled
0.9235 rests on a floor that requires no sequence understanding beyond a 2nt check, and 0.6054 is
the figure that reflects ranking targets against each other.

The consequence for attribution is direct: a position profile computed over the pooled population
is dominated by positions 25-26, and its headline reads "the model uses the PAM" — true before the
analysis began, and therefore not a finding. Restricting the generalisation question to NGG removes
that confound so the protospacer signal is visible. This refines the stratum decision rather than
replacing it: the two strata were already assigned different questions, and each now gets the
population its question needs.

**Reporting constraint on the library stratum:** its n is **5 targets**, not 1,297 samples. Every
library-stratum finding is reported as "observed on 5 targets" with the per-target breakdown
beside it. Treating 1,297 PAM variants of 5 protospacers as 1,297 independent observations would
overstate the evidence by roughly the replication factor.

## Head selection

v3 has nine heads; Integrated Gradients, ISM and attention rollout are each defined against **one
scalar output**, so a selection is unavoidable. Analysed heads:

- **`SpCas9`** — primary.
- **`SpCas9-NG`** — contrast: relaxed PAM requirement, so its PAM-position sensitivity should
  separate from `SpCas9`'s.
- **`evoCas9`** — contrast: the most conservative variant.

Measured coverage on the Testset is `SpCas9` 5,021, `SpCas9-NG` 3,982, `evoCas9` 5,161; **3,851
rows have all three measured**.

**The analysis intersection is those 3,851 rows**, and every cross-head comparison is computed on
it. Rationale: head-agreement is undefined unless the heads are compared on the same rows — if
`SpCas9-NG`'s ~1,000 unmeasured rows entered the other heads' figures, "the heads disagree" could
be a population difference rather than a model property.

Single-head absolute figures (one head's own profile, metrics, error distribution) may use that
head's full coverage. The two are **named differently in all artifacts** so an intersection figure
is never read as a coverage figure.

## Token resolution: PAM attribution belongs to ISM, not IG

NT uses a **non-overlapping 6-mer** tokenizer (vocabulary: 4,097 6-mers, 5 single bases, 5 special
tokens). Verified empirically with the base vocabulary:

```
'ACGTACGTACGTACGTACGTACGTAGGTCA'  (30bp)
  -> ['<cls>', 'ACGTAC', 'GTACGT', 'ACGTAC', 'GTACGT', 'AGGTCA']
```

30 is divisible by 6, so the window is exactly 5 tokens with boundaries
`0-5 | 6-11 | 12-17 | 18-23 | 24-29`. The final token carries **PAM (24-26) and the right flank
(27-29) together**.

Integrated Gradients produces one attribution per token. Dividing a token's attribution evenly
across its 6 bases — the SaCas9 pipeline's **Deterministic Projection Rule** — cannot recover
which half of token 4 the signal came from. A number labelled "PAM attribution" obtained that way
is a PAM-plus-flank average.

**Therefore:**

- **IG** reports at **5-token resolution** and makes no per-base claim. Token 4 is named
  `pam_plus_right_flank`, never `pam`.
- **ISM** carries the per-base PAM question. A single-base substitution is indifferent to token
  boundaries, so the 30 x 3 sweep gives direct base-level resolution at positions 24, 25 and 26.

This is recorded as an ADR because the obvious implementation (project the token attribution and
read positions 24-26 off it) produces a confidently wrong answer to the analysis's second
objective. SaCas9 did not hit this: its 36bp window is also a multiple of 6, but its PAM window at
25-30 straddles two tokens, so the failure mode was diffuse rather than a clean PAM/flank merge.

## Techniques

Applied:

1. **Integrated Gradients** — token-level attribution, **200 integration steps**, computed on the
   **encoder input-embedding tensor** with a
   zero-embedding baseline. See ADR on attribution depth. Implemented in `v3_ig.py`, which
   attaches to the embedding by **replacing the word-embedding module's output through a forward
   hook** rather than by passing `inputs_embeds`: ESM's `token_dropout` rescale is guarded on
   `input_ids is not None`, so the `inputs_embeds` path skips it and would explain a function the
   deployed model never computes. The Riemann sum uses the **midpoint** rule — at the same step
   count, a markedly smaller completeness residual, and the residual is the audit signal for
   ADR 0006 rather than a diagnostic to be explained away.

   **The step count is 200 because 50 was measured and found short.** The SaCas9 pipeline's 50
   midpoint steps gave the first real 400-row run a completeness pass rate of **0.85**, with
   `|residual|` reaching 0.69 on the standardised scale — ~15 percentage points once multiplied
   by `sd`. Re-running three of the worst rows against the same checkpoint at 200 steps dropped
   their residuals to 0.0026 / 0.0194 / 0.0026: a 32-265x fall where the midpoint rule's `O(h^2)`
   predicts 16x, so the 50-step residual was dominated by integration error rather than by a
   missing share of the integral. The fix is steps, not a looser tolerance — a completeness
   report failing 15% of its checks cannot tell an IG-vs-ISM disagreement from its own
   integration error, which is the one thing the cross-technique comparison needs it to do.
2. **In-silico saturation mutagenesis (ISM)** — all 30 positions x 3 alternative bases, delta
   against wild type. Carries the per-base PAM result. Implemented in `v3_ism.py`. The delta is
   **absolute**, on the fraction scale (`(mutant - wild type) * sd`, so `mu` cancels), not
   SaCas9's relative `(mutant - wt) / |wt|`: that ratio is unbounded as the wild-type score
   approaches zero, which is why `ism_sweep.py` NaN-masks it — and NaN is forbidden here. The
   sweep runs in **row shards** written to a checkpoint directory as they complete, so a Colab
   disconnect or an OOM costs the unfinished shards rather than all ~350K forward passes, and the
   **batch size is an argument recorded in the sidecar** rather than a constant: `outOfMemory.md`
   records SaCas9's ISM sweep dying inside a forward pass, and a batch size that is not written
   down makes that report unactionable.
3. **Attention rollout** — per-layer `0.5*A + 0.5*I` then row-normalise, `<cls>` removed, reported
   as **information dependency** between token positions, not physical interaction. Stored and
   reported at **5x5 token resolution** and *not* expanded to nucleotide coordinates: attention is
   a token-to-token quantity, and projecting it onto 30 positions would re-introduce exactly the
   PAM/flank merge that ADR 0007 rules out. The SaCas9 pipeline did expand its rollout to 36x36
   nucleotide coordinates; that choice is not carried over. Implemented in `v3_rollout.py`, which
   takes the token contract from `v3_ig.py` rather than restating it — ADR 0007 governs both
   analyses, so there is one copy of the boundaries. Two consequences of the above are worth
   stating because an artifact alone does not show them: the run must hold the model on the
   **`eager` attention path**, since a fused kernel never forms the probability matrix and
   `transformers` 5.x answers `output_attentions=True` on one with an *empty tuple* rather than an
   error (the implementation actually in force is recorded in the sidecar, so a run cannot claim a
   path it did not set); and each stored row sums to **less than 1**, because `<cls>` takes part in
   the rollout and is dropped after it, with no renormalisation — rescaling would assert a
   token-to-token dependency the encoder routed through `<cls>`. The sidecar records the discarded
   share as `cls_mass`, and the rows are asserted to put no mass on the padding, which is the only
   symptom an attention mask that stopped reaching the encoder would have.
4. **Positional / PAM profiling** — aggregate ISM deltas by region and by PAM identity, per
   stratum and per head.
5. **Cross-head attribution agreement** — rank correlation between heads' ISM profiles on the
   3,851-row intersection. This axis exists only because v3 is multi-head, and it requires no
   comparison model; it is the analysis's single-model-native contribution. Implemented in
   `v3_agreement.py`, which reads `ism.npz` and nothing else: rollout has no head axis (ADR 0008)
   and IG is not a prerequisite, which is why this is its own ticket. It reports per head pair and
   per position range, with both named in the artifact, and carries analysis question 2 — the
   `SpCas9` / `SpCas9-NG` PAM contrast — as a **reported finding either way**, under a separation
   margin recorded beside the verdict.

**Deliberately not applied**, recorded here so the omission is not re-litigated:

- **Tree SHAP** and **Token-grouped SHAP** — the SaCas9 pipeline's SHAP operated on Model B's
  XGBoost regressor over an 8,964-feature vector. v3 has no tree model and no tabular feature
  vector.
- **Feature ablation (embedding / physical / full)** — v3 has no physical or thermodynamic feature
  channel to ablate; it is sequence-only by ADR 0005.
- **Paired bootstrap delta CIs** — these measure the difference between two models. There is no
  second model in scope.

## Budget

One forward pass yields all nine heads, so head selection does **not** multiply inference cost.
Against the 3,851-row intersection:

| analysis | inference | population |
|---|---|---|
| ISM | 3,851 x 91 ~ 350K forward | **full 3,851** |
| Attention rollout | 3,851 forward | **full 3,851** |
| Integrated Gradients | 400 x 200 x 3 ~ 240K forward+backward | **400-row stratified subset** |

ISM and rollout run exhaustively because they can: at 30bp and ~32 GFLOP per sequence the cost is
tens of minutes on a single GPU, and the stored arrays are small — the ISM delta `(3851, 30, 4, 9)`
is 16 MB as float32, and a rolled-out matrix `(3851, 5, 5)` is ~385 KB (24x that if every layer is
kept separately rather than only the rollout). Sampling an analysis that fits leaves a permanent
"is this an artifact of the sample?" question for no gain — and ISM in particular carries the PAM
result, so reducing it would undercut the token-resolution decision above.

**IG subset**: 200 rows per stratum, 400 total, drawn independently within each stratum. Seeded and
written to disk as a fixed membership list with a checksum, so every later IG artifact addresses
the same rows.

`v3_selection.py` draws it into `artifacts/v3_ig_subset/ig_subset.json`, under seed 129, with two
pins that answer different questions: the **membership digest** inside the manifest says which
rows an IG artifact must address, and `ig_subset_checksum.json` says the manifest's bytes have not
been edited since it was drawn. The manifest is stored **`row_index` ascending**, with selection
order, stratum and cell recorded as fields — overlaying the draw order on the storage axis is how
a re-drawn manifest silently reorders an array that was joined to it positionally.

"Every later IG artifact addresses the same rows" is enforced rather than asked for:
`write_ig_artifact` takes the manifest as a **path**, loads it through `load_ig_subset` so the
checksum is verified, refuses rows that are not the set it claims, and records the membership
digest in its sidecar. Taking a path rather than an object is the point: a hand-built manifest
would satisfy a row comparison while proving nothing about which bytes the subset was drawn into.
Its `stratum` defaults to `ALL_STRATA` — one archive over all 400 rows, which is the `.npz` schema
below — and a combined write's own rows are the whole manifest's, so its `stratum_digest` equals
the `membership_digest`. Naming a stratum narrows it to that stratum's 200 rows; what the writer
enforces either way is that the artifact holds exactly the rows it says it does, so a 200-row
archive cannot read as the 400-row one.

**Decided in #131, where the AC was silent:** the library stratum's two balancing axes are
applied as a **nesting** — NGG / non-NGG first, the error quartiles within each half — and a cell
too thin to take its even share has the shortfall redistributed over the cells that have
capacity. Both choices are forced by the data. The stratum holds only 134 NGG rows and they
cluster in the high-error quartiles, so evening all eight cells as one flat axis returns
**80 NGG / 120 non-NGG**, which balances the quartiles by sacrificing the axis the stratum exists
to contrast; nested, it is **100 / 100** with the NGG half allocated 7 / 14 / 28 / 51 and the
non-NGG half 25 each. Without redistribution the draw would come back at 183 rows rather than
200. The per-cell allocation is in the manifest, so a reader can see the unevenness rather than
infer it.

- Non-library (NGG-restricted, 895 available): balanced on `SpCas9` error quartile only — the
  stratum is already single-valued on NGG, so there is nothing to balance there.
- Library (all PAMs, 1,287 available): balanced on NGG / non-NGG and on `SpCas9` error quartile,
  since the PAM contrast is what this stratum is for. Its 200 rows are **5 targets' 200 PAM
  variants** and inherit the stratum's reporting constraint.

Precedent for freezing a budget-selected detailed subset:
`docs/adr/0006-cas12a-xai-freezes-a-budget-selected-detailed-subset.md` (repo root).

## Execution split

The boundary is **whether a model forward pass is required**.

**Colab (GPU, reads the checkpoint from Drive):** ISM sweep, IG, attention rollout. Each writes
`.npz` arrays plus a `.json` metadata sidecar and nothing else. The driver notebook pins its
`CODE_REF` to a reviewed commit and **pins the `transformers` version**, recording it in the
metadata sidecar. The pin is enforced where the artifact is written, not only asked for in the
notebook: `write_ism_artifact` refuses a `code_ref` that is not a 40-character commit sha — a
branch name moves and cannot address a run — and refuses to write a sidecar whose `transformers`
or `torch` version it cannot record. The driver additionally checks out `CODE_REF` and asserts
`git rev-parse HEAD` matches it, and asserts the consumed `test.csv` has the sha256 that
`SpCas9/published_split/split_checksum.json` pins, so "which split did this run read?" is answered
before the GPU time is spent rather than after.

**Local (no checkpoint, no GPU, fully tested):** IG subset selection, case selection, all
aggregation and stratification, cross-head agreement, the IG-vs-ISM validation, and report
rendering. Case selection reads predictions and labels only, so it needs no model — which is what
lets it carry tests.

Consequences:

- The local test suite builds a **small randomly-initialised 9-output ESM model inside the test**
  for shape and gradient-flow checks. No download, no checkpoint, no network.
- `SpCas9_NT_analysis/result/` holds the checkpoint's **metadata only** — `config.json`
  (`EsmForSequenceClassification`, `problem_type: regression`, nine `id2label` entries, 24 layers,
  hidden 1280), the tokenizer files, `training_args.bin`, and `mu_sd.npy` as `(2, 9)` float32. The
  weights are not there and are not needed locally.
- v3's **per-head Testset predictions** are produced by `v3_predict.py` into
  `artifacts/v3_testset_predictions/`, in the `predictions.csv` +
  `predictions_checksum.json` shape `spcas9_eval.py` uses for v2. The notebook writes only a
  metrics summary, so this artifact did not previously exist.

**Scale, settled.** Predictions are **fractions of 1** (v3's `predict` path de-standardises with
`mu_sd.npy`), while the split's label columns are **raw percentages**. Divide the labels by 100
before taking an error; mixing the two is a 100x mistake. Both scales are named in every sidecar.

The export was validated end to end against the run's own `nt_v3_seed42_results.csv`, recomputed
from the written predictions:

| reported | recomputed |
|---|---|
| `test_Sp_all` 0.9235 | 0.9235 |
| `test_Sp_ngg` 0.6054 | 0.6054 |
| `test_mean_all` 0.8554 | 0.8554 |
| `test_mean_ngg` 0.7020 | 0.7020 |
| `test_SpCas9_MAE` 0.0510 | 0.05098 |

The consumed split's SHA-256 matches `published_split/split_checksum.json`'s `files/test.csv`
entry, so the predictions are pinned to the same bytes v3 was evaluated on.
- The locally installed `transformers` is 5.8.1; Colab's may differ. Version drift inside
  `EsmForSequenceClassification` is the exact mechanism that silently zeroed Model B's IG in the
  SaCas9 pipeline, hence the pin.

## Case selection

The SaCas9 pipeline's Concordant / Discordant classes are definitionally two-model (`e_A` vs
`e_B` quantiles) and **do not carry over**. Replacement, applied **within each stratum
separately**:

1. **Error quantile on `SpCas9`** — well-predicted (lowest quartile of absolute error) and
   badly-predicted (highest quartile) rows.
2. **Cross-head error disagreement** — the spread (max − min) of the three analysed heads'
   absolute errors on a row, taken from **each stratum's own top decile**. The two strata differ
   about twofold in scale (generalisation median 0.075, library 0.038), so one absolute threshold
   would be p90 in one and p75 in the other. This is a failure mode that exists only in a
   multi-head model and needs no second model to define.

**Head-ordering reversal is not a selection criterion.** It was planned as one and does not
survive measurement: 203 of the 3,851 intersection rows have Kendall tau < 0, but requiring every
inverted pair to be separated by more than the **replicate noise floor** (pooled median 0.053,
measured over 79 repeated 30-mers) leaves **one row**. Lowering the margin to manufacture a pool
would report measurement noise as model failure. Reversal is instead a **reported result** under
analysis question 3, as a per-stratum rate plus the distribution of inverted-pair gaps against
that noise floor — and it is a substantive one, since 169 of the 203 fall in the library stratum,
which is a third of the rows. Reports that lean on the noise floor state that it rests on 79
groups.

The floor itself is produced by `v3_noise.py` into `artifacts/v3_noise_floor.json` (AC14), from
**labels only** — the spread of a repeated 30-mer's measurements is a property of the assay, not
of the model, so a floor measured through predictions would confound the two. The artifact reports
the floor **per head as well as pooled**, because the three are far apart: `SpCas9` 0.064,
`SpCas9-NG` 0.070, `evoCas9` 0.0099, a 7.1x span the artifact computes as `head_floor_ratio`
(the "sixfold" quoted elsewhere is the `SpCas9`/`evoCas9` pair, 6.4x). The pooled median is about
five times `evoCas9`'s own floor, so a single constant is five times too generous for that head.
`v3_noise.floor_for_head` is the only lookup and refuses `"pooled"` for that reason.

The 79 groups and 379 rows go into the artifact rather than a hand-written footnote, since AC13
has to report that limit — and with them the sharper number, which is that the split holds **779**
repeated 30-mers over 2,247 rows and only 79 of those groups survive the intersection cut. The
floor is bounded by how few repeated rows have all three analysed heads measured, not by how
rarely the assay repeats a target.

### Budget and identity

Six cells: `{generalisation, pam_library}` x `{ERRLO, ERRHI, HEADGAP}`, **3 cases per cell**, at
most 18. In the library stratum a cell takes **at most one row per distinct protospacer**, so its
three cases are three different targets rather than three PAM variants of one; the error-quartile
cells there span all 5 targets (322 rows each), so the constraint is satisfiable. Every library
case record carries the "observed in one of 5 targets" qualifier.

`case_id` is `<STRATUM>-<CRITERION>-<NN>` — `GEN-ERRLO-01`, `LIB-HEADGAP-03` — uppercase,
hyphenated, two-digit zero-padded, prefixes derived from the `v3_strata` constants
(`GENERALISATION` → `GEN`, `PAM_LIBRARY` → `LIB`). `HEADGAP`, not `DISAG`, because the latter
reads as the SaCas9 pipeline's `Discordant`. Ties break on `row_index` ascending, asserted by test.

**Decided in #131, where the AC bounded the pool but not the pick:** the quartile and the decile
bound the **pool**; within it the three cases are the pool's extremes (lowest error for `ERRLO`,
highest for `ERRHI`, widest gap for `HEADGAP`). Keeping the two separate is what makes the library
stratum's distinct-protospacer rule safe: a cell that has to look past a second PAM variant of a
target it already used walks further down the rank but cannot leave the pool, and a cell that
cannot fill under the rule comes back **short** rather than reaching outside it — which is what
the budget's "at most 18" allows for.

**A row is a case once**, and the criteria are taken in `ERRLO, ERRHI, HEADGAP` order, so
`HEADGAP` yields to the two error cells and takes the widest gaps they did not claim. This was
not anticipated: the six cells were assumed to be six different rows, and on the real split they
are not. Taking each cell's own extremes independently put **2 of 3** generalisation and **all 3**
library `HEADGAP` cases on rows that were already `ERRHI` cases — the widest cross-head gaps are
the worst-predicted rows. `HEADGAP` is the criterion that moves because a row already reported as
max-error teaches nothing about multi-head disagreement, which is the only thing that cell is
for; `ERRLO` and `ERRHI` keep the true extremes a reader will expect of "best" and "worst
predicted". The collision count is written to `cases.json` as `cell_overlap_before_dedup` rather
than resolved silently, because "this model's cross-head disagreement is concentrated in its
high-error tail" is a result under analysis question 3, not an implementation detail.

Attribution outliers are **excluded from selection** and used only to describe selected cases:
selecting cases by the attribution that the case study then explains is circular. This is enforced
at the artifact level, not by convention. `artifacts/v3_cases/cases.json` is written locally from
predictions and labels and **does not reference any attribution artifact**; once the `.npz` files
exist, `cases_annotated.json` points at it by `case_id` and adds attribution description, and
**never rewrites the selection**. A reviewer can therefore confirm non-circularity from the
dependency direction of two files, which is what makes the one-line regression that restores the
circularity visible.

## Artifacts

Layout, under `SpCas9_NT_analysis/`:

```
SpCas9_v3.ipynb          # training, plus one CODE_REF-pinned GPU driver cell per Colab analysis
docs/xai-design.md       # this document
<modules>.py             # analysis logic, no model dependency at import time
test/                    # local pytest suite
artifacts/               # .npz + .json outputs
```

The two **frozen lists** are tracked JSON, unlike the `.npz` arrays and the `*.csv` inputs, and so
is the measured noise floor:

```
artifacts/v3_ig_subset/ig_subset.json  + ig_subset_checksum.json        # AC7
artifacts/v3_cases/cases.json          + cases_checksum.json            # AC9
artifacts/v3_noise_floor.json          + v3_noise_floor_checksum.json   # AC14
```

The noise floor sits at the artifacts root rather than in a directory of its own, which is where
the other two live: AC14 names that path outright and AC13's consumer reads it.

The frozen lists are written by `python v3_selection.py` from the split and the prediction export
alone, and both record those two files' SHA-256 — `test.csv`'s against the pin in
`published_split/split_checksum.json`, which stops a selection drawn against the wrong bytes at
the point it is drawn. A test redraws them and compares the committed membership digest, so the
tracked lists are checkable rather than merely trusted.

The subtree does **not** import the root SaCas9 analysis modules (`integrated_gradients.py`,
`attention_rollout.py`, `ism_sweep.py`, ...). Those are written against a 36bp window, a
single-output regressor and the SaCas9 tokenizer; generalising them to 30bp and nine heads would
put both pipelines on one changing contract for no shared benefit. Techniques are reused,
implementations are not. This also keeps the repo's test-scope rule ("the subtree you touched")
cleanly applicable.

Every artifact carries a metadata sidecar recording the checkpoint digest, the consumed split's
checksums, the `transformers` and `torch` versions, the analysed heads, the stratum, and the
population or subset membership digest. `v3_noise_floor.json` is the one artifact without a
checkpoint digest or a model library version, and deliberately so: it is measured from labels
alone, so pinning `torch` there would imply a checkpoint touched a number that is a property of
the assay. It still carries the split's checksum and the membership digest of the rows it rests
on.

### `.npz` schema

**One `.npz` per analysis, each with its own sidecar** — not one combined archive. The three
analyses run as separate Colab sessions over different populations, so a combined file would make
re-running one rewrite another's arrays, and a single sidecar cannot hold three runs'
`transformers` versions. SaCas9's combined `model_analysis_arrays.npz` existed to join *two
models* and is not inherited.

```
artifacts/v3_ism/ism.npz          + ism.sidecar.json
artifacts/v3_rollout/rollout.npz  + rollout.sidecar.json
artifacts/v3_ig/ig.npz            + ig.sidecar.json
```

Every archive carries `row_index` (int64) aligned to axis 0, and axis 0 is ordered **`row_index`
ascending** in all three. The ordering is asserted by test and the array's digest goes in the
sidecar. Implicit positional identity — SaCas9's `sample_id`-by-row-order — is the mechanism by
which a 3,851-row and a 400-row artifact silently mis-join.

| archive | key | shape | dtype |
|---|---|---|---|
| `ism.npz` | `delta` | `(3851, 30, 4, 9)` | float32 |
| | `wt_prediction` | `(3851, 9)` | float32 |
| | `reference_base` | `(3851, 30)` | int8 |
| | `row_index` | `(3851,)` | int64 |
| `rollout.npz` | `rollout` | `(3851, 5, 5)` | float32 |
| | `row_index` | `(3851,)` | int64 |
| `ig.npz` | `attribution` | `(400, 3, 5)` | float32 |
| | `completeness_residual` | `(400, 3)` | float64 |
| | `special_attribution` | `(400, 3)` | float64 |
| | `row_index` | `(400,)` | int64 |

`special_attribution` is the share of the integral that landed on `<cls>` and the padding. It is
not part of the reported profile, and it is in the archive because without it the completeness
check can only be *re-read*, not *re-derived*: the residual is taken against the sum over all 32
token positions, while only the 5 sequence tokens are reported, so `f(x)` is recoverable from the
artifact alone only as `f(0) + attribution.sum() + special_attribution + completeness_residual`.

**ISM is `(n, 30, 4, 9)`, ACGT in fixed order, with the reference base's slot exactly 0.** The
fourth slot costs 33% more (16MB total) and buys an axis that means the same thing in every row;
SaCas9's `(n, 36, 3)` alt-only layout makes "slot 0" a different base at each position, and this
analysis indexes PAM bases directly because ADR 0007 assigns base-resolution PAM work to ISM. A
zero in the reference slot is true, not a sentinel.

`reference_base` and `wt_prediction` are in the archive for the same reason `special_attribution`
is in `ig.npz`: without them the array can only be *re-read*, not *checked*. `reference_base` says
which slot each row's zero belongs to, so "reference" stays distinguishable from "no effect" for a
reader holding the artifact alone rather than the artifact plus the split. `wt_prediction` is the
wild-type prediction per row and head on the fraction scale, so it must equal `predictions.csv` at
the same `row_index` — which turns a mis-join into a failed comparison instead of a result. Both
are small: 139 KB and 115 KB against the delta's 16 MB.

**ISM stores all nine heads; IG stores three.** One forward returns all nine, so the extra six
cost nothing and avoid a re-run; IG needs a backward pass per head, which is what fixes its budget
at the three analysed heads. The sidecar lists `heads` in axis order, and the ISM sidecar states
that the six unanalysed heads are **stored but not analysed**, so they are not read as validated
results.

**Rollout has no head axis** — see ADR 0008. Only the final rollout is stored, not the 24
per-layer matrices.

**IG's hidden axis is summed away.** Keeping `(400, 3, 5, 1280)` would be 7.7M values with no
interpretation path. The per-row per-head `completeness_residual` is kept at float64 because it is
the audit trail for the all-zero-attribution failure ADR 0006 exists to prevent: without it in the
artifact, the check cannot be re-run after the Colab session ends. The sidecar additionally records
`f_baseline`, the nine scalars `f(0)` at the zero embedding, for the same reason. No key named
`pam` exists in `ig.npz`; the token is `pam_plus_right_flank`.

**NaN is forbidden.** Each module asserts `np.isfinite(...).all()` before writing. A head with no
measurement on a row still has a prediction, so an ISM delta is defined everywhere and a NaN here
can only be a bug — unlike SaCas9, which masked genuine prediction failures. Measurement coverage
is handled upstream by the 3,851-row intersection.

## Validation

Two independent layers, because they catch different failures:

1. **Implementation** — local unit tests: IG completeness (attribution sum ~ f(x) - f(baseline)
   within tolerance) and a non-zero attribution check, both against the small in-test model.
   This is what the SaCas9 pipeline lacked when Model B's IG came back identically zero.
2. **Interpretation** — rank correlation between each head's IG token profile and its ISM profile
   aggregated to the same 5 tokens, reported per stratum. Disagreement here is not a bug: it marks
   the limit of what the token-level attribution supports, and is reported as such.

## Tracking

`shin584/project#129` — "[SpCas9 XAI] 9-Hydra v3 단독 설명가능성 분석 계획" — holds the plan
summary and the twelve acceptance criteria, and points at this document as the authoritative
design. Its body is kept in `xai-issue-body.md` beside this file. It is split into child tickets
once the open items above are settled.
