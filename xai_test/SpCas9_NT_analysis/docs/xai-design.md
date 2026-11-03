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

1. **Integrated Gradients** — token-level attribution, 50 integration steps (matching the SaCas9
   pipeline's `DEFAULT_IG_STEPS`), computed on the **encoder input-embedding tensor** with a
   zero-embedding baseline. See ADR on attribution depth.
2. **In-silico saturation mutagenesis (ISM)** — all 30 positions x 3 alternative bases, delta
   against wild type. Carries the per-base PAM result.
3. **Attention rollout** — per-layer `0.5*A + 0.5*I` then row-normalise, `<cls>` removed, reported
   as **information dependency** between token positions, not physical interaction. Stored and
   reported at **5x5 token resolution** and *not* expanded to nucleotide coordinates: attention is
   a token-to-token quantity, and projecting it onto 30 positions would re-introduce exactly the
   PAM/flank merge that ADR 0007 rules out. The SaCas9 pipeline did expand its rollout to 36x36
   nucleotide coordinates; that choice is not carried over.
4. **Positional / PAM profiling** — aggregate ISM deltas by region and by PAM identity, per
   stratum and per head.
5. **Cross-head attribution agreement** — rank correlation between heads' ISM profiles on the
   3,851-row intersection. This axis exists only because v3 is multi-head, and it requires no
   comparison model; it is the analysis's single-model-native contribution.

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
| Integrated Gradients | 400 x 50 x 3 ~ 60K forward+backward | **400-row stratified subset** |

ISM and rollout run exhaustively because they can: at 30bp and ~32 GFLOP per sequence the cost is
tens of minutes on a single GPU, and the stored arrays are small — ISM delta `(3851, 30, 3)` is
~1.4 MB as float32, and a rolled-out matrix `(3851, 5, 5)` is ~385 KB (24x that if every layer is
kept separately rather than only the rollout). Sampling an analysis that fits leaves a permanent
"is this an artifact of the sample?" question for no gain — and ISM in particular carries the PAM
result, so reducing it would undercut the token-resolution decision above.

**IG subset**: 200 rows per stratum, 400 total, drawn independently within each stratum. Seeded and
written to disk as a fixed membership list with a checksum, so every later IG artifact addresses
the same rows.

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
metadata sidecar.

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
2. **Head-ordering reversal** — rows where the measured ordering of the three analysed heads
   disagrees with the predicted ordering (e.g. measured `SpCas9` > `SpCas9-NG`, predicted the
   reverse). This is a failure mode that exists only in a multi-head model and needs no second
   model to define.

Attribution outliers are **excluded from selection** and used only to describe selected cases:
selecting cases by the attribution that the case study then explains is circular.

**Open:** the per-stratum case budget and the `case_id` scheme.

## Artifacts

Layout, under `SpCas9_NT_analysis/`:

```
SpCas9_v3.ipynb          # training (exists)
docs/xai-design.md       # this document
<modules>.py             # analysis logic, no model dependency at import time
test/                    # local pytest suite
colab_*.ipynb            # GPU drivers, CODE_REF-pinned
artifacts/               # .npz + .json outputs
```

The subtree does **not** import the root SaCas9 analysis modules (`integrated_gradients.py`,
`attention_rollout.py`, `ism_sweep.py`, ...). Those are written against a 36bp window, a
single-output regressor and the SaCas9 tokenizer; generalising them to 30bp and nine heads would
put both pipelines on one changing contract for no shared benefit. Techniques are reused,
implementations are not. This also keeps the repo's test-scope rule ("the subtree you touched")
cleanly applicable.

Every artifact carries a metadata sidecar recording the checkpoint digest, the consumed split's
checksums, the `transformers` and `torch` versions, the analysed heads, the stratum, and the
population or subset membership digest.

**Open:** the `.npz` key schema and array shapes.

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
