# 9-Hydra NGG evaluation (#82)

`SpCas9.spcas9_data.is_ngg(frame)` checks the normalized 30mer's PAM at
positions 24–26: its last two bases must be `GG`. Non-NGG is its complement.
PAM-library membership is independent and remains unchanged metadata.

`SpCas9.spcas9_metrics.population_metrics(predictions)` scores NGG and non-NGG
using each head's non-missing labels. Measured zero contributes; missing labels
do not. Labels and predictions retain the percentage/100 scale and are not
clipped. Repeated targets and their original condition metadata remain rows.

Each population reports all nine heads and pooled MAE, MSE, Spearman and Pearson.
`n` counts measured head/row pairs; for a single head it equals `n_rows`.
Pooled `n_rows` counts rows with at least one measured head, while pooled `n`
counts their measured observations across heads. `unique_30mers` and
`unique_protospacers` describe those eligible rows, without reweighting them.
Population `rows` counts all rows in that PAM population, including any rows
without measured labels.

`equal_head_ngg_summary(populations)` returns the arithmetic mean of the nine
NGG head MAEs and the arithmetic mean of their Spearman correlations. Pooled
correlations instead use concatenated measured observations. A missing required
head metric makes its summary unavailable; available heads are never silently
averaged alone. Empty groups, non-finite measured values, insufficient
correlation samples and constant arrays have explicit `unavailable_reasons`.
The existing minimum of ten measured rows for correlations is preserved.
Undefined figures are JSON `null`, never NaN or Infinity.

`spcas9_eval.build_report` adds `populations`, `equal_head_ngg` and
`metrics_schema_version: 2` to the existing head/stratum report. Validation
callers and future comparison/monitor code can reuse the same arithmetic
without importing a training framework. Existing study runs without a report
schema version continue to verify against the original schema 1, without
rewriting their evidence or weakening exact report/hash checks.

## Re-evaluate existing evidence

From `xai_test/`, choose a destination that does not exist:

```powershell
python -m SpCas9.spcas9_eval --predictions SpCas9/v2_eval/predictions.csv --output-dir SpCas9/ngg_eval_new
```

The default sidecar is `predictions_checksum.json` beside the input CSV;
`--prediction-sidecar` can specify another path. The checkpoint, published
split and pinned source spreadsheet are still required to verify provenance.
All splits use independent input pins, including validation; self-issued
sidecars cannot authorize substituting Test rows. Saved CSV/checkpoint/split
digests and row/metadata/label alignment must match. Re-evaluation preserves
all prediction rows and additional metadata, writes a new CSV, sidecar and
`results.json`, and records input/output prediction digests, input sidecar hash,
checkpoint hash, consumed split/dataset identity, source snapshot, interpreter
and installed library versions. Existing destinations and source overlap are
refused. Original evidence is never overwritten.

For fresh inference on validation using an existing checkpoint:

```powershell
python -m SpCas9.spcas9_eval --split val --checkpoint SpCas9/v2_train/9hydra_v2.keras --output-dir SpCas9/ngg_val_new
```

`--split test` is restricted to independently pinned historical v2 artifacts.
An alternative checkpoint or Test prediction file is refused before inference
or report creation. Candidate Test evaluation still requires the study's
finalist/freeze stage; this command supplies no bypass and performs no training.

The historical v2 replay produced `SpCas9/ngg_eval_issue82_final/results.json`:
5,186 rows, equal-head NGG MAE 0.0916146510360614 and mean NGG Spearman
0.571974207844954. This recomputes already inspected stored predictions; it is
neither a new candidate evaluation nor evidence of performance improvement.
