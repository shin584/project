# NGG validation acceptance and supplementary intervals (#86)

From `xai_test/`, rebuild a controlled study's validation report without loading
a training backend:

```python
from pathlib import Path
from SpCas9.spcas9_study import StudyProtocol

study = StudyProtocol(Path("path/to/study"))
report = study.validation_report(
    Path("path/to/new-report-directory"),
    stored_predictions=Path("path/to/stored-v2-validation/predictions.csv"),
    # Defaults to predictions_checksum.json beside the supplied CSV:
    # stored_sidecar=Path("path/to/predictions_checksum.json"),
)
```

The destination must be new. `validation.json` and its SHA-256 sidecar preserve
the frozen protocol, execution configuration, source, environment, input and
checkpoint/prediction identities. Reconstruction records its own source and
environment separately. The stored-v2 sidecar must match the independently
pinned historical checkpoint and the same published validation split. Missing
or damaged required run/stored-v2 evidence produces an incomplete report with
reasons. Reconstruction uses saved predictions; it does not refit or select a
different checkpoint.

The paired comparison averages each head/population metric over seeds 42, 43
and 44 before taking an equal-weight mean over nine NGG heads. Relative MAE
compares these means, not seed-level ratios. The representative comparison uses
only candidate seed42 and the single stored-v2 model. Both comparisons require
at least 2% lower equal-head NGG MAE and at least +0.02 mean NGG Spearman. Every
head in NGG and non-NGG must stay within +5% MAE and -0.02 Spearman. Boundaries
are inclusive; a zero reference MAE requires zero candidate MAE. Missing required
seeds or undefined required metrics make the decision incomplete.

Per-run/head and pooled MAE, MSE, Spearman and Pearson remain available with row,
unique-30mer and protospacer counts. Pooled correlations are calculated on
concatenated measured head/row pairs; they are distinct from mean head
correlations. Full epoch histories and checkpoint selection records are retained.
New controlled runs also persist selected-checkpoint training predictions and
NGG diagnostics. Empty/undefined training groups retain their reasons and do not
participate in adoption. Earlier run formats remain readable, with unavailable
training diagnostics explicitly represented as `null` in the comparison report.

Supplementary 95% percentile intervals concern the paired comparison only:
relative change `(candidate_mean_mae - baseline_mean_mae) / baseline_mean_mae`
and the change in mean NGG Spearman. With seed 20261027, sample NGG protospacer
clusters with replacement, retaining every selected cluster's rows and
multiplicity. One draw is shared by both methods, all heads and all three seeds.
Recompute measured-row metrics, average seeds then heads, and accept only draws
where both summary changes are finite. Request 2,000 valid draws in at most
20,000 attempts. A zero bootstrap reference MAE leaves relative change undefined.
Report attempts, valid/invalid counts and quota/unavailable reasons. A short
quota produces no interval; it never changes the completed point decision or
eligibility for a subsequent finalist freeze. These intervals are conditional on
the fixed seeds and do not claim robustness to arbitrary training seeds.

Only the paired comparison estimates the ranking auxiliary term's contribution
under a common selection/stopping/LR policy. Comparing seed42 with stored v2 also
includes changes to checkpoint selection, stopping and LR scheduling.

Gate and bootstrap constants share the execution configuration's source; the
configuration also lists their behavioral fixtures. Full-fit readiness and
finalist/Test stages remain guarded pending #87/#88. A passing validation report
does not itself authorize candidate Test access, checkpoint replacement or
deployment. This implementation runs prediction fixtures and disposable fake
backends only; it does not report trained candidate performance.

Focused verification from the repository root:

```powershell
python -m pytest xai_test/test_spcas9_acceptance.py -q -p no:cacheprovider
```

Run the full suite from `xai_test/` so the legacy tests' relative checkpoint
paths resolve: `python -m pytest -q -p no:cacheprovider`. Retain normal CPU
thread settings for the historical Model B prediction-regression check; forcing
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` changes its pinned Spearman result.
