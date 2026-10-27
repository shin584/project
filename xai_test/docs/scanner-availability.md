# Scanner prediction availability (#80)

The repository-root `multi_cas_scanner_수정.py` returns a `PredictionResult`
from `ModelBundle.predict(site)`. Check `prediction.available` before consuming
`prediction.efficiency`:

- Available: a finite efficiency on the original model scale, with `reason=None`.
  Zero and negative linear outputs remain valid. Values are neither clipped nor
  normalized to a bounded range.
- Unavailable: `efficiency=None` and a nonempty `reason`. Unknown families,
  invalid guide inputs, missing models, inference exceptions and non-finite
  results cannot become numeric zero. Load errors retain their path and cause.

`MultiCasScanner.predict` stores the result on `site.prediction` and exposes its
efficiency through `site.raw_score`. Only available results receive the existing
`round(raw * gaussian_penalty(distance, sigma) * 100, 2)` final score. A failed
repeat prediction clears any previous score. Before prediction, sites are
unavailable with reason `Prediction not requested`.

`ScanResult.sorted_by_score()` returns available, scored sites only.
`build_score_dict()` retains every candidate so callers can show failure reasons:
available entries retain the existing `..._rankN` keys and per-family ranks;
unavailable entries use `..._unavailableN` keys with `rank`, `raw_score` and
`score_pct` all `None`. Every entry includes `available` and `reason`. Consumers
must filter `available` before sorting numeric scores or recommending sites.
`print_report()` ranks successful candidates and prints unavailable candidates
with their reasons outside the ranking. A zero score prints as `0.00%`.

Each family fails independently. A missing configured SaCas9 ensemble member
makes SaCas9 unavailable instead of silently changing the ensemble membership;
healthy families remain usable. Complete ensembles retain their existing mean.

The approved scanner request-to-report seam is covered in
`xai_test/test_multi_cas_scanner.py`, including external inference failures and
disposable saved Keras models. No models are trained by these checks. This
ticket establishes availability; the 30bp/named-head v2 adapter and actual stored
v2 checkpoint parity belong to #81.
