# Scanner 30bp / named-head v2 adapter (#81)

The repository-root `multi_cas_scanner_수정.py` accepts the stored 9-Hydra v2
checkpoint through `ModelBundle(path_A=..., ...)`. Models load with
`compile=False`; inference does not require training losses or optimizer state.

`scan_pam` preserves `CandidateSite.guide_seq` as the 20nt SpCas9 protospacer.
It extracts `model_input_seq` from the same oriented `search_seq` as
`search_seq[guide_start - 4:pam_end + 3]`: four left-flank bases, 20 protospacer
bases, three PAM bases and three right-flank bases. Reverse-strand candidates
use the reverse complement before extraction. Existing strand-relative
`pam_start`, original-region `cut_pos`, discovery/distance filters and PAM
policy are preserved.

A discovered candidate with insufficient context or a non-ACGT window retains
its guide and a `model_input_reason`, but has no model input. Prediction returns
unavailable with that reason. It never pads or infers context from the guide.
Manually supplied candidates also require a complete valid window agreeing
with their guide and PAM. Region validation continues to allow `N`; an `N`
inside an SpCas9 model window makes that prediction unavailable.

`HYDRA_OUTPUT_LAYERS` explicitly maps the nine canonical protein names to the
saved output layer names. Only the v2 dictionary format is supported; legacy
arrays/lists are unavailable with a format error. Positional `hydra_head_idx`
selection has been removed. Each required output must be a real numeric
`(1, 1)` array with a finite value. Any invalid or absent head makes the whole
call unavailable with the offending protein/layer in its reason; no partial
head dictionary is returned.

Available SpCas9 predictions expose all nine original-scale values through
`PredictionResult.efficiencies_by_protein`. `build_score_dict` includes a copy
of that mapping plus `model_input_seq`. The existing scalar `efficiency` /
`raw_score` is the canonical SpCas9 head, preserving family ranking and the
existing `round(raw * gaussian_penalty(distance, sigma) * 100, 2)` policy. No
additional protein-ranking or variant-PAM policy is introduced. Zero and
negative values stay available. Unavailable entries have an empty mapping,
no numeric score/rank, and retain their reason in reports. Other model families
continue independently under the #80 contract.

## Verification

Run from the repository root with an interpreter supplying TensorFlow, NumPy
and pytest:

```powershell
python -m pytest xai_test/test_multi_cas_scanner.py -q -p no:cacheprovider
```

The actual-checkpoint test requires the original local file
`xai_test/SpCas9/v2_train/9hydra_v2.keras` with SHA-256
`ece5009215482d3fd2506688dc8e2423437abdcd28dfe10e4ac21c5bed3d19d6`.
It fails if that evidence is absent or changed. Synthetic complete windows
avoid reading candidate-study Test results. Independent A/C/G/T encoding feeds
the original model directly; scanner consumers use a saved/reloaded disposable
copy on both strands. All nine outputs must match at `rtol=1e-5`, `atol=1e-6`.
The test confirms the original checkpoint digest remains unchanged and writes
`parity.json` under its pytest temporary directory, including checkpoint,
source, region/window identifiers, environment versions, and per-head errors.

A retained execution record is in
[`evidence/issue81-scanner-v2-parity.json`](evidence/issue81-scanner-v2-parity.json).
This establishes scanner compatibility with stored v2, independently of #79's
training experiment or model adoption. No fit, partition change, checkpoint
replacement or earlier evaluation artifact overwrite is required.
