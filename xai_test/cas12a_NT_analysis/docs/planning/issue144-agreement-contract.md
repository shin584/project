# #144 model-free IG–ISM agreement contract

Scope: CPU implementation of #144 from `fbfaf2a`. Read committed #140–#142
artifacts; never load a model, execute GPU work or change existing artifacts.
Scientific freeze/validation remains #146 and actual Test agreement remains
#148. This contract does not approve real IG quality thresholds.

Public boundaries: `sequence_agreement` computes one sequence's diagnostics;
`read_agreement(study, run_id)` joins the existing checked readers by source ID;
`save_agreement(study, run_id, directory)` writes a new immutable export;
`load_agreement(directory, checksum=...)` verifies and recomputes that export.
`compare_saved(predictions, ig, ism)` also accepts the existing checked-reader
reports for downstream callers. It verifies joins and rebuilds the population
plan; committed bytes and method quality must first be checked by those readers.
Exports preserve a normalized input snapshot and derived JSON, with a trusted
returned manifest SHA256. No ledger job or GPU reservation is needed for reading.
The export never writes into the study; reject resolved output paths within
the study root. New output directories must be empty. Record NumPy/SciPy/Python
versions in the export; replay uses the saved protocol and verifies exact
canonical results, rejecting a numerically different recomputation.

Join requires identical full run identities (source, partition, model,
tokenizer/config, prediction/IG/ISM protocol, code), run IDs, source sequences,
sequence/prediction hashes, actual token IDs/masks/spans and WT predictions.
Upstream readers verify committed bytes, method protocols, plans and quality.
Record complete method coverage and missing/failed IDs. The immutable IG plan
defines the expected union, statistical IDs and additional case IDs. Full and
random-subset summaries use only `statistics_ids`; case-only and validation
probe rows have separate descriptive summaries. An absent plan is an error.
Missing/failed explanations retain a row and reason; they block completion,
but normal undefined correlations and low rho do not. An unresolved job also
blocks completion. Never silently intersect IDs or extend the denominator.

For each actual DNA span, retain signed IG mass and `abs(mass)/span_length`.
The ISM input is raw mutant-minus-WT delta in fraction units on 34 x A/C/G/T.
Eligibility is base-valid AND non-WT, additionally excluding PAM-breaking
alternatives at invariant T positions 4,5,6 for `exclude_pam_breaking`.
For each base, save the four-alternative mask, eligible count, mean absolute
delta or null+reason. Token ISM mean averages only valid base means; span-sum
adds those base means. Preserve valid base coordinates, counts and actual span
length. No valid alternatives/base gives null, never a fabricated zero.
PAM exclusion leaves t1 with 4bp and t2 with 5bp ISM support; their IG still
describes their full 6bp spans. Do not project/reallocate IG onto remaining bases.

For both PAM versions and each token set (`five_6mer` t1–t5; `nine_token`
t1–t9), save two per-sequence Spearmans: density (abs IG / actual length versus
token mean ISM) and mass (abs IG versus span-sum ISM). Primary is exclusively
`five_6mer/all_nonWT/density`. Midranks handle exact ties. Use complete paired
support: a missing/nonfinite token nulls that correlation with a reason;
fewer than two tokens or either constant axis also yields null+reason.
Store n_total, n_valid and token IDs. Population summaries preserve sorted
rho distributions, median, total/valid counts and exclusion reasons. Also
correlate per-token population mean profiles, preserving each token's paired
denominator and requiring complete profiles. Case-only rows cannot enter these
population profiles. No bootstrap or significance tests are added.

For each PAM version's nine-token density axes, independently partition
(a) 6-mer t1–t5 versus 1bp t6–t9 and (b) pure 3′ flank t7–t9 versus remainder.
Use global midranks X,Y; rank the nine values once, never within each group.
Save group IDs/counts, mean ranks and first-minus-second mean-rank differences.
Use population covariance/variance (divide by N):

    W = sum_g p_g mean_g((X-mean_g X)*(Y-mean_g Y))
    B = sum_g p_g (mean_g X-mean X)*(mean_g Y-mean Y)
    D = std_population(X)*std_population(Y)
    rho = Cov_population(X,Y)/D = W/D + B/D

Save raw covariance terms, common D, normalized within/between contributions,
sum, residual and an arithmetic check (atol=1e-12, rtol=1e-12). This tolerance
checks rank arithmetic only, not model quality. Constant/missing axes preserve
null+reason; do not normalize by group standard deviations. The partitions are
separate diagnostics, not additive across partitions, a confound fraction of
rho_9-rho_5, or causal effects.

Exports bind schema/protocol, source input digest, result digest and exact file
set. Load checks the independently retained manifest checksum, verifies bytes,
then recomputes every primary/auxiliary value from the saved inputs and compares
canonical JSON. Transfer the checksum separately. Export integrity preserves
the checked-reader provenance at export time; it does not independently prove
remote study history or scientific validity. Reject corruption and overwrite.

The user confirmed the sequence calculation, checked artifact join and immutable
export/recalculation test boundaries with `y`. Use independent worked examples
and SciPy for ties/constant axes, PAM exclusion/no valid bases, group-only
agreement and covariance sums; shuffle IDs, test mismatches and subset cases,
and reload in a subprocess that forbids torch/Transformers imports.

## Independent design review

Standards: 0 blockers. Enforce the read-only study boundary with resolved paths,
preserve original base-valid masks and bind calculation environment versions.
Spec: 0 blockers. Use the final accepted IG resolution, actual WT token records,
global ranks with a shared normalization, and all expected-ID denominators.
