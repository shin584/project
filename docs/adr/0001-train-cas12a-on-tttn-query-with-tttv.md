# Cas12a trains on the TTTN library and is queried with TTTV

The Cas12a model's training data is the Kim 2018 library, and that library's PAM
is `TTTN`, not the canonical `TTTV`: of its 15,000 usable rows, the fourth PAM
base is C in 5,068, G in 4,897, **T in 3,635** and A in 1,400. The scanner, on
the other hand, searches genomic input for `TTTV` (`e_system/pam_scanner.py`),
because `TTTT` is reported as a poor PAM for AsCas12a. So the model is asked
about a narrower population than the one it was fitted on. This is Kim 2018's
own design, not a bug, and we are keeping it.

The alternative was to drop the `TTTT` rows and train on the `TTTV` subset
alone, matching training population to query population exactly. That costs
**24% of the data** (3,635 rows, leaving 11,365) on a dataset where the current
architecture already performs at ridge-regression level — see
`cas12a/docs/cas12a-model-review.md` §2. Spending a quarter of the rows to
remove a mismatch we have not measured the cost of is the worse trade at this
stage: the model's problem today is that it is under-validated and
under-trained, not that it has seen too many PAMs.

## Consequences

- `cas12a/data.py` keeps every row whose sequence and target parse, and does not
  filter on the PAM. Code that assumes "every loaded row is a TTTV site" is
  wrong; `cas12a/test/test_data.py` fixes only `TTT` at positions 4-6 as
  invariant, leaving position 7 free.
- The model is evaluated on the same TTTN population it is trained on, so its
  reported metrics do not describe TTTV-only performance. No number we have
  quantifies the gap. A TTTV-only holdout evaluation would, and is not built
  here; it is the cheap way to revisit this decision later, well before
  retraining on a filtered library.
- Reversing this is expensive: it means re-deriving the training set and
  re-running the 5-fold study, so the published numbers from before and after
  would not be comparable.
- The scanner's `TTTV` query stays as it is. Widening the scan to `TTTN` to
  match the training data would make the two consistent, but at the cost of
  proposing target sites the literature expects to cut poorly — a worse failure
  than a population mismatch, because it is visible to the user as a bad
  recommendation.

## Considered Options

- **Train on the TTTV subset only**: rejected. 24% data loss for an unmeasured
  gain, on a model that is currently data-hungry.
- **Widen the scanner to TTTN**: rejected. It would surface `TTTT` sites as
  candidates, which is a user-facing regression.
- **Keep both and report TTTV/TTTN metrics separately**: deferred, not rejected.
  This is the measurement that should precede any reopening of this ADR, and it
  needs no retraining — only a stratified evaluation.
