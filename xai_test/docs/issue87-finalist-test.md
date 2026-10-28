# #87: finalist freeze and guarded Test

`StudyProtocol.freeze_selection(validation_dir, stored_checkpoint=...)` verifies
the existing #79 fixed execution, all six completed runs, and the stored-v2
validation predictions before writing `STUDY/finalist/freeze.json` and its
checksum. It recomputes both validation comparisons from verified prediction
bytes. Both must pass; an unavailable supplementary CI does not block a pass.
The representative remains candidate seed42.

The freeze records the recipe, acceptance and CI settings, source/environment,
protocol/configuration/input identities, six checkpoint hashes and evidence
identities, the stored-v2 checkpoint and validation identities, and the passing
validation comparisons. Repeating freeze never overwrites it. Existing #51
studies cannot use this stage without the fixed #79 execution and complete
evidence.

`StudyProtocol.evaluate_test(fresh_destination, backend=...)` revalidates the
freeze and its original evidence before inference. It scores exactly the six
frozen models and stored v2 against identical published Test rows. Verification
mode requires a disposable backend double. Full training remains blocked by the
separate final readiness work; implementing this boundary does not authorize an
experiment, checkpoint replacement or deployment.

Disposable studies may pass `verification_stored_checkpoint` to
`StudyProtocol.plan` to pin their fixture's stored model. This input is accepted
only in verification mode; full executions retain the published stored-v2 pin.

Results retain all seven predictions, their hashes, the full finalist identity,
both comparisons, and supplementary protospacer-bootstrap CIs. Test applies the
same improvement and protection thresholds as validation. A failed or incomplete
Test decision keeps stored v2; it cannot change the recipe, thresholds or seed.
Test has been inspected historically and is not new external confirmation.

`StudyProtocol.test_report(destination)` verifies saved results and recomputes
the two decisions without loading a backend. Original checkpoints, validation
evidence and split files are read-only. New outputs require a fresh destination
outside their source paths. Failed inference may leave partial outputs in that
destination but cannot produce a sealed eligible report.

For #87, run only its fixture workflow and the existing acceptance/study guards:

```powershell
cd xai_test
python -m pytest test_spcas9_finalist.py test_spcas9_acceptance.py test_spcas9_study.py::test_inputs_are_rechecked_before_backend_use_and_test_prediction_remains_closed -q -p no:cacheprovider
```

These checks use disposable fixtures and backend doubles. They do not train
models or evaluate real candidate Test predictions. The repository-wide suite
is deliberately omitted as requested for this ticket.
