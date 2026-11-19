# Preserve the provenance of each runtime verification bundle

For #205, freeze the run identity, contract, relevant code, models and settings
before measurement, and link each stage's results and retained raw artifacts by
checksum. Keep the existing #193 evidence and failures as historical records,
and create a separate bundle for the repaired verification: adding the current
contract hash while assembling old results cannot establish their provenance.
This accepts retaining the raw evidence and rerunning the stages instead of
reusing untraceable measurements; it prevents later reports from silently
combining incompatible results.

Accepted in Q1 of the [#205 design interview](../planning/issue205-interview.md).
This records a design decision, not completed implementation or a passing run.
