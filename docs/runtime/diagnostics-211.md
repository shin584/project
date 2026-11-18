# Fixed failure-location diagnostics (#211)

This implements an investigation producer, not official model remeasurement.
#212/#213 own official execution after integration. The historical #193 contract,
report, checkpoints and deployed float32 runtime remain unchanged.

Run the existing commands in [verification-206.md](verification-206.md), starting
with a **new empty** bundle and this additive protocol:

```powershell
python docs/runtime/verify_contract.py init --work <new-bundle> --run-id <new-id> --protocol docs/runtime/protocol-211.json
# Collect reference, verify, diagnose, RAM and independent recovery as documented.
python docs/runtime/verify_contract.py investigation --work <new-bundle>
python docs/runtime/assemble_verification.py --work <new-bundle> --output <new-summary.json>
```

Protocol 211 runs diagnostics and explicitly leaves candidate investigations
unverified. To run both producers, before `init` copy the registered candidate
protocol, add `"diagnostic_producer": 211` and copy the `diagnostic_protocol` from
`protocol-211.json`. Keep the candidate command, selection rules and inventories.
Pass that combined file with `--protocol`; the journaled CLI merges both outputs
without changing candidates or reusing another bundle's measurements.

Initialization freezes historical `verification.json` as
`snapshot/diagnostic-history.json`. After verify/diagnose, before any sweep,
`diagnostic-plan.json` fixes the union of historical and current failed checks,
including model, verified head, window, input, point and original check/step/value.
Coordinates cover the entire affected input gradient, not selected components.
Directional failures retain all 32 existing seed-193 directions and all inputs at
alpha 0.5, including their original float64 normalization of float32 random
draws (only model inputs and computation use float32). Historical reports did not retain individual direction indices, so
none are guessed or omitted. Current original arrays remain in the checksummed
core/diagnostic stage archives. No other interpolation points are searched.

Passing controls are fixed in advance: SpCas9 S3/alpha0.5/In_Seq and Cas12a
C2/x/Input_Seq. Both historical coordinate criteria must pass for each control;
the plan retains that evidence. A new failure at a control is preserved, never
replaced by a different control.

Every float32 location and control uses exactly 0.03, 0.01, 0.003, 0.001, 0.0003,
0.0001. Each observation retains the analytic gradient, numeric derivative,
residual, plus/minus outputs, base inputs and perturbation vectors. Adjacent-step
comparisons use the existing comparison lines. Coordinate comparisons retain
cosine >= 0.999 and relative L2 <= 0.02; directional comparisons retain median
<= 0.02 and maximum <= 0.1. The original step is explicitly retained even when
another step agrees better. Diagnostic stencil batching is fixed at 16 pairs;
its new numeric measurements do not overwrite the original required results.

Boundary arrays are actual pre-activations at those plus/minus stencil inputs,
restricted to ancestors of the selected head. Explicit ReLU/LeakyReLU inputs
are probed directly. Embedded activations use the same Keras layer configuration
and weights with a linear activation in a separate probe; no recurrent forward
or backward is reimplemented. Per-stencil crossing indices are replayed from
these arrays during assembly. A crossing is supporting evidence, not a unique
causal conclusion. Failure locations remain causally unresolved after this
descriptive sweep; passing controls without failures are ineligible for float64.

Only those unresolved locations enter one float64 attempt. The child clones
the serialized graph with float64 policies and exactly expands stored float32
weight values. It verifies graph/head/inference identity, variable names/shapes,
weight values and hashes, declared policies, actual layer input/output dtypes,
and actual output/gradient dtypes. Keras may omit a Bidirectional wrapper prefix
from copied variable paths; both paths are retained and correspondence follows
the identical serialized graph order, names, shapes and exact values. Assembly
also checks source weights against the frozen TensorFlow reference inventory.
An input-only double conversion does not qualify.

Float64 runs the same six steps plus only 1e-5 and 1e-6. It retains every
same-step residual comparison with float32 and adjacent-step agreement. Labels
distinguish precision-effect support, boundary-crossing evidence, mixed
possibilities and unresolved evidence; improvement alone never proves a unique
cause or turns an original failure into a pass.

Float32 work finishes for both models before the float64 clock starts. One
isolated child runs at a time in SpCas9/Cas12a order, with 300 seconds for model
load/qualification, 600 seconds for diagnosis, and 1800 seconds total. The parent
enforces these deadlines and starts no model after the total boundary. Each
completed step is atomically checkpointed under `float32-<model>/` or
`float64-<model>/`; completed and planned step counts survive a planned kill.
The final checksummed investigation archive retains all published partial
arrays, process commands/exits/logs, qualification and budget metadata. A
crashed process aborts the bundle; checkpoints remain for inspection, but
measurement cannot resume. Use a new bundle for all required measurements.

Q17 keeps required failure precedence. Unsupported/failed qualification is
`unavailable`, completed diagnosis without a unique cause is `unresolved`, a
planned deadline is `time_limited`, and no eligible location is
`not_applicable`. Nonfinite evidence is a required failure, including nonfinite
qualification. An unavailable/time-limited investigation means incomplete/nonzero
exit, not numerical success. Completed unresolved findings allow exit zero only
when all required checks pass. Missing location/step/raw/dtype/budget evidence
cannot be replaced by a passing marker array.
