# #199 isolated SpCas9 XAI evidence

This is a new #199 run, not a replacement or reconstruction of any #205 bundle.
The source snapshot, frozen contract copy and raw arrays belong to this run.
`report.json` records their hashes, measurement scope and failed numerical checks.

- Numerical checks: **671 passed / 269 failed / 0 skipped**, exit 1.
- Real WebView2/child-process checks: **12 passed / 0 failed / 0 skipped**,
  separately recorded under `window/`.
- All nine heads × four frozen cases: function match, finite gradients,
  IG32/256 and SHAP64/4 seeds 0/1/2/repeat.
- The SHAP512/32 comparison covers the registered SpCas9-NG head only (four cases).
  It does not establish finer-setting agreement for the other eight heads.

See [implementation and commands](../../issue199-implementation.md) for the
remaining numerical failures, scope, display policy and the type guard added
after the measurement. Do not interpret successful graph rendering or fixed-seed
repeatability as numerical stability or biological causality.
