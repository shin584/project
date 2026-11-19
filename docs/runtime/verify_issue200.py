"""Measure production Cas12a XAI against frozen #195/#193 cases and limits.

py -3.13 -X utf8 docs/runtime/verify_issue200.py NEW_OUTPUT_DIRECTORY
Exit 1 preserves numerical failures; display acceptance is measured separately.
Does not modify or replace #205 evidence. SHAP always bypasses the disk cache.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(output: Path) -> int:
    from non_nt_cas12a_predict import NonNTCas12aPredictor
    from non_nt_runtime import cas12a_derive, one_hot
    from predict import NON_NT_REGISTRY
    from xai import explain
    from verify_cas12a_ig import cosine, feature_vjp

    output.mkdir(parents=True, exist_ok=False)
    sources = ["docs/runtime/contract.json", "docs/runtime/cas12a-ig-contract.json",
               "docs/runtime/verify_issue200.py", "docs/runtime/verify_cas12a_ig.py",
               "non_nt_cas12a_predict.py", "non_nt_runtime.py", "predict.py",
               "cas12a/continuous.py", "cas12a/features.py", "model_source.py",
               "xai/explain.py", "xai/units.py", "xai/identity.py", "xai/quality.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    with zipfile.ZipFile(output / "measured-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(ROOT / source, source)
    (output / "registration.json").write_text(json.dumps(
        {"command": sys.argv, "source_sha256": hashes}, indent=2), encoding="utf-8")
    contract = json.loads((ROOT / "docs/runtime/contract.json").read_text("utf-8"))
    selected = json.loads((ROOT / "docs/runtime/cas12a-ig-contract.json").read_text("utf-8"))
    criteria = contract["criteria"]
    predictor = NonNTCas12aPredictor(NON_NT_REGISTRY[("A", "Cas12a")])
    handle = predictor.explain_handle("Cas12a")
    report = {"issue": 200, "identity": predictor.explanation_identity().as_dict(),
              "source_sha256": hashes, "checks": [], "scope": __doc__}
    arrays = {}
    checks = report["checks"]
    started = time.monotonic()

    def check(name, value, bound, passed):
        checks.append({"check": name, "measured": value, "bound": bound, "passed": bool(passed)})

    def compare(prefix, a, b, correlation, difference=None, signs=False):
        r = float(np.corrcoef(a, b)[0, 1])
        check(prefix + "/pearson", r, correlation, np.isfinite(r) and r >= correlation)
        if difference is not None:
            error = float(np.max(np.abs(a - b)))
            check(prefix + "/max_abs_diff", error, difference, error <= difference)
        if signs:
            strong = np.abs(b) >= .1 * np.max(np.abs(b))
            agreement = float(np.mean(np.sign(a[strong]) == np.sign(b[strong])))
            check(prefix + "/sign_agreement", agreement, 1, agreement == 1)

    original_forward = predictor.runtime.forward
    for case in selected["cases"]:
        window, prefix = case["window"], case["id"]
        x, z = one_hot(window)[None], cas12a_derive(window)[None]
        arrays[prefix + "/input"] = x
        arrays[prefix + "/features"] = z
        for alpha in selected["settings"]["alphas"]:
            soft = alpha * x
            features = np.concatenate([alpha*z[:, :2], alpha**2*z[:, 2:]], axis=1)
            original = original_forward([soft, features])
            y, gradient = handle.ig_grad_fn([soft])
            raw = handle.raw_fn(soft)
            arrays[f"{prefix}/forward/{alpha}"] = [*original, *y, *raw]
            arrays[f"{prefix}/gradient/{alpha}"] = gradient[0]
            error = float(np.max(np.abs(np.array([y, raw]) - original)))
            bound = selected["criteria"]["function_match"]["max_abs_error"]
            check(f"{prefix}/function/{alpha}", error, bound, error <= bound)
            if alpha in selected["settings"]["gradient_alphas"]:
                _, partial = predictor.runtime.gradients([soft, features])
                chain = partial[0] + feature_vjp(soft, partial[1])
                arrays[f"{prefix}/chain/{alpha}"] = chain
                error = float(np.max(np.abs(gradient[0] - chain)))
                bound = selected["criteria"]["chain_rule"]["max_abs_error"]
                check(f"{prefix}/chain/{alpha}/max_abs", error, bound, error <= bound)
                similarity = cosine(gradient[0], chain)
                bound = selected["criteria"]["chain_rule"]["min_cosine"]
                check(f"{prefix}/chain/{alpha}/cosine", similarity, bound, similarity >= bound)
        igs = {}
        for steps in selected["settings"]["ig_steps"]:
            result = explain.integrated_gradients(handle, window, steps=steps)
            igs[steps] = result.attribution
            arrays[f"{prefix}/ig/{steps}"] = result.attribution
            arrays[f"{prefix}/ig/{steps}/terms"] = [result.pred, result.baseline_pred, result.completeness_error]
            bound = selected["criteria"]["completeness"]["max_abs_error_raw"]
            check(f"{prefix}/ig/{steps}/completeness", result.completeness_error, bound,
                  result.completeness_error <= bound)
        compare(prefix + "/ig_resolution", igs[32], igs[256],
                selected["criteria"]["resolution"]["min_pearson"],
                .05 * float(np.max(np.abs(igs[256]))), signs=True)

        # Inspect the actual production model boundary for EVERY SHAP mutation,
        # including PAM changes. Compare all 354 features with the training
        # extractor, and outputs with M(x, training_features(x)).
        audit = {"rows": 0, "features_max_abs": 0.0, "function_max_abs": 0.0, "calls": 0}

        evidence_arrays = arrays

        def audited_forward(arrays, head=0, batch_size=64, on_batch=None):
            batch_x, features = arrays
            sequences = ["".join("ACGT"[i] for i in row.argmax(axis=1)) for row in batch_x]
            expected_features = np.stack([cas12a_derive(seq) for seq in sequences])
            audit["features_max_abs"] = max(audit["features_max_abs"], float(np.max(np.abs(features - expected_features))))
            y = original_forward(arrays, head=head, batch_size=batch_size, on_batch=on_batch)
            reference = original_forward([batch_x, expected_features], head=head, batch_size=batch_size)
            audit["function_max_abs"] = max(audit["function_max_abs"], float(np.max(np.abs(y - reference))))
            evidence_arrays[f"{prefix}/shap_inputs/{audit['calls']}"] = batch_x
            audit["rows"] += len(batch_x)
            audit["calls"] += 1
            return y

        predictor.runtime.forward = audited_forward
        shaps = []
        try:
            for label, seed, samples, backgrounds in [
                    ("seed0", 0, 64, 4), ("seed1", 1, 64, 4), ("seed2", 2, 64, 4),
                    ("repeat", 0, 64, 4), ("finer", 0, 512, 32)]:
                result = explain.kernel_shap(handle, window, nsamples=samples,
                                             n_background=backgrounds, seed=seed, use_disk_cache=False)
                shaps.append(result)
                arrays[f"{prefix}/shap/{label}"] = result.values
                arrays[f"{prefix}/shap/{label}/terms"] = [result.pred, result.base_value, result.additivity_error]
                bound = criteria["shap_additivity"]["max_abs_error"]["cas12a"]
                check(f"{prefix}/shap/{label}/additivity", result.additivity_error, bound,
                      result.additivity_error <= bound)
        finally:
            predictor.runtime.forward = original_forward
        report.setdefault("shap_mutation_audits", {})[prefix] = audit
        bound = selected["criteria"]["feature_parity"]["max_abs_error"]
        check(prefix + "/shap_all_mutations/features", audit["features_max_abs"], bound, audit["features_max_abs"] <= bound)
        bound = criteria["function_match"]["max_abs_diff"]
        check(prefix + "/shap_all_mutations/function", audit["function_max_abs"], bound, audit["function_max_abs"] <= bound)
        bound = criteria["shap_repeat_stability"]["same_seed_determinism"]["max_abs_diff"]
        error = float(np.max(np.abs(shaps[0].values - shaps[3].values)))
        check(prefix + "/shap_determinism", error, bound, error <= bound)
        for a, b in itertools.combinations(range(3), 2):
            compare(f"{prefix}/shap_cross_seed/{a}-{b}", shaps[a].values, shaps[b].values,
                    criteria["shap_repeat_stability"]["cross_seed"]["min_pairwise_pearson_r"],
                    .25 * float(np.max(np.abs(shaps[0].values))))
        compare(prefix + "/shap_default_vs_finer", shaps[0].values, shaps[4].values,
                criteria["shap_repeat_stability"]["default_vs_finer"]["min_pearson_r"])
        check(prefix + "/finite", True, "all arrays finite", all(np.isfinite(a).all() for a in arrays.values()))
        print(prefix, "completed", flush=True)
    unchanged = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    check("sources_unchanged_during_measurement", unchanged, True, unchanged)
    report["seconds"] = time.monotonic() - started
    report["passed"] = sum(row["passed"] for row in checks)
    report["failed"] = len(checks) - report["passed"]
    report["verdict"] = "fail" if report["failed"] else "pass"
    report["skipped"] = 0
    np.savez_compressed(output / "raw.npz", **arrays)
    report["evidence_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                for name in ("raw.npz", "registration.json", "measured-source.zip")}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(report["verdict"], report["passed"], "passed", report["failed"], "failed", flush=True)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
