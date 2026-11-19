"""Measure #199 independently of the preserved #205 bundles (exit 1 on quality failure).

python -X utf8 docs/runtime/verify_issue199.py OUTPUT_DIRECTORY
Frozen cases and tolerances come from contract.json. All nine heads: function
match, finiteness, IG32/256, SHAP64/4 seeds0/1/2/repeat. The registered #193 head
(SpCas9-NG) also gets the frozen SHAP512/32 comparison on all four cases.
No disk cache. This is UI integration evidence, not a replacement #205 bundle.
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
    from non_nt_predict import NonNTSpCas9Predictor
    from non_nt_runtime import one_hot
    from pam_scanner import SPCAS9_VARIANTS
    from predict import NON_NT_REGISTRY
    from xai import explain

    output.mkdir(parents=True, exist_ok=False)
    contract_path = ROOT / "docs/runtime/contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    criteria = contract["criteria"]
    predictor = NonNTSpCas9Predictor(NON_NT_REGISTRY[("A", "SpCas9")])
    sources = ["docs/runtime/contract.json", "docs/runtime/verify_issue199.py",
               "non_nt_predict.py", "non_nt_runtime.py", "predict.py", "xai/explain.py",
               "xai/units.py", "xai/identity.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    (output / "contract.json").write_bytes(contract_path.read_bytes())
    with zipfile.ZipFile(output / "measured-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(ROOT / source, source)
    report = {"issue": 199, "identity": predictor.explanation_identity().as_dict(),
              "source_sha256": hashes, "checks": [], "scope": __doc__}
    arrays = {}
    checks = report["checks"]
    started = time.monotonic()

    def check(name, value, bound, passed):
        checks.append({"check": name, "measured": value, "bound": bound,
                       "passed": bool(passed)})

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

    for case in contract["cases"]["spcas9"]:
        window = case["window"]
        x = one_hot(window)[None]
        original = predictor.runtime.forward_all([x])[0]
        arrays[case["id"] + "/original"] = original
        for index, head in enumerate(SPCAS9_VARIANTS):
            prefix = f"{case['id']}/{head}"
            handle = predictor.explain_handle(head)
            for alpha in (0, .5, 1):
                y, gradients = handle.ig_grad_fn([x * alpha])
                raw = handle.raw_fn(x * alpha)
                arrays[f"{prefix}/forward/{alpha}"] = [*y, *raw]
                arrays[f"{prefix}/gradient/{alpha}"] = gradients[0]
                error = float(np.max(np.abs(y - raw)))
                check(f"{prefix}/function/{alpha}", error, criteria["function_match"]["max_abs_diff"],
                      error <= criteria["function_match"]["max_abs_diff"])
                check(f"{prefix}/gradient_finite/{alpha}", bool(np.isfinite(gradients[0]).all()),
                      "all finite", np.isfinite([*y, *raw]).all() and np.isfinite(gradients[0]).all())
            error = float(abs(handle.raw_fn(x)[0] - original[index]))
            check(prefix + "/prediction_function", error, criteria["function_match"]["max_abs_diff"],
                  error <= criteria["function_match"]["max_abs_diff"])
            igs = {}
            for steps in (32, 256):
                ig = explain.integrated_gradients(handle, window, steps=steps)
                igs[steps] = ig.attribution
                arrays[f"{prefix}/ig/{steps}"] = ig.attribution
                arrays[f"{prefix}/ig/{steps}/terms"] = [ig.pred, ig.baseline_pred, ig.completeness_error]
                bound = criteria["ig_completeness"]["max_abs_error"]["spcas9"]
                check(f"{prefix}/ig_completeness/{steps}", ig.completeness_error, bound,
                      ig.completeness_error <= bound)
                finite = np.isfinite([ig.pred, ig.baseline_pred, ig.completeness_error]).all() and np.isfinite(ig.attribution).all()
                check(f"{prefix}/ig_finite/{steps}", bool(finite), "all finite", finite)
            compare(prefix + "/ig_resolution", igs[32], igs[256],
                    criteria["ig_resolution_stability"]["min_pearson_r"],
                    .05 * float(np.max(np.abs(igs[256]))), signs=True)
            shaps = []
            for label, seed, samples, backgrounds in [
                    ("seed0", 0, 64, 4), ("seed1", 1, 64, 4),
                    ("seed2", 2, 64, 4), ("repeat", 0, 64, 4)]:
                result = explain.kernel_shap(handle, window, nsamples=samples,
                                             n_background=backgrounds, seed=seed,
                                             use_disk_cache=False)
                shaps.append(result)
                arrays[f"{prefix}/shap/{label}"] = result.values
                arrays[f"{prefix}/shap/{label}/terms"] = [result.pred, result.base_value, result.additivity_error]
                finite = np.isfinite(result.values).all() and np.isfinite([result.pred, result.base_value, result.additivity_error]).all()
                check(f"{prefix}/shap_finite/{label}", bool(finite), "all finite", finite)
            default = shaps[0]
            bound = criteria["shap_additivity"]["max_abs_error"]["spcas9"]
            check(prefix + "/shap_additivity", default.additivity_error, bound,
                  default.additivity_error <= bound)
            bound = criteria["shap_repeat_stability"]["same_seed_determinism"]["max_abs_diff"]
            error = float(np.max(np.abs(default.values - shaps[3].values)))
            check(prefix + "/shap_determinism", error, bound, error <= bound)
            cross = criteria["shap_repeat_stability"]["cross_seed"]
            for a, b in itertools.combinations(range(3), 2):
                compare(f"{prefix}/shap_cross_seed/{a}-{b}", shaps[a].values, shaps[b].values,
                        cross["min_pairwise_pearson_r"], .25 * float(np.max(np.abs(default.values))))
            if index == contract["models"]["spcas9_non_nt"]["verified_head"]["index"]:
                heavy = explain.kernel_shap(handle, window, nsamples=512, n_background=32,
                                            seed=0, use_disk_cache=False)
                arrays[prefix + "/shap/finer"] = heavy.values
                compare(prefix + "/shap_default_vs_finer", default.values, heavy.values,
                        criteria["shap_repeat_stability"]["default_vs_finer"]["min_pearson_r"])
            print(prefix, "completed", flush=True)
    report["sources_unchanged_during_measurement"] = all(
        hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    report["seconds"] = time.monotonic() - started
    report["passed"] = sum(row["passed"] for row in checks)
    report["failed"] = len(checks) - report["passed"]
    report["verdict"] = "fail" if report["failed"] else "pass"
    report["skipped"] = 0
    np.savez_compressed(output / "raw.npz", **arrays)
    report["raw_sha256"] = hashlib.sha256((output / "raw.npz").read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(report["verdict"], report["passed"], "passed", report["failed"], "failed", flush=True)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
