"""#195 fixed-case comparison. Run from the worktree root with a NEW output path:

python -X utf8 docs/runtime/verify_cas12a_ig.py --output <new-directory>

Reads the preregistered contract; preserves raw arrays, registration and checks.
Exit 1 means numerical quality failed, including when adoption is feasible.
This does not replace or modify the historical #193 dual-input measurements.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/runtime/cas12a-ig-contract.json"
SOURCES = (
    "docs/runtime/cas12a-ig-contract.json", "docs/runtime/contract.json",
    "docs/runtime/verify_cas12a_ig.py", "non_nt_runtime.py",
    "cas12a/features.py", "cas12a/continuous.py", "predict.py",
    "model_source.py", "xai/explain.py", "xai/identity.py",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_vjp(x, weights):
    """Independent analytic derivative of weighted GC sums / adjacent products.

    Deliberately computes the derivative directly, without autograd or the
    continuous feature implementation, so the chain-rule comparison can detect
    a missing/transposed feature term.
    """
    out = np.zeros_like(x, dtype=np.float64)
    out[:, 8:31, 1:3] += weights[:, 0, None, None] / 23
    out[:, 23:31, 1:3] += weights[:, 1, None, None] / 8
    for i in range(22):
        for a in range(4):
            for b in range(4):
                w = weights[:, 2 + 16*i + 4*a + b]
                out[:, 8+i, a] += w * x[:, 9+i, b]
                out[:, 9+i, b] += w * x[:, 8+i, a]
    return out


def cosine(a, b):
    a, b = np.ravel(a).astype(float), np.ravel(b).astype(float)
    scale = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a.dot(b)/scale) if scale else float(np.array_equal(a, b))


def finite_difference(function, inputs, index, step):
    """Central coordinate differences, batching all perturbations via forward."""
    n = inputs[index].size
    plus = [np.repeat(x, n, axis=0) for x in inputs]
    minus = [x.copy() for x in plus]
    rows = np.arange(n)
    plus[index].reshape(n, -1)[rows, rows] += step
    minus[index].reshape(n, -1)[rows, rows] -= step
    return ((function(plus).astype(float) - function(minus).astype(float)) /
            (2*step)).reshape(inputs[index].shape)


def measure(contract):
    import torch
    import non_nt_runtime as rt
    from cas12a.continuous import derive_features
    from xai.explain import integrated_gradients

    checks, arrays = [], {}
    criteria = contract["criteria"]

    def check(name, case, passed, **values):
        checks.append(dict(name=name, case=case, passed=bool(passed), **values))

    def error_check(name, case, actual, expected, bound):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        check(name, case, np.isfinite(error) and error <= bound,
              max_abs_error=error, bound=bound)

    sequences = [c["window"] for c in contract["cases"]]
    sequences += [(a+b)*17 for a in "ACGT" for b in "ACGT"]
    first = sequences[0]
    sequences += [first[:i] + "N" + first[i+1:] for i in range(34)]
    encoded = np.stack([rt.one_hot(s) for s in sequences])
    actual = derive_features(torch.from_numpy(encoded)).numpy()
    expected = np.stack([rt.cas12a_derive(s) for s in sequences])
    arrays["features/actual"], arrays["features/reference"] = actual, expected
    error_check("feature_parity", "54 sequences x 354 features", actual, expected,
                criteria["feature_parity"]["max_abs_error"])

    soft = (((np.arange(136)*7) % 19 + 1)/20).reshape(1, 34, 4)
    weights = ((np.arange(354) % 17 - 8)/17)[None]
    tensor = torch.tensor(soft, requires_grad=True)
    loss = (derive_features(tensor)*torch.tensor(weights)).sum()
    grad, = torch.autograd.grad(loss, tensor)
    analytic = feature_vjp(soft, weights)
    fd = finite_difference(
        lambda xs: (derive_features(torch.tensor(xs[0])).numpy()*weights).sum(axis=1),
        [soft], 0, contract["settings"]["fd_step_feature"])
    arrays["soft/input"], arrays["soft/weights"] = soft, weights
    arrays["soft/autograd"], arrays["soft/analytic"], arrays["soft/fd"] = grad.numpy(), analytic, fd
    for name, reference in (("analytic", analytic), ("fd", fd)):
        error_check("feature_soft_vjp", name, grad.numpy(), reference,
                    criteria["feature_soft_vjp"]["max_abs_error"])

    runtime = rt.NonNTRuntime.load("cas12a")
    checkpoint = ROOT / contract["checkpoint"]["path"]
    if digest(checkpoint) != contract["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint differs from preregistration")
    versions = rt.runtime_versions()
    frozen_runtime = json.loads((ROOT / "docs/runtime/contract.json").read_text("utf-8"))
    for name, version in frozen_runtime["runtime"]["versions"].items():
        if versions[name] != version:
            raise ValueError(f"Unregistered runtime {name}: {versions[name]} != {version}")
    handles = {p: rt.cas12a_explain_handle(runtime, path=p) for p in ("composite", "dual")}
    for path, handle in handles.items():
        if handle.model_key.protocol != contract["protocols"][path]:
            raise ValueError(f"Protocol differs from preregistration: {path}")

    for case in contract["cases"]:
        case_id = case["id"]
        x = rt.one_hot(case["window"])[None]
        z = rt.cas12a_derive(case["window"])[None]
        arrays[f"{case_id}/input"], arrays[f"{case_id}/features"] = x, z
        for alpha in contract["settings"]["alphas"]:
            soft_x = (alpha*x).astype(np.float32)
            derived = derive_features(torch.from_numpy(soft_x)).numpy()
            scaling = np.concatenate([alpha*z[:, :2], alpha**2*z[:, 2:]], axis=1)
            error_check("path_scaling", f"{case_id}/{alpha}", derived, scaling,
                        criteria["path_scaling"]["max_abs_error"])
            for path, handle in handles.items():
                inputs = [soft_x] if path == "composite" else [soft_x, alpha*z]
                model_inputs = [soft_x, derived if path == "composite" else alpha*z]
                y, gradients = handle.ig_grad_fn(inputs)
                reference = runtime.forward(model_inputs)
                key = f"{case_id}/{path}/{alpha}"
                arrays[key+"/y"] = y
                for i, gradient in enumerate(gradients):
                    arrays[key+f"/gradient/{i}"] = gradient
                error_check("function_match", key, y, reference,
                            criteria["function_match"]["max_abs_error"])
                if path == "composite":
                    error_check("raw_function_match", key, handle.raw_fn(soft_x), reference,
                                criteria["function_match"]["max_abs_error"])

                if alpha not in contract["settings"]["gradient_alphas"]:
                    continue
                if path == "composite":
                    _, partial = runtime.gradients(model_inputs)
                    expected_gradient = partial[0] + feature_vjp(soft_x, partial[1])
                    arrays[key+"/chain_rule"] = expected_gradient
                    error_check("chain_rule", key, gradients[0], expected_gradient,
                                criteria["chain_rule"]["max_abs_error"])
                    c = cosine(gradients[0], expected_gradient)
                    check("chain_rule_cosine", key, c >= criteria["chain_rule"]["min_cosine"],
                          cosine=c, bound=criteria["chain_rule"]["min_cosine"])
                    forward = lambda xs: handle.raw_fn(xs[0])
                else:
                    forward = runtime.forward
                for index, gradient in enumerate(gradients):
                    fd = finite_difference(forward, inputs, index,
                                           contract["settings"]["fd_step_model"])
                    arrays[key+f"/fd/{index}"] = fd
                    relative = float(np.linalg.norm(fd-gradient) /
                                     max(np.linalg.norm(gradient), 1e-12))
                    c = cosine(fd, gradient)
                    limits = criteria["model_coordinate_fd"]
                    check("model_coordinate_fd", key+f"/input{index}",
                          relative <= limits["max_relative_l2"] and c >= limits["min_cosine"],
                          relative_l2=relative, cosine=c, bounds=limits)

        for path, handle in handles.items():
            results = []
            for steps in contract["settings"]["ig_steps"]:
                result = integrated_gradients(handle, case["window"], steps=steps)
                results.append(result)
                key = f"{case_id}/{path}/ig{steps}"
                arrays[key] = result.attribution
                bound = criteria["completeness"]["max_abs_error_raw"]
                check("completeness", key, result.completeness_error <= bound,
                      error_raw=result.completeness_error, bound=bound,
                      prediction=result.pred, baseline=result.baseline_pred,
                      signed_input_totals=handle.input_totals.copy())
            low, high = (r.attribution for r in results)
            peak = float(np.max(np.abs(high)))
            difference = float(np.max(np.abs(low-high)))
            correlation = float(np.corrcoef(low, high)[0, 1]) if peak else float(np.array_equal(low, high))
            significant = np.abs(high) >= 0.1*peak
            signs = float(np.mean(np.sign(low[significant]) == np.sign(high[significant])))
            limits = criteria["resolution"]
            scaled = difference/peak if peak else difference
            check("resolution", f"{case_id}/{path}",
                  correlation >= limits["min_pearson"] and
                  scaled <= limits["max_difference_over_reference_peak"] and
                  signs >= limits["min_sign_agreement_on_10pct_peak"],
                  pearson=correlation, max_difference_over_peak=scaled,
                  sign_agreement=signs, bounds=limits)

    check("finiteness", "all raw arrays", all(np.isfinite(a).all() for a in arrays.values()))
    # No cross-path attribution similarity criterion: they explain different paths.
    feasibility = {"feature_parity", "feature_soft_vjp", "path_scaling", "function_match",
                   "raw_function_match", "chain_rule", "chain_rule_cosine", "finiteness"}
    feasible = all(c["passed"] for c in checks if c["name"] in feasibility)
    failures = [c for c in checks if not c["passed"]]
    return dict(
        issue=195, versions=versions, checkpoint_sha256=digest(checkpoint),
        tensorflow_imported="tensorflow" in sys.modules,
        identities={p: h.model_key.as_dict() for p, h in handles.items()},
        adoption_feasible=feasible, selected_path="composite" if feasible else None,
        numerical_outcome="fail" if failures else "pass",
        counts=dict(passed=len(checks)-len(failures), failed=len(failures), skipped=0),
        checks=checks,
    ), arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshot"
    snapshot.mkdir()
    hashes = {}
    for relative in SOURCES:
        source = ROOT / relative
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = digest(target)
    registration = dict(
        registered_at=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        executable=sys.executable, source_sha256=hashes,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
    )
    (output / "registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    try:
        contract = json.loads((snapshot / "docs/runtime/cas12a-ig-contract.json").read_text("utf-8"))
        report, arrays = measure(contract)
        # Reject source drift during the measurement; the report must describe
        # the bytes registered before it started.
        if any(digest(ROOT / relative) != expected for relative, expected in hashes.items()):
            raise ValueError("Source changed during measurement")
        np.savez_compressed(output / "arrays.npz", **arrays)
        report["evidence_sha256"] = {name: digest(output / name)
                                     for name in ("registration.json", "arrays.npz")}
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("counts", "adoption_feasible", "selected_path",
                                                "numerical_outcome")}))
        return 1 if report["numerical_outcome"] == "fail" else 0
    except Exception as error:
        (output / "error.json").write_text(json.dumps({"error": repr(error)}), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
