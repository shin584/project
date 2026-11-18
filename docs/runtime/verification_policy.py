"""Frozen required-check inventory and Q17 reporting policy (#206)."""

from collections import Counter
import math

MODELS = {"spcas9": "spcas9_non_nt", "cas12a": "cas12a_non_nt"}
VERSIONS = ("python", "keras", "torch", "numpy", "shap", "h5py")
SHAP_RUNS = ("seed0", "seed1", "seed2", "repeat", "finer")
RAM_STAGES = ("interpreter", "runtime_import", "model_loaded", "after_forward",
              "after_ig", "peak_working_set")


def layer_parity(got, want, criteria):
    """The frozen #193 absolute-scaled AND relative gate, used in collection/replay."""
    import numpy as np
    got, want = np.asarray(got), np.asarray(want)
    if got.shape != want.shape or not got.size:
        raise ValueError("Layer shape mismatch/empty activation")
    absolute = float(np.max(np.abs(got - want)))
    relative = float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-6)))
    bound = 1e-4 * max(1.0, float(np.max(np.abs(want))))
    finite = all_finite([got, want, absolute, relative, bound])
    return {"max_abs_diff": absolute, "max_rel_diff": relative, "scaled_bound": bound,
            "abs_passed": finite and absolute <= bound,
            "rel_passed": finite and relative <= criteria["max_rel_diff"]}


def runtime_version_failures(versions, contract):
    expected = contract["runtime"]
    errors = [f"Runtime version missing/mismatch: {name}" for name in VERSIONS
              if versions.get(name) != expected["versions"][name]]
    if versions.get("backend") != expected["backend_env"]["KERAS_BACKEND"]:
        errors.append("Target backend must be torch")
    if versions.get("torch_threads") != str(expected["threads"]["torch_num_threads"]):
        errors.append("Target thread configuration mismatch")
    return errors


def require_planned_termination(data):
    """Q8 overrides every state, including inapplicable/unverified declarations."""
    for record in [data, *data.get("investigations", [])]:
        if (record.get("termination") not in (None, "planned_budget")
                or record.get("state") in ("interrupted", "aborted")):
            raise ValueError("Unexpected interruption requires a new bundle/full remeasurement")


def expected_checks(contract, topology):
    core = [f"version/{name}" for name in VERSIONS] + [
        "runtime/backend", "runtime/torch_threads", "finiteness/raw"]
    diag = ["finiteness/raw"]
    for key, model_key in MODELS.items():
        spec = contract["models"][model_key]
        core += [f"{key}/{suffix}" for suffix in (
            "model_identity", "weight_placement/variable_set", "weight_placement/bitwise",
            "forward_parity/max_abs_diff", "forward_parity/max_rel_diff",
            "layerwise_parity/abs_scaled", "layerwise_parity/rel")]
        for case in contract["cases"][key]:
            prefix = f"{key}/{case['id']}"
            core += [f"{prefix}/{suffix}" for suffix in (
                "output_set", "layerwise/layer_set", "function_match", "finiteness/forward",
                "ig_completeness/default", "ig_completeness/finer",
                "finiteness/ig_default", "finiteness/ig_finer",
                "ig_resolution/pearson", "ig_resolution/max_abs_diff",
                "ig_resolution/sign_agreement", "shap_additivity", "finiteness/shap",
                "shap_same_seed_determinism", "shap_cross_seed/pearson",
                "shap_cross_seed/max_abs_diff", "shap_default_vs_finer")]
            for run in SHAP_RUNS:
                core.append(f"{prefix}/finiteness/shap_{run}")
            for layer in topology[key]:
                core += [f"{prefix}/layerwise/{layer}/{metric}"
                         for metric in ("abs_scaled", "rel")]
            for point in ("x", "alpha0.5"):
                for input_spec in spec["inputs"]:
                    name = input_spec["name"]
                    core += [f"{prefix}/gradient_check/{point}/{name}/{metric}"
                             for metric in ("cosine", "rel_l2")]
                    diag += [f"{prefix}/cross_runtime_gradient/{point}/{name}/{metric}"
                             for metric in ("max_abs_diff", "cosine")]
            diag += [f"{prefix}/directional_fd/{metric}" for metric in ("median", "max")]
    return {"verify": sorted(core), "diagnose": sorted(diag)}


def expected_arrays(contract, topology):
    """Known dimensions are contractual; layer dimensions must match reference."""
    result = {"reference": {}, "verify": {}, "diagnose": {}, "offline": {}}
    for key, model_key in MODELS.items():
        spec = contract["models"][model_key]
        result["offline"][f"{key}/output"] = [1]
        result["offline"][f"{key}/gradient_output"] = [1]
        for index, input_spec in enumerate(spec["inputs"]):
            result["offline"][f"{key}/gradient/{index}"] = [1, *input_spec["shape"]]
        for case in contract["cases"][key]:
            prefix = f"{key}/{case['id']}"
            for stage in ("reference", "verify"):
                result[stage][prefix + "/outputs"] = [spec["n_outputs"]]
                for layer in topology[key]:
                    result[stage][f"{prefix}/layer/{layer}"] = None
            for point in ("x", "alpha0.5"):
                for index, input_spec in enumerate(spec["inputs"]):
                    shape = input_spec["shape"]
                    result["reference"][f"{prefix}/grad/{point}/{index}"] = [1, *shape]
                    for stage in ("verify", "diagnose"):
                        result[stage][f"{prefix}/grad/{point}/{index}"] = [1, *shape]
                    result["verify"][f"{prefix}/fd/{point}/{index}"] = shape
            length = spec["inputs"][0]["shape"][0]
            for run in ("default", "finer"):
                result["verify"][f"{prefix}/ig/{run}/attribution"] = [length]
                result["verify"][f"{prefix}/ig/{run}/terms"] = [3]
                result["verify"][f"{prefix}/ig/{run}/input_totals"] = [len(spec["inputs"])]
                for index, input_spec in enumerate(spec["inputs"]):
                    result["verify"][f"{prefix}/ig/{run}/input_attribution/{index}"] = input_spec["shape"]
            for run in SHAP_RUNS:
                result["verify"][f"{prefix}/shap/{run}/values"] = [length]
                result["verify"][f"{prefix}/shap/{run}/terms"] = [3]
            result["diagnose"][prefix + "/directional/terms"] = [32, 3]
    return result


def all_finite(value):
    import numpy as np
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "biufc" and bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(value)
    return True


def inventory_failures(checks, expected):
    names = [row["check"] for row in checks]
    counts = Counter(names)
    errors = [f"missing check: {name}" for name in sorted(set(expected) - set(names))]
    errors += [f"unexpected check: {name}" for name in sorted(set(names) - set(expected))]
    errors += [f"duplicate check: {name}" for name, count in counts.items() if count != 1]
    errors += [f"failed check: {row['check']}" for row in checks
               if row.get("passed") is not True or not all_finite(row)]
    return errors


def investigation_outcome(data, protocol):
    """Validate a finite registered investigation inventory before applying Q17."""
    require_planned_termination(data)
    rows = data["investigations"]
    expected = protocol["investigations"]
    counts_by_id = Counter(r["id"] for r in rows)
    if set(counts_by_id) - set(expected) or any(n != 1 for n in counts_by_id.values()):
        raise ValueError("Investigation inventory mismatch")
    missing = set(expected) - set(counts_by_id)
    incomplete, findings = bool(missing), False
    counts = {name: {"completed": 0, "incomplete": expected[name]["planned"]} for name in sorted(missing)}
    for row in rows:
        name, state = row["id"], row["state"]
        definition = expected[name]
        planned, completed = row["planned"], row["completed"]
        if (type(planned) is not int or type(completed) is not int
                or not 0 <= completed <= planned or planned != definition["planned"]):
            raise ValueError(f"Invalid investigation counts: {name}")
        counts[name] = {"completed": completed, "incomplete": planned - completed}
        if state == "not_applicable":
            if not definition.get("conditional") or not row.get("reason") or completed != 0:
                raise ValueError(f"Invalid inapplicable investigation: {name}")
            continue
        if state not in ("pass", "unverified", "unavailable", "time_limited",
                         "unresolved", "no_eligible_candidate"):
            raise ValueError(f"Unknown investigation state: {name}/{state}")
        if state in ("unavailable", "time_limited") and not row.get("reason"):
            raise ValueError(f"Missing investigation terminal reason: {name}")
        if state == "time_limited" and row.get("termination") != "planned_budget":
            raise ValueError(f"Unexpected interruption requires a new bundle: {name}")
        if state in ("unverified", "unavailable", "time_limited") or completed < planned:
            incomplete = True
        if state in ("unresolved", "no_eligible_candidate"):
            findings = True
        if state in ("pass", "unresolved", "no_eligible_candidate") and completed != planned:
            raise ValueError(f"Completed investigation has missing measurements: {name}")
        if completed and not row.get("raw_arrays"):
            raise ValueError(f"Missing investigation raw inventory: {name}")
        if state in ("pass", "unresolved", "no_eligible_candidate"):
            if set(definition.get("required_arrays", [])) - set(row["raw_arrays"]):
                raise ValueError(f"Incomplete registered investigation raw inventory: {name}")
    return ("incomplete" if incomplete else "completed_with_findings" if findings else "pass"), counts
