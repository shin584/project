"""Fixed-location diagnostics for #211; never changes a required check result.

The public experiment boundary is sweep(). The journaled verification CLI owns
execution; the optional float64 worker runs this module in an isolated process.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

from verification_bundle import digest, read_json, verify_receipt, write_json
from verification_policy import MODELS, all_finite

STEPS32 = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
STEPS64 = STEPS32 + [1e-5, 1e-6]
ROOT = Path(__file__).resolve().parents[2]

# Passing historical controls, fixed before new measurements. A new failure at
# a control is retained too; it cannot be silently replaced by a nicer case.
CONTROLS = {"spcas9": ("S3", "alpha0.5", "In_Seq"),
            "cas12a": ("C2", "x", "Input_Seq")}


def _target_metadata(contract, model, case, point, identity):
    spec = contract["models"][MODELS[model]]
    return {"id": identity, "model": model, "case": case, "point": point,
            "head": spec["verified_head"]["index"], "head_name": spec["verified_head"]["name"],
            "window": next(c["window"] for c in contract["cases"][model] if c["id"] == case),
            "original_failures": []}


def boundary_layer_inventory(work, contract):
    """Derive selected-head ancestors from frozen graph bytes without loading it."""
    result = {}

    def histories(value):
        if isinstance(value, dict):
            if "keras_history" in value:
                yield value["keras_history"][0]
            for item in value.values():
                yield from histories(item)
        elif isinstance(value, list):
            for item in value:
                yield from histories(item)

    for model, model_key in MODELS.items():
        spec = contract["models"][model_key]
        with zipfile.ZipFile(work / "snapshot/assets" / spec["path"]) as archive:
            config = json.loads(archive.read("config.json"))["config"]
        layers = {l["config"]["name"]: l for l in config["layers"]}
        outputs = config["output_layers"]
        output = outputs if isinstance(outputs[0], str) else outputs[spec["verified_head"]["index"]]
        pending, ancestors = [output[0]], set()
        while pending:
            name = pending.pop()
            if name in ancestors:
                continue
            ancestors.add(name)
            pending.extend(histories(layers[name].get("inbound_nodes", [])))
        result[model] = [name for name, layer in layers.items() if name in ancestors and
                         (layer["class_name"] in ("ReLU", "LeakyReLU")
                          or layer["config"].get("activation") == "relu")]
    return result


def plan_locations(contract, historical, core, diagnostic):
    """Union check identities without changing their original failure verdicts."""
    targets = {}
    sources = [("historical", historical.get("core", {})),
               ("historical", historical.get("diagnostics", {})),
               ("current", core), ("current", diagnostic)]
    for source, document in sources:
        for check in document["checks"]:
            parts = check["check"].split("/")
            if len(parts) < 4 or check["passed"] is not False:
                continue
            model, case, kind = parts[:3]
            if kind not in ("gradient_check", "directional_fd"):
                continue
            spec = contract["models"][MODELS[model]]
            if case not in {c["id"] for c in contract["cases"][model]}:
                raise ValueError("Unknown diagnostic case")
            if kind == "gradient_check":
                _, _, _, point, name, metric = parts
                index = [i["name"] for i in spec["inputs"]].index(name)
                if point not in ("x", "alpha0.5") or metric not in ("cosine", "rel_l2"):
                    raise ValueError("Unknown coordinate failure identity")
                identity = f"{model}/{case}/{point}/{name}"
                location = {"input": index, "input_name": name, "kind": "coordinate"}
                step = 0.01
            else:
                if len(parts) != 4 or parts[-1] not in ("median", "max"):
                    raise ValueError("Unknown direction failure identity")
                point, step = "alpha0.5", 0.001
                identity = f"{model}/{case}/{point}/directions"
                location = {"kind": "directional", "directions": list(range(32)),
                            "inputs": [i["name"] for i in spec["inputs"]]}
            target = targets.setdefault(identity, {
                **_target_metadata(contract, model, case, point, identity), **location})
            target["original_failures"].append({"source": source, "step": step, **check})
    for model, (case, point, name) in CONTROLS.items():
        spec = contract["models"][MODELS[model]]
        identity = f"{model}/{case}/{point}/{name}"
        target = targets.setdefault(identity, {
            **_target_metadata(contract, model, case, point, identity),
            "input": [i["name"] for i in spec["inputs"]].index(name), "input_name": name,
            "kind": "coordinate"})
        target["control"] = True
        historical_checks = {c["check"]: c for c in historical["core"]["checks"]}
        control_checks = [historical_checks.get(f"{model}/{case}/gradient_check/{point}/{name}/{metric}")
                          for metric in ("cosine", "rel_l2")]
        if any(c is None or c["passed"] is not True for c in control_checks):
            raise ValueError("Preregistered control lacks its historical passing evidence")
        target["control_reference"] = control_checks
    return [targets[name] for name in sorted(targets)]


class DiagnosticRuntime:
    """Investigation-only Keras graph; deployment runtime is never modified."""

    def __init__(self, model, dtype, head):
        self.model, self.dtype, self.head = model, np.dtype(dtype), head
        self.boundary_probe = None
        self.boundary_names = []

    def call(self, arrays):
        return self.model(arrays[0] if len(arrays) == 1 else arrays, training=False)

    def picked(self, outputs, head):
        return (outputs[head] if isinstance(outputs, (list, tuple)) else outputs).reshape(-1)

    def forward(self, arrays, head=0, **kwargs):
        import torch
        with torch.no_grad():
            values = self.call([torch.from_numpy(np.asarray(a, dtype=self.dtype)) for a in arrays])
        return self.picked(values, head).detach().numpy().copy()

    def gradients(self, arrays, head=0):
        import torch
        tensors = [torch.from_numpy(np.array(a, dtype=self.dtype)).requires_grad_(True) for a in arrays]
        y = self.picked(self.call(tensors), head)
        gradients = torch.autograd.grad(y.sum(), tensors)
        return y.detach().numpy().copy(), [g.detach().numpy().copy() for g in gradients]

    def boundary_values(self, arrays):
        import keras
        import torch
        if self.boundary_probe is None:
            # Restrict probes to ancestors of the selected head. Other Hydra
            # heads cannot supply evidence about this head's failing stencil.
            selected = keras.Model(self.model.inputs, self.model.outputs[self.head])
            outputs, linear_layers = [], []
            for layer in selected.layers:
                activation = getattr(getattr(layer, "activation", None), "__name__", "")
                if type(layer).__name__ in ("ReLU", "LeakyReLU"):
                    outputs.append(layer.input)
                elif activation == "relu":
                    config = layer.get_config()
                    config.update(activation="linear", name="diagnostic_pre_" + layer.name)
                    linear = type(layer).from_config(config)
                    outputs.append(linear(layer.input))
                    linear.set_weights(layer.get_weights())
                    linear_layers.append(linear)
                else:
                    continue
                self.boundary_names.append(layer.name + "/preactivation")
            self.boundary_probe = keras.Model(self.model.inputs, outputs) if outputs else False
        if self.boundary_probe is False:
            return {}
        tensors = [torch.from_numpy(np.asarray(a, dtype=self.dtype)) for a in arrays]
        with torch.no_grad():
            values = self.boundary_probe(tensors[0] if len(tensors) == 1 else tensors, training=False)
        values = values if isinstance(values, (list, tuple)) else [values]
        return {name: v.detach().numpy().copy() for name, v in zip(self.boundary_names, values)}


def _precision_config(value, dtype):
    if isinstance(value, list):
        return [_precision_config(v, dtype) for v in value]
    if isinstance(value, dict):
        result = {k: _precision_config(v, dtype) for k, v in value.items()}
        if "dtype" in result:
            result["dtype"] = dtype
        return result
    return value


def qualify_float64(model, inputs, head):
    """Clone the graph, expand stored weight values exactly, probe real dtypes.

    Unsupported backend operations propagate as unavailable at the child boundary.
    No manual recurrent forward/backward or input-only conversion is accepted.
    """
    import keras
    original_config = json.loads(model.to_json())["config"]
    config = _precision_config(original_config, "float64")
    copy = keras.Model.from_config(config)
    if _precision_config(json.loads(copy.to_json())["config"], "ignored") != _precision_config(original_config, "ignored"):
        raise ValueError("Float64 architecture/head differs")
    if copy.output_names != model.output_names:
        raise ValueError("Float64 head identity differs")
    original = model.get_weights()
    if any(w.dtype != np.float32 for w in original):
        raise ValueError("Stored weights must all be float32")
    if not all_finite(original):
        raise ValueError("Nonfinite stored float32 weights")
    copy.set_weights([w.astype(np.float64) for w in original])
    if len(copy.weights) != len(model.weights):
        raise ValueError("Float64 variable inventory differs")
    weights = []
    for source, expanded in zip(model.weights, copy.weights):
        a, b = source.numpy(), expanded.numpy()
        # Keras deserialization can drop a Bidirectional wrapper's variable
        # path prefix. The identical serialized graph fixes variable order;
        # retain both paths and check name/shape/value without renaming weights.
        if (source.name != expanded.name or a.shape != b.shape or b.dtype != np.float64
                or not np.array_equal(a.astype(np.float64), b)):
            raise ValueError("Weight values were not exactly expanded")
        import hashlib
        weights.append({"path": source.path, "copy_path": expanded.path,
                        "dtype": str(b.dtype), "shape": list(b.shape),
                        "source_dtype": str(a.dtype),
                        "source_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                        "expanded_source_sha256": hashlib.sha256(a.astype(np.float64).tobytes()).hexdigest(),
                        "copy_sha256": hashlib.sha256(b.tobytes()).hexdigest()})
    runtime = DiagnosticRuntime(copy, "float64", head)
    output, gradients = runtime.gradients(inputs, head)
    # Probe actual layer inputs/outputs, in addition to declared dtype policy.
    layers = [l for l in copy.layers if not isinstance(l, keras.layers.InputLayer)]
    probe = keras.Model(copy.inputs, [t for l in layers for t in
                        [*keras.tree.flatten(l.input), *keras.tree.flatten(l.output)]])
    tensors = [np.asarray(a, np.float64) for a in inputs]
    import torch
    with torch.no_grad():
        observed = probe(tensors[0] if len(tensors) == 1 else tensors, training=False)
    observed = observed if isinstance(observed, (list, tuple)) else [observed]
    compute_dtypes = [str(v.dtype) for v in observed]
    if (any(str(l.compute_dtype) != "float64" for l in layers)
            or any(d != "torch.float64" for d in compute_dtypes)
            or output.dtype != np.float64 or any(g.dtype != np.float64 for g in gradients)):
        raise ValueError("Actual compute/variable/gradient dtype is not float64")
    if not all_finite([original, copy.get_weights(), output, gradients,
                       [v.detach().numpy() for v in observed]]):
        raise ValueError("Nonfinite float64 qualification")
    return runtime, {"qualified": True, "architecture_matched": True,
                     "exact_weight_expansion": True, "inference_mode": True,
                     "head_names": copy.output_names, "weights": weights,
                     "compute_dtypes": compute_dtypes, "output_dtype": str(output.dtype),
                     "gradient_dtypes": [str(g.dtype) for g in gradients]}


def directions(inputs):
    """Exactly the #193 seed/normalization, expanded unchanged for float64."""
    rng = np.random.default_rng(193)
    result = []
    for _ in range(32):
        values = [rng.standard_normal(p.shape).astype(np.float32) for p in inputs]
        norm = np.sqrt(sum(float((d ** 2).sum()) for d in values))
        result.append([d / norm for d in values])
    return result


def comparison(analytic, numeric, kind):
    a, b = np.asarray(analytic, np.float64), np.asarray(numeric, np.float64)
    residual = a - b
    if kind == "coordinate":
        product = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosine = float(a @ b / product) if product else float(np.array_equal(a, b))
        relative = float(np.linalg.norm(residual) / max(float(np.linalg.norm(b)), 1e-12))
        return {"cosine": cosine, "rel_l2": relative,
                "passed": all_finite([a, b]) and cosine >= 0.999 and relative <= 0.02}
    errors = np.abs(residual) / np.maximum(np.abs(b), 1e-6)
    median, maximum = float(np.median(errors)), float(np.max(errors))
    return {"median": median, "max": maximum,
            "passed": all_finite([a, b]) and median <= 0.02 and maximum <= 0.1}


def sweep(runtime, targets, case_inputs, precision, *, checkpoint=None):
    """Retain every fixed stencil, gradient and boundary witness at each target.

    checkpoint receives completed locations and raw arrays after EVERY step so a
    planned child timeout can retain partial work. No favorable step is selected.
    """
    dtype = np.dtype(precision)
    steps = STEPS32 if precision == "float32" else STEPS64
    results, raw = [], {}
    for target in targets:
        factor = 1.0 if target["point"] == "x" else 0.5
        inputs = [np.asarray(a, dtype=dtype) * factor for a in case_inputs[target["case"]]]
        output, gradients = runtime.gradients(inputs, head=target["head"])
        prefix = precision + "/" + target["id"]
        raw[prefix + "/output"] = output
        for i, (p, g) in enumerate(zip(inputs, gradients)):
            raw[f"{prefix}/input/{i}"] = p
            raw[f"{prefix}/gradient/{i}"] = g
        if target["kind"] == "coordinate":
            index = target["input"]
            analytic = gradients[index].reshape(-1).astype(np.float64)
            vectors = []
            for coordinate in range(inputs[index].size):
                vector = [np.zeros_like(p) for p in inputs]
                vector[index].reshape(-1)[coordinate] = 1
                vectors.append(vector)
        else:
            # #193 draws float32 normals but divides by a NumPy float64 norm.
            # Preserve those normalized float64 directions exactly, including
            # for float32 diagnostics. Only model inputs/compute are float32.
            vectors = directions(inputs)
            analytic = np.array([sum(float((g * d).sum()) for g, d in zip(gradients, ds))
                                 for ds in vectors], dtype=np.float64)
        for i in range(len(inputs)):
            raw[f"{prefix}/vectors/{i}"] = np.concatenate([ds[i] for ds in vectors])
        row = {"target": target, "steps": [], "original_failures_preserved": True}
        results.append(row)
        previous = None
        for step in steps:
            names = {}
            base = f"{prefix}/h{step:g}"
            values = []
            boundary = []
            # Chunk stencils to bound activation memory; each plus/minus pair
            # uses exactly the same point, input/direction and head as the FD.
            for start in range(0, len(vectors), 16):
                block = vectors[start:start + 16]
                batch = [np.concatenate([p + sign * step * ds[i]
                         for ds in block for sign in (1, -1)])
                         for i, p in enumerate(inputs)]
                batch = [np.asarray(a, dtype=dtype) for a in batch]
                predictions = runtime.forward(batch, head=target["head"], batch_size=32)
                values.extend(predictions.tolist())
                activations = runtime.boundary_values(batch)
                crossing = np.zeros(len(block), dtype=bool)
                for name, activation in activations.items():
                    raw[f"{base}/boundary/{start}/{name}"] = activation
                    pairs = activation.reshape(len(block), 2, -1)
                    crossing |= np.any((pairs[:, 0] > 0) != (pairs[:, 1] > 0), axis=1)
                boundary.extend((start + np.flatnonzero(crossing)).tolist())
            values = np.array(values, dtype=dtype).reshape(-1, 2)
            numeric = (values[:, 0].astype(np.float64) - values[:, 1].astype(np.float64)) / (2 * step)
            for name, value in {"analytic": analytic, "numeric": numeric,
                                "residual": analytic - numeric, "outputs": values}.items():
                names[name] = base + "/" + name
                raw[names[name]] = value
            observation = {"step": step, "raw": names,
                           "comparison": comparison(analytic, numeric, target["kind"]),
                           "boundary_crossing_stencils": boundary,
                           "boundary_available": bool(activations)}
            if previous is not None:
                observation["adjacent_agreement"] = comparison(previous, numeric, target["kind"])
            previous = numeric
            row["steps"].append(observation)
            original_step = 0.01 if target["kind"] == "coordinate" else 0.001
            if step == original_step:
                row["original_step"] = observation
            if checkpoint:
                checkpoint(results, raw)
        # Boundary evidence is local to the actual original failing stencil.
        # Even complete boundary evidence is support, not a unique causal proof.
        original = row["original_step"]
        row["boundary_support"] = bool(original["boundary_crossing_stencils"])
        row["unresolved"] = bool(target["original_failures"])
    return results, raw


def _replace_json(path, document):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(path)


def _checkpoint(directory, rows, raw):
    temporary = directory / "raw.tmp.npz"
    np.savez_compressed(temporary, **raw)
    temporary.replace(directory / "raw.npz")
    # Publish the inventory only after its arrays are durable. A later raw
    # checkpoint may be a superset of an earlier JSON checkpoint after a kill.
    _replace_json(directory / "progress.json", {"locations": rows})


def _inputs(work, contract, model):
    with np.load(work / "snapshot/inputs.npz", allow_pickle=False) as archive:
        return {c["id"]: [archive[f"{model}/{c['id']}/input/{i}"].copy()
                          for i in range(len(contract["models"][MODELS[model]]["inputs"]))]
                for c in contract["cases"][model]}


def _load_model(work, contract, model):
    sys.path.insert(0, str(ROOT))
    import non_nt_runtime as rt
    keras = rt.import_keras()
    spec = contract["models"][MODELS[model]]
    path = work / "snapshot/assets" / spec["path"]
    if digest(path) != spec["sha256"]:
        raise ValueError("Diagnostic checkpoint checksum mismatch")
    return keras.saving.load_model(path, compile=False)


def float64_worker(work, model, directory):
    """One model, one attempt. Parent enforces qualification/diagnosis deadlines."""
    contract = read_json(work / "snapshot/contract.json")
    targets = read_json(directory / "targets.json")
    inputs = _inputs(work, contract, model)
    try:
        original = _load_model(work, contract, model)
        first = targets[0]
        factor = 1.0 if first["point"] == "x" else 0.5
        runtime, qualification = qualify_float64(original,
            [a * factor for a in inputs[first["case"]]], first["head"])
    except (ValueError, RuntimeError, TypeError, NotImplementedError, ImportError) as error:
        _replace_json(directory / "qualification.json", {
            "qualified": False, "nonfinite": "Nonfinite" in str(error),
            "reason": f"{type(error).__name__}: {error}"})
        return 0
    # Retain the separate original only for exact-value qualification, then drop it.
    del original
    gc.collect()
    _replace_json(directory / "qualification.json", qualification)
    while not (directory / "start-diagnosis").exists():
        time.sleep(0.02)
    try:
        rows, raw = sweep(runtime, targets, inputs, "float64",
                          checkpoint=lambda rows, raw: _checkpoint(directory, rows, raw))
        _checkpoint(directory, rows, raw)
        _replace_json(directory / "completed.json", {"locations": rows})
    except (RuntimeError, NotImplementedError, TypeError) as error:
        _replace_json(directory / "unavailable.json", {"reason": f"{type(error).__name__}: {error}"})
    return 0


def run_float64(work, model, targets, deadline, *, qualification_seconds=300,
                diagnosis_seconds=600):
    """Bounded process execution. A crash is an interruption, never unavailable."""
    if time.monotonic() >= deadline:
        return {"state": "time_limited", "termination": "planned_budget",
                "reason": "Total float64 budget exhausted before model start", "locations": [],
                "completed_steps": 0, "planned_steps": len(targets) * 8}, {}
    directory = work / ("float64-" + model)
    directory.mkdir()  # refuse reattempts, including after a previous timeout
    write_json(directory / "targets.json", targets)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", model,
               "--work", str(work), "--directory", str(directory)]
    started = time.monotonic()
    phase_deadline = min(deadline, started + qualification_seconds)
    phase = "qualification"
    state, reason = "unresolved", "Investigation complete; a unique cause is not established"
    with (directory / "stdout.txt").open("w", encoding="utf-8") as stdout, \
            (directory / "stderr.txt").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        try:
            while True:
                if time.monotonic() >= phase_deadline:
                    process.kill()
                    process.wait()
                    state, reason = "time_limited", f"Prespecified {phase}/total budget exhausted"
                    break
                qualified = directory / "qualification.json"
                if phase == "qualification" and qualified.exists():
                    qualification = read_json(qualified)
                    if qualification.get("qualified") is not True:
                        if process.poll() is not None:
                            if process.returncode != 0:
                                raise RuntimeError("Unexpected qualification child exit; new bundle required")
                            state, reason = "unavailable", qualification["reason"]
                            break
                    else:
                        phase = "diagnosis"
                        phase_deadline = min(deadline, time.monotonic() + diagnosis_seconds)
                        (directory / "start-diagnosis").touch()
                code = process.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError(f"Unexpected float64 child exit {model}/{code}; new bundle required")
                    if qualified.exists() and read_json(qualified).get("qualified") is not True:
                        state, reason = "unavailable", read_json(qualified)["reason"]
                    elif (directory / "unavailable.json").exists():
                        state, reason = "unavailable", read_json(directory / "unavailable.json")["reason"]
                    elif not (directory / "completed.json").exists():
                        raise RuntimeError("Unexpected float64 child interruption; new bundle required")
                    break
                time.sleep(0.02)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    rows, raw = [], {}
    if (directory / "progress.json").exists():
        rows = read_json(directory / "progress.json")["locations"]
        with np.load(directory / "raw.npz", allow_pickle=False) as archive:
            raw = {n: archive[n] for n in archive.files}
    evidence = {"state": state, "reason": reason, "locations": rows,
                "execution": {"command": command, "exit_code": process.returncode,
                              "elapsed_seconds": time.monotonic() - started,
                              "phase": phase,
                              "stdout": (directory / "stdout.txt").read_text(encoding="utf-8"),
                              "stderr": (directory / "stderr.txt").read_text(encoding="utf-8"),
                              "qualification_limit_seconds": qualification_seconds,
                              "diagnosis_limit_seconds": diagnosis_seconds},
                "qualification": read_json(directory / "qualification.json")
                    if (directory / "qualification.json").exists() else {"qualified": False},
                "completed_steps": sum(len(r["steps"]) for r in rows),
                "planned_steps": len(targets) * len(STEPS64)}
    if state == "time_limited":
        evidence["termination"] = "planned_budget"
    return evidence, raw


def precision_effects(rows32, rows64, raw):
    by_id = {r["target"]["id"]: r for r in rows32}
    for row in rows64:
        baseline = {s["step"]: s for s in by_id[row["target"]["id"]]["steps"]}
        support = False
        for step in row["steps"]:
            if step["step"] not in baseline:
                continue
            previous = baseline[step["step"]]
            a = raw[previous["raw"]["residual"]]
            b = raw[step["raw"]["residual"]]
            improved = float(np.linalg.norm(b)) < float(np.linalg.norm(a))
            step["same_step_precision_effect"] = {
                "float32_residual_l2": float(np.linalg.norm(a)),
                "float64_residual_l2": float(np.linalg.norm(b)), "improved": improved}
            # All same-step comparisons are retained. The original h alone
            # supplies this support label, so no favorable h is selected.
            if step["step"] == (0.01 if row["target"]["kind"] == "coordinate" else 0.001):
                support = improved and bool(step["comparison"]["passed"])
        boundary = bool(by_id[row["target"]["id"]]["original_step"]["boundary_crossing_stencils"])
        row["interpretation"] = ("mixed_possibilities" if boundary and support else
                                 "precision_effect_support" if support else
                                 "boundary_crossing_evidence" if boundary else "unresolved")
        row["unique_cause_established"] = False
        original = next((s for s in row["steps"] if s["step"] ==
                         (0.01 if row["target"]["kind"] == "coordinate" else 0.001)), None)
        if original:
            row["original_step"] = original


def run_diagnostics(work, contract, protocol, *, precision="both"):
    """Produce the four registered rows; callers may merge candidate rows."""
    config = protocol["diagnostic_protocol"]
    validate_settings(config)
    for stage in ("verify", "diagnose"):
        verify_receipt(work, stage)
    if precision == "float64":
        verify_receipt(work, "float32")
        document = read_json(work / "float32.json")
        with np.load(work / "float32-raw.npz", allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        rows = document["investigations"]
        return _run_precision_followup(work, rows, raw)
    historical = read_json(work / "snapshot/diagnostic-history.json")
    targets = plan_locations(contract, historical, read_json(work / "verification-core.json"),
                             read_json(work / "diagnose.json"))
    write_json(work / "diagnostic-plan.json", {
        "targets": targets, "controls": CONTROLS,
        "boundary_layers": boundary_layer_inventory(work, contract),
        "history_sha256": digest(work / "snapshot/diagnostic-history.json"),
        "core_sha256": digest(work / "verification-core.json"),
        "diagnose_sha256": digest(work / "diagnose.json"),
        "settings": config})
    rows, raw = [], {}
    # Mandatory float32 work is outside the optional float64 budget and all
    # models finish it before the first optional child may be launched.
    for model, model_key in MODELS.items():
        head = contract["models"][model_key]["verified_head"]["index"]
        directory = work / ("float32-" + model)
        directory.mkdir()
        runtime = DiagnosticRuntime(_load_model(work, contract, model), "float32", head)
        selected = [t for t in targets if t["model"] == model]
        result, arrays = sweep(runtime, selected, _inputs(work, contract, model), "float32",
                               checkpoint=lambda r, a: _checkpoint(directory, r, a))
        _checkpoint(directory, result, arrays)
        raw.update(arrays)
        name = "float32_" + model
        raw[name] = np.array([sum(len(r["steps"]) for r in result), len(selected) * 6])
        rows.append({"id": name, "state": "unresolved" if any(r["unresolved"] for r in result) else "pass",
                     "planned": 1, "completed": 1, "raw_arrays": [name, *arrays],
                     "locations": result})
        del runtime
        gc.collect()
    if precision == "float32":
        return rows, raw
    return _run_precision_followup(work, rows, raw)


def _run_precision_followup(work, rows, raw):
    float32 = {model: next(r["locations"] for r in rows if r["id"] == "float32_" + model)
               for model in MODELS}
    batch_start = time.monotonic()
    deadline = batch_start + 1800
    for model in MODELS:
        name = "float64_" + model
        selected = [r["target"] for r in float32[model] if r["unresolved"]]
        if not selected:
            rows.append({"id": name, "state": "not_applicable", "planned": 1, "completed": 0,
                         "raw_arrays": [], "reason": "No unresolved original/current failure locations"})
            continue
        start_offset = time.monotonic() - batch_start
        evidence, arrays = run_float64(work, model, selected, deadline)
        evidence["budget"] = {"total_seconds": 1800, "start_offset": start_offset,
                              "end_offset": time.monotonic() - batch_start,
                              "model_order": list(MODELS)}
        raw.update(arrays)
        precision_effects(float32[model], evidence["locations"], raw)
        complete = evidence["state"] == "unresolved"
        raw[name] = np.array([evidence["completed_steps"], evidence["planned_steps"]])
        rows.append({"id": name, "planned": 1, "completed": int(complete),
                     "raw_arrays": [name, *arrays], **evidence})
    return rows, raw


def validate_settings(config):
    if (config["float32_steps"] != STEPS32
            or config["float64_extra_steps"] != [1e-5, 1e-6]
            or config["direction_seed"] != 193 or config["directions"] != 32
            or config["float64_budget_seconds"] != 1800
            or config["qualification_seconds_per_model"] != 300
            or config["diagnosis_seconds_per_model"] != 600
            or config.get("controls") != {m: {"case": c, "point": p, "input": i}
                                          for m, (c, p, i) in CONTROLS.items()}):
        raise ValueError("Diagnostic protocol changed the fixed #211 settings/budgets")


def validate_diagnostic_report(document, raw, targets, reference=None, frozen_inputs=None, boundary_layers=None,
                               *, float32_only=False):
    """Replay the public report inventory, dtypes and fixed comparison lines."""
    rows = {r["id"]: r for r in document["investigations"]}
    previous_end = 0.0
    for model in MODELS:
        row32 = rows["float32_" + model]
        selected = [t for t in targets if t["model"] == model]
        if (row32["completed"] != 1 or row32["state"] not in ("pass", "unresolved")
                or [r["target"] for r in row32["locations"]] != selected):
            raise ValueError("Float32 failed-location/control inventory mismatch")
        expected_boundary = boundary_layers[model] if boundary_layers is not None else None
        _validate_locations(row32, raw, "float32", complete=True, frozen_inputs=frozen_inputs,
                            boundary_layers=expected_boundary)
        selected64 = [r["target"] for r in row32["locations"] if r["unresolved"]]
        if any(r["unresolved"] != bool(r["target"]["original_failures"]) for r in row32["locations"]):
            raise ValueError("Unresolved original failure was discarded")
        if row32["state"] != ("unresolved" if selected64 else "pass"):
            raise ValueError("Float32 findings state differs from preserved failures")
        counts = raw.get("float32_" + model)
        if counts is not None and not np.array_equal(counts, [len(selected) * 6, len(selected) * 6]):
            raise ValueError("Float32 completed/incomplete step counts disagree")
        if float32_only:
            if counts is None or set(row32.get("raw_arrays", [])) - set(raw):
                raise ValueError("Missing mandatory float32 raw inventory")
            continue
        row64 = rows["float64_" + model]
        if row64["state"] == "not_applicable":
            if (selected64 or row64.get("locations") or row64.get("execution")
                    or any(n.startswith(f"float64/{model}/") for n in raw)):
                raise ValueError("Float64 inapplicability hides unresolved failures")
            continue
        if not selected64:
            raise ValueError("Float64 attempted without eligible locations")
        budget = row64["budget"]
        if (budget["total_seconds"] != 1800 or budget["model_order"] != list(MODELS)
                or not previous_end <= budget["start_offset"] <= budget["end_offset"] <= 1801):
            raise ValueError("Float64 total budget/model sequence mismatch")
        previous_end = budget["end_offset"]
        qualification = row64.get("qualification", {})
        if qualification.get("nonfinite"):
            raise ValueError("Nonfinite qualification evidence")
        observed = row64["locations"]
        if [r["target"] for r in observed] != selected64[:len(observed)]:
            raise ValueError("Float64 failed-location inventory mismatch")
        complete = row64["state"] in ("unresolved", "pass")
        if complete and (len(observed) != len(selected64) or row64["completed"] != 1):
            raise ValueError("Float64 completion hides missing locations")
        if row64["state"] == "pass":
            raise ValueError("Precision improvement cannot establish a unique cause/pass")
        if observed or complete:
            weights = qualification.get("weights", [])
            if (qualification.get("qualified") is not True
                    or qualification.get("architecture_matched") is not True
                    or qualification.get("exact_weight_expansion") is not True
                    or qualification.get("inference_mode") is not True
                    or not weights or any(w["dtype"] != "float64" or
                        w.get("source_dtype") != "float32" or
                        w["expanded_source_sha256"] != w["copy_sha256"] for w in weights)
                    or not qualification.get("compute_dtypes")
                    or set(qualification["compute_dtypes"]) != {"torch.float64"}
                    or qualification.get("output_dtype") != "float64"
                    or not qualification.get("gradient_dtypes")
                    or set(qualification["gradient_dtypes"]) != {"float64"}):
                raise ValueError("Ineligible float64 dtype/weight evidence")
            heads = qualification.get("head_names", [])
            if any(t["head"] >= len(heads) or heads[t["head"]] != t["head_name"] for t in selected64):
                raise ValueError("Float64 selected head identity differs")
            if reference is not None:
                original_weights = reference[model]["weights"]
                actual_weights = {w["path"]: w for w in weights}
                if (set(actual_weights) != set(original_weights) or any(
                        actual_weights[name]["source_sha256"] != original_weights[name]["sha256"]
                        or actual_weights[name]["shape"] != original_weights[name]["shape"]
                        for name in original_weights)):
                    raise ValueError("Float64 weight source differs from frozen reference")
        _validate_locations(row64, raw, "float64", complete=complete, frozen_inputs=frozen_inputs,
                            boundary_layers=expected_boundary)
        replayed = json.loads(json.dumps(observed))
        precision_effects(row32["locations"], replayed, raw)
        if replayed != observed:
            raise ValueError("Same-step precision interpretation differs from raw evidence")
        completed = sum(len(r["steps"]) for r in observed)
        if row64["completed_steps"] != completed or row64["planned_steps"] != len(selected64) * 8:
            raise ValueError("Float64 completed/incomplete step counts disagree")
        execution = row64.get("execution", {})
        if not execution and not (row64["state"] == "time_limited" and budget["start_offset"] >= 1800):
            raise ValueError("Missing isolated float64 child execution")
        if execution and (execution["qualification_limit_seconds"] != 300
                          or execution["diagnosis_limit_seconds"] != 600
                          or execution["elapsed_seconds"] > 901):
            raise ValueError("Float64 budget exceeded registered limits")


def _validate_locations(row, raw, precision, *, complete, frozen_inputs=None, boundary_layers=None):
    steps = STEPS32 if precision == "float32" else STEPS64
    for location in row["locations"]:
        target = location["target"]
        observed = location["steps"]
        if [s["step"] for s in observed] != steps[:len(observed)] or (complete and len(observed) != len(steps)):
            raise ValueError("Diagnostic fixed step inventory mismatch")
        if location.get("original_failures_preserved") is not True:
            raise ValueError("Original failure was not preserved")
        prefix = precision + "/" + target["id"]
        count = 0
        while f"{prefix}/input/{count}" in raw:
            for kind in ("input", "gradient"):
                if raw[f"{prefix}/{kind}/{count}"].dtype != np.dtype(precision):
                    raise ValueError("Actual diagnostic input/gradient dtype mismatch")
            count += 1
        if not count:
            raise ValueError("Missing raw diagnostic gradients")
        expected_inputs = {f"{prefix}/input/{i}" for i in range(count)}
        expected_gradients = {f"{prefix}/gradient/{i}" for i in range(count)}
        if ({n for n in raw if n.startswith(prefix + "/input/")} != expected_inputs
                or {n for n in raw if n.startswith(prefix + "/gradient/")} != expected_gradients):
            raise ValueError("Diagnostic input/gradient inventory mismatch")
        if raw[f"{prefix}/output"].dtype != np.dtype(precision):
            raise ValueError("Diagnostic output dtype mismatch")
        for i in range(count):
            point = raw[f"{prefix}/input/{i}"]
            if point.shape != raw[f"{prefix}/gradient/{i}"].shape:
                raise ValueError("Diagnostic gradient shape mismatch")
        if frozen_inputs is not None:
            original_prefix = f"{target['model']}/{target['case']}/input/"
            expected_original = {n for n in frozen_inputs if n.startswith(original_prefix)}
            if expected_original != {original_prefix + str(i) for i in range(count)}:
                raise ValueError("Diagnostic frozen input inventory mismatch")
            factor = 1.0 if target["point"] == "x" else 0.5
            for i in range(count):
                original = np.asarray(frozen_inputs[original_prefix + str(i)], dtype=precision) * factor
                if not np.array_equal(raw[f"{prefix}/input/{i}"], original):
                    raise ValueError("Diagnostic input/point differs from the frozen failure location")
        if target["kind"] == "coordinate":
            expected_analytic = raw[f"{prefix}/gradient/{target['input']}"].reshape(-1).astype(np.float64)
            for i in range(count):
                shape = raw[f"{prefix}/input/{i}"].shape[1:]
                expected_vectors = np.zeros((expected_analytic.size, *shape), dtype=precision)
                if i == target["input"]:
                    expected_vectors.reshape(expected_analytic.size, -1)[:] = np.eye(expected_analytic.size)
                if not np.array_equal(raw[f"{prefix}/vectors/{i}"], expected_vectors):
                    raise ValueError("Diagnostic coordinate vectors do not cover the entire input")
        else:
            gradients = [raw[f"{prefix}/gradient/{i}"] for i in range(count)]
            points = [raw[f"{prefix}/input/{i}"] for i in range(count)]
            vectors = directions(points)
            expected_analytic = np.array([sum(float((g * d).sum())
                for g, d in zip(gradients, ds)) for ds in vectors])
            for i in range(count):
                if not np.array_equal(raw[f"{prefix}/vectors/{i}"],
                                      np.concatenate([ds[i] for ds in vectors])):
                    raise ValueError("Diagnostic directions differ from seed 193")
        previous = None
        for step in observed:
            names = step["raw"]
            expected = {n: f"{prefix}/h{step['step']:g}/{n}"
                        for n in ("analytic", "numeric", "residual", "outputs")}
            if names != expected or any(n not in raw for n in names.values()):
                raise ValueError("Diagnostic raw stencil inventory mismatch")
            a, b, residual, outputs = [raw[names[n]] for n in ("analytic", "numeric", "residual", "outputs")]
            if (a.shape != b.shape or a.ndim != 1 or outputs.shape != (a.size, 2)
                    or outputs.dtype != np.dtype(precision)
                    or not np.array_equal(residual, a - b, equal_nan=True)
                    or not np.array_equal(b, (outputs[:, 0].astype(np.float64) -
                         outputs[:, 1].astype(np.float64)) / (2 * step["step"]), equal_nan=True)):
                raise ValueError("Diagnostic raw residual/stencil/dtype mismatch")
            if step["comparison"] != comparison(a, b, target["kind"]):
                raise ValueError("Diagnostic comparison lines/results disagree")
            if not np.array_equal(a, expected_analytic, equal_nan=True):
                raise ValueError("Diagnostic analytic gradient/input inventory mismatch")
            crossing = set()
            boundary_prefix = f"{prefix}/h{step['step']:g}/boundary/"
            if boundary_layers is not None:
                required = {f"{boundary_prefix}{start}/{layer}/preactivation"
                            for start in range(0, a.size, 16) for layer in boundary_layers}
                if (required != {n for n in raw if n.startswith(boundary_prefix)}
                        or step["boundary_available"] != bool(boundary_layers)):
                    raise ValueError("Missing/wrong selected-head boundary stencil inventory")
            for name, activation in raw.items():
                if name.startswith(boundary_prefix):
                    start = int(name[len(boundary_prefix):].split("/", 1)[0])
                    if (activation.shape[0] != 2 * min(16, a.size - start)
                            or activation.dtype != np.dtype(precision)):
                        raise ValueError("Boundary stencil shape/dtype mismatch")
                    pairs = activation.reshape(activation.shape[0] // 2, 2, -1)
                    crossing.update((start + np.flatnonzero(np.any(
                        (pairs[:, 0] > 0) != (pairs[:, 1] > 0), axis=1))).tolist())
            if step["boundary_crossing_stencils"] != sorted(crossing):
                raise ValueError("Boundary evidence does not match the actual stencil")
            if target["kind"] == "directional" and a.size != 32:
                raise ValueError("Diagnostic must retain all 32 directions")
            if previous is not None and step.get("adjacent_agreement") != comparison(previous, b, target["kind"]):
                raise ValueError("Diagnostic adjacent-step agreement missing/mismatch")
            previous = b
        original = next((s for s in observed if s["step"] ==
                         (0.01 if target["kind"] == "coordinate" else 0.001)), None)
        if original and location.get("original_step") != original:
            raise ValueError("Original failure stencil was replaced")
        if complete and location.get("boundary_support") != bool(original["boundary_crossing_stencils"]):
            raise ValueError("Boundary interpretation differs from the original actual stencil")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=list(MODELS), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(float64_worker(args.work, args.worker, args.directory))
