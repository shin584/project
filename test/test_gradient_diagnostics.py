"""Public diagnostic experiment/report regressions, not #212/#213 measurements."""

import sys
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/runtime"))
import gradient_diagnostics as diagnostics


class QuadraticRuntime:
    """Independent analytic fixture: f(x) = sum(x*x), gradient = 2*x."""

    def forward(self, arrays, head=0, **kwargs):
        return sum((a * a).reshape(len(a), -1).sum(axis=1) for a in arrays)

    def gradients(self, arrays, head=0):
        return self.forward(arrays, head), [2 * a for a in arrays]

    def boundary_values(self, arrays):
        return {"relu/preactivation": arrays[0]}


def test_fixed_coordinate_sweep_keeps_original_failure_and_actual_stencils():
    target = {"id": "spcas9/S1/x/In_Seq", "model": "spcas9", "case": "S1",
              "point": "x", "input": 0, "kind": "coordinate", "head": 1,
              "original_failures": [{"source": "historical", "step": 0.01,
                                     "check": "rel_l2", "measured": 0.03}]}
    result, raw = diagnostics.sweep(
        QuadraticRuntime(), [target], {"S1": [np.array([[0.0, 0.25]], np.float32)]},
        "float32")
    row = result[0]
    assert row["target"] == target
    assert [s["step"] for s in row["steps"]] == [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    assert row["original_failures_preserved"] is True
    assert row["original_step"]["step"] == 0.01
    assert row["steps"][0]["boundary_crossing_stencils"] == [0]
    assert all("adjacent_agreement" in s for s in row["steps"][1:])
    np.testing.assert_allclose(raw[row["steps"][1]["raw"]["analytic"]], [0, 0.5])
    assert "residual" in row["steps"][1]["raw"]


def test_plan_preserves_union_heads_inputs_points_and_predeclared_controls():
    historical = {"core": {"checks": [
        {"check": "spcas9/S2/gradient_check/alpha0.5/In_Seq/rel_l2", "passed": False,
         "measured": 0.1}]}, "diagnostics": {"checks": []}}
    historical["core"]["checks"] += [
        {"check": f"{model}/{case}/gradient_check/{point}/{name}/{metric}", "passed": True, "measured": value}
        for model, case, point, name in [("spcas9", "S3", "alpha0.5", "In_Seq"),
                                         ("cas12a", "C2", "x", "Input_Seq")]
        for metric, value in [("cosine", 1.0), ("rel_l2", 0.001)]]
    core = {"checks": [
        {"check": "cas12a/C4/gradient_check/x/Input_Feat/cosine", "passed": False,
         "measured": 0.9}]}
    diag = {"checks": [{"check": "cas12a/C3/directional_fd/max", "passed": False,
                        "measured": 0.12}]}
    import json
    contract = json.loads((Path(diagnostics.__file__).parent / "contract.json").read_text())
    targets = diagnostics.plan_locations(contract, historical, core, diag)
    failures = [t for t in targets if not t.get("control")]
    assert len(failures) == 3
    feature = next(t for t in failures if t["case"] == "C4")
    assert feature["input"] == 1 and feature["point"] == "x"
    direction = next(t for t in failures if t["kind"] == "directional")
    assert direction["point"] == "alpha0.5" and direction["directions"] == list(range(32))
    assert {t["model"] for t in targets if t.get("control")} == {"spcas9", "cas12a"}
    assert next(t for t in failures if t["model"] == "spcas9")["head"] == 1


def test_direction_sweep_retains_all_32_seeded_directions_and_no_extra_steps():
    target = {"id": "cas12a/C3/alpha0.5/directions", "case": "C3", "point": "alpha0.5",
              "kind": "directional", "head": 0, "original_failures": []}
    points = {"C3": [np.array([[0.0, 0.25]], np.float64)]}
    rows, raw = diagnostics.sweep(QuadraticRuntime(), [target], points, "float64")
    assert [s["step"] for s in rows[0]["steps"]] == [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 1e-5, 1e-6]
    original = rows[0]["original_step"]
    assert raw[original["raw"]["numeric"]].shape == (32,)
    np.testing.assert_allclose(raw[original["raw"]["analytic"]], raw[original["raw"]["numeric"]], atol=1e-12)


def test_original_seed193_normalized_direction_values_are_preserved_in_float32_diagnosis():
    target = {"id": "fixture/directions", "case": "S1", "point": "alpha0.5",
              "kind": "directional", "head": 0, "original_failures": []}
    rows, raw = diagnostics.sweep(QuadraticRuntime(), [target],
                                 {"S1": [np.array([[0.0, 0.5]], np.float32)]}, "float32")
    # Frozen #193 values under the registered NumPy version; normalizing the
    # float32 draws produces float64 values, which must not be rounded again.
    np.testing.assert_array_equal(raw["float32/fixture/directions/vectors/0"][0],
                                  [-0.9925739159576034, 0.12164314843280657])
    assert raw[rows[0]["original_step"]["raw"]["analytic"]][0] == 0.06082157421640329


def test_nonfinite_diagnostic_is_retained_as_required_failure():
    class InvalidRuntime(QuadraticRuntime):
        def gradients(self, arrays, head=0):
            return np.array([1.0]), [np.full_like(a, np.nan) for a in arrays]

    target = {"id": "fixture", "case": "S1", "point": "x", "input": 0,
              "kind": "coordinate", "head": 0, "original_failures": []}
    rows, raw = diagnostics.sweep(InvalidRuntime(), [target],
                                 {"S1": [np.array([[0.25]], np.float32)]}, "float32")
    assert not diagnostics.all_finite(raw)
    assert not rows[0]["original_step"]["comparison"]["passed"]


@pytest.mark.parametrize("structure", ["dense", "lstm", "bidirectional"])
def test_qualified_copy_expands_weights_and_uses_actual_double_compute(structure):
    import non_nt_runtime as rt
    keras = rt.import_keras()
    inputs = keras.Input(shape=(2,) if structure == "dense" else (3, 2), name="test_input")
    hidden = inputs
    if structure == "lstm":
        hidden = keras.layers.LSTM(2, name="test_lstm")(hidden)
    elif structure == "bidirectional":
        hidden = keras.layers.Bidirectional(keras.layers.LSTM(2), name="test_bilstm")(hidden)
    output = keras.layers.Dense(1, activation="relu", name="test_output")(hidden)
    model = keras.Model(inputs, output)
    if structure == "dense":
        model.set_weights([np.array([[0.1], [0.2]], np.float32), np.array([0.3], np.float32)])
    before = [w.copy() for w in model.get_weights()]
    point = np.array([[1, 0]], np.float32) if structure == "dense" else np.ones((1, 3, 2), np.float32)
    runtime, evidence = diagnostics.qualify_float64(model, [point], 0)
    assert evidence["qualified"] and evidence["exact_weight_expansion"]
    assert set(evidence["compute_dtypes"]) == {"torch.float64"}
    assert evidence["gradient_dtypes"] == ["float64"]
    assert all(w["dtype"] == "float64" for w in evidence["weights"])
    for original, unchanged in zip(before, model.get_weights()):
        np.testing.assert_array_equal(original, unchanged)
        assert unchanged.dtype == np.float32
    if structure == "dense":
        boundary = runtime.boundary_values([np.array([[-10, 0], [10, 0]], np.float64)])
        assert boundary["test_output/preactivation"][0, 0] < 0
        assert boundary["test_output/preactivation"][1, 0] > 0


def test_input_only_double_is_not_a_qualified_runtime():
    import non_nt_runtime as rt
    keras = rt.import_keras()

    @keras.saving.register_keras_serializable(package="DiagnosticFixture")
    class Float32Compute(keras.layers.Layer):
        def call(self, inputs):
            return keras.ops.cast(inputs, "float32")

    inputs = keras.Input(shape=(2,))
    output = Float32Compute()(keras.layers.Dense(1)(inputs))
    model = keras.Model(inputs, output)
    with pytest.raises(ValueError, match="Actual compute/variable/gradient dtype"):
        diagnostics.qualify_float64(model, [np.ones((1, 2), np.float64)], 0)


def test_nonfinite_qualification_weights_have_an_explicit_nonfinite_reason():
    import non_nt_runtime as rt
    keras = rt.import_keras()
    inputs = keras.Input(shape=(2,))
    model = keras.Model(inputs, keras.layers.Dense(1)(inputs))
    model.set_weights([np.array([[np.nan], [0.0]], np.float32), np.array([0.0], np.float32)])
    with pytest.raises(ValueError, match="Nonfinite stored float32 weights"):
        diagnostics.qualify_float64(model, [np.ones((1, 2), np.float32)], 0)


def test_float64_child_executes_a_real_qualified_graph_without_changing_checkpoint(tmp_path):
    import non_nt_runtime as rt
    keras = rt.import_keras()
    work = tmp_path / "bundle"
    assets = work / "snapshot/assets"
    assets.mkdir(parents=True)
    x = keras.Input((2,), name="fixture_input")
    model = keras.Model(x, keras.layers.Dense(1, name="fixture_output")(x))
    model.set_weights([np.array([[0.125], [0.5]], np.float32), np.array([0.25], np.float32)])
    path = assets / "fixture.keras"
    model.save(path)
    checksum = diagnostics.digest(path)
    contract = {"models": {"spcas9_non_nt": {"path": "fixture.keras", "sha256": checksum,
                 "inputs": [{"shape": [2]}]}}, "cases": {"spcas9": [{"id": "S1"}]}}
    (work / "snapshot/contract.json").write_text(json.dumps(contract), encoding="utf-8")
    np.savez(work / "snapshot/inputs.npz", **{"spcas9/S1/input/0": np.array([[0.5, 1.0]], np.float32)})
    target = {"id": "spcas9/S1/x/fixture_input", "model": "spcas9", "case": "S1",
              "point": "x", "input": 0, "kind": "coordinate", "head": 0,
              "original_failures": [{"step": 0.01, "source": "fixture"}]}
    evidence, raw = diagnostics.run_float64(work, "spcas9", [target], time.monotonic() + 120)
    assert evidence["state"] == "unresolved", evidence
    assert evidence["qualification"]["qualified"] is True
    assert evidence["completed_steps"] == evidence["planned_steps"] == 8
    assert raw[evidence["locations"][0]["steps"][0]["raw"]["outputs"]].dtype == np.float64
    assert diagnostics.digest(path) == checksum
    with pytest.raises(FileExistsError):
        diagnostics.run_float64(work, "spcas9", [target], time.monotonic() + 120)


@pytest.mark.parametrize("outcome", ["timeout", "partial", "unavailable", "crash"])
def test_child_terminal_states_preserve_partial_evidence_and_crash_is_not_unavailable(tmp_path, monkeypatch, outcome):
    original_popen = subprocess.Popen
    # A real external process supplies controlled IPC/termination behavior.
    # The OS process boundary is the fixture seam; the runner is unmodified.
    def process(command, **kwargs):
        directory = Path(command[-1])
        script = ["import json,time,sys", "from pathlib import Path", f"p=Path({str(directory)!r})"]
        if outcome == "unavailable":
            script += ["(p/'qualification.json').write_text(json.dumps({'qualified':False,'reason':'unsupported op'}))"]
        elif outcome == "crash":
            script += ["sys.exit(7)"]
        else:
            script += ["(p/'qualification.json').write_text(json.dumps({'qualified':True}))"]
            if outcome == "partial":
                script += ["import numpy as np", "np.savez(p/'raw.npz', witness=[1.,2.])",
                           "(p/'progress.json').write_text(json.dumps({'locations':[{'steps':[{'step':0.03}]}]}))"]
            script += ["time.sleep(30)"]
        return original_popen([sys.executable, "-c", "\n".join(script)], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", process)
    if outcome == "crash":
        with pytest.raises(RuntimeError, match="Unexpected.*exit"):
            diagnostics.run_float64(tmp_path, "spcas9", [{}], time.monotonic() + 20,
                                   qualification_seconds=10, diagnosis_seconds=1)
        return
    evidence, raw = diagnostics.run_float64(tmp_path, "spcas9", [{}], time.monotonic() + 20,
                                           qualification_seconds=10, diagnosis_seconds=1)
    assert evidence["state"] == ("unavailable" if outcome == "unavailable" else "time_limited")
    assert evidence["planned_steps"] == 8
    assert evidence["completed_steps"] == (1 if outcome == "partial" else 0)
    if outcome == "partial":
        np.testing.assert_array_equal(raw["witness"], [1, 2])
    if outcome != "unavailable":
        assert evidence["termination"] == "planned_budget"


def test_total_budget_does_not_launch_a_new_child(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A child was started after the total budget")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    evidence, raw = diagnostics.run_float64(tmp_path, "cas12a", [{}], time.monotonic() - 1)
    assert evidence["state"] == "time_limited" and evidence["completed_steps"] == 0
    assert not raw


def report_fixture():
    targets = []
    rows, raw = [], {}
    for model in ("spcas9", "cas12a"):
        target = {"id": model + "/S1/x/input", "model": model, "case": "S1", "point": "x",
                  "input": 0, "kind": "coordinate", "head": 0, "original_failures": []}
        locations, arrays = diagnostics.sweep(QuadraticRuntime(), [target],
                            {"S1": [np.array([[0.0, 0.25]], np.float32)]}, "float32")
        targets.append(target)
        raw.update(arrays)
        rows += [{"id": "float32_" + model, "completed": 1, "state": "pass", "locations": locations},
                 {"id": "float64_" + model, "completed": 0, "state": "not_applicable", "locations": []}]
    return {"investigations": rows}, raw, targets


@pytest.mark.parametrize("fault", ["target", "step", "gradient_dtype", "residual", "comparison",
                                   "boundary", "adjacent", "inapplicable", "boundary_support"])
def test_report_rejects_lost_failure_locations_dtype_and_stencil_evidence(fault):
    document, raw, targets = report_fixture()
    diagnostics.validate_diagnostic_report(document, raw, targets)
    row = document["investigations"][0]
    location = row["locations"][0]
    step = location["steps"][1]
    if fault == "target":
        row["locations"] = []
    elif fault == "step":
        step["step"] = 0.1
    elif fault == "gradient_dtype":
        name = "float32/" + targets[0]["id"] + "/gradient/0"
        raw[name] = raw[name].astype(np.float64)
    elif fault == "residual":
        raw[step["raw"]["residual"]] += 1
    elif fault == "comparison":
        step["comparison"]["cosine"] = 0.1
    elif fault == "boundary":
        step["boundary_crossing_stencils"] = []
    elif fault == "adjacent":
        step.pop("adjacent_agreement")
    elif fault == "boundary_support":
        location["boundary_support"] = not location["boundary_support"]
    else:
        targets[0]["original_failures"] = [{"source": "fixture", "step": 0.01}]
        location["unresolved"] = True
    with pytest.raises(ValueError):
        diagnostics.validate_diagnostic_report(document, raw, targets)


@pytest.mark.parametrize("fault", [None, "missing_location", "dtype", "nonfinite", "inapplicable",
                                   "input_value", "coordinate_vectors", "boundary_omission"])
def test_diagnostic_inventory_and_q17_reach_existing_assembler_cli(tmp_path, fault):
    # The existing CLI fixture builder freezes real assets but supplies wholly
    # synthetic measurements. This never claims official model validation.
    from test_runtime_verification import build_complete_bundle, cli, load, write, refresh_receipt, sha
    docs = Path(diagnostics.__file__).parent
    work = build_complete_bundle(tmp_path, docs / "protocol-211.json", "fixture-211")
    contract = load(work / "snapshot/contract.json")
    protocol = load(work / "snapshot/protocol.json")
    targets = diagnostics.plan_locations(contract, load(work / "snapshot/diagnostic-history.json"),
                load(work / "verification-core.json"), load(work / "diagnose.json"))
    plan = {"targets": targets, "settings": protocol["diagnostic_protocol"],
            "boundary_layers": diagnostics.boundary_layer_inventory(work, contract),
            "history_sha256": sha(work / "snapshot/diagnostic-history.json"),
            "core_sha256": sha(work / "verification-core.json"), "diagnose_sha256": sha(work / "diagnose.json")}
    write(work / "diagnostic-plan.json", plan)
    document = load(work / "investigations.json")
    document["investigations"] = [r for r in document["investigations"] if not r["id"].startswith("float")]
    with np.load(work / "investigation-raw.npz") as archive:
        raw = {n: archive[n] for n in archive.files if not n.startswith("float")}
    for model in ("spcas9", "cas12a"):
        selected = [t for t in targets if t["model"] == model]
        with np.load(work / "snapshot/inputs.npz") as archive:
            points = {c["id"]: [archive[f"{model}/{c['id']}/input/{i}"]
                      for i in range(len(contract["models"][model + "_non_nt"]["inputs"]))]
                      for c in contract["cases"][model]}
        class BoundaryFixture(QuadraticRuntime):
            def boundary_values(self, arrays):
                return {name + "/preactivation": arrays[0]
                        for name in plan["boundary_layers"][model]}

        locations, arrays = diagnostics.sweep(BoundaryFixture(), selected,
                            points, "float32")
        name = "float32_" + model
        arrays[name] = np.array([len(selected) * 6, len(selected) * 6])
        raw.update(arrays)
        document["investigations"] += [
            {"id": name, "state": "unresolved", "planned": 1, "completed": 1,
             "locations": locations, "raw_arrays": list(arrays)},
            {"id": "float64_" + model, "state": "unavailable", "planned": 1, "completed": 0,
             "locations": [], "raw_arrays": [], "reason": "unsupported fixture operation",
             "qualification": {"qualified": False}, "completed_steps": 0,
             "budget": {"total_seconds": 1800, "model_order": ["spcas9", "cas12a"],
                        "start_offset": 0 if model == "spcas9" else 1,
                        "end_offset": 1 if model == "spcas9" else 2},
             "execution": {"qualification_limit_seconds": 300,
                           "diagnosis_limit_seconds": 600, "elapsed_seconds": 1},
             "planned_steps": sum(bool(t["original_failures"]) for t in selected) * 8}]
    first32 = next(r for r in document["investigations"] if r["id"] == "float32_spcas9")
    first64 = next(r for r in document["investigations"] if r["id"] == "float64_spcas9")
    if fault == "missing_location":
        first32["locations"].pop()
    elif fault == "dtype":
        name = "float32/" + first32["locations"][0]["target"]["id"] + "/gradient/0"
        raw[name] = raw[name].astype(np.float64)
    elif fault == "nonfinite":
        name = first32["locations"][0]["steps"][0]["raw"]["residual"]
        raw[name].reshape(-1)[0] = np.inf
    elif fault == "inapplicable":
        first64["state"] = "not_applicable"
    elif fault == "input_value":
        name = "float32/" + first32["locations"][0]["target"]["id"] + "/input/0"
        raw[name] = raw[name] + 0.25
    elif fault == "coordinate_vectors":
        name = "float32/" + first32["locations"][0]["target"]["id"] + "/vectors/0"
        raw[name] = raw[name][:-1]
    elif fault == "boundary_omission":
        for name in list(raw):
            if "/boundary/" in name:
                raw.pop(name)
        for row in document["investigations"]:
            row["raw_arrays"] = [n for n in row["raw_arrays"] if n in raw]
            for location in row.get("locations", []):
                for step in location["steps"]:
                    step.update(boundary_crossing_stencils=[], boundary_available=False)
    write(work / "investigations.json", document)
    np.savez_compressed(work / "investigation-raw.npz", **raw)
    refresh_receipt(work, "investigation")
    receipt = load(work / "investigation.receipt.json")
    receipt["auxiliary_artifacts"] = {"diagnostic-plan.json": sha(work / "diagnostic-plan.json")}
    write(work / "investigation.receipt.json", receipt)
    output = tmp_path / "report.json"
    result = cli("assemble_verification.py", "--work", work, "--output", output)
    report = load(output)
    assert result.returncode == 1, report
    assert report["status"]["overall"] == ("incomplete" if fault is None else "fail"), report


@pytest.mark.parametrize("fault", [None, "input_only", "weight_expansion", "head", "total_budget", "precision_effect"])
def test_completed_float64_report_checks_qualification_and_same_step_effects(fault):
    document, raw, targets = report_fixture()
    target = targets[0]
    target["head_name"] = "fixture_head"
    target["original_failures"] = [{"source": "fixture", "step": 0.01}]
    row32 = document["investigations"][0]
    row32["locations"][0]["unresolved"] = True
    row32["state"] = "unresolved"
    locations, arrays = diagnostics.sweep(QuadraticRuntime(), [target],
                          {"S1": [np.array([[0.0, 0.25]], np.float32)]}, "float64")
    raw.update(arrays)
    diagnostics.precision_effects(row32["locations"], locations, raw)
    qualification = {"qualified": True, "architecture_matched": True, "exact_weight_expansion": True,
        "inference_mode": True, "head_names": ["fixture_head"], "compute_dtypes": ["torch.float64"],
        "output_dtype": "float64", "gradient_dtypes": ["float64"],
        "weights": [{"path": "fixture/kernel", "dtype": "float64", "source_dtype": "float32",
                     "expanded_source_sha256": "a" * 64, "copy_sha256": "a" * 64}]}
    row64 = document["investigations"][1]
    row64.update(state="unresolved", completed=1, locations=locations, qualification=qualification,
        completed_steps=8, planned_steps=8,
        execution={"qualification_limit_seconds": 300, "diagnosis_limit_seconds": 600, "elapsed_seconds": 1},
        budget={"total_seconds": 1800, "start_offset": 0, "end_offset": 1, "model_order": ["spcas9", "cas12a"]})
    if fault == "input_only":
        qualification["compute_dtypes"] = ["torch.float32"]
    elif fault == "weight_expansion":
        qualification["weights"][0]["copy_sha256"] = "b" * 64
    elif fault == "head":
        qualification["head_names"] = ["other"]
    elif fault == "total_budget":
        row64["budget"]["end_offset"] = 1802
    elif fault == "precision_effect":
        locations[0]["steps"][0].pop("same_step_precision_effect")
    if fault is None:
        diagnostics.validate_diagnostic_report(document, raw, targets)
    else:
        with pytest.raises(ValueError):
            diagnostics.validate_diagnostic_report(document, raw, targets)
