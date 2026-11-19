"""#212 planned stage boundary through the existing journal/report CLIs."""

import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_runtime_verification import build_complete_bundle, cli, load, write, sha
from test_gradient_diagnostics import QuadraticRuntime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/runtime"))
import gradient_diagnostics as diagnostics
from verification_bundle import stage_execution


@pytest.mark.parametrize("fault", [None, "missing_raw", "changed_input", "interrupted"])
def test_preregistered_float32_boundary_preserves_required_evidence(tmp_path, monkeypatch, fault):
    docs = Path(diagnostics.__file__).parent
    work = build_complete_bundle(tmp_path, docs / "protocol-212.json", "fixture-212")
    # The builder supplies synthetic prior stages. Neither investigation has
    # started: publishing a completed float32 stage is a planned handoff.
    for stage in ("float32", "investigation"):
        manifest = load(work / "manifest.json")
        for name in [*manifest["stages"][stage], f"{stage}.receipt.json"]:
            (work / name).unlink()
    write(work / "journal.json", {"state": "ready", "completed": [
        s for s in manifest["stages"] if s not in ("float32", "investigation")]})
    contract = load(work / "snapshot/contract.json")
    protocol = load(work / "snapshot/protocol.json")
    targets = diagnostics.plan_locations(contract, load(work / "snapshot/diagnostic-history.json"),
        load(work / "verification-core.json"), load(work / "diagnose.json"))
    layers = diagnostics.boundary_layer_inventory(work, contract)
    rows, raw = [], {}
    for model in diagnostics.MODELS:
        class BoundaryFixture(QuadraticRuntime):
            def boundary_values(self, arrays):
                return {name + "/preactivation": arrays[0] for name in layers[model]}
        with np.load(work / "snapshot/inputs.npz") as inputs:
            points = {c["id"]: [inputs[f"{model}/{c['id']}/input/{i}"] for i in
                range(len(contract["models"][model + "_non_nt"]["inputs"]))]
                for c in contract["cases"][model]}
        selected = [t for t in targets if t["model"] == model]
        locations, arrays = diagnostics.sweep(BoundaryFixture(), selected, points, "float32")
        name = "float32_" + model
        arrays[name] = np.array([len(selected) * 6, len(selected) * 6])
        raw.update(arrays)
        rows.append({"id": name, "state": "unresolved", "planned": 1, "completed": 1,
                     "locations": locations, "raw_arrays": list(arrays)})
    write(work / "diagnostic-plan.json", {"targets": targets, "controls": diagnostics.CONTROLS,
        "settings": protocol["diagnostic_protocol"], "boundary_layers": layers,
        "history_sha256": sha(work / "snapshot/diagnostic-history.json"),
        "core_sha256": sha(work / "verification-core.json"),
        "diagnose_sha256": sha(work / "diagnose.json")})
    if fault == "missing_raw":
        raw.pop(rows[0]["locations"][0]["steps"][0]["raw"]["residual"])
    elif fault == "changed_input":
        name = "float32/" + targets[0]["id"] + "/input/0"
        raw[name] = raw[name] + 0.25
    with stage_execution(work, "float32") as result:
        write(work / "float32.json", {"investigations": rows})
        np.savez_compressed(work / "float32-raw.npz", **raw)
        result["exit_code"] = 0
    if fault == "interrupted":
        write(work / "journal.json", {"state": "running", "active": "investigation",
                                      "completed": load(work / "journal.json")["completed"]})
    output = tmp_path / "summary.json"
    result = cli("assemble_verification.py", "--work", work, "--output", output)
    report = load(output)
    assert result.returncode == 1
    assert report["status"]["overall"] == ("incomplete" if fault is None else "fail"), report
    if fault is None:
        assert not report["evidence_errors"], report
        assert report["investigation_counts"]["float32_spcas9"]["completed"] == 1
        assert report["investigation_counts"]["float64_spcas9"]["completed"] == 0
        assert "investigation" not in report["execution"]
        assert sha(work / "diagnostic-plan.json") == load(work / "float32.receipt.json")["auxiliary_artifacts"]["diagnostic-plan.json"]
        duplicate = cli("verify_contract.py", "float32", "--work", work)
        assert duplicate.returncode == 1
        assert "cannot resume" in duplicate.stdout
        # The existing child-process seam supplies unsupported qualification.
        # Followup must consume the completed float32 evidence unchanged.
        before = {p.name: sha(p) for p in work.glob("float32*") if p.is_file()}
        original_popen = subprocess.Popen
        def child(command, **kwargs):
            directory = Path(command[-1])
            script = ("from pathlib import Path; "
                f"Path({str(directory / 'qualification.json')!r}).write_text("
                "'{\"qualified\": false, \"reason\": \"unsupported fixture op\"}')")
            return original_popen([sys.executable, "-c", script], **kwargs)
        monkeypatch.setattr(subprocess, "Popen", child)
        followup, followup_raw = diagnostics.run_diagnostics(work, contract, protocol, precision="float64")
        assert followup[:2] == rows
        assert [r["state"] for r in followup[2:]] == ["unavailable", "unavailable"]
        assert before == {p.name: sha(p) for p in work.glob("float32*") if p.is_file()}
        for name, value in raw.items():
            np.testing.assert_array_equal(followup_raw[name], value)
