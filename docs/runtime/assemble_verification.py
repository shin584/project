"""Assemble an immutable verification bundle without measuring or using live inputs.

python docs/runtime/assemble_verification.py --work <bundle> --output <report.json>
Historical #193 evidence is preserved and cannot be adopted into a new run.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from verification_bundle import (STAGES, digest, identity, load_manifest,
                                 read_json, verify_receipt)
from verification_policy import (MODELS, VERSIONS, RAM_STAGES, all_finite, expected_arrays,
                                 expected_checks, inventory_failures,
                                 investigation_outcome)

EVIDENCE_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError,
                   AttributeError, EOFError, zipfile.BadZipFile)

#: 판정 이름 조각 → 요약에서 쓸 주장 이름. 한 조각이라도 실패하면 그 주장은 실패다.
CLAIM_GROUPS = {
    "forward_parity": "forward_parity_vs_tensorflow_backend",
    "layerwise_parity": "layerwise_parity_including_lstm_and_bilstm",
    "weight_placement": "weight_placement_bitwise",
    "function_match": "ig_path_function_match",
    "cross_runtime_gradient": "cross_runtime_gradient_parity",
    "finiteness": "finiteness",
    "gradient_check": "coordinate_finite_difference_gradient_check",
    "directional_fd": "directional_finite_difference",
    "ig_completeness": "ig_completeness_at_user_default_steps",
    "ig_resolution": "ig_resolution_stability",
    "shap_additivity": "shap_additivity",
    "shap_same_seed_determinism": "shap_same_seed_determinism",
    "shap_cross_seed": "shap_cross_seed_stability",
    "shap_default_vs_finer": "shap_default_vs_finer_agreement",
    "version": "runtime_versions_match_contract",
    "runtime": "runtime_versions_match_contract",
    "layerwise": "layerwise_parity_including_lstm_and_bilstm",
    "offline_tensorflow_free_run": "offline_tensorflow_free_run",
}


def classify(checks: list[dict]) -> dict[str, dict]:
    """판정 목록을 주장별로 묶어 통과/실패를 센다."""
    groups: dict[str, dict] = {}
    for check in checks:
        parts = check["check"].split("/")
        claim = next((CLAIM_GROUPS[p] for p in parts if p in CLAIM_GROUPS), None)
        if "ig_completeness" in parts and parts[-1] == "finer":
            claim = "ig_completeness_at_finer_steps"
        if claim is None:
            claim = f"unclassified:{check['check']}"
        row = groups.setdefault(claim, {"checks": 0, "failed": 0, "failed_names": []})
        row["checks"] += 1
        if not check["passed"]:
            row["failed"] += 1
            row["failed_names"].append(check["check"])
    for row in groups.values():
        row["status"] = "pass" if row["failed"] == 0 else "fail"
    return groups



def assemble(work: Path) -> dict:
    report = {"schema_version": 3, "issue": 206, "status": {"overall": "fail"},
              "claims": {}, "verified_claims": [], "open_failures": {},
              "evidence_errors": [], "investigation_counts": {}}
    errors, checks = report["evidence_errors"], []
    try:
        manifest = load_manifest(work)
        report["run_id"] = manifest["run_id"]
        report["manifest_sha256"] = digest(work / "manifest.json")
        report["contract_sha256"] = digest(work / "snapshot/contract.json")
        contract = read_json(work / "snapshot/contract.json")
        protocol = read_json(work / "snapshot/protocol.json")
        if protocol["frozen_contract_sha256"] != report["contract_sha256"]:
            raise ValueError("Protocol/contract identity mismatch")
        expected = expected_checks(contract, manifest["topology"])
        arrays_expected = expected_arrays(contract, manifest["topology"])
        if (manifest["expected_checks"] != expected
                or manifest["expected_arrays"] != arrays_expected
                or manifest["stages"] != {k: list(v) for k, v in STAGES.items()}):
            raise ValueError("Frozen inventory disagrees with contract/protocol/topology")
        journal = read_json(work / "journal.json")
        if journal["state"] != "ready" or (work / "active-stage.json").exists():
            errors.append("Interrupted bundle; new bundle/full remeasurement required")
        report["execution"] = {}
        report["inventory"] = {}
        raw = {}
        documents = {}
        for stage, files in STAGES.items():
            try:
                receipt = verify_receipt(work, stage)
                report["execution"][stage] = receipt
                if stage not in journal["completed"]:
                    errors.append(f"Stage missing from journal: {stage}")
                document = read_json(work / files[0])
                documents[stage] = document
                if document.get("identity") != identity(work, stage):
                    errors.append(f"Result identity mismatch: {stage}")
                if not all_finite(document):
                    errors.append(f"Nonfinite result: {stage}")
                if stage in expected:
                    stage_checks = document.get("checks", [])
                    errors.extend(f"{stage}: {e}" for e in inventory_failures(stage_checks, expected[stage]))
                    checks.extend(stage_checks)
                    report["inventory"][stage] = {"expected": len(expected[stage]), "observed": len(stage_checks)}
                    failed = [r for r in stage_checks if r.get("passed") is not True]
                    if (document.get("n_checks") != len(stage_checks)
                            or document.get("n_failed") != len(failed)
                            or document.get("status") != ("fail" if failed else "pass")):
                        errors.append(f"Inconsistent check summary: {stage}")
                    if receipt["exit_code"] != (1 if failed else 0):
                        errors.append(f"Check result/exit mismatch: {stage}")
                elif receipt["exit_code"] != 0:
                    errors.append(f"Stage failed: {stage}")
                if len(files) > 1:
                    with np.load(work / files[1], allow_pickle=False) as archive:
                        raw[stage] = {name: archive[name] for name in archive.files}
                    for name, value in raw[stage].items():
                        if not value.size or not all_finite(value):
                            errors.append(f"Nonfinite/empty raw array: {stage}/{name}")
                    for name, shape in arrays_expected.get(stage, {}).items():
                        if name not in raw[stage]:
                            errors.append(f"Missing raw array: {stage}/{name}")
                        elif shape is not None and list(raw[stage][name].shape) != shape:
                            errors.append(f"Wrong raw array shape: {stage}/{name}")
                    if stage in arrays_expected:
                        for name in set(raw[stage]) - set(arrays_expected[stage]):
                            observed = any(name.startswith(f"{key}/observed/") for key in MODELS)
                            kink = stage == "diagnose" and any(
                                name.startswith(f"{key}/{c['id']}/kink/")
                                for key in MODELS for c in contract["cases"][key])
                            if not (stage in ("verify", "diagnose") and observed) and not kink:
                                errors.append(f"Unexpected raw identity: {stage}/{name}")
                    if stage in ("reference", "verify"):
                        want_layers = {n for n in arrays_expected[stage] if "/layer/" in n}
                        got_layers = {n for n in raw[stage] if "/layer/" in n}
                        if want_layers != got_layers:
                            errors.append(f"Raw layer set mismatch: {stage}")
            except EVIDENCE_ERRORS as error:
                errors.append(f"{stage}: {error}")

        if "verify" in documents:
            core = documents["verify"]
            report["core"] = core
            report["measured_at"] = core.get("measured_at")
            report["status"].update(core=core.get("status"), core_checks=core.get("n_checks"),
                                     core_failed=core.get("n_failed"))
            if core.get("contract_sha256") != report["contract_sha256"]:
                errors.append("Core/frozen contract identity mismatch")
            versions = core.get("runtime", {})
            report["runtime"] = versions
            for name in VERSIONS:
                if versions.get(name) != contract["runtime"]["versions"][name]:
                    errors.append(f"Runtime version missing/mismatch: {name}")
            if versions.get("backend") != "torch":
                errors.append("Target backend must be torch")
            if versions.get("torch_threads") != str(contract["runtime"]["threads"]["torch_num_threads"]):
                errors.append("Target thread configuration mismatch")
            measurements = core.get("measurements", {})
            if set(measurements) != set(MODELS):
                errors.append("Model inventory mismatch")
            for key, model_key in MODELS.items():
                model, spec = measurements.get(key, {}), contract["models"][model_key]
                if (model.get("verified_head") != spec["verified_head"]
                        or model.get("checkpoint_sha256") != spec["sha256"]
                        or model.get("input_names") != [i["name"] for i in spec["inputs"]]
                        or len(model.get("head_names", [])) != spec["n_outputs"]
                        or model.get("head_names", [])[spec["verified_head"]["index"]] != spec["verified_head"]["name"]):
                    errors.append(f"Model/head/input identity mismatch: {key}")
                cases = model.get("cases", {})
                if set(cases) != {c["id"] for c in contract["cases"][key]}:
                    errors.append(f"Case inventory mismatch: {key}")
                for case in contract["cases"][key]:
                    if cases.get(case["id"], {}).get("window") != case["window"]:
                        errors.append(f"Case/window identity mismatch: {key}/{case['id']}")
        if "reference" in documents:
            ref = documents["reference"]
            report["reference_runtime"] = {k: ref.get(k) for k in ("keras", "tensorflow", "backend")}
            if (ref.get("backend") != "tensorflow" or ref.get("keras") != "3.14.1"
                    or ref.get("tensorflow") != "2.21.0"):
                errors.append("Reference runtime must be the frozen TF backend")
            for key in MODELS:
                if set(ref["models"][key]["layer_names"]) != set(manifest["topology"][key]):
                    errors.append(f"Reference layer set mismatch: {key}")
                spec = contract["models"][MODELS[key]]
                if (ref["models"][key].get("verified_head") != spec["verified_head"]
                        or ref["models"][key].get("checkpoint_sha256") != spec["sha256"]):
                    errors.append(f"Reference model/head identity mismatch: {key}")
            if "verify" in documents and documents["verify"].get("reference_runtime") != report["reference_runtime"]:
                errors.append("Core/reference runtime identity mismatch")

        # Every layer gets BOTH bounds, independent of which is largest in absolute units.
        if "reference" in raw and "verify" in raw:
            for key in MODELS:
                for case in contract["cases"][key]:
                    prefix = f"{key}/{case['id']}"
                    for layer in manifest["topology"][key]:
                        name = f"{prefix}/layer/{layer}"
                        if name not in raw["reference"] or name not in raw["verify"]:
                            continue  # already a missing-array failure above
                        want, got = raw["reference"][name], raw["verify"][name]
                        if got.shape != want.shape:
                            errors.append(f"Layer shape mismatch: {name}")
                            continue
                        absolute = float(np.max(np.abs(got - want)))
                        relative = float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-6)))
                        if (not all_finite([absolute, relative])
                                or absolute > 1e-4 * max(1.0, float(np.max(np.abs(want))))
                                or relative > contract["criteria"]["layerwise_parity"]["max_rel_diff"]):
                            errors.append(f"Layer parity violation: {name}")

        if "offline" in documents:
            offline = documents["offline"].get("run", {})
            offline_status = offline.get("status")
            report["offline_staging"] = documents["offline"]
            report["status"]["offline_staged_run"] = offline_status
            offline_ok = (offline_status == "pass"
                          and report["execution"].get("offline", {}).get("exit_code") == 0
                          and offline.get("network_blocked") is True
                          and offline.get("tensorflow_importable") is False
                          and offline.get("tensorflow_in_sys_modules_after_load") is False)
            if set(offline.get("models", {})) != set(MODELS):
                offline_ok = False
            for key, model_key in MODELS.items():
                observed = offline.get("models", {}).get(key, {})
                if (observed.get("all_finite") is not True
                        or observed.get("n_outputs") != contract["models"][model_key]["n_outputs"]
                        or len(observed.get("gradient_abs_sum", [])) != len(contract["models"][model_key]["inputs"])
                        or not all(value > 0 for value in observed.get("gradient_abs_sum", []))):
                    offline_ok = False
            checks.append({"check": "offline_tensorflow_free_run", "passed": offline_ok})
            if not offline_ok:
                errors.append("Offline recovery failed")
        if "diagnose" in documents:
            report["diagnostics"] = documents["diagnose"]
            report["status"].update(diagnostics=documents["diagnose"].get("status"),
                                     diagnostic_checks=documents["diagnose"].get("n_checks"),
                                     diagnostic_failed=documents["diagnose"].get("n_failed"))
        if "ram" in documents:
            report["ram"] = documents["ram"]
            for key in MODELS:
                observed = documents["ram"].get(key, {})
                if (not all(observed.get(stage, 0) > 0 for stage in RAM_STAGES)
                        or observed.get("all_finite") is not True or observed.get("exit_code") != 0
                        or observed.get("held_models") != [key] or not observed.get("command")):
                    errors.append(f"Missing/failed RAM measurement: {key}")
        outcome = "incomplete"
        if "investigation" in documents:
            investigation = documents["investigation"]
            outcome, counts = investigation_outcome(investigation, protocol)
            report["investigation_counts"] = counts
            report["investigations"] = investigation["investigations"]
            for row in investigation["investigations"]:
                for name in row.get("raw_arrays", []):
                    if name not in raw.get("investigation", {}):
                        errors.append(f"Missing investigation raw array: {name}")
        report["claims"] = classify(checks)
        report["verified_claims"] = sorted(name for name, row in report["claims"].items()
                                            if row["status"] == "pass")
        report["open_failures"] = {name: row for name, row in report["claims"].items()
                                   if row["status"] == "fail"}
        report["status"]["overall"] = "fail" if errors or report["open_failures"] else outcome
    except EVIDENCE_ERRORS as error:
        errors.append(str(error))
    if errors:
        report["open_failures"]["evidence"] = errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, work = args.output.resolve(), args.work.resolve()
    # Reports may be rebuilt, but must never overwrite evidence or historical reports.
    if output.is_relative_to(work) or output == Path(__file__).with_name("verification.json"):
        parser.error("Output must be outside the immutable bundle and historical report")
    if output.exists():
        try:
            previous = read_json(output)
            if previous.get("schema_version") != 3 or previous.get("issue") != 206:
                raise ValueError("Not a generated #206 summary")
        except EVIDENCE_ERRORS:
            parser.error("Refusing to overwrite existing inputs or historical evidence")
    report = assemble(work)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["status"]))
    return 0 if report["status"]["overall"] in ("pass", "completed_with_findings") else 1


if __name__ == "__main__":
    raise SystemExit(main())
