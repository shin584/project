"""#210: budgeted IG/SHAP candidate comparison at the experiment/report boundary.

Raw-result fixtures drive the registered producer (`candidate_investigation.
collect`) with a synthetic measurement and a controlled clock, and the bundle is
judged by the existing assembler CLI. These are harness regressions, not the
official measurement (#213).
"""

import json
import shutil
import sys

import numpy as np
import pytest

from test_runtime_verification import (ROOT, build_complete_bundle, cli, load,
                                       refresh_receipt, sha, write)

sys.path.insert(0, str(ROOT / "docs/runtime"))
import candidate_investigation  # noqa: E402
import candidate_policy  # noqa: E402

PROTOCOL = ROOT / "docs/runtime/protocol-210.json"


@pytest.fixture(scope="module")
def base_bundle(tmp_path_factory):
    return build_complete_bundle(tmp_path_factory.mktemp("candidate-fixture"), PROTOCOL,
                                 run_id="fixture-210")


@pytest.fixture
def bundle(tmp_path, base_bundle):
    work = tmp_path / "bundle"
    shutil.copytree(base_bundle, work)
    return work


class Synthetic:
    """Deterministic measurements; each call advances a controlled clock."""

    def __init__(self, noisy=lambda call: False, cost=None, values=None, fresh_shift=0.0,
                 ig_error=None, fail_at=None):
        self.now = 0.0
        self.noisy, self.values, self.fresh_shift = noisy, values, fresh_shift
        self.cost = cost or (lambda call: call["steps"] / 32 if call["kind"] == "ig"
                             else call["nsamples"] * call["n_background"] / 256)
        self.ig_error = ig_error or (lambda call: (0.32 if call["model"] == "spcas9" else 0.064)
                                     / call["steps"])
        self.fail_at, self.started = fail_at, []

    def clock(self):
        return self.now

    def measure(self, call):
        self.started.append(call["name"])
        if self.fail_at is not None and len(self.started) == self.fail_at:
            raise RuntimeError("synthetic interruption")
        self.now += self.cost(call)
        length = 30 if call["model"] == "spcas9" else 34
        base = np.sin(np.arange(length) + len(call["case"]) + ord(call["case"][1]))
        if call["kind"] == "ig":
            attribution = base * (1 + 0.5 / call["steps"])
            return {"attribution": attribution,
                    "terms": [1.0, 0.0, self.ig_error(call)]}
        seed = 1000 * call["background_seed"] + call["subset_seed"]
        noise = np.random.default_rng(seed).standard_normal(length)
        values = base + (2.0 if self.noisy(call) else 1e-4) * noise
        if self.values is not None:
            values = self.values(call, values)
        if call["name"].endswith("fresh00"):
            values = values + self.fresh_shift
        return {"values": values, "terms": [0.1, 0.1 + float(values.sum()), 0.0]}


def contract_and_protocol(work):
    return load(work / "snapshot/contract.json"), load(work / "snapshot/protocol.json")


def produce(work, synthetic, float_state="pass"):
    """Write the producer's report into a complete fixture bundle."""
    contract, protocol = contract_and_protocol(work)
    report, raw = candidate_investigation.collect(protocol, contract, synthetic.measure,
                                                  clock=synthetic.clock)
    for row in report["investigations"]:
        if row["id"].startswith("float"):
            # #211 rows are another producer's; fixture them independently.
            row.update(state=float_state, completed=1 if float_state == "pass" else 0,
                       raw_arrays=[row["id"]] if float_state == "pass" else [])
            if float_state == "pass":
                raw[row["id"]] = np.ones(1)
                row.pop("reason", None)
    report["identity"] = load(work / "investigations.json")["identity"]
    write(work / "investigations.json", report)
    np.savez_compressed(work / "investigation-raw.npz", **raw)
    refresh_receipt(work, "investigation")
    return report, raw


def rows(report):
    return {row["id"]: row for row in report["investigations"]}


def assemble(work, tmp_path):
    output = tmp_path / "report.json"
    result = cli("assemble_verification.py", "--work", work, "--output", output)
    assert output.exists(), result.stdout + result.stderr
    return result, load(output)


def default_noisy(call):
    return call["nsamples"] < 256 and call["n_background"] == 4


def test_registered_protocol_is_additive_and_frozen():
    base = load(ROOT / "docs/runtime/protocol-206.json")
    protocol = load(PROTOCOL)
    assert protocol["frozen_contract_sha256"] == sha(ROOT / "docs/runtime/contract.json")
    assert protocol["extends"]["sha256"] == sha(ROOT / "docs/runtime/protocol-206.json")
    for key in ("baseline", "diagnostic_protocol", "q17", "restart_rule"):
        assert protocol[key] == base[key]
    for name, definition in base["investigations"].items():
        for field, value in definition.items():
            assert protocol["investigations"][name][field] == value
    contract = load(ROOT / "docs/runtime/contract.json")
    plan = candidate_policy.initial_plan(protocol, contract)
    shap = [c for c in plan if c["kind"] == "shap"]
    assert len([c for c in plan if c["kind"] == "ig"]) == 32
    assert len(shap) == 184 and len({c["name"] for c in shap}) == 184
    for model in ("spcas9", "cas12a"):
        for case in candidate_policy.cases(contract, model):
            assert sum(c["model"] == model and c["case"] == case for c in shap) == 23
    # Every candidate fits the 240-call ceiling and its registered count.
    for ns, bg, axes in candidate_policy.screen_settings(protocol):
        candidate = {"nsamples": ns, "n_background": bg, "axes": axes,
                     "setting": candidate_policy.setting_key(ns, bg)}
        counts = [len(candidate_policy.confirmation_plan(protocol, contract, m, candidate))
                  for m in ("spcas9", "cas12a")]
        definition = protocol["investigations"]["shap_confirmation_spcas9"]
        assert counts[0] == definition.get("planned_by_candidate", {}).get(
            candidate["setting"], definition["planned"])
        assert 184 + sum(counts) <= protocol["candidate_protocol"]["max_shap_calls"]


def test_full_run_selects_candidates_and_q17_passes(bundle, tmp_path):
    synthetic = Synthetic(noisy=default_noisy)
    report, raw = produce(bundle, synthetic)
    found = rows(report)
    for model in ("spcas9", "cas12a"):
        ig = found[f"ig_{model}"]
        assert ig["state"] == "pass" and ig["completed"] == 16
        assert ig["candidate"]["steps"] == 64
        assert ig["candidate"]["label"] == "completeness_improvement_candidate"
        assert {r["steps"] for r in ig["resolution"]} == {32, 64, 128}  # 256 not self-compared
        screen = found[f"shap_screen_{model}"]
        assert screen["completed"] == 92 and screen["candidate"]["setting"] == "ns256/bg4"
        assert next(s for s in screen["settings"] if s["setting"] == "ns256/bg4")[
            "unverified_axes"] == ["background"]
        confirmation = found[f"shap_confirmation_{model}"]
        assert confirmation["state"] == "pass"
        assert confirmation["label"] == "seed_stability_confirmed_candidate"
        assert confirmation["completed"] == confirmation["planned"] == 28
        assert all(case["n_pairs"] == 36 for case in confirmation["cases"])
    assert report["candidate_summary"]["shap_calls"] == 240
    assert "termination" not in report
    # Fresh calls only: the fixture measured every call exactly once.
    assert len(synthetic.started) == len(set(synthetic.started)) == 32 + 240
    result, assembled = assemble(bundle, tmp_path)
    assert assembled["status"]["overall"] == "pass", assembled["open_failures"]
    assert result.returncode == 0


def test_without_the_211_producer_the_bundle_stays_incomplete(bundle, tmp_path):
    produce(bundle, Synthetic(noisy=default_noisy), float_state="unverified")
    result, report = assemble(bundle, tmp_path)
    assert report["status"]["overall"] == "incomplete"
    assert result.returncode == 1


def test_one_failing_case_blocks_a_setting_and_intersection_reuses_five(bundle, tmp_path):
    def noisy(call):
        return default_noisy(call) or (call["nsamples"] == 256 and call["case"] == "S4")
    report, _ = produce(bundle, Synthetic(noisy=noisy))
    found = rows(report)
    assert found["shap_screen_spcas9"]["candidate"]["setting"] == "ns512/bg4"
    assert found["shap_screen_cas12a"]["candidate"]["setting"] == "ns256/bg4"
    confirmation = found["shap_confirmation_spcas9"]
    assert confirmation["planned"] == confirmation["completed"] == 20
    assert len(confirmation["reused_from"]) == 4 * 5
    assert found["shap_confirmation_spcas9"]["state"] == "pass"
    result, assembled = assemble(bundle, tmp_path)
    assert assembled["status"]["overall"] == "pass", assembled["open_failures"]


@pytest.mark.parametrize("cost, expected", [
    (lambda call: 1.0, "ns256/bg4"),  # tie on wall time -> smaller ns*bg
    (lambda call: 0.5 if (call.get("nsamples"), call.get("n_background")) == (512, 16) else 1.0,
     "ns512/bg16"),  # wall time decides before the product
])
def test_deterministic_shap_tie_breaking(bundle, cost, expected):
    report, _ = produce(bundle, Synthetic(noisy=default_noisy, cost=cost))
    assert rows(report)["shap_screen_spcas9"]["candidate"]["setting"] == expected


def test_ig_tie_prefers_fewer_steps_and_resolution_is_reported_separately(bundle):
    synthetic = Synthetic(noisy=default_noisy, cost=lambda call: 1.0,
                          ig_error=lambda call: 0.0)
    report, _ = produce(bundle, synthetic)
    ig = rows(report)["ig_spcas9"]
    assert ig["candidate"]["steps"] == 32
    assert all(row["state"] == "pass" for row in ig["resolution"])


def test_ig_without_complete_setting_is_a_finding(bundle, tmp_path):
    synthetic = Synthetic(noisy=default_noisy,
                          ig_error=lambda call: 1.0 if call["case"] in ("S3", "C2") else 0.0)
    report, _ = produce(bundle, synthetic)
    assert rows(report)["ig_spcas9"]["state"] == "no_eligible_candidate"
    assert rows(report)["ig_spcas9"]["candidate"] is None
    result, assembled = assemble(bundle, tmp_path)
    assert assembled["status"]["overall"] == "completed_with_findings"
    assert result.returncode == 0


def test_undefined_correlation_cannot_pass(bundle, tmp_path):
    synthetic = Synthetic(values=lambda call, values: np.zeros_like(values)
                          if call["case"] == "S1" else values)
    report, _ = produce(bundle, synthetic)
    found = rows(report)
    screen = found["shap_screen_spcas9"]
    assert screen["state"] == "no_eligible_candidate"
    pairs = screen["settings"][0]["cases"][0]["axes"]["subset"]["pairs"]
    assert pairs[0]["pearson_r"] is None and not pairs[0]["passed"]
    assert found["shap_confirmation_spcas9"]["state"] == "not_applicable"
    assert found["shap_confirmation_spcas9"]["completed"] == 0
    assert not any(c["row"] == "shap_confirmation_spcas9" for c in report["candidate_calls"])
    result, assembled = assemble(bundle, tmp_path)
    assert assembled["status"]["overall"] == "completed_with_findings"
    assert result.returncode == 0


@pytest.mark.parametrize("fault", ["cross_seed", "fresh"])
def test_failed_confirmation_ends_search_without_another_candidate(bundle, tmp_path, fault):
    if fault == "cross_seed":
        synthetic = Synthetic(noisy=lambda call: default_noisy(call) or (
            call["background_seed"] and call["subset_seed"]))
    else:
        synthetic = Synthetic(noisy=default_noisy, fresh_shift=1e-6)
    report, _ = produce(bundle, synthetic)
    confirmation = rows(report)["shap_confirmation_spcas9"]
    assert confirmation["state"] == "no_eligible_candidate"
    assert "no other candidate" in confirmation["reason"]
    assert confirmation["completed"] == 28
    tested = {c["name"].split("/")[2] for c in report["candidate_calls"]
              if c["row"] == "shap_confirmation_spcas9"}
    assert len([c for c in report["candidate_calls"]
                if c["row"] == "shap_confirmation_spcas9"]) == 28 and tested == {"S1", "S2", "S3", "S4"}
    result, assembled = assemble(bundle, tmp_path)
    assert assembled["status"]["overall"] == "completed_with_findings"
    assert result.returncode == 0


def test_no_call_starts_after_the_budget_and_stop_is_planned(bundle, tmp_path):
    budget = load(bundle / "snapshot/protocol.json")["candidate_protocol"]["budget_seconds"]
    synthetic = Synthetic(noisy=default_noisy,
                          cost=lambda call: budget / 50 if call["kind"] == "shap" else 1.0)
    report, _ = produce(bundle, synthetic)
    calls = report["candidate_calls"]
    assert all(call["started_offset"] < budget for call in calls)
    # The call in flight at the boundary finished and was kept; nothing else began.
    assert calls[-1]["started_offset"] + calls[-1]["wall_seconds"] >= budget
    assert len(synthetic.started) == len(calls)
    assert report["termination"] == "planned_budget"
    assert report["candidate_budget"]["exhausted"] is True
    found = rows(report)
    assert found["ig_spcas9"]["state"] == "pass"
    screen = found["shap_screen_spcas9"]
    assert screen["state"] == "time_limited" and screen["termination"] == "planned_budget"
    assert 0 < screen["completed"] < 92
    assert found["shap_screen_cas12a"]["completed"] == 0
    assert found["shap_confirmation_spcas9"]["state"] == "time_limited"
    assert report["candidate_summary"]["unstarted_calls"][0] not in {c["name"] for c in calls}
    checksum = sha(bundle / "investigation-raw.npz")
    result, assembled = assemble(bundle, tmp_path)
    assert assembled["status"]["overall"] == "incomplete"
    assert result.returncode == 1
    assert assembled["investigation_counts"]["shap_screen_spcas9"] == {
        "completed": screen["completed"], "incomplete": 92 - screen["completed"]}
    assert sha(bundle / "investigation-raw.npz") == checksum
    again = cli("verify_contract.py", "investigation", "--work", bundle)
    assert again.returncode == 1 and sha(bundle / "investigation-raw.npz") == checksum


def test_unexpected_interruption_writes_no_result(tmp_path):
    synthetic = Synthetic(noisy=default_noisy, fail_at=40)
    contract = load(ROOT / "docs/runtime/contract.json")
    with pytest.raises(RuntimeError):
        candidate_investigation.collect(load(PROTOCOL), contract, synthetic.measure,
                                        clock=synthetic.clock)
    work = tmp_path / "not-a-bundle"
    work.mkdir()
    assert candidate_investigation.main([str(work)]) == 2
    assert list(work.iterdir()) == []


def test_aborted_producer_stage_cannot_resume_or_mix(tmp_path):
    """The registered producer aborting mid-collection aborts the journaled stage."""
    producer = tmp_path / "aborting_producer.py"
    producer.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'docs/runtime')!r})\n"
        "import candidate_investigation as ci\n"
        "def failing(call):\n"
        "    raise KeyboardInterrupt('synthetic kill')\n"
        "ci.real_measure = lambda contract, windows: (failing, {})\n"
        "raise SystemExit(ci.main(sys.argv[1:]))\n", encoding="utf-8")
    protocol = load(PROTOCOL)
    protocol["investigation_command"] = ["{python}", str(producer), "{work}"]
    path = tmp_path / "protocol-210-abort.json"
    write(path, protocol)
    work = tmp_path / "aborted"
    initialized = cli("verify_contract.py", "init", "--work", work, "--run-id", "abort-210",
                      "--protocol", path, "--extra-code", producer)
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    result = cli("verify_contract.py", "investigation", "--work", work)
    assert result.returncode == 1
    assert "aborted" in (work / "investigation-stderr.txt").read_text()
    assert load(work / "journal.json")["state"] == "aborted"
    assert not (work / "investigations.json").exists()
    assert not (work / "investigation.receipt.json").exists()
    again = cli("verify_contract.py", "investigation", "--work", work)
    assert again.returncode == 1 and "new bundle" in again.stdout
    assert not (work / "investigations.json").exists()


def _edit_report(work, change):
    path = work / "investigations.json"
    report = load(path)
    change(report)
    write(path, report)
    refresh_receipt(work, "investigation")


@pytest.mark.parametrize("fault", ["state", "candidate", "extra_search", "late_call",
                                   "unplanned_stop", "reuse_copy"])
def test_report_rejects_rows_not_derived_from_raw_evidence(bundle, tmp_path, fault):
    produce(bundle, Synthetic(noisy=lambda call: default_noisy(call) or (
        call["background_seed"] and call["subset_seed"])))
    if fault == "state":
        _edit_report(bundle, lambda r: rows(r)["shap_confirmation_spcas9"].update(
            state="pass", label="seed_stability_confirmed_candidate"))
    elif fault == "candidate":
        _edit_report(bundle, lambda r: rows(r)["shap_screen_cas12a"]["candidate"].update(
            setting="ns512/bg32"))
    elif fault == "extra_search":
        _edit_report(bundle, lambda r: r["candidate_calls"].append(
            {"name": "confirmation/spcas9/S1/background1/subset1/retry", "row": "shap_confirmation_spcas9",
             "started_offset": 1.0, "wall_seconds": 1.0}))
    elif fault == "late_call":
        _edit_report(bundle, lambda r: r["candidate_calls"][-1].update(started_offset=10 ** 6))
    elif fault == "unplanned_stop":
        def drop(report):
            report["candidate_calls"].pop()
        _edit_report(bundle, drop)
    else:
        with np.load(bundle / "investigation-raw.npz") as archive:
            raw = {name: archive[name] for name in archive.files}
        name = "confirmation/spcas9/S1/background0/subset1/values"
        raw[name] = raw[name] + 1e-3
        np.savez_compressed(bundle / "investigation-raw.npz", **raw)
        refresh_receipt(bundle, "investigation")
    result, assembled = assemble(bundle, tmp_path)
    assert assembled["status"]["overall"] == "fail"
    assert result.returncode == 1
    assert any("andidate" in e or "Reused" in e for e in assembled["open_failures"]["evidence"])


def test_required_failure_dominates_candidate_findings(bundle, tmp_path):
    produce(bundle, Synthetic(noisy=lambda call: True))
    path = bundle / "verification-core.json"
    doc = load(path)
    doc["checks"][0]["passed"] = False
    doc["n_failed"], doc["status"] = 1, "fail"
    write(path, doc)
    receipt = load(bundle / "verify.receipt.json")
    receipt["exit_code"] = 1
    write(bundle / "verify.receipt.json", receipt)
    refresh_receipt(bundle, "verify")
    result, assembled = assemble(bundle, tmp_path)
    assert assembled["status"]["overall"] == "fail"
    assert result.returncode == 1


def test_registered_producer_runs_real_models_and_stops_at_budget(tmp_path):
    """Real adopted runtime through the journaled stage, with a 1 s budget copy."""
    protocol = load(PROTOCOL)
    protocol["candidate_protocol"]["budget_seconds"] = 1.0
    path = tmp_path / "protocol-210-smoke.json"
    write(path, protocol)
    work = tmp_path / "smoke"
    initialized = cli("verify_contract.py", "init", "--work", work, "--run-id", "smoke-210",
                      "--protocol", path)
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    result = cli("verify_contract.py", "investigation", "--work", work)
    assert result.returncode == 0, result.stdout + (work / "investigation-stderr.txt").read_text()
    journal = load(work / "journal.json")
    assert journal["state"] == "ready" and journal["completed"] == ["investigation"]
    report = load(work / "investigations.json")
    assert report["termination"] == "planned_budget"
    assert report["runtime"]["backend"] == "torch"
    calls = report["candidate_calls"]
    assert calls and calls[0]["name"] == "ig/spcas9/S1/steps32"
    assert all(call["started_offset"] < 1.0 for call in calls)
    with np.load(work / "investigation-raw.npz") as archive:
        raw = {name: archive[name] for name in archive.files}
    sys.path.insert(0, str(ROOT / "docs/runtime"))
    import assemble_verification
    assert assemble_verification.candidate_evidence_errors(
        report, raw, load(work / "snapshot/protocol.json"),
        load(work / "snapshot/contract.json")) == []
    assert rows(report)["ig_spcas9"]["state"] == "time_limited"
    assert rows(report)["float32_spcas9"]["state"] == "unverified"
