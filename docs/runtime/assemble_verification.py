"""측정 산출물을 모아 `docs/runtime/verification.json`을 만든다 (#193).

요약을 손으로 쓰지 않는다. `adoption_claims_verified`와 `open_failures`를 실제
판정 목록에서 **유도**한다. 처음 이 요약을 손으로 적었을 때 등록된 실패 하나
(방향 차분의 최대값)가 요약에서 빠졌고, 리뷰가 그것을 잡았다. 유도하면 같은
방식으로 틀릴 수 없다.

실행 (`verify`, `diagnose`, `ram`, `stage_offline` 뒤에):

    python docs/runtime/assemble_verification.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "runtime"
WORK = ROOT / ".packaging" / "issue193"

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
}


def classify(checks: list[dict]) -> dict[str, dict]:
    """판정 목록을 주장별로 묶어 통과/실패를 센다."""
    groups: dict[str, dict] = {}
    for check in checks:
        parts = check["check"].split("/")
        claim = next((CLAIM_GROUPS[p] for p in parts if p in CLAIM_GROUPS), None)
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


def main() -> int:
    core = json.loads((WORK / "verification-core.json").read_text(encoding="utf-8"))
    diag = json.loads((WORK / "diagnose.json").read_text(encoding="utf-8"))
    ram = json.loads((WORK / "ram.json").read_text(encoding="utf-8"))
    staged = json.loads((WORK / "staged.json").read_text(encoding="utf-8"))

    groups = classify(core["checks"] + diag["checks"])
    groups["offline_tensorflow_free_run"] = {
        "checks": 1,
        "failed": 0 if staged["run"]["status"] == "pass" else 1,
        "failed_names": [] if staged["run"]["status"] == "pass" else ["stage_offline"],
        "status": staged["run"]["status"],
    }

    # per-layer 상세는 빼고 케이스별 최악값만 남긴다 (근거는 유지, 크기는 줄인다).
    measurements = {}
    for key, model in core["measurements"].items():
        cases = {}
        for case_id, case in model["cases"].items():
            trimmed = {k: v for k, v in case.items() if k != "layerwise"}
            trimmed["layerwise_layers_compared"] = len(case["layerwise"])
            cases[case_id] = trimmed
        measurements[key] = {**{k: v for k, v in model.items() if k != "cases"},
                             "cases": cases}

    report = {
        "schema_version": 2,
        "issue": 193,
        "measured_at": core["measured_at"],
        "contract_sha256": hashlib.sha256((DOCS / "contract.json").read_bytes()).hexdigest(),
        "machine": "development PC, Windows 11, CPU only, torch threads 4",
        "runtime": core["runtime"],
        "reference_runtime": core["reference_runtime"],
        "status": {
            "core": core["status"],
            "core_checks": core["n_checks"], "core_failed": core["n_failed"],
            "diagnostics": diag["status"],
            "diagnostic_checks": diag["n_checks"], "diagnostic_failed": diag["n_failed"],
            "offline_staged_run": staged["run"]["status"],
            "overall": "fail" if (core["n_failed"] or diag["n_failed"]) else "pass",
        },
        "claims": groups,
        "verified_claims": sorted(name for name, row in groups.items()
                                  if row["status"] == "pass"),
        "open_failures": {name: row for name, row in groups.items()
                          if row["status"] == "fail"},
        "corrections": [
            {
                "where": "contract.json addendum_1.why, and the first draft of this report",
                "said": "failed gradient_check on 9 of 32 case/point combinations",
                "actually": ("9 of 32 individual checks, which is 6 of 16 case/point "
                             "combinations - each combination contributes a cosine and a "
                             "rel_l2 check. The frozen text is left as written; this "
                             "records the miscount rather than editing the frozen record."),
            },
            {
                "where": "the first draft of this report's summary",
                "said": "the gradient diagnostics passed",
                "actually": ("cross_runtime_gradient_parity passed, but "
                             "directional_finite_difference failed its max bound in 3 of 16 "
                             "cases. The summary is now derived from the check list, so a "
                             "registered failure cannot be left out by hand."),
            },
            {
                "where": "the first verification run's gradient_check",
                "said": "one cosine and one rel_l2 per case/point",
                "actually": ("the two Cas12a inputs were pooled into a single vector, so "
                             "the sequence gradient the screen renders per base could hide "
                             "behind the ten-times-larger feature gradient. The criterion "
                             "says 'every model input', so each input is now gated "
                             "separately and the pooled value is kept alongside."),
            },
        ],
        "core": {"failed": core["failed"], "checks": core["checks"],
                 "measurements": measurements},
        "diagnostics": {"failed": diag["failed"], "checks": diag["checks"],
                        "measurements": diag["measurements"]},
        "ram": {
            "development_pc_with_tensorflow_installed": ram,
            "offline_staging_without_tensorflow": staged["run"]["rss"],
            "reading": ("With TensorFlow absent - the shipped configuration - one process "
                        "holding both models peaks at "
                        f"{staged['run']['rss']['peak_working_set']} bytes. On this machine "
                        "TensorFlow is installed, so the Bidirectional deserialization probe "
                        "imports it and the same Cas12a path reports "
                        f"{ram['cas12a']['peak_working_set']} bytes. The difference is the "
                        "probe, not the adopted runtime. Full UI + computation + WebView2 "
                        "tree RAM is #194."),
        },
        "offline_staging": {
            "command": "python docs/runtime/stage_offline.py <staging path>",
            "staging": staged["staging"],
            "run": {k: v for k, v in staged["run"].items() if k != "rss"},
            "method": ("Dependencies copied from the local site-packages with no network, "
                       "then run with python -S so site-packages is off sys.path and "
                       "TensorFlow is not importable at all. The network_probes strings are "
                       "this script's own in-process socket blocker reporting back; the "
                       "independent evidence next to them is the real ModuleNotFoundError "
                       "for tensorflow. OS-level blocking and a clean offline PC are #204."),
        },
        "evidence_files": {
            "full_core_report": ".packaging/issue193/verification-core.json",
            "full_diagnostics": ".packaging/issue193/diagnose.json",
            "reference_backend_values": ".packaging/issue193/reference.npz + reference.json",
            "staged_run": ".packaging/issue193/staged.json",
            "note": (".packaging is Git-excluded. Regenerate with "
                     "docs/runtime/verify_contract.py (reference, verify, diagnose, ram), "
                     "docs/runtime/stage_offline.py, then this script."),
        },
    }
    out = DOCS / "verification.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"overall": report["status"]["overall"],
                      "verified": report["verified_claims"],
                      "open_failures": sorted(report["open_failures"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
