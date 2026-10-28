"""Final #88 execution readiness and separate implementation/V2-A/V2-B reports."""

import math
from pathlib import Path

from SpCas9.spcas9_budget import RUN_ORDER, read
from SpCas9.spcas9_data import BASE_DIR, HEAD_NAMES
from SpCas9.spcas9_execution import reserve_destination, source_record
from SpCas9.spcas9_finalist import _seal, _sealed
from SpCas9.spcas9_split import file_sha256
from SpCas9.spcas9_study_inputs import StudyInputs, checked_bytes


def scanner_result(path: Path | None, checkpoint_sha256: str) -> dict:
    """Verify independent stored-v2 parity evidence; never gate V2-A fits on it."""
    if path is None:
        return {"status": "incomplete", "reason": "scanner evidence not supplied"}
    try:
        record = read(Path(path))
        if (
            record["checkpoint_sha256"] != checkpoint_sha256
            or record["scanner_sha256"]
            != file_sha256(BASE_DIR.parent / "multi_cas_scanner_수정.py")
            or record["rtol"] != 1e-5
            or record["atol"] != 1e-6
            or {case["strand"] for case in record["cases"]} != {"+", "-"}
        ):
            raise ValueError("scanner identity or parity evidence differs")
        if record["passed"] is not True:
            return {
                "status": "failed",
                "reason": "stored-v2 scanner parity check failed",
                "sha256": file_sha256(Path(path)),
            }
        for case in record["cases"]:
            if set(case["heads"]) != set(HEAD_NAMES):
                raise ValueError("scanner evidence requires all nine heads")
            for values in case["heads"].values():
                direct, adapted = values["direct"], values["scanner"]
                if not (
                    math.isfinite(direct)
                    and math.isfinite(adapted)
                    and abs(adapted - direct) <= 1e-6 + 1e-5 * abs(direct)
                ):
                    return {
                        "status": "failed",
                        "reason": "stored-v2 scanner numerical parity failed",
                        "sha256": file_sha256(Path(path)),
                    }
        return {
            "status": "verified",
            "path": str(Path(path).resolve()),
            "sha256": file_sha256(Path(path)),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"status": "incomplete", "reason": f"{type(error).__name__}: {error}"}


def research_report(
    study,
    destination: Path,
    *,
    validation: Path | None = None,
    test: Path | None = None,
    scanner: Path | None = None,
) -> dict:
    """Rebuild available verified evidence without fitting or reselecting a model."""
    accounting_error = None
    try:
        config = study.execution_configuration()
    except (OSError, ValueError, KeyError) as error:
        # A damaged journal forbids training without suppressing independent
        # reporting. Protocol/study pins must still verify these config bytes.
        config = read(study.path / "execution-config.json")
        if config["protocol"] != study._protocol():
            raise ValueError("unverifiable execution configuration") from error
        accounting_error = f"{type(error).__name__}: {error}"
    inputs = StudyInputs.from_record(config["protocol"]["inputs"])
    destination = reserve_destination(
        Path(destination), (study.path, inputs.source_path, inputs.split_dir)
    )
    unavailable = {}
    if accounting_error is not None:
        unavailable["budget"] = accounting_error
    stored = None
    if validation is not None:
        try:
            previous = _sealed(Path(validation), "validation.json")
            identity = previous["identity"]["stored_v2"]
            if identity is not None:
                for key in ("input_predictions", "input_sidecar"):
                    checked_bytes(Path(identity[key]), identity[f"{key}_sha256"])
                stored = (
                    Path(identity["input_predictions"]),
                    Path(identity["input_sidecar"]),
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            unavailable["validation"] = f"{type(error).__name__}: {error}"
    try:
        rebuilt = study.validation_report(
            destination / "validation",
            stored_predictions=stored[0] if stored else None,
            stored_sidecar=stored[1] if stored else None,
        )
    except (OSError, ValueError, KeyError, PermissionError) as error:
        unavailable["validation_reconstruction"] = f"{type(error).__name__}: {error}"
        rebuilt = {
            "decision": "incomplete",
            "bootstrap": {
                "status": "unavailable",
                "reason": str(error),
                "valid_draws": 0,
                "attempts": 0,
            },
            "attribution": {
                "paired_seeds": "fixed three seeds under common NGG checkpoint policy",
                "representative": "single candidate seed42 vs stored v2; ranking and selection/stopping/LR policy changes combined",
            },
        }
    outcome = "incomplete"
    test_result = {
        "status": "not_evaluated",
        "reason": "both validation comparisons and a finalist freeze are required",
    }
    if rebuilt["decision"] == "fail":
        outcome = "negative_validation"
    elif rebuilt["decision"] == "pass" and test is not None:
        try:
            test_result = study.test_report(Path(test))
            outcome = {
                "pass": "improved",
                "fail": "negative_test",
                "incomplete": "incomplete",
            }.get(test_result["decision"], "inconclusive")
        except (OSError, ValueError, KeyError, PermissionError) as error:
            unavailable["test"] = f"{type(error).__name__}: {error}"
    if unavailable:
        outcome = "incomplete"
    try:
        budget = study.budget_status()
    except (OSError, ValueError, KeyError) as error:
        unavailable["budget"] = f"{type(error).__name__}: {error}"
        budget = {
            "attempts": [],
            "training_seconds": None,
            "remaining_seconds": None,
            "evaluation_seconds": None,
            "bootstrap_seconds": None,
            "blocked": True,
            "accounting": "unknown; corrupt or missing attempt/time journal; training forbidden",
        }
    if budget["blocked"]:
        outcome = "incomplete"
    planned = []
    for expected in RUN_ORDER:
        attempt = next(
            (
                a
                for a in budget["attempts"]
                if all(a[k] == v for k, v in expected.items())
            ),
            None,
        )
        planned.append(
            {
                **expected,
                **(
                    attempt
                    or {
                        "status": "unknown"
                        if "budget" in unavailable
                        else "not_started"
                    }
                ),
            }
        )
    implementation = {
        "status": "verification_only"
        if config["mode"] == "verification"
        else "incomplete"
    }
    if config["mode"] == "full":
        try:
            frozen = _sealed(study.path / "pre-fit", "freeze.json")
            for key, expected_identity in (
                (
                    "execution_config_sha256",
                    file_sha256(study.path / "execution-config.json"),
                ),
                ("protocol_sha256", file_sha256(study.path / "protocol.json")),
                ("source", config["source"]),
                ("environment", config["environment"]),
            ):
                if frozen[key] != expected_identity:
                    raise ValueError("pre-fit freeze identity differs from this study")
            readiness = Path(frozen["readiness"]["path"])
            checked_bytes(readiness / "readiness.json", frozen["readiness"]["sha256"])
            recorded = _sealed(readiness, "readiness.json")
            for name in ("tests.xml", "tests.log"):
                checked_bytes(readiness / name, recorded["files"][name])
            if (
                recorded["source"] != config["source"]
                or recorded["environment"] != config["environment"]
            ):
                raise ValueError("pre-fit readiness identity differs from this study")
            implementation = {"status": "complete", "readiness": frozen["readiness"]}
        except (OSError, ValueError, KeyError) as error:
            implementation["reason"] = str(error)
        if implementation["status"] != "complete":
            outcome = "incomplete"
    report = {
        "issue": 88,
        "execution_mode": config["mode"],
        "implementation": implementation,
        "v2_a": {"status": outcome, "unresolved": outcome != "improved"},
        "v2_b": scanner_result(
            scanner, config["acceptance"]["historical_checkpoint_sha256"]
        ),
        "planned_runs": planned,
        "budget": budget,
        "validation": rebuilt,
        "test": test_result,
        "supplementary_ci": rebuilt["bootstrap"],
        "unavailable_evidence": unavailable,
        "selected_checkpoint": "candidate seed42"
        if outcome == "improved"
        else "stored v2",
        "test_exposure": "historically inspected; not new external confirmation",
        "attribution": rebuilt["attribution"],
        "limitations": "fixed seeds 42/43/44 and target-resampling evidence only; no arbitrary-seed or external-population guarantee",
        "deployment_authorized": False,
        "source_study": str(study.path.resolve()),
        "execution_config_sha256": file_sha256(study.path / "execution-config.json"),
        "reconstruction_source": source_record(),
    }
    _seal(report, destination, "results.json")
    return report
