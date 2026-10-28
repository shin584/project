"""Execute the single frozen #79 study, or recover reports without retraining.

From xai_test: python -m SpCas9.spcas9_research_run --help
No search, pilot fit, retry, resume, deployment or checkpoint replacement.
"""

import argparse
import logging
from pathlib import Path

from SpCas9.spcas9_budget import RUN_ORDER, read
from SpCas9.spcas9_data import BASE_DIR
from SpCas9.spcas9_execution import reserve_destination
from SpCas9.spcas9_split import write_json
from SpCas9.spcas9_study import StudyProtocol
from SpCas9.spcas9_study_inputs import StudyInputs

logger = logging.getLogger(__name__)


def execute(
    destination: Path,
    *,
    readiness: Path,
    synthetic_ledger: Path,
    scanner: Path | None = None,
    retry_from: Path | None = None,
    retry_authorization: str | None = None,
) -> dict:
    """One fresh execution; failed fits still produce partial evidence reports."""
    from SpCas9.spcas9_eval import CHECKPOINT_PATH
    from SpCas9.spcas9_eval import main as evaluate_stored

    destination = reserve_destination(
        Path(destination), (StudyInputs().split_dir, CHECKPOINT_PATH)
    )
    study = StudyProtocol.plan(
        destination / "studies",
        study_id="ngg-issue88",
        checkpoint_selection="ngg",
        execution_mode="full",
        synthetic_ledger=synthetic_ledger,
        retry_from=retry_from,
        retry_authorization=retry_authorization,
    )
    study.freeze_execution(readiness=readiness, stored_checkpoint=CHECKPOINT_PATH)
    runs = [
        study.plan_run(
            recipe=str(plan["recipe"]),
            seed=int(plan["seed"]),
            run_id=f"{plan['recipe']}-{plan['seed']}",
        )
        for plan in RUN_ORDER
    ]
    errors = {}
    for run_id in runs:
        print(f"Starting {run_id}", flush=True)
        try:
            study.train_validation(run_id)
        except (Exception, KeyboardInterrupt) as error:
            logger.exception("Full fit stopped; all later fits are prohibited")
            errors[run_id] = f"{type(error).__name__}: {error}"
            print(f"Stopped all subsequent fits: {errors[run_id]}", flush=True)
            break
    # Evaluation/reporting remains independent of training failure. It is
    # recorded separately and may later be recovered in another destination.
    stored = destination / "stored-v2-validation"
    try:
        with study.evaluation_timing():
            evaluate_stored(
                [
                    "--split",
                    "val",
                    "--checkpoint",
                    str(CHECKPOINT_PATH),
                    "--output-dir",
                    str(stored),
                ]
            )
    except (Exception, KeyboardInterrupt) as error:
        logger.exception("Stored-v2 validation evaluation unavailable")
        errors["stored_validation"] = f"{type(error).__name__}: {error}"
    validation = destination / "validation"
    validation_result = None
    try:
        validation_result = study.validation_report(
            validation,
            stored_predictions=stored / "predictions.csv"
            if (stored / "predictions_checksum.json").is_file()
            else None,
        )
    except (Exception, KeyboardInterrupt) as error:
        logger.exception("Validation reporting incomplete")
        errors["validation"] = f"{type(error).__name__}: {error}"
    test = None
    if validation_result is not None and validation_result["decision"] == "pass":
        try:
            study.freeze_selection(validation, stored_checkpoint=CHECKPOINT_PATH)
            test = destination / "test"
            study.evaluate_test(test)
        except (Exception, KeyboardInterrupt) as error:
            logger.exception("Guarded Test evaluation incomplete")
            errors["test"] = f"{type(error).__name__}: {error}"
    write_json(errors, destination / "execution-errors.json")
    report = study.research_report(
        destination / "report",
        validation=validation if validation_result is not None else None,
        test=test,
        scanner=scanner,
    )
    print(
        f"Implementation={report['implementation']['status']}; V2-A={report['v2_a']['status']}; V2-B={report['v2_b']['status']}",
        flush=True,
    )
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser(
        "verify", help="only essential implementation fixtures; no full suite or fits"
    )
    verify.add_argument("--destination", type=Path, required=True)
    run = commands.add_parser(
        "execute", help="one fresh frozen study; never retries or resumes"
    )
    run.add_argument("--destination", type=Path, required=True)
    run.add_argument("--readiness", type=Path, required=True)
    run.add_argument("--synthetic-ledger", type=Path, required=True)
    run.add_argument("--retry-from", type=Path)
    run.add_argument("--retry-authorization")
    run.add_argument(
        "--scanner",
        type=Path,
        default=BASE_DIR / "docs/evidence/issue81-scanner-v2-parity.json",
    )
    report = commands.add_parser(
        "report", help="fresh report from completed verified evidence; no fitting"
    )
    report.add_argument("--study", type=Path, required=True)
    report.add_argument("--destination", type=Path, required=True)
    report.add_argument("--validation", type=Path)
    report.add_argument("--test", type=Path)
    report.add_argument("--scanner", type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify":
        from SpCas9.spcas9_readiness import verify_implementation

        return verify_implementation(args.destination)
    if args.command == "execute":
        return execute(
            args.destination,
            readiness=args.readiness,
            synthetic_ledger=args.synthetic_ledger,
            scanner=args.scanner,
            retry_from=args.retry_from,
            retry_authorization=args.retry_authorization,
        )
    study = StudyProtocol(args.study)
    # Reconcile only abandoned training records; this never authorizes fits.
    try:
        status = study.budget_status()
        if status["attempts"] and status["attempts"][-1]["status"] == "running":
            study.recover_reports(
                args.destination.with_name(args.destination.name + "-recovery")
            )
    except (OSError, ValueError, KeyError):
        # Preserve unknown accounting as a permanent training stop, while
        # reporting whatever independent evidence can still be verified.
        logger.exception("Recovery evidence incomplete; continuing partial reporting")
    result = study.research_report(
        args.destination,
        validation=args.validation,
        test=args.test,
        scanner=args.scanner,
    )
    print(read(args.destination / "results.json")["v2_a"])
    return result


if __name__ == "__main__":
    main()
