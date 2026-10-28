"""Approved study reporting seam, using worked prediction fixtures only."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_spcas9_budget import FixedBackend
from test_spcas9_budget import controlled_inputs as _controlled_inputs
from test_spcas9_budget import synthetic_ledger as _synthetic_ledger
from test_spcas9_study import inputs as _inputs

from SpCas9.spcas9_acceptance import build_report
from SpCas9.spcas9_data import HEAD_NAMES
from SpCas9.spcas9_split import file_sha256
from SpCas9.spcas9_study import StudyProtocol
from SpCas9.spcas9_train import split_record

controlled_inputs = _controlled_inputs
synthetic_ledger = _synthetic_ledger
inputs = _inputs


class ChangedDiagnosticCheckpointBackend(FixedBackend):
    def predict(self, checkpoint_path, inputs):
        predicted = super().predict(checkpoint_path, inputs)
        if len(inputs) == 29134:
            checkpoint_path.write_bytes(b"changed during training diagnostics")
        return predicted


# For n=25, these disjoint swaps have sum(d^2)=52, hence rho=0.98.
NEAR_PERFECT = list(range(25))
NEAR_PERFECT[0], NEAR_PERFECT[5] = 5, 0
NEAR_PERFECT[7], NEAR_PERFECT[8] = 8, 7


def predictions(offset=1.0, *, ordered=False, non_ngg_offset=None):
    rank = np.arange(25) if ordered else np.array(NEAR_PERFECT)
    frames = []
    for population, pam in (("ngg", "AGG"), ("non_ngg", "AAA")):
        frame = pd.DataFrame(
            {
                "row_index": np.arange(25) + (25 if population == "non_ngg" else 0),
                "sequence": ["AAAA" + "ACGT" * 5 + pam + "AAA"] * 25,
                "split": "val",
                **{f"{h}_true": np.arange(25) / 100 for h in HEAD_NAMES},
            }
        )
        error = non_ngg_offset if population == "non_ngg" else None
        for head in HEAD_NAMES:
            frame[f"{head}_pred"] = rank / 100 + (offset if error is None else error)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def runs(candidate_offset=0.98):
    return {
        "baseline": {s: predictions() for s in (42, 43, 44)},
        "candidate": {
            s: predictions(candidate_offset, ordered=True) for s in (42, 43, 44)
        },
    }


def two_protospacer_clusters(original):
    frame = pd.concat([original, original], ignore_index=True)
    frame["row_index"] = np.arange(len(frame))
    frame.loc[len(original) :, "sequence"] = frame.loc[
        len(original) :, "sequence"
    ].str.replace("ACGT" * 5, "TGCA" * 5)
    return frame


def test_both_comparisons_pass_at_inclusive_improvement_boundaries():
    report = build_report(runs(), predictions(), bootstrap=False)
    assert report["decision"] == "pass"
    assert report["candidate_test_eligible"] is True
    for comparison in report["comparisons"].values():
        assert comparison["decision"] == "pass"
        assert comparison["candidate"]["equal_head_ngg"]["mae"] == pytest.approx(0.98)
        assert comparison["reference"]["equal_head_ngg"]["spearman"] == pytest.approx(
            0.98
        )


@pytest.mark.parametrize(
    "offset,decision", [(0.98 - 1e-8, "pass"), (0.98 + 1e-8, "fail")]
)
def test_mae_improvement_on_either_side_of_boundary(offset, decision):
    assert (
        build_report(runs(offset), predictions(), bootstrap=False)["decision"]
        == decision
    )


@pytest.mark.parametrize(
    "offset,decision", [(1.05 - 1e-8, "pass"), (1.05, "pass"), (1.05 + 1e-8, "fail")]
)
def test_one_head_non_ngg_mae_protection_cannot_be_hidden_by_average(offset, decision):
    study = runs(0.9)
    for frame in study["candidate"].values():
        frame.loc[25:, "evoCas9_pred"] = np.arange(25) / 100 + offset
    report = build_report(study, predictions(), bootstrap=False)
    assert report["decision"] == decision
    assert report["comparisons"]["paired_seeds"]["checks"][0]["passed"]


@pytest.mark.parametrize(
    "change,decision", [("below", "fail"), ("equal", "pass"), ("above", "pass")]
)
def test_spearman_improvement_around_exact_boundary(change, decision):
    study = runs(0.9)
    if change != "equal":
        # One extra disjoint adjacent swap lowers rho by 1/1300.
        reference = list(NEAR_PERFECT)
        reference[10], reference[11] = 11, 10
        if change == "above":
            for frame in study["baseline"].values():
                for head in HEAD_NAMES:
                    frame.loc[:24, f"{head}_pred"] = np.array(reference) / 100 + 1
        else:
            for frame in study["candidate"].values():
                for head in HEAD_NAMES:
                    frame.loc[:24, f"{head}_pred"] = np.array(NEAR_PERFECT) / 100 + 0.9
    report = build_report(study, predictions(), bootstrap=False)
    assert report["comparisons"]["paired_seeds"]["decision"] == decision


@pytest.mark.parametrize("extra_swap,decision", [(False, "pass"), (True, "fail")])
def test_non_ngg_spearman_protection_is_inclusive(extra_swap, decision):
    study = runs(0.9)
    rank = list(NEAR_PERFECT)
    if extra_swap:
        rank[10], rank[11] = 11, 10
    for frame in study["baseline"].values():
        frame.loc[25:, "evoCas9_pred"] = np.arange(25) / 100 + 1
    for frame in study["candidate"].values():
        frame.loc[25:, "evoCas9_pred"] = np.array(rank) / 100 + 0.9
    assert (
        build_report(study, predictions(), bootstrap=False)["comparisons"][
            "paired_seeds"
        ]["decision"]
        == decision
    )


def test_relative_mae_uses_seed_means_before_ratios_and_preserves_every_run():
    study = {
        "baseline": {s: predictions(e) for s, e in zip((42, 43, 44), (0.1, 0.1, 2.8))},
        "candidate": {
            s: predictions(e, ordered=True)
            for s, e in zip((42, 43, 44), (0.19, 0.19, 2.53))
        },
    }
    report = build_report(study, predictions(), bootstrap=False)
    paired = report["comparisons"]["paired_seeds"]
    assert paired["decision"] == "pass"
    assert paired["candidate"]["equal_head_ngg"]["mae"] == pytest.approx(0.97)
    assert paired["reference"]["equal_head_ngg"]["mae"] == pytest.approx(1)
    assert report["runs"]["candidate"][42]["populations"]["ngg"]["heads"]["SpCas9"][
        "mae"
    ] == pytest.approx(0.19)
    assert report["runs"]["baseline"][42]["counts"] == {
        "n_rows": 50,
        "unique_30mers": 2,
        "unique_protospacers": 1,
    }
    assert "pearson" in report["runs"]["baseline"][43]["pooled"]


def test_representative_failure_blocks_a_passing_three_seed_mean():
    study = runs(0.9)
    report = build_report(study, predictions(0.8), bootstrap=False)
    assert report["comparisons"]["paired_seeds"]["decision"] == "pass"
    assert report["comparisons"]["representative"]["decision"] == "fail"
    assert report["candidate_test_eligible"] is False


@pytest.mark.parametrize("missing", ["seed", "stored", "metric"])
def test_missing_required_evidence_is_incomplete(missing):
    study, stored = runs(0.9), predictions()
    if missing == "seed":
        del study["candidate"][44]
    elif missing == "stored":
        stored = None
    else:
        study["candidate"][43]["evoCas9_pred"] = 1
    report = build_report(study, stored, bootstrap=False)
    assert report["decision"] == "incomplete"
    assert report["candidate_test_eligible"] is False


@pytest.mark.parametrize("error,passed", [(0, True), (1e-8, False)])
def test_zero_reference_mae_requires_zero_candidate_mae(error, passed):
    study = runs(0.9)
    for frame in study["baseline"].values():
        frame.loc[25:, "evoCas9_pred"] = np.arange(25) / 100
    for frame in study["candidate"].values():
        frame.loc[25:, "evoCas9_pred"] = np.arange(25) / 100 + error
    checks = build_report(study, predictions(), bootstrap=False)["comparisons"][
        "paired_seeds"
    ]["checks"]
    assert (
        next(c for c in checks if c["name"] == "non_ngg.evoCas9.mae")["passed"]
        is passed
    )


def test_cluster_bootstrap_keeps_all_observations_and_is_repeatable():
    study = runs(0.9)
    for recipe, seeds in study.items():
        for seed, original in seeds.items():
            frame = two_protospacer_clusters(original)
            if recipe == "candidate":
                frame.loc[50:, [f"{h}_pred" for h in HEAD_NAMES]] += 0.09
            seeds[seed] = frame
    first = build_report(study, None)
    second = build_report(copy.deepcopy(study), None)
    assert first["bootstrap"] == second["bootstrap"]
    ci = first["bootstrap"]
    assert ci["valid_draws"] == 2000
    assert ci["attempts"] == 2000
    assert ci["intervals"]["relative_ngg_mae_change"] == pytest.approx([-0.1, -0.01])


def test_saved_validation_report_reconstructs_verified_partial_evidence_without_test(
    controlled_inputs, synthetic_ledger, tmp_path
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    run = study.plan_run(seed=42, recipe="baseline")
    completed = study.train_validation(run, backend=FixedBackend())
    report = study.validation_report(tmp_path / "report")
    assert report["decision"] == "incomplete"
    assert report["candidate_test_eligible"] is False
    assert (
        report["run_evidence"]["baseline"][42]["history"]
        == completed["training"]["history"]
    )
    assert report["identity"]["source"] == study.execution_configuration()["source"]
    assert (
        report["identity"]["environment"]
        == study.execution_configuration()["environment"]
    )
    assert report["run_evidence"]["baseline"][42]["prediction_sha256"]
    assert (
        report["run_evidence"]["baseline"][42]["training_ngg"]["heads"]["SpCas9"][
            "unavailable_reasons"
        ]["mae"]
        == "no measured rows"
    )
    assert (
        StudyProtocol(study.path).validation_report(tmp_path / "rebuilt")["comparisons"]
        == report["comparisons"]
    )
    with pytest.raises(FileExistsError):
        study.validation_report(tmp_path / "report")
    with pytest.raises(PermissionError, match="freeze"):
        study.evaluate_test()
    # Stored-model evidence is independent of the six new models. No
    # historical seeds are fabricated, and the pinned checkpoint must match.
    stored = tmp_path / "stored.csv"
    stored.write_bytes((study.path / "runs" / run / "predictions.csv").read_bytes())
    sidecar = tmp_path / "stored-sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "rows": 3676,
                "file_sha256": file_sha256(stored),
                "checkpoint_sha256": study.execution_configuration()["acceptance"][
                    "historical_checkpoint_sha256"
                ],
                "split": split_record(controlled_inputs.split_dir),
            }
        )
    )
    with_stored = study.validation_report(
        tmp_path / "with-stored", stored_predictions=stored, stored_sidecar=sidecar
    )
    assert with_stored["stored_v2"]["populations"]["ngg"]["heads"]["SpCas9"]["mae"] == 5
    assert with_stored["identity"]["stored_v2"][
        "input_predictions_sha256"
    ] == file_sha256(stored)
    sidecar.write_text(sidecar.read_text().replace("5", "6"))
    damaged = study.validation_report(
        tmp_path / "damaged-stored", stored_predictions=stored, stored_sidecar=sidecar
    )
    assert damaged["stored_v2"] is None
    assert "stored_v2" in damaged["unavailable_evidence"]
    (study.path / "runs" / run / "predictions.csv").write_bytes(
        b"corrupted prediction bytes"
    )
    corrupted = study.validation_report(tmp_path / "corrupted")
    assert corrupted["runs"]["baseline"] == {}
    assert run in corrupted["unavailable_evidence"]


def test_bootstrap_quota_failure_does_not_reverse_complete_point_decision():
    report = build_report(
        runs(0.9),
        predictions(),
        bootstrap_settings={"valid_draws": 3, "maximum_attempts": 2},
    )
    assert report["decision"] == "pass"
    assert report["candidate_test_eligible"] is True
    assert report["bootstrap"]["status"] == "unavailable"
    assert report["bootstrap"]["attempts"] == 2
    assert report["bootstrap"]["valid_draws"] == 2
    assert report["bootstrap"]["intervals"] is None
    assert "quota" in report["bootstrap"]["reason"]


def test_candidate_test_rows_and_misaligned_arrays_are_refused():
    study = runs()
    study["candidate"][44]["split"] = "test"
    with pytest.raises(PermissionError, match="validation"):
        build_report(study, predictions(), bootstrap=False)
    study = runs()
    study["candidate"][44] = study["candidate"][44].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="align"):
        build_report(study, predictions(), bootstrap=False)


def test_bootstrap_shares_multiplicity_across_all_heads_methods_and_unequal_seeds():
    study = {"baseline": {}, "candidate": {}}
    for recipe, offsets in (("baseline", (1, 2, 3)), ("candidate", (0.5, 1.7, 3.2))):
        for seed, error in zip((42, 43, 44), offsets):
            original = predictions(error, ordered=recipe == "candidate")
            frame = two_protospacer_clusters(original)
            for i, head in enumerate(HEAD_NAMES):
                # Different errors per seed/head/cluster. The shared draw's
                # ratio of seed-then-head means is always -10%, whereas the
                # mean seed ratio is about -19.44%.
                frame[f"{head}_pred"] += error * i
                frame.loc[50:, f"{head}_pred"] += error * (i + 1)
            study[recipe][seed] = frame
    report = build_report(
        study, None, bootstrap_settings={"valid_draws": 30, "maximum_attempts": 300}
    )
    assert report["bootstrap"]["intervals"]["relative_ngg_mae_change"] == pytest.approx(
        [-0.1, -0.1]
    )


def test_undefined_joint_draws_consume_attempts_and_report_quota_shortfall():
    study = runs(0.9)
    for seeds in study.values():
        for seed, original in seeds.items():
            frame = two_protospacer_clusters(original)
            for head in HEAD_NAMES[1:]:
                frame.loc[50:, f"{head}_true"] = np.nan
            seeds[seed] = frame
    stored = study["baseline"][42].copy()
    report = build_report(
        study, stored, bootstrap_settings={"valid_draws": 10, "maximum_attempts": 10}
    )
    assert report["decision"] == "pass"
    assert report["bootstrap"]["attempts"] == 10
    assert 0 < report["bootstrap"]["valid_draws"] < 10
    assert report["bootstrap"]["invalid_draws"] > 0
    assert report["bootstrap"]["status"] == "unavailable"
    assert report["candidate_test_eligible"]


def test_training_diagnostics_cannot_change_the_selected_validation_checkpoint(
    controlled_inputs, synthetic_ledger, tmp_path
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    run = study.plan_run(seed=42, recipe="baseline")
    with pytest.raises(ValueError, match="checkpoint"):
        study.train_validation(run, backend=ChangedDiagnosticCheckpointBackend())
    assert not study.report_only(run)["eligible"]
