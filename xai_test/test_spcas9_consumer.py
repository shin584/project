"""Read-only e_system checks at the model-argument seam agreed in #94/#97."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from SpCas9.spcas9_consumer_check import characterize_consumer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "e_system"
EVIDENCE = Path(__file__).parent / "docs/evidence/issue97-consumer"


def test_actual_consumer_maps_every_head_and_receives_uncast_encoder_array(tmp_path):
    report = characterize_consumer(SOURCE, tmp_path / "observations")

    assert report["status"] == "verified"
    case = report["cases"]["distinct_heads"]
    assert case["exception"] is None
    assert [row["cas_name"] for row in case["variants"]] == [
        "SpCas9(WT)",
        "SpCas9-NG",
        "VRQR",
        "xCas9",
        "Sniper",
        "SpCas9-HF1",
        "eSpCas9(1.1)",
        "HypaCas9",
        "evoCas9",
    ]
    assert [row["raw_score"] for row in case["variants"]] == pytest.approx(
        [0, -0.25, 1.5, 0.3, 0.4, 0.56, 0.67, 0.8, 0.9]
    )
    assert [row["final_score"] for row in case["variants"]] == pytest.approx(
        [
            0,
            -15.163266492815836,
            90.979598956895,
            18.195919791379,
            24.26122638850534,
            33.96571694390748,
            40.63755420151968,
            48.52245277701067,
            54.58775937413701,
        ]
    )
    call = case["model_calls"][0]
    assert call["shape"] == [1, 30, 4]
    assert call["dtype_kind"] == "i"
    assert call["encoded_array"][0][:4] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert (
        call["window_sha256"]
        == hashlib.sha256(b"ACGTACGTACGTACGTACGTACGTAGGTCA").hexdigest()
    )
    assert report["real_checkpoint_inference_executed"] is False
    assert report["keras_dtype_acceptance_verified"] is False
    saved = json.loads((tmp_path / "observations/report.json").read_text())
    assert saved == report


def test_negative_cases_record_known_defect_or_correct_current_selection(tmp_path):
    report = characterize_consumer(SOURCE, tmp_path / "negative-cases")

    assert report["selection_status"] in {"known_defect_present", "fixed"}
    cases = report["cases"]
    assert cases["first_below_minus_one"]["expected_final"] == [[-2.0] * 9]
    assert cases["exact_minus_one"]["expected_final"] == [[-1.0] * 9]
    assert cases["consecutive_negative"]["expected_final"] == [
        [20.0] * 9,
        [-2.0] * 9,
        [-4.0] * 9,
    ]
    # A negative head alone is insufficient: a positive maximum still wins.
    assert cases["distinct_heads"]["exception"] is None
    assert cases["distinct_heads"]["inputs"][0]["raw_score"] == 1.5
    assert cases["zero_and_tie"]["inputs"][0]["raw_score"] == 0
    # Historical reproduction is pinned to preserved source, never required
    # of a future consumer repaired under #95.
    if report["selection_status"] == "known_defect_present":
        assert (
            cases["first_below_minus_one"]["exception"]["type"] == "UnboundLocalError"
        )
        assert cases["exact_minus_one"]["exception"]["type"] == "UnboundLocalError"
        assert [
            site["raw_score"] for site in cases["consecutive_negative"]["inputs"]
        ] == [0.2] * 3
        assert [
            site["final_score"] for site in cases["consecutive_negative"]["inputs"]
        ] == [20, -1, -1]
    else:
        assert all(
            cases[name]["exception"] is None
            for name in (
                "first_below_minus_one",
                "exact_minus_one",
                "consecutive_negative",
            )
        )
        assert [
            site["raw_score"] for site in cases["consecutive_negative"]["inputs"]
        ] == [0.2, -0.02, -0.04]
        assert [
            site["final_score"] for site in cases["consecutive_negative"]["inputs"]
        ] == [20, -2, -4]


def test_discovery_and_nonfinite_limits_are_recorded_without_consumer_edits(tmp_path):
    report = characterize_consumer(SOURCE, tmp_path / "limits")

    assert [row["three_base_pam"] for row in report["ng_discovery"]] == [
        "AGG",
        "AGA",
        "AGC",
        "AGT",
    ]
    assert all(row["matched_pam"] == "AG" for row in report["ng_discovery"])
    assert report["source_preserved"] is True
    for name, digest in report["source_sha256"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
        assert (
            hashlib.sha256((tmp_path / "limits/source" / name).read_bytes()).hexdigest()
            == digest
        )
    for name in ("nan_head", "positive_infinity_head", "negative_infinity_head"):
        case = report["cases"][name]
        assert case["exception"] is None or case["exception"]["type"] == "ValueError"
        if case["exception"] is None:
            assert case["variants"][1]["raw_score"] in {"NaN", "Infinity", "-Infinity"}


def test_existing_evidence_is_never_overwritten(tmp_path):
    destination = tmp_path / "prior-result"
    destination.mkdir()
    marker = destination / "report.json"
    marker.write_text("preserved")

    with pytest.raises(FileExistsError):
        characterize_consumer(SOURCE, destination)

    assert marker.read_text() == "preserved"


def test_evidence_destination_cannot_write_inside_consumer_source(tmp_path):
    source = tmp_path / "consumer"
    source.mkdir()
    destination = source / "new-evidence"

    with pytest.raises(ValueError, match="outside"):
        characterize_consumer(source, destination)

    assert not destination.exists()


def test_preserved_source_reproduces_history_without_pin_on_current_consumer(tmp_path):
    preserved = json.loads((EVIDENCE / "report.json").read_text())
    prior = EVIDENCE / "prior-review-evidence.json"

    report = characterize_consumer(
        EVIDENCE / "source",
        tmp_path / "historical",
        prior_evidence=prior,
    )

    assert report["source_sha256"] == preserved["source_sha256"]
    assert report["selection_status"] == "known_defect_present"
    assert report["prior_evidence"]["sha256"] == preserved["prior_evidence"]["sha256"]
    assert (
        tmp_path / "historical/prior-review-evidence.json"
    ).read_bytes() == prior.read_bytes()


def test_corrected_disposable_consumer_is_accepted_without_reintroducing_defect(
    tmp_path,
):
    # Illustrative finite-selection repair in a disposable historical copy.
    # This neither edits e_system nor implements/validates the #95 follow-up.
    source = tmp_path / "corrected-consumer"
    shutil.copytree(EVIDENCE / "source", source)
    prediction = source / "predict.py"
    original = prediction.read_bytes()
    assert original.count(b"best_score = -1.0") == 1
    prediction.write_bytes(
        original.replace(b"best_score = -1.0", b"best_score = -float('inf')")
    )

    report = characterize_consumer(source, tmp_path / "corrected-observations")

    assert report["selection_status"] == "fixed"
    cases = report["cases"]
    assert cases["exact_minus_one"]["inputs"][0]["final_score"] == -1
    assert [site["raw_score"] for site in cases["consecutive_negative"]["inputs"]] == [
        0.2,
        -0.02,
        -0.04,
    ]
    assert [
        site["final_score"] for site in cases["consecutive_negative"]["inputs"]
    ] == [20, -2, -4]


@pytest.mark.parametrize(
    "condition",
    [
        "any(float(head[0][0]) > 1 for head in preds)",
        "max(float(head[0][0]) for head in preds) == 0",
    ],
)
def test_rejected_finite_cases_cannot_publish_verified_evidence(tmp_path, condition):
    source = tmp_path / "rejecting-consumer"
    shutil.copytree(EVIDENCE / "source", source)
    prediction = source / "predict.py"
    original = prediction.read_bytes()
    anchor = b"preds = model.predict(X_np, verbose=0)"
    assert original.count(anchor) == 1
    replacement = (
        anchor
        + (
            f'\n            if {condition}:\n                raise ValueError("controlled rejection")'
        ).encode()
    )
    prediction.write_bytes(original.replace(anchor, replacement))
    destination = tmp_path / "failed-observations"

    with pytest.raises(AssertionError, match="Finite case"):
        characterize_consumer(source, destination)

    assert not (destination / "report.json").exists()
