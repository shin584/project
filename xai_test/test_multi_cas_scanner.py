"""Scanner request to reported availability, at the agreed scanner boundary."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCANNER_PATH = Path(__file__).resolve().parents[1] / "multi_cas_scanner_수정.py"
SPEC = importlib.util.spec_from_file_location("multi_cas_scanner", SCANNER_PATH)
scanner_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner_module
SPEC.loader.exec_module(scanner_module)


class InferenceModel:
    """Stand in for the external Keras inference boundary, without training."""

    def __init__(self, value):
        self.value = value

    def predict(self, inputs, verbose=0):
        if isinstance(self.value, Exception):
            raise self.value
        return np.array([[self.value]])


def bundle_for(monkeypatch, *, sp=0.0, sa=-0.25, cas12=0.75):
    models = {
        "sp.keras": InferenceModel(sp),
        "cas12.keras": InferenceModel(cas12),
        "sa.keras": InferenceModel(sa),
    }

    def load_model(path):
        model = models[Path(path).name]
        if model.value == "missing":
            raise OSError("fixture checkpoint missing")
        return model

    monkeypatch.setattr(scanner_module.tf.keras.models, "load_model", load_model)
    return scanner_module.ModelBundle(
        "sp.keras", "cas12.keras", ".", sacas9_files=["sa.keras"]
    )


def scan_all_families():
    sequence = list("A" * 81)
    sequence[25:29] = "TTTA"
    sequence[35:38] = "AGG"
    sequence[45:51] = "AAGAAT"
    scanner = scanner_module.MultiCasScanner()
    result = scanner.scan("".join(sequence))
    assert {site.cas_type for site in result.sites} == {"SpCas9", "SaCas9", "Cas12a"}
    return scanner, result


def test_scanned_zero_and_negative_efficiencies_remain_ranked_and_visible(
    monkeypatch, capsys
):
    bundle = bundle_for(monkeypatch)
    scanner, result = scan_all_families()
    scanner.predict(result, bundle)
    rows = scanner.build_score_dict(result)

    assert len(result.sorted_by_score()) == len(result.sites)
    assert all(row["available"] for row in rows.values())
    assert all(row["reason"] is None for row in rows.values())
    assert {s.raw_score for s in result.by_cas("SpCas9")} == {0.0}
    assert {s.raw_score for s in result.by_cas("SaCas9")} == {-0.25}
    # Cas12a cuts at 49, nine bases from the center; keep the existing scale.
    cas12 = next(row for row in rows.values() if row["cas_type"] == "Cas12a")
    assert cas12["score_pct"] == 50.02
    scanner.print_report(result)
    report = capsys.readouterr().out
    assert "0.000" in report
    assert "0.00%" in report
    assert "-0.250" in report


@pytest.mark.parametrize(
    "family,argument", [("SpCas9", "sp"), ("SaCas9", "sa"), ("Cas12a", "cas12")]
)
@pytest.mark.parametrize(
    "failure,reason",
    [
        ("missing", "fixture checkpoint missing"),
        (RuntimeError("fixture inference failed"), "fixture inference failed"),
        (np.nan, "Non-finite"),
        (np.inf, "Non-finite"),
    ],
)
def test_family_failure_reaches_report_without_scores_or_ranks(
    monkeypatch, capsys, family, argument, failure, reason
):
    bundle = bundle_for(monkeypatch, **{argument: failure})
    scanner, result = scan_all_families()
    scanner.predict(result, bundle)
    rows = scanner.build_score_dict(result)

    assert len(rows) == len(result.sites)
    failed = [row for row in rows.values() if row["cas_type"] == family]
    assert failed
    for row in failed:
        assert row["available"] is False
        assert reason in row["reason"]
        assert row["raw_score"] is None
        assert row["score_pct"] is None
        assert row["rank"] is None
    assert all(site.cas_type != family for site in result.sorted_by_score())
    assert all(row["available"] for row in rows.values() if row["cas_type"] != family)
    capsys.readouterr()  # Check the consumer report, independently of loader logs.
    scanner.print_report(result)
    report = capsys.readouterr().out
    assert family in report and reason in report
    assert "unavailable" in report


@pytest.mark.parametrize("guide", [None, 7, "A" * 19, "Z" + "A" * 19])
def test_invalid_candidate_input_is_unavailable_through_the_consumer(
    monkeypatch, capsys, guide
):
    bundle = bundle_for(monkeypatch)
    scanner, result = scan_all_families()
    site = result.by_cas("SpCas9")[0]
    site.guide_seq = guide
    scanner.predict(result, bundle)

    assert site.prediction.available is False
    assert "invalid guide" in site.prediction.reason
    assert site not in result.sorted_by_score()
    row = next(
        row
        for row in scanner.build_score_dict(result).values()
        if row["cas_type"] == "SpCas9"
    )
    assert row["score_pct"] is None
    assert row["rank"] is None
    capsys.readouterr()
    scanner.print_report(result)
    assert site.prediction.reason in capsys.readouterr().out


def test_unknown_family_is_reported_without_a_numeric_fallback(monkeypatch, capsys):
    bundle = bundle_for(monkeypatch)
    scanner, result = scan_all_families()
    site = result.sites[0]
    site.cas_type = "UnknownCas"
    scanner.predict(result, bundle)
    row = next(
        row
        for row in scanner.build_score_dict(result).values()
        if row["cas_type"] == "UnknownCas"
    )
    assert row["available"] is False
    assert "Unknown Cas family" in row["reason"]
    assert row["score_pct"] is None
    assert site not in result.sorted_by_score()
    capsys.readouterr()
    scanner.print_report(result)
    assert row["reason"] in capsys.readouterr().out


def test_unpredicted_sites_have_reasons_and_no_rank(capsys):
    scanner, result = scan_all_families()
    assert result.sorted_by_score() == []
    rows = scanner.build_score_dict(result)
    assert len(rows) == len(result.sites)
    assert all(
        row["rank"] is None and row["score_pct"] is None for row in rows.values()
    )
    scanner.print_report(result)
    assert "Prediction not requested" in capsys.readouterr().out


def test_all_unavailable_skips_distance_penalties(monkeypatch, capsys):
    bundle = bundle_for(monkeypatch, sp="missing", sa="missing", cas12="missing")
    scanner, result = scan_all_families()
    scanner.SIGMA = 0  # A penalty calculation would divide by zero.
    scanner.predict(result, bundle)
    assert result.sorted_by_score() == []
    assert all(
        row["score_pct"] is None for row in scanner.build_score_dict(result).values()
    )
    capsys.readouterr()
    scanner.print_report(result)
    report = capsys.readouterr().out
    assert report.count("fixture checkpoint missing") == len(result.sites)


def test_reprediction_failure_clears_previous_scores(monkeypatch):
    scanner, result = scan_all_families()
    scanner.predict(result, bundle_for(monkeypatch))
    assert len(result.sorted_by_score()) == len(result.sites)
    scanner.predict(
        result, bundle_for(monkeypatch, sp="missing", sa="missing", cas12="missing")
    )
    assert result.sorted_by_score() == []
    assert all(
        site.raw_score is None and site.final_score is None for site in result.sites
    )


def test_missing_configured_ensemble_member_is_reported(monkeypatch):
    bundle_for(monkeypatch)  # Install the external checkpoint loader fixture.
    bundle = scanner_module.ModelBundle(
        "sp.keras", "cas12.keras", ".", sacas9_files=["sa.keras", "absent.keras"]
    )
    scanner, result = scan_all_families()
    scanner.predict(result, bundle)
    rows = scanner.build_score_dict(result)
    failed = next(row for row in rows.values() if row["cas_type"] == "SaCas9")
    assert failed["available"] is False
    assert "absent.keras" in failed["reason"]
    assert failed["rank"] is None
    assert all(row["available"] for row in rows.values() if row["cas_type"] != "SaCas9")


def test_saved_keras_models_deliver_zero_and_negative_results(tmp_path, capsys):
    """Exercise actual model loading and inference with disposable models."""
    tf = scanner_module.tf
    for config, value in zip(
        scanner_module.CAS_CONFIGS, [0.0, -0.25, 0.75], strict=True
    ):
        inputs = tf.keras.Input(shape=(config.guide_len, 4))
        pooled = tf.keras.layers.GlobalAveragePooling1D()(inputs)
        outputs = tf.keras.layers.Dense(
            1,
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(value),
        )(pooled)
        tf.keras.Model(inputs, outputs).save(tmp_path / f"{config.name}.keras")
    bundle = scanner_module.ModelBundle(
        str(tmp_path / "SpCas9.keras"),
        str(tmp_path / "Cas12a.keras"),
        str(tmp_path),
        sacas9_files=["SaCas9.keras"],
    )
    scanner, result = scan_all_families()
    scanner.predict(result, bundle)
    rows = scanner.build_score_dict(result)
    assert len(rows) == len(result.sites)
    assert all(row["available"] for row in rows.values())
    assert {row["raw_score"] for row in rows.values()} == {0.0, -0.25, 0.75}
    scanner.print_report(result)
    assert "-0.250" in capsys.readouterr().out
