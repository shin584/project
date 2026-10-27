"""Scanner request to reported availability, at the agreed scanner boundary."""

import hashlib
import importlib.util
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

SCANNER_PATH = Path(__file__).resolve().parents[1] / "multi_cas_scanner_수정.py"
SPEC = importlib.util.spec_from_file_location("multi_cas_scanner", SCANNER_PATH)
scanner_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner_module
SPEC.loader.exec_module(scanner_module)

HEAD_LAYERS = {
    "SpCas9": "Output_SpCas9",
    "SpCas9-NG": "Output_SpCas9_NG",
    "VRQR variant": "Output_VRQR_variant",
    "xCas": "Output_xCas",
    "Sniper-Cas9": "Output_Sniper_Cas9",
    "SpCas9-HF.1": "Output_SpCas9_HF_1",
    "eSpCas9(1.1)": "Output_eSpCas9_1_1_",
    "HypaCas9": "Output_HypaCas9",
    "evoCas9": "Output_evoCas9",
}


class InferenceModel:
    """Stand in for the external Keras inference boundary, without training."""

    def __init__(self, value):
        self.value = value

    def predict(self, inputs, verbose=0):
        if isinstance(self.value, Exception):
            raise self.value
        if isinstance(self.value, dict):
            return self.value
        return np.array([[self.value]])


def bundle_for(monkeypatch, *, sp=0.0, sa=-0.25, cas12=0.75):
    if not isinstance(sp, (str, dict, Exception)):
        sp = {layer: np.array([[sp]]) for layer in HEAD_LAYERS.values()}
    models = {
        "sp.keras": InferenceModel(sp),
        "cas12.keras": InferenceModel(cas12),
        "sa.keras": InferenceModel(sa),
    }

    def load_model(path, **kwargs):
        model = models[Path(path).name]
        if isinstance(model.value, str) and model.value == "missing":
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


WINDOW = "TGCA" + "ACGT" * 5 + "AGG" + "TAC"


@pytest.mark.parametrize("strand", ["+", "-"])
def test_scanner_preserves_oriented_model_window_separately_from_guide(strand):
    region = "A" * 15 + WINDOW + "A" * 36
    if strand == "-":
        region = scanner_module.reverse_complement(region)
    scanner = scanner_module.MultiCasScanner(max_dist=81)
    result = scanner.scan(region)
    site = next(s for s in result.by_cas("SpCas9") if s.strand == strand)
    assert site.model_input_seq == WINDOW
    assert site.guide_seq == "ACGT" * 5
    assert site.pam_seq == "AGG"
    assert site.pam_start == 39  # Existing strand-relative PAM coordinate.
    assert site.cut_pos == (36 if strand == "+" else 44)
    assert site.model_input_reason is None


@pytest.mark.parametrize("reverse_order", [False, True])
def test_all_named_efficiencies_reach_consumer_independent_of_dict_order(
    monkeypatch, reverse_order
):
    expected = dict(
        zip(HEAD_LAYERS, [0.0, -0.25, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], strict=True)
    )
    outputs = {
        HEAD_LAYERS[name]: np.array([[value]]) for name, value in expected.items()
    }
    if reverse_order:
        outputs = dict(reversed(list(outputs.items())))
    bundle = bundle_for(monkeypatch, sp=outputs)
    scanner, result = scan_all_families()
    scanner.predict(result, bundle)
    row = next(
        r
        for r in scanner.build_score_dict(result).values()
        if r["cas_type"] == "SpCas9"
    )
    assert row["available"] is True
    assert row["raw_score"] == 0.0
    assert row["rank"] == 1
    assert row["efficiencies_by_protein"] == expected


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("offset", [0, 3, 56, 57, 58])
def test_incomplete_flanks_remain_discovered_but_unavailable(
    monkeypatch, strand, offset
):
    # 23bp guide+PAM at offset; a full window needs >=4 left and >=3 right bases.
    region = "A" * offset + "ACGT" * 5 + "AGG" + "A" * (58 - offset)
    if strand == "-":
        region = scanner_module.reverse_complement(region)
    scanner = scanner_module.MultiCasScanner(max_dist=81)
    result = scanner.scan(region)
    site = next(s for s in result.by_cas("SpCas9") if s.strand == strand)
    scanner.predict(result, bundle_for(monkeypatch))
    row = next(
        r
        for r in scanner.build_score_dict(result).values()
        if r["strand"] == strand and r["cas_type"] == "SpCas9"
    )
    assert site.model_input_seq is None
    assert row["available"] is False
    assert "insufficient flanking context" in row["reason"]
    assert row["efficiencies_by_protein"] == {}
    assert row["score_pct"] is None and row["rank"] is None
    assert site not in result.sorted_by_score()


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("offset", [4, 55])
def test_exactly_sufficient_flanks_remain_available(monkeypatch, strand, offset):
    region = "A" * offset + "ACGT" * 5 + "AGG" + "A" * (58 - offset)
    if strand == "-":
        region = scanner_module.reverse_complement(region)
    scanner = scanner_module.MultiCasScanner(max_dist=81)
    result = scanner.scan(region)
    scanner.predict(result, bundle_for(monkeypatch))
    row = next(
        r
        for r in scanner.build_score_dict(result).values()
        if r["cas_type"] == "SpCas9" and r["strand"] == strand
    )
    assert row["available"] is True
    assert row["model_input_seq"] == "AAAA" + "ACGT" * 5 + "AGGAAA"


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("position", [0, 3, 4, 23, 27, 29])
def test_invalid_context_base_is_unavailable_through_consumer(
    monkeypatch, strand, position
):
    window = WINDOW[:position] + "N" + WINDOW[position + 1 :]
    region = "A" * 15 + window + "A" * 36
    if strand == "-":
        region = scanner_module.reverse_complement(region)
    scanner = scanner_module.MultiCasScanner(max_dist=81)
    result = scanner.scan(region)
    scanner.predict(result, bundle_for(monkeypatch))
    row = next(
        r
        for r in scanner.build_score_dict(result).values()
        if r["strand"] == strand and r["cas_type"] == "SpCas9"
    )
    assert row["available"] is False
    assert "invalid base" in row["reason"]
    assert row["rank"] is None and row["raw_score"] is None


@pytest.mark.parametrize("layer", HEAD_LAYERS.values())
@pytest.mark.parametrize(
    "failure", ["missing", "shape", "nan", "inf", "text", "complex"]
)
def test_any_invalid_head_invalidates_the_complete_call(
    monkeypatch, capsys, layer, failure
):
    outputs = {name: np.array([[0.5]]) for name in HEAD_LAYERS.values()}
    if failure == "missing":
        del outputs[layer]
    else:
        outputs[layer] = {
            "shape": np.array([0.5]),
            "nan": np.array([[np.nan]]),
            "inf": np.array([[np.inf]]),
            "text": np.array([["0.5"]]),
            "complex": np.array([[0.5 + 1j]]),
        }[failure]
    scanner, result = scan_all_families()
    scanner.predict(result, bundle_for(monkeypatch, sp=outputs))
    rows = list(scanner.build_score_dict(result).values())
    failed = next(r for r in rows if r["cas_type"] == "SpCas9")
    assert failed["available"] is False
    assert layer in failed["reason"]
    assert failed["efficiencies_by_protein"] == {}
    assert failed["rank"] is None and failed["score_pct"] is None
    assert all(r["available"] for r in rows if r["cas_type"] != "SpCas9")
    capsys.readouterr()
    scanner.print_report(result)
    assert layer in capsys.readouterr().out


def test_legacy_positional_output_is_explicitly_unavailable(monkeypatch):
    bundle = bundle_for(monkeypatch)
    bundle.model_A.value = 0.5  # A single-output array is not a complete v2 prediction.
    scanner, result = scan_all_families()
    scanner.predict(result, bundle)
    row = next(
        r
        for r in scanner.build_score_dict(result).values()
        if r["cas_type"] == "SpCas9"
    )
    assert row["available"] is False
    assert "requires a dict keyed by output layer name" in row["reason"]


def test_negative_spcas9_linear_efficiency_is_ranked(monkeypatch, capsys):
    scanner, result = scan_all_families()
    scanner.predict(result, bundle_for(monkeypatch, sp=-0.25))
    row = next(
        r
        for r in scanner.build_score_dict(result).values()
        if r["cas_type"] == "SpCas9"
    )
    assert row["available"] is True and row["raw_score"] == -0.25
    assert row["rank"] == 1 and row["score_pct"] == -18.15
    scanner.print_report(result)
    assert "-0.250" in capsys.readouterr().out


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
        inputs = tf.keras.Input(
            shape=(30 if config.name == "SpCas9" else config.guide_len, 4)
        )
        pooled = tf.keras.layers.GlobalAveragePooling1D()(inputs)
        outputs = tf.keras.layers.Dense(
            1,
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(value),
        )(pooled)
        if config.name == "SpCas9":
            outputs = {
                layer: tf.keras.layers.Dense(
                    1,
                    name=layer,
                    kernel_initializer="zeros",
                    bias_initializer=tf.keras.initializers.Constant(value),
                )(pooled)
                for layer in HEAD_LAYERS.values()
            }
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


def test_stored_v2_checkpoint_matches_direct_model_on_oriented_windows(tmp_path):
    """Actual pinned v2, independent encoding, all heads, both strands; no fits."""
    checkpoint = Path(__file__).parent / "SpCas9/v2_train/9hydra_v2.keras"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert (
        checkpoint_hash
        == "ece5009215482d3fd2506688dc8e2423437abdcd28dfe10e4ac21c5bed3d19d6"
    )
    tf = scanner_module.tf
    direct = tf.keras.models.load_model(checkpoint, compile=False)
    # Reload a disposable copy too: inference must not need a training-only loss.
    reloaded_path = tmp_path / "v2_copy.keras"
    direct.save(reloaded_path)
    bundle = scanner_module.ModelBundle(
        str(reloaded_path),
        str(tmp_path / "absent_cas12.keras"),
        str(tmp_path),
        sacas9_files=["absent_sa.keras"],
    )
    cases = []
    for window in [WINDOW, "CATG" + "TGCA" * 5 + "TGG" + "ATC"]:
        encoded = np.eye(4, dtype=np.float32)[[["ACGT".index(base) for base in window]]]
        expected = direct(encoded, training=False)
        for strand in ["+", "-"]:
            region = "A" * 15 + window + "A" * 36
            if strand == "-":
                region = scanner_module.reverse_complement(region)
            scanner = scanner_module.MultiCasScanner(max_dist=81)
            result = scanner.scan(region)
            scanner.predict(result, bundle)
            row = next(
                r
                for r in scanner.build_score_dict(result).values()
                if r["cas_type"] == "SpCas9" and r["strand"] == strand
            )
            assert row["available"] is True, row["reason"]
            assert row["model_input_seq"] == window
            comparisons = {}
            for protein, layer in HEAD_LAYERS.items():
                value = float(np.asarray(expected[layer])[0, 0])
                adapted = row["efficiencies_by_protein"][protein]
                np.testing.assert_allclose(adapted, value, rtol=1e-5, atol=1e-6)
                comparisons[protein] = {
                    "direct": value,
                    "scanner": adapted,
                    "absolute_error": abs(value - adapted),
                }
            cases.append(
                {
                    "region": region,
                    "region_sha256": hashlib.sha256(region.encode()).hexdigest(),
                    "strand": strand,
                    "window": window,
                    "window_sha256": hashlib.sha256(window.encode()).hexdigest(),
                    "heads": comparisons,
                }
            )
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    report = {
        "issue": 81,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "scanner_sha256": hashlib.sha256(SCANNER_PATH.read_bytes()).hexdigest(),
        "reload_copy_sha256": hashlib.sha256(reloaded_path.read_bytes()).hexdigest(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "libraries": {
                name: version(name)
                for name in ["tensorflow", "keras", "numpy", "pytest"]
            },
            "devices": [str(d) for d in tf.config.list_physical_devices()],
        },
        "rtol": 1e-5,
        "atol": 1e-6,
        "passed": True,
        "cases": cases,
    }
    (tmp_path / "parity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
