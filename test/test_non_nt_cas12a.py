"""#197: public Cas12a scan/prediction boundaries using the adopted checkpoint."""

import json
import hashlib
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from non_nt_cas12a_predict import MANIFEST_CAS_CONFIGS, NonNTCas12aPredictor
from non_nt_runtime import cas12a_derive, one_hot
from pam_scanner import reverse_complement, scan_all, scan_pam
from predict import NON_NT_REGISTRY, run_prediction

ROOT = Path(__file__).resolve().parents[1]
DNA = json.loads((ROOT / "docs/equivalence-baseline/capture.json").read_text())['input_seq']
SPEC = NON_NT_REGISTRY[("A", "Cas12a")]


@pytest.mark.parametrize("reverse", [False, True])
def test_manifest_window_contains_the_exact_scanned_guide_on_both_strands(reverse):
    dna = reverse_complement(DNA) if reverse else DNA
    sites = scan_all(dna, configs=MANIFEST_CAS_CONFIGS).by_cas("Cas12a")
    assert len(sites) == 2
    expected = {
        "TCGCTTTAAGAATAAATAAAAGCGCCACAGGATT",
        "GTGGTTTAGTCAATCCTGTGGCGCTTTTATTTAT",
    }
    assert {site.model_input_seq for site in sites} == expected
    for site in sites:
        assert site.model_input_seq[4:8] == site.pam_seq
        assert site.model_input_seq[8:31] == site.guide_seq
    # The legacy NT scan deliberately keeps its original input window.
    assert {s.model_input_seq for s in scan_all(dna).by_cas("Cas12a")} == {
        "TTTAAGAATAAATAAAAGCGCCACAGGATTGACT",
        "TTTAGTCAATCCTGTGGCGCTTTTATTTATTCTT",
    }


@pytest.mark.parametrize("pam_start,accepted", [(3, False), (4, True), (51, True), (52, False)])
def test_manifest_requires_all_four_upstream_and_three_downstream_bases(pam_start, accepted):
    dna = "A" * pam_start + "TTTA" + "A" * (77 - pam_start)
    cfg = next(c for c in MANIFEST_CAS_CONFIGS if c.name == "Cas12a")
    sites = scan_pam(dna, cfg, max_dist=81)
    assert any(s.strand == "+" and s.pam_start == pam_start for s in sites) == accepted


def test_feature_order_and_values_match_the_preserved_training_source():
    archive = ROOT / "models/non_nt/cas12a-cnn-20261102/source_snapshot.zip"
    with zipfile.ZipFile(archive) as source:
        namespace = {}
        exec(compile(source.read("cas12a/features.py"), "training/features.py", "exec"), namespace)
    windows = [s.model_input_seq for s in scan_all(DNA, configs=MANIFEST_CAS_CONFIGS).by_cas("Cas12a")]
    # All 16 pair types and ambiguous bases exercise positional blocks and zeros.
    windows += ["AAAA" + "TTTA" + "AACAGATCCGCTGGTTAACAGAT" + "AAA", "N" * 34]
    for window in windows:
        gc, dinucleotide = namespace["extract_seq_features"](window)
        np.testing.assert_array_equal(cas12a_derive(window), np.concatenate([gc, dinucleotide]))
        np.testing.assert_array_equal(one_hot(window), namespace["one_hot_encode"](window))
        assert cas12a_derive(window).shape == (354,)
        assert cas12a_derive(window).dtype == np.float32


@pytest.fixture(scope="module")
def predictor():
    return NonNTCas12aPredictor(SPEC)


def test_real_predictions_have_one_head_fraction_units_and_checkpoint_identity(predictor):
    sites = scan_all(DNA, configs=MANIFEST_CAS_CONFIGS).by_cas("Cas12a")
    progress = []
    rows = predictor.predict_batch(sites, on_batch=lambda d, t: progress.append((d, t)))
    assert len(rows) == 2
    assert progress == [(2, 2)]
    for site, row in zip(sites, rows):
        assert row.site_key == site.site_key
        assert row.window_seq == site.model_input_seq
        assert row.cas_name == "Cas12a"
        assert row.efficiency_pct == pytest.approx(row.raw_score * 100)
        assert row.score_kind == "efficiency"
        assert row.identity.family == "non_nt"
        assert row.identity.checkpoint_sha256 == "94ac977865efa1cdad734ae05e033cbaf40b38e109013a6d2fe2b9361aa8b28f"
        assert predictor.explain_heads(site) == ["Cas12a"]
    assert all(0 < row.raw_score < 1 for row in rows)


def test_prediction_rejects_a_legacy_nt_window_instead_of_silently_shifting_features(predictor):
    sites = scan_all(DNA).by_cas("Cas12a")
    with pytest.raises(ValueError, match="4 flank"):
        predictor.predict_batch(sites)


def test_actual_model_inputs_and_outputs_match_independent_tensorflow_reference(predictor):
    reference = json.loads((ROOT / "docs/benchmarks/issue197-cas12a-reference.json").read_text())
    [template, _] = scan_all(DNA, configs=MANIFEST_CAS_CONFIGS).by_cas("Cas12a")
    sites = []
    for case in reference["cases"]:
        window = case["window"]
        assert hashlib.sha256(one_hot(window).tobytes()).hexdigest() == case["sequence_input_sha256"]
        assert hashlib.sha256(cas12a_derive(window).tobytes()).hexdigest() == case["feature_input_sha256"]
        sites.append(replace(template, model_input_seq=window, pam_seq=window[4:8], guide_seq=window[8:31]))
    rows = predictor.predict_batch(sites)
    actual = np.array([row.raw_score for row in rows])
    expected = np.array([case["raw_score"] for case in reference["cases"]])
    delta = np.abs(actual - expected)
    assert delta.max() <= reference["tolerance"]["max_abs_diff"]
    assert (delta / np.maximum(np.abs(expected), 1e-12)).max() <= reference["tolerance"]["max_rel_diff"]


def test_batch_cancellation_propagates_through_the_predictor(predictor):
    [site, _] = scan_all(DNA, configs=MANIFEST_CAS_CONFIGS).by_cas("Cas12a")

    class Cancelled(Exception):
        pass

    def cancel(done, total):
        assert (done, total) == (64, 65)
        raise Cancelled()

    with pytest.raises(Cancelled):
        predictor.predict_batch([site] * 65, on_batch=cancel)


def test_prediction_replaces_previous_family_rows(predictor):
    scan = scan_all(DNA, configs=MANIFEST_CAS_CONFIGS)
    [site, _] = scan.by_cas("Cas12a")
    scan.variant_results = [replace(predictor.predict_batch([site])[0], raw_score=-999)]
    result = run_prediction(scan, {("A", "Cas12a"): predictor})
    assert len(result.variant_results) == 2
    assert all(row.raw_score > 0 and row.identity.family == "non_nt" for row in result.variant_results)
    for site in result.by_cas("Cas12a"):
        assert site.efficiency_pct > 0


def test_explanations_use_same_checkpoint_as_prediction(predictor):
    handle = predictor.explain_handle("Cas12a")
    assert handle.model_key.checkpoints == (predictor.model_identity().checkpoint_sha256,)
    assert handle.ig_path == "composite" and handle.head == "Cas12a"
