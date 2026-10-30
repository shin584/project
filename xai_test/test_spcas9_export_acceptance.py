"""Real acceptance at #94's agreed export and consumer model-argument seams."""

import hashlib
import json
from pathlib import Path

import pytest

from SpCas9.spcas9_export_acceptance import verify_acceptance

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "xai_test/SpCas9/test_run/9hydra_v2.keras"


def test_selected_checkpoint_export_reaches_actual_consumer_without_cast(tmp_path):
    destination = tmp_path / "acceptance"
    report = verify_acceptance(SOURCE, ROOT / "e_system", destination)

    assert report["status"] == "verified"
    assert report["source_final"]["sha256"] == (
        "24e31cc6b9df8520e2bf761d33f7bb475c4658a9a479242b67db882946d0d2b8"
    )
    assert report["preservation"]["status"] == "unchanged"
    assert report["consumer"]["model_input_dtype"] == "float32"
    assert report["consumer"]["keras_integer_input_accepted"] is True
    cases = report["consumer"]["cases"]
    assert len(cases) == 3
    assert [case["penalty"] for case in cases] == pytest.approx(
        [1.0, 0.6065306597126334, 0.1353352832366127]
    )
    assert all(case["input"]["dtype_kind"] == "i" for case in cases)
    assert all(case["input"]["shape"] == [1, 30, 4] for case in cases)
    assert all(len(case["heads"]) == 9 for case in cases)
    first = cases[0]["heads"]
    assert first[0]["raw"] == pytest.approx(0.5581128597259521, abs=1e-6)
    assert first[5]["protein"] == "SpCas9-HF1"
    assert first[5]["raw"] == pytest.approx(0.14267730712890625, abs=1e-6)
    assert first[6]["protein"] == "eSpCas9(1.1)"
    assert first[6]["raw"] == pytest.approx(0.3090905249118805, abs=1e-6)
    artifact = destination / "export/model.keras"
    assert (
        hashlib.sha256(artifact.read_bytes()).hexdigest()
        == report["export"]["artifact"]["sha256"]
    )
    assert json.loads((destination / "report.json").read_text()) == report
