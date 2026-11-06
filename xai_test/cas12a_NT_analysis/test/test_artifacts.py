import pytest

from xai_test.cas12a_NT_analysis.nt_artifacts import prediction_rows, read_results
from xai_test.cas12a_NT_analysis.nt_inputs import (
    encode_sequences,
    row_identity,
    verify_inputs,
)
from xai_test.cas12a_NT_analysis.nt_run import Study


def test_model_free_readback_reconstructs_errors_metrics_and_null_reasons(
    inputs, tmp_path
):
    from transformers import AutoTokenizer

    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    source_rows = [
        dict(verified.rows[0], target=0.0),
        dict(verified.rows[1], target=1.0),
    ]
    source_rows = [
        dict(row, sequence=row["sequence"][:4] + "TTTA" + row["sequence"][8:])
        for row in source_rows
    ]
    from xai_test.cas12a_NT_analysis.nt_inputs import sequence_hash

    for row in source_rows:
        row["sequence_sha256"] = sequence_hash(row["sequence"])
        row["raw"] = dict(row["raw"])
        from cas12a.data import SEQUENCE_COLUMN, TARGET_COLUMN

        row["raw"][SEQUENCE_COLUMN] = row["sequence"]
        row["raw"][TARGET_COLUMN] = row["target"] * 100
    tokenizer = AutoTokenizer.from_pretrained(inputs.export, local_files_only=True)
    tokens = encode_sequences(tokenizer, [row["sequence"] for row in source_rows], 4105)
    rows = prediction_rows(source_rows, [-1.0, 2.0], tokens)
    identity = {
        "validation_source_rows": [row["source_row"] for row in source_rows],
        "validation_rows": [row_identity(row) for row in source_rows],
    }
    study = Study.create(tmp_path / "drive", fixture=True)
    study.register("a", identity)
    study.start("a", "cpu", 100, {})
    study.save_artifact(
        "shard",
        {"schema": 1, "identity": identity, "rows": rows},
        identity["validation_source_rows"],
    )
    study.finish()
    report = read_results(study, "a", require_complete=True)
    assert report["metrics"]["all"]["raw"]["mae"] == 1.0
    assert report["metrics"]["all"]["raw"]["mse"] == 1.0
    assert report["metrics"]["all"]["clipped"]["mae"] == 0.0
    assert report["metrics"]["all"]["raw"]["pearson"] == pytest.approx(1.0)
    assert report["metrics"]["TTTT"]["raw"]["spearman"] is None
    assert (
        report["metrics"]["TTTT"]["raw"]["spearman_undefined_reason"]
        == "empty population"
    )
    assert report["gpu_executed"] is False

    # The exported run is sufficient even when both inputs and model disappear.
    import json
    import subprocess
    import sys
    from pathlib import Path

    from xai_test.cas12a_NT_analysis.nt_run import file_hash

    inputs.export.rename(tmp_path / "model-unavailable")
    inputs.source.unlink()
    inputs.manifest.unlink()
    receipt_path = tmp_path / "external-receipt.json"
    receipt_path.write_text(json.dumps(study.receipt), encoding="utf-8")
    output = tmp_path / "model-free-report"
    script = """import sys
class NoModelImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise ModuleNotFoundError('model libraries unavailable')
        return None
sys.meta_path.insert(0, NoModelImports())
from xai_test.cas12a_NT_analysis.analyze import main
main(sys.argv[1:])
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "inspect",
            "--study-root",
            str(study.store.root),
            "--receipt",
            str(receipt_path),
            "--run-id",
            "a",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    exported = json.loads((output / "manifest.json").read_bytes())
    assert exported["complete"] is True
    assert exported["files"]["predictions.csv"] == file_hash(output / "predictions.csv")
    assert json.loads((output / "metrics.json").read_bytes()) == report["metrics"]
