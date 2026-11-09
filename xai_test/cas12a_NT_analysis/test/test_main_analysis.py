import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xai_test.cas12a_NT_analysis.nt_inputs import verify_test_inputs
from xai_test.cas12a_NT_analysis.nt_main import prepare_main, run_main
from xai_test.cas12a_NT_analysis.nt_main_artifacts import read_main
from xai_test.cas12a_NT_analysis.nt_main_report import report_main


def test_test_population_runs_to_model_free_report_and_reuses_results(inputs, tmp_path):
    verified = verify_test_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    expected = sorted(inputs.partition["partitions"]["test"])
    assert [r["source_row"] for r in verified.rows] == expected
    assert "validation_source_rows" not in verified.identity
    root = tmp_path / "main"
    plan = prepare_main(verified, root)
    assert plan["statistics_ids"] == expected
    progress = []
    run_main(verified, root, device="cpu", on_progress=progress.append)
    saved = read_main(root)
    assert saved["complete"]
    assert saved["coverage"]["ism"]["available"] == expected
    assert saved["coverage"]["ig"]["requested"] == expected
    assert all(p["output_dir"] == str(root.resolve()) for p in progress)
    report = report_main(root)
    output = report["output_dir"]
    with open(
        os.path.join(output, "predictions.csv"), newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    assert [int(r["source_row"]) for r in rows] == expected
    assert report["metrics"]["all"]["raw"]["count"] == 2
    with np.load(os.path.join(output, "ism.npz"), allow_pickle=False) as arrays:
        assert arrays["source_row"].tolist() == expected
        assert arrays["raw_delta"].shape == (2, 34, 4)
        assert arrays["non_wt_mask"].sum() == 204
        assert np.all(arrays["raw_delta"][arrays["wt_mask"]] == 0)
    with np.load(os.path.join(output, "ig.npz"), allow_pickle=False) as arrays:
        assert arrays["signed_mass"].shape == (2, 12)
    assert os.path.isfile(os.path.join(output, "report.ko.md"))
    assert os.path.isfile(os.path.join(output, "position_profiles.svg"))
    before = {p: p.read_bytes() for p in (root / "shards").rglob("*") if p.is_file()}
    run_main(verified, root, device="cpu")
    assert all(p.read_bytes() == data for p, data in before.items())
    script = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'transformers'):
        raise AssertionError('model import during reporting: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from xai_test.cas12a_NT_analysis.nt_main_report import report_main
assert report_main(sys.argv[1])['complete']
"""
    subprocess.run([sys.executable, "-c", script, str(root)], check=True)


def test_interrupted_method_keeps_completed_shard_and_reports_missing_coverage(
    inputs, tmp_path
):
    verified = verify_test_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    root = tmp_path / "interrupted"
    prepare_main(verified, root)

    def disconnect(progress):
        if progress["method"] == "ism" and progress["completed"] == 1:
            raise RuntimeError("simulated Colab disconnection")

    with pytest.raises(RuntimeError, match="Colab disconnection"):
        run_main(verified, root, device="cpu", on_progress=disconnect)
    partial = report_main(root)
    assert not partial["complete"]
    assert len(partial["coverage"]["ism"]["available"]) == 1
    assert len(partial["coverage"]["ism"]["not_computed"]) == 1
    assert len(partial["coverage"]["ig"]["not_computed"]) == 2
    completed = next((root / "shards/ism").glob("*.json"))
    data = completed.read_bytes()
    # A write interrupted before the final JSON acknowledgement is not reused.
    (root / "shards/ism/orphan.npz").write_bytes(b"incomplete")
    (root / "shards/ism/ignored.json.pending-test").write_bytes(b"incomplete")
    run_main(verified, root, device="cpu")
    assert read_main(root)["complete"]
    assert completed.read_bytes() == data
    incompatible = dict(verified.identity)
    incompatible["table_sha256"] = "0" * 64
    from dataclasses import replace

    with pytest.raises(ValueError, match="incompatible"):
        run_main(replace(verified, identity=incompatible), root, device="cpu")


def test_zero_signal_and_known_raw_clipped_metrics_remain_valid(inputs, tmp_path):
    import torch
    from transformers import EsmForSequenceClassification

    from xai_test.cas12a_NT_analysis.nt_run import file_hash

    model = EsmForSequenceClassification.from_pretrained(
        inputs.export, local_files_only=True
    )
    with torch.no_grad():
        model.classifier.out_proj.weight.zero_()
        model.classifier.out_proj.bias.fill_(1.25)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(inputs.export, local_files_only=True)
    export = tmp_path / "constant_export"
    model.save_pretrained(export)
    tokenizer.save_pretrained(export)
    inputs.export = export
    receipt = json.loads(inputs.receipt.read_bytes())
    receipt["files"] = {
        f"selected_export/{p.name}": file_hash(p)
        for p in inputs.export.iterdir()
        if p.is_file()
    }
    inputs.receipt.write_text(json.dumps(receipt), encoding="utf-8")
    verified = verify_test_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    root = tmp_path / "zero"
    prepare_main(verified, root)
    run_main(verified, root, device="cpu")
    report = report_main(root)
    assert report["complete"]
    assert report["metrics"]["all"]["raw"]["spearman"] is None
    assert (
        report["metrics"]["all"]["raw"]["spearman_undefined_reason"]
        == "constant predictions"
    )
    assert report["metrics"]["all"]["raw"]["mae"] == pytest.approx(0.8)
    assert report["metrics"]["all"]["raw"]["mse"] == pytest.approx(0.7625)
    assert report["metrics"]["all"]["clipped"]["mae"] == pytest.approx(0.55)
    assert report["metrics"]["all"]["clipped"]["mse"] == pytest.approx(0.425)
    with open(
        os.path.join(report["output_dir"], "predictions.csv"),
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))
    assert {r["raw_prediction"] for r in rows} == {"1.25"}
    assert {r["clipped_prediction"] for r in rows} == {"1.0"}
    agreement = json.loads((Path(report["output_dir"]) / "agreement.json").read_bytes())
    assert agreement["populations"]["all"]["primary"]["n_valid"] == 0
    assert agreement["populations"]["all"]["n_available"] == 2
    with np.load(os.path.join(report["output_dir"], "ig.npz")) as arrays:
        assert arrays["accepted"].tolist() == [True, True]
        assert np.all(arrays["signed_mass"] == 0)


def test_unresolved_ig_numerics_are_failed_and_never_exported_as_zero(inputs, tmp_path):
    import torch
    from transformers import AutoTokenizer, EsmForSequenceClassification

    from xai_test.cas12a_NT_analysis.nt_run import file_hash

    model = EsmForSequenceClassification.from_pretrained(
        inputs.export, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(inputs.export, local_files_only=True)
    # A very sharp zero-reference path makes finite midpoint quadrature miss
    # the endpoint difference; the existing quality policy must reject it.
    with torch.no_grad():
        model.esm.embeddings.word_embeddings.weight.mul_(100000)
    export = tmp_path / "sharp_export"
    model.save_pretrained(export)
    tokenizer.save_pretrained(export)
    receipt = json.loads(inputs.receipt.read_bytes())
    receipt["files"] = {
        f"selected_export/{p.name}": file_hash(p)
        for p in export.iterdir()
        if p.is_file()
    }
    inputs.receipt.write_text(json.dumps(receipt), encoding="utf-8")
    verified = verify_test_inputs(
        inputs.source, inputs.manifest, export, inputs.receipt, fixture=True
    )
    root = tmp_path / "quality_failure"
    prepare_main(verified, root)
    run_main(verified, root, device="cpu", stages=("ig",))
    report = report_main(root)
    assert not report["complete"]
    assert len(report["coverage"]["ig"]["failed"]) == 2
    with np.load(Path(report["output_dir"]) / "ig.npz") as arrays:
        assert arrays["accepted"].tolist() == [False, False]
        assert np.isnan(arrays["signed_mass"]).all()
    agreement = json.loads((Path(report["output_dir"]) / "agreement.json").read_bytes())
    assert agreement["populations"]["all"]["n_available"] == 0
    assert all(
        r["agreement"] is None and r["exclusion_reasons"] for r in agreement["rows"]
    )


def test_main_keeps_unresolved_historical_accounting_blocked(inputs, tmp_path):
    from xai_test.cas12a_NT_analysis.nt_run import Study

    verified = verify_test_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    root = tmp_path / "blocked_main"
    prepare_main(verified, root)
    study = Study.create(tmp_path / "historical_study", fixture=True)
    study.register("old", {"historical": True})
    study.start("old", "old-session", 300, {"device": "cpu"})
    before = study.status()
    receipt = study.receipt
    with pytest.raises(ValueError, match="unresolved"):
        run_main(verified, root, device="cpu", study=study, reserved_seconds=300)
    assert study.status() == before
    assert study.receipt == receipt
    assert not (root / "shards").exists()
    assert not report_main(root)["complete"]


def test_resume_rejects_changed_numerical_environment(inputs, tmp_path, monkeypatch):
    import platform

    verified = verify_test_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    root = tmp_path / "environment"
    prepare_main(verified, root)
    run_main(verified, root, device="cpu", stages=())
    monkeypatch.setattr(platform, "python_version", lambda: "changed-python-version")
    with pytest.raises(ValueError, match="execution environment"):
        run_main(verified, root, device="cpu", stages=("ism",))


def test_ism_numerical_failure_is_saved_as_failed_coverage(inputs, tmp_path):
    import torch
    from transformers import AutoTokenizer, EsmForSequenceClassification

    from xai_test.cas12a_NT_analysis.nt_run import file_hash

    tokenizer = AutoTokenizer.from_pretrained(inputs.export, local_files_only=True)
    model = EsmForSequenceClassification.from_pretrained(
        inputs.export, local_files_only=True
    )
    with torch.no_grad():
        # CC GTTT is visited by a first-base mutant, never by these WT targets.
        mutant_id = tokenizer.convert_tokens_to_ids("CCGTTT")
        model.esm.embeddings.word_embeddings.weight[mutant_id].fill_(float("nan"))
    export = tmp_path / "nonfinite_mutant_export"
    model.save_pretrained(export)
    tokenizer.save_pretrained(export)
    receipt = json.loads(inputs.receipt.read_bytes())
    receipt["files"] = {
        f"selected_export/{p.name}": file_hash(p)
        for p in export.iterdir()
        if p.is_file()
    }
    inputs.receipt.write_text(json.dumps(receipt), encoding="utf-8")
    verified = verify_test_inputs(
        inputs.source, inputs.manifest, export, inputs.receipt, fixture=True
    )
    root = tmp_path / "ism_failure"
    prepare_main(verified, root)
    with pytest.raises(ValueError, match="nonfinite"):
        run_main(verified, root, device="cpu", stages=("ism",))
    report = report_main(root)
    assert report["coverage"]["ism"]["failed"] == [1]
    assert report["coverage"]["ism"]["not_computed"] == [8]
    annotated = json.loads(
        (Path(report["output_dir"]) / "cases_annotated.json").read_bytes()
    )
    failed = next(c for c in annotated["cases"] if c["source_row"] == 1)
    assert failed["methods"]["ism"]["status"] == "failed"
    assert "nonfinite" in failed["methods"]["ism"]["failure"]["reason"]


def test_frozen_subset_and_case_only_targets_keep_separate_denominators(
    inputs, tmp_path
):
    import hashlib

    import pandas as pd

    from cas12a.data import SEQUENCE_COLUMN, TARGET_COLUMN
    from cas12a.nt_data import (
        SplitConfig,
        canonical_bytes,
        partition_library,
        read_library,
    )
    from xai_test.cas12a_NT_analysis.nt_selection import freeze_coverage, read_coverage
    from xai_test.cas12a_NT_analysis.test.test_selection import sequence

    # This larger synthetic population checks the existing 400-row fallback
    # contract using predictions only; it does not compute 400 IGs on CPU.
    source, manifest = tmp_path / "subset.csv", tmp_path / "subset-partition.json"
    config = SplitConfig.fixture(4, 4, 402)
    frame = pd.DataFrame(
        {SEQUENCE_COLUMN: [sequence(i) for i in range(410)], TARGET_COLUMN: [0] * 410}
    )
    frame.to_csv(source, index=False)
    initial = partition_library(read_library(source, config), config)
    population = sorted(initial["partitions"]["test"])
    prefix = np.random.Generator(np.random.PCG64(42)).permutation(402)[:400]
    statistics = sorted(population[int(i)] for i in prefix)
    case_only = sorted(set(population) - set(statistics))
    frame.loc[case_only, TARGET_COLUMN] = 100
    frame.to_csv(source, index=False)
    partition = partition_library(read_library(source, config), config)
    manifest.write_bytes(canonical_bytes(partition))
    verified = verify_test_inputs(
        source, manifest, inputs.export, inputs.receipt, fixture=True
    )
    decision = {
        "schema": 1,
        "partition": "validation",
        "partition_sha256": partition["partition_sha256"],
        "analysis_identity": {
            "study_id": "synthetic",
            "run_id": "fixture",
            "model_sha256": verified.identity["export_files"]["model.safetensors"],
            "protocol_sha256": "e" * 64,
        },
        "full_rejected": True,
        "reason": "synthetic fixed-subset contract fixture, no GPU evidence",
        "measurement_reference": "fixture://cost",
        "measurement_sha256": "f" * 64,
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_bytes(canonical_bytes(decision))
    coverage_dir = tmp_path / "coverage"
    checksum = freeze_coverage(
        manifest,
        decision_path,
        coverage_dir,
        decision_checksum=hashlib.sha256(decision_path.read_bytes()).hexdigest(),
        mode="subset",
        size=400,
        fixture=True,
    )
    assert read_coverage(coverage_dir, checksum=checksum)["storage_ids"] == statistics
    root = tmp_path / "subset_main"
    plan = prepare_main(
        verified, root, coverage_directory=coverage_dir, coverage_checksum=checksum
    )
    assert plan["statistics_ids"] == statistics
    run_main(verified, root, device="cpu", stages=())
    report = report_main(root)
    agreement = json.loads((Path(report["output_dir"]) / "agreement.json").read_bytes())
    assert agreement["case_only_ids"] == case_only
    assert agreement["populations"]["all"]["n_total"] == 400
    assert set(case_only).isdisjoint(agreement["populations"]["all"]["source_ids"])
    assert set(case_only) <= set(report["coverage"]["ig"]["requested"])
    assert len(report["coverage"]["ig"]["requested"]) == 402
