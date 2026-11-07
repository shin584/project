import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cas12a.nt_data import canonical_bytes, digest
from xai_test.cas12a_NT_analysis.nt_artifacts import prediction_rows, read_results
from xai_test.cas12a_NT_analysis.nt_ig import PROTOCOL as IG_PROTOCOL
from xai_test.cas12a_NT_analysis.nt_ig import quality, summarize
from xai_test.cas12a_NT_analysis.nt_ig_artifacts import (
    plan_ig,
    read_ig,
    save_ig,
    save_ig_plan,
)
from xai_test.cas12a_NT_analysis.nt_inputs import (
    DNA_SPANS,
    encode_sequences,
    sequence_hash,
    verify_inputs,
)
from xai_test.cas12a_NT_analysis.nt_ism import PROTOCOL as ISM_PROTOCOL
from xai_test.cas12a_NT_analysis.nt_ism import compute_ism, read_ism, save_ism
from xai_test.cas12a_NT_analysis.nt_run import STUDY_ID, Study
from xai_test.cas12a_NT_analysis.nt_selection import (
    read_cases,
    read_coverage,
    save_cases,
)
from xai_test.cas12a_NT_analysis.test.test_ig import POLICY, scalar_predictor
from xai_test.cas12a_NT_analysis.test.test_selection import freeze, sequence


@pytest.fixture
def agreement_study(inputs, tmp_path):
    predictor = scalar_predictor(inputs)
    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    sequences = [row["sequence"] for row in verified.rows]
    rows = prediction_rows(
        verified.rows,
        predictor.predict(sequences),
        encode_sequences(predictor.tokenizer, sequences, 4105),
    )
    identity = {
        **verified.identity,
        "protocol": {"ig": IG_PROTOCOL, "ism": ISM_PROTOCOL, "batch_size": 32},
        "code_sha256": "a" * 64,
    }
    study = Study.create(tmp_path / "study", fixture=True)
    study.register(
        "a",
        identity,
        snapshots={
            f"selected_export/{name}": (inputs.export / name).read_bytes()
            for name in ("vocab.txt", "config.json")
        },
    )
    study.start("a", "predictions", 300, {"device": "cpu"})
    study.save_artifact(
        "predictions",
        {"schema": 1, "identity": identity, "rows": rows},
        [row["source_row"] for row in rows],
    )
    study.finish()
    # This fixture tests stored-reader contracts, not attribution computation.
    # Deliberately distinct accepted resolutions exercise final-resolution use.
    policy = {**POLICY, "convergence_atol": 1.0}
    plan = plan_ig(
        read_results(study, "a"),
        run_id="a",
        policy=policy,
        probe_ids=[row["source_row"] for row in rows],
    )
    study.start("a", "explanations", 300, {"device": "cpu"})
    study.record_execution(
        {"device": "cpu"}, {"validated": True, "injected_cpu_model": True}
    )
    save_ig_plan(study, "a", plan)
    # Opposite method shard order deliberately forbids positional joining.
    for row in reversed(rows):
        scale = 1 + row["source_row"] / 16
        final = np.array([1, *[scale * i for i in range(1, 10)], 0, 0])
        final *= 0.125 / final.sum()
        mass = np.array([final[[0, *range(9, 0, -1), 10, 11]], final])
        outputs = {"native": 0.375, "endpoint": 0.375, "reference": 0.25}
        attempts = [
            {
                "steps": steps,
                "batch_sizes": [16] * (steps // 16)
                + ([steps % 16] if steps % 16 else []),
                "quality": quality(
                    mass[i], mass[i - 1] if i else None, outputs, policy
                ),
            }
            for i, steps in enumerate((200, 400))
        ]
        binding = {
            key: row[key]
            for key in ("source_row", "sequence", "sequence_sha256", "raw_prediction")
        }
        save_ig(
            study,
            "a",
            {
                "schema": 1,
                "row": {**binding, "prediction_sha256": digest(row)},
                "tokens": row["tokens"],
                "protocol": IG_PROTOCOL,
                "policy": policy,
                "integration_batch_size": 16,
                "vocabulary": {
                    "vocabulary_sha256": digest(predictor.tokenizer.get_vocab()),
                    "vocabulary_size": 4107,
                    "embedding_rows": 4105,
                },
                "execution": {
                    "parameter_dtype": "float32",
                    "output_dtype": "float32",
                    "autocast": False,
                    "tf32": False,
                    "backend": "eager",
                    "eval": True,
                    "input_gradient_enabled": True,
                    "device": "cpu",
                },
                "reference": {
                    "kind": IG_PROTOCOL["reference"],
                    "shape": [12, 8],
                    "dtype": "float32",
                    "input_ids": row["tokens"]["input_ids"],
                    "attention_mask": row["tokens"]["attention_mask"],
                },
                "arrays": {"signed_mass": mass},
                "outputs": outputs,
                "attempts": attempts,
                "accepted": True,
                "failure": None,
                "summary": summarize(mass, attempts, outputs),
            },
        )
    for row in rows:

        def mutant_forward(sequences, reference=row["sequence"]):
            return np.array(
                [
                    0.375
                    + sum(
                        (i + 1) / 128
                        for i, (a, b) in enumerate(zip(seq, reference, strict=True))
                        if a != b
                    )
                    for seq in sequences
                ],
                dtype=np.float32,
            )

        save_ism(
            study, "a", compute_ism([row], mutant_forward, predictor.tokenizer, 4105)
        )
    study.finish()
    return study, rows


def test_checked_join_is_by_source_id_and_exports_recompute_without_changing_study(
    agreement_study, inputs, tmp_path
):
    from xai_test.cas12a_NT_analysis.nt_agreement_artifacts import (
        load_agreement,
        read_agreement,
        save_agreement,
    )

    study, rows = agreement_study
    before = copy.deepcopy(study.receipt)
    result = read_agreement(study, "a")
    ids = sorted(row["source_row"] for row in rows)
    assert [row["source_row"] for row in result["rows"]] == ids
    assert result["complete"] is True
    assert result["populations"]["statistics"]["n_total"] == 0
    summary = result["populations"]["validation_probes"]
    assert summary["source_ids"] == ids
    assert summary["primary"]["n_total"] == len(ids)
    assert summary["primary"]["n_valid"] == len(ids)
    assert summary["primary"]["median"] == pytest.approx(1)
    assert summary["primary"]["excluded_reasons"] == {}
    output = tmp_path / "agreement"
    checksum = save_agreement(study, "a", output)
    assert load_agreement(output, checksum=checksum) == result
    assert study.receipt == before
    assert result["rows"][0]["tokens"] == rows[0]["tokens"]
    assert (
        result["rows"][0]["agreement"]["versions"]["exclude_pam_breaking"]["tokens"][0][
            "n_valid_bases"
        ]
        == 4
    )

    # The complete saved study and its standalone export both read model-free.
    unavailable_export = tmp_path / "unavailable-export"
    assert inputs.export.resolve().is_relative_to(tmp_path.resolve())
    assert unavailable_export.resolve().is_relative_to(tmp_path.resolve())
    inputs.export.rename(unavailable_export)
    inputs.source.unlink()
    inputs.manifest.unlink()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(study.receipt))
    code = """
import builtins, json, sys
original_import = builtins.__import__
def model_free(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'transformers'):
        raise AssertionError('model import attempted: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = model_free
from xai_test.cas12a_NT_analysis.nt_run import Study
from xai_test.cas12a_NT_analysis.nt_agreement_artifacts import read_agreement, load_agreement
study = Study.reopen(sys.argv[1], receipt=json.load(open(sys.argv[2])))
assert read_agreement(study, 'a') == load_agreement(sys.argv[3], checksum=sys.argv[4])
"""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(study.store.root),
            str(receipt),
            str(output),
            checksum,
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr


def test_join_rejects_identity_token_and_source_mismatches_and_preserves_missing_ids(
    agreement_study,
):
    from xai_test.cas12a_NT_analysis.nt_agreement_artifacts import compare_saved

    study, _ = agreement_study
    predictions, ig, ism = (
        read_results(study, "a"),
        read_ig(study, "a"),
        read_ism(study, "a"),
    )
    # Explicitly shuffle both reader views, maintaining their own array/ID alignment.
    ig["results"].reverse()
    ism["rows"].reverse()
    ism["arrays"] = {key: value[::-1] for key, value in ism["arrays"].items()}
    baseline = compare_saved(predictions, ig, ism)
    assert baseline["complete"] is True
    for method, field in (
        ("ig", "code_sha256"),
        ("ig", "export_files"),
        ("ism", "protocol"),
        ("ism", "source_sha256"),
    ):
        changed_ig, changed_ism = copy.deepcopy(ig), copy.deepcopy(ism)
        changed = changed_ig if method == "ig" else changed_ism
        changed["identity"][field] = "mismatched"
        with pytest.raises(ValueError, match="identity"):
            compare_saved(predictions, changed_ig, changed_ism)
    for field in ("sequence_sha256", "prediction_sha256"):
        changed = copy.deepcopy(ig)
        changed["results"][0]["row"][field] = "mismatched"
        with pytest.raises(ValueError, match="identity"):
            compare_saved(predictions, changed, ism)
    changed = copy.deepcopy(ig)
    changed["results"][0]["tokens"]["dna_spans"][0][1] = 5
    with pytest.raises(ValueError, match="token/span"):
        compare_saved(predictions, changed, ism)
    partial = copy.deepcopy(ig)
    missing_id = partial["results"].pop()["row"]["source_row"]
    partial["coverage"].remove(missing_id)
    partial["accepted_coverage"].remove(missing_id)
    partial["missing"] = [missing_id]
    partial["complete"] = False
    result = compare_saved(predictions, partial, ism)
    assert result["complete"] is False
    assert result["unavailable_ids"] == [missing_id]
    summary = result["populations"]["validation_probes"]["primary"]
    assert summary["n_total"] == len(ig["coverage"])
    assert summary["excluded_reasons"]["missing IG"] == 1
    failed = copy.deepcopy(ig)
    failed["results"][0]["accepted"] = False
    failed["results"][0]["failure"] = {"stage": "quality", "reason": "fixture failure"}
    failed["complete"] = False
    failed_result = compare_saved(predictions, failed, ism)
    assert failed_result["complete"] is False
    assert (
        "failed IG: quality: fixture failure"
        in failed_result["populations"]["validation_probes"]["primary"][
            "excluded_reasons"
        ]
    )


@pytest.mark.parametrize("mode,size", [("full", None), ("subset", 400)])
def test_frozen_full_and_subset_reports_keep_cases_outside_statistical_denominators(
    selection_inputs, tmp_path, mode, size
):
    from xai_test.cas12a_NT_analysis.nt_agreement_artifacts import compare_saved

    partition_path, decision_path, _ = selection_inputs
    protocol = {"ig": IG_PROTOCOL, "ism": ISM_PROTOCOL, "batch_size": 32}
    analysis_identity = {
        "study_id": STUDY_ID,
        "run_id": "a",
        "model_sha256": "d" * 64,
        "protocol_sha256": digest(protocol),
    }
    decision = json.loads(decision_path.read_bytes())
    decision["analysis_identity"] = analysis_identity
    decision_path.write_bytes(canonical_bytes(decision))
    root, case_root = tmp_path / "coverage", tmp_path / "cases"
    coverage_sha = freeze(selection_inputs, root, mode=mode, size=size)
    coverage = read_coverage(root, checksum=coverage_sha)
    rows = []
    for source_id in coverage["population_ids"]:
        seq = sequence(source_id)
        token_ids = []
        for start, end in DNA_SPANS:
            text = seq[start:end]
            token_ids.append(
                4101 + "ACGT".index(text)
                if len(text) == 1
                else 5 + int("".join(str("ACGT".index(base)) for base in text), 4)
            )
        rows.append(
            {
                "source_row": source_id,
                "sequence": seq,
                "sequence_sha256": sequence_hash(seq),
                "raw_prediction": float((source_id % 16) / 16),
                "label": 0.0,
                "tokens": {
                    "input_ids": [3, *token_ids, 1, 1],
                    "attention_mask": [1] * 10 + [0, 0],
                    "special_mask": [True] + [False] * 9 + [True, True],
                    "dna_spans": DNA_SPANS,
                },
            }
        )
    case_sha = save_cases(
        root,
        case_root,
        coverage_checksum=coverage_sha,
        predictions=rows,
        prediction_identity=analysis_identity,
    )
    cases = read_cases(
        root, case_root, coverage_checksum=coverage_sha, checksum=case_sha
    )
    partition = json.loads(partition_path.read_bytes())
    identity = {
        "fixture": True,
        "partition": "test",
        "protocol": protocol,
        "export_files": {"model.safetensors": "d" * 64},
        "partition_sha256": partition["partition_sha256"],
        "source_sha256": partition["source_file_sha256"],
        "table_sha256": partition["table_sha256"],
    }
    predictions = {
        "identity": identity,
        "run_id": "a",
        "rows": rows,
        "coverage": coverage["population_ids"],
        "coverage_complete": True,
        "complete": True,
        "missing": [],
        "job_unresolved": False,
    }
    plan = plan_ig(
        predictions, run_id="a", policy=POLICY, coverage=coverage, cases=cases
    )
    statistics = set(plan["work"]["statistics_ids"])
    ig_results = []
    ism_rows, delta = [], []
    for row in rows:
        source_id = row["source_row"]
        binding = {
            key: row[key]
            for key in ("source_row", "sequence", "sequence_sha256", "raw_prediction")
        }
        binding["prediction_sha256"] = digest(row)
        if source_id in plan["work"]["ids"]:
            # Case-only examples deliberately disagree, while random targets agree.
            density = (
                [1, 2, 3, 4, 5, 6, 7, 8, 9]
                if source_id in statistics
                else [5, 4, 3, 2, 1, 6, 7, 8, 9]
            )
            mass = np.array(density) * np.array([6] * 5 + [1] * 4)
            ig_results.append(
                {
                    "row": binding,
                    "tokens": row["tokens"],
                    "accepted": True,
                    "arrays": {
                        "signed_mass": np.array([[0, *mass, 0, 0]], dtype=float)
                    },
                }
            )
        ism_rows.append({**binding, "tokens": [row["tokens"]]})
        values = np.empty((34, 4), dtype=float)
        for i, (start, end) in enumerate(DNA_SPANS):
            values[start:end] = i + 1
        for i, base in enumerate(row["sequence"]):
            values[i, "ACGT".index(base)] = 0
        delta.append(values)
    ig = {
        "run_id": "a",
        "identity": identity,
        "plan": plan,
        "results": list(reversed(ig_results)),
        "coverage": plan["work"]["ids"],
        "accepted_coverage": plan["work"]["ids"],
        "missing": [],
        "failed": [],
        "complete": True,
        "job_unresolved": False,
    }
    ism = {
        "run_id": "a",
        "identity": identity,
        "rows": list(reversed(ism_rows)),
        "coverage": coverage["population_ids"],
        "missing": [],
        "complete": True,
        "job_unresolved": False,
        "arrays": {
            "source_row": np.array(coverage["population_ids"])[::-1],
            "raw_delta": np.array(delta)[::-1],
            "base_valid_mask": np.ones((len(rows), 34, 4), dtype=bool),
        },
    }
    result = compare_saved(predictions, ig, ism)
    stats = result["populations"]["statistics"]
    assert stats["n_total"] == (1500 if mode == "full" else 400)
    assert stats["source_ids"] == coverage["storage_ids"]
    assert stats["primary"]["median"] == pytest.approx(1)
    assert stats["primary"]["n_valid"] == len(statistics)
    assert stats["primary"]["distribution"] == pytest.approx([1] * len(statistics))
    profile = stats["versions"]["all_nonWT"]["five_6mer"]["density"]["mean_profile"]
    assert [p["n_paired"] for p in profile["profile"]] == [len(statistics)] * 5
    assert profile["correlation"]["rho"] == pytest.approx(1)
    additional = result["populations"]["additional_cases"]
    assert additional["source_ids"] == plan["work"]["additional_case_ids"]
    if mode == "subset":
        assert additional["n_total"] > 0
        assert additional["primary"]["median"] == pytest.approx(-1)
    else:
        assert additional["n_total"] == 0


def test_exports_reject_overwrite_study_paths_corruption_and_fabricated_summaries(
    agreement_study, tmp_path
):
    from xai_test.cas12a_NT_analysis.nt_agreement_artifacts import (
        load_agreement,
        save_agreement,
    )

    study, _ = agreement_study
    output = tmp_path / "export"
    checksum = save_agreement(study, "a", output)
    with pytest.raises(ValueError, match="new empty"):
        save_agreement(study, "a", output)
    with pytest.raises(ValueError, match="outside"):
        save_agreement(study, "a", study.store.root / "new-report")
    assert not (study.store.root / "new-report").exists()
    with pytest.raises(ValueError, match="manifest checksum"):
        load_agreement(output, checksum="0" * 64)
    for name in (
        "inputs.json",
        "results.json",
        "manifest.json",
        "missing",
        "extra",
        "fabricated",
    ):
        destination = tmp_path / ("corrupt-" + name)
        shutil.copytree(output, destination)
        trusted_sha = checksum
        if name == "missing":
            (destination / "inputs.json").unlink()
        elif name == "extra":
            (destination / "extra.txt").write_text("uncommitted")
        elif name == "fabricated":
            result = json.loads((destination / "results.json").read_bytes())
            result["populations"]["validation_probes"]["primary"]["median"] = -1
            result_bytes = canonical_bytes(result)
            (destination / "results.json").write_bytes(result_bytes)
            manifest = json.loads((destination / "manifest.json").read_bytes())
            import hashlib

            manifest["files"]["results.json"] = hashlib.sha256(result_bytes).hexdigest()
            raw = canonical_bytes(manifest)
            (destination / "manifest.json").write_bytes(raw)
            trusted_sha = hashlib.sha256(raw).hexdigest()
        else:
            (destination / name).write_bytes(b"corrupt")
        with pytest.raises(ValueError):
            load_agreement(destination, checksum=trusted_sha)
    # A partial export is neither silently completed nor replaced.
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "inputs.json").write_bytes(b"partial")
    with pytest.raises(ValueError, match="new empty"):
        save_agreement(study, "a", partial)
