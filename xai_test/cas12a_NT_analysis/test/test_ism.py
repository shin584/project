import numpy as np
import pytest

from xai_test.cas12a_NT_analysis.nt_inputs import sequence_hash


def test_full_sequence_mutations_preserve_wt_and_distinguish_delta_views(inputs):
    from transformers import AutoTokenizer

    from xai_test.cas12a_NT_analysis.nt_ism import compute_ism

    sequence = "AAAA" + "TTTA" + "A" * 26
    row = {
        "source_row": 72,
        "sequence": sequence,
        "sequence_sha256": sequence_hash(sequence),
        "raw_prediction": -0.5,
    }
    observed = []

    def forward(sequences):
        assert len(sequences) <= 16
        observed.extend(sequences)
        return np.asarray(
            [{"A": -0.5, "C": 0.5, "G": 1.5, "T": 2.5}[s[0]] for s in sequences],
            dtype=np.float32,
        )

    tokenizer = AutoTokenizer.from_pretrained(inputs.export, local_files_only=True)
    result = compute_ism([row], forward, tokenizer, 4105, batch_size=16)
    arrays = result["arrays"]
    assert len(observed) == len(set(observed)) == 103
    assert observed[0] == sequence
    assert all(sum(a != b for a, b in zip(s, sequence)) == 1 for s in observed[1:])
    assert arrays["prediction"].shape == (1, 34, 4)
    np.testing.assert_array_equal(arrays["prediction"][0, 0], [-0.5, 0.5, 1.5, 2.5])
    np.testing.assert_array_equal(arrays["raw_delta"][0, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(arrays["clipped_delta"][0, 0], [0, 0.5, 1, 1])
    np.testing.assert_array_equal(arrays["relative_delta"][0, 0], [0, 2, 4, 6])
    assert arrays["non_wt_mask"].sum() == 102
    assert arrays["wt_mask"].sum() == 34
    assert arrays["pam_breaking"].sum() == 9
    assert arrays["tttn_preserving"][0, 7].all()
    np.testing.assert_array_equal(
        arrays["preserving_denominator"][0, 4:8], [0, 0, 0, 3]
    )
    assert np.all(arrays["raw_delta"][arrays["wt_mask"]] == 0)
    assert len(result["rows"][0]["tokens"]) == 103


@pytest.fixture
def saved_predictions(inputs, tmp_path):
    from transformers import AutoTokenizer

    from xai_test.cas12a_NT_analysis.nt_artifacts import prediction_rows
    from xai_test.cas12a_NT_analysis.nt_inputs import encode_sequences, verify_inputs
    from xai_test.cas12a_NT_analysis.nt_ism import PROTOCOL
    from xai_test.cas12a_NT_analysis.nt_run import Study

    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    tokenizer = AutoTokenizer.from_pretrained(inputs.export, local_files_only=True)
    tokens = encode_sequences(
        tokenizer, [row["sequence"] for row in verified.rows], 4105
    )
    rows = prediction_rows(verified.rows, [-0.5, 0.0, 1.5, 0.25], tokens)
    identity = {
        **verified.identity,
        "protocol": {"ism": PROTOCOL, "batch_size": 16},
        "code_sha256": "synthetic-code",
        "packages": {"numpy": np.__version__},
        "python": "fixture",
    }
    study = Study.create(tmp_path / "study", fixture=True)
    study.register(
        "a",
        identity,
        snapshots={
            "selected_export/vocab.txt": (inputs.export / "vocab.txt").read_bytes()
        },
    )
    study.start("a", "predictions", 300, {"device": "cpu"})
    study.save_artifact(
        "predictions",
        {"schema": 1, "identity": identity, "rows": rows},
        [row["source_row"] for row in rows],
    )
    study.finish()
    return study, rows, tokenizer


def test_ism_roundtrip_joins_ids_and_reconstructs_undefined_values(saved_predictions):
    from xai_test.cas12a_NT_analysis.nt_ism import compute_ism, read_ism, save_ism

    study, rows, tokenizer = saved_predictions
    study.start("a", "ism-cpu", 300, {"device": "cpu"})
    study.record_execution(
        {"device": "cpu"}, {"validated": True, "injected_forward": True}
    )

    def forward(sequences):
        return np.asarray(
            [0.0 if s == rows[1]["sequence"] else -0.5 for s in sequences],
            dtype=np.float32,
        )

    result = compute_ism(rows[:2], forward, tokenizer, 4105, batch_size=16)
    save_ism(study, "a", result)
    sidecar = study.artifacts("a", kind="ism")[0][1]
    assert sidecar["summaries"][1]["relative_delta"] is None
    assert sidecar["summaries"][1]["relative_undefined_reason"] == "abs(WT)<=1e-4"
    partial = read_ism(
        study, "a", source_ids=[rows[1]["source_row"], rows[0]["source_row"]]
    )
    assert partial["coverage"] == [row["source_row"] for row in rows[:2]]
    assert partial["arrays"]["source_row"].tolist() == [
        rows[1]["source_row"],
        rows[0]["source_row"],
    ]
    assert partial["complete"] is False
    assert partial["mutant_count"] == 204
    assert partial["summaries"][0]["relative_delta"] is None
    assert partial["summaries"][0]["relative_undefined_reason"] == "abs(WT)<=1e-4"
    assert np.isnan(partial["arrays"]["relative_delta"][0]).all()
    assert partial["summaries"][0]["preserving_sensitivity"][4] is None
    assert (
        partial["summaries"][0]["preserving_undefined_reason"][4]
        == "no eligible non-WT alternatives"
    )
    with pytest.raises(ValueError, match="incomplete"):
        read_ism(study, "a", source_ids=[rows[0]["source_row"]], require_complete=True)
    with pytest.raises(ValueError, match="overlapping"):
        save_ism(study, "a", result)
    study.finish()
    report = read_ism(study, "a")
    assert report["coverage_complete"] is False
    assert report["missing"] == [row["source_row"] for row in rows[2:]]
    assert len(study.artifacts("a")) == 1  # prediction coverage coexists with ISM


@pytest.fixture
def ism_shard(saved_predictions):
    from xai_test.cas12a_NT_analysis.nt_ism import compute_ism

    study, rows, tokenizer = saved_predictions
    study.start("a", "ism", 300, {"device": "cpu"})
    study.record_execution(
        {"device": "cpu"}, {"validated": True, "injected_forward": True}
    )
    result = compute_ism(
        rows[:1],
        lambda sequences: np.full(len(sequences), -0.5, dtype=np.float32),
        tokenizer,
        4105,
        batch_size=16,
    )
    return study, rows, result


@pytest.mark.parametrize(
    "failure",
    ["raw_delta", "wt_slot", "mutant_tokens", "source_binding", "denominator"],
)
def test_writer_rejects_inconsistent_computation_before_committing(ism_shard, failure):
    from xai_test.cas12a_NT_analysis.nt_ism import save_ism

    study, _, result = ism_shard
    if failure == "raw_delta":
        result["arrays"]["raw_delta"][0, 0, 0] = 42
    elif failure == "wt_slot":
        result["arrays"]["prediction"][result["arrays"]["wt_mask"]] = 0.5
    elif failure == "mutant_tokens":
        result["rows"][0]["tokens"][1]["input_ids"][1] = 4106
    elif failure == "source_binding":
        result["rows"][0]["prediction_sha256"] = "other-population"
    else:
        result["arrays"]["preserving_denominator"][0, 4] = 3
    with pytest.raises(ValueError):
        save_ism(study, "a", result)
    assert study.artifacts("a", kind="ism") == []


@pytest.mark.parametrize("failure", ["corrupt", "missing"])
def test_reader_blocks_damaged_committed_arrays(ism_shard, failure):
    from xai_test.cas12a_NT_analysis.nt_ism import read_ism, save_ism

    study, _, result = ism_shard
    save_ism(study, "a", result)
    path = next(study.store.root.glob("runs/a/artifacts/ism-*.npz"))
    if failure == "corrupt":
        path.write_bytes(b"damaged synthetic shard")
    else:
        path.unlink()
    with pytest.raises((ValueError, OSError)):
        read_ism(study, "a")
    assert study.poisoned is True


def test_model_free_reader_needs_only_committed_study_and_ignores_orphans(
    ism_shard, inputs, tmp_path
):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from xai_test.cas12a_NT_analysis.nt_ism import read_ism, save_ism

    study, _, result = ism_shard
    save_ism(study, "a", result)
    study.finish()
    study.store.put(
        "runs/a/artifacts/ism-orphan.npz", b"unfinished unreferenced fixture"
    )
    assert read_ism(study, "a")["mutant_count"] == 102
    inputs.export.rename(tmp_path / "model-unavailable")
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
from xai_test.cas12a_NT_analysis.nt_ism import read_ism
study = Study.reopen(sys.argv[1], receipt=json.load(open(sys.argv[2])))
result = read_ism(study, 'a')
assert result['mutant_count'] == 102
assert result['arrays']['prediction'].shape == (1, 34, 4)
assert len(result['missing']) == 3
print(json.dumps(result['summaries'], allow_nan=False))
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(study.store.root), str(receipt)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize("wt", [-1e-4, 0.0, 1e-4, 1.1e-4])
def test_relative_delta_floor_preserves_undefined_mask(inputs, wt):
    from transformers import AutoTokenizer

    from xai_test.cas12a_NT_analysis.nt_ism import compute_ism

    sequence = "AAAA" + "TTTA" + "A" * 26
    wt = float(np.float32(wt))
    row = {
        "source_row": 1,
        "sequence": sequence,
        "sequence_sha256": sequence_hash(sequence),
        "raw_prediction": wt,
    }
    result = compute_ism(
        [row],
        lambda sequences: np.full(len(sequences), wt, dtype=np.float32),
        AutoTokenizer.from_pretrained(inputs.export, local_files_only=True),
        4105,
    )
    arrays = result["arrays"]
    assert bool(arrays["relative_valid"][0]) == (abs(wt) > 1e-4)
    if abs(wt) <= 1e-4:
        assert np.isnan(arrays["relative_delta"]).all()
        assert result["rows"][0]["relative_undefined_reason"] == "abs(WT)<=1e-4"
    else:
        assert np.all(arrays["relative_delta"] == 0)
        assert result["rows"][0]["relative_undefined_reason"] is None


@pytest.mark.parametrize("mode", ["float64", "nonfinite", "shape", "wt_drift"])
def test_invalid_forward_is_rejected(inputs, mode):
    from transformers import AutoTokenizer

    from xai_test.cas12a_NT_analysis.nt_ism import compute_ism

    sequence = "AAAA" + "TTTA" + "A" * 26
    row = {
        "source_row": 1,
        "sequence": sequence,
        "sequence_sha256": sequence_hash(sequence),
        "raw_prediction": 0.5,
    }

    def forward(sequences):
        if mode == "float64":
            return np.full(len(sequences), 0.5, dtype=np.float64)
        if mode == "nonfinite":
            return np.full(len(sequences), np.nan, dtype=np.float32)
        if mode == "shape":
            return np.full((len(sequences), 1), 0.5, dtype=np.float32)
        return np.full(len(sequences), 0.6, dtype=np.float32)

    with pytest.raises(ValueError, match="FP32|parity"):
        compute_ism(
            [row],
            forward,
            AutoTokenizer.from_pretrained(inputs.export, local_files_only=True),
            4105,
        )


def test_mutant_tokenization_is_checked_before_forward_and_budget_stops_next_batch(
    inputs,
):
    from transformers import AutoTokenizer

    from xai_test.cas12a_NT_analysis.nt_ism import compute_ism

    sequence = "AAAA" + "TTTA" + "A" * 26
    row = {
        "source_row": 1,
        "sequence": sequence,
        "sequence_sha256": sequence_hash(sequence),
        "raw_prediction": 0.5,
    }
    tokenizer = AutoTokenizer.from_pretrained(inputs.export, local_files_only=True)

    class DamagedTokenizer:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def __call__(self, value, **options):
            encoded = tokenizer(value, **options)
            if value != sequence:
                encoded["input_ids"][1] = 4106
            return encoded

    calls = []

    def forward(sequences):
        calls.append(sequences)
        return np.full(len(sequences), 0.5, dtype=np.float32)

    with pytest.raises(ValueError, match="token|span"):
        compute_ism([row], forward, DamagedTokenizer(), 4105, batch_size=16)
    assert calls == []

    def expired():
        if calls:
            raise ValueError("reservation expired")

    with pytest.raises(ValueError, match="reservation expired"):
        compute_ism([row], forward, tokenizer, 4105, batch_size=16, guard=expired)
    assert len(calls) == 1


def test_positive_out_of_range_wt_uses_clipped_endpoints(inputs):
    from transformers import AutoTokenizer

    from xai_test.cas12a_NT_analysis.nt_ism import compute_ism

    sequence = "AAAA" + "TTTA" + "A" * 26
    row = {
        "source_row": 9,
        "sequence": sequence,
        "sequence_sha256": sequence_hash(sequence),
        "raw_prediction": 1.5,
    }

    def forward(sequences):
        return np.asarray(
            [1.5 if s[0] == "A" else 0.5 for s in sequences], dtype=np.float32
        )

    result = compute_ism(
        [row],
        forward,
        AutoTokenizer.from_pretrained(inputs.export, local_files_only=True),
        4105,
    )
    np.testing.assert_array_equal(result["arrays"]["raw_delta"][0, 0], [0, -1, -1, -1])
    np.testing.assert_array_equal(
        result["arrays"]["clipped_delta"][0, 0], [0, -0.5, -0.5, -0.5]
    )


@pytest.mark.parametrize("failure", ["read", "hash"])
@pytest.mark.parametrize("file_kind", ["vocabulary", "npz"])
def test_transient_read_after_integrity_scan_poisoning_requires_reopen(
    ism_shard, monkeypatch, file_kind, failure
):
    from xai_test.cas12a_NT_analysis.nt_ism import read_ism, save_ism

    study, _, result = ism_shard
    save_ism(study, "a", result)
    original = study.store.read
    calls = 0

    def broken_read(name):
        nonlocal calls
        matching = (
            name.endswith("vocab.txt")
            if file_kind == "vocabulary"
            else name.endswith(".npz")
        )
        if matching:
            calls += 1
            if calls >= 3:  # transient failure after the preceding ledger scans
                if failure == "read":
                    raise OSError("transient external read failure")
                return original(name) + b"changed after integrity scan"
        return original(name)

    monkeypatch.setattr(study.store, "read", broken_read)
    with pytest.raises((OSError, ValueError), match="transient external|hash mismatch"):
        read_ism(study, "a")
    assert study.poisoned is True
    monkeypatch.setattr(study.store, "read", original)
    with pytest.raises(ValueError, match="poisoned"):
        study.checkpoint()


@pytest.mark.parametrize("failed_file", ["npz", "json"])
def test_failed_shard_write_cannot_commit_partial_coverage(
    ism_shard, monkeypatch, failed_file
):
    from xai_test.cas12a_NT_analysis.nt_ism import save_ism

    study, _, result = ism_shard
    original = study.store.put

    def broken_put(name, data):
        if "/artifacts/ism-" in name and name.endswith("." + failed_file):
            raise OSError("external shard write failed")
        return original(name, data)

    monkeypatch.setattr(study.store, "put", broken_put)
    with pytest.raises(OSError, match="shard write failed"):
        save_ism(study, "a", result)
    assert study.poisoned is True
    assert not any(
        record["kind"] in ("ism", "ism-array")
        for record in study.status()["runs"]["a"]["files"].values()
    )
    if failed_file == "json":
        assert len(list(study.store.root.glob("runs/a/artifacts/ism-*.npz"))) == 1


def test_full_coverage_requires_closed_job_even_when_selecting_one_id(
    saved_predictions,
):
    from xai_test.cas12a_NT_analysis.nt_ism import compute_ism, read_ism, save_ism

    study, rows, tokenizer = saved_predictions
    study.start("a", "ism", 300, {"device": "cpu"})
    study.record_execution(
        {"device": "cpu"}, {"validated": True, "injected_forward": True}
    )
    originals = {row["sequence"]: row["raw_prediction"] for row in rows}

    def forward(sequences):
        return np.asarray(
            [originals.get(sequence, 0.75) for sequence in sequences], dtype=np.float32
        )

    result = compute_ism(rows, forward, tokenizer, 4105, batch_size=16)
    save_ism(study, "a", result)
    report = read_ism(study, "a")
    assert report["coverage_complete"] is True
    assert report["complete"] is False
    assert report["job_unresolved"] is True
    with pytest.raises(ValueError, match="unresolved"):
        read_ism(study, "a", source_ids=[rows[-1]["source_row"]], require_complete=True)
    study.finish()
    selected = read_ism(
        study, "a", source_ids=[rows[-1]["source_row"]], require_complete=True
    )
    assert selected["complete"] is True
    assert selected["mutant_count"] == 408
    assert selected["arrays"]["source_row"].tolist() == [rows[-1]["source_row"]]
