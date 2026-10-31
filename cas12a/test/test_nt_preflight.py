"""Offline NT preflight through the same public lifecycle used on Colab."""

import itertools
import json

import pytest

from cas12a.nt_workflow import NTWorkflow, SplitConfig
from cas12a.test.test_nt_workflow import synthetic_table


@pytest.fixture
def tiny_nt(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    path = tmp_path / "generic-tiny-nt"
    path.mkdir()
    vocabulary = ["<cls>", "<pad>", "<mask>", "<unk>"]
    vocabulary += ["".join(bases) for bases in itertools.product("ACGT", repeat=6)]
    vocabulary += list("ACGTN")
    (path / "vocab.txt").write_text("\n".join(vocabulary) + "\n")
    tokenizer = transformers.EsmTokenizer(str(path / "vocab.txt"), eos_token=None)
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    config = transformers.EsmConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
        pad_token_id=1,
        mask_token_id=2,
        token_dropout=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    transformers.EsmForMaskedLM(config).save_pretrained(path)
    return path


def prepared(tmp_path, *, production=False, counts=(8, 1, 1)):
    source = tmp_path / "source.csv"
    synthetic_table(15000 if production else sum(counts)).to_csv(source, index=False)
    workflow = NTWorkflow(
        source,
        tmp_path / "partition.json",
        tmp_path / "local",
        tmp_path / "drive",
        SplitConfig() if production else SplitConfig.fixture(*counts),
    )
    return workflow, workflow.prepare(run_id="preflight")


def artifact(run, name):
    local = (run.local_dir / name).read_bytes()
    assert local == (run.durable_dir / name).read_bytes()
    return json.loads(local)


def test_fixture_preflight_records_complete_tokenization_and_two_real_updates(
    tmp_path, tiny_nt
):
    from cas12a.nt_workflow import ModelSource

    workflow, run = prepared(tmp_path)
    result = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))

    assert result.status["state"] == "preflight_passed"
    assert result.status["model_completed"] is False
    evidence = artifact(result, "preflight.json")
    assert evidence["evidence_kind"] == "offline_fixture"
    assert evidence["source"]["kind"] == "fixture"
    assert evidence["gpu_preflight_executed"] is False
    assert evidence["attempts"][0]["optimizer_updates"] == 2
    assert evidence["attempts"][0]["post_adam_backward_finite"] is True
    assert evidence["attempts"][0]["validation_examples"] == 1
    assert evidence["training_source_rows"] == [5, 6, 0, 7, 3, 2, 4, 9]
    assert evidence["validation_source_rows"] == [1]
    tokens = artifact(result, "tokenization.json")
    assert tokens["checked_sequences"] == 10
    assert tokens["unpadded_length_distribution"] == {"10": 10}
    assert tokens["max_length"] == 12
    assert tokens["truncation"] is False
    assert tokens["vocabulary_size"] == 4105
    execution = artifact(result, "execution.json")
    assert execution["settings"]["train_microbatch"] == 32
    assert execution["settings"]["gradient_accumulation"] == 1
    assert execution["settings"]["evaluation_microbatch"] == 32
    assert execution["settings"]["gradient_checkpointing"] is False
    assert execution["settings"]["precision"] == "float32"
    assert (
        execution["model"]["trainable_parameters"]
        == execution["model"]["total_parameters"]
    )
    assert execution["model"]["problem_type"] == "regression"
    assert execution["environment"]["packages"]["torch"]
    assert execution["frozen"] is True


def test_fit_initialization_discards_preflight_and_recreates_all_training_state(
    tmp_path, tiny_nt
):
    import torch

    from cas12a.nt_workflow import ModelSource

    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    first = workflow.initialize_fit(run)
    assert first.global_step == 0
    assert not first.optimizer.state
    assert first.scheduler.last_epoch == 0
    assert not first.scaler.is_enabled()
    assert (
        first.initialization_sha256
        == artifact(run, "execution.json")["initialization_sha256"]
    )
    original = {name: value.clone() for name, value in first.model.state_dict().items()}
    expected_random = torch.rand(4)
    with torch.no_grad():
        next(first.model.parameters()).add_(100)
    torch.manual_seed(999)

    second = workflow.initialize_fit(run)

    assert torch.equal(torch.rand(4), expected_random)
    assert all(
        torch.equal(original[name], value)
        for name, value in second.model.state_dict().items()
    )
    assert not second.optimizer.state
    assert second.scheduler.last_epoch == 0
    assert second.global_step == 0
    assert artifact(run, "status.json")["model_completed"] is False


def test_memory_retries_preserve_effective_batch_and_reset_initialization(
    tmp_path, tiny_nt, monkeypatch
):
    import torch

    from cas12a.nt_workflow import ModelSource

    workflow, run = prepared(tmp_path, counts=(64, 32, 2))
    allocate = torch.tensor

    def limited_allocator(values, *args, **kwargs):
        # Fault injection at the external tensor allocator, while model,
        # gradients, Adam, scheduler and serialization remain real.
        if isinstance(values, list) and len(values) > 4 and isinstance(values[0], list):
            raise torch.OutOfMemoryError("simulated tensor allocation limit")
        return allocate(values, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", limited_allocator)
    result = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))

    evidence = artifact(result, "preflight.json")
    attempts = evidence["attempts"]
    assert [a["settings"]["train_microbatch"] for a in attempts] == [
        32,
        16,
        8,
        4,
        4,
        4,
        4,
    ]
    assert [a["settings"]["evaluation_microbatch"] for a in attempts] == [
        32,
        32,
        32,
        32,
        16,
        8,
        4,
    ]
    assert len({a["initialization_sha256"] for a in attempts}) == 1
    assert all(
        a["settings"]["train_microbatch"] * a["settings"]["gradient_accumulation"] == 32
        for a in attempts
    )
    assert attempts[-1]["optimizer_updates"] == 2
    assert attempts[-1]["validation_examples"] == 4
    assert attempts[-1]["state"] == "passed"
    assert workflow.initialize_fit(result).optimizer.state == {}


@pytest.mark.parametrize(
    "damage, message",
    [
        ("unknown", "unknown token"),
        ("missing", "incomplete token reconstruction"),
        ("too-long", "exceeds 12 positions"),
    ],
)
def test_bad_tokenizers_fail_before_training_and_record_failure(
    tmp_path, tiny_nt, damage, message
):
    from transformers import EsmTokenizer

    from cas12a.nt_workflow import ModelSource

    if damage == "missing":
        tokenizer = EsmTokenizer.from_pretrained(tiny_nt)
        tokenizer.add_special_tokens({"additional_special_tokens": ["A"]})
        tokenizer.save_pretrained(tiny_nt)
    else:
        vocabulary = ["<cls>", "<pad>", "<mask>", "<unk>"]
        if damage == "too-long":
            vocabulary += list("ACGTN")
        (tiny_nt / "vocab.txt").write_text("\n".join(vocabulary))
    workflow, run = prepared(tmp_path)

    with pytest.raises(ValueError, match=message):
        workflow.preflight(run, model=ModelSource.fixture(tiny_nt))

    status = artifact(run, "status.json")
    assert status["state"] == "preflight_failed"
    assert status["model_completed"] is False
    assert artifact(run, "preflight.json")["attempts"] == []
    assert not (run.local_dir / "execution.json").exists()


def test_exhausted_memory_reports_resource_constraint_without_frozen_settings(
    tmp_path, tiny_nt, monkeypatch
):
    import torch

    from cas12a.nt_workflow import ModelSource

    allocate = torch.tensor

    def no_batch_memory(values, *args, **kwargs):
        if isinstance(values, list) and values and isinstance(values[0], list):
            raise torch.OutOfMemoryError("simulated exhausted device")
        return allocate(values, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", no_batch_memory)
    workflow, run = prepared(tmp_path)
    with pytest.raises(RuntimeError, match="microbatch 1 with gradient checkpointing"):
        workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    evidence = artifact(run, "preflight.json")
    assert [a["settings"]["train_microbatch"] for a in evidence["attempts"]] == [
        32,
        16,
        8,
        4,
        2,
        1,
        1,
    ]
    assert [a["settings"]["gradient_checkpointing"] for a in evidence["attempts"]] == [
        False
    ] * 6 + [True]
    assert artifact(run, "status.json")["state"] == "preflight_failed"
    assert not (run.local_dir / "execution.json").exists()


def test_nonfinite_model_cannot_pass_preflight(tmp_path, tiny_nt):
    import torch
    from transformers import EsmForMaskedLM, EsmTokenizer

    from cas12a.nt_workflow import ModelSource

    model = EsmForMaskedLM.from_pretrained(tiny_nt)
    with torch.no_grad():
        model.esm.embeddings.word_embeddings.weight.fill_(float("nan"))
    damaged = tmp_path / "nonfinite-nt"
    model.save_pretrained(damaged)
    EsmTokenizer.from_pretrained(tiny_nt).save_pretrained(damaged)
    workflow, run = prepared(tmp_path)
    with pytest.raises(FloatingPointError, match="nonfinite training"):
        workflow.preflight(run, model=ModelSource.fixture(damaged))
    assert artifact(run, "status.json")["state"] == "preflight_failed"
    assert artifact(run, "preflight.json")["attempts"][0]["state"] == "failed"


@pytest.mark.parametrize(
    "change", ["execution", "preflight", "tokenization", "model", "dependency"]
)
def test_fit_refuses_changes_after_execution_settings_are_frozen(
    tmp_path, tiny_nt, change, monkeypatch
):
    import importlib.metadata

    from cas12a.nt_workflow import ModelSource

    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    if change in ("execution", "preflight", "tokenization"):
        filename = change + ".json"
        document = artifact(run, filename)
        document["changed_after_preflight"] = True
        for directory in (run.local_dir, run.durable_dir):
            (directory / filename).write_text(json.dumps(document))
    elif change == "model":
        with (tiny_nt / "vocab.txt").open("a") as handle:
            handle.write("\n<added-after-preflight>")
    else:
        version = importlib.metadata.version
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: "changed" if name == "torch" else version(name),
        )
    with pytest.raises(ValueError, match="identity changed|configuration changed"):
        workflow.initialize_fit(run)


def test_production_cannot_use_fixture_weights_or_tokenizer(tmp_path, tiny_nt):
    from cas12a.nt_workflow import ModelSource

    workflow, run = prepared(tmp_path, production=True)
    with pytest.raises(ValueError, match="production cannot use a local fixture"):
        workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    assert artifact(run, "status.json")["state"] == "preflight_failed"
    assert artifact(run, "status.json")["run_kind"] == "production"
    assert artifact(run, "preflight.json")["attempts"] == []


def test_fit_requires_a_successful_preflight_and_passed_settings_cannot_be_reselected(
    tmp_path, tiny_nt
):
    from cas12a.nt_workflow import ModelSource

    workflow, run = prepared(tmp_path)
    with pytest.raises(ValueError, match="state/configuration"):
        workflow.initialize_fit(run)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    with pytest.raises(ValueError, match="state/configuration"):
        workflow.preflight(run, model=ModelSource.fixture(tiny_nt))


def test_tiny_fixture_gpu_preflight_when_cuda_is_available(tmp_path, tiny_nt):
    import torch

    from cas12a.nt_workflow import ModelSource

    if not torch.cuda.is_available():
        pytest.skip("actual CUDA preflight unexecuted: CPU-only runtime")
    workflow, run = prepared(tmp_path)
    result = workflow.preflight(run, model=ModelSource.fixture(tiny_nt), device="cuda")
    evidence = artifact(result, "preflight.json")
    assert evidence["gpu_preflight_executed"] is True
    assert evidence["evidence_kind"] == "gpu_fixture"
    assert evidence["attempts"][-1]["peak_allocated_bytes"] > 0
    assert evidence["attempts"][-1]["optimizer_updates"] == 2
    assert evidence["attempts"][-1]["post_adam_backward_finite"] is True
    state = workflow.initialize_fit(result)
    assert state.scaler.is_enabled() == (
        state.execution["settings"]["precision"] == "float16"
    )


def test_requested_cuda_unavailable_is_recorded_as_unexecuted(
    tmp_path, tiny_nt, monkeypatch
):
    import torch

    from cas12a.nt_workflow import ModelSource

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    workflow, run = prepared(tmp_path)
    with pytest.raises(RuntimeError, match="CUDA GPU unavailable"):
        workflow.preflight(run, model=ModelSource.fixture(tiny_nt), device="cuda")
    assert artifact(run, "status.json")["state"] == "preflight_failed"
    evidence = artifact(run, "preflight.json")
    assert evidence["gpu_preflight_executed"] is False
    assert evidence["evidence_kind"] == "unexecuted"
    assert evidence["attempts"] == []
    assert not (run.local_dir / "execution.json").exists()
