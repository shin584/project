import ast
import json
import re

import numpy as np
import pandas as pd
import pytest
from ablation import XGB_HYPERPARAMS
from build_colab_grouped_model_b import (
    NOTEBOOK_MODULES,
    NOTEBOOK_PATH,
    check_embedded_modules,
)
from grouped_model_b import (
    EARLY_STOPPING_PATIENCE,
    FINETUNE_ARGS,
    MAX_LENGTH,
    NT_BASE_MODEL,
    PREDICTION_COLUMNS,
    build_test_predictions,
    build_training_log,
    finetune_compute_metrics,
    finetune_nt,
    join_grouped_splits,
    sequence_mean,
    summarize_finetune_history,
    train_heads,
    verify_grouped_testset,
    verify_no_sequence_overlap,
)
from grouped_split import BASE_DIR, DATASET_PATH, OUTPUT_DIR, load_full_dataset
from grouped_testset import CHECKSUM_FILENAME, TESTSET_FILENAME

NT_PRETRAIN_PATH = BASE_DIR / "nt_pretrain.py"
LOCAL_TOKENIZER_DIR = BASE_DIR / "NT_sacas9_fintuned_model"


def _toy_full_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence": ["AAA", "CCC", "GGG", "AAA", "TTT", "CCC"],
            "true_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "original_id": [10, 11, 12, 13, 14, 15],
        }
    )


def _toy_assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "original_id": [10, 11, 12, 13, 14, 15],
            "normalized_sequence": ["AAA", "CCC", "GGG", "AAA", "TTT", "CCC"],
            "cell_line_block": ["L1", "L1", "L1", "L2", "L2", "L2"],
            "split": ["train", "test", "train", "train", "val", "test"],
        }
    )


@pytest.fixture(scope="module")
def full_df() -> pd.DataFrame:
    return load_full_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def assignments() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "assignments.csv")


@pytest.fixture(scope="module")
def testset_metadata() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / TESTSET_FILENAME)


@pytest.fixture(scope="module")
def checksum_record() -> dict:
    with open(OUTPUT_DIR / CHECKSUM_FILENAME, encoding="utf-8") as f:
        return json.load(f)


def test_join_grouped_splits_routes_rows_by_original_id():
    splits = join_grouped_splits(_toy_full_df(), _toy_assignments())

    assert splits["train"]["original_id"].tolist() == [10, 12, 13]
    assert splits["val"]["original_id"].tolist() == [14]
    assert splits["test"]["original_id"].tolist() == [11, 15]
    assert splits["train"]["true_score"].tolist() == [0.1, 0.3, 0.4]
    assert splits["test"]["cell_line_block"].tolist() == ["L1", "L2"]


def test_join_grouped_splits_rejects_unassigned_rows():
    assignments = _toy_assignments().iloc[:-1]

    with pytest.raises(ValueError, match="original_id"):
        join_grouped_splits(_toy_full_df(), assignments)


def test_verify_no_sequence_overlap_rejects_shared_sequence():
    assignments = _toy_assignments()
    assignments.loc[4, ["normalized_sequence"]] = "AAA"
    full_df = _toy_full_df()
    full_df.loc[4, "sequence"] = "AAA"
    splits = join_grouped_splits(full_df, assignments)

    with pytest.raises(ValueError, match="train/val"):
        verify_no_sequence_overlap(splits)


def test_real_grouped_split_has_expected_sizes_and_no_overlap(full_df, assignments):
    splits = join_grouped_splits(full_df, assignments)

    verify_no_sequence_overlap(splits)
    assert {name: len(df) for name, df in splits.items()} == {
        "train": 4118,
        "val": 513,
        "test": 514,
    }


def test_verify_grouped_testset_accepts_committed_testset(
    full_df, assignments, testset_metadata, checksum_record
):
    test_df = join_grouped_splits(full_df, assignments)["test"]

    verify_grouped_testset(test_df, testset_metadata, checksum_record)


def test_verify_grouped_testset_rejects_tampered_scores(
    full_df, assignments, testset_metadata, checksum_record
):
    test_df = join_grouped_splits(full_df, assignments)["test"].copy()
    test_df.loc[test_df.index[0], "true_score"] += 1.0

    with pytest.raises(ValueError, match="test_metadata.csv"):
        verify_grouped_testset(test_df, testset_metadata, checksum_record)


def test_verify_grouped_testset_rejects_checksum_mismatch(
    full_df, assignments, testset_metadata, checksum_record
):
    test_df = join_grouped_splits(full_df, assignments)["test"]
    record = {**checksum_record, "dataset_sha256": "0" * 64}

    with pytest.raises(ValueError, match="checksum"):
        verify_grouped_testset(test_df, testset_metadata, record)


def _nt_pretrain_training_arguments() -> dict:
    source = NT_PRETRAIN_PATH.read_text(encoding="utf-8")
    call = re.search(r"TrainingArguments\((.*?)\n\)", source, re.DOTALL).group(0)
    node = ast.parse(call, mode="eval").body
    return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}


def test_finetune_args_match_nt_pretrain():
    reference = _nt_pretrain_training_arguments()
    # Paths are Colab-run-specific; everything else is the recipe.
    for path_arg in ("output_dir", "logging_dir"):
        reference.pop(path_arg)

    assert {k: v for k, v in FINETUNE_ARGS.items() if k != "seed"} == reference
    assert FINETUNE_ARGS["seed"] == 42


def test_nt_pretrain_model_and_tokenization_constants():
    source = NT_PRETRAIN_PATH.read_text(encoding="utf-8")

    assert f'model_name = "{NT_BASE_MODEL}"' in source
    assert f"max_length={MAX_LENGTH}" in source
    assert f"early_stopping_patience={EARLY_STOPPING_PATIENCE}" in source


def test_finetune_compute_metrics_reports_nt_pretrain_keys():
    labels = np.array([0.1, 0.5, 0.3, 0.9])
    predictions = np.array([[0.2], [0.6], [0.1], [0.8]])

    metrics = finetune_compute_metrics((predictions, labels))

    assert set(metrics) == {"spearman_rho", "pearson_r", "mae", "mse"}
    assert metrics["spearman_rho"] == pytest.approx(0.8)
    assert metrics["mae"] == pytest.approx(0.125)


def test_finetune_compute_metrics_handles_constant_predictions():
    metrics = finetune_compute_metrics((np.zeros((3, 1)), np.array([0.1, 0.2, 0.3])))

    assert metrics == {"spearman_rho": 0.0, "pearson_r": 0.0, "mae": 0.0, "mse": 0.0}


def test_summarize_finetune_history_picks_lowest_eval_loss_epoch():
    log_history = [
        {"loss": 0.5, "epoch": 0.5, "step": 10},
        {"eval_loss": 0.30, "eval_spearman_rho": 0.4, "epoch": 1.0, "step": 20},
        {"eval_loss": 0.20, "eval_spearman_rho": 0.6, "epoch": 2.0, "step": 40},
        {"eval_loss": 0.25, "eval_spearman_rho": 0.7, "epoch": 3.0, "step": 60},
        {"train_runtime": 12.0, "epoch": 3.0, "step": 60},
    ]

    summary = summarize_finetune_history(log_history)

    assert [e["epoch"] for e in summary["epochs"]] == [1.0, 2.0, 3.0]
    assert summary["epochs"][1]["eval_spearman_rho"] == 0.6
    assert summary["best_epoch"] == 2.0
    assert summary["best_eval_loss"] == 0.20


def test_sequence_mean_averages_labels_and_keeps_one_feature_row_per_sequence():
    X = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [3.0, 3.0]])
    y = np.array([0.2, 0.5, 0.4, 0.9])
    groups = pd.Series(["AAA", "CCC", "AAA", "GGG"])

    X_seq, y_seq = sequence_mean(X, y, groups)

    np.testing.assert_array_equal(X_seq, [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    np.testing.assert_allclose(y_seq, [0.3, 0.5, 0.9])


def test_sequence_mean_rejects_replicates_with_different_features():
    X = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError, match="features"):
        sequence_mean(X, np.array([0.1, 0.2]), pd.Series(["AAA", "AAA"]))


def test_train_heads_fits_all_rows_and_seq_mean_on_their_own_sample_counts():
    rng = np.random.default_rng(0)
    train_seqs = pd.Series([f"S{i}" for i in range(30)] * 3)
    val_seqs = pd.Series([f"V{i}" for i in range(10)] * 2)
    base_train = rng.normal(size=(30, 5))
    base_val = rng.normal(size=(10, 5))
    X_train = np.vstack([base_train] * 3)
    X_val = np.vstack([base_val] * 2)
    y_train = X_train[:, 0] + rng.normal(scale=0.1, size=len(X_train))
    y_val = X_val[:, 0] + rng.normal(scale=0.1, size=len(X_val))

    heads = train_heads(X_train, y_train, train_seqs, X_val, y_val, val_seqs)

    assert set(heads) == {"all_rows", "seq_mean"}
    assert heads["all_rows"].n_train_samples == 90
    assert heads["seq_mean"].n_train_samples == 30
    assert heads["seq_mean"].n_val_samples == 10
    for fit in heads.values():
        model = fit.model
        assert model.get_params()["max_depth"] == XGB_HYPERPARAMS["max_depth"]
        assert model.n_features_in_ == 5
        assert model.best_iteration >= 0

    log = build_training_log(
        [{"eval_loss": 0.2, "epoch": 1.0}], heads, seed=42, n_rows={"train": 90}
    )
    assert log["seed"] == 42
    assert log["finetune"]["best_epoch"] == 1.0
    assert set(log["xgboost"]) == {"all_rows", "seq_mean"}
    assert log["xgboost"]["seq_mean"]["n_train_samples"] == 30
    assert log["xgboost"]["all_rows"]["n_train_samples"] == 90
    assert log["recipe"]["finetune_args"] == FINETUNE_ARGS
    assert "xgboost" in log["library_versions"]
    json.dumps(log)


def test_build_test_predictions_has_documented_columns(testset_metadata):
    n = len(testset_metadata)

    table = build_test_predictions(testset_metadata, np.arange(n) / n, np.ones(n))

    assert table.columns.tolist() == PREDICTION_COLUMNS
    assert len(table) == 514
    assert table["sample_id"].tolist() == list(range(514))
    assert table["true_score"].tolist() == testset_metadata["true_score"].tolist()
    assert not table.isna().any().any()


def test_build_test_predictions_rejects_length_mismatch(testset_metadata):
    with pytest.raises(ValueError, match="514"):
        build_test_predictions(testset_metadata, np.zeros(3), np.zeros(3))


@pytest.mark.skipif(
    not (LOCAL_TOKENIZER_DIR / "vocab.txt").exists(), reason="needs NT tokenizer files"
)
def test_finetune_nt_runs_recipe_end_to_end_on_tiny_model(tmp_path):
    # Tiny random ESM with the NT tokenizer: exercises the Trainer wiring
    # (eval/save per epoch, early stopping, best-checkpoint reload, metrics)
    # on CPU, not the real 500M fine-tune.
    from transformers import AutoTokenizer, EsmConfig, EsmForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_DIR)
    config = EsmConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=MAX_LENGTH + 2,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=1,
    )
    tiny_dir = tmp_path / "tiny_base"
    EsmForSequenceClassification(config).save_pretrained(tiny_dir)
    tokenizer.save_pretrained(tiny_dir)

    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 36)) for _ in range(12)]
    train_df = pd.DataFrame({"sequence": seqs[:8], "true_score": rng.random(8)})
    val_df = pd.DataFrame({"sequence": seqs[8:], "true_score": rng.random(4)})

    trainer, tokenizer = finetune_nt(
        train_df,
        val_df,
        tmp_path / "checkpoints",
        model_name=str(tiny_dir),
        overrides={"fp16": False, "num_train_epochs": 2, "use_cpu": True},
    )

    summary = summarize_finetune_history(trainer.state.log_history)
    assert [e["epoch"] for e in summary["epochs"]] == [1.0, 2.0]
    assert "eval_spearman_rho" in summary["epochs"][0]
    assert tokenizer.model_max_length >= MAX_LENGTH


def test_notebook_embeds_modules_byte_identical_to_repo():
    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        notebook = json.load(f)

    embedded = check_embedded_modules(notebook)

    assert sorted(embedded) == sorted(NOTEBOOK_MODULES)
