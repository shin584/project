"""Tests for the 9-Hydra v3 prediction export (#129).

Every test runs without the checkpoint, without torch and without a GPU: the
model is injected as `forward_fn`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import v3_predict
from v3_predict import HEADS

SPLIT_COLUMNS = [
    "row_index",
    "sequence",
    "protospacer",
    "in_pam_library",
    "trna_feature",
]


def make_sequence(tail: str = "AGGTCA") -> str:
    return "ACGT" + "ACGT" * 5 + tail


def make_split(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": list(range(100, 100 + n)),
            "sequence": [make_sequence() for _ in range(n)],
            "protospacer": [make_sequence()[4:24] for _ in range(n)],
            "in_pam_library": [False, True, False][:n],
            "trna_feature": [0, 1, 0][:n],
        },
        columns=SPLIT_COLUMNS,
    )


def write_checkpoint(tmp_path, mu=None, sd=None):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    mu = np.arange(len(HEADS), dtype=np.float32) / 100 if mu is None else mu
    sd = np.full(len(HEADS), 0.2, dtype=np.float32) if sd is None else sd
    np.save(checkpoint / "mu_sd.npy", np.vstack([mu, sd]).astype(np.float32))
    (checkpoint / "config.json").write_text('{"problem_type": "regression"}\n')
    return checkpoint


# --- head order ------------------------------------------------------------


def test_head_order_matches_the_training_notebook():
    # mu_sd.npy and the model's output columns follow this order; a reordering
    # would mislabel every prediction while still producing a valid-looking CSV.
    assert HEADS == (
        "SpCas9",
        "SpCas9-NG",
        "VRQR variant",
        "xCas",
        "Sniper-Cas9",
        "SpCas9-HF.1",
        "eSpCas9(1.1)",
        "HypaCas9",
        "evoCas9",
    )


def test_head_order_matches_the_published_split():
    xai_test_root = Path(v3_predict.__file__).resolve().parent.parent
    split_csv = xai_test_root / "SpCas9" / "published_split" / "test.csv"
    if not split_csv.exists():
        pytest.skip(f"{split_csv} is not available (it is git-ignored)")
    columns = pd.read_csv(split_csv, nrows=0).columns
    assert [head for head in columns if head in set(HEADS)] == list(HEADS)


# --- scale -----------------------------------------------------------------


def test_destandardise_inverts_the_standardisation():
    mu = np.linspace(0.05, 0.3, len(HEADS))
    sd = np.linspace(0.1, 0.25, len(HEADS))
    raw = np.array([mu, mu + sd, mu - 2 * sd])
    standardised = (raw - mu) / sd
    assert np.allclose(v3_predict.destandardise(standardised, mu, sd), raw)


def test_destandardise_reproduces_the_notebook_smoke_test():
    # The notebook printed SpCas9 0.497 for its synthetic sequence, against a
    # recorded mu of 0.2489 and sd of 0.2255 - i.e. ~1.10 standardised.
    mu = np.full(len(HEADS), 0.2489)
    sd = np.full(len(HEADS), 0.2255)
    standardised = np.full((1, len(HEADS)), (0.497 - 0.2489) / 0.2255)
    assert v3_predict.destandardise(standardised, mu, sd)[0, 0] == pytest.approx(0.497)


def test_destandardise_rejects_the_wrong_head_count():
    mu = sd = np.ones(len(HEADS))
    with pytest.raises(ValueError, match="standardised outputs"):
        v3_predict.destandardise(np.zeros((2, len(HEADS) - 1)), mu, sd)


def test_load_mu_sd_rejects_a_bad_shape(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    np.save(checkpoint / "mu_sd.npy", np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match=r"must be \(2, 9\)"):
        v3_predict.load_mu_sd(checkpoint)


def test_load_mu_sd_rejects_a_zero_standard_deviation(tmp_path):
    checkpoint = write_checkpoint(tmp_path, sd=np.zeros(len(HEADS), dtype=np.float32))
    with pytest.raises(ValueError, match="non-positive standard deviation"):
        v3_predict.load_mu_sd(checkpoint)


# --- sequence validation ---------------------------------------------------


def test_validate_sequences_rejects_a_length_other_than_30():
    with pytest.raises(ValueError, match="sequence length 29"):
        v3_predict.validate_sequences([make_sequence()[:29]])


def test_validate_sequences_rejects_a_non_acgt_base():
    # A length whose remainder is not zero, or an N, changes the tokenisation,
    # which the whole analysis assumes is exactly five 6-mers.
    with pytest.raises(ValueError, match="non-ACGT base"):
        v3_predict.validate_sequences(["N" + make_sequence()[1:]])


# --- frame -----------------------------------------------------------------


def test_build_prediction_frame_keeps_metadata_and_adds_one_column_per_head():
    split = make_split(3)
    predictions = np.arange(3 * len(HEADS), dtype=float).reshape(3, len(HEADS))
    frame = v3_predict.build_prediction_frame(split, predictions)

    assert list(frame.columns) == list(v3_predict.METADATA_COLUMNS) + [
        f"pred_{head}" for head in HEADS
    ]
    assert frame["row_index"].tolist() == [100, 101, 102]
    assert frame["pred_SpCas9"].tolist() == [0.0, 9.0, 18.0]
    assert frame["pred_evoCas9"].tolist() == [8.0, 17.0, 26.0]


def test_build_prediction_frame_survives_a_non_default_index():
    split = make_split(3)
    split.index = [7, 8, 9]
    predictions = np.arange(3 * len(HEADS), dtype=float).reshape(3, len(HEADS))
    frame = v3_predict.build_prediction_frame(split, predictions)
    assert frame["row_index"].tolist() == [100, 101, 102]
    assert frame["pred_SpCas9"].tolist() == [0.0, 9.0, 18.0]


def test_build_prediction_frame_rejects_a_row_count_mismatch():
    with pytest.raises(ValueError, match="expected \\(3, 9\\) predictions"):
        v3_predict.build_prediction_frame(make_split(3), np.zeros((2, len(HEADS))))


def test_build_prediction_frame_reports_missing_metadata():
    split = make_split(2).drop(columns=["in_pam_library"])
    with pytest.raises(ValueError, match="in_pam_library"):
        v3_predict.build_prediction_frame(split, np.zeros((2, len(HEADS))))


# --- export ----------------------------------------------------------------


def test_export_predictions_writes_the_csv_and_the_sidecar(tmp_path):
    split = make_split(3)
    split_csv = tmp_path / "test.csv"
    split.to_csv(split_csv, index=False)
    checkpoint = write_checkpoint(tmp_path)
    mu, _sd = v3_predict.load_mu_sd(checkpoint)

    calls = []

    def forward_fn(sequences):
        calls.append(list(sequences))
        return np.zeros((len(sequences), len(HEADS)))

    csv_path = v3_predict.export_predictions(
        split_csv, checkpoint, tmp_path / "out", forward_fn=forward_fn
    )

    assert calls == [split["sequence"].tolist()]
    frame = pd.read_csv(csv_path)
    assert len(frame) == 3
    # A zero standardised output must come back as the head's own mean.
    assert frame["pred_SpCas9"].tolist() == pytest.approx([mu[0]] * 3)
    assert frame["pred_evoCas9"].tolist() == pytest.approx([mu[8]] * 3)

    sidecar = json.loads((tmp_path / "out" / "predictions_checksum.json").read_text())
    assert sidecar["rows"] == 3
    assert sidecar["heads"] == list(HEADS)
    assert sidecar["prediction_scale"] == "fraction"
    assert sidecar["label_scale_in_split"] == "percent"
    assert sidecar["predictions_sha256"] == v3_predict.sha256_of(csv_path)
    assert sidecar["split"]["sha256"] == v3_predict.sha256_of(split_csv)
    assert set(sidecar["checkpoint"]["files"]) == {"config.json", "mu_sd.npy"}
    assert sidecar["checkpoint"]["weights_hashed"] is False


def test_export_predictions_refuses_a_bad_sequence(tmp_path):
    split = make_split(2)
    split.loc[0, "sequence"] = make_sequence()[:29]
    split_csv = tmp_path / "test.csv"
    split.to_csv(split_csv, index=False)
    checkpoint = write_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="sequence length 29"):
        v3_predict.export_predictions(
            split_csv,
            checkpoint,
            tmp_path / "out",
            forward_fn=lambda sequences: np.zeros((len(sequences), len(HEADS))),
        )
    assert not (tmp_path / "out" / "predictions.csv").exists()


def test_export_predictions_is_byte_reproducible(tmp_path):
    split_csv = tmp_path / "test.csv"
    make_split(3).to_csv(split_csv, index=False)
    checkpoint = write_checkpoint(tmp_path)
    rng = np.random.default_rng(0)
    fixed = rng.normal(size=(3, len(HEADS)))

    digests = []
    for name in ("one", "two"):
        path = v3_predict.export_predictions(
            split_csv,
            checkpoint,
            tmp_path / name,
            forward_fn=lambda sequences: fixed,
        )
        digests.append(v3_predict.sha256_of(path))
    assert digests[0] == digests[1]
