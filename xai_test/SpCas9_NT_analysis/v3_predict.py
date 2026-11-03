"""Export 9-Hydra v3's per-head predictions over the 9-Hydra Testset (#129).

The XAI analysis selects its cases locally, from per-row error, so it needs one
predicted value per Testset row per head. v3's training notebook only prints a
metrics summary and a single-sequence smoke test, so this module produces the
missing artifact: a predictions CSV plus a checksum sidecar, in the shape
`SpCas9/spcas9_eval.py` writes for 9-Hydra v2.

Two scales are in play and mixing them is a 100x error:

- v3 was trained on the published percentage divided by 100, and its
  `predict` path de-standardises with `mu_sd.npy`, so **predictions are
  fractions** (0.497 means 49.7% efficiency).
- `SpCas9/published_split/*.csv` holds the **raw percentages** (49.6).

This module emits fractions, names the scale in every artifact, and never
converts the labels itself.

The model is reached only through `forward_fn`, a callable from a list of 30bp
strings to a `(n, 9)` array of standardised outputs. `torch` and `transformers`
are imported lazily inside `make_forward_fn`, so this module imports — and its
tests run — without them, with no checkpoint and no GPU.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

#: The nine heads in the order v3 was trained on. `mu_sd.npy` and the model's
#: output columns both follow it, so reordering this silently mislabels every
#: prediction.
HEADS: tuple[str, ...] = (
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

#: Carried through from the split so the analysis can stratify without
#: re-deriving anything.
METADATA_COLUMNS: tuple[str, ...] = (
    "row_index",
    "protospacer",
    "in_pam_library",
    "trna_feature",
)

SEQUENCE_LENGTH = 30
PREDICTION_SCALE = "fraction"
LABEL_SCALE_IN_SPLIT = "percent"

ForwardFn = Callable[[Sequence[str]], np.ndarray]


def load_mu_sd(checkpoint_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read `mu_sd.npy` as `(mu, sd)`, one value per head."""
    array = np.load(Path(checkpoint_dir) / "mu_sd.npy")
    if array.shape != (2, len(HEADS)):
        raise ValueError(f"mu_sd.npy must be (2, {len(HEADS)}), got {array.shape}")
    mu, sd = array[0].astype(np.float64), array[1].astype(np.float64)
    if not np.all(sd > 0):
        raise ValueError("mu_sd.npy holds a non-positive standard deviation")
    return mu, sd


def destandardise(
    standardised: np.ndarray, mu: np.ndarray, sd: np.ndarray
) -> np.ndarray:
    """Undo the per-head standardisation, giving fractions of 1."""
    standardised = np.asarray(standardised, dtype=np.float64)
    if standardised.ndim != 2 or standardised.shape[1] != len(HEADS):
        raise ValueError(
            f"expected (n, {len(HEADS)}) standardised outputs, got {standardised.shape}"
        )
    return standardised * sd + mu


def validate_sequences(sequences: Sequence[str]) -> None:
    """Reject anything the 6-mer tokenisation would not split into 5 tokens."""
    allowed = set("ACGT")
    for position, sequence in enumerate(sequences):
        if len(sequence) != SEQUENCE_LENGTH:
            raise ValueError(
                f"row {position}: sequence length {len(sequence)}, "
                f"expected {SEQUENCE_LENGTH}"
            )
        if not set(sequence) <= allowed:
            raise ValueError(f"row {position}: sequence holds a non-ACGT base")


def build_prediction_frame(
    split: pd.DataFrame, predictions: np.ndarray
) -> pd.DataFrame:
    """Join de-standardised predictions onto the split's metadata columns."""
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(split), len(HEADS)):
        raise ValueError(
            f"expected ({len(split)}, {len(HEADS)}) predictions, "
            f"got {predictions.shape}"
        )
    missing = [column for column in METADATA_COLUMNS if column not in split.columns]
    if missing:
        raise ValueError(f"the split is missing {missing}")

    frame = split.loc[:, list(METADATA_COLUMNS)].reset_index(drop=True).copy()
    for index, head in enumerate(HEADS):
        frame[f"pred_{head}"] = predictions[:, index]
    return frame


def make_forward_fn(
    checkpoint_dir: str | Path,
    *,
    device: str | None = None,
    batch_size: int = 64,
    max_length: int = 32,
) -> ForwardFn:
    """Build the real forward pass. Imports torch/transformers lazily."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint_dir = str(checkpoint_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = (
        AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        .eval()
        .to(device)
    )

    def forward_fn(sequences: Sequence[str]) -> np.ndarray:
        outputs = []
        for start in range(0, len(sequences), batch_size):
            batch = list(sequences[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs.append(model(**encoded).logits.float().cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, len(HEADS)))

    return forward_fn


def sha256_of(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for name in ("torch", "transformers"):
        try:
            versions[name] = __import__(name).__version__
        except (ImportError, AttributeError):
            # Recorded as null rather than fatal: the export runs with an
            # injected forward_fn and needs neither library present.
            versions[name] = None
    return versions


def export_predictions(
    split_csv: str | Path,
    checkpoint_dir: str | Path,
    out_dir: str | Path,
    *,
    forward_fn: ForwardFn | None = None,
    hash_weights: bool = False,
) -> Path:
    """Write `predictions.csv` and `predictions_checksum.json`; return the CSV."""
    split_csv, checkpoint_dir = Path(split_csv), Path(checkpoint_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(split_csv)
    sequences = split["sequence"].tolist()
    validate_sequences(sequences)

    if forward_fn is None:
        forward_fn = make_forward_fn(checkpoint_dir)
    mu, sd = load_mu_sd(checkpoint_dir)
    frame = build_prediction_frame(split, destandardise(forward_fn(sequences), mu, sd))

    csv_path = out_dir / "predictions.csv"
    frame.to_csv(csv_path, index=False, lineterminator="\n")

    checkpoint_files = {
        name: sha256_of(checkpoint_dir / name)
        for name in ("config.json", "mu_sd.npy")
        if (checkpoint_dir / name).exists()
    }
    weights = checkpoint_dir / "model.safetensors"
    if hash_weights and weights.exists():
        checkpoint_files["model.safetensors"] = sha256_of(weights)

    sidecar = {
        "model": "9-Hydra v3",
        "heads": list(HEADS),
        "rows": len(frame),
        "prediction_scale": PREDICTION_SCALE,
        "label_scale_in_split": LABEL_SCALE_IN_SPLIT,
        "scale_note": (
            "Predictions are fractions of 1; the split's label columns are raw "
            "percentages. Divide the labels by 100 before taking an error."
        ),
        "predictions_sha256": sha256_of(csv_path),
        "split": {"path": split_csv.as_posix(), "sha256": sha256_of(split_csv)},
        "checkpoint": {
            "path": checkpoint_dir.as_posix(),
            "files": checkpoint_files,
            "weights_hashed": bool(hash_weights and weights.exists()),
        },
        "versions": _library_versions(),
    }
    (out_dir / "predictions_checksum.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return csv_path
