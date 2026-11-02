"""Loading the Kim 2018 Cas12a dataset.

`read_cas12a_table` is the contract layer: it normalises sequences and targets
and keeps the source row index, so a row can be traced back to the spreadsheet.
`load_cas12a_dataset` is the array view the model trains on, and returns
exactly what `cas12a/model/Cas12a_model_final.ipynb` used to build inline.

pandas only - no TensorFlow - so `cas12a/test/` runs without it (#91).
"""

import numpy as np
import pandas as pd

from cas12a.features import (
    SEQUENCE_LENGTH,
    extract_seq_features,
    one_hot_encode,
)
from cas12a.units import SCORE_MAX, SCORE_MIN

#: Column headers as they appear in `kim_2018_cas12a_dataset.xlsx`, newline and
#: "freqeuncy" typo included.
SEQUENCE_COLUMN = (
    "34 bp synthetic target and target context sequence\n"
    "(4 bp + PAM + 23 bp protospacer + 3 bp)"
)
TARGET_COLUMN = "Indel freqeuncy\n(Background substracted, %)"

PAD_BASE = "N"


def _read_by_extension(filepath):
    """Read a table, dispatching on extension.

    The real dataset is `.xlsx` and the test fixture is `.csv`; the `.txt`,
    `.tsv` and `.xls` branches are the notebook's and are kept so that a path
    the notebook accepted does not start raising here.
    """
    name = str(filepath).lower()
    if name.endswith((".csv", ".txt")):
        return pd.read_csv(filepath)
    if name.endswith(".tsv"):
        return pd.read_csv(filepath, sep="\t")
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(filepath)
    raise ValueError(f"Unsupported file format: {filepath}")


def _normalise_sequence(value):
    """Force one cell to `SEQUENCE_LENGTH` bases, upper-cased.

    The real file is already 34 bp on every row, so truncation and padding only
    ever fire on malformed input; padding uses `N`, which `one_hot_encode`
    turns into all zeros.
    """
    sequence = str(value).strip().upper()
    if len(sequence) > SEQUENCE_LENGTH:
        return sequence[:SEQUENCE_LENGTH]
    return sequence.ljust(SEQUENCE_LENGTH, PAD_BASE)


def read_cas12a_table(filepath):
    """Return a `sequence` / `target` frame indexed by source row position.

    Rows with a missing sequence, or a target that is blank or unparseable, are
    dropped - the real file loses two rows this way. `target` is the indel
    frequency as a 0-1 ratio: background subtraction leaves some values
    negative, and those are clipped to 0 (see ADR 0001 and review §P5).
    """
    frame = _read_by_extension(filepath)
    targets = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    usable = frame[SEQUENCE_COLUMN].notna() & targets.notna()

    return pd.DataFrame(
        {
            "sequence": frame.loc[usable, SEQUENCE_COLUMN].map(_normalise_sequence),
            "target": (targets[usable] / 100.0).clip(SCORE_MIN, SCORE_MAX),
        }
    )


def load_cas12a_dataset(filepath):
    """Return `(X_seq, X_feat, y)` as float32 arrays, one row per usable row.

    `X_seq` is `(n, 34, 4)` one-hot, `X_feat` is `(n, 354)` GC + dinucleotide
    features, `y` is `(n,)` in `[0, 1]`.
    """
    table = read_cas12a_table(filepath)

    x_seq, x_feat = [], []
    for sequence in table["sequence"]:
        gc_feat, dinuc_feat = extract_seq_features(sequence)
        x_seq.append(one_hot_encode(sequence))
        x_feat.append(np.concatenate([gc_feat, dinuc_feat]))

    return (
        np.array(x_seq, dtype=np.float32),
        np.array(x_feat, dtype=np.float32),
        table["target"].to_numpy(dtype=np.float32),
    )
