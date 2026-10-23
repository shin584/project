"""Shared loader for the Kim 2020 SpCas9 nine-variant dataset (issue #43).

Single source of truth for reading `dataset/kim_2020_table_s8.xlsx`, so that
the 9-Hydra v1 leakage audit and the 9-Hydra v2 build see exactly the same
rows, the same sequence windows and the same labels.

- `load_spcas9_table` reads the spreadsheet, coerces the nine head columns to
  numbers (the literal string `"n.a."` becomes NaN) and drops every row whose
  sequence is not 30bp. It also carries the **tRNA feature** column through as
  metadata - the published split stratifies on it, but it is not a model input
  (see issue #42's Out of Scope).
- Two filters over that frame: `filter_all_heads` (29,303 rows - the frame
  9-Hydra v1 was trained on) and `filter_any_head` (37,996 rows - the frame
  the v2 masked-loss build uses).
- `protospacers` / `pams` cut the SpCas9 target window: 20nt protospacer at
  0-indexed 4-23, 3nt PAM at 24-26 (see `CONTEXT.md`).
- `pam_library_protospacers` names the 30 protospacers of the embedded PAM
  library, a property of the dataset that both the v1 audit and the v2 split
  stratify by.
- `head_labels` returns labels **keyed by head name**, never by position. The
  spreadsheet orders `eSpCas9(1.1)` before `SpCas9-HF.1`; the saved model
  outputs them the other way round, so positional arrays would silently swap
  those two heads.
- `one_hot` and `output_name` are the two pieces of model plumbing every
  9-Hydra version shares: the `(n, 30, 4)` A/C/G/T encoding the input layer
  expects, and the `Output_<clean name>` layer name that keys a head's
  predictions, labels and sample weights. They live here so the v1 audit and
  the v2 training path cannot drift apart on either.
- `library_versions` reports installed versions through
  `importlib.metadata`, so a provenance record can be built without importing
  the frameworks themselves. Shared for the same reason as the plumbing: the
  v1 audit's report and the **9-Hydra v2 training log** should say versions
  the same way.

Read-only: this module opens the spreadsheet and writes nothing.
"""

import re
from collections.abc import Iterable, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "kim_2020_table_s8.xlsx"

# Every library whose version could change a SpCas9 number. A caller that
# needs fewer (the v2 training log) passes its own subset.
LIBRARIES = ("tensorflow", "keras", "numpy", "pandas", "scipy", "scikit-learn")

# Head order as the saved 9-Hydra checkpoint emits its outputs (the `targets`
# list in `models/Multi_Cas_1IN_9Hydra_Divide_Testset.ipynb`), which is *not*
# the spreadsheet's column order.
HEAD_NAMES = (
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

SEQUENCE_LENGTH = 30
PROTOSPACER_SLICE = slice(4, 24)
PAM_SLICE = slice(24, 27)

# The notebook trained on percentages divided by 100, so labels are scaled the
# same way here to stay comparable with the checkpoint's outputs.
LABEL_SCALE = 100.0

# One-hot layout the v1 notebook encoded with; any other base is all zeros.
ONE_HOT = {"A": 0, "C": 1, "G": 2, "T": 3}

SEQUENCE_COLUMN = "sequence"
USE_COLUMN = "use"
TRNA_COLUMN = "trna_feature"

# The spreadsheet's own vocabulary for the published split. These are the only
# two values `Use (Training or Test)` is defined to take.
TRAINING_USE_VALUE = "Training"
TEST_USE_VALUE = "Test"
USE_VALUES = (TRAINING_USE_VALUE, TEST_USE_VALUE)


def _find_column(columns: pd.Index, needle: str) -> str:
    matches = [c for c in columns if needle.lower() in str(c).lower()]
    if not matches:
        raise KeyError(f"no column containing {needle!r} in {list(columns)}")
    return matches[0]


def load_spcas9_table(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Read the spreadsheet into a frame with normalized columns.

    Returns `sequence`, `use`, `trna_feature` and the nine head columns
    (numeric, `"n.a."` coerced to NaN). Rows whose sequence is not 30bp are
    dropped; row order is otherwise the spreadsheet's, and the index is reset.
    """
    raw = pd.read_excel(path)
    return normalize_table(raw)


def normalize_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize an already-read spreadsheet frame. Seam for testing."""
    sequence_column = _find_column(raw.columns, "Input sequence")
    use_column = _find_column(raw.columns, "Use (Training or Test)")
    trna_column = _find_column(raw.columns, "tRNA feature")

    table = pd.DataFrame(
        {
            SEQUENCE_COLUMN: raw[sequence_column].astype(str).str.strip().str.upper(),
            USE_COLUMN: raw[use_column].astype(str).str.strip(),
            TRNA_COLUMN: pd.to_numeric(raw[trna_column], errors="coerce"),
        }
    )
    for head in HEAD_NAMES:
        table[head] = pd.to_numeric(raw[head], errors="coerce")

    is_30bp = table[SEQUENCE_COLUMN].str.len() == SEQUENCE_LENGTH
    return table[is_30bp].reset_index(drop=True)


def filter_all_heads(table: pd.DataFrame) -> pd.DataFrame:
    """Rows with a measured value for all nine heads (9-Hydra v1's frame)."""
    return table[table[list(HEAD_NAMES)].notna().all(axis=1)].reset_index(drop=True)


def filter_any_head(table: pd.DataFrame) -> pd.DataFrame:
    """Rows with at least one measured head (the 9-Hydra v2 build's frame)."""
    return table[table[list(HEAD_NAMES)].notna().any(axis=1)].reset_index(drop=True)


def protospacers(table: pd.DataFrame) -> pd.Series:
    """The 20nt protospacer at 0-indexed positions 4-23."""
    return table[SEQUENCE_COLUMN].str[PROTOSPACER_SLICE]


def pams(table: pd.DataFrame) -> pd.Series:
    """The 3nt PAM at 0-indexed positions 24-26."""
    return table[SEQUENCE_COLUMN].str[PAM_SLICE]


def pam_library_protospacers(sequences: Iterable[str]) -> set[str]:
    """Protospacers that appear with more than one distinct PAM.

    The dataset holds a PAM library: 30 protospacers repeated across up to 268
    rows each that vary only at positions 24-29. Every other repeated
    protospacer in the frame repeats as a byte-identical 30-mer, so "more than
    one PAM" separates the library cleanly without a frequency threshold.
    """
    frame = pd.DataFrame(
        {
            "protospacer": [s[PROTOSPACER_SLICE] for s in sequences],
            "pam": [s[PAM_SLICE] for s in sequences],
        }
    )
    distinct_pams = frame.groupby("protospacer")["pam"].nunique()
    return set(distinct_pams[distinct_pams > 1].index)


def head_labels(
    table: pd.DataFrame, scale: float = LABEL_SCALE
) -> dict[str, np.ndarray]:
    """Labels keyed by head name, scaled the way the checkpoint was trained."""
    return {head: table[head].to_numpy(dtype=float) / scale for head in HEAD_NAMES}


def clean_head_name(head: str) -> str:
    """`head` reduced to the characters a Keras layer name may hold."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", head)


def output_name(head: str) -> str:
    """The output layer name for `head`, as the v1 notebook built it.

    Every 9-Hydra version keys a head's predictions, labels and sample
    weights by this name rather than by position.
    """
    return "Output_" + clean_head_name(head)


def one_hot(sequences: Sequence[str]) -> np.ndarray:
    """`(n, 30, 4)` float32 one-hot encoding in A/C/G/T order."""
    encoded = np.zeros((len(sequences), len(sequences[0]) if sequences else 0, 4))
    for row, sequence in enumerate(sequences):
        for position, base in enumerate(sequence):
            channel = ONE_HOT.get(base)
            if channel is not None:
                encoded[row, position, channel] = 1.0
    return encoded.astype(np.float32)


def is_published_test(table: pd.DataFrame) -> pd.Series:
    """The paper's own held-out rows (`Use (Training or Test)` reads `Test`).

    Works on any frame carrying the `use` column, including the membership
    frame built from one.
    """
    return table[USE_COLUMN] == TEST_USE_VALUE


def library_versions(names: Sequence[str] = LIBRARIES) -> dict[str, str | None]:
    """Installed version of each of `names`, `None` where it is not installed.

    Read from the installed distribution metadata rather than from each
    library's `__version__`, so building a provenance record costs no
    framework import.
    """
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found
