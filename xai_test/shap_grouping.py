"""Token-grouped SHAP collapse for Model B (shared by the dashboard and the CLI).

Model B's embedding is NT's last hidden state (7 tokens x 1280) flattened
token-major (`model_b_wrapper.extract_nt_embeddings`), followed by the 4
physical features in `analysis_export.PHYSICAL_FEATURE_KEYS` order. This
module collapses one SHAP row over that layout into the 11 Token-grouped
SHAP groups (see CONTEXT.md).

Kept free of xgboost/torch so the dashboard can import it with its own
light requirements; `shap_analysis.py` holds the Tree SHAP computation.
"""

from dataclasses import dataclass

import numpy as np
from analysis_export import PHYSICAL_FEATURE_KEYS

N_TOKENS = 7
HIDDEN_DIM = 1280
TOKEN_NT_SPAN = 6
PHYSICAL_FEATURE_LABELS = {"mfe": "MFE", "dg": "ΔG", "tm": "Tm", "gc": "GC"}
SHAP_GROUP_COUNT = N_TOKENS + len(PHYSICAL_FEATURE_KEYS)
CLS_GROUP_NAME = "[CLS] (global sequence context)"


@dataclass(frozen=True)
class ShapGroup:
    """One Token-grouped SHAP bar: a signed contribution plus the input it summarizes."""

    name: str
    shap_value: float
    feature_value: str  # the 6-mer / physical value; "" for [CLS]

    @property
    def label(self) -> str:
        return f"{self.name}: {self.feature_value}" if self.feature_value else self.name


def token_grouped_shap(
    shap_row: np.ndarray, sequence: str, physical_features: dict[str, float]
) -> tuple[ShapGroup, ...]:
    """Collapse one 8,964-wide SHAP row into the 11 Token-grouped SHAP groups.

    Each token's 1,280 embedding dimensions are summed with sign, so the group
    keeps its push-up/push-down direction; physical features stay individual.
    """
    n_embedding = N_TOKENS * HIDDEN_DIM
    per_token = shap_row[:n_embedding].reshape(N_TOKENS, HIDDEN_DIM).sum(axis=1)
    groups = [ShapGroup(CLS_GROUP_NAME, float(per_token[0]), "")]
    for k in range(1, N_TOKENS):
        start = (k - 1) * TOKEN_NT_SPAN
        kmer = sequence[start : start + TOKEN_NT_SPAN]
        groups.append(
            ShapGroup(
                f"Token {k} (pos {start}-{start + TOKEN_NT_SPAN - 1})",
                float(per_token[k]),
                kmer,
            )
        )
    for j, key in enumerate(PHYSICAL_FEATURE_KEYS):
        groups.append(
            ShapGroup(
                PHYSICAL_FEATURE_LABELS[key],
                float(shap_row[n_embedding + j]),
                f"{physical_features[key]:.2f}",
            )
        )
    return tuple(groups)
