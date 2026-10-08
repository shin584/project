import numpy as np
import pytest
from analysis_export import PHYSICAL_FEATURE_KEYS
from shap_analysis import EMBEDDING_DIM, PHYSICAL_DIM
from shap_grouping import (
    CLS_GROUP_NAME,
    HIDDEN_DIM,
    N_TOKENS,
    SHAP_GROUP_COUNT,
    TOKEN_NT_SPAN,
    token_grouped_shap,
)

SEQUENCE = "ACGTAC" * 6
PHYSICAL = {"mfe": -3.21, "dg": -12.5, "tm": 61.0, "gc": 0.5}


def _shap_row(seed=0):
    return np.random.default_rng(seed).normal(size=EMBEDDING_DIM + PHYSICAL_DIM)


def test_token_grouped_shap_yields_eleven_groups_summing_to_the_row():
    row = _shap_row()

    groups = token_grouped_shap(row, SEQUENCE, PHYSICAL)

    assert len(groups) == SHAP_GROUP_COUNT == 11
    assert sum(g.shap_value for g in groups) == pytest.approx(row.sum())


def test_token_grouped_shap_is_cls_then_six_kmers_then_physical_features():
    row = _shap_row(1)

    groups = token_grouped_shap(row, SEQUENCE, PHYSICAL)

    per_token = row[:EMBEDDING_DIM].reshape(N_TOKENS, HIDDEN_DIM).sum(axis=1)
    assert [g.shap_value for g in groups[:N_TOKENS]] == pytest.approx(per_token)
    assert groups[0].name == CLS_GROUP_NAME and groups[0].feature_value == ""
    for k, g in enumerate(groups[1:N_TOKENS]):
        start = k * TOKEN_NT_SPAN
        assert g.feature_value == SEQUENCE[start : start + TOKEN_NT_SPAN]
    assert [g.shap_value for g in groups[N_TOKENS:]] == pytest.approx(
        row[-PHYSICAL_DIM:]
    )
    assert [g.feature_value for g in groups[N_TOKENS:]] == [
        f"{PHYSICAL[k]:.2f}" for k in PHYSICAL_FEATURE_KEYS
    ]


def test_grouping_layout_matches_the_shap_module():
    assert N_TOKENS * HIDDEN_DIM == EMBEDDING_DIM
    assert len(PHYSICAL_FEATURE_KEYS) == PHYSICAL_DIM
