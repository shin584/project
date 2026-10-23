import json

import numpy as np
import pandas as pd
import pytest
from test_spcas9_data import make_sequence, numbered_protospacer

from SpCas9.spcas9_compare import (
    ABSENT,
    ABSENT_ROW,
    CATEGORY_COLUMN,
    EXACT_DUPLICATE_COLUMN,
    MIN_N_FOR_COMPARISON,
    PAM_VARIANT_OVERLAP,
    STRATA,
    UNSEEN_TARGET,
    V1,
    V1_PREDICTIONS_PATH,
    V1_ROLE_COLUMN,
    V1_ROW_INDEX_COLUMN,
    V2,
    V2_PREDICTIONS_PATH,
    as_model_predictions,
    attach_v1_predictions,
    attach_v1_roles,
    build_basis,
    build_report,
    comparison_basis,
    composition,
    frame_row_map,
    label_categories,
    missing_v1_sequences,
    model_frame,
    model_pred_column,
    v1_predictions_by_sequence,
    write_basis,
)
from SpCas9.spcas9_data import HEAD_NAMES, load_spcas9_table
from SpCas9.spcas9_metrics import pred_column, true_column
from SpCas9.spcas9_split import PROTOSPACER_COLUMN, ROW_INDEX_COLUMN, file_sha256
from SpCas9.spcas9_v1_membership import ROLE_COLUMN, TRAIN, VAL

SEQUENCE_COLUMN = "sequence"

# Enough rows per stratum that the default minimum-n rule keeps both, and
# enough beyond it that dropping one head's coverage can push it under.
STRATUM_ROWS = MIN_N_FOR_COMPARISON + 20
THIN_HEAD = "SpCas9-NG"


def _table(head_patterns: list[tuple[str, ...]]) -> pd.DataFrame:
    """A normalized-table shape: one row per pattern of measured heads."""
    rows = []
    for i, measured in enumerate(head_patterns):
        row = {SEQUENCE_COLUMN: make_sequence(numbered_protospacer(i))}
        for head in HEAD_NAMES:
            row[head] = 1.0 if head in measured else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --- unit: bridging v1's frame and the split's frame ---


def test_frame_row_map_numbers_each_frame_in_its_own_order():
    """The bridge counts the any-head frame and the all-9-heads frame apart."""
    all_heads = tuple(HEAD_NAMES)
    some_heads = (HEAD_NAMES[0],)
    row_map = frame_row_map(
        _table([all_heads, some_heads, all_heads, (), all_heads, some_heads])
    )

    # The empty row is in neither frame, so it is absent from the map entirely.
    assert row_map[ROW_INDEX_COLUMN].tolist() == [0, 1, 2, 3, 4]
    assert row_map[V1_ROW_INDEX_COLUMN].tolist() == [0, ABSENT_ROW, 1, 2, ABSENT_ROW]


def _v1_frame(roles: dict[str, str], preds: float = 0.5) -> pd.DataFrame:
    """A v1 audit predictions frame: one row per sequence, with its role."""
    rows = []
    for i, (sequence, role) in enumerate(roles.items()):
        row = {
            ROW_INDEX_COLUMN: i,
            SEQUENCE_COLUMN: sequence,
            PROTOSPACER_COLUMN: sequence[4:24],
            ROLE_COLUMN: role,
        }
        for head in HEAD_NAMES:
            row[true_column(head)] = 0.1
            row[pred_column(head)] = preds
        rows.append(row)
    return pd.DataFrame(rows)


def test_attach_v1_roles_calls_rows_outside_v1s_frame_absent():
    """A test row v1 never held is `absent`, not quietly folded into a role."""
    sequences = [make_sequence(numbered_protospacer(i)) for i in range(3)]
    row_map = pd.DataFrame(
        {ROW_INDEX_COLUMN: [0, 1, 2], V1_ROW_INDEX_COLUMN: [0, ABSENT_ROW, 1]}
    )
    v1 = _v1_frame({sequences[0]: TRAIN, sequences[2]: "test"})
    test_frame = pd.DataFrame({ROW_INDEX_COLUMN: [0, 1, 2], SEQUENCE_COLUMN: sequences})

    attached = attach_v1_roles(test_frame, row_map, v1)

    assert attached[V1_ROLE_COLUMN].tolist() == [TRAIN, ABSENT, "test"]


def test_comparison_basis_drops_only_the_rows_v1_fitted():
    """Train and val go; v1's own test role and the absent rows stay."""
    frame = pd.DataFrame({V1_ROLE_COLUMN: [TRAIN, VAL, "test", ABSENT]})

    assert comparison_basis(frame)[V1_ROLE_COLUMN].tolist() == ["test", ABSENT]


# --- unit: the two strata and what they are made of ---


def test_label_categories_judges_against_every_row_v1_fitted():
    """Unseen means the protospacer is in no train *or* val row."""
    seen_in_train = make_sequence(numbered_protospacer(0))
    seen_in_val = make_sequence(numbered_protospacer(1))
    # Same protospacer as the train row, different PAM: a real PAM variant.
    pam_variant = make_sequence(numbered_protospacer(0), pam="AGG")
    unseen = make_sequence(numbered_protospacer(2))
    fitted = _v1_frame({seen_in_train: TRAIN, seen_in_val: VAL})
    basis = pd.DataFrame(
        {
            SEQUENCE_COLUMN: [seen_in_train, seen_in_val, pam_variant, unseen],
            PROTOSPACER_COLUMN: [
                seen_in_train[4:24],
                seen_in_val[4:24],
                pam_variant[4:24],
                unseen[4:24],
            ],
        }
    )

    labelled = label_categories(basis, fitted)

    assert labelled[CATEGORY_COLUMN].tolist() == [
        PAM_VARIANT_OVERLAP,
        PAM_VARIANT_OVERLAP,
        PAM_VARIANT_OVERLAP,
        UNSEEN_TARGET,
    ]
    # The two 30-mers v1 fitted outright are flagged, the PAM variant is not.
    assert labelled[EXACT_DUPLICATE_COLUMN].tolist() == [True, True, False, False]


# --- unit: reaching v1's predictions without re-running the audit ---


def test_v1_predictions_are_looked_up_by_sequence():
    """v1 is deterministic per 30-mer, so a repeat reuses the audit's row."""
    fitted = make_sequence(numbered_protospacer(0))
    unscored = make_sequence(numbered_protospacer(1))
    v1 = _v1_frame({fitted: TRAIN}, preds=0.25)
    basis = pd.DataFrame({SEQUENCE_COLUMN: [fitted, unscored]})

    assert missing_v1_sequences(basis, v1_predictions_by_sequence([v1])) == [unscored]

    paired = attach_v1_predictions(basis, v1_predictions_by_sequence([v1]))
    column = model_pred_column(V1, "SpCas9")
    assert paired[column].tolist()[0] == pytest.approx(0.25)
    assert np.isnan(paired[column].tolist()[1])


def test_as_model_predictions_tags_a_scoring_steps_output_with_its_model():
    """A scoring step writes `<head>_pred`; the comparison holds two models."""
    scored = pd.DataFrame({pred_column("SpCas9"): [0.3]})

    tagged = as_model_predictions(scored, V2)

    assert tagged.columns.tolist() == [model_pred_column(V2, "SpCas9")]
    assert model_frame(tagged, V2)[pred_column("SpCas9")].tolist() == [0.3]


def test_model_frame_reads_each_models_predictions_under_one_name():
    """Both models are scored by the shared metric helpers, unchanged."""
    paired = pd.DataFrame(
        {
            true_column("SpCas9"): [0.1],
            model_pred_column(V1, "SpCas9"): [0.2],
            model_pred_column(V2, "SpCas9"): [0.3],
        }
    )

    assert model_frame(paired, V1)[pred_column("SpCas9")].tolist() == [0.2]
    assert model_frame(paired, V2)[pred_column("SpCas9")].tolist() == [0.3]


def _paired_frame(
    rows_per_stratum: int = STRATUM_ROWS, thin_head_rows: int = 0
) -> pd.DataFrame:
    """A paired basis frame where v2 beats v1 by a known margin.

    Labels rise with the row number and each model's prediction is the label
    plus a constant error, so every group's Spearman is 1.0 and its MAE is
    that constant. v1's error is twice v2's, so every delta has a known sign.
    `thin_head_rows` is how many rows `THIN_HEAD` was measured on, to exercise
    the minimum-n rule.
    """
    rows = []
    for stratum in STRATA:
        for i in range(rows_per_stratum):
            label = 0.001 * i
            row = {
                ROW_INDEX_COLUMN: len(rows),
                SEQUENCE_COLUMN: make_sequence(numbered_protospacer(len(rows))),
                PROTOSPACER_COLUMN: numbered_protospacer(len(rows)),
                CATEGORY_COLUMN: stratum,
                EXACT_DUPLICATE_COLUMN: stratum == PAM_VARIANT_OVERLAP and i % 2 == 0,
                V1_ROLE_COLUMN: ABSENT,
            }
            for head in HEAD_NAMES:
                measured = head != THIN_HEAD or i < thin_head_rows
                row[true_column(head)] = label if measured else np.nan
                row[model_pred_column(V1, head)] = label + 0.02
                row[model_pred_column(V2, head)] = label + 0.01
            rows.append(row)
    return pd.DataFrame(rows)


# --- the report builder ---


@pytest.fixture(scope="module")
def report() -> dict:
    return build_report(_paired_frame(), n_resamples=200)


def test_report_covers_both_strata_and_every_head(report):
    assert sorted(report["categories"]) == sorted(STRATA)
    for stratum in STRATA:
        assert sorted(report["categories"][stratum]["heads"]) == sorted(HEAD_NAMES)
        assert report["categories"][stratum]["rows"] == STRATUM_ROWS


def test_report_emits_no_pooled_figure(report):
    """Pooling would call the partially advantaged rows a fair comparison."""
    serialized = json.dumps(report)
    assert "pooled" not in json.loads(serialized)
    for forbidden in ("overall", "all_rows", "combined"):
        assert forbidden not in serialized
    assert report["pooling"]


def test_delta_is_v2_minus_v1_on_every_metric(report):
    """v2's error is half v1's here, so MAE falls and the correlations tie."""
    head = report["categories"][UNSEEN_TARGET]["heads"]["SpCas9"]

    assert head["n"] == STRATUM_ROWS
    assert head["v1"]["mae"] == pytest.approx(0.02)
    assert head["v2"]["mae"] == pytest.approx(0.01)
    assert head["delta"]["mae"] == pytest.approx(-0.01)
    assert head["delta"]["spearman"] == pytest.approx(0.0)


def test_only_the_spcas9_head_carries_a_paired_bootstrap_ci(report):
    for stratum in STRATA:
        heads = report["categories"][stratum]["heads"]
        for key in ("delta_ci95", "delta_ci95_by_sequence"):
            assert set(heads["SpCas9"][key]) == {"spearman", "mae"}
            lower, upper = heads["SpCas9"][key]["mae"]
            assert lower <= heads["SpCas9"]["delta"]["mae"] <= upper
        for head in HEAD_NAMES:
            if head != "SpCas9":
                assert "delta_ci95" not in heads[head]


def test_a_head_under_the_minimum_is_excluded_with_its_reason():
    """Thin coverage is named, not reported as a comparison on a few rows."""
    thin = MIN_N_FOR_COMPARISON - 1
    report = build_report(_paired_frame(thin_head_rows=thin), n_resamples=50)

    for stratum in STRATA:
        head = report["categories"][stratum]["heads"][THIN_HEAD]
        assert head["excluded"] is True
        assert head["n"] == thin
        assert str(thin) in head["reason"]
        assert str(MIN_N_FOR_COMPARISON) in head["reason"]
        assert "delta" not in head
    assert report["excluded"][UNSEEN_TARGET] == [THIN_HEAD]


def test_pam_variant_stratum_reports_how_much_of_it_is_memorized(report):
    """1,134 rows are not 1,134 PAM variants; the exact repeats are counted."""
    composition = report["categories"][PAM_VARIANT_OVERLAP]["composition"]

    assert composition["exact_duplicate"] + composition["pam_variant"] == STRATUM_ROWS
    assert composition["exact_duplicate"] == STRATUM_ROWS // 2 + STRATUM_ROWS % 2
    assert report["categories"][UNSEEN_TARGET]["composition"]["exact_duplicate"] == 0


# --- the basis file and its checksum sidecar ---


def test_write_basis_pins_the_frame_and_both_sources(tmp_path):
    frame = _paired_frame(rows_per_stratum=4)
    v1_source = tmp_path / "v1.csv"
    v2_source = tmp_path / "v2.csv"
    v1_source.write_text("v1", encoding="utf-8")
    v2_source.write_text("v2", encoding="utf-8")

    checkpoint = tmp_path / "v1.keras"
    checkpoint.write_text("weights", encoding="utf-8")

    record = write_basis(
        frame,
        v1_source,
        v2_source,
        rescored=3,
        checkpoint_path=checkpoint,
        output_dir=tmp_path,
    )

    assert record["rows"] == len(frame)
    assert record["v1_predictions"]["rescored"] == 3
    assert record["v1_predictions"]["from_audit"] == len(frame) - 3
    # The rows the audit never held went through this checkpoint, so it is pinned.
    assert record["v1_predictions"]["checkpoint_sha256"] == file_sha256(checkpoint)
    assert record["sources"]["v1"]["file"] == "v1.csv"
    assert len(record["file_sha256"]) == 64
    written = pd.read_csv(tmp_path / record["file"])
    assert len(written) == len(frame)
    assert json.loads((tmp_path / "basis_checksum.json").read_text())["rows"] == len(
        frame
    )


# --- pinned regression over the real basis (no model loaded) ---

# What #42 pins the comparison basis at, and what it is made of. Rebuilt here
# from the spreadsheet and the two scoring steps' own prediction files, so a
# change to any of the three that moves these counts fails loudly.
PINNED_BASIS_ROWS = 1740
PINNED_CATEGORY_ROWS = {UNSEEN_TARGET: 606, PAM_VARIANT_OVERLAP: 1134}
PINNED_PAM_VARIANT_COMPOSITION = {"exact_duplicate": 990, "pam_variant": 144}
# The strata hold fewer distinct 30-mers than rows, which is why a row count
# and a target count are reported side by side.
PINNED_DISTINCT_SEQUENCES = {UNSEEN_TARGET: 387, PAM_VARIANT_OVERLAP: 784}


@pytest.fixture(scope="module")
def real_basis() -> pd.DataFrame:
    return build_basis(
        load_spcas9_table(),
        pd.read_csv(V1_PREDICTIONS_PATH),
        pd.read_csv(V2_PREDICTIONS_PATH),
    )


def test_the_real_basis_is_the_rows_v1_neither_trained_nor_validated_on(real_basis):
    assert len(real_basis) == PINNED_BASIS_ROWS
    assert real_basis[CATEGORY_COLUMN].value_counts().to_dict() == PINNED_CATEGORY_ROWS
    assert not real_basis[V1_ROLE_COLUMN].isin([TRAIN, VAL]).any()


def test_the_real_pam_variant_stratum_is_mostly_memorized(real_basis):
    """87% of the stratum repeats a 30-mer v1 fitted - not a new PAM."""
    stratum = real_basis[real_basis[CATEGORY_COLUMN] == PAM_VARIANT_OVERLAP]

    assert composition(stratum) == PINNED_PAM_VARIANT_COMPOSITION
    unseen = real_basis[real_basis[CATEGORY_COLUMN] == UNSEEN_TARGET]
    assert composition(unseen)["exact_duplicate"] == 0


def test_the_real_strata_hold_fewer_targets_than_rows(real_basis):
    distinct = {
        category: int(
            real_basis.loc[
                real_basis[CATEGORY_COLUMN] == category, SEQUENCE_COLUMN
            ].nunique()
        )
        for category in STRATA
    }

    assert distinct == PINNED_DISTINCT_SEQUENCES
    for category in STRATA:
        assert distinct[category] < PINNED_CATEGORY_ROWS[category]
