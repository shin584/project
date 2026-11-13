"""The demo reader's defence line: the four misreads these artifacts invite.

The manifest detects a changed file; it cannot detect an intact file read
wrongly. These tests do, on synthetic data whose right answers are known.
"""

from __future__ import annotations

import numpy as np
import pytest

from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo.test import reader_fixtures as fx


@pytest.fixture
def data_dir(tmp_path):
    return fx.build_data_dir(tmp_path / "data")


@pytest.fixture
def spcas9(data_dir):
    return rd.SpCas9Reader(data_dir)


@pytest.fixture
def cas12a(data_dir):
    return rd.Cas12aReader(data_dir)


# --- (1) head is chosen by name, on each artifact's own axis ---------------- #


def test_head_is_resolved_by_name_on_each_artifacts_own_axis(spcas9):
    assert spcas9.ism_heads.width == 9 and spcas9.ig_heads.width == 3
    # The two axes order the analysed heads differently, so a position reused
    # from one axis on the other would land on another head.
    assert spcas9.ism_heads.index("evoCas9") == 8
    assert spcas9.ig_heads.index("evoCas9") == 0
    assert spcas9.ism_heads.index("SpCas9") == 0
    assert spcas9.ig_heads.index("SpCas9") == 1


def test_ism_profile_reads_the_named_heads_slice_not_the_first(spcas9):
    row = fx.ISM_ROWS[1]
    row_position = fx.ISM_ROWS.index(row)

    evo = spcas9.ism_profile(row, "evoCas9")
    primary = spcas9.ism_profile(row, "SpCas9")

    assert evo[3] == pytest.approx(fx.expected_profile(row_position, 3, 8), rel=1e-5)
    assert primary[3] == pytest.approx(
        fx.expected_profile(row_position, 3, 0), rel=1e-5
    )
    assert not np.allclose(evo, primary)


def test_spcas9_max_profile_reads_the_named_heads_slice_not_the_first(spcas9):
    """The case screen's bar: per position, the single most disruptive base.

    The fixture's three alternatives are 0.1/0.2/0.3 plus a per-head and
    per-row offset, so the max is the 0.3 one and it differs per head. A max
    taken over a wrongly selected head slice is caught by value here.
    """
    row = fx.ISM_ROWS[1]
    row_position = fx.ISM_ROWS.index(row)

    primary = spcas9.ism_max_profile(row, "SpCas9")
    evo = spcas9.ism_max_profile(row, "evoCas9")

    assert primary.shape == (fx.POSITIONS,)
    for position in (0, 7, fx.POSITIONS - 1):
        expected = 0.01 + 0.3 + row_position * 0.001 + position * 0.0001
        assert primary[position] == pytest.approx(expected, rel=1e-5)
    assert evo[3] == pytest.approx(
        0.09 + 0.3 + row_position * 0.001 + 3 * 0.0001, rel=1e-5
    )
    assert not np.allclose(primary, evo)


def test_spcas9_max_profile_keeps_the_mean_profile_beside_it(spcas9):
    """Both statistics stay: the bar is the max, the recorded stratum profiles
    the analyses check against are the mean."""
    row = fx.ISM_ROWS[1]
    row_position = fx.ISM_ROWS.index(row)

    assert spcas9.ism_profile(row, "SpCas9")[4] == pytest.approx(
        fx.expected_profile(row_position, 4, 0), rel=1e-5
    )
    assert (
        spcas9.ism_max_profile(row, "SpCas9")[4] > spcas9.ism_profile(row, "SpCas9")[4]
    )


def test_spcas9_max_profile_refuses_a_head_that_is_stored_but_not_analysed(spcas9):
    with pytest.raises(rd.UnknownHead):
        spcas9.ism_max_profile(fx.ISM_ROWS[0], "HypaCas9")


def test_spcas9_rollout_referenced_is_the_stored_blocks_column_mean(spcas9):
    """How much each token is attended to, joined by `row_index` not position.

    The fixture's matrices count upward row by row, so each row's column means
    are its own and reading another row's block is caught by value. There is
    no head argument: the stored matrix has no head axis (ADR 0008).
    """
    for position, row in enumerate(fx.ISM_ROWS):
        referenced = spcas9.rollout_referenced(row)
        assert referenced.shape == (5,)
        base = position * 25 + 10
        assert referenced == pytest.approx([base + token for token in range(5)])


def test_ig_tokens_read_the_named_heads_slice(spcas9):
    row = fx.IG_ROWS[0]

    evo = spcas9.ig_tokens(row, "evoCas9")
    primary = spcas9.ig_tokens(row, "SpCas9")

    assert evo[0] == pytest.approx(10.0, rel=1e-6)  # IG axis position 0
    assert primary[0] == pytest.approx(11.0, rel=1e-6)  # IG axis position 1


def test_a_stored_but_unanalysed_head_is_refused_with_both_lists(spcas9):
    with pytest.raises(rd.UnknownHead) as refusal:
        spcas9.ism_profile(fx.ISM_ROWS[0], "HypaCas9")

    assert "저장만" in str(refusal.value)
    assert "SpCas9-NG" in str(refusal.value)

    with pytest.raises(rd.UnknownHead):
        spcas9.ism_profile(fx.ISM_ROWS[0], "SaCas9")


def test_a_head_axis_that_disagrees_with_its_sidecar_is_refused(spcas9, data_dir):
    np.savez(
        data_dir / "spcas9/v3_ig/ig.npz",
        attribution=np.zeros((len(fx.IG_ROWS), 2, 5), dtype=np.float32),
        completeness_residual=np.zeros((len(fx.IG_ROWS), 2)),
        special_attribution=np.zeros((len(fx.IG_ROWS), 2)),
        row_index=np.array(fx.IG_ROWS, dtype=np.int64),
    )

    with pytest.raises(rd.ArtifactMismatch):
        spcas9.ig_tokens(fx.IG_ROWS[0], "SpCas9")


# --- (2) rows are joined by ID ---------------------------------------------- #


def test_rows_join_by_id_across_two_misaligned_populations(spcas9):
    shared = fx.IG_ROWS[0]  # position 1 in ISM, position 0 in IG
    assert spcas9.ism_rows.position(shared) == 1
    assert spcas9.ig_rows.position(shared) == 0

    # The same row's two results, each from its own axis position.
    assert spcas9.ig_tokens(shared, "SpCas9")[0] == pytest.approx(11.0, rel=1e-6)
    assert spcas9.ism_profile(shared, "SpCas9")[0] == pytest.approx(
        fx.expected_profile(1, 0, 0), rel=1e-5
    )


def test_wt_prediction_joins_by_id_and_catches_a_mis_join(spcas9):
    for position, row in enumerate(fx.ISM_ROWS):
        assert spcas9.wt_prediction(row, "SpCas9") == pytest.approx(
            0.5 + position * 0.1, rel=1e-5
        )


def test_a_row_outside_a_population_is_reported_as_absent_not_as_a_gap(spcas9):
    missing = fx.ISM_ROWS[0]
    assert spcas9.has_ig(missing) is False

    absence = spcas9.ig_absence(missing)
    assert absence is not None
    assert str(spcas9.ig_rows.rows) in absence.reason
    assert "IG 고정 명단" in absence.reason
    assert spcas9.ig_absence(fx.IG_ROWS[0]) is None

    with pytest.raises(rd.RowNotPresent) as refusal:
        spcas9.ig_tokens(missing, "SpCas9")
    assert str(missing) in str(refusal.value)


def test_cas12a_rows_join_by_source_row(cas12a):
    row = fx.CAS12A_ROWS[2]
    assert cas12a.ism_rows.position(row) == 2
    assert cas12a.prediction(row)["pam_stratum"] == "TTTV"
    with pytest.raises(rd.RowNotPresent):
        cas12a.prediction(99999)


# --- (3) percent vs fraction ------------------------------------------------ #


def test_labels_are_converted_from_percent_and_predictions_are_left_alone(spcas9):
    row = fx.ISM_ROWS[0]

    label = spcas9.label(row, "SpCas9")

    assert label == pytest.approx(fx.LABEL_PERCENT[row] / 100.0)
    assert label != pytest.approx(fx.LABEL_PERCENT[row])
    assert spcas9.prediction(row, "SpCas9") == pytest.approx(0.1)


def test_a_label_scale_the_reader_was_not_written_against_is_refused(tmp_path):
    data = fx.build_data_dir(tmp_path / "data", label_scale="fraction")

    with pytest.raises(rd.ArtifactMismatch) as refusal:
        rd.SpCas9Reader(data).label(fx.ISM_ROWS[0], "SpCas9")

    assert "label_scale_in_split" in str(refusal.value)


def test_a_cas12a_output_unit_that_is_not_fraction_is_refused(tmp_path):
    data = fx.build_data_dir(tmp_path / "data", output_unit="percent")

    with pytest.raises(rd.ArtifactMismatch):
        rd.Cas12aReader(data).prediction(fx.CAS12A_ROWS[0])


def test_an_unmeasured_label_stays_absent_instead_of_becoming_zero(spcas9):
    row, head = fx.UNMEASURED

    assert spcas9.label(row, head) is None
    assert spcas9.label(row, "SpCas9") == pytest.approx(fx.LABEL_PERCENT[row] / 100.0)


def test_split_role_is_read_per_row_not_assumed(spcas9):
    assert spcas9.split_role(fx.ISM_ROWS[0]) == "test"
    assert spcas9.split_role(30) == "val"
    assert spcas9.split_role(999999) is None


# --- (4) the WT slot is excluded from an ISM profile ------------------------ #


def test_ism_profile_averages_the_three_alternatives_not_four_slots():
    delta = np.array([[[0.0, 0.1, 0.2, 0.3]]])  # reference base is A (slot 0)
    reference = np.array([[0]])

    profile = rd.ism_position_profile(delta, reference)

    assert profile[0, 0] == pytest.approx(0.2)  # mean of 3, not 0.15 over 4


def test_ism_profile_excludes_the_wt_slot_at_whatever_position_it_sits(spcas9):
    row = fx.ISM_ROWS[0]
    profile = spcas9.ism_profile(row, "SpCas9")

    expected = [fx.expected_profile(0, position, 0) for position in range(fx.POSITIONS)]
    assert profile == pytest.approx(expected, rel=1e-5)


def test_a_nonzero_reference_slot_is_refused_as_a_mis_read_axis():
    delta = np.array([[[0.5, 0.1, 0.2, 0.3]]])
    reference = np.array([[0]])

    with pytest.raises(rd.ArtifactMismatch):
        rd.ism_position_profile(delta, reference)


def test_a_profile_asked_for_before_the_head_axis_is_reduced_is_refused():
    delta, reference = fx.ism_delta(1, 4, 9)

    with pytest.raises(rd.ArtifactMismatch) as refusal:
        rd.ism_position_profile(delta, reference)

    assert "head" in str(refusal.value)


def test_cas12a_profile_divides_by_the_valid_alternatives_only(cas12a, data_dir):
    row = fx.CAS12A_ROWS[0]
    arrays = np.load(data_dir / "cas12a/ism.npz")

    profile = cas12a.ism_profile(row)

    divisor = int(arrays["denominator"][0, 0])
    assert divisor == 2
    valid = arrays["non_wt_mask"][0, 0] & arrays["base_valid_mask"][0, 0]
    assert profile[0] == pytest.approx(np.abs(arrays["raw_delta"][0, 0])[valid].mean())
    assert profile[1] == pytest.approx(fx.expected_profile(0, 1, 0), rel=1e-5)


# --- the rest of the reader's contract -------------------------------------- #


def test_the_reader_loads_no_model_and_imports_no_analysis_package():
    """Read the import statements themselves, not the prose that explains them."""
    import ast
    import subprocess
    import sys
    from pathlib import Path

    package = Path(rd.__file__).parent
    imported: set[str] = set()
    for module in (
        package / "reader.py",
        package / "render.py",
        package / "shell.py",
        package / "cas12a_views.py",
    ):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])

    assert not imported & {"torch", "transformers", "jax", "tensorflow"}
    assert not any(
        name.endswith("_NT_analysis") or name == "SpCas9" for name in imported
    )
    assert "xai_test" not in imported  # no sibling analysis module reaches in

    # Transitively too, in a process of its own. Asserting on *this* process's
    # `sys.modules` only held while this subtree ran alone: inside the wider
    # suite another test has already imported torch, and the assertion failed
    # for a reason that had nothing to do with this reader.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import xai_test.nt_xai_demo.shell; "
                "print(sorted({'torch', 'transformers'} & set(sys.modules)))"
            ),
        ],
        cwd=Path(rd.__file__).parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "[]", probe.stdout


def test_token_aggregation_uses_the_artifacts_own_boundaries(spcas9):
    profile = spcas9.ism_profile(fx.IG_ROWS[0], "SpCas9")

    tokens = rd.aggregate_to_tokens(profile, spcas9.token_boundaries)

    assert tokens.shape == (5,)
    assert tokens[-1] == pytest.approx(np.mean(profile[24:30]), rel=1e-6)


def test_validation_figures_stay_within_one_layer(spcas9):
    figures = spcas9.validation_figures("generalisation", "SpCas9")

    assert figures["rows"] == 1 and figures["stratum"] == "generalisation"
    assert figures["statistic"] == "spearman_rank_correlation"
    assert "ism_token_profile" in figures and len(figures["ig_token_profile"]) == 5
    assert figures["mismatch_note"]
    with pytest.raises(KeyError):
        spcas9.validation_figures("generalisation+pam_library", "SpCas9")


def test_claim_limit_comes_from_the_artifacts(spcas9, cas12a):
    assert spcas9.claim_limit == fx.CLAIM_LIMIT
    assert fx.CAS12A_LIMIT_TEXT in cas12a.claim_limit
    assert fx.CAS12A_ROLLOUT_TEXT in cas12a.claim_limit


def test_disagreeing_claim_limits_are_refused_rather_than_one_being_picked(
    spcas9, data_dir
):
    import json

    path = data_dir / "spcas9/v3_report/head_order_reversal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["claim_limit"] = "다른 문구"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(rd.ArtifactMismatch):
        _ = spcas9.claim_limit


def test_cas12a_rollout_degeneracy_is_read_as_stored(cas12a):
    degeneracy = cas12a.rollout_degeneracy("all")

    assert degeneracy["near_identical_query_count"] == degeneracy["n_accepted"] == 3


def test_cases_keep_their_own_identifiers(spcas9, cas12a):
    assert [case["case_id"] for case in spcas9.cases] == [
        case_id for case_id, *_ in fx.SPCAS9_CASES
    ]
    assert [(case["source_row"], case["role"]) for case in cas12a.cases] == [
        (row, role) for row, role, *_ in fx.CAS12A_CASES
    ]


def test_metrics_are_per_pam_population_with_their_own_counts(cas12a):
    assert cas12a.metrics_for("all")["count"] == 3
    assert cas12a.metrics_for("TTTT", "clipped")["count"] == 1
    with pytest.raises(KeyError):
        cas12a.metrics_for("TTTN")
    with pytest.raises(KeyError):
        cas12a.metrics_for("all", "rounded")


def test_the_prediction_table_is_gated_by_its_own_head_axis(spcas9):
    """The split and prediction tables store nine heads; ISM's axis is not theirs."""
    assert spcas9.prediction_heads.width == 9
    assert spcas9.prediction_heads.analysed == spcas9.analysed_heads

    with pytest.raises(rd.UnknownHead):
        spcas9.label(fx.ISM_ROWS[0], "HypaCas9")
    with pytest.raises(rd.UnknownHead):
        spcas9.prediction(fx.ISM_ROWS[0], "HypaCas9")


def test_ig_values_carry_their_own_scale_and_are_not_fractions(spcas9):
    assert "표준화" in spcas9.ig_scale
    assert "fraction 아님" in spcas9.ig_scale
    assert spcas9.ig_completeness_residual(fx.IG_ROWS[0], "SpCas9") == pytest.approx(
        0.001
    )


def test_an_ig_archive_that_disagrees_with_its_declared_row_count_is_refused(
    spcas9, data_dir
):
    np.savez(
        data_dir / "spcas9/v3_ig/ig.npz",
        attribution=np.zeros((1, len(fx.IG_HEADS), 5), dtype=np.float32),
        completeness_residual=np.zeros((1, len(fx.IG_HEADS))),
        row_index=np.array(fx.IG_ROWS[:1], dtype=np.int64),
    )

    with pytest.raises(rd.ArtifactMismatch) as refusal:
        spcas9.ig_tokens(fx.IG_ROWS[0], "SpCas9")

    assert "IG 행 수" in str(refusal.value)


def test_a_stratum_outside_the_declared_ones_is_refused_everywhere(spcas9, cas12a):
    for call in (
        lambda: spcas9.require_stratum("nope"),
        lambda: spcas9.ig_subset_rows("nope"),
        lambda: cas12a.require_stratum("TTTN"),
        lambda: cas12a.position_profile("TTTN"),
        lambda: cas12a.rollout_degeneracy("TTTN"),
    ):
        with pytest.raises(KeyError):
            call()


def test_cas12a_declares_which_population_pools_the_pam_groups(cas12a):
    assert cas12a.pools_pam_groups("all") is True
    assert cas12a.pools_pam_groups("TTTV") is False
    assert cas12a.pools_pam_groups("TTTT") is False


def test_a_divisor_that_contradicts_the_stored_denominator_is_refused(cas12a, data_dir):
    arrays = dict(np.load(data_dir / "cas12a/ism.npz"))
    arrays["denominator"] = arrays["denominator"] + 1
    np.savez(data_dir / "cas12a/ism.npz", **arrays)

    with pytest.raises(rd.ArtifactMismatch) as refusal:
        cas12a.ism_profile(fx.CAS12A_ROWS[0])

    assert "denominator" in str(refusal.value)


def test_the_per_row_reads_views_need_are_each_keyed_by_id(spcas9, cas12a):
    row = fx.IG_ROWS[0]

    assert spcas9.ig_stratum(row) == "generalisation"
    assert spcas9.ig_stratum(fx.ISM_ROWS[0]) is None
    assert len(spcas9.reference_sequence(row)) == fx.POSITIONS
    assert set(spcas9.reference_sequence(row)) <= set(rd.BASES)
    assert spcas9.rollout(row).shape == (5, 5)
    assert spcas9.noise_floor_by_head("SpCas9")["median"] == pytest.approx(
        fx.NOISE_FLOOR_MEDIAN["SpCas9"]
    )
    with pytest.raises(rd.UnknownHead):
        spcas9.noise_floor_by_head("xCas")

    cas_row = fx.CAS12A_ROWS[1]
    assert cas12a.ig_signed_mass(cas_row).shape == (12,)
    assert cas12a.rollout(cas_row).shape == (9, 9)
    assert cas12a.accepted(cas_row) is True
    assert cas12a.head == rd.CAS12A_HEAD


# --- Cas12a token spans and PAM: read from layout.json, never re-derived ---- #


def test_cas12a_token_spans_and_pam_are_read_from_the_layout_file(cas12a):
    assert cas12a.token_spans == tuple(
        tuple(span) for span in fx.CAS12A_LAYOUT["dna_spans"]
    )
    assert cas12a.pam_span == (5, 9)


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"coordinates": "1-based closed"}, "coordinates"),
        (
            {"layout": {**fx.CAS12A_LAYOUT["layout"], "coordinate_system": "x"}},
            "coordinate_system",
        ),
        # Swapped: the spans would no longer follow IG's token columns in order.
        (
            {"dna_spans": [[1, 2], [0, 1], *fx.CAS12A_LAYOUT["dna_spans"][2:]]},
            "dna_spans",
        ),
        # A gap between tokens: a base no token covers.
        (
            {"dna_spans": [[0, 1], [2, 3], *fx.CAS12A_LAYOUT["dna_spans"][2:]]},
            "dna_spans",
        ),
        (
            {"layout": {**fx.CAS12A_LAYOUT["layout"], "pam": [30, 40]}},
            "pam",
        ),
    ],
)
def test_a_layout_the_reader_was_not_written_against_is_refused(tmp_path, change, why):
    data = tmp_path / "data"
    fx.build_cas12a(data, layout={**fx.CAS12A_LAYOUT, **change})

    with pytest.raises(rd.ArtifactMismatch, match=why):
        _ = rd.Cas12aReader(data).token_spans


# --- Cas12a per-row reads for the case screen (#171) ------------------------ #


@pytest.fixture
def shuffled(data_dir):
    """Each array file in its own row order, one position with no valid base."""
    cas12a_dir = data_dir / "cas12a"
    fx.reorder_cas12a_arrays(cas12a_dir, fx.CAS12A_SHUFFLED_ORDERS)
    fx.drop_alternatives(cas12a_dir, fx.CAS12A_ROWS[1], 10)
    return rd.Cas12aReader(data_dir)


def test_cas12a_max_profile_is_the_largest_alternative_magnitude(cas12a):
    """Per position, the most sensitive of the three substitutions, unsigned.

    The fixture's alternatives are 0.1/0.2/0.3 plus offsets with alternating
    sign, so the max is the 0.3 one in magnitude - a signed max would pick a
    negative value's neighbour on odd positions.
    """
    for index, row in enumerate(fx.CAS12A_ROWS):
        profile = cas12a.ism_max_profile(row)
        assert profile.shape == (fx.CAS12A_POSITIONS,)
        for position in (0, 1, 33):
            expected = 0.01 + 0.3 + index * 0.001 + position * 0.0001
            assert profile[position] == pytest.approx(expected, rel=1e-5)


def test_cas12a_max_profile_keeps_the_mean_profile_beside_it(cas12a):
    row = fx.CAS12A_ROWS[1]
    assert cas12a.ism_profile(row)[1] == pytest.approx(
        fx.expected_profile(1, 1, 0), rel=1e-5
    )
    assert cas12a.ism_max_profile(row)[1] > cas12a.ism_profile(row)[1]


def test_a_position_with_no_valid_alternative_is_missing_not_zero(shuffled):
    """The WT slot is a true zero; keeping it in the max would print 0 here."""
    row = fx.CAS12A_ROWS[1]
    profile = shuffled.ism_max_profile(row)
    assert np.isnan(profile[10])
    assert not np.isnan(profile[9]) and not np.isnan(profile[11])
    assert np.isnan(shuffled.ism_profile(row)[10])


def test_cas12a_per_row_reads_join_by_source_row_in_every_file(shuffled, data_dir):
    row = fx.CAS12A_ROWS[1]  # stored first, second and second-from-last
    original = rd.Cas12aReader(fx.build_cas12a(data_dir.parent / "plain").parent)

    np.testing.assert_allclose(
        shuffled.ism_max_profile(row)[:9], original.ism_max_profile(row)[:9]
    )
    np.testing.assert_allclose(shuffled.ig_dna_mass(row), original.ig_dna_mass(row))
    np.testing.assert_allclose(
        shuffled.rollout_referenced(row), original.rollout_referenced(row)
    )


def test_cas12a_ig_dna_mass_drops_the_three_special_tokens(cas12a, data_dir):
    """IG's 12 columns are CLS, the 9 DNA tokens, then two specials (nt_ig)."""
    row = fx.CAS12A_ROWS[1]
    stored = np.load(data_dir / "cas12a/ig.npz")["signed_mass"][1]
    np.testing.assert_array_equal(cas12a.ig_dna_mass(row), stored[1:10])
    assert len(cas12a.ig_dna_mass(row)) == len(cas12a.token_spans)


def test_cas12a_rollout_referenced_is_each_key_tokens_mean_over_queries(cas12a):
    """Rows of the DNA block are queries; a column is how much a token is attended."""
    row = fx.CAS12A_ROWS[1]
    matrix = cas12a.rollout(row)
    np.testing.assert_allclose(cas12a.rollout_referenced(row), matrix.mean(axis=0))
    assert cas12a.rollout_referenced(row)[0] == pytest.approx(81 + 36)


def test_cas12a_sequence_is_the_rows_own_and_matches_its_ism_reference(cas12a):
    row = fx.CAS12A_ROWS[2]
    reference = fx.reference_bases(len(fx.CAS12A_ROWS), fx.CAS12A_POSITIONS)[2]
    assert cas12a.sequence(row) == "".join(rd.BASES[b] for b in reference)


def test_a_sequence_that_disagrees_with_the_ism_reference_is_refused(data_dir):
    """predictions.csv and ism.npz meet only at source_row; their bases must agree."""
    path = data_dir / "cas12a/predictions.csv"
    text = path.read_text(encoding="utf-8").splitlines()
    fields = text[3].split(",")
    fields[1] = "A" * fx.CAS12A_POSITIONS
    text[3] = ",".join(fields)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")

    with pytest.raises(rd.ArtifactMismatch) as refusal:
        rd.Cas12aReader(data_dir).sequence(fx.CAS12A_ROWS[2])
    assert "reference" in str(refusal.value)
