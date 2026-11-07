import numpy as np
import pytest
from scipy.stats import spearmanr

from xai_test.cas12a_NT_analysis.nt_inputs import DNA_SPANS

SEQUENCE = "ACGTTTTA" + "A" * 23 + "CGA"


def deltas(profile):
    values = np.empty((34, 4), dtype=np.float64)
    for value, (start, end) in zip(profile, DNA_SPANS, strict=True):
        values[start:end] = value
    for position, base in enumerate(SEQUENCE):
        values[position, "ACGT".index(base)] = 0
    return values


def test_primary_equal_length_ranks_and_nine_token_density_use_midrank_ties():
    from xai_test.cas12a_NT_analysis.nt_agreement import sequence_agreement

    # Worked example: primary rho=-2/3; adding four low 1bp tokens gives 13/18.
    density = [1, 1, 3, 4, 4, 0, 0, 0, 0]
    sensitivity = [2, 2, 4, 1, 1, 0, 0, 0, 0]
    mass = [-6, 6, 18, -24, 24, 0, 0, 0, 0]
    result = sequence_agreement(SEQUENCE, mass, deltas(sensitivity))
    primary = result["versions"]["all_nonWT"]["correlations"]["five_6mer"]["density"]
    nine = result["versions"]["all_nonWT"]["correlations"]["nine_token"]["density"]
    assert primary == {
        "rho": pytest.approx(-2 / 3),
        "reason": None,
        "n_total": 5,
        "n_valid": 5,
        "token_ids": [1, 2, 3, 4, 5],
    }
    assert nine["rho"] == pytest.approx(13 / 18)
    assert nine["rho"] == pytest.approx(spearmanr(density, sensitivity).statistic)
    assert result["versions"]["all_nonWT"]["tokens"][0]["ig_signed_mass"] == -6


def test_pam_exclusion_preserves_alternative_denominators_and_full_ig_support():
    from xai_test.cas12a_NT_analysis.nt_agreement import sequence_agreement

    values = deltas([2] * 9)
    values[4:7][values[4:7] != 0] = 50
    result = sequence_agreement(SEQUENCE, [6] * 5 + [1] * 4, values)
    included = result["versions"]["all_nonWT"]
    excluded = result["versions"]["exclude_pam_breaking"]
    assert included["tokens"][0]["ism_mean"] == 18
    assert included["tokens"][1]["ism_mean"] == 10
    for position in (4, 5, 6):
        assert excluded["bases"][position] == {
            "position": position,
            "alternative_mask": [False] * 4,
            "n_alternatives": 0,
            "mean_abs_delta": None,
            "reason": "no eligible alternatives",
        }
    assert excluded["tokens"][0]["valid_base_positions"] == [0, 1, 2, 3]
    assert excluded["tokens"][1]["valid_base_positions"] == [7, 8, 9, 10, 11]
    assert [t["n_valid_bases"] for t in excluded["tokens"]] == [
        4,
        5,
        6,
        6,
        6,
        1,
        1,
        1,
        1,
    ]
    assert excluded["tokens"][0]["ig_density"] == 1
    assert excluded["tokens"][0]["ism_sum"] == 8
    assert excluded["tokens"][0]["support_mismatch"] is True


def test_ism_averages_alternatives_then_valid_bases_with_distinct_denominators():
    from xai_test.cas12a_NT_analysis.nt_agreement import sequence_agreement

    values = deltas([3] * 9)
    values[0, 1:] = 9
    valid = np.ones((34, 4), dtype=bool)
    valid[0] = [False, True, False, False]
    result = sequence_agreement(SEQUENCE, range(9), values, base_valid_mask=valid)
    included = result["versions"]["all_nonWT"]
    assert included["bases"][0]["n_alternatives"] == 1
    assert included["tokens"][0]["ism_mean"] == 4
    assert included["tokens"][0]["ism_sum"] == 24
    assert result["versions"]["exclude_pam_breaking"]["tokens"][0]["ism_mean"] == 4.5
    assert result["base_valid_mask"][0] == [False, True, False, False]


def test_global_covariance_detects_group_only_agreement_without_within_group_signal():
    from xai_test.cas12a_NT_analysis.nt_agreement import sequence_agreement

    result = sequence_agreement(SEQUENCE, [6] * 5 + [0] * 4, deltas([1] * 5 + [0] * 4))
    for version in result["versions"].values():
        groups = version["groups"]["length"]
        assert groups["rho"] == pytest.approx(1)
        assert groups["within_contribution"] == 0
        assert groups["between_contribution"] == pytest.approx(1)
        assert groups["rank_mean_difference"] == {"ig": 4.5, "ism": 4.5}
        assert groups["groups"][0]["token_ids"] == [1, 2, 3, 4, 5]
        assert groups["groups"][0]["mean_ig_rank"] == 7
        assert groups["groups"][1]["mean_ig_rank"] == 2.5
        assert groups["sum_check_pass"] is True
        assert version["correlations"]["five_6mer"]["density"]["rho"] is None
        assert version["groups"]["pure_flank"]["groups"][0]["token_ids"] == [7, 8, 9]


@pytest.mark.parametrize("seed", range(4))
def test_both_global_rank_partitions_match_independent_covariance(seed):
    from xai_test.cas12a_NT_analysis.nt_agreement import sequence_agreement

    rng = np.random.default_rng(seed)
    density = rng.integers(0, 5, 9).astype(float)
    sensitivity = rng.integers(0, 5, 9).astype(float)
    mass = density * np.array([6] * 5 + [1] * 4)
    result = sequence_agreement(SEQUENCE, mass, deltas(sensitivity))
    from scipy.stats import rankdata

    x, y = rankdata(density), rankdata(sensitivity)
    for version in result["versions"].values():
        for name, first in (("length", np.arange(5)), ("pure_flank", np.arange(6, 9))):
            # E[XY] = E[Cov(X,Y|g)] + E[E(X|g)E(Y|g)].
            partitions = (first, np.setdiff1d(np.arange(9), first))
            conditional_product = (
                sum(len(g) * x[g].mean() * y[g].mean() for g in partitions) / 9
            )
            between = conditional_product - x.mean() * y.mean()
            total = np.cov(x, y, ddof=0)[0, 1]
            diagnostic = version["groups"][name]
            assert diagnostic["rho"] == pytest.approx(
                spearmanr(density, sensitivity).statistic
            )
            assert diagnostic["between_covariance"] == pytest.approx(between)
            assert diagnostic["within_covariance"] == pytest.approx(total - between)
            assert diagnostic["contribution_sum"] == pytest.approx(diagnostic["rho"])


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("zero", "constant IG axis"),
        ("missing", "missing or nonfinite token axis"),
        ("invalid", "missing or nonfinite token axis"),
        ("constant_ism", "constant ISM axis"),
        ("nonfinite_delta", "missing or nonfinite token axis"),
    ],
)
def test_undefined_axes_and_empty_span_preserve_null_reasons(mode, reason):
    from xai_test.cas12a_NT_analysis.nt_agreement import sequence_agreement

    mass = np.arange(9, dtype=float)
    valid = np.ones((34, 4), dtype=bool)
    values = deltas(range(9))
    if mode == "zero":
        mass[:] = 0
    elif mode == "missing":
        mass[0] = np.nan
    elif mode == "invalid":
        valid[:6] = False
    elif mode == "constant_ism":
        values = deltas([0] * 9)
    else:
        values[0, 1] = np.nan
    result = sequence_agreement(SEQUENCE, mass, values, base_valid_mask=valid)
    for version in result["versions"].values():
        assert version["correlations"]["five_6mer"]["density"]["rho"] is None
        assert version["correlations"]["five_6mer"]["density"]["reason"] == reason
        assert version["groups"]["length"]["within_contribution"] is None
        if mode == "invalid":
            assert version["tokens"][0]["ism_mean"] is None
            assert version["tokens"][0]["reason"] == "no valid bases"
