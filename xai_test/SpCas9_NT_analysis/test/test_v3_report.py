"""Tests for v3's aggregation and reporting step (#134, AC10 and AC13).

Four groups:

- **Token aggregation** — ISM is swept per base and IG is attributed per
  6-mer token, so the comparison only exists once ISM is folded onto IG's own
  five tokens. The fold's boundaries are asserted against `v3_ig`'s rather
  than restated, because a report that aggregates to its own idea of a token
  compares two different windows and reads as disagreement.
- **Reversal arithmetic** — Kendall tau over the three head pairs, the tie
  policy, and the noise-floor margin. The margin is the whole of AC13: a
  reversal whose inverted pairs sit inside the replicate noise floor is a
  measurement, not a model failure, and the tests pin that a per-head floor is
  used and that the pooled constant is reported apart rather than applied.
- **Reporting discipline** — what #134 states as acceptance criteria: the two
  strata are never summed, the library stratum carries its "5 targets"
  qualifier, the floor's limit counts are read from #135's artifact rather
  than written by hand, reversal is marked as *not* a selection pool, the
  annotation points at `cases.json` by `case_id` and cannot rewrite the
  selection, and nothing claims physical binding.
- **Real artifacts** — against the committed `ism.npz`, `ig.npz`, `cases.json`
  and `v3_noise_floor.json`, skipped when they or the split are absent since
  the arrays and the split are git-ignored.

`test_real_reversal_report_reproduces_the_pinned_reversals` is the AC's
reference figures written as a test: 203 reversals of 3,851 intersection rows,
169 of them in the library stratum, and **one** row whose inverted pairs clear
the noise floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import v3_agreement
import v3_ig
import v3_predict
import v3_report
import v3_selection
import v3_strata
from v3_agreement import pair_key
from v3_strata import ANALYSED_HEADS, GENERALISATION, PAM_LIBRARY

SUBTREE_ROOT = Path(v3_report.__file__).resolve().parent
XAI_TEST_ROOT = SUBTREE_ROOT.parent
SPLIT_CSV = XAI_TEST_ROOT / "SpCas9" / "published_split" / "test.csv"
ARTIFACTS = SUBTREE_ROOT / "artifacts"
ISM_NPZ = ARTIFACTS / "v3_ism" / "ism.npz"
IG_NPZ = ARTIFACTS / "v3_ig" / "ig.npz"
CASES_JSON = ARTIFACTS / "v3_cases" / "cases.json"
NOISE_FLOOR = ARTIFACTS / "v3_noise_floor.json"
PREDICTIONS_CSV = ARTIFACTS / "v3_testset_predictions" / "predictions.csv"

#: The figures #134's AC13 quotes, and the populations they are measured on.
INTERSECTION_ROWS = 3851
REVERSAL_ROWS = 203
LIBRARY_REVERSALS = 169
CLEARING_ROWS = 1
IG_SUBSET_ROWS = 400
IG_ROWS_PER_STRATUM = 200
CASE_COUNT = 18

BASES = v3_agreement.BASES
CODE_REF = "0" * 40
HEAD_PAIRS = v3_agreement.HEAD_PAIRS


# --- fixtures --------------------------------------------------------------


def sequence(*, protospacer: str = "A" * 20, pam: str = "AGG") -> str:
    return "ACGT" + protospacer + pam + "TCA"


def make_split(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_index": index,
                "sequence": row.get("sequence", sequence()),
                "in_pam_library": row.get("in_pam_library", False),
                **{head: row.get(head, 10.0) for head in v3_predict.HEADS},
            }
            for index, row in enumerate(rows)
        ]
    )


def make_predictions(
    split: pd.DataFrame, rows: list[dict] | None = None
) -> pd.DataFrame:
    """`row_index` plus one `pred_<head>` column per analysed head."""
    frame = pd.DataFrame({"row_index": split["row_index"].to_numpy(dtype=np.int64)})
    for position, head in enumerate(ANALYSED_HEADS):
        if rows is None:
            values = split[head].to_numpy(dtype=float) / v3_strata.LABEL_DIVISOR
        else:
            values = np.array(
                [row.get(f"pred_{head}", 0.1 * (position + 1)) for row in rows],
                dtype=float,
            )
        frame[f"pred_{head}"] = values
    return frame


def make_delta(
    profiles: dict[str, np.ndarray],
    sequences: list[str],
    *,
    sign: float = -1.0,
) -> np.ndarray:
    """A `(rows, 30, 4, 9)` delta whose per-position profile is `profiles`.

    The magnitude is spread evenly over a position's three alternative bases
    with the reference slot left at exactly 0, which is the layout `v3_ism`
    writes. `sign` flips the delta's direction without changing `|delta|`, so
    a test can separate the magnitude profile from the signed one.
    """
    rows = len(sequences)
    delta = np.zeros((rows, 30, len(BASES), len(v3_predict.HEADS)), dtype=np.float32)
    for head, profile in profiles.items():
        head_axis = v3_predict.HEADS.index(head)
        values = np.broadcast_to(np.asarray(profile, dtype=np.float64), (rows, 30))
        for row, text in enumerate(sequences):
            for position in range(30):
                for base_axis, base in enumerate(BASES):
                    if base != text[position]:
                        delta[row, position, base_axis, head_axis] = (
                            sign * values[row, position]
                        )
    return delta


def ism_sidecar(**overrides) -> dict:
    payload = {
        "heads": list(v3_predict.HEADS),
        "code_ref": CODE_REF,
        "population": "analysis_intersection",
        "population_digest": "f" * 64,
        "delta_scale": "fraction",
    }
    payload.update(overrides)
    return payload


def write_ism(
    directory: Path,
    split: pd.DataFrame,
    delta: np.ndarray,
    *,
    sidecar: dict | None = None,
) -> Path:
    sequences = list(split["sequence"])
    reference = np.array(
        [[BASES.index(base) for base in text] for text in sequences], dtype=np.int8
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ism.npz"
    np.savez_compressed(
        path,
        delta=delta.astype(np.float32),
        wt_prediction=np.zeros((len(split), len(v3_predict.HEADS)), dtype=np.float32),
        reference_base=reference,
        row_index=split["row_index"].to_numpy(dtype=np.int64),
    )
    (directory / "ism.sidecar.json").write_text(
        json.dumps(ism_sidecar() if sidecar is None else sidecar), encoding="utf-8"
    )
    return path


def ig_sidecar(**overrides) -> dict:
    payload = {
        "heads": list(ANALYSED_HEADS),
        "code_ref": CODE_REF,
        "steps": v3_ig.DEFAULT_IG_STEPS,
        "token_names": list(v3_ig.TOKEN_NAMES),
        "token_boundaries": [list(bounds) for bounds in v3_ig.TOKEN_BOUNDARIES],
        "attribution_point": v3_ig.ATTRIBUTION_POINT,
        "baseline": v3_ig.BASELINE,
        "subset": {"manifest": "ig_subset.json", "membership_digest": "a" * 64},
        "attribution_sha256": "b" * 64,
    }
    payload.update(overrides)
    return payload


def write_ig(
    directory: Path,
    row_index: np.ndarray,
    attribution: np.ndarray,
    *,
    sidecar: dict | None = None,
) -> Path:
    """`ig.npz` in the shape `v3_ig.write_ig_artifact` writes it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ig.npz"
    rows = len(row_index)
    np.savez_compressed(
        path,
        attribution=np.asarray(attribution, dtype=np.float32),
        completeness_residual=np.zeros((rows, len(ANALYSED_HEADS)), dtype=np.float64),
        special_attribution=np.zeros((rows, len(ANALYSED_HEADS)), dtype=np.float64),
        row_index=np.asarray(row_index, dtype=np.int64),
    )
    (directory / "ig.sidecar.json").write_text(
        json.dumps(ig_sidecar() if sidecar is None else sidecar), encoding="utf-8"
    )
    return path


def noise_floor_payload(
    *,
    medians: dict[str, float] | None = None,
    pooled_median: float = 0.05,
    **overrides,
) -> dict:
    medians = medians or {"SpCas9": 0.06, "SpCas9-NG": 0.07, "evoCas9": 0.01}
    payload = {
        "model": "9-Hydra v3",
        "analysis": "replicate_noise_floor",
        "inputs": ["labels"],
        "limit_note": "the floor rests on the split's repeated 30-mers",
        "single_constant_note": "use by_head, not pooled",
        "groups": 79,
        "rows": 379,
        "heads": list(ANALYSED_HEADS),
        "by_head": {
            head: {
                "groups": 79,
                "median": value,
                "p90": value * 2,
                "mean": value,
                "min": value / 10,
                "max": value * 3,
            }
            for head, value in medians.items()
        },
        "pooled": {
            "measurements": 237,
            "median": pooled_median,
            "p90": pooled_median * 2,
            "mean": pooled_median,
            "min": pooled_median / 10,
            "max": pooled_median * 3,
        },
        "head_floor_ratio": max(medians.values()) / min(medians.values()),
        "label_scale": "fraction",
        "intersection_rows": 3851,
        "split_rows": 5186,
        "repeated_groups_in_split": 779,
        "repeated_rows_in_split": 2247,
    }
    payload.update(overrides)
    return payload


def write_noise_floor(directory: Path, payload: dict | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = v3_predict.write_sidecar(
        directory,
        v3_report.NOISE_NAME,
        noise_floor_payload() if payload is None else payload,
    )
    v3_predict.write_sidecar(
        directory,
        v3_report.NOISE_CHECKSUM_NAME,
        {"files": {v3_report.NOISE_NAME: v3_predict.sha256_of(path)}},
    )
    return path


def cases_payload(cases: list[dict]) -> dict:
    return {
        "model": "9-Hydra v3",
        "analysis": "case_selection",
        "inputs": ["predictions", "labels"],
        "criteria": ["ERRLO", "ERRHI", "HEADGAP"],
        "heads": list(ANALYSED_HEADS),
        "cases": cases,
    }


def write_cases(directory: Path, cases: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = v3_predict.write_sidecar(
        directory, v3_report.CASES_NAME, cases_payload(cases)
    )
    v3_predict.write_sidecar(
        directory,
        v3_report.CASES_CHECKSUM_NAME,
        {"files": {v3_report.CASES_NAME: v3_predict.sha256_of(path)}},
    )
    return path


def ramp_profile() -> np.ndarray:
    """Rising over the window, so the five token means rise too."""
    return np.linspace(0.1, 3.0, 30)


def token_ramp(*, reverse: bool = False) -> np.ndarray:
    values = np.arange(1.0, v3_report.TOKEN_COUNT + 1)
    return values[::-1] if reverse else values


# --- token aggregation -----------------------------------------------------


def test_token_bounds_are_the_ig_artifacts_own_tokens():
    # The comparison only means anything if both sides aggregate to the same
    # five windows, so the bounds are derived from `v3_ig` rather than
    # restated here.
    assert v3_report.TOKEN_COUNT == v3_ig.TOKEN_COUNT == 5
    assert v3_report.TOKEN_NAMES == v3_ig.TOKEN_NAMES
    assert v3_report.token_bounds() == tuple(
        (start, stop + 1) for start, stop in v3_ig.TOKEN_BOUNDARIES
    )
    assert v3_report.token_bounds()[-1] == (24, 30)


def test_ism_token_profile_averages_each_tokens_six_positions():
    profile = np.zeros((1, 30, len(ANALYSED_HEADS)))
    for token, (start, stop) in enumerate(v3_report.token_bounds()):
        profile[0, start:stop, :] = token + 1.0
    tokens = v3_report.ism_token_profile(profile)
    assert tokens.shape == (1, 5, len(ANALYSED_HEADS))
    assert tokens[0, :, 0] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_ism_token_profile_refuses_a_window_it_cannot_tokenise():
    with pytest.raises(ValueError, match="30"):
        v3_report.ism_token_profile(np.zeros((1, 29, 3)))


def test_ism_tokens_keep_the_sign_only_in_the_signed_variant():
    sequences = [sequence()]
    profiles = {head: ramp_profile() for head in ANALYSED_HEADS}
    negative = make_delta(profiles, sequences, sign=-1.0)
    positive = make_delta(profiles, sequences, sign=+1.0)

    magnitude = v3_report.ism_token_tensor(
        v3_agreement.select_heads(negative, v3_predict.HEADS), v3_report.MAGNITUDE
    )
    flipped = v3_report.ism_token_tensor(
        v3_agreement.select_heads(positive, v3_predict.HEADS), v3_report.MAGNITUDE
    )
    assert magnitude == pytest.approx(flipped)
    assert (magnitude > 0).all()

    signed = v3_report.ism_token_tensor(
        v3_agreement.select_heads(negative, v3_predict.HEADS), v3_report.SIGNED
    )
    assert (signed < 0).all()
    assert signed == pytest.approx(-magnitude)


def test_ig_token_tensor_puts_the_tokens_on_the_middle_axis():
    attribution = np.arange(2 * 3 * 5, dtype=np.float64).reshape(2, 3, 5)
    tokens = v3_report.ig_token_tensor(attribution, v3_report.SIGNED)
    assert tokens.shape == (2, 5, 3)
    assert tokens[0, :, 0] == pytest.approx(attribution[0, 0, :])
    assert tokens[1, :, 2] == pytest.approx(attribution[1, 2, :])


def test_ig_token_tensor_magnitude_drops_the_sign():
    attribution = -np.ones((1, 3, 5))
    assert (v3_report.ig_token_tensor(attribution, v3_report.MAGNITUDE) > 0).all()


# --- the correlation -------------------------------------------------------


def one_row(values: np.ndarray) -> np.ndarray:
    """`(1, 5, 1)` from five token values — one row, one head."""
    return np.asarray(values, dtype=np.float64).reshape(1, -1, 1)


def test_identical_token_profiles_correlate_perfectly():
    tokens = one_row(token_ramp())
    assert v3_report.profile_correlation(tokens, tokens).ravel() == pytest.approx([1.0])


def test_reversed_token_profiles_anticorrelate_perfectly():
    assert v3_report.profile_correlation(
        one_row(token_ramp()), one_row(token_ramp(reverse=True))
    ).ravel() == pytest.approx([-1.0])


def test_a_constant_profile_is_undefined_not_zero():
    correlation = v3_report.profile_correlation(
        one_row(token_ramp()), one_row(np.ones(5))
    )
    assert np.isnan(correlation).all()
    summary = v3_agreement.summarise_values(correlation)
    assert summary == {"rows": 0, "undefined_rows": 1}


def test_mean_profile_correlation_is_one_number_per_head():
    rising = np.stack([token_ramp(), token_ramp() * 2.0])[:, :, None]
    falling = np.stack([token_ramp(reverse=True)] * 2)[:, :, None]
    assert v3_report.mean_profile_correlation(rising, rising) == pytest.approx([1.0])
    assert v3_report.mean_profile_correlation(rising, falling) == pytest.approx([-1.0])


# --- reversal arithmetic ---------------------------------------------------


def test_kendall_tau_is_one_for_the_same_ordering():
    labels = np.array([[0.1, 0.2, 0.3]])
    assert v3_report.kendall_tau(labels, labels * 2.0) == pytest.approx([1.0])


def test_kendall_tau_is_minus_one_for_a_fully_reversed_ordering():
    labels = np.array([[0.1, 0.2, 0.3]])
    assert v3_report.kendall_tau(labels, labels[:, ::-1]) == pytest.approx([-1.0])


def test_kendall_tau_is_a_third_when_one_pair_of_three_is_discordant():
    labels = np.array([[0.1, 0.2, 0.3]])
    # Swapping the two smallest leaves two concordant pairs of three.
    predictions = np.array([[0.2, 0.1, 0.3]])
    assert v3_report.kendall_tau(labels, predictions) == pytest.approx([1 / 3])


def test_inverted_pairs_are_the_discordant_ones_in_head_pair_order():
    labels = np.array([[0.1, 0.2, 0.3]])
    predictions = np.array([[0.2, 0.1, 0.3]])
    inverted = v3_report.inverted_pairs(labels, predictions)
    assert inverted.shape == (1, len(HEAD_PAIRS))
    # Only `SpCas9` vs `SpCas9-NG` swapped.
    assert list(inverted[0]) == [True, False, False]


def test_a_tied_label_pair_has_no_ordering_to_reverse():
    labels = np.array([[0.2, 0.2, 0.3]])
    predictions = np.array([[0.3, 0.2, 0.1]])
    assert v3_report.tied_pairs(labels)[0].any()
    assert v3_report.tied_rows(labels, predictions)[0]
    assert not v3_report.reversal_rows(labels, predictions)[0]


def test_a_reversal_needs_a_negative_tau_and_no_tie():
    labels = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    predictions = np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.3]])
    assert list(v3_report.reversal_rows(labels, predictions)) == [True, False]


def test_pair_gaps_are_absolute_differences_in_head_pair_order():
    values = np.array([[0.1, 0.4, 0.9]])
    gaps = v3_report.pair_gaps(values)
    assert gaps[0] == pytest.approx([0.3, 0.8, 0.5])


def test_pair_floor_is_the_larger_of_the_two_heads_own_floors():
    floors = v3_report.pair_floors(noise_floor_payload())
    assert floors[pair_key("SpCas9", "SpCas9-NG")] == pytest.approx(0.07)
    assert floors[pair_key("SpCas9", "evoCas9")] == pytest.approx(0.06)
    assert floors[pair_key("SpCas9-NG", "evoCas9")] == pytest.approx(0.07)


def test_pair_floors_never_reach_for_the_pooled_constant():
    # `v3_noise.floor_for_head` refuses `"pooled"`; the pair rule must go
    # through it rather than reading `pooled` itself.
    payload = noise_floor_payload()
    payload.pop("by_head")
    with pytest.raises(ValueError, match="by_head|unknown head"):
        v3_report.pair_floors(payload)


def test_a_reversal_inside_the_noise_floor_does_not_clear_it():
    labels = np.array([[0.10, 0.11, 0.9]])
    predictions = np.array([[0.11, 0.10, 0.9]])
    inverted = v3_report.inverted_pairs(labels, predictions)
    gaps = v3_report.pair_gaps(labels)
    floors = v3_report.floor_vector(noise_floor_payload())
    assert not v3_report.clears_noise_floor(gaps, inverted, floors)[0]


def test_clearing_requires_every_inverted_pair_to_clear():
    # Two inverted pairs: one separated by 0.5, one by 0.01. The row clears
    # only if both do, which is what keeps a wide gap from carrying a narrow
    # one through.
    labels = np.array([[0.10, 0.11, 0.80]])
    predictions = np.array([[0.11, 0.10, 0.05]])
    inverted = v3_report.inverted_pairs(labels, predictions)
    assert inverted[0].sum() == 3
    floors = v3_report.floor_vector(noise_floor_payload())
    assert not v3_report.clears_noise_floor(
        v3_report.pair_gaps(labels), inverted, floors
    )[0]

    wide = np.array([[0.10, 0.60, 0.95]])
    wide_predictions = np.array([[0.95, 0.60, 0.10]])
    assert v3_report.clears_noise_floor(
        v3_report.pair_gaps(wide),
        v3_report.inverted_pairs(wide, wide_predictions),
        floors,
    )[0]


def test_a_row_with_no_inverted_pair_does_not_clear_the_floor():
    labels = np.array([[0.1, 0.5, 0.9]])
    assert not v3_report.clears_noise_floor(
        v3_report.pair_gaps(labels),
        v3_report.inverted_pairs(labels, labels),
        v3_report.floor_vector(noise_floor_payload()),
    )[0]


# --- reading #135's artifact ------------------------------------------------


def test_the_floor_is_read_from_the_artifact_with_its_limit_counts(tmp_path):
    path = write_noise_floor(tmp_path)
    payload = v3_report.read_noise_floor(path)
    assert payload["groups"] == 79
    assert payload["rows"] == 379


@pytest.mark.parametrize("key", ["by_head", "groups", "rows", "limit_note", "pooled"])
def test_a_floor_that_cannot_state_its_limit_is_refused(tmp_path, key):
    payload = noise_floor_payload()
    payload.pop(key)
    path = write_noise_floor(tmp_path, payload)
    with pytest.raises(ValueError, match="noise floor"):
        v3_report.read_noise_floor(path)


def test_a_floor_whose_bytes_do_not_match_its_checksum_is_refused(tmp_path):
    path = write_noise_floor(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["by_head"]["evoCas9"]["median"] = 0.5
    v3_predict.write_sidecar(tmp_path, v3_report.NOISE_NAME, payload)
    with pytest.raises(ValueError, match="checksum"):
        v3_report.read_noise_floor(path)


# --- synthetic reports -----------------------------------------------------


def synthetic_rows() -> list[dict]:
    """Four generalisation rows, five library rows, one in neither, three out.

    Two kinds of row sit outside the strata, and the report has to keep both
    straight. The non-library non-NGG row is **inside** the analysis
    intersection and in neither stratum, which is what makes the intersection
    larger than the two strata summed; the last three each leave one analysed
    head unmeasured, so they are outside the intersection altogether.
    """
    rows: list[dict] = [
        {"sequence": sequence(protospacer="A" * 19 + base, pam="AGG")}
        for base in "ACGT"
    ]
    rows += [
        {"sequence": sequence(protospacer="C" * 20, pam=pam), "in_pam_library": True}
        for pam in ("AGG", "AGA", "TGG")
    ]
    rows += [
        {"sequence": sequence(protospacer="G" * 20, pam=pam), "in_pam_library": True}
        for pam in ("AGG", "AGA")
    ]
    rows += [{"sequence": sequence(protospacer="T" * 19 + "C", pam="AGA")}]
    rows += [
        {"sequence": sequence(protospacer="T" * 20, pam="AGA"), head: np.nan}
        for head in ANALYSED_HEADS
    ]
    return rows


def synthetic_inputs(tmp_path: Path, *, ig_rows: int | None = None) -> dict:
    """Split, predictions, ISM, IG and a noise floor, all on disk."""
    rows = synthetic_rows()
    split = make_split(rows)
    split_csv = tmp_path / "test.csv"
    split.to_csv(split_csv, index=False)

    predictions_csv = tmp_path / "predictions.csv"
    make_predictions(split).to_csv(predictions_csv, index=False)

    measured = split.loc[:, list(ANALYSED_HEADS)].notna().all(axis=1)
    swept = split.loc[measured].reset_index(drop=True)
    profiles = {head: ramp_profile() for head in ANALYSED_HEADS}
    ism = write_ism(
        tmp_path / "v3_ism",
        swept,
        make_delta(profiles, list(swept["sequence"])),
    )

    # IG agrees with ISM on the first head and runs against it on the last,
    # so the report has a per-head spread to show.
    subset = swept["row_index"].to_numpy(dtype=np.int64)
    if ig_rows is not None:
        subset = subset[:ig_rows]
    attribution = np.zeros((subset.size, len(ANALYSED_HEADS), 5), dtype=np.float32)
    attribution[:, 0, :] = token_ramp()
    attribution[:, 1, :] = token_ramp()
    attribution[:, 2, :] = token_ramp(reverse=True)
    ig = write_ig(tmp_path / "v3_ig", subset, attribution)

    return {
        "split_csv": split_csv,
        "predictions_csv": predictions_csv,
        "ism_npz": ism,
        "ig_npz": ig,
        "noise_floor": write_noise_floor(tmp_path),
    }


def synthetic_validation(tmp_path: Path, **overrides) -> dict:
    inputs = synthetic_inputs(tmp_path)
    return v3_report.build_validation_report(
        ism_npz=inputs["ism_npz"],
        ig_npz=inputs["ig_npz"],
        split_csv=inputs["split_csv"],
        split_checksum=None,
        **overrides,
    )


def synthetic_reversal(tmp_path: Path, **overrides) -> dict:
    inputs = synthetic_inputs(tmp_path)
    return v3_report.build_reversal_report(
        split_csv=inputs["split_csv"],
        predictions_csv=inputs["predictions_csv"],
        noise_floor=inputs["noise_floor"],
        split_checksum=None,
        **overrides,
    )


def walk_values(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from walk_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk_values(value)
    else:
        yield node


def walk_keys(node, trail=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield f"{trail}/{key}".lower()
            yield from walk_keys(value, f"{trail}/{key}")
    elif isinstance(node, list):
        for value in node:
            yield from walk_keys(value, trail)


def assert_claims_no_binding(payload: dict) -> None:
    forbidden = ("bind", "docking", "affinity", "contact")
    assert not [key for key in walk_keys(payload) if any(w in key for w in forbidden)]
    assert payload["claim_limit"] == v3_report.CLAIM_LIMIT


def assert_finite(payload: dict) -> None:
    assert all(
        np.isfinite(value) for value in walk_values(payload) if isinstance(value, float)
    )
    # NaN is not JSON, so a NaN that reached the payload would be written as a
    # literal nothing else parses.
    json.dumps(payload, allow_nan=False)


# --- AC10: the IG <-> ISM validation report --------------------------------


def test_validation_report_is_per_stratum_and_per_head(tmp_path):
    report = synthetic_validation(tmp_path)
    assert set(report["strata"]) == {GENERALISATION, PAM_LIBRARY}
    assert report["statistic"] == v3_report.VALIDATION_STATISTIC
    for stratum in report["strata"].values():
        assert set(stratum["by_head"]) == set(ANALYSED_HEADS)
        for head in stratum["by_head"].values():
            assert set(head) == set(v3_report.PROFILE_VARIANTS)
            for variant in head.values():
                assert set(variant) >= {"per_row", "mean_profile"}


def test_validation_report_has_no_pooled_entry(tmp_path):
    report = synthetic_validation(tmp_path)
    assert report["pooling"] == "forbidden"
    rows = [report["strata"][name]["rows"] for name in (GENERALISATION, PAM_LIBRARY)]
    pooled = sum(rows)
    assert pooled not in [
        value for value in walk_values(report["strata"]) if isinstance(value, int)
    ]


def test_validation_report_carries_the_library_qualifier(tmp_path):
    report = synthetic_validation(tmp_path)
    library = report["strata"][PAM_LIBRARY]
    assert library["qualifier"] == v3_report.LIBRARY_QUALIFIER
    assert library["qualifier_note"] == v3_report.LIBRARY_QUALIFIER_NOTE
    assert "distinct_protospacers" in library
    assert "qualifier" not in report["strata"][GENERALISATION]


def test_a_stratum_aggregate_is_not_labelled_as_one_target(tmp_path):
    # The case table's wording is a claim about one row. On a stratum pooled
    # over every target it would read as "this figure came from one target",
    # so the two spellings stay two different limits.
    assert v3_report.LIBRARY_QUALIFIER == v3_agreement.LIBRARY_QUALIFIER
    assert v3_report.LIBRARY_QUALIFIER != v3_selection.LIBRARY_QUALIFIER
    report = synthetic_validation(tmp_path)
    assert v3_selection.LIBRARY_QUALIFIER not in json.dumps(
        report["strata"][PAM_LIBRARY]
    )


def test_validation_report_breaks_the_library_stratum_down_by_target(tmp_path):
    # Pooling PAM variants of a handful of protospacers reads as that many
    # independent observations, so the pool is reported with its targets.
    library = synthetic_validation(tmp_path)["strata"][PAM_LIBRARY]
    per_target = library["per_target"]
    assert len(per_target) == library["distinct_protospacers"] == 2
    assert sum(entry["rows"] for entry in per_target) == library["rows"]
    for entry in per_target:
        assert set(entry["per_row_median"]) == set(v3_report.PROFILE_VARIANTS)
        assert set(entry["per_row_median"][v3_report.MAGNITUDE]) == set(ANALYSED_HEADS)
    assert "per_target" not in synthetic_validation(tmp_path)["strata"][GENERALISATION]


def test_validation_report_states_the_token_mismatch_limit(tmp_path):
    report = synthetic_validation(tmp_path)
    assert report["mismatch_note"] == v3_report.MISMATCH_NOTE
    assert "0007" in report["mismatch_note"]
    assert v3_ig.PAM_TOKEN_NAME in json.dumps(report)


def test_validation_report_holds_no_base_resolution_pam_key(tmp_path):
    report = synthetic_validation(tmp_path)
    assert not [key for key in walk_keys(report) if key.endswith("/pam")]
    assert report["token_names"] == list(v3_ig.TOKEN_NAMES)


def test_validation_report_reports_a_disagreeing_head_rather_than_raising(tmp_path):
    # The last head's IG runs against its ISM profile by construction. AC10
    # says such a mismatch is a limit to report, not a failure to raise on.
    report = synthetic_validation(tmp_path)
    for stratum in report["strata"].values():
        magnitude = stratum["by_head"]["evoCas9"][v3_report.MAGNITUDE]
        assert magnitude["mean_profile"] == pytest.approx(-1.0)
        assert stratum["by_head"]["SpCas9"][v3_report.MAGNITUDE][
            "mean_profile"
        ] == pytest.approx(1.0)


def test_validation_report_says_which_way_round_the_signed_variant_agrees(tmp_path):
    # IG's sign is what the base present contributes; an ISM delta's is what
    # substituting it away does. The synthetic ISM delta is negative where the
    # synthetic IG attribution is positive, which is the *agreeing* case, and
    # a reader who takes the resulting -1 for disagreement has mis-read it.
    report = synthetic_validation(tmp_path)
    assert report["agreement_sign"] == {v3_report.MAGNITUDE: 1, v3_report.SIGNED: -1}
    assert report["signed_convention_note"] == v3_report.SIGNED_CONVENTION_NOTE
    for stratum in report["strata"].values():
        head = stratum["by_head"]["SpCas9"]
        assert head[v3_report.MAGNITUDE]["agreement_sign"] == 1
        assert head[v3_report.MAGNITUDE]["mean_profile"] == pytest.approx(1.0)
        signed = head[v3_report.SIGNED]
        assert signed["agreement_sign"] == -1
        assert signed["mean_profile"] == pytest.approx(-1.0)
        # Agreement read through the convention: every row agrees.
        assert signed["per_row"]["share_negative"] == pytest.approx(1.0)


def test_validation_report_claims_no_binding_and_is_finite(tmp_path):
    report = synthetic_validation(tmp_path)
    assert_claims_no_binding(report)
    assert_finite(report)


def test_validation_report_records_its_inputs_by_name_and_digest_only(tmp_path):
    report = synthetic_validation(tmp_path)
    consumed = report["consumed"]
    assert consumed["ism"]["name"] == "ism.npz"
    assert consumed["ig"]["name"] == "ig.npz"
    assert consumed["split"]["name"] == "test.csv"
    assert len(consumed["ig"]["sha256"]) == 64
    assert consumed["ig"]["sidecar"]["code_ref"] == CODE_REF
    assert str(tmp_path) not in json.dumps(report)


def test_validation_report_refuses_ig_rows_the_sweep_never_covered(tmp_path):
    inputs = synthetic_inputs(tmp_path)
    attribution = np.ones((1, len(ANALYSED_HEADS), 5), dtype=np.float32)
    outside = write_ig(tmp_path / "v3_ig_outside", np.array([9999]), attribution)
    with pytest.raises(ValueError, match="swept|intersection"):
        v3_report.build_validation_report(
            ism_npz=inputs["ism_npz"],
            ig_npz=outside,
            split_csv=inputs["split_csv"],
            split_checksum=None,
        )


def test_validation_report_refuses_an_ig_artifact_with_other_token_names(tmp_path):
    inputs = synthetic_inputs(tmp_path)
    swept = np.load(inputs["ism_npz"])["row_index"]
    attribution = np.ones((swept.size, len(ANALYSED_HEADS), 5), dtype=np.float32)
    renamed = write_ig(
        tmp_path / "v3_ig_renamed",
        swept,
        attribution,
        sidecar=ig_sidecar(token_names=["pam", "b", "c", "d", "e"]),
    )
    with pytest.raises(ValueError, match="token"):
        v3_report.build_validation_report(
            ism_npz=inputs["ism_npz"],
            ig_npz=renamed,
            split_csv=inputs["split_csv"],
            split_checksum=None,
        )


def test_write_validation_report_round_trips(tmp_path):
    report = synthetic_validation(tmp_path)
    path = v3_report.write_validation_report(tmp_path / "out", report)
    assert path.name == v3_report.VALIDATION_NAME
    assert json.loads(path.read_text(encoding="utf-8")) == report


# --- AC13: the head-order reversal report ----------------------------------


def test_reversal_report_is_not_a_selection_pool(tmp_path):
    report = synthetic_reversal(tmp_path)
    assert report["selection_pool"] == "forbidden"
    assert "margin" in report["selection_pool_note"]
    # No row list to pick from: the report carries counts and distributions.
    assert "row_index" not in json.dumps(report)


def test_reversal_report_separates_the_intersection_from_the_strata(tmp_path):
    report = synthetic_reversal(tmp_path)
    assert report["population"]["name"] == v3_report.POPULATION
    assert set(report["strata"]) == {GENERALISATION, PAM_LIBRARY}
    assert report["pooling"] == "forbidden"
    strata_rows = sum(
        report["strata"][name]["rows"] for name in (GENERALISATION, PAM_LIBRARY)
    )
    assert report["population"]["rows"] != strata_rows
    assert "neither" in report["pooling_note"]


def test_reversal_report_uses_per_head_floors_and_reports_pooled_apart(tmp_path):
    report = synthetic_reversal(tmp_path)
    floor = report["floor"]
    assert floor["statistic"] == "median"
    assert floor["by_head"] == {"SpCas9": 0.06, "SpCas9-NG": 0.07, "evoCas9": 0.01}
    assert floor["pair_floors"][pair_key("SpCas9", "evoCas9")] == pytest.approx(0.06)
    assert floor["pooled_median"] == pytest.approx(0.05)
    assert floor["pooled_applied"] is False
    assert "head_floor_ratio" in floor
    for population in (report["population"], *report["strata"].values()):
        clearing = population["clearing_floor"]
        assert clearing["rule"] == v3_report.PAIR_FLOOR_RULE
        assert "rows" in clearing
        assert "pooled_rule_rows" in clearing


def test_reversal_report_reads_the_floors_limit_from_the_artifact(tmp_path):
    report = synthetic_reversal(tmp_path)
    limit = report["floor"]["limit"]
    assert limit["groups"] == 79
    assert limit["rows"] == 379
    assert limit["repeated_groups_in_split"] == 779
    assert limit["source"] == v3_report.NOISE_NAME
    assert limit["note"]


def test_reversal_report_counts_tied_rows_apart_from_reversals(tmp_path):
    report = synthetic_reversal(tmp_path)
    population = report["population"]
    assert population["reversal_rows"] <= population["tau_negative_rows"]
    assert (
        population["reversal_rows"] + population["tied_rows_with_negative_tau"]
        == population["tau_negative_rows"]
    )
    assert report["tie_policy"]


def test_reversal_report_distributes_the_inverted_pair_gaps(tmp_path):
    report = synthetic_reversal(tmp_path)
    for population in (report["population"], *report["strata"].values()):
        pairs = population["inverted_pairs"]
        assert set(pairs) == {pair_key(*pair) for pair in HEAD_PAIRS}
        for entry in pairs.values():
            assert set(entry) >= {"rows", "floor", "label_gap", "rows_clearing_floor"}
        assert "gap_over_floor" in population


def test_reversal_report_carries_the_library_qualifier(tmp_path):
    report = synthetic_reversal(tmp_path)
    library = report["strata"][PAM_LIBRARY]
    assert library["qualifier"] == v3_report.LIBRARY_QUALIFIER
    assert library["qualifier_note"] == v3_report.LIBRARY_QUALIFIER_NOTE
    assert "qualifier" not in report["strata"][GENERALISATION]


def test_reversal_report_breaks_the_library_stratum_down_by_target(tmp_path):
    # A pooled rate that one target carries is a different result from one
    # that holds on all of them, so the rate is reported target by target.
    library = synthetic_reversal(tmp_path)["strata"][PAM_LIBRARY]
    per_target = library["per_target"]
    assert len(per_target) == library["distinct_protospacers"] == 2
    assert sum(entry["rows"] for entry in per_target) == library["rows"]
    assert (
        sum(entry["reversal_rows"] for entry in per_target) == library["reversal_rows"]
    )
    for entry in per_target:
        assert set(entry) == {
            "protospacer",
            "rows",
            "reversal_rows",
            "reversal_rate",
            "rows_clearing_floor",
            "gap_over_floor_median",
        }
    assert "per_target" not in synthetic_reversal(tmp_path)["strata"][GENERALISATION]


def test_reversal_report_claims_no_binding_and_is_finite(tmp_path):
    report = synthetic_reversal(tmp_path)
    assert_claims_no_binding(report)
    assert_finite(report)


def test_reversal_report_counts_a_constructed_reversal(tmp_path):
    rows = synthetic_rows()
    split = make_split(rows)
    split_csv = tmp_path / "test.csv"
    split.to_csv(split_csv, index=False)
    # Labels are equal across heads by default, so give the first row a wide
    # label spread and a reversed prediction order: one reversal that clears
    # the floor, and no other row able to produce one.
    split.loc[0, "SpCas9"] = 10.0
    split.loc[0, "SpCas9-NG"] = 50.0
    split.loc[0, "evoCas9"] = 90.0
    split.to_csv(split_csv, index=False)
    predictions = make_predictions(split)
    predictions.loc[0, [f"pred_{head}" for head in ANALYSED_HEADS]] = [0.9, 0.5, 0.1]
    predictions_csv = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_csv, index=False)

    report = v3_report.build_reversal_report(
        split_csv=split_csv,
        predictions_csv=predictions_csv,
        noise_floor=write_noise_floor(tmp_path),
        split_checksum=None,
    )
    assert report["population"]["reversal_rows"] == 1
    assert report["population"]["clearing_floor"]["rows"] == 1
    assert report["strata"][GENERALISATION]["reversal_rows"] == 1
    assert report["strata"][PAM_LIBRARY]["reversal_rows"] == 0


def test_write_reversal_report_round_trips(tmp_path):
    report = synthetic_reversal(tmp_path)
    path = v3_report.write_reversal_report(tmp_path / "out", report)
    assert path.name == v3_report.REVERSAL_NAME
    assert json.loads(path.read_text(encoding="utf-8")) == report


# --- the annotated case table ----------------------------------------------


def synthetic_cases(tmp_path: Path, inputs: dict) -> Path:
    """Two cases over rows the synthetic ISM covers, one per stratum."""
    split = pd.read_csv(inputs["split_csv"])
    return write_cases(
        tmp_path / "v3_cases",
        [
            {
                "case_id": "GEN-ERRHI-01",
                "stratum": GENERALISATION,
                "criterion": "ERRHI",
                "selection_rank": 1,
                "row_index": int(split["row_index"].iloc[0]),
                "sequence": str(split["sequence"].iloc[0]),
                "reporting_qualifier": "",
            },
            {
                "case_id": "LIB-ERRLO-01",
                "stratum": PAM_LIBRARY,
                "criterion": "ERRLO",
                "selection_rank": 1,
                "row_index": int(split["row_index"].iloc[4]),
                "sequence": str(split["sequence"].iloc[4]),
                "reporting_qualifier": v3_selection.LIBRARY_QUALIFIER,
            },
        ],
    )


def synthetic_annotation(tmp_path: Path, **overrides) -> dict:
    inputs = synthetic_inputs(tmp_path)
    cases = synthetic_cases(tmp_path, inputs)
    arguments = {
        "cases_json": cases,
        "ism_npz": inputs["ism_npz"],
        "ig_npz": inputs["ig_npz"],
        "split_csv": inputs["split_csv"],
        "predictions_csv": inputs["predictions_csv"],
        "noise_floor": inputs["noise_floor"],
        "split_checksum": None,
    }
    arguments.update(overrides)
    return v3_report.annotate_cases(**arguments)


def test_annotation_points_at_the_case_table_by_case_id(tmp_path):
    annotated = synthetic_annotation(tmp_path)
    assert annotated["points_at"]["file"] == v3_report.CASES_NAME
    assert annotated["points_at"]["key"] == "case_id"
    assert len(annotated["points_at"]["sha256"]) == 64
    assert annotated["rewrites_selection"] is False
    assert [case["case_id"] for case in annotated["cases"]] == [
        "GEN-ERRHI-01",
        "LIB-ERRLO-01",
    ]


def test_annotation_adds_attribution_and_keeps_the_selection_fields(tmp_path):
    annotated = synthetic_annotation(tmp_path)
    for case in annotated["cases"]:
        assert set(case) >= {
            "case_id",
            "row_index",
            "stratum",
            "criterion",
            "ism",
            "head_order",
            "attribution_outlier",
        }
        assert set(case["ism"]["token_profile"]) == set(ANALYSED_HEADS)
        assert len(case["ism"]["token_profile"]["SpCas9"]) == 5
        assert set(case["ism"]["top_position"]) == set(ANALYSED_HEADS)
        assert set(case["head_order"]) >= {"tau", "reversal", "clears_noise_floor"}


def test_annotation_holds_no_selection_score_it_could_re_rank_on(tmp_path):
    annotated = synthetic_annotation(tmp_path)
    # The selection's own scores stay in `cases.json`: copying them here is
    # what a re-ranking would read.
    forbidden = ("abs_error", "head_gap", "error_quartile", "selection_rank")
    assert not [
        key for key in walk_keys(annotated["cases"]) if any(w in key for w in forbidden)
    ]


def test_annotation_refuses_to_rewrite_the_selection(tmp_path):
    annotated = synthetic_annotation(tmp_path)
    cases = cases_payload(
        [
            {
                "case_id": "GEN-ERRHI-01",
                "stratum": GENERALISATION,
                "criterion": "ERRHI",
                "row_index": 0,
            },
        ]
    )
    with pytest.raises(ValueError, match="selection"):
        v3_report.assert_selection_unchanged(cases, annotated)

    reordered = dict(annotated)
    reordered["cases"] = list(reversed(annotated["cases"]))
    with pytest.raises(ValueError, match="selection"):
        v3_report.assert_selection_unchanged(
            json.loads(
                (tmp_path / "v3_cases" / v3_report.CASES_NAME).read_text(
                    encoding="utf-8"
                )
            ),
            reordered,
        )


def test_annotation_records_a_case_outside_the_ig_subset_as_absent(tmp_path):
    inputs = synthetic_inputs(tmp_path, ig_rows=1)
    cases = synthetic_cases(tmp_path, inputs)
    annotated = v3_report.annotate_cases(
        cases_json=cases,
        ism_npz=inputs["ism_npz"],
        ig_npz=inputs["ig_npz"],
        split_csv=inputs["split_csv"],
        predictions_csv=inputs["predictions_csv"],
        noise_floor=inputs["noise_floor"],
        split_checksum=None,
    )
    absent = [case for case in annotated["cases"] if case["ig"] is None]
    assert len(absent) == 1
    assert annotated["ig_coverage"]["cases_with_ig"] == 1
    assert annotated["ig_coverage"]["note"]


def test_annotation_marks_outliers_as_description_only(tmp_path):
    annotated = synthetic_annotation(tmp_path)
    outlier = annotated["attribution_outlier"]
    assert outlier["use"] == v3_report.OUTLIER_USE
    assert outlier["threshold_percentile"] == v3_report.OUTLIER_PERCENTILE
    for case in annotated["cases"]:
        assert set(case["attribution_outlier"]) == set(ANALYSED_HEADS)
        assert all(
            isinstance(flag, bool) for flag in case["attribution_outlier"].values()
        )


def test_annotation_annotates_every_case_including_the_non_reversals(tmp_path):
    annotated = synthetic_annotation(tmp_path)
    assert len(annotated["cases"]) == 2
    assert not any(case["head_order"]["reversal"] for case in annotated["cases"])


def test_a_case_that_is_no_reversal_never_reads_as_clearing_the_floor(tmp_path):
    # One discordant pair of three leaves tau positive, so the row is not a
    # reversal — and a `clears_noise_floor` computed without that condition
    # would report it as a misordering that beat the noise.
    annotated = synthetic_annotation(tmp_path)
    for case in annotated["cases"]:
        if not case["head_order"]["reversal"]:
            assert case["head_order"]["clears_noise_floor"] is False
    assert all(
        case["head_order"]["clears_noise_floor_rule"] == v3_report.PAIR_FLOOR_RULE
        for case in annotated["cases"]
    )


def test_annotation_does_not_read_the_rollout(tmp_path):
    annotated = synthetic_annotation(tmp_path)
    assert annotated["inputs"] == ["ism", "ig", "predictions", "labels", "noise_floor"]
    assert "rollout" in annotated["rollout_excluded"]
    assert "rollout.npz" not in json.dumps(annotated)


def test_annotation_claims_no_binding_and_is_finite(tmp_path):
    annotated = synthetic_annotation(tmp_path)
    assert_claims_no_binding(annotated)
    assert_finite(annotated)


def test_annotation_refuses_a_case_the_sweep_never_covered(tmp_path):
    inputs = synthetic_inputs(tmp_path)
    cases = write_cases(
        tmp_path / "v3_cases",
        [
            {
                "case_id": "GEN-ERRHI-01",
                "stratum": GENERALISATION,
                "criterion": "ERRHI",
                "row_index": 9999,
                "reporting_qualifier": "",
            }
        ],
    )
    with pytest.raises(ValueError, match="swept|intersection"):
        v3_report.annotate_cases(
            cases_json=cases,
            ism_npz=inputs["ism_npz"],
            ig_npz=inputs["ig_npz"],
            split_csv=inputs["split_csv"],
            predictions_csv=inputs["predictions_csv"],
            noise_floor=inputs["noise_floor"],
            split_checksum=None,
        )


def test_write_annotated_cases_round_trips(tmp_path):
    annotated = synthetic_annotation(tmp_path)
    path = v3_report.write_annotated_cases(tmp_path / "out", annotated)
    assert path.name == v3_report.ANNOTATED_NAME
    assert json.loads(path.read_text(encoding="utf-8")) == annotated


# --- the real artifacts ----------------------------------------------------

REAL_INPUTS = (SPLIT_CSV, PREDICTIONS_CSV, NOISE_FLOOR)
real_only = pytest.mark.skipif(
    not all(path.exists() for path in REAL_INPUTS),
    reason="the split and the prediction export are git-ignored",
)


@real_only
def test_real_reversal_report_reproduces_the_pinned_reversals():
    report = v3_report.build_reversal_report()
    population = report["population"]
    assert population["rows"] == INTERSECTION_ROWS
    assert population["reversal_rows"] == REVERSAL_ROWS
    assert report["strata"][PAM_LIBRARY]["reversal_rows"] == LIBRARY_REVERSALS
    # AC13's headline: v3 misorders heads essentially only where the heads
    # were measured within noise of each other.
    assert population["clearing_floor"]["rows"] == CLEARING_ROWS
    assert population["clearing_floor"]["pooled_rule_rows"] == CLEARING_ROWS
    assert report["floor"]["limit"]["groups"] == 79
    assert report["floor"]["limit"]["rows"] == 379
    # The library stratum is a third of the intersection and holds five sixths
    # of the reversals, which is the result rather than a rounding of it.
    assert report["strata"][PAM_LIBRARY]["rows"] * 3 < INTERSECTION_ROWS * 1.1
    assert LIBRARY_REVERSALS / REVERSAL_ROWS > 0.8


@pytest.mark.skipif(
    not (ISM_NPZ.exists() and IG_NPZ.exists() and SPLIT_CSV.exists()),
    reason="ism.npz, ig.npz and test.csv are git-ignored",
)
def test_real_validation_report_covers_the_frozen_ig_subset():
    report = v3_report.build_validation_report()
    assert report["population"]["rows"] == IG_SUBSET_ROWS
    for name in (GENERALISATION, PAM_LIBRARY):
        stratum = report["strata"][name]
        assert stratum["rows"] == IG_ROWS_PER_STRATUM
        for head in ANALYSED_HEADS:
            summary = stratum["by_head"][head][v3_report.MAGNITUDE]["per_row"]
            assert -1.0 <= summary["median"] <= 1.0
            assert summary["rows"] + summary["undefined_rows"] == stratum["rows"]


@pytest.mark.skipif(
    not (ISM_NPZ.exists() and IG_NPZ.exists() and CASES_JSON.exists()),
    reason="the arrays and the split are git-ignored",
)
def test_real_annotation_covers_every_case_without_rewriting_it():
    cases = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    annotated = v3_report.annotate_cases()
    assert len(annotated["cases"]) == len(cases["cases"]) == CASE_COUNT
    v3_report.assert_selection_unchanged(cases, annotated)
    assert annotated["points_at"]["sha256"] == v3_predict.sha256_of(CASES_JSON)
