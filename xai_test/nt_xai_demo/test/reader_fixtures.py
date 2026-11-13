"""Synthetic stand-ins for a prepared `data/` tree, built to catch misreads.

The shapes, keys and declared scales are the real ones; the values are small
and closed-form so a test can state the right answer. Three things are
deliberately adversarial, because each is a misread the real artifacts invite:

* the ISM head axis holds nine heads and the IG head axis holds three **in a
  different order**, so a reader that reuses one axis' position gets another
  head's numbers;
* the IG population is a subset of the ISM one and their row orders do not line
  up, so a reader that joins by position reads the wrong row;
* the split label columns are percent while predictions and deltas are
  fractions.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ISM_HEADS = (
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
ANALYSED_HEADS = ("SpCas9", "SpCas9-NG", "evoCas9")
#: Scrambled on purpose: the IG artifact's own order, which is not the ISM one.
IG_HEADS = ("evoCas9", "SpCas9", "SpCas9-NG")
TOKEN_NAMES = (
    "left_flank_plus_protospacer_5prime",
    "protospacer_6_11",
    "protospacer_12_17",
    "protospacer_18_23",
    "pam_plus_right_flank",
)
TOKEN_BOUNDARIES = ((0, 5), (6, 11), (12, 17), (18, 23), (24, 29))
POSITIONS = 30
BASES = ("A", "C", "G", "T")

#: Row 44 is appended, never inserted: the tests above address these by
#: position, and it exists so the generalisation stratum has an `ERRLO` case
#: (the demo's `A1`) beside its `ERRHI` and `HEADGAP` ones.
ISM_ROWS = (5, 11, 30, 44)
IG_ROWS = (11, 30)  # misaligned with ISM_ROWS on purpose
SPLIT_ROLE = {5: "test", 11: "test", 30: "val", 44: "test"}
#: Rows in the PAM library; the rest are non-library. With an NGG PAM (below)
#: the non-library rows are the generalisation stratum, as `v3_strata` cuts it.
PAM_LIBRARY_ROWS = frozenset({30})
#: The generalisation stratum's PAM: G at 25 and 26 (position 24 is the free N).
NGG_POSITIONS = (25, 26)
#: Percent, as the split files store it. 12.5 percent is 0.125 as a fraction.
#: Row 44 sits at the prediction, so its primary head's error is 0 - an
#: `ERRLO` case, and a value a percent/fraction slip cannot hide.
LABEL_PERCENT = {5: 12.5, 11: 40.0, 30: 1.25, 44: 10.0}
#: A blank label cell: a head that was not measured on that row. Real and common.
UNMEASURED = (30, "SpCas9-NG")
CLAIM_LIMIT = (
    "여기의 모든 수치는 모델 내부 정보 사용을 기술한다. 물리적 DNA/단백질 상호작용에 "
    "대한 주장이 아니다."
)

CAS12A_ROWS = (4, 9, 21)
CAS12A_POSITIONS = 34
CAS12A_LIMIT_TEXT = "TTTV/TTTT는 관측된 각 집단의 행 수를 분모로 사용합니다."
CAS12A_ROLLOUT_TEXT = "encoder token attention flow입니다. 출력 attribution이 아닙니다."


def reference_bases(rows: int, positions: int) -> np.ndarray:
    return np.array(
        [
            [(row * 7 + position * 3) % 4 for position in range(positions)]
            for row in range(rows)
        ],
        dtype=np.int8,
    )


def ism_delta(
    rows: int, positions: int, heads: int, reference: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Deltas whose per-head, per-row profile has a closed form.

    The reference slot stays exactly 0 and the three alternatives carry
    0.1/0.2/0.3 plus a per-head and per-row offset, so the mean over the
    alternatives is `expected_profile`. Signs alternate, so a profile that
    forgot the absolute value is visibly wrong.
    """
    if reference is None:
        reference = reference_bases(rows, positions)
    delta = np.zeros((rows, positions, len(BASES), heads), dtype=np.float32)
    for row in range(rows):
        for position in range(positions):
            alternatives = [
                b for b in range(len(BASES)) if b != reference[row, position]
            ]
            for order, base in enumerate(alternatives):
                for head in range(heads):
                    value = (
                        (head + 1) * 0.01
                        + (order + 1) * 0.1
                        + row * 0.001
                        + position * 0.0001
                    )
                    delta[row, position, base, head] = value * (-1) ** position
    return delta, reference


def expected_profile(row: int, position: int, head_index: int) -> float:
    """The mean |delta| over the three alternative bases, WT slot excluded."""
    return (head_index + 1) * 0.01 + 0.2 + row * 0.001 + position * 0.0001


#: Per-head replicate noise floors (median spread, fraction). Distinct on
#: purpose, and the pooled median below is none of them: a report that shows
#: the pooled value instead of the per-head ones is caught by value.
NOISE_FLOOR_MEDIAN = {"SpCas9": 0.0637, "SpCas9-NG": 0.0697, "evoCas9": 0.0099}
NOISE_FLOOR_POOLED_MEDIAN = 0.0531
#: A pair's floor is the larger of its two heads' own floors.
PAIR_FLOORS = {
    "SpCas9_vs_SpCas9-NG": 0.0697,
    "SpCas9_vs_evoCas9": 0.0637,
    "SpCas9-NG_vs_evoCas9": 0.0697,
}
#: The pam_library stratum's rows are PAM variants of 5 targets, so its
#: reversal rows are repeated target by target. Deliberately uneven: one target
#: carries far more of the pool's reversals than the others.
REVERSAL_PER_TARGET = (
    {
        "protospacer": "GTCGCTTCCTCTTCTCCTTC",
        "rows": 268,
        "reversal_rows": 18,
        "rows_clearing_floor": 0,
    },
    {
        "protospacer": "GTTCACGACGTCATGCTCAC",
        "rows": 264,
        "reversal_rows": 7,
        "rows_clearing_floor": 0,
    },
    {
        "protospacer": "GACCCAAGCCACGAACGAGC",
        "rows": 258,
        "reversal_rows": 45,
        "rows_clearing_floor": 0,
    },
    {
        "protospacer": "GGTCAGGCTCCATGTGCCGC",
        "rows": 258,
        "reversal_rows": 54,
        "rows_clearing_floor": 0,
    },
    {
        "protospacer": "GGCGCCCACACTCGTGCAGC",
        "rows": 239,
        "reversal_rows": 45,
        "rows_clearing_floor": 0,
    },
)
REVERSAL_STRATA = {
    # stratum: rows, targets, reversal rows, rows clearing the floor
    "generalisation": (895, 818, 13, 1),
    "pam_library": (1287, 5, 169, 0),
}


def noise_floor() -> dict:
    return {
        "spread_definition": "max - min of a repeated 30-mer's labels for one head, "
        "on the fraction scale.",
        "single_constant_note": "Use by_head, not pooled.",
        "limit_note": "The floor rests on the repeated 30-mers of one split.",
        "label_scale": "fraction",
        "groups": 79,
        "rows": 379,
        "heads": list(ANALYSED_HEADS),
        "by_head": {
            head: {"groups": 79, "median": median, "p90": median * 2}
            for head, median in NOISE_FLOOR_MEDIAN.items()
        },
        "pooled": {"measurements": 237, "median": NOISE_FLOOR_POOLED_MEDIAN},
        "head_floor_ratio": 7.06,
        "intersection_rows": 3851,
        "split_rows": 5186,
        "repeated_groups_in_split": 779,
        "repeated_rows_in_split": 2247,
    }


def _reversal_stratum(rows: int, targets: int, reversals: int, clearing: int) -> dict:
    return {
        "rows": rows,
        "distinct_protospacers": targets,
        "tau": {"rows": rows, "undefined_rows": 0, "median": 1.0},
        "tau_negative_rows": reversals + 1,
        "tied_rows": 2,
        "tied_rows_with_negative_tau": 1,
        "reversal_rows": reversals,
        "reversal_rate": reversals / rows,
        "inverted_pairs": {
            pair: {
                "rows": reversals - index,
                "floor": floor,
                "label_gap": {"rows": reversals - index, "median": 0.01 + index / 1000},
                "prediction_gap": {"rows": reversals - index, "median": 0.02},
                "rows_clearing_floor": min(clearing, 1),
            }
            for index, (pair, floor) in enumerate(PAIR_FLOORS.items())
        },
        # How far the inverted pairs sit from their own floor, in units of that
        # floor: under 1.0 the ordering was decided inside measurement noise.
        "gap_over_floor": {
            "rows": reversals * 2,
            "undefined_rows": 0,
            "median": 0.42 if clearing else 0.15,
        },
        "clearing_floor": {
            "rows": clearing,
            "rule": "every inverted pair's measured label gap exceeds that pair's floor",
            "pooled_rule_rows": clearing,
        },
    }


def head_order_reversal() -> dict:
    strata = {
        name: _reversal_stratum(*shape) for name, shape in REVERSAL_STRATA.items()
    }
    strata["pam_library"]["qualifier"] = "observed on 5 targets"
    strata["pam_library"]["qualifier_note"] = (
        "This stratum's rows are PAM variants of 5 targets, so its n is 5 targets "
        "rather than its row count, and that count is never added to the other "
        "stratum's."
    )
    strata["pam_library"]["per_target"] = list(REVERSAL_PER_TARGET)
    return {
        "model": "9-Hydra v3",
        "analysis": "head_order_reversal",
        "statistic": "kendall_tau_a",
        "heads": list(ANALYSED_HEADS),
        "reversal_definition": "A row whose predicted ordering of the analysed "
        "heads disagrees with their measured ordering.",
        "selection_pool": "forbidden",
        "selection_pool_note": "Head-order reversal is a reported result and not "
        "a case-selection pool.",
        "floor": {
            "source": "v3_noise_floor.json",
            "statistic": "median",
            "by_head": dict(NOISE_FLOOR_MEDIAN),
            "pair_floors": dict(PAIR_FLOORS),
            "pair_floor_rule": "the larger of its two heads' own floors",
            "pooled_median": NOISE_FLOOR_POOLED_MEDIAN,
            "pooled_applied": False,
            "gap_source": "measured_labels",
        },
        "label_scale": "fraction",
        "prediction_scale": "fraction",
        "population": {"name": "analysis_intersection", "rows": 3851},
        "pooling": "forbidden",
        "claim_limit": CLAIM_LIMIT,
        "strata": strata,
    }


#: The case table, keyed as the real one is: `<STRATUM>-<CRITERION>-<NN>`.
#: GEN-HEADGAP-02 is the one case inside the IG list (as on the real data);
#: GEN-ERRLO-01 and GEN-ERRHI-01 are outside it, so the demo's screen has both
#: the IG block and the "this row is not in the sample" line to print, in the
#: same proportion the real selection has them. LIB-ERRLO-01 sits on a `val`
#: row with no prediction and one unmeasured head - impossible on the real
#: data, and here so the split-role safeguard and the missing-prediction path
#: have a case; the demo offers no number for it, since it is not
#: generalisation. The selection ranks leave the `HEADGAP` letter at 2 with no
#: 1 or 3 beside it, so a number range printed as a span would be wrong.
SPCAS9_CASES = (
    ("GEN-ERRLO-01", "generalisation", "ERRLO", 44),
    ("GEN-ERRHI-01", "generalisation", "ERRHI", 5),
    ("GEN-HEADGAP-02", "generalisation", "HEADGAP", 11),
    ("LIB-ERRLO-01", "pam_library", "ERRLO", 30),
)
CASE_STRATA = {
    "generalisation": {"rows": 895, "distinct_protospacers": 818},
    "pam_library": {"rows": 1287, "distinct_protospacers": 5},
}
LIB_QUALIFIER = "observed in one of 5 targets"


#: Labels step twice as fast across the head axis as predictions do, so each
#: head's absolute error is its own: a screen that shows one head's error under
#: another head's name is caught by value, and so is a comparison head chosen
#: by anything but the stored errors.
LABEL_HEAD_STEP = 2


def label_fraction(row: int, head: str) -> float | None:
    """What the reader must return: the split file's percent, divided by 100."""
    if (row, head) == UNMEASURED:
        return None
    return (LABEL_PERCENT[row] + ISM_HEADS.index(head) * LABEL_HEAD_STEP) / 100.0


def prediction_fraction(row: int, head: str) -> float | None:
    if SPLIT_ROLE[row] != "test":
        return None
    return 0.1 + ISM_HEADS.index(head) * 0.01


def spcas9_cases() -> dict:
    cases = []
    for case_id, stratum, criterion, row in SPCAS9_CASES:
        case: dict = {
            "case_id": case_id,
            "stratum": stratum,
            "criterion": criterion,
            "selection_rank": int(case_id[-1]),
            "row_index": row,
            "reporting_qualifier": LIB_QUALIFIER if stratum == "pam_library" else "",
        }
        errors = {}
        for head in ANALYSED_HEADS:
            label = label_fraction(row, head)
            prediction = prediction_fraction(row, head)
            case[f"label_{head}"] = label
            case[f"pred_{head}"] = prediction
            error = (
                None if label is None or prediction is None else abs(label - prediction)
            )
            case[f"abs_error_{head}"] = error
            if error is not None:
                errors[head] = error
        # The selection's own recorded gap: max - min of the analysed heads'
        # absolute errors. It is not the same quantity as the distance from the
        # primary head to one comparison head, and the screen must not print
        # one under the other's name.
        case["head_gap"] = (
            max(errors.values()) - min(errors.values()) if errors else None
        )
        cases.append(case)
    return {
        "primary_head": "SpCas9",
        "heads": list(ANALYSED_HEADS),
        "criteria": ["ERRLO", "ERRHI", "HEADGAP"],
        "label_scale": "fraction",
        "prediction_scale": "fraction",
        "strata": CASE_STRATA,
        "cases": cases,
    }


#: Per-head ISM token profiles recorded for a case. Distinct per head so a
#: view that shows another head's numbers is caught by value.
CASE_ISM_TOKENS = {
    "SpCas9": [0.019, 0.022, 0.026, 0.036, 0.152],
    "SpCas9-NG": [0.031, 0.020, 0.025, 0.037, 0.083],
    "evoCas9": [0.003, 0.0046, 0.0051, 0.0027, 0.0036],
}
CASE_IG_TOKENS = {
    "SpCas9": [-0.0409, -0.129, 0.154, 0.124, 2.088],
    "SpCas9-NG": [0.376, -0.195, 0.694, 0.110, 0.061],
    "evoCas9": [0.668, -0.319, 0.572, -0.0924, 0.0296],
}
CASE_PAM_SHARE = {"SpCas9": 5.34, "SpCas9-NG": 2.81, "evoCas9": 1.31}
CASE_PAM_PERCENTILE = {"SpCas9": 46.6, "SpCas9-NG": 83.6, "evoCas9": 38.2}


def spcas9_annotation(case: tuple[str, str, str, int]) -> dict:
    case_id, stratum, criterion, row = case
    in_ig = row in IG_ROWS
    return {
        "case_id": case_id,
        "row_index": row,
        "stratum": stratum,
        "criterion": criterion,
        "reporting_qualifier": LIB_QUALIFIER if stratum == "pam_library" else "",
        "ism": {
            "token_profile": {h: list(v) for h, v in CASE_ISM_TOKENS.items()},
            "top_position": {"SpCas9": 26, "SpCas9-NG": 25, "evoCas9": 15},
            "top_token": {
                "SpCas9": "pam_plus_right_flank",
                "SpCas9-NG": "pam_plus_right_flank",
                "evoCas9": "protospacer_12_17",
            },
            "pam_share": dict(CASE_PAM_SHARE),
            "pam_share_percentile_in_stratum": dict(CASE_PAM_PERCENTILE),
        },
        "ig": (
            {
                "token_attribution": {h: list(v) for h, v in CASE_IG_TOKENS.items()},
                "top_token": {
                    "SpCas9": "pam_plus_right_flank",
                    "SpCas9-NG": "protospacer_12_17",
                    "evoCas9": "left_flank_plus_protospacer_5prime",
                },
            }
            if in_ig
            else None
        ),
        "head_order": {
            "tau": -1 / 3,
            "reversal": True,
            "tied_pair": False,
            "clears_noise_floor": False,
            "clears_noise_floor_rule": "every inverted pair's measured label gap "
            "exceeds that pair's floor",
            "inverted_pairs": [
                {
                    "pair": "SpCas9_vs_SpCas9-NG",
                    "label_gap": 0.00885,
                    "floor": PAIR_FLOORS["SpCas9_vs_SpCas9-NG"],
                    "clears_floor": False,
                }
            ],
        },
        "attribution_outlier": {head: False for head in ANALYSED_HEADS},
    }


#: Recorded IG-ISM agreement per stratum. Distinct on purpose, so a figure
#: that pooled the two strata, or read one under the other's name, is caught.
VALIDATION_MEDIAN = {"generalisation": 0.7, "pam_library": 0.45}
#: Added per head, so each head's recorded median is its own value. On the
#: real data the three differ widely (0.7 / 0.55 / 0.0 in `generalisation`); a
#: fixture that gave them one value could not tell a report that shows the
#: primary head's figure under another head's name.
VALIDATION_HEAD_OFFSET = {"SpCas9": 0.0, "SpCas9-NG": -0.15, "evoCas9": -0.4}


def validation_median(stratum: str, head: str) -> float:
    return round(VALIDATION_MEDIAN[stratum] + VALIDATION_HEAD_OFFSET[head], 2)


VALIDATION_MEAN_PROFILE = {"generalisation": 1.0, "pam_library": 0.9}
#: The IG subset row each stratum holds (see `ig_subset.json` below).
STRATUM_ROW = {"generalisation": 11, "pam_library": 30}


def spcas9_stratum_profile(stratum: str, head: str) -> list[float]:
    """The stratum's mean ISM position profile, WT slot excluded, in closed form."""
    row = ISM_ROWS.index(STRATUM_ROW[stratum])
    head_index = ISM_HEADS.index(head)
    return [
        expected_profile(row, position, head_index) for position in range(POSITIONS)
    ]


def spcas9_recorded_ism_tokens(stratum: str, head: str) -> list[float]:
    profile = spcas9_stratum_profile(stratum, head)
    return [
        sum(profile[start : end + 1]) / (end - start + 1)
        for start, end in TOKEN_BOUNDARIES
    ]


def spcas9_reference() -> np.ndarray:
    """The ISM reference bases, with every non-library row given an NGG PAM.

    Only the bases at 25-26 change; `ism_delta` keys its values to the
    alternatives' order, not to which base is WT, so no profile moves.
    """
    reference = reference_bases(len(ISM_ROWS), POSITIONS)
    for at, row in enumerate(ISM_ROWS):
        if row not in PAM_LIBRARY_ROWS:
            reference[at, list(NGG_POSITIONS)] = BASES.index("G")
    return reference


def spcas9_sequence(row: int) -> str:
    """The 30bp window a row's split record spells, the same as its ISM reference."""
    bases = spcas9_reference()[ISM_ROWS.index(row)]
    return "".join(BASES[int(base)] for base in bases)


def build_spcas9(root: Path, *, label_scale: str = "percent") -> Path:
    spcas9 = root / "spcas9"
    (spcas9 / "published_split").mkdir(parents=True, exist_ok=True)
    for name in (
        "v3_ism",
        "v3_ig",
        "v3_ig_subset",
        "v3_rollout",
        "v3_cases",
        "v3_report",
        "v3_head_agreement",
        "v3_testset_predictions",
    ):
        (spcas9 / name).mkdir(parents=True, exist_ok=True)

    def write_json(relative: str, payload: object) -> None:
        (spcas9 / relative).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    delta, reference = ism_delta(
        len(ISM_ROWS), POSITIONS, len(ISM_HEADS), spcas9_reference()
    )
    wt_prediction = np.array(
        [
            [0.5 + head * 0.01 + row * 0.1 for head in range(len(ISM_HEADS))]
            for row in range(len(ISM_ROWS))
        ],
        dtype=np.float32,
    )
    np.savez(
        spcas9 / "v3_ism/ism.npz",
        delta=delta,
        wt_prediction=wt_prediction,
        reference_base=reference,
        row_index=np.array(ISM_ROWS, dtype=np.int64),
    )
    write_json(
        "v3_ism/ism.sidecar.json",
        {
            "model": "9-Hydra v3",
            "analysis": "in_silico_saturation_mutagenesis",
            "population": "analysis_intersection",
            "rows": len(ISM_ROWS),
            "positions": POSITIONS,
            "bases": list(BASES),
            "heads": list(ISM_HEADS),
            "analysed_heads": list(ANALYSED_HEADS),
            "delta_scale": "fraction",
            "nan_policy": "forbidden",
        },
    )

    attribution = np.array(
        [
            [
                [
                    (row + 1) * 10 + head + token * 0.1
                    for token in range(len(TOKEN_NAMES))
                ]
                for head in range(len(IG_HEADS))
            ]
            for row in range(len(IG_ROWS))
        ],
        dtype=np.float32,
    )
    np.savez(
        spcas9 / "v3_ig/ig.npz",
        attribution=attribution,
        completeness_residual=np.full((len(IG_ROWS), len(IG_HEADS)), 0.001),
        special_attribution=np.full((len(IG_ROWS), len(IG_HEADS)), 0.01),
        row_index=np.array(IG_ROWS, dtype=np.int64),
    )
    write_json(
        "v3_ig/ig.sidecar.json",
        {
            "model": "9-Hydra v3",
            "analysis": "integrated_gradients",
            "rows": len(IG_ROWS),
            "heads": list(IG_HEADS),
            "attribution_point": "encoder_input_embedding",
            "baseline": "zero_embedding",
            "token_resolution": 5,
            "token_names": list(TOKEN_NAMES),
            "token_boundaries": [list(pair) for pair in TOKEN_BOUNDARIES],
        },
    )
    write_json(
        "v3_ig_subset/ig_subset.json",
        {
            "rows": len(IG_ROWS),
            "rows_per_stratum": 1,
            "primary_head": "SpCas9",
            "heads": list(IG_HEADS),
            "label_scale": "fraction",
            "prediction_scale": "fraction",
            "entries": [
                {"row_index": 11, "stratum": "generalisation", "cell": "gen-errq1"},
                {"row_index": 30, "stratum": "pam_library", "cell": "ngg-errq4"},
            ],
        },
    )

    np.savez(
        spcas9 / "v3_rollout/rollout.npz",
        rollout=np.arange(len(ISM_ROWS) * 25, dtype=np.float32).reshape(
            len(ISM_ROWS), 5, 5
        ),
        row_index=np.array(ISM_ROWS, dtype=np.int64),
    )

    write_json(
        "v3_report/ig_ism_validation.json",
        {
            "statistic": "spearman_rank_correlation",
            "heads": list(ANALYSED_HEADS),
            "token_names": list(TOKEN_NAMES),
            "token_boundaries": [list(pair) for pair in TOKEN_BOUNDARIES],
            "mismatch_note": "낮은 상관은 토큰 수준 attribution의 한계이며 버그가 아니다.",
            "population": {
                "name": "ig_subset",
                "rows": len(IG_ROWS),
                "swept_rows": len(ISM_ROWS),
            },
            "pooling": "forbidden",
            "claim_limit": CLAIM_LIMIT,
            "strata": {
                stratum: {
                    "rows": 1,
                    "distinct_protospacers": 1,
                    "by_head": {
                        head: {
                            "magnitude": {
                                "agreement_sign": 1,
                                "per_row": {
                                    "rows": 1,
                                    "undefined_rows": 0,
                                    "median": validation_median(stratum, head),
                                    "q1": validation_median(stratum, head) - 0.3,
                                    "q3": validation_median(stratum, head) + 0.2,
                                },
                                "mean_profile": VALIDATION_MEAN_PROFILE[stratum],
                                "mean_token_profile": {
                                    # The recorded ISM token profile is the
                                    # closed-form one, so the view's own
                                    # position profile can be checked against it.
                                    "ism": spcas9_recorded_ism_tokens(stratum, head),
                                    "ig": [0.14, 0.15, 0.17, 0.27, 2.02],
                                },
                            }
                        }
                        for head in ANALYSED_HEADS
                    },
                }
                for stratum in ("generalisation", "pam_library")
            },
        },
    )
    write_json("v3_report/head_order_reversal.json", head_order_reversal())
    write_json(
        "v3_head_agreement/head_agreement.json",
        {
            "population": "analysis_intersection",
            "rows": len(ISM_ROWS),
            "pooling": "forbidden",
        },
    )
    write_json("v3_noise_floor.json", noise_floor())
    write_json("v3_cases/cases.json", spcas9_cases())
    write_json(
        "v3_cases/cases_annotated.json",
        {
            "claim_limit": CLAIM_LIMIT,
            "heads": list(ANALYSED_HEADS),
            "token_names": list(TOKEN_NAMES),
            "cases": [spcas9_annotation(case) for case in SPCAS9_CASES],
        },
    )

    heads_header = list(ISM_HEADS)
    for split in ("train", "val", "test"):
        rows = [row for row, role in SPLIT_ROLE.items() if role == split]
        with open(
            spcas9 / f"published_split/{split}.csv", "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "row_index",
                    "sequence",
                    "protospacer",
                    "in_pam_library",
                    "trna_feature",
                    "use",
                    "split",
                    *heads_header,
                ]
            )
            for row in rows:
                labels = [
                    ""
                    if (row, head) == UNMEASURED
                    else LABEL_PERCENT[row] + index * LABEL_HEAD_STEP
                    for index, head in enumerate(heads_header)
                ]
                writer.writerow(
                    [
                        row,
                        spcas9_sequence(row),
                        spcas9_sequence(row)[4:24],
                        str(row in PAM_LIBRARY_ROWS),
                        0,
                        split.capitalize(),
                        split,
                    ]
                    + labels
                )

    test_rows = [row for row, role in SPLIT_ROLE.items() if role == "test"]
    with open(
        spcas9 / "v3_testset_predictions/predictions.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row_index",
                "protospacer",
                "in_pam_library",
                "trna_feature",
                *[f"pred_{head}" for head in heads_header],
            ]
        )
        for row in test_rows:
            writer.writerow(
                [row, "A" * 20, "True", 0]
                + [0.1 + index * 0.01 for index in range(len(heads_header))]
            )
    write_json(
        "v3_testset_predictions/predictions_checksum.json",
        {
            "model": "9-Hydra v3",
            "heads": heads_header,
            "rows": len(test_rows),
            "prediction_scale": "fraction",
            "label_scale_in_split": label_scale,
        },
    )
    return spcas9


#: The described cases, identified as the source data does: source_row + role.
#: Row 9's IG failed, so the demo has an absence path with the artifact's own
#: recorded reason rather than a gap on screen.
CAS12A_CASES = (
    (4, "out_of_range", 1, "raw < 0 or raw > 1"),
    (9, "low_error", 1, "|error| ascending"),
)
#: Each role's pool, as the real `cases.json` records it: `eligible` rows meet
#: the criterion, `available` is what was left after the earlier categories
#: took their rows (`exclude_previous`). They differ for `low_error` on the
#: real data (375 eligible, 372 available), and they differ here too - the
#: denominator a case was actually drawn from is `available`, and a fixture
#: that made the two equal could not tell a reader picking the wrong one.
CAS12A_CATEGORY_POOLS = {
    # role: available (drawn from), eligible (met the criterion)
    "out_of_range": (16, 16),
    "high_error": (375, 375),
    "low_error": (372, 375),
}
#: `layout.json` as `prepare` lifts it from the IG shards. The singles sit
#: first and the PAM is not the real `[4, 8]`, so a reader that assumed the
#: real 6-mer split or PAM instead of reading this file reads it wrong.
CAS12A_LAYOUT = {
    "coordinates": "0-based half-open",
    "dna_spans": [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 10],
        [10, 16],
        [16, 22],
        [22, 28],
        [28, 34],
    ],
    "layout": {
        "coordinate_system": "0-based half-open",
        "left_flank": [0, 5],
        "pam": [5, 9],
        "protospacer": [9, 31],
        "right_flank": [31, 34],
    },
}
CAS12A_IG_FAILED_ROW = 9
CAS12A_IG_FAILURE = "completeness residual above limit"
#: Per-token mean |signed mass| for a case: 9 DNA blocks, as the real one has.
CAS12A_CASE_TOKEN_SENSITIVITY = (
    0.035,
    0.011,
    0.023,
    0.010,
    0.029,
    0.003,
    0.011,
    0.004,
    0.003,
)
CAS12A_CASE_PAM_SHARE = 0.1558
CAS12A_CASE_TOP_POSITION = 5
CAS12A_ROLLOUT_INTERPRETATION = (
    "attention flow; not output attribution or selective token-pair interaction"
)


def cas12a_annotation(row: int, role: str, reason: str) -> dict:
    """One annotated case, with each method's own `status` and `failure`."""
    label, raw, clipped, stratum = (
        0.0 if row == 4 else 0.3,
        -0.05 if row == 4 else 0.35,
        0.0 if row == 4 else 0.35,
        "TTTT" if row == 4 else "TTTV",
    )
    ig_failed = row == CAS12A_IG_FAILED_ROW
    return {
        "source_row": row,
        "role": role,
        "roles": ["full_population", role],
        "pam_stratum": stratum,
        "rank_in_category": 1,
        "selection_reason": reason,
        "label": label,
        "raw_prediction": raw,
        "clipping_distance": abs(raw - clipped),
        "raw_absolute_error": abs(raw - label),
        "methods": {
            "ism": {
                "status": "available",
                "failure": None,
                "summary": {
                    "source_row": row,
                    "pam_share": CAS12A_CASE_PAM_SHARE,
                    "pam_share_undefined_reason": None,
                    "top_position": CAS12A_CASE_TOP_POSITION,
                    "token_mean_sensitivity": list(CAS12A_CASE_TOKEN_SENSITIVITY),
                },
            },
            "ig": (
                {
                    "status": "failed",
                    "failure": CAS12A_IG_FAILURE,
                    "summary": None,
                }
                if ig_failed
                else {
                    "status": "available",
                    "failure": None,
                    "summary": {
                        "dna_signed_mass": [0.1 * (i + 1) for i in range(9)],
                        "total_signed_mass": -0.7883,
                        "quality": {
                            "completeness_pass": True,
                            "convergence_pass": True,
                            "absolute_residual": 2.5e-06,
                            "completeness_limit": 0.0395,
                        },
                        "steps": 400,
                    },
                }
            ),
            "rollout": {
                "status": "available",
                "failure": None,
                "summary": {
                    "interpretation": CAS12A_ROLLOUT_INTERPRETATION,
                    "sequence_query_diagnostic": {
                        "status": "near_identical_queries",
                        "near_identical_fraction": 1.0,
                    },
                },
            },
        },
    }


CAS12A_AGREEMENT_SECTION = "설명과 IG–ISM 일치도"
CAS12A_AGREEMENT_TEXT = (
    "주 통계는 같은 길이 t1–t5의 6-mer, PAM-breaking 포함, density Spearman입니다.\n"
    "낮은 상관은 설명 정확도 실패가 아닙니다."
)
CAS12A_SVGS = (
    ("ISM 위치별 평균", "position_profiles.svg"),
    ("IG 토큰별 평균", "token_profiles.svg"),
    ("IG–ISM 분포", "agreement_distribution.svg"),
)
#: Each PAM population's per-row agreement median and its mean-profile rho.
#: Distinct per population, so a pooled figure read under a group's name shows.
CAS12A_AGREEMENT = {
    # population: (rows, median, mean-profile rho)
    "all": (3, 0.4, 0.5),
    "TTTV": (2, 0.3, 0.6),
    "TTTT": (1, 0.6, 0.9),
}
CAS12A_PRIMARY = "five_6mer/all_nonWT/density"
#: Where each population's ISM position profile peaks: inside the PAM (4-7).
CAS12A_PROFILE_PEAK = {"all": 7, "TTTV": 7, "TTTT": 6}


def cas12a_position_profile(stratum: str) -> list[float]:
    profile = [0.01 * (index % 5 + 1) for index in range(CAS12A_POSITIONS)]
    profile[CAS12A_PROFILE_PEAK[stratum]] = 0.24
    return profile


def cas12a_agreement() -> dict:
    """The sliced agreement: populations / group_diagnostics / protocol, as real.

    The non-primary variants carry different medians on purpose: a reader that
    took the first variant it found instead of the protocol's `primary` path
    shows the wrong number.
    """

    def variant(rows: int, median: float, rho: float) -> dict:
        return {
            "mean_profile": {
                "correlation": {
                    "n_total": 5,
                    "n_valid": 5,
                    "reason": None,
                    "rho": rho,
                    "token_ids": [1, 2, 3, 4, 5],
                },
                "profile": [
                    {
                        "mean_ig": 0.02 * token,
                        "mean_ism": 0.05 * token,
                        "n_paired": rows,
                        "n_total": rows,
                        "token_id": token,
                    }
                    for token in range(1, 6)
                ],
            },
            "sequences": {
                "distribution": [median] * rows,
                "excluded_reasons": {},
                "median": median,
                "n_total": rows,
                "n_valid": rows,
                "reason": None,
            },
        }

    tokens_p, support_p, kind_p = CAS12A_PRIMARY.split("/")
    populations = {}
    for name, (rows, median, rho) in CAS12A_AGREEMENT.items():
        versions = {
            support: {
                tokens: {
                    kind: (
                        variant(rows, median, rho)
                        if (tokens, support, kind) == (tokens_p, support_p, kind_p)
                        else variant(rows, -0.05, -0.2)
                    )
                    for kind in ("density", "mass")
                }
                for tokens in ("five_6mer", "nine_token")
            }
            for support in ("all_nonWT", "exclude_pam_breaking")
        }
        populations[name] = {
            "n_available": rows,
            "n_total": rows,
            "primary": dict(variant(rows, median, rho)["sequences"]),
            "source_ids": list(CAS12A_ROWS[:rows]),
            "versions": versions,
        }
    return {
        "populations": populations,
        "group_diagnostics": {},
        "protocol": {
            "primary": CAS12A_PRIMARY,
            "ism": "mean eligible alternative absolute raw delta, then mean valid bases",
            "output_unit": "native raw fraction",
            "version": "cas12a-ig-ism-agreement-v1",
        },
    }


def build_cas12a(
    root: Path, *, output_unit: str = "fraction", layout: dict | None = None
) -> Path:
    cas12a = root / "cas12a"
    cas12a.mkdir(parents=True, exist_ok=True)

    def write_json(name: str, payload: object) -> None:
        (cas12a / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    rows = len(CAS12A_ROWS)
    delta, reference = ism_delta(rows, CAS12A_POSITIONS, 1)
    raw_delta = delta[:, :, :, 0].astype(np.float64)
    wt_mask = np.eye(len(BASES), dtype=bool)[reference]
    non_wt_mask = ~wt_mask
    base_valid_mask = np.ones_like(non_wt_mask)
    # One position of the first row has a single invalid alternative: its
    # divisor is 2, not 3, and a reader that hardcodes 3 reads it low.
    invalid_base = next(b for b in range(len(BASES)) if b != reference[0, 0])
    base_valid_mask[0, 0, invalid_base] = False
    raw_delta[0, 0, invalid_base] = 0.0
    denominator = (non_wt_mask & base_valid_mask).sum(axis=-1).astype(np.int64)

    np.savez(
        cas12a / "ism.npz",
        source_row=np.array(CAS12A_ROWS, dtype=np.int64),
        accepted=np.ones(rows, dtype=bool),
        wt_prediction=np.linspace(0.4, 0.6, rows).astype(np.float32),
        prediction=np.zeros((rows, CAS12A_POSITIONS, len(BASES)), dtype=np.float32),
        reference_base=reference,
        wt_mask=wt_mask,
        non_wt_mask=non_wt_mask,
        base_valid_mask=base_valid_mask,
        raw_delta=raw_delta,
        clipped_delta=raw_delta,
        relative_delta=raw_delta,
        denominator=denominator,
    )
    np.savez(
        cas12a / "ig.npz",
        source_row=np.array(CAS12A_ROWS, dtype=np.int64),
        accepted=np.ones(rows, dtype=bool),
        signed_mass=np.arange(rows * 12, dtype=np.float64).reshape(rows, 12),
    )
    np.savez(
        cas12a / "rollout.npz",
        source_row=np.array(CAS12A_ROWS, dtype=np.int64),
        accepted=np.ones(rows, dtype=bool),
        full=np.zeros((rows, 10, 10)),
        sequence=np.arange(rows * 81, dtype=np.float64).reshape(rows, 9, 9),
    )

    with open(cas12a / "predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_row",
                "sequence",
                "label",
                "raw_prediction",
                "clipped_prediction",
                "pam_stratum",
                "raw_signed_error",
                "raw_absolute_error",
                "raw_squared_error",
                "clipped_signed_error",
                "clipped_absolute_error",
                "clipped_squared_error",
                "output_unit",
            ]
        )
        case_rows = {row: reason for row, _, _, reason in CAS12A_CASES}
        for index, row in enumerate(CAS12A_ROWS):
            # A described case's stored label and prediction are its row's, as
            # on the real data, where cases.csv and predictions.csv agree.
            if row in case_rows:
                case = cas12a_annotation(row, "", case_rows[row])
                label, raw = case["label"], case["raw_prediction"]
                clipped, stratum = min(max(raw, 0.0), 1.0), case["pam_stratum"]
                error = case["raw_absolute_error"]
            else:
                label, raw, clipped = (
                    0.2 + index * 0.1,
                    0.25 + index * 0.1,
                    0.25 + index * 0.1,
                )
                stratum = "TTTV" if index else "TTTT"
                error = 0.05
            writer.writerow(
                [
                    row,
                    # The row's own reference bases, as on the real data: the
                    # reader checks the two agree, which is a row join check.
                    "".join(BASES[base] for base in reference[index]),
                    label,
                    raw,
                    clipped,
                    stratum,
                    0.05,
                    error,
                    0.0025,
                    0.05,
                    0.05,
                    0.0025,
                    output_unit,
                ]
            )
    with open(cas12a / "cases.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_row",
                "raw_prediction",
                "label",
                "raw_absolute_error",
                "clipping_distance",
                "pam_stratum",
                "role",
                "rank_in_category",
                "selection_reason",
            ]
        )
        for row, role, rank, reason in CAS12A_CASES:
            annotation = cas12a_annotation(row, role, reason)
            writer.writerow(
                [
                    row,
                    annotation["raw_prediction"],
                    annotation["label"],
                    annotation["raw_absolute_error"],
                    annotation["clipping_distance"],
                    annotation["pam_stratum"],
                    role,
                    rank,
                    reason,
                ]
            )

    for name, payload in {
        "metrics.json": {
            stratum: {
                clipping: {
                    "count": count,
                    "mae": 0.17,
                    "spearman": spearman,
                    "pearson": 0.7,
                }
                for clipping in ("raw", "clipped")
            }
            for stratum, count, spearman in (
                ("all", 3, 0.7007),
                ("TTTV", 2, 0.5423),
                ("TTTT", 1, 0.2995),
            )
        },
        "position_profiles.json": {
            stratum: {
                "mean_abs_raw_delta": cas12a_position_profile(stratum),
                "n_available": count,
                "requested": count,
            }
            for stratum, count in (("all", 3), ("TTTV", 2), ("TTTT", 1))
        },
        "token_profiles.json": {
            stratum: {
                "mean_abs_signed_mass": [0.1] * 9,
                "n_accepted": count,
                "requested": count,
            }
            for stratum, count in (("all", 3), ("TTTV", 2), ("TTTT", 1))
        },
        "rollout_summary.json": {
            stratum: {
                "mean_removed_cls_mass": [0.17] * 9,
                "n_accepted": count,
                "near_identical_query_count": count,
                "requested": count,
            }
            for stratum, count in (("all", 3), ("TTTV", 2), ("TTTT", 1))
        },
        "agreement.sliced.json": cas12a_agreement(),
        "layout.json": CAS12A_LAYOUT if layout is None else layout,
        "summary.json": {
            "complete": True,
            "population_count": 3,
            "partition": "test",
            "ig_mode": "full",
            # Each coverage entry is a list of row IDs, as the real
            # summary.json stores them: the count is the list's length.
            "coverage": {
                method: {
                    "available": list(CAS12A_ROWS),
                    "failed": [],
                    "not_computed": [],
                    "requested": list(CAS12A_ROWS),
                }
                for method in ("prediction", "ism", "ig", "rollout")
            },
        },
        "cases.json": {
            "cases": [
                cas12a_annotation(row, role, reason)
                for row, role, _, reason in CAS12A_CASES
            ],
            "categories": {
                role: {
                    "available": available,
                    "eligible": eligible,
                    "selected": sum(
                        1 for _, case_role, *_ in CAS12A_CASES if case_role == role
                    ),
                    "unfilled": 0,
                }
                for role, (available, eligible) in CAS12A_CATEGORY_POOLS.items()
            },
            "protocol": {
                "category_order": list(CAS12A_CATEGORY_POOLS),
                "quantile_population": "all Test raw absolute errors before exclusions",
            },
        },
        "cases_annotated.json": {
            "cases": [
                cas12a_annotation(row, role, reason)
                for row, role, _, reason in CAS12A_CASES
            ],
            "selection": {},
        },
    }.items():
        write_json(name, payload)

    (cas12a / "report.ko.md").write_text(
        "# Cas12a Test 결과\n\n"
        "## 성능\n\n본문.\n\n"
        f"## {CAS12A_AGREEMENT_SECTION}\n\n{CAS12A_AGREEMENT_TEXT}\n\n"
        + "".join(f"![{label}]({name})\n\n" for label, name in CAS12A_SVGS)
        + f"## rollout 해석\n\n{CAS12A_ROLLOUT_TEXT}\n\n"
        f"## 범위와 한계\n\n{CAS12A_LIMIT_TEXT}\n",
        encoding="utf-8",
    )
    for _, name in CAS12A_SVGS:
        (cas12a / name).write_text("<svg/>", encoding="utf-8")
    return cas12a


#: A storage order per `.npz` in which no file keeps the default order's
#: position for row 9 (the shown case) and no two files agree.
CAS12A_SHUFFLED_ORDERS = {
    "ism.npz": (CAS12A_ROWS[2], CAS12A_ROWS[0], CAS12A_ROWS[1]),
    "ig.npz": (CAS12A_ROWS[1], CAS12A_ROWS[2], CAS12A_ROWS[0]),
    "rollout.npz": (CAS12A_ROWS[1], CAS12A_ROWS[0], CAS12A_ROWS[2]),
}


def reorder_cas12a_arrays(cas12a: Path, orders: dict[str, Sequence[int]]) -> None:
    """Store each `.npz`'s rows in its own `source_row` order.

    The real ISM/IG/rollout arrays happen to share one row order, and so do
    the default fixture's, so a lookup by array position instead of by
    `source_row` would pass. Giving each file a different order makes that
    misjoin print another row's numbers.
    """
    for name, rows in orders.items():
        path = cas12a / name
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        index = {int(row): i for i, row in enumerate(arrays["source_row"])}
        take = np.array([index[int(row)] for row in rows], dtype=np.int64)
        np.savez(path, **{key: value[take] for key, value in arrays.items()})


def drop_alternatives(cas12a: Path, row: int, position: int) -> None:
    """Mark every alternative base at one ISM position invalid.

    The raw deltas are left nonzero there on purpose: a profile that ignored
    `base_valid_mask` would print them, and one that kept the WT slot in the
    max would print 0 - only "no valid alternative" is right.
    """
    path = cas12a / "ism.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    at = int(np.flatnonzero(arrays["source_row"] == row)[0])
    # Only the alternatives: the WT slot stays valid, as on the real data,
    # so a max that forgot to exclude it would pick up its 0.
    arrays["base_valid_mask"][at, position] = arrays["wt_mask"][at, position]
    arrays["denominator"][at, position] = 0
    np.savez(path, **arrays)


def copy_sequence(cas12a: Path, source: int, target: int) -> None:
    """Give `target` the same 34bp window as `source`, as a replicate row has.

    Both places the sequence is spelled change together - `predictions.csv`
    and the ISM `reference_base` (with its WT mask, a zero WT delta and the
    denominator that follows) - so the reader's row-join check still holds for
    the target row.
    """
    path = cas12a / "ism.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    rows = arrays["source_row"]
    at, of = (int(np.flatnonzero(rows == row)[0]) for row in (target, source))
    reference = arrays["reference_base"][of].copy()
    wt_mask = np.eye(len(BASES), dtype=bool)[reference]
    arrays["reference_base"][at] = reference
    arrays["wt_mask"][at] = wt_mask
    arrays["non_wt_mask"][at] = ~wt_mask
    arrays["raw_delta"][at] = np.where(wt_mask, 0.0, arrays["raw_delta"][at])
    arrays["denominator"][at] = (~wt_mask & arrays["base_valid_mask"][at]).sum(axis=-1)
    np.savez(path, **arrays)

    predictions = cas12a / "predictions.csv"
    with open(predictions, newline="", encoding="utf-8") as handle:
        records = list(csv.reader(handle))
    column = records[0].index("sequence")
    spelled = "".join(BASES[int(base)] for base in reference)
    for record in records[1:]:
        if int(record[0]) == target:
            record[column] = spelled
    with open(predictions, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(records)


def copy_spcas9_sequence(spcas9: Path, source: int, target: int) -> None:
    """Give `target` the same 30bp window as `source`, as a replicate row has.

    Both places the sequence is spelled change together - the split file and
    the ISM `reference_base`. Where the WT base moves, its slot's delta swaps
    with the new WT slot's, so the WT slot keeps its true zero and the three
    alternatives keep the same values: the row's profiles do not move.
    """
    path = spcas9 / "v3_ism/ism.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    rows = arrays["row_index"]
    at, of = (int(np.flatnonzero(rows == row)[0]) for row in (target, source))
    old = arrays["reference_base"][at].copy()
    new = arrays["reference_base"][of].copy()
    for position in np.flatnonzero(old != new):
        was, now = int(old[position]), int(new[position])
        slots = arrays["delta"][at, position]
        slots[[was, now]] = slots[[now, was]]
    arrays["reference_base"][at] = new
    np.savez(path, **arrays)

    split = spcas9 / f"published_split/{SPLIT_ROLE[target]}.csv"
    with open(split, newline="", encoding="utf-8") as handle:
        records = list(csv.reader(handle))
    columns = records[0]
    spelled = "".join(BASES[int(base)] for base in new)
    for record in records[1:]:
        if int(record[0]) == target:
            record[columns.index("sequence")] = spelled
            record[columns.index("protospacer")] = spelled[4:24]
    with open(split, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(records)


def build_data_dir(root: Path, **kwargs) -> Path:
    """A full synthetic `data/` directory: both nucleases, no manifest needed."""
    root.mkdir(parents=True, exist_ok=True)
    build_cas12a(root, **{k: v for k, v in kwargs.items() if k == "output_unit"})
    build_spcas9(root, **{k: v for k, v in kwargs.items() if k == "label_scale"})
    return root
