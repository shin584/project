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

ISM_ROWS = (5, 11, 30)
IG_ROWS = (11, 30)  # misaligned with ISM_ROWS on purpose
SPLIT_ROLE = {5: "test", 11: "test", 30: "val"}
#: Percent, as the split files store it. 12.5 percent is 0.125 as a fraction.
LABEL_PERCENT = {5: 12.5, 11: 40.0, 30: 1.25}
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


def ism_delta(rows: int, positions: int, heads: int) -> tuple[np.ndarray, np.ndarray]:
    """Deltas whose per-head, per-row profile has a closed form.

    The reference slot stays exactly 0 and the three alternatives carry
    0.1/0.2/0.3 plus a per-head and per-row offset, so the mean over the
    alternatives is `expected_profile`. Signs alternate, so a profile that
    forgot the absolute value is visibly wrong.
    """
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

    delta, reference = ism_delta(len(ISM_ROWS), POSITIONS, len(ISM_HEADS))
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
                                    "median": 0.7,
                                },
                                "mean_profile": 1.0,
                                "mean_token_profile": {
                                    "ism": [0.01, 0.02, 0.03, 0.04, 0.11],
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
    write_json(
        "v3_report/head_order_reversal.json",
        {"statistic": "head_order_reversal", "claim_limit": CLAIM_LIMIT, "strata": {}},
    )
    write_json(
        "v3_head_agreement/head_agreement.json",
        {
            "population": "analysis_intersection",
            "rows": len(ISM_ROWS),
            "pooling": "forbidden",
        },
    )
    write_json(
        "v3_noise_floor.json",
        {
            "spread_definition": "max - min of a repeated 30-mer's labels",
            "label_scale": "fraction",
            "by_head": {head: {"median": 0.06, "groups": 3} for head in ANALYSED_HEADS},
            "pooled": {"median": 0.05},
            "single_constant_note": "head별 값을 쓴다",
        },
    )
    write_json(
        "v3_cases/cases.json",
        {
            "primary_head": "SpCas9",
            "heads": list(ANALYSED_HEADS),
            "label_scale": "fraction",
            "prediction_scale": "fraction",
            "cases": [
                {
                    "case_id": "GEN-ERRHI-01",
                    "stratum": "generalisation",
                    "row_index": 11,
                },
                {
                    "case_id": "GEN-HEADGAP-02",
                    "stratum": "generalisation",
                    "row_index": 30,
                },
            ],
        },
    )
    write_json(
        "v3_cases/cases_annotated.json", {"claim_limit": CLAIM_LIMIT, "cases": []}
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
                    "" if (row, head) == UNMEASURED else LABEL_PERCENT[row] + index
                    for index, head in enumerate(heads_header)
                ]
                writer.writerow(
                    [
                        row,
                        "A" * POSITIONS,
                        "A" * 20,
                        "True",
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


def build_cas12a(root: Path, *, output_unit: str = "fraction") -> Path:
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
        for index, row in enumerate(CAS12A_ROWS):
            writer.writerow(
                [
                    row,
                    "T" * CAS12A_POSITIONS,
                    0.2 + index * 0.1,
                    0.25 + index * 0.1,
                    0.25 + index * 0.1,
                    "TTTV" if index else "TTTT",
                    0.05,
                    0.05,
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
        writer.writerow(
            [4, -0.05, 0.0, 0.05, 0.05, "TTTT", "out_of_range", 1, "raw < 0 or raw > 1"]
        )
        writer.writerow(
            [9, 0.35, 0.3, 0.05, 0.0, "TTTV", "low_error", 1, "|error| ascending"]
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
                "mean_abs_raw_delta": [
                    0.01 * (index + 1) for index in range(CAS12A_POSITIONS)
                ],
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
        "agreement.sliced.json": {
            "populations": {"all": {"n_available": 3, "n_total": 3}},
            "group_diagnostics": {},
            "protocol": {"statistic": "spearman"},
        },
        "summary.json": {"complete": True, "population_count": 3},
        "cases_annotated.json": {"cases": [], "selection": {}},
    }.items():
        write_json(name, payload)

    (cas12a / "report.ko.md").write_text(
        "# Cas12a Test 결과\n\n"
        "## 성능\n\n본문.\n\n"
        f"## rollout 해석\n\n{CAS12A_ROLLOUT_TEXT}\n\n"
        f"## 범위와 한계\n\n{CAS12A_LIMIT_TEXT}\n",
        encoding="utf-8",
    )
    return cas12a


def build_data_dir(root: Path, **kwargs) -> Path:
    """A full synthetic `data/` directory: both nucleases, no manifest needed."""
    root.mkdir(parents=True, exist_ok=True)
    build_cas12a(root, **{k: v for k, v in kwargs.items() if k == "output_unit"})
    build_spcas9(root, **{k: v for k, v in kwargs.items() if k == "label_scale"})
    return root
