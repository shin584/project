"""Tests for the v3 replicate noise floor (#135, #129's AC14).

Labels only: no predictions, no checkpoint, no GPU. The synthetic splits are
built at the shape `v3_strata.analysis_intersection` consumes — raw percent
label columns per head — because that is what this module reads.

One block runs the real split to pin the reference values the design and
`CONTEXT.md` already quote, and skips when the split is absent since it is
git-ignored.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import v3_noise
import v3_predict
import v3_strata
from v3_strata import ANALYSED_HEADS

XAI_TEST_ROOT = Path(v3_strata.__file__).resolve().parent.parent
SPLIT_CSV = XAI_TEST_ROOT / "SpCas9" / "published_split" / "test.csv"

#: The design's measured reference, in fractions of 1. Pinned here so a change
#: to the measurement has to change a number a report already quotes.
REFERENCE_GROUPS = 79
REFERENCE_ROWS = 379
REFERENCE_INTERSECTION_ROWS = 3851
#: What the *uncut* split holds, which is the attrition the intersection cut
#: costs: 779 repeated groups become 79 once every analysed head has to be
#: measured on the row.
REFERENCE_SPLIT_ROWS = 5186
REFERENCE_SPLIT_REPEAT_GROUPS = 779
REFERENCE_SPLIT_REPEAT_ROWS = 2247
REFERENCE_POOLED = {"median": 0.053, "p90": 0.105}
REFERENCE_BY_HEAD = {"SpCas9": 0.064, "SpCas9-NG": 0.070, "evoCas9": 0.0099}


def window(index: int) -> str:
    """A distinct 30-mer per index, stable across tests."""
    body = f"{index:026d}".translate(str.maketrans("0123456789", "ACGTACGTAC"))
    return "ACGT" + body


def split_frame(rows: list[dict]) -> pd.DataFrame:
    """A split-shaped frame: `sequence` plus one raw-percent column per head.

    `labels` gives the three analysed heads' percentages in `ANALYSED_HEADS`
    order; a head left as None is unmeasured, which is how a row falls out of
    the analysis intersection.
    """
    records = []
    for row_index, row in enumerate(rows):
        record = {
            "row_index": row.get("row_index", row_index),
            "sequence": row["sequence"],
            "in_pam_library": row.get("in_pam_library", False),
        }
        for head, label in zip(ANALYSED_HEADS, row["labels"]):
            record[head] = label
        record.update(row.get("extra", {}))
        records.append(record)
    return pd.DataFrame(records)


def flat(sequence: str, value: float, **extra) -> dict:
    """One row whose three heads all carry the same percentage."""
    return {"sequence": sequence, "labels": (value, value, value), **extra}


# --- which rows the measurement rests on -----------------------------------


def test_repeated_rows_keep_only_sequences_seen_at_least_twice():
    frame = split_frame(
        [
            flat(window(0), 10.0),
            flat(window(0), 20.0),
            flat(window(1), 30.0),
        ]
    )
    repeated = v3_noise.repeated_sequence_rows(frame)
    assert list(repeated["sequence"].unique()) == [window(0)]
    assert len(repeated) == 2


def test_repeated_rows_are_empty_when_every_sequence_is_unique():
    frame = split_frame([flat(window(index), 10.0) for index in range(4)])
    assert len(v3_noise.repeated_sequence_rows(frame)) == 0


def test_repeat_minimum_is_two_replicates():
    assert v3_noise.REPEAT_MIN == 2


# --- the spread itself ------------------------------------------------------


def test_spread_is_max_minus_min_divided_by_a_hundred():
    frame = split_frame([flat(window(0), 10.0), flat(window(0), 35.0)])
    spreads = v3_noise.group_spreads(frame)
    assert len(spreads) == 1
    for head in ANALYSED_HEADS:
        assert spreads.at[0, f"spread_{head}"] == pytest.approx(0.25)


def test_spread_is_reported_on_the_fraction_scale_not_in_percent():
    assert v3_noise.LABEL_DIVISOR == v3_strata.LABEL_DIVISOR == 100.0
    frame = split_frame([flat(window(0), 0.0), flat(window(0), 100.0)])
    spreads = v3_noise.group_spreads(frame)
    assert spreads.at[0, f"spread_{ANALYSED_HEADS[0]}"] == pytest.approx(1.0)


def test_spread_is_measured_per_head_rather_than_across_heads():
    frame = split_frame(
        [
            {"sequence": window(0), "labels": (10.0, 50.0, 90.0)},
            {"sequence": window(0), "labels": (12.0, 50.0, 90.0)},
        ]
    )
    spreads = v3_noise.group_spreads(frame)
    assert spreads.at[0, "spread_SpCas9"] == pytest.approx(0.02)
    assert spreads.at[0, "spread_SpCas9-NG"] == pytest.approx(0.0)
    assert spreads.at[0, "spread_evoCas9"] == pytest.approx(0.0)


def test_group_spreads_carry_the_rows_each_group_rests_on():
    frame = split_frame(
        [flat(window(0), 10.0), flat(window(0), 20.0), flat(window(0), 30.0)]
    )
    spreads = v3_noise.group_spreads(frame)
    assert spreads.at[0, "rows"] == 3


def test_group_spreads_are_ordered_by_sequence():
    frame = split_frame(
        [
            flat(window(1), 10.0),
            flat(window(1), 20.0),
            flat(window(0), 10.0),
            flat(window(0), 20.0),
        ]
    )
    spreads = v3_noise.group_spreads(frame)
    assert list(spreads["sequence"]) == sorted([window(0), window(1)])


def test_unanalysed_heads_are_not_measured():
    frame = split_frame(
        [
            flat(window(0), 10.0, extra={"VRQR variant": 0.0}),
            flat(window(0), 20.0, extra={"VRQR variant": 100.0}),
        ]
    )
    spreads = v3_noise.group_spreads(frame)
    assert "spread_VRQR variant" not in spreads.columns
    summary = v3_noise.summarise(spreads)
    assert set(summary["by_head"]) == set(ANALYSED_HEADS)


# --- the summary ------------------------------------------------------------


def test_summary_reports_the_group_count_and_the_row_count():
    frame = split_frame(
        [
            flat(window(0), 10.0),
            flat(window(0), 20.0),
            flat(window(1), 10.0),
            flat(window(1), 20.0),
            flat(window(1), 30.0),
        ]
    )
    summary = v3_noise.summarise(v3_noise.group_spreads(frame))
    assert summary["groups"] == 2
    assert summary["rows"] == 5


def test_summary_reports_every_head_separately_as_well_as_pooled():
    frame = split_frame(
        [
            {"sequence": window(0), "labels": (10.0, 20.0, 30.0)},
            {"sequence": window(0), "labels": (20.0, 40.0, 60.0)},
        ]
    )
    summary = v3_noise.summarise(v3_noise.group_spreads(frame))
    assert set(summary["by_head"]) == set(ANALYSED_HEADS)
    assert summary["by_head"]["SpCas9"]["median"] == pytest.approx(0.10)
    assert summary["by_head"]["SpCas9-NG"]["median"] == pytest.approx(0.20)
    assert summary["by_head"]["evoCas9"]["median"] == pytest.approx(0.30)
    assert summary["pooled"]["median"] == pytest.approx(0.20)


def test_pooled_statistics_are_the_heads_spreads_concatenated():
    frame = split_frame(
        [
            {"sequence": window(0), "labels": (10.0, 20.0, 30.0)},
            {"sequence": window(0), "labels": (20.0, 40.0, 60.0)},
            {"sequence": window(1), "labels": (10.0, 20.0, 30.0)},
            {"sequence": window(1), "labels": (20.0, 40.0, 60.0)},
        ]
    )
    summary = v3_noise.summarise(v3_noise.group_spreads(frame))
    assert summary["pooled"]["measurements"] == 2 * len(ANALYSED_HEADS)
    assert summary["by_head"]["SpCas9"]["groups"] == 2


def test_pooled_entry_counts_measurements_and_heads_count_groups():
    frame = split_frame([flat(window(0), 10.0), flat(window(0), 20.0)])
    summary = v3_noise.summarise(v3_noise.group_spreads(frame))
    assert "groups" not in summary["pooled"]
    assert "measurements" not in summary["by_head"]["SpCas9"]
    assert summary["by_head"]["SpCas9"]["groups"] == 1


def test_the_tail_key_is_derived_from_the_percentile():
    assert v3_noise.TAIL_KEY == f"p{v3_noise.TAIL_PERCENTILE}"
    assert v3_noise.TAIL_KEY in v3_noise.FLOOR_STATISTICS
    frame = split_frame([flat(window(0), 10.0), flat(window(0), 20.0)])
    summary = v3_noise.summarise(v3_noise.group_spreads(frame))
    assert v3_noise.TAIL_KEY in summary["pooled"]


def test_summary_records_how_far_apart_the_heads_are():
    frame = split_frame(
        [
            {"sequence": window(0), "labels": (10.0, 20.0, 30.0)},
            {"sequence": window(0), "labels": (20.0, 40.0, 60.0)},
        ]
    )
    summary = v3_noise.summarise(v3_noise.group_spreads(frame))
    assert summary["head_floor_ratio"] == pytest.approx(3.0)


def test_summary_refuses_an_empty_measurement():
    frame = split_frame([flat(window(index), 10.0) for index in range(3)])
    with pytest.raises(ValueError, match="no repeated"):
        v3_noise.summarise(v3_noise.group_spreads(frame))


# --- reading a floor back out ----------------------------------------------


def test_floor_for_head_returns_that_heads_statistic():
    frame = split_frame(
        [
            {"sequence": window(0), "labels": (10.0, 20.0, 30.0)},
            {"sequence": window(0), "labels": (20.0, 40.0, 60.0)},
        ]
    )
    summary = v3_noise.summarise(v3_noise.group_spreads(frame))
    assert v3_noise.floor_for_head(summary, "evoCas9") == pytest.approx(0.30)
    assert v3_noise.floor_for_head(summary, "evoCas9", statistic="p90") == (
        pytest.approx(0.30)
    )


def test_floor_for_head_refuses_the_pooled_value_as_a_head():
    frame = split_frame([flat(window(0), 10.0), flat(window(0), 20.0)])
    summary = v3_noise.summarise(v3_noise.group_spreads(frame))
    with pytest.raises(ValueError, match="pooled"):
        v3_noise.floor_for_head(summary, "pooled")


def test_floor_for_head_refuses_an_unknown_head():
    frame = split_frame([flat(window(0), 10.0), flat(window(0), 20.0)])
    summary = v3_noise.summarise(v3_noise.group_spreads(frame))
    with pytest.raises(ValueError, match="unknown head"):
        v3_noise.floor_for_head(summary, "HypaCas9")


def test_floor_for_head_refuses_an_unknown_statistic():
    frame = split_frame([flat(window(0), 10.0), flat(window(0), 20.0)])
    summary = v3_noise.summarise(v3_noise.group_spreads(frame))
    with pytest.raises(ValueError, match="statistic"):
        v3_noise.floor_for_head(summary, "SpCas9", statistic="mean_of_medians")


# --- the measurement over a split ------------------------------------------


def test_measure_works_from_labels_alone():
    frame = split_frame(
        [flat(window(0), 10.0), flat(window(0), 20.0), flat(window(1), 30.0)]
    )
    measured = v3_noise.measure(frame)
    assert measured.summary["groups"] == 1
    assert measured.summary["intersection_rows"] == 3
    assert len(measured.spreads) == 1


def test_measure_rests_on_the_analysis_intersection():
    frame = split_frame(
        [
            flat(window(0), 10.0),
            flat(window(0), 20.0),
            {"sequence": window(1), "labels": (10.0, 20.0, None)},
            {"sequence": window(1), "labels": (20.0, 40.0, None)},
        ]
    )
    measured = v3_noise.measure(frame)
    assert measured.summary["intersection_rows"] == 2
    assert measured.summary["groups"] == 1
    assert list(measured.spreads["sequence"]) == [window(0)]


def test_measure_refuses_a_split_missing_a_head():
    frame = split_frame([flat(window(0), 10.0), flat(window(0), 20.0)])
    with pytest.raises(ValueError, match="evoCas9"):
        v3_noise.measure(frame.drop(columns=["evoCas9"]))


def test_measure_refuses_a_split_missing_row_index():
    frame = split_frame([flat(window(0), 10.0), flat(window(0), 20.0)])
    with pytest.raises(ValueError, match="row_index"):
        v3_noise.measure(frame.drop(columns=["row_index"]))


def test_measure_pins_the_rows_the_floor_rests_on():
    frame = split_frame(
        [
            flat(window(0), 10.0, row_index=7),
            flat(window(0), 20.0, row_index=3),
            flat(window(1), 30.0, row_index=5),
        ]
    )
    summary = v3_noise.measure(frame).summary
    assert summary["population_digest"] == v3_predict.population_digest([3, 7])


def test_measure_records_what_the_intersection_cut_costs():
    frame = split_frame(
        [
            flat(window(0), 10.0),
            flat(window(0), 20.0),
            {"sequence": window(1), "labels": (10.0, 20.0, None)},
            {"sequence": window(1), "labels": (20.0, 40.0, None)},
            flat(window(2), 30.0),
        ]
    )
    summary = v3_noise.measure(frame).summary
    assert summary["split_rows"] == 5
    assert summary["repeated_groups_in_split"] == 2
    assert summary["repeated_rows_in_split"] == 4
    assert summary["groups"] == 1
    assert summary["rows"] == 2


# --- the artifact -----------------------------------------------------------


def repeated_split() -> pd.DataFrame:
    return split_frame(
        [
            {"sequence": window(0), "labels": (10.0, 20.0, 30.0)},
            {"sequence": window(0), "labels": (20.0, 40.0, 60.0)},
            {"sequence": window(1), "labels": (11.0, 21.0, 31.0)},
            {"sequence": window(1), "labels": (15.0, 25.0, 35.0)},
            {"sequence": window(2), "labels": (50.0, 50.0, 50.0)},
        ]
    )


def test_written_artifact_carries_the_group_and_row_counts(tmp_path):
    measured = v3_noise.measure(repeated_split())
    path = v3_noise.write_noise_floor(tmp_path, measured)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == v3_noise.NOISE_NAME == "v3_noise_floor.json"
    assert payload["groups"] == 2
    assert payload["rows"] == 4
    assert payload["intersection_rows"] == 5


def test_written_artifact_reports_each_head_not_only_the_pooled_floor(tmp_path):
    measured = v3_noise.measure(repeated_split())
    payload = json.loads(
        v3_noise.write_noise_floor(tmp_path, measured).read_text(encoding="utf-8")
    )
    assert set(payload["by_head"]) == set(ANALYSED_HEADS)
    assert payload["pooled"]["median"] > 0
    assert payload["single_constant_note"]


def test_written_artifact_pins_no_model_library(tmp_path):
    measured = v3_noise.measure(repeated_split())
    payload = json.loads(
        v3_noise.write_noise_floor(tmp_path, measured).read_text(encoding="utf-8")
    )
    assert set(payload["versions"]) == set(v3_noise.LIBRARIES_READ)
    assert "torch" not in payload["versions"]
    assert "transformers" not in payload["versions"]


def test_written_artifact_pins_the_rows_and_states_the_cut(tmp_path):
    measured = v3_noise.measure(repeated_split())
    payload = json.loads(
        v3_noise.write_noise_floor(tmp_path, measured).read_text(encoding="utf-8")
    )
    assert payload["population_digest"]
    assert payload["repeated_groups_in_split"] >= payload["groups"]
    assert "repeated_groups_in_split" in payload["limit_note"]


def test_written_artifact_records_the_label_scale(tmp_path):
    measured = v3_noise.measure(repeated_split())
    payload = json.loads(
        v3_noise.write_noise_floor(tmp_path, measured).read_text(encoding="utf-8")
    )
    assert payload["label_scale"] == "fraction"
    assert payload["label_divisor"] == 100.0


def test_written_artifact_keeps_the_per_group_spreads(tmp_path):
    measured = v3_noise.measure(repeated_split())
    payload = json.loads(
        v3_noise.write_noise_floor(tmp_path, measured).read_text(encoding="utf-8")
    )
    assert len(payload["group_spreads"]) == 2
    assert payload["group_spreads"][0]["rows"] == 2
    assert "spread_SpCas9" in payload["group_spreads"][0]


def test_written_artifact_is_byte_pinned_by_its_checksum_file(tmp_path):
    measured = v3_noise.measure(repeated_split())
    path = v3_noise.write_noise_floor(tmp_path, measured)
    checksum = json.loads(
        (tmp_path / v3_noise.NOISE_CHECKSUM_NAME).read_text(encoding="utf-8")
    )
    assert checksum["files"][v3_noise.NOISE_NAME] == v3_predict.sha256_of(path)


def test_consumed_split_refuses_a_split_that_misses_its_pin(tmp_path):
    split = tmp_path / "test.csv"
    split.write_text("row_index\n0\n", encoding="utf-8")
    pin = tmp_path / "split_checksum.json"
    pin.write_text(json.dumps({"files": {"test.csv": "0" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="split_checksum.json"):
        v3_noise.consumed_split(split, split_checksum=pin)


def test_consumed_split_records_the_digest_and_whether_the_pin_was_checked(tmp_path):
    split = tmp_path / "test.csv"
    split.write_text("row_index\n0\n", encoding="utf-8")
    consumed = v3_noise.consumed_split(split, split_checksum=None)
    assert consumed["digests"]["test.csv"] == v3_predict.sha256_of(split)
    assert consumed["split_checksum_pin_checked"] is False
    assert "predictions" not in consumed


def test_build_writes_the_artifact_from_a_split_csv(tmp_path):
    split = tmp_path / "test.csv"
    repeated_split().to_csv(split, index=False)
    path = v3_noise.build_noise_floor(
        split_csv=split, artifacts_root=tmp_path / "artifacts", split_checksum=None
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["groups"] == 2
    assert payload["consumed"]["split"] == "test.csv"
    assert payload["inputs"] == ["labels"]


# --- the real split ---------------------------------------------------------


def real_split() -> pd.DataFrame:
    if not SPLIT_CSV.exists():
        pytest.skip("the published split is absent")
    return pd.read_csv(SPLIT_CSV)


def test_real_noise_floor_rests_on_the_designs_group_and_row_counts():
    summary = v3_noise.measure(real_split()).summary
    assert summary["intersection_rows"] == REFERENCE_INTERSECTION_ROWS
    assert summary["groups"] == REFERENCE_GROUPS
    assert summary["rows"] == REFERENCE_ROWS


def test_real_intersection_cut_is_recorded_because_it_is_most_of_the_repeats():
    summary = v3_noise.measure(real_split()).summary
    assert summary["split_rows"] == REFERENCE_SPLIT_ROWS
    assert summary["repeated_groups_in_split"] == REFERENCE_SPLIT_REPEAT_GROUPS
    assert summary["repeated_rows_in_split"] == REFERENCE_SPLIT_REPEAT_ROWS
    # The floor keeps 79 of 779 groups: the cut, not the split's repetition
    # rate, is what bounds it, which is what AC13 has to report.
    assert summary["groups"] < summary["repeated_groups_in_split"] / 5


def test_real_pooled_floor_reproduces_the_reference_values():
    summary = v3_noise.measure(real_split()).summary
    assert summary["pooled"]["median"] == pytest.approx(
        REFERENCE_POOLED["median"], abs=5e-4
    )
    assert summary["pooled"]["p90"] == pytest.approx(REFERENCE_POOLED["p90"], abs=5e-4)
    assert summary["pooled"]["measurements"] == REFERENCE_GROUPS * len(ANALYSED_HEADS)


def test_real_per_head_floors_reproduce_the_reference_values():
    summary = v3_noise.measure(real_split()).summary
    for head, expected in REFERENCE_BY_HEAD.items():
        assert v3_noise.floor_for_head(summary, head) == pytest.approx(
            expected, abs=5e-4
        )


def test_real_heads_differ_enough_that_one_constant_would_misreport_evocas9():
    summary = v3_noise.measure(real_split()).summary
    assert summary["head_floor_ratio"] > 5
    pooled = summary["pooled"]["median"]
    evocas9 = v3_noise.floor_for_head(summary, "evoCas9")
    assert pooled / evocas9 > 5


def test_real_group_spreads_are_one_row_per_repeated_sequence():
    measured = v3_noise.measure(real_split())
    assert len(measured.spreads) == REFERENCE_GROUPS
    assert measured.spreads["sequence"].is_unique
    assert int(measured.spreads["rows"].sum()) == REFERENCE_ROWS
    assert measured.spreads["rows"].min() >= v3_noise.REPEAT_MIN
    assert np.isfinite(measured.spreads["spread_SpCas9"]).all()
