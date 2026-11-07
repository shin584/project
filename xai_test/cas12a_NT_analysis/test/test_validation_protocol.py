import copy

import pytest

from xai_test.cas12a_NT_analysis.nt_validation_protocol import (
    PRESET,
    forecast,
    prepare_protocol,
)


def test_protocol_uses_only_existing_disjoint_validation_ids(inputs):
    first = prepare_protocol(inputs.manifest, fixture=True)
    assert first == prepare_protocol(inputs.manifest, fixture=True)
    ids = [i for group in first["groups"].values() for i in group]
    assert sorted(ids) == sorted(inputs.partition["partitions"]["validation"])
    assert len(ids) == len(set(ids))
    assert not set(ids) & set(inputs.partition["partitions"]["test"])
    assert first["fixture"] is True
    with pytest.raises(ValueError, match="mismatch"):
        prepare_protocol(inputs.manifest)


def test_forecast_counts_full_reservations_and_102_mutants_per_source():
    # Four methods each take 1 second per source. Eight loads take 10 seconds
    # each. Common costs are 2 seconds/source. Before rounding: future=14220.
    # Each method's two jobs round 1777.5 up to 1800: future charge=14400.
    result = forecast(
        {"prediction": 1, "ism": 1, "rollout": 1, "ig": 1, "common": 2, "loading": 10},
        population=1500,
        charged_seconds=500,
        policy=PRESET["forecast"],
    )
    assert result["mode"] == "full"
    assert result["full"]["total_seconds"] == 14900
    assert result["selected"]["additional_case_allowance"] == 0
    assert result["selected"]["reservation_plan"]["ism"] == {
        "sessions": 2,
        "per_session_seconds": 1800,
        "charged_seconds": 3600,
    }


def test_forecast_fallback_includes_all_15_case_only_rows_and_400_floor():
    rates = {
        "prediction": 0,
        "ism": 0,
        "rollout": 0,
        "ig": 20,
        "common": 0,
        "loading": 0,
    }
    result = forecast(
        rates, population=1500, charged_seconds=8000, policy=PRESET["forecast"]
    )
    assert result["full_rejected"] is True
    assert result["mode"] == "subset"
    # Three non-IG methods each reserve 2*120 = 240. IG reserves 2*6420.
    assert result["selected"]["size"] == 408
    assert result["selected"]["additional_case_allowance"] == 15
    assert result["selected"]["total_seconds"] == 21560
    stopped = forecast(
        rates, population=1500, charged_seconds=8400, policy=PRESET["forecast"]
    )
    assert stopped["mode"] is None
    assert stopped["selected"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_forecast_rejects_unknown_or_invalid_costs(value):
    rates = {
        "prediction": 1,
        "ism": 1,
        "rollout": 1,
        "ig": value,
        "common": 1,
        "loading": 1,
    }
    with pytest.raises(ValueError, match="finite"):
        forecast(rates, population=1500, charged_seconds=0, policy=PRESET["forecast"])
    changed = copy.deepcopy(PRESET["forecast"])
    changed["fallback_floor"] = 25
    with pytest.raises(ValueError, match="frozen"):
        forecast({**rates, "ig": 1}, population=1500, charged_seconds=0, policy=changed)
