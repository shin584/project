import json

import pytest

from cas12a.nt_data import canonical_bytes, digest
from xai_test.cas12a_NT_analysis.nt_ig import PROTOCOL
from xai_test.cas12a_NT_analysis.nt_ig_artifacts import plan_ig
from xai_test.cas12a_NT_analysis.nt_run import STUDY_ID
from xai_test.cas12a_NT_analysis.nt_selection import (
    read_cases,
    read_coverage,
    save_cases,
)
from xai_test.cas12a_NT_analysis.test.test_ig import POLICY
from xai_test.cas12a_NT_analysis.test.test_selection import (
    freeze,
    sequence,
)


@pytest.mark.parametrize("mode,size", [("full", None), ("subset", 400)])
def test_frozen_plan_matches_unique_work_union_and_preserves_statistics(
    selection_inputs, tmp_path, mode, size
):
    partition_path, decision_path, _ = selection_inputs
    decision = json.loads(decision_path.read_bytes())
    protocol = {"ig": PROTOCOL}
    selection_identity = {
        "study_id": STUDY_ID,
        "run_id": "a",
        "model_sha256": "d" * 64,
        "protocol_sha256": digest(protocol),
    }
    decision["analysis_identity"] = selection_identity
    decision_path.write_bytes(canonical_bytes(decision))
    root, cases_root = tmp_path / "coverage", tmp_path / "cases"
    checksum = freeze(selection_inputs, root, mode=mode, size=size)
    coverage = read_coverage(root, checksum=checksum)
    # Select an overlapping error case, plus case-only rows in subset mode.
    overlapping = coverage["storage_ids"][0]
    rows = [
        {
            "source_row": i,
            "sequence": sequence(i),
            "label": 0.0,
            "raw_prediction": -0.5 if i == overlapping else float((i % 16) / 16),
        }
        for i in coverage["population_ids"]
    ]
    case_checksum = save_cases(
        root,
        cases_root,
        coverage_checksum=checksum,
        predictions=rows,
        prediction_identity=selection_identity,
    )
    cases = read_cases(
        root, cases_root, coverage_checksum=checksum, checksum=case_checksum
    )
    partition = json.loads(partition_path.read_bytes())
    identity = {
        "fixture": True,
        "partition": "test",
        "protocol": protocol,
        "export_files": {"model.safetensors": "d" * 64},
        "partition_sha256": partition["partition_sha256"],
        "source_sha256": partition["source_file_sha256"],
        "table_sha256": partition["table_sha256"],
    }
    predictions = {
        "identity": identity,
        "rows": rows,
        "coverage": coverage["population_ids"],
        "coverage_complete": True,
    }
    plan = plan_ig(
        predictions, run_id="a", policy=POLICY, coverage=coverage, cases=cases
    )
    work = plan["work"]
    case_ids = {row["source_row"] for row in cases["cases"]}
    assert work["ids"] == sorted(set(coverage["storage_ids"]) | case_ids)
    assert work["statistics_ids"] == coverage["storage_ids"]
    role = next(
        row["roles"] for row in work["rows"] if row["source_row"] == overlapping
    )
    assert role == [
        "full_population" if mode == "full" else "random_subset",
        "out_of_range",
    ]
    assert work["additional_case_ids"] == sorted(
        case_ids - set(coverage["storage_ids"])
    )
    if mode == "full":
        assert work["additional_case_ids"] == []
    else:
        assert work["additional_case_ids"]
    for field, bad in (("partition", "validation"), ("source_sha256", "0" * 64)):
        with pytest.raises(ValueError, match="provenance"):
            plan_ig(
                {**predictions, "identity": {**identity, field: bad}},
                run_id="a",
                policy=POLICY,
                coverage=coverage,
                cases=cases,
            )
    with pytest.raises(ValueError, match="population"):
        plan_ig(
            {
                **predictions,
                "rows": rows[:-1],
                "coverage": predictions["coverage"][:-1],
            },
            run_id="a",
            policy=POLICY,
            coverage=coverage,
            cases=cases,
        )
