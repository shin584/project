import hashlib
import json

import pytest

from cas12a.nt_data import canonical_bytes, digest
from xai_test.cas12a_NT_analysis.nt_selection import freeze_coverage, read_coverage


def sequence(source_row):
    bases = "".join("ACGT"[(source_row >> shift) & 3] for shift in range(20, -1, -2))
    return "ACGT" + ("TTTA" if source_row % 2 else "TTTT") + bases + "A" * 12 + "CGA"


def freeze(selection_inputs, directory, *, mode="full", size=None):
    partition_path, decision_path, _ = selection_inputs
    decision = json.loads(decision_path.read_bytes())
    decision["full_rejected"] = mode == "subset"
    decision_path.write_bytes(canonical_bytes(decision))
    return freeze_coverage(
        partition_path,
        decision_path,
        directory,
        decision_checksum=hashlib.sha256(decision_path.read_bytes()).hexdigest(),
        mode=mode,
        size=size,
        fixture=True,
    )


def test_full_saved_membership_preserves_all_1500_source_rows(
    selection_inputs, tmp_path
):
    root = tmp_path / "full"
    checksum = freeze(selection_inputs, root)
    coverage = read_coverage(root, checksum=checksum)
    assert coverage["population_ids"] == list(range(4, 1504))
    assert (
        coverage["selection_ids"]
        == coverage["storage_ids"]
        == coverage["population_ids"]
    )
    assert coverage["size"] == 1500
    assert coverage["mode"] == "full"
    assert coverage["historical_fp16_test_known"] is True
    assert read_coverage(root, checksum=checksum) == coverage
    with pytest.raises(ValueError, match="empty"):
        freeze(selection_inputs, root)


def test_fallback_is_a_nested_random_prefix_with_separate_storage_order(
    selection_inputs, tmp_path
):
    first = freeze(selection_inputs, tmp_path / "400", mode="subset", size=400)
    larger = freeze(selection_inputs, tmp_path / "500", mode="subset", size=500)
    small = read_coverage(tmp_path / "400", checksum=first)
    big = read_coverage(tmp_path / "500", checksum=larger)
    # Fixed PCG64(42) known-answer vector over source-row-sorted membership.
    assert small["selection_ids"][:10] == [
        794,
        862,
        313,
        709,
        142,
        473,
        1426,
        478,
        467,
        1171,
    ]
    assert small["selection_ids"] == big["selection_ids"][:400]
    assert small["storage_ids"] == sorted(small["selection_ids"])
    assert small["membership_sha256"] == digest(small["storage_ids"])
    assert small["size"] == len(set(small["storage_ids"])) == 400


def resize_population(selection_inputs, count):
    partition_path, decision_path, _ = selection_inputs
    partition = json.loads(partition_path.read_bytes())
    partition["partitions"]["test"] = list(range(count + 3, 3, -1))
    partition["counts"]["test"] = count
    partition["rows"] = partition["rows"][: count + 4]
    partition["partition_sha256"] = digest(partition["partitions"])
    partition_path.write_bytes(canonical_bytes(partition))
    decision = json.loads(decision_path.read_bytes())
    decision["partition_sha256"] = partition["partition_sha256"]
    decision_path.write_bytes(canonical_bytes(decision))


def predictions(values, start=4):
    return [
        {
            "source_row": i,
            "sequence": sequence(i),
            "label": 0.0,
            "raw_prediction": value,
            "explanation_artifact": object(),
        }
        for i, value in enumerate(values, start)
    ]


def test_cases_reproduce_global_linear_quantiles_priority_ties_and_shortages(
    selection_inputs, tmp_path
):
    from xai_test.cas12a_NT_analysis.nt_selection import read_cases, save_cases

    resize_population(selection_inputs, 12)
    coverage_root, cases_root = tmp_path / "coverage", tmp_path / "cases"
    coverage_checksum = freeze(selection_inputs, coverage_root)
    rows = predictions(
        [-0.5, 1.5, -0.25, 1.25, 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    )
    checksum = save_cases(
        coverage_root,
        cases_root,
        coverage_checksum=coverage_checksum,
        predictions=list(reversed(rows)),
        prediction_identity=selection_inputs[2],
    )
    result = read_cases(
        coverage_root,
        cases_root,
        coverage_checksum=coverage_checksum,
        checksum=checksum,
    )
    assert result["thresholds"] == {"q25": 0.25, "q75": 0.78125, "degenerate": False}
    assert [(r["source_row"], r["role"]) for r in result["cases"]] == [
        (5, "out_of_range"),
        (4, "out_of_range"),
        (7, "out_of_range"),
        (6, "out_of_range"),
        (15, "high_error"),
        (8, "low_error"),
        (9, "low_error"),
        (10, "low_error"),
    ]
    assert result["categories"]["high_error"] == {
        "eligible": 3,
        "available": 1,
        "selected": 1,
        "unfilled": 4,
    }
    assert result["categories"]["low_error"]["unfilled"] == 2
    assert result["cases"][0]["pam_stratum"] == "TTTV"
    assert result["cases"][1]["pam_stratum"] == "TTTT"
    assert all(row["selection_reason"] for row in result["cases"])
    assert all(
        set(row) == {"source_row", "sequence", "label", "raw_prediction"}
        for row in result["predictions"]
    )
    assert result["protocol"]["quantile_method"] == "linear"
    assert result["coverage_checksum"] == coverage_checksum
    second = save_cases(
        coverage_root,
        tmp_path / "cases-again",
        coverage_checksum=coverage_checksum,
        predictions=rows,
        prediction_identity=selection_inputs[2],
    )
    assert second == checksum


@pytest.mark.parametrize("mode,size", [("full", None), ("subset", 400)])
def test_ig_union_reuses_overlap_and_keeps_cases_out_of_random_statistics(
    selection_inputs, tmp_path, mode, size
):
    from xai_test.cas12a_NT_analysis.nt_selection import ig_work, read_cases, save_cases

    root = tmp_path / "coverage"
    coverage_checksum = freeze(selection_inputs, root, mode=mode, size=size)
    coverage = read_coverage(root, checksum=coverage_checksum)
    overlap = coverage["selection_ids"][0]
    outside = next(
        (i for i in coverage["population_ids"] if i not in coverage["storage_ids"]),
        None,
    )
    rows = predictions([0.25] * 1500)
    rows[overlap - 4]["raw_prediction"] = 2.0
    if outside is not None:
        rows[outside - 4]["raw_prediction"] = 2.0
    checksum = save_cases(
        root,
        tmp_path / "cases",
        coverage_checksum=coverage_checksum,
        predictions=rows,
        prediction_identity=selection_inputs[2],
    )
    cases = read_cases(
        root, tmp_path / "cases", coverage_checksum=coverage_checksum, checksum=checksum
    )
    work = ig_work(coverage, cases)
    assert work["statistics_ids"] == coverage["storage_ids"]
    assert work["ids"] == sorted(
        set(work["statistics_ids"]) | {r["source_row"] for r in cases["cases"]}
    )
    overlap_roles = next(r["roles"] for r in work["rows"] if r["source_row"] == overlap)
    assert overlap_roles == [
        ("full_population" if mode == "full" else "random_subset"),
        "out_of_range",
    ]
    assert overlap not in work["additional_case_ids"]
    if mode == "full":
        assert work["additional_case_ids"] == []
        assert len(work["ids"]) == 1500
    else:
        assert outside in work["additional_case_ids"]
        assert outside not in work["statistics_ids"]
        assert len(work["statistics_ids"]) == 400
    assert work["case_checksum"] == checksum


def test_posthoc_notes_reference_original_selection_without_changing_it(
    selection_inputs, tmp_path
):
    from xai_test.cas12a_NT_analysis.nt_selection import (
        read_annotations,
        read_cases,
        save_annotations,
        save_cases,
    )

    resize_population(selection_inputs, 4)
    root, cases_root = tmp_path / "coverage", tmp_path / "cases"
    coverage_checksum = freeze(selection_inputs, root)
    checksum = save_cases(
        root,
        cases_root,
        coverage_checksum=coverage_checksum,
        predictions=predictions([0, 0.25, 0.5, 1.5]),
        prediction_identity=selection_inputs[2],
    )
    original = (cases_root / "manifest.json").read_bytes()
    annotations = [
        {
            "source_row": 7,
            "note": "Synthetic explanation note added after case selection",
        }
    ]
    notes_checksum = save_annotations(
        root,
        cases_root,
        tmp_path / "notes",
        coverage_checksum=coverage_checksum,
        selection_checksum=checksum,
        annotations=annotations,
    )
    notes = read_annotations(
        root,
        cases_root,
        tmp_path / "notes",
        coverage_checksum=coverage_checksum,
        selection_checksum=checksum,
        checksum=notes_checksum,
    )
    assert notes["selection_checksum"] == checksum
    assert notes["annotations"] == annotations
    assert (cases_root / "manifest.json").read_bytes() == original
    assert (
        read_cases(
            root, cases_root, coverage_checksum=coverage_checksum, checksum=checksum
        )["cases"][0]["source_row"]
        == 7
    )
    with pytest.raises(ValueError, match="selected"):
        save_annotations(
            root,
            cases_root,
            tmp_path / "bad-notes",
            coverage_checksum=coverage_checksum,
            selection_checksum=checksum,
            annotations=[{"source_row": 1500, "note": "not a case"}],
        )


@pytest.mark.parametrize(
    "value,out_ids,low_ids",
    [(0.25, [], list(range(4, 9))), (2.0, list(range(4, 9)), list(range(9, 14)))],
)
def test_equal_quartiles_keep_high_empty_and_resolve_all_ties_by_source_id(
    selection_inputs, tmp_path, value, out_ids, low_ids
):
    from xai_test.cas12a_NT_analysis.nt_selection import read_cases, save_cases

    resize_population(selection_inputs, 20)
    root = tmp_path / "coverage"
    coverage_checksum = freeze(selection_inputs, root)
    checksum = save_cases(
        root,
        tmp_path / "cases",
        coverage_checksum=coverage_checksum,
        predictions=predictions([value] * 20),
        prediction_identity=selection_inputs[2],
    )
    cases = read_cases(
        root, tmp_path / "cases", coverage_checksum=coverage_checksum, checksum=checksum
    )
    assert cases["thresholds"] == {"q25": value, "q75": value, "degenerate": True}
    assert [
        r["source_row"] for r in cases["cases"] if r["role"] == "out_of_range"
    ] == out_ids
    assert [
        r["source_row"] for r in cases["cases"] if r["role"] == "low_error"
    ] == low_ids
    assert cases["categories"]["high_error"] == {
        "eligible": 0,
        "available": 0,
        "selected": 0,
        "unfilled": 5,
    }


def test_high_error_ties_use_source_id_without_pam_quota(selection_inputs, tmp_path):
    from xai_test.cas12a_NT_analysis.nt_selection import read_cases, save_cases

    resize_population(selection_inputs, 20)
    root = tmp_path / "coverage"
    coverage_checksum = freeze(selection_inputs, root)
    checksum = save_cases(
        root,
        tmp_path / "cases",
        coverage_checksum=coverage_checksum,
        predictions=predictions([0.5] * 10 + [0.875] * 10),
        prediction_identity=selection_inputs[2],
    )
    cases = read_cases(
        root, tmp_path / "cases", coverage_checksum=coverage_checksum, checksum=checksum
    )
    assert cases["thresholds"] == {"q25": 0.5, "q75": 0.875, "degenerate": False}
    assert [r["source_row"] for r in cases["cases"]] == list(range(14, 19)) + list(
        range(4, 9)
    )
    assert cases["categories"]["out_of_range"]["unfilled"] == 5


@pytest.mark.parametrize(
    "mode,size",
    [
        ("bad", None),
        ("full", 1499),
        ("full", True),
        ("subset", None),
        ("subset", 399),
        ("subset", 400.0),
        ("subset", 1500),
    ],
)
def test_invalid_coverage_mode_or_size_leaves_no_artifact(
    selection_inputs, tmp_path, mode, size
):
    with pytest.raises(ValueError):
        freeze(selection_inputs, tmp_path / "invalid", mode=mode, size=size)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-partition",
        "not-validation",
        "no-measurement",
        "no-rejection",
        "no-identity",
        "checksum",
        "production",
    ],
)
def test_cost_evidence_and_fixture_identity_are_required(
    selection_inputs, tmp_path, mutation
):
    partition_path, decision_path, _ = selection_inputs
    decision = json.loads(decision_path.read_bytes())
    decision["full_rejected"] = True
    if mutation == "wrong-partition":
        decision["partition_sha256"] = "0" * 64
    elif mutation == "not-validation":
        decision["partition"] = "test"
    elif mutation == "no-measurement":
        decision["measurement_reference"] = ""
    elif mutation == "no-rejection":
        decision["full_rejected"] = False
    elif mutation == "no-identity":
        decision["analysis_identity"] = {"x": 1}
    decision_path.write_bytes(canonical_bytes(decision))
    checksum = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        freeze_coverage(
            partition_path,
            decision_path,
            tmp_path / "bad",
            mode="subset",
            size=400,
            fixture=mutation != "production",
            decision_checksum="0" * 64 if mutation == "checksum" else checksum,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "unexpected",
        "nan",
        "infinite",
        "label",
        "sequence",
        "identity",
        "bool-id",
    ],
)
def test_cases_reject_incomplete_or_incompatible_prediction_inputs(
    selection_inputs, tmp_path, mutation
):
    from xai_test.cas12a_NT_analysis.nt_selection import save_cases

    resize_population(selection_inputs, 12)
    root = tmp_path / "coverage"
    coverage_checksum = freeze(selection_inputs, root)
    rows = predictions([0.25] * 12)
    identity = dict(selection_inputs[2])
    if mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "unexpected":
        rows[-1] = predictions([0.25], start=100)[0]
    elif mutation in ("nan", "infinite"):
        rows[0]["raw_prediction"] = float("nan" if mutation == "nan" else "inf")
    elif mutation == "label":
        rows[0]["label"] = 25.0
    elif mutation == "sequence":
        rows[0]["sequence"] = sequence(100)
    elif mutation == "identity":
        identity["model_sha256"] = "0" * 64
    elif mutation == "bool-id":
        rows[0]["source_row"] = True
    with pytest.raises(ValueError):
        save_cases(
            root,
            tmp_path / "bad",
            coverage_checksum=coverage_checksum,
            predictions=rows,
            prediction_identity=identity,
        )
    assert not (tmp_path / "bad").exists()


def rewrite_manifest(root, mutate):
    manifest = json.loads((root / "manifest.json").read_bytes())
    mutate(manifest)
    (root / "manifest.json").write_bytes(canonical_bytes(manifest))
    checksum = digest(manifest)
    (root / "checksum.json").write_bytes(canonical_bytes({"sha256": checksum}))
    return checksum


@pytest.mark.parametrize(
    "mutation",
    [
        "bytes",
        "missing",
        "input",
        "sidecar",
        "duplicate",
        "wrong-mode",
        "order",
        "floor",
    ],
)
def test_coverage_readback_rejects_tampering_even_with_recomputed_sidecars(
    selection_inputs, tmp_path, mutation
):
    root = tmp_path / "coverage"
    checksum = freeze(selection_inputs, root, mode="subset", size=400)
    if mutation == "bytes":
        (root / "manifest.json").write_bytes(b"{}")
    elif mutation == "missing":
        (root / "decision.json").unlink()
    elif mutation == "input":
        (root / "partition.json").write_bytes(b"{}")
    elif mutation == "sidecar":
        rewrite_manifest(root, lambda m: m.update(size=401))
        # Keep independently retained original checksum.
    elif mutation == "duplicate":
        checksum = rewrite_manifest(
            root, lambda m: m["storage_ids"].__setitem__(0, m["storage_ids"][1])
        )
    elif mutation == "wrong-mode":
        checksum = rewrite_manifest(root, lambda m: m.update(mode="invalid"))
    elif mutation == "order":
        checksum = rewrite_manifest(root, lambda m: m["selection_ids"].reverse())
    elif mutation == "floor":
        checksum = rewrite_manifest(root, lambda m: m.update(size=399))
    with pytest.raises(ValueError):
        read_coverage(root, checksum=checksum)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate-case", "missing-prediction", "threshold", "binding", "checksum"],
)
def test_case_readback_reconstructs_and_rejects_altered_selections(
    selection_inputs, tmp_path, mutation
):
    from xai_test.cas12a_NT_analysis.nt_selection import read_cases, save_cases

    resize_population(selection_inputs, 12)
    root, cases_root = tmp_path / "coverage", tmp_path / "cases"
    coverage_checksum = freeze(selection_inputs, root)
    checksum = save_cases(
        root,
        cases_root,
        coverage_checksum=coverage_checksum,
        predictions=predictions([0.25] * 12),
        prediction_identity=selection_inputs[2],
    )
    if mutation == "duplicate-case":
        checksum = rewrite_manifest(
            cases_root, lambda m: m["cases"].append(m["cases"][0])
        )
    elif mutation == "missing-prediction":
        checksum = rewrite_manifest(cases_root, lambda m: m["predictions"].pop())
    elif mutation == "threshold":
        checksum = rewrite_manifest(
            cases_root, lambda m: m["thresholds"].update(q25=0.5)
        )
    elif mutation == "binding":
        checksum = rewrite_manifest(
            cases_root, lambda m: m.update(coverage_checksum="0" * 64)
        )
    elif mutation == "checksum":
        checksum = "0" * 64
    with pytest.raises(ValueError):
        read_cases(
            root, cases_root, coverage_checksum=coverage_checksum, checksum=checksum
        )
