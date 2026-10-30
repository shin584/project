"""Read-only characterization of the local e_system consumer (#97).

Uses actual imported modules, NumPy and the model-argument seam; no AST copies,
encoder substitutions, model loading, training, or operating-model selection.
"""

import argparse
import hashlib
import importlib
import json
import math
import platform
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

PROTEINS = (
    "SpCas9(WT)",
    "SpCas9-NG",
    "VRQR",
    "xCas9",
    "Sniper",
    "SpCas9-HF1",
    "eSpCas9(1.1)",
    "HypaCas9",
    "evoCas9",
)
WINDOWS = (
    "ACGT" + "ACGT" * 5 + "AGG" + "TCA",
    "TGCA" + "TGCA" * 5 + "TGA" + "ATC",
    "CATG" + "GATTACAGATTACAGATTAC" + "CGT" + "GTC",
)
SOURCE_FILES = ("predict.py", "pam_scanner.py", "adapters.py", "backbones.py", "app.py")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _consumer(source):
    # e_system uses sibling absolute imports. Isolate those names from other
    # projects' identically named modules, then restore the caller's imports.
    names = ("predict", "pam_scanner", "adapters", "backbones")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    original_path = sys.path[:]
    original_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        predict = importlib.import_module("predict")
        scanner = importlib.import_module("pam_scanner")
        for name in names:
            assert Path(sys.modules[name].__file__).resolve() == source / f"{name}.py"
        yield predict, scanner
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


class _ControlledModel:
    def __init__(self, outputs, sites):
        self.outputs = outputs
        self.sites = sites
        self.calls = []

    def predict(self, inputs, verbose=0):
        site = self.sites[len(self.calls)]
        self.calls.append(
            {
                "window": site.model_input_seq,
                "window_sha256": hashlib.sha256(
                    site.model_input_seq.encode()
                ).hexdigest(),
                "dtype": str(inputs.dtype),
                "dtype_kind": inputs.dtype.kind,
                "shape": list(inputs.shape),
                "encoded_array": inputs.tolist(),
            }
        )
        return [np.array([[raw]]) for raw in self.outputs[len(self.calls) - 1]]


def _same_number(observed, expected):
    return (
        math.isnan(observed)
        and math.isnan(expected)
        or observed == expected
        or math.isclose(observed, expected, abs_tol=1e-12)
    )


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _case(predict, scanner, outputs, distances):
    sites = [
        scanner.CandidateSite(
            cas_type="SpCas9",
            strand="+",
            pam_start=24,
            cut_pos=21,
            distance=distance,
            guide_seq=WINDOWS[i][4:24],
            pam_seq=WINDOWS[i][24:26],
            model_input_seq=WINDOWS[i],
            encoded_seq=["must not be consumed"],
        )
        for i, distance in enumerate(distances)
    ]
    model = _ControlledModel(outputs, sites)
    result = scanner.ScanResult(input_seq="A" * 81, sites=sites)
    error = None
    try:
        returned = predict.run_prediction(result, model, None, None)
        assert returned is result
    except (UnboundLocalError, ValueError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    # The oracle is independent of the consumer's Gaussian function and scale.
    expected = [
        [raw * math.exp(-distance * distance / 200) * 100 for raw in heads]
        for heads, distance in zip(outputs, distances, strict=True)
    ]
    for index, row in enumerate(result.variant_results):
        site_index, head_index = divmod(index, 9)
        assert row.cas_name == PROTEINS[head_index], (
            "Positional protein mapping changed"
        )
        assert _same_number(row.raw_score, outputs[site_index][head_index])
        assert _same_number(row.final_score, expected[site_index][head_index])
    if error is None:
        assert len(result.variant_results) == len(outputs) * 9
        assert len(model.calls) == len(outputs)
    for call in model.calls:
        assert call["dtype_kind"] == "i", "Actual encoder no longer emits integers"
        assert call["shape"] == [1, 30, 4]
        expected_encoding = [
            [int(base == channel) for channel in "ACGT"] for base in call["window"]
        ]
        assert call["encoded_array"] == [expected_encoding], (
            "Unexpected encoder channels"
        )
    return {
        "inputs": [asdict(site) for site in sites],
        "controlled_raw": outputs,
        "expected_final": expected,
        "model_calls": model.calls,
        "variants": [asdict(row) for row in result.variant_results],
        "exception": error,
    }


def _ng_discovery(scanner):
    config = next(config for config in scanner.CAS_CONFIGS if config.name == "SpCas9")
    observations = []
    for pam in ("AGG", "AGA", "AGC", "AGT"):
        window = WINDOWS[0][:24] + pam + WINDOWS[0][27:]
        region = "A" * 15 + window + "A" * 36
        sites = scanner.scan_pam(region, config, max_dist=81)
        site = next(
            site
            for site in sites
            if site.strand == "+" and site.model_input_seq == window
        )
        observations.append(
            {
                "three_base_pam": window[24:27],
                "matched_pam": site.pam_seq,
                "input_region": region,
                "site": asdict(site),
            }
        )
    return observations


def _selection_status(cases):
    negative = [
        cases[name]
        for name in ("first_below_minus_one", "exact_minus_one", "consecutive_negative")
    ]
    correct = all(
        case["exception"] is None
        and all(
            math.isclose(site["final_score"], max(finals), abs_tol=1e-12)
            and math.isclose(site["raw_score"], raw[finals.index(max(finals))])
            for site, raw, finals in zip(
                case["inputs"],
                case["controlled_raw"],
                case["expected_final"],
                strict=True,
            )
        )
        for case in negative
    )
    if correct:
        return "fixed"
    first, boundary, later = negative
    if (
        all(
            case["exception"]
            and case["exception"]["type"] == "UnboundLocalError"
            and len(case["variants"]) == 9
            and len(case["model_calls"]) == 1
            for case in (first, boundary)
        )
        and later["exception"] is None
        and [site["raw_score"] for site in later["inputs"]] == [0.2] * 3
        and [site["final_score"] for site in later["inputs"]] == [20, -1, -1]
    ):
        return "known_defect_present"
    raise AssertionError(
        "Unrecognized or partially repaired negative-score selection; inspect observations"
    )


def characterize_consumer(source, destination, *, prior_evidence=None):
    """Preserve source snapshots and observations under a new destination.

    Missing source/dependencies or unrecognized behavior fails explicitly;
    report.json is written only after all checks and source preservation pass.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError(
            "Evidence destination must be outside the consumer source directory"
        )
    before = {name: _sha256(source / name) for name in SOURCE_FILES}
    prior_bytes = (
        Path(prior_evidence).read_bytes() if prior_evidence is not None else None
    )
    destination.mkdir()  # Never overwrite a previous attempt or its evidence.
    snapshots = destination / "source"
    snapshots.mkdir()
    for name in SOURCE_FILES:
        (snapshots / name).write_bytes((source / name).read_bytes())
    if prior_bytes is not None:
        (destination / "prior-review-evidence.json").write_bytes(prior_bytes)
    with _consumer(source) as (predict, scanner):
        cases = {
            "distinct_heads": _case(
                predict,
                scanner,
                [[0, -0.25, 1.5, 0.3, 0.4, 0.56, 0.67, 0.8, 0.9]],
                [10],
            )
        }
        for name, outputs, distances in (
            ("first_below_minus_one", [[-0.02] * 9], [0]),
            ("exact_minus_one", [[-0.01] * 9], [0]),
            ("consecutive_negative", [[0.2] * 9, [-0.02] * 9, [-0.04] * 9], [0, 0, 0]),
            (
                "zero_and_tie",
                [[0, 0, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08]],
                [0],
            ),
        ):
            cases[name] = _case(predict, scanner, outputs, distances)
        for name in ("distinct_heads", "zero_and_tie"):
            case = cases[name]
            assert case["exception"] is None, (
                f"Finite case {name} failed: {case['exception']}"
            )
            for site, raw, finals in zip(
                case["inputs"],
                case["controlled_raw"],
                case["expected_final"],
                strict=True,
            ):
                winning_index = finals.index(max(finals))
                assert _same_number(site["raw_score"], raw[winning_index]), (
                    f"Finite case {name}: raw"
                )
                assert _same_number(site["final_score"], finals[winning_index]), (
                    f"Finite case {name}: final"
                )
        selection_status = _selection_status(cases)
        for name, invalid in (
            ("nan_head", float("nan")),
            ("positive_infinity_head", float("inf")),
            ("negative_infinity_head", -float("inf")),
        ):
            cases[name] = _case(predict, scanner, [[0.2, invalid] + [0.1] * 7], [0])
        discovery = _ng_discovery(scanner)
    after = {name: _sha256(source / name) for name in SOURCE_FILES}
    assert before == after, "Consumer source changed during characterization"
    assert before == {name: _sha256(snapshots / name) for name in SOURCE_FILES}
    if prior_bytes is not None:
        assert Path(prior_evidence).read_bytes() == prior_bytes, (
            "Prior evidence changed"
        )
    report = {
        "issue": 97,
        "status": "verified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verification_code_sha256": _sha256(__file__),
        "verification_mode": "actual imported run_prediction with controlled model argument",
        "source_directory": str(source),
        "source_sha256": before,
        "source_preserved": True,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{
                name: version(name)
                for name in ("numpy", "tensorflow", "keras", "torch", "streamlit")
            },
        },
        "real_checkpoint_inference_executed": False,
        "keras_dtype_acceptance_verified": False,
        "models_loaded": False,
        "channels": list("ACGT"),
        "cases": cases,
        "selection_status": selection_status,
        "ng_discovery": discovery,
        "prior_evidence": None
        if prior_bytes is None
        else {
            "path": str(Path(prior_evidence).resolve()),
            "sha256": hashlib.sha256(prior_bytes).hexdigest(),
            "record": json.loads(prior_bytes),
        },
    }
    report = _json_safe(report)
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Local e_system directory"
    )
    parser.add_argument(
        "--destination", type=Path, required=True, help="Fresh evidence directory"
    )
    parser.add_argument(
        "--prior-evidence", type=Path, help="Preserve an earlier defect record verbatim"
    )
    args = parser.parse_args(argv)
    characterize_consumer(
        args.source, args.destination, prior_evidence=args.prior_evidence
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
