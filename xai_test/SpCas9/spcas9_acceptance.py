"""Fixed #79 validation comparisons over persisted measured-row predictions.

No inference or training happens here. Head means and pooled correlations are
separate, and the supplementary cluster interval never participates in a gate.
"""

import math

import numpy as np
import pandas as pd
from leakage_check import MIN_N_FOR_SPEARMAN
from paired_bootstrap import joint_clustered_bootstrap_ci
from scipy.stats import rankdata

from SpCas9.spcas9_data import HEAD_NAMES, is_ngg, protospacers
from SpCas9.spcas9_metrics import (
    full_metrics,
    head_arrays,
    population_metrics,
    pred_column,
    target_counts,
    true_column,
)

SEEDS = (42, 43, 44)
METRICS = ("mae", "mse", "spearman", "pearson")
GATE_SETTINGS = {
    "relative_ngg_mae_reduction": 0.02,
    "ngg_spearman_increase": 0.02,
    "maximum_relative_mae_increase": 0.05,
    "maximum_spearman_decrease": 0.02,
}
BOOTSTRAP_SETTINGS = {
    "level": 0.95,
    "seed": 20261027,
    "valid_draws": 2000,
    "maximum_attempts": 20000,
}


def _mean(values: list) -> float | None:
    if not values or any(v is None or not math.isfinite(v) for v in values):
        return None
    return math.fsum(values) / len(values)


def _score(frame: pd.DataFrame) -> dict:
    frame = frame.reindex(
        columns=list(
            dict.fromkeys(
                [
                    *frame.columns,
                    *(true_column(h) for h in HEAD_NAMES),
                    *(pred_column(h) for h in HEAD_NAMES),
                ]
            )
        )
    )
    labels, predictions = [], []
    heads = {}
    for head in HEAD_NAMES:
        rows = frame[frame[true_column(head)].notna()]
        y, p = head_arrays(rows, head)
        heads[head] = {**full_metrics(y, p), **target_counts(rows)}
        labels.append(y)
        predictions.append(p)
    return {
        "counts": target_counts(frame),
        "heads": heads,
        "pooled": full_metrics(np.concatenate(labels), np.concatenate(predictions)),
        "populations": population_metrics(frame),
    }


def _aggregate(scores: dict, seeds: tuple) -> dict:
    populations = {}
    for population in ("ngg", "non_ngg"):
        heads = {}
        for head in HEAD_NAMES:
            heads[head] = {
                metric: _mean(
                    [
                        scores[seed]["populations"][population]["heads"][head][metric]
                        if seed in scores
                        else None
                        for seed in seeds
                    ]
                )
                for metric in METRICS
            }
        populations[population] = {"heads": heads}
    return {
        "seeds": list(seeds),
        "populations": populations,
        "equal_head_ngg": {
            metric: _mean([populations["ngg"]["heads"][h][metric] for h in HEAD_NAMES])
            for metric in ("mae", "spearman")
        },
    }


def _comparison(candidate: dict, reference: dict) -> dict:
    checks = []

    def check(name: str, c, b, *, metric: str, factor: float):
        reason: str | None
        if c is None or b is None:
            passed = None
            reason = "missing required seed result or undefined required metric"
        else:
            passed = c <= b * factor if metric == "mae" else c >= b + factor
            reason = None if passed else "frozen point-estimate threshold exceeded"
        change = None
        if c is not None and b is not None:
            if metric != "mae":
                change = c - b
            elif b != 0:
                change = (c - b) / b
            elif c == 0:
                change = 0.0
        checks.append(
            {
                "name": name,
                "candidate": c,
                "reference": b,
                "change": change,
                "passed": passed,
                "reason": reason,
            }
        )

    c, b = candidate["equal_head_ngg"], reference["equal_head_ngg"]
    check(
        "equal_head_ngg.mae",
        c["mae"],
        b["mae"],
        metric="mae",
        factor=1 - GATE_SETTINGS["relative_ngg_mae_reduction"],
    )
    check(
        "equal_head_ngg.spearman",
        c["spearman"],
        b["spearman"],
        metric="spearman",
        factor=GATE_SETTINGS["ngg_spearman_increase"],
    )
    for population in ("ngg", "non_ngg"):
        for head in HEAD_NAMES:
            c = candidate["populations"][population]["heads"][head]
            b = reference["populations"][population]["heads"][head]
            check(
                f"{population}.{head}.mae",
                c["mae"],
                b["mae"],
                metric="mae",
                factor=1 + GATE_SETTINGS["maximum_relative_mae_increase"],
            )
            check(
                f"{population}.{head}.spearman",
                c["spearman"],
                b["spearman"],
                metric="spearman",
                factor=-GATE_SETTINGS["maximum_spearman_decrease"],
            )
    decision = (
        "incomplete"
        if any(c["passed"] is None for c in checks)
        else "pass"
        if all(c["passed"] for c in checks)
        else "fail"
    )
    return {
        "decision": decision,
        "checks": checks,
        "candidate": candidate,
        "reference": reference,
    }


def bootstrap_intervals(runs: dict, settings: dict) -> dict:
    configuration = {**BOOTSTRAP_SETTINGS, **settings}
    result = {
        "configuration": configuration,
        "unit": "NGG protospacer; retain all observations and cluster multiplicity",
        "scope": "supplementary 95% percentile intervals, conditional on fixed seeds; never an adoption gate",
    }
    if any(s not in runs.get(r, {}) for r in ("baseline", "candidate") for s in SEEDS):
        return {
            **result,
            "status": "unavailable",
            "reason": "missing required paired seed predictions",
            "valid_draws": 0,
            "attempts": 0,
            "invalid_draws": 0,
            "intervals": None,
        }
    required = {true_column(h) for h in HEAD_NAMES} | {
        pred_column(h) for h in HEAD_NAMES
    }
    if any(
        not required <= set(frame)
        for seeds in runs.values()
        for frame in seeds.values()
    ):
        return {
            **result,
            "status": "unavailable",
            "reason": "missing required head label/prediction columns",
            "valid_draws": 0,
            "attempts": 0,
            "invalid_draws": 0,
            "intervals": None,
        }
    base = runs["baseline"][42]
    mask = is_ngg(base).to_numpy()
    frames = [
        runs[r][s].loc[mask].reset_index(drop=True)
        for r in ("baseline", "candidate")
        for s in SEEDS
    ]
    labels = np.column_stack(
        [frames[0][true_column(h)].to_numpy(dtype=float) for h in HEAD_NAMES]
    )
    predicted = np.stack(
        [
            np.column_stack([f[f"{h}_pred"].to_numpy(dtype=float) for h in HEAD_NAMES])
            for f in frames
        ]
    )

    def statistic(idx: np.ndarray) -> tuple[float, float] | None:
        maes, correlations = [], []
        for h in range(len(HEAD_NAMES)):
            y = labels[idx, h]
            measured = ~np.isnan(y)
            y = y[measured]
            p = predicted[:, idx, h][:, measured].T
            if (
                len(y) < MIN_N_FOR_SPEARMAN
                or not np.isfinite(y).all()
                or not np.isfinite(p).all()
            ):
                return None
            # Compute all six models together with average-tie ranks, exactly
            # the measured-row Spearman definition used by full_metrics.
            yr = rankdata(y)
            pr = rankdata(p, axis=0)
            yr -= yr.mean()
            pr -= pr.mean(axis=0)
            denom = np.sqrt(np.sum(yr**2) * np.sum(pr**2, axis=0))
            if np.any(denom == 0):
                return None
            correlations.append(np.sum(yr[:, None] * pr, axis=0) / denom)
            maes.append(np.mean(np.abs(p - y[:, None]), axis=0))
        # Columns are baseline 42/43/44, candidate 42/43/44. First
        # average seeds for each head, then give the nine heads equal weight.
        mae = np.array(maes).reshape(len(HEAD_NAMES), 2, 3).mean(axis=2).mean(axis=0)
        rho = (
            np.array(correlations)
            .reshape(len(HEAD_NAMES), 2, 3)
            .mean(axis=2)
            .mean(axis=0)
        )
        if mae[0] == 0:
            return None  # A relative change of a zero denominator is undefined.
        return float((mae[1] - mae[0]) / mae[0]), float(rho[1] - rho[0])

    sampled = joint_clustered_bootstrap_ci(
        protospacers(base.loc[mask]).to_numpy(),
        statistic,
        n_resamples=configuration["valid_draws"],
        maximum_attempts=configuration["maximum_attempts"],
        ci=configuration["level"],
        seed=configuration["seed"],
    )
    intervals = sampled.pop("intervals")
    return {
        **result,
        **sampled,
        "intervals": None
        if intervals is None
        else dict(zip(("relative_ngg_mae_change", "ngg_spearman_change"), intervals)),
    }


def build_report(
    runs: dict[str, dict[int, pd.DataFrame]],
    stored_v2: pd.DataFrame | None,
    *,
    bootstrap: bool = True,
    bootstrap_settings: dict | None = None,
) -> dict:
    """Study reporting boundary: both fixed comparisons, without candidate Test."""
    return _build_report(
        runs,
        stored_v2,
        split="val",
        bootstrap=bootstrap,
        bootstrap_settings=bootstrap_settings,
    )


def _build_report(
    runs: dict[str, dict[int, pd.DataFrame]],
    stored_v2: pd.DataFrame | None,
    *,
    split: str,
    bootstrap: bool = True,
    bootstrap_settings: dict | None = None,
) -> dict:
    """Shared arithmetic; Test inference remains owned by the finalist guard."""
    if split not in ("val", "test"):
        raise ValueError("unknown comparison split")
    frames = [f for r in ("baseline", "candidate") for f in runs.get(r, {}).values()]
    if stored_v2 is not None:
        frames.append(stored_v2)
    for frame in frames:
        if "split" not in frame or not frame["split"].eq(split).all():
            raise PermissionError(
                f"only {'validation' if split == 'val' else 'Test'} predictions may enter this report"
            )
        if frame["row_index"].isna().any() or frame["row_index"].duplicated().any():
            raise ValueError("prediction rows require unique aligned row identities")
        if not frame["sequence"].str.fullmatch("[ACGT]{30}").all():
            raise ValueError("prediction rows require normalized 30bp sequences")
        columns = ["row_index", "sequence", "split"] + [
            true_column(h)
            for h in HEAD_NAMES
            if true_column(h) in frames[0] and true_column(h) in frame
        ]
        try:
            pd.testing.assert_frame_equal(
                frame[columns].reset_index(drop=True),
                frames[0][columns].reset_index(drop=True),
                check_dtype=False,
                rtol=1e-12,
                atol=1e-14,
            )
        except AssertionError as error:
            raise ValueError(
                "paired predictions must align on rows, sequences and measured labels"
            ) from error
    scored = {
        recipe: {s: _score(frame) for s, frame in runs.get(recipe, {}).items()}
        for recipe in ("baseline", "candidate")
    }
    stored = _score(stored_v2) if stored_v2 is not None else None
    comparisons = {
        "paired_seeds": _comparison(
            _aggregate(scored["candidate"], SEEDS),
            _aggregate(scored["baseline"], SEEDS),
        ),
        "representative": _comparison(
            _aggregate(scored["candidate"], (42,)),
            _aggregate({42: stored} if stored is not None else {}, (42,)),
        ),
    }
    decisions = [c["decision"] for c in comparisons.values()]
    decision = (
        "incomplete"
        if "incomplete" in decisions
        else "pass"
        if all(d == "pass" for d in decisions)
        else "fail"
    )
    return {
        "schema_version": 1,
        "issue": 86,
        "split": split,
        "decision": decision,
        "candidate_test_eligible": decision == "pass",
        "test_access": "requires separate finalist freeze; this report never reads candidate Test",
        "aggregation": "metric mean over required seeds per head/population, then equal-head NGG mean; relative MAE compares means",
        "settings": GATE_SETTINGS,
        "runs": scored,
        "stored_v2": stored,
        "comparisons": comparisons,
        "bootstrap": bootstrap_intervals(runs, bootstrap_settings or {})
        if bootstrap
        else {
            "status": "unavailable",
            "reason": "bootstrap not requested",
            "configuration": BOOTSTRAP_SETTINGS,
            "valid_draws": 0,
            "attempts": 0,
            "invalid_draws": 0,
            "intervals": None,
        },
        "attribution": {
            "paired_seeds": "ranking auxiliary contribution under common checkpoint selection, stopping and LR policy; fixed seeds only",
            "representative": "single candidate seed42 vs single stored v2; combines ranking and checkpoint selection/stopping/LR policy changes",
        },
    }
