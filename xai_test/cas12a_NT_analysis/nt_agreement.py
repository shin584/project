"""Model-free token IG–ISM ranks, population summaries and group diagnostics."""

from collections import Counter

import numpy as np
from scipy.stats import rankdata

from .nt_inputs import DNA_SPANS

TOKEN_SETS = {"five_6mer": list(range(5)), "nine_token": list(range(9))}
PAM_VERSIONS = ("all_nonWT", "exclude_pam_breaking")
AXES = {"density": ("ig_density", "ism_mean"), "mass": ("ig_abs_mass", "ism_sum")}
PROTOCOL = {
    "version": "cas12a-ig-ism-agreement-v1",
    "primary": "five_6mer/all_nonWT/density",
    "ties": "global exact midrank",
    "missing": "complete paired token support required; null plus reason",
    "constant": "exact constant axis: null plus reason",
    "ism": "mean eligible alternative absolute raw delta, then mean valid bases",
    "mass": "absolute signed token IG versus sum of base mean absolute deltas",
    "ig_support": "full actual token span, including PAM-excluded bases",
    "covariance": "population moments; shared global rank std product",
    "covariance_check_atol": 1e-12,
    "covariance_check_rtol": 1e-12,
    "profile": "paired per-token population means; no case-only IDs",
    "coordinates": "0-based half-open actual DNA spans; t1–t9 are 1-based token labels",
    "output_unit": "native raw fraction",
}


def _correlation(x, y, token_ids):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    reason = None
    if not valid.all():
        reason = "missing or nonfinite token axis"
    elif len(x) < 2:
        reason = "fewer than two tokens"
    elif np.ptp(x) == 0:
        reason = "constant IG axis"
    elif np.ptp(y) == 0:
        reason = "constant ISM axis"
    rho = None
    if reason is None:
        rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
        dx, dy = rx - rx.mean(), ry - ry.mean()
        rho = float(np.clip(np.mean(dx * dy) / (np.std(rx) * np.std(ry)), -1, 1))
    return {
        "rho": rho,
        "reason": reason,
        "n_total": len(x),
        "n_valid": int(valid.sum()),
        "token_ids": token_ids,
    }


def _correlations(tokens):
    return {
        name: {
            axis: _correlation(
                [tokens[i][x] for i in positions],
                [tokens[i][y] for i in positions],
                [i + 1 for i in positions],
            )
            for axis, (x, y) in AXES.items()
        }
        for name, positions in TOKEN_SETS.items()
    }


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _groups(tokens, first, names):
    x = np.asarray([token["ig_density"] for token in tokens], dtype=float)
    y = np.asarray([token["ism_mean"] for token in tokens], dtype=float)
    correlation = _correlation(x, y, list(range(1, 10)))
    positions = [np.asarray(first), np.asarray([i for i in range(9) if i not in first])]
    finite = np.isfinite(x).all() and np.isfinite(y).all()
    rx = rankdata(x, method="average") if finite else None
    ry = rankdata(y, method="average") if finite else None
    groups = [
        {
            "name": name,
            "token_ids": (index + 1).tolist(),
            "count": len(index),
            "mean_ig_rank": float(rx[index].mean()) if finite else None,
            "mean_ism_rank": float(ry[index].mean()) if finite else None,
        }
        for name, index in zip(names, positions, strict=True)
    ]
    difference = {
        axis: groups[0][f"mean_{axis}_rank"] - groups[1][f"mean_{axis}_rank"]
        if finite
        else None
        for axis in ("ig", "ism")
    }
    fields = {
        key: None
        for key in (
            "total_covariance",
            "within_covariance",
            "between_covariance",
            "global_std_product",
            "within_contribution",
            "between_contribution",
            "contribution_sum",
            "sum_residual",
            "sum_check_pass",
        )
    }
    if correlation["reason"] is None:
        total = float(np.mean((rx - rx.mean()) * (ry - ry.mean())))
        within = float(
            sum(
                len(index)
                / 9
                * np.mean(
                    (rx[index] - rx[index].mean()) * (ry[index] - ry[index].mean())
                )
                for index in positions
            )
        )
        between = float(
            sum(
                len(index)
                / 9
                * (rx[index].mean() - rx.mean())
                * (ry[index].mean() - ry.mean())
                for index in positions
            )
        )
        denominator = float(np.std(rx) * np.std(ry))
        contribution_sum = within / denominator + between / denominator
        fields = {
            "total_covariance": total,
            "within_covariance": within,
            "between_covariance": between,
            "global_std_product": denominator,
            "within_contribution": within / denominator,
            "between_contribution": between / denominator,
            "contribution_sum": contribution_sum,
            "sum_residual": contribution_sum - correlation["rho"],
            "sum_check_pass": bool(
                np.isclose(
                    contribution_sum,
                    correlation["rho"],
                    atol=PROTOCOL["covariance_check_atol"],
                    rtol=PROTOCOL["covariance_check_rtol"],
                )
            ),
        }
        if not fields["sum_check_pass"]:
            raise ValueError("global rank covariance sum check failed")
    return {
        **correlation,
        **fields,
        "groups": groups,
        "rank_mean_difference": difference,
        "ig_global_ranks": rx.tolist() if finite else None,
        "ism_global_ranks": ry.tolist() if finite else None,
    }


def sequence_agreement(
    sequence, signed_mass, raw_delta, *, spans=None, base_valid_mask=None
):
    """Compare nine actual signed token masses with raw 34 x A/C/G/T deltas."""
    spans = DNA_SPANS if spans is None else spans
    if (
        spans != DNA_SPANS
        or len(sequence) != 34
        or set(sequence) - set("ACGT")
        or sequence[4:7] != "TTT"
    ):
        raise ValueError("invalid actual Cas12a sequence/spans")
    mass, delta = (
        np.asarray(signed_mass, dtype=float),
        np.asarray(raw_delta, dtype=float),
    )
    valid = (
        np.ones((34, 4), dtype=bool)
        if base_valid_mask is None
        else np.asarray(base_valid_mask)
    )
    if (
        mass.shape != (9,)
        or delta.shape != (34, 4)
        or valid.shape != (34, 4)
        or valid.dtype != bool
    ):
        raise ValueError("invalid IG/ISM array shape or base-valid mask")
    non_wt = np.array(
        [[base != alternative for alternative in "ACGT"] for base in sequence]
    )
    if not np.all(delta[~non_wt] == 0):
        raise ValueError("WT raw delta must be zero")
    pam = non_wt.copy()
    pam[:4] = False
    pam[7:] = False
    versions = {}
    for version in PAM_VERSIONS:
        eligible = valid & non_wt
        if version == "exclude_pam_breaking":
            eligible &= ~pam
        bases = []
        for position in range(34):
            values = np.abs(delta[position, eligible[position]])
            reason = (
                "no eligible alternatives"
                if not len(values)
                else (
                    "nonfinite eligible delta"
                    if not np.isfinite(values).all()
                    else None
                )
            )
            bases.append(
                {
                    "position": position,
                    "alternative_mask": eligible[position].tolist(),
                    "n_alternatives": len(values),
                    "mean_abs_delta": float(values.mean()) if reason is None else None,
                    "reason": reason,
                }
            )
        tokens = []
        for index, (start, end) in enumerate(spans):
            support = [base for base in bases[start:end] if base["n_alternatives"]]
            reason = (
                "no valid bases"
                if not support
                else (
                    "nonfinite eligible delta"
                    if any(b["reason"] for b in support)
                    else None
                )
            )
            values = [base["mean_abs_delta"] for base in support]
            tokens.append(
                {
                    "token_id": index + 1,
                    "span": [start, end],
                    "span_length": end - start,
                    "ig_signed_mass": _finite(mass[index]),
                    "ig_abs_mass": _finite(abs(mass[index])),
                    "ig_density": _finite(abs(mass[index]) / (end - start)),
                    "valid_base_positions": [base["position"] for base in support],
                    "n_valid_bases": len(support),
                    "ism_mean": float(np.mean(values)) if reason is None else None,
                    "ism_sum": float(np.sum(values)) if reason is None else None,
                    "reason": reason,
                    "support_mismatch": len(support) != end - start,
                }
            )
        versions[version] = {
            "bases": bases,
            "tokens": tokens,
            "correlations": _correlations(tokens),
            "groups": {
                "length": _groups(tokens, list(range(5)), ("6mer", "1bp")),
                "pure_flank": _groups(
                    tokens, [6, 7, 8], ("pure_3prime_flank", "remainder")
                ),
            },
        }
    return {
        "base_valid_mask": valid.tolist(),
        "non_wt_mask": non_wt.tolist(),
        "pam_breaking_mask": pam.tolist(),
        "versions": versions,
    }


def _distribution(rows, version, token_set, axis):
    correlations = [
        row["agreement"]["versions"][version]["correlations"][token_set][axis]
        if row["agreement"] is not None
        else {"rho": None, "reason": "; ".join(row["exclusion_reasons"])}
        for row in rows
    ]
    values = sorted(item["rho"] for item in correlations if item["rho"] is not None)
    return {
        "n_total": len(rows),
        "n_valid": len(values),
        "distribution": values,
        "median": float(np.median(values)) if values else None,
        "reason": None if values else "no valid sequence correlations",
        "excluded_reasons": dict(
            sorted(
                Counter(
                    item["reason"] for item in correlations if item["rho"] is None
                ).items()
            )
        ),
    }


def _mean_profile(rows, version, token_set, axis):
    x_key, y_key = AXES[axis]
    profile = []
    for position in TOKEN_SETS[token_set]:
        pairs = []
        for row in rows:
            if row["agreement"] is None:
                continue
            token = row["agreement"]["versions"][version]["tokens"][position]
            if token[x_key] is not None and token[y_key] is not None:
                pairs.append((token[x_key], token[y_key]))
        profile.append(
            {
                "token_id": position + 1,
                "n_total": len(rows),
                "n_paired": len(pairs),
                "mean_ig": float(np.mean([p[0] for p in pairs])) if pairs else None,
                "mean_ism": float(np.mean([p[1] for p in pairs])) if pairs else None,
            }
        )
    return {
        "profile": profile,
        "correlation": _correlation(
            [p["mean_ig"] for p in profile],
            [p["mean_ism"] for p in profile],
            [p["token_id"] for p in profile],
        ),
    }


def summarize_population(rows):
    """Summarize an explicitly selected population, retaining all missing rows."""
    versions = {
        version: {
            token_set: {
                axis: {
                    "sequences": _distribution(rows, version, token_set, axis),
                    "mean_profile": _mean_profile(rows, version, token_set, axis),
                }
                for axis in AXES
            }
            for token_set in TOKEN_SETS
        }
        for version in PAM_VERSIONS
    }
    return {
        "source_ids": [row["source_row"] for row in rows],
        "n_total": len(rows),
        "n_available": sum(row["agreement"] is not None for row in rows),
        "primary": versions["all_nonWT"]["five_6mer"]["density"]["sequences"],
        "versions": versions,
    }
