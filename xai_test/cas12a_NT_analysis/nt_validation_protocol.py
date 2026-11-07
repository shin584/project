"""Pre-observation validation protocol and model-free budget arithmetic."""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cas12a.cnn_manifest import verify_manifest
from cas12a.nt_data import digest

from .nt_run import LIMIT_SECONDS

PRESET = {
    "correctness_rows": 8,
    "warmup_rows": 2,
    "cost_rows": 32,
    "cost_repeats": 3,
    "batch_size": 32,
    "integration_batch_size": 16,
    "rollout_batch_size": 1,
    "ig_policy": {
        "completeness_atol": 1e-4,
        "completeness_rtol": 0.05,
        "near_zero": 1e-6,
        "convergence_atol": 1e-4,
        "convergence_rtol": 0.05,
        "max_steps": 1600,
    },
    "rollout_policy": {"query_l1_threshold": 0.01},
    "reservations": {"correctness": 1800, "warmup": 900, "cost": 1800},
    "forecast": {
        "factor": 1.5,
        "fixed_seconds": 600,
        "sessions_per_method": 2,
        "reservation_quantum": 60,
        "case_allowance": 15,
        "fallback_floor": 400,
    },
}


def code_snapshot():
    from .nt_predict import source_snapshot

    return source_snapshot()


def snapshot_digest(snapshot):
    return digest(
        {name: hashlib.sha256(data).hexdigest() for name, data in snapshot.items()}
    )


def prepare_protocol(partition_path, *, fixture=False):
    """Choose IDs from existing validation membership, without model/data outputs."""
    data = Path(partition_path).read_bytes()
    partition = json.loads(data)
    verify_manifest(partition)
    if (partition["run_kind"] == "fixture") != fixture:
        raise ValueError("protocol fixture/production partition mismatch")
    config = copy.deepcopy(PRESET)
    if fixture:
        config.update(correctness_rows=1, warmup_rows=1, cost_rows=1, cost_repeats=2)
        config.update(batch_size=1, integration_batch_size=200)
        config["ig_policy"].update(
            completeness_atol=0.1,
            completeness_rtol=0.1,
            convergence_atol=0.1,
            convergence_rtol=0.1,
            max_steps=400,
        )
        config["reservations"] = {"correctness": 300, "warmup": 300, "cost": 300}
    ordered = sorted(partition["partitions"]["validation"])
    needed = (
        config["correctness_rows"]
        + config["warmup_rows"]
        + config["cost_rows"] * config["cost_repeats"]
    )
    if len(ordered) < needed:
        raise ValueError("not enough existing validation rows for disjoint probes")
    permutation = np.random.Generator(np.random.PCG64(42)).permutation(len(ordered))
    selected = [ordered[int(i)] for i in permutation[:needed]]
    groups, offset = {}, 0
    counts = {
        "correctness": config["correctness_rows"],
        "warmup": config["warmup_rows"],
    }
    counts.update(
        {f"cost-{i + 1}": config["cost_rows"] for i in range(config["cost_repeats"])}
    )
    for name, count in counts.items():
        groups[name] = sorted(selected[offset : offset + count])
        offset += count
    return {
        "schema": 1,
        "kind": "cas12a-validation-146",
        "fixture": fixture,
        "partition_file_sha256": hashlib.sha256(data).hexdigest(),
        "partition_sha256": partition["partition_sha256"],
        "source_sha256": partition["source_file_sha256"],
        "test_count": partition["counts"]["test"],
        "selection": "ascending validation IDs; PCG64(42) permutation; disjoint groups",
        "groups": groups,
        "phase_order": list(groups),
        "config": config,
        "code_sha256": snapshot_digest(code_snapshot()),
        "historical_fp16_test_known": True,
    }


def forecast(rates, *, population, charged_seconds, policy):
    """Full first, then largest affordable random N plus 15 case-only rows.

    Rates describe whole-source work: ISM already includes all 102 mutants;
    IG is scaled to the entire frozen maximum-resolution retry ladder.
    """
    keys = {"prediction", "ism", "rollout", "ig", "common", "loading"}
    if set(rates) != keys or any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v < 0
        for v in [*rates.values(), charged_seconds]
    ):
        raise ValueError("forecast requires finite nonnegative measured rates/charges")
    if type(population) is not int or population < 1:
        raise ValueError("invalid Test population")
    if policy != PRESET["forecast"]:
        raise ValueError("forecast policy differs from frozen engineering preset")

    def estimate(size, extra):
        sessions = policy["sessions_per_method"]
        quantum = policy["reservation_quantum"]
        plan = {}
        for method in ("prediction", "ism", "rollout", "ig"):
            count = size + extra if method == "ig" else population
            measured = (
                rates[method] * count
                + rates["common"] * population / 4
                + sessions * rates["loading"]
            )
            projected = policy["factor"] * measured + policy["fixed_seconds"] / 4
            reservation = math.ceil(projected / sessions / quantum) * quantum
            plan[method] = {
                "sessions": sessions,
                "per_session_seconds": reservation,
                "charged_seconds": sessions * reservation,
            }
        return {
            "size": size,
            "additional_case_allowance": extra,
            "reservation_plan": plan,
            "total_seconds": charged_seconds
            + sum(p["charged_seconds"] for p in plan.values()),
        }

    full = estimate(population, 0)
    result = {
        "schema": 1,
        "rates": rates,
        "charged_seconds": charged_seconds,
        "limit_seconds": LIMIT_SECONDS,
        "population": population,
        "full": full,
        "full_rejected": full["total_seconds"] > LIMIT_SECONDS,
        "mode": None,
        "selected": None,
    }
    if not result["full_rejected"]:
        result.update(mode="full", selected=full)
        return result
    for size in range(population - 1, policy["fallback_floor"] - 1, -1):
        selected = estimate(size, policy["case_allowance"])
        if selected["total_seconds"] <= LIMIT_SECONDS:
            result.update(mode="subset", selected=selected)
            break
    return result
