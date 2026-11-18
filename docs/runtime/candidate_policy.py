"""Preregistered #210 candidate plan, selection and confirmation rules.

The producer (`candidate_investigation.py`) uses these functions to decide what
to measure next; the assembler uses the same functions to re-derive every
candidate row from the immutable raw arrays and call log. A producer row that
does not equal its re-derivation is an evidence failure, so a hand-edited
state, candidate or count cannot reach the report.

Scope: frozen 8 cases, the contract's verified SpCas9 head (NG) and the current
Cas12a dual-input path. Results are named only as completeness-improvement
(IG) or seed-stability-confirmed (SHAP) candidates; nothing here promotes a
default or approves other heads/paths (#199/#200/#201).
"""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np

MODEL_ORDER = ("spcas9", "cas12a")
LABELS = {"ig": "completeness_improvement_candidate",
          "screen": "first_stage_candidate",
          "confirmation": "seed_stability_confirmed_candidate"}


# ── naming ──────────────────────────────────────────────────────────────

def ig_name(model, case, steps):
    return f"ig/{model}/{case}/steps{steps}"


def screen_name(model, case, ns, bg, b, s):
    return f"shap/{model}/{case}/ns{ns}/bg{bg}/background{b}/subset{s}"


def confirmation_name(model, case, b, s):
    return f"confirmation/{model}/{case}/background{b}/subset{s}"


def fresh_name(model, case):
    return f"confirmation/{model}/{case}/fresh00"


def setting_key(ns, bg):
    return f"ns{ns}/bg{bg}"


def cases(contract, model):
    return [case["id"] for case in contract["cases"][model]]


# ── preregistered plan ──────────────────────────────────────────────────

def screen_settings(protocol):
    """(ns, bg, measured axes) in a fixed order; the intersection is shared."""
    cp = protocol["candidate_protocol"]
    sample_bg, background_ns = cp["sample_axis"]["n_background"], cp["background_axis"]["nsamples"]
    found = {}
    for ns in cp["shap_nsamples"]:
        found.setdefault((ns, sample_bg), []).append("subset")
    for bg in cp["shap_backgrounds"]:
        found.setdefault((background_ns, bg), []).append("background")
    return [(ns, bg, axes) for (ns, bg), axes in found.items()]


def axis_runs(protocol, ns, bg, axis):
    """(background seed, subset seed) pairs measured on one axis of a setting."""
    cp = protocol["candidate_protocol"]
    if axis == "subset":
        return [(cp["sample_axis"]["background_seed"], s) for s in cp["seeds"]]
    return [(b, cp["background_axis"]["subset_seed"]) for b in cp["seeds"]]


def setting_runs(protocol, ns, bg, axes):
    runs = []
    for axis in axes:
        runs += [run for run in axis_runs(protocol, ns, bg, axis) if run not in runs]
    return runs


def initial_plan(protocol, contract):
    """IG for both models, then the SHAP screen for both models (case-major)."""
    cp = protocol["candidate_protocol"]
    plan = []
    for model in MODEL_ORDER:
        for case in cases(contract, model):
            for steps in cp["ig_steps"]:
                plan.append({"kind": "ig", "row": f"ig_{model}", "model": model, "case": case,
                             "steps": steps, "name": ig_name(model, case, steps)})
    for model in MODEL_ORDER:
        for case in cases(contract, model):
            seen = set()
            for axis in ("subset", "background"):
                for ns, bg, axes in screen_settings(protocol):
                    if axis not in axes:
                        continue
                    for b, s in axis_runs(protocol, ns, bg, axis):
                        name = screen_name(model, case, ns, bg, b, s)
                        if name in seen:
                            continue  # the shared intersection is measured once
                        seen.add(name)
                        plan.append({"kind": "shap", "row": f"shap_screen_{model}", "model": model,
                                     "case": case, "nsamples": ns, "n_background": bg,
                                     "background_seed": b, "subset_seed": s, "name": name})
    return plan


def confirmation_plan(protocol, contract, model, candidate):
    """New 3x3 combinations (not measured in the screen), then one fresh (0,0)."""
    cp = protocol["candidate_protocol"]
    ns, bg = candidate["nsamples"], candidate["n_background"]
    measured = set(setting_runs(protocol, ns, bg, candidate["axes"]))
    plan = []
    for case in cases(contract, model):
        for b in cp["seeds"]:
            for s in cp["seeds"]:
                if (b, s) not in measured:
                    plan.append({"kind": "shap", "row": f"shap_confirmation_{model}",
                                 "model": model, "case": case, "nsamples": ns,
                                 "n_background": bg, "background_seed": b, "subset_seed": s,
                                 "name": confirmation_name(model, case, b, s)})
        plan.append({"kind": "shap", "row": f"shap_confirmation_{model}", "model": model,
                     "case": case, "nsamples": ns, "n_background": bg,
                     "background_seed": 0, "subset_seed": 0, "name": fresh_name(model, case)})
    return plan


def reused_arrays(protocol, contract, model, candidate):
    """Confirmation names filled from identical screen measurements in this bundle."""
    ns, bg = candidate["nsamples"], candidate["n_background"]
    mapping = {}
    for case in cases(contract, model):
        for b, s in setting_runs(protocol, ns, bg, candidate["axes"]):
            mapping[confirmation_name(model, case, b, s)] = screen_name(model, case, ns, bg, b, s)
    return mapping


# ── numerical rules (frozen contract numbers) ───────────────────────────

def pearson(a, b):
    """None when undefined (constant/nonfinite/mismatched); never passes."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    if a.std() == 0 or b.std() == 0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else None


def finite(*values):
    return all(np.isfinite(np.asarray(v, dtype=np.float64)).all() for v in values)


def json_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def pair_checks(runs, reference, contract):
    """Every pair of `runs` against the seed-(0,0) scale of `reference`."""
    cross = contract["criteria"]["shap_repeat_stability"]["cross_seed"]
    bound = 0.25 * float(np.max(np.abs(reference))) if finite(reference) else math.nan
    rows, passed = [], True
    for a, b in combinations(sorted(runs), 2):
        r = pearson(runs[a], runs[b])
        diff = float(np.max(np.abs(np.asarray(runs[a]) - np.asarray(runs[b]))))
        ok = (r is not None and r >= cross["min_pairwise_pearson_r"]
              and math.isfinite(diff) and math.isfinite(bound) and diff <= bound)
        passed &= ok
        rows.append({"pair": [a, b], "pearson_r": r, "max_abs_diff": json_float(diff),
                     "bound": json_float(bound), "passed": ok})
    return passed, rows


def shap_run_ok(entry, model, contract):
    bound = contract["criteria"]["shap_additivity"]["max_abs_error"][model]
    terms = np.asarray(entry["terms"], dtype=np.float64)
    return bool(finite(entry["values"], terms) and terms.shape == (3,) and terms[2] <= bound)


# ── evaluation ──────────────────────────────────────────────────────────

def _wall(calls):
    return {call["name"]: call["wall_seconds"] for call in calls}


def _arrays(raw, prefix, fields):
    if any(f"{prefix}/{field}" not in raw for field in fields):
        return None
    return {field: raw[f"{prefix}/{field}"] for field in fields}


def evaluate_ig(protocol, contract, model, raw, calls):
    cp = protocol["candidate_protocol"]
    wall = _wall(calls)
    bound = contract["criteria"]["ig_completeness"]["max_abs_error"][model]
    stability = contract["criteria"]["ig_resolution_stability"]
    settings, measured, arrays = [], 0, []
    for steps in cp["ig_steps"]:
        rows, ok, times = [], True, []
        for case in cases(contract, model):
            name = ig_name(model, case, steps)
            entry = _arrays(raw, name, ("attribution", "terms"))
            if entry is None or name not in wall:
                rows.append({"case": case, "measured": False})
                ok = False
                continue
            measured += 1
            arrays += [f"{name}/attribution", f"{name}/terms"]
            terms = np.asarray(entry["terms"], dtype=np.float64)
            fine = finite(entry["attribution"], terms)
            passed = bool(fine and terms[2] <= bound)
            ok &= passed
            times.append(wall[name])
            rows.append({"case": case, "measured": True, "finite": fine,
                         "completeness_error": json_float(terms[2]), "bound": bound,
                         "passed": passed, "wall_seconds": wall[name]})
        settings.append({"steps": steps, "eligible": ok, "cases": rows,
                         "max_wall_seconds": max(times) if times else None})
    eligible = [row for row in settings if row["eligible"]]
    candidate = None
    if eligible:
        best = min(eligible, key=lambda row: (row["max_wall_seconds"], row["steps"]))
        candidate = {"steps": best["steps"], "max_wall_seconds": best["max_wall_seconds"],
                     "label": LABELS["ig"]}
    # Separate resolution report: each lower step count against the largest one.
    top = max(cp["ig_steps"])
    resolution = []
    for steps in cp["ig_steps"]:
        if steps == top:
            continue  # 256 against itself is not evidence of its resolution
        for case in cases(contract, model):
            low = _arrays(raw, ig_name(model, case, steps), ("attribution",))
            high = _arrays(raw, ig_name(model, case, top), ("attribution",))
            if low is None or high is None:
                resolution.append({"steps": steps, "case": case, "state": "unverified"})
                continue
            coarse = np.asarray(low["attribution"], dtype=np.float64)
            ref = np.asarray(high["attribution"], dtype=np.float64)
            scale = float(np.max(np.abs(ref)))
            r = pearson(coarse, ref)
            diff = float(np.max(np.abs(coarse - ref)))
            strong = np.abs(ref) >= 0.1 * scale
            signs = (float(np.mean(np.sign(coarse[strong]) == np.sign(ref[strong])))
                     if strong.any() else 1.0)
            passed = bool(r is not None and r >= stability["min_pearson_r"]
                          and math.isfinite(diff) and diff <= 0.05 * scale and signs == 1.0)
            resolution.append({"steps": steps, "against": top, "case": case,
                               "pearson_r": r, "max_abs_diff": json_float(diff),
                               "bound": json_float(0.05 * scale),
                               "sign_agreement": json_float(signs),
                               "state": "pass" if passed else "fail"})
    return {"settings": settings, "candidate": candidate, "resolution": resolution,
            "measured": measured, "arrays": arrays}


def screen_setting(protocol, contract, model, raw, wall, ns, bg, axes):
    rows, ok, times = [], True, []
    runs = setting_runs(protocol, ns, bg, axes)
    unmeasured_axes = [axis for axis in ("subset", "background") if axis not in axes]
    for case in cases(contract, model):
        entries = {}
        for b, s in runs:
            name = screen_name(model, case, ns, bg, b, s)
            entry = _arrays(raw, name, ("values", "terms"))
            if entry is not None and name in wall:
                entries[(b, s)] = entry
                times.append(wall[name])
        if len(entries) != len(runs):
            rows.append({"case": case, "measured": False})
            ok = False
            continue
        run_ok = all(shap_run_ok(entry, model, contract) for entry in entries.values())
        axis_rows, axis_ok = {}, True
        for axis in axes:
            subset = {f"background{b}/subset{s}": entries[(b, s)]["values"]
                      for b, s in axis_runs(protocol, ns, bg, axis)}
            passed, pairs = pair_checks(subset, entries[(0, 0)]["values"], contract)
            axis_ok &= passed
            axis_rows[axis] = {"passed": passed, "pairs": pairs}
        passed = bool(run_ok and axis_ok)
        ok &= passed
        rows.append({"case": case, "measured": True, "finite_and_additive": run_ok,
                     "axes": axis_rows, "passed": passed})
    return {"setting": setting_key(ns, bg), "nsamples": ns, "n_background": bg,
            "axes": list(axes), "unverified_axes": unmeasured_axes,
            "eligible": ok, "cases": rows, "max_wall_seconds": max(times) if times else None}


def evaluate_screen(protocol, contract, model, raw, calls):
    wall = _wall(calls)
    settings = [screen_setting(protocol, contract, model, raw, wall, ns, bg, axes)
                for ns, bg, axes in screen_settings(protocol)]
    names = {call["name"] for call in calls if call["row"] == f"shap_screen_{model}"}
    arrays = sorted(f"{name}/{field}" for name in names for field in ("values", "terms"))
    eligible = [row for row in settings if row["eligible"]]
    candidate = None
    if eligible:
        best = min(eligible, key=lambda row: (row["max_wall_seconds"],
                                              row["nsamples"] * row["n_background"],
                                              row["nsamples"], row["n_background"]))
        candidate = {"setting": best["setting"], "nsamples": best["nsamples"],
                     "n_background": best["n_background"], "axes": best["axes"],
                     "max_wall_seconds": best["max_wall_seconds"], "label": LABELS["screen"]}
    return {"settings": settings, "candidate": candidate, "measured": len(names), "arrays": arrays}


def evaluate_confirmation(protocol, contract, model, raw, calls, candidate):
    cp = protocol["candidate_protocol"]
    wall = _wall(calls)
    determinism = contract["criteria"]["shap_repeat_stability"]["same_seed_determinism"]["max_abs_diff"]
    reused = reused_arrays(protocol, contract, model, candidate)
    planned_names = {call["name"] for call in confirmation_plan(protocol, contract, model, candidate)}
    measured = sorted(name for name in planned_names if name in wall)
    rows, ok = [], True
    arrays = []
    for case in cases(contract, model):
        entries = {}
        for b in cp["seeds"]:
            for s in cp["seeds"]:
                name = confirmation_name(model, case, b, s)
                entry = _arrays(raw, name, ("values", "terms"))
                if entry is not None and (name in wall or name in reused):
                    entries[f"background{b}/subset{s}"] = entry
                    arrays += [f"{name}/values", f"{name}/terms"]
        fresh = _arrays(raw, fresh_name(model, case), ("values", "terms"))
        if fresh is not None and fresh_name(model, case) in wall:
            arrays += [f"{fresh_name(model, case)}/values", f"{fresh_name(model, case)}/terms"]
        if len(entries) != len(cp["seeds"]) ** 2 or fresh is None or fresh_name(model, case) not in wall:
            rows.append({"case": case, "measured": False})
            ok = False
            continue
        run_ok = all(shap_run_ok(entry, model, contract) for entry in [*entries.values(), fresh])
        reference = entries["background0/subset0"]["values"]
        pairs_ok, pairs = pair_checks({k: v["values"] for k, v in entries.items()}, reference, contract)
        fresh_diff = float(np.max(np.abs(np.asarray(fresh["values"]) - np.asarray(reference))))
        fresh_ok = math.isfinite(fresh_diff) and fresh_diff <= determinism
        passed = bool(run_ok and pairs_ok and fresh_ok and len(pairs) == 36)
        ok &= passed
        rows.append({"case": case, "measured": True, "finite_and_additive": run_ok,
                     "pairs_passed": pairs_ok, "n_pairs": len(pairs), "pairs": pairs,
                     "fresh00_max_abs_diff": json_float(fresh_diff),
                     "fresh00_bound": determinism, "passed": passed})
    return {"cases": rows, "confirmed": ok, "measured": len(measured),
            "planned": len(planned_names), "reused_from": reused, "arrays": sorted(set(arrays))}


def _incomplete(row, budget):
    row.update(state="time_limited", termination="planned_budget",
               reason=f"Candidate budget of {budget['seconds']} s exhausted between completed "
                      "calls; unstarted work remains unverified")


def evaluate(protocol, contract, raw, calls, budget):
    """Re-derive every candidate row and check the call log against the plan.

    `raw` maps array names to arrays, `calls` is the ordered call log (name, row,
    started_offset, wall_seconds) and `budget` records seconds/exhausted/
    exhausted_offset. Raises ValueError when the log is not the preregistered
    sequence, a call started after exhaustion, or an incomplete run claims no
    planned budget stop.
    """
    cp = protocol["candidate_protocol"]
    investigations = protocol["investigations"]
    seconds = cp["budget_seconds"]
    if budget.get("seconds") != seconds:
        raise ValueError("Candidate budget differs from the registered protocol")
    for call in calls:
        if not (call["started_offset"] < seconds and call["wall_seconds"] >= 0
                and math.isfinite(call["wall_seconds"])):
            raise ValueError(f"Candidate call started after the budget boundary: {call['name']}")

    plan = initial_plan(protocol, contract)
    rows, selections = {}, {}
    for model in MODEL_ORDER:
        ig = evaluate_ig(protocol, contract, model, raw, calls)
        row = {"id": f"ig_{model}", "planned": investigations[f"ig_{model}"]["planned"],
               "completed": ig["measured"], "raw_arrays": ig["arrays"],
               "settings": ig["settings"], "resolution": ig["resolution"],
               "candidate": ig["candidate"]}
        if ig["measured"] < row["planned"]:
            _incomplete(row, budget)
            row["candidate"] = None
        elif ig["candidate"]:
            row["state"] = "pass"
        else:
            row.update(state="no_eligible_candidate",
                       reason="No IG setting was finite and complete on all four cases")
        rows[row["id"]] = row
    for model in MODEL_ORDER:
        screen = evaluate_screen(protocol, contract, model, raw, calls)
        ident = f"shap_screen_{model}"
        row = {"id": ident, "planned": investigations[ident]["planned"],
               "completed": screen["measured"], "raw_arrays": screen["arrays"],
               "settings": screen["settings"], "candidate": screen["candidate"]}
        if screen["measured"] < row["planned"]:
            _incomplete(row, budget)
            row["candidate"] = None
        elif screen["candidate"]:
            row["state"] = "pass"
        else:
            row.update(state="no_eligible_candidate",
                       reason="No SHAP setting met the criteria on every measured axis of all four cases")
        rows[ident] = row
        selections[model] = row["candidate"]
    for model in MODEL_ORDER:
        ident = f"shap_confirmation_{model}"
        definition = investigations[ident]
        candidate = selections[model]
        row = {"id": ident, "planned": definition["planned"], "completed": 0, "raw_arrays": []}
        if rows[f"shap_screen_{model}"]["state"] == "time_limited":
            _incomplete(row, budget)
        elif candidate is None:
            row.update(state="not_applicable",
                       reason="No first-stage SHAP candidate; confirmation is conditional")
        else:
            plan += confirmation_plan(protocol, contract, model, candidate)
            confirmation = evaluate_confirmation(protocol, contract, model, raw, calls, candidate)
            row.update(planned=definition.get("planned_by_candidate", {}).get(
                           candidate["setting"], definition["planned"]),
                       completed=confirmation["measured"], raw_arrays=confirmation["arrays"],
                       candidate={k: candidate[k] for k in ("setting", "nsamples", "n_background")},
                       reused_from=confirmation["reused_from"], cases=confirmation["cases"])
            if confirmation["planned"] != row["planned"]:
                raise ValueError(f"Registered confirmation count mismatch: {ident}")
            if confirmation["measured"] < row["planned"]:
                _incomplete(row, budget)
            elif confirmation["confirmed"]:
                row.update(state="pass", label=LABELS["confirmation"])
            else:
                row.update(state="no_eligible_candidate",
                           reason="Candidate confirmation failed; no other candidate is tested")
        rows[ident] = row

    names = [call["name"] for call in calls]
    expected = [call["name"] for call in plan]
    if names != expected[:len(names)]:
        raise ValueError("Candidate calls deviate from the preregistered order/inventory")
    shap_calls = sum(1 for call in calls if call["name"].startswith(("shap/", "confirmation/")))
    if shap_calls > cp["max_shap_calls"]:
        raise ValueError("SHAP call ceiling exceeded")
    stopped = len(names) < len(expected)
    if stopped != bool(budget.get("exhausted")):
        raise ValueError("Unfinished candidate work without a planned budget stop")
    if stopped:
        last = max([c["started_offset"] + c["wall_seconds"] for c in calls], default=0.0)
        offset = budget.get("exhausted_offset")
        if not (isinstance(offset, (int, float)) and offset >= seconds and offset >= last):
            raise ValueError("Budget stop is not supported by the call log")
    summary = {"unstarted_calls": expected[len(names):], "shap_calls": shap_calls,
               "planned_calls": len(expected),
               "budget_seconds": seconds}
    return rows, summary
