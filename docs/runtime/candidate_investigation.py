"""#210 producer: budgeted IG/SHAP precision-candidate comparison.

Registered as `investigation_command` in `protocol-210.json` and run only by

    python docs/runtime/verify_contract.py investigation --work <bundle>

inside the journaled investigation stage. It reads the bundle's frozen
contract/protocol, measures fresh (cache-bypassing) explanations in the
preregistered order of `candidate_policy`, and writes `investigations.json`
plus `investigation-raw.npz` into the bundle.

Exit 0 means collection finished, including a planned budget stop. Any
exception exits 2 without writing results, so the stage is aborted and a new
bundle with full remeasurement is required (Q8); there is no partial resume.
The float32/float64 rows belong to #211 and stay unverified here.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

import candidate_policy as policy
from verification_policy import MODELS

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ROWS = tuple(f"{prefix}_{model}" for prefix in ("ig", "shap_screen", "shap_confirmation")
                       for model in policy.MODEL_ORDER)


class Budget:
    """Checked only between completed calls; a running call always finishes."""

    def __init__(self, seconds, clock):
        self.seconds, self.clock = seconds, clock
        self.start = clock()
        self.exhausted = False
        self.exhausted_offset = None

    def offset(self):
        return self.clock() - self.start

    def allows_next_call(self):
        offset = self.offset()
        if offset >= self.seconds:
            self.exhausted, self.exhausted_offset = True, offset
            return False
        return True

    def record(self):
        return {"seconds": self.seconds, "exhausted": self.exhausted,
                "exhausted_offset": self.exhausted_offset}


def collect(protocol, contract, measure, clock=time.perf_counter):
    """Run the preregistered plan under the budget; return (report, raw arrays).

    `measure(call)` returns {"attribution"|"values": array, "terms": [3]} for one
    fresh call. Selection between phases uses the same rules as the assembler.
    """
    budget = Budget(protocol["candidate_protocol"]["budget_seconds"], clock)
    raw, calls = {}, []

    def execute(plan):
        for call in plan:
            if not budget.allows_next_call():
                return False
            started = budget.offset()
            result = measure(call)
            wall = budget.offset() - started
            for field, value in result.items():
                raw[f"{call['name']}/{field}"] = np.asarray(value, dtype=np.float64)
            calls.append({"name": call["name"], "row": call["row"],
                          "started_offset": started, "wall_seconds": wall})
        return True

    running = execute(policy.initial_plan(protocol, contract))
    for model in policy.MODEL_ORDER:
        screen = policy.evaluate_screen(protocol, contract, model, raw, calls)
        candidate = screen["candidate"]
        planned = protocol["investigations"][f"shap_screen_{model}"]["planned"]
        if candidate is None or screen["measured"] < planned:
            continue  # incomplete screen or no first-stage candidate
        # Exactly identical screen measurements of this bundle fill the 3x3 grid.
        for target, source in policy.reused_arrays(protocol, contract, model, candidate).items():
            for field in ("values", "terms"):
                raw[f"{target}/{field}"] = raw[f"{source}/{field}"]
        running = running and execute(policy.confirmation_plan(protocol, contract, model, candidate))

    rows, summary = policy.evaluate(protocol, contract, raw, calls, budget.record())
    investigations = []
    for name, definition in protocol["investigations"].items():
        if name in rows:
            investigations.append(rows[name])
        else:
            investigations.append({"id": name, "state": "unverified", "planned": definition["planned"],
                                   "completed": 0, "raw_arrays": [],
                                   "reason": "#211 producer is not registered in this protocol"})
    report = {"producer": "candidate_investigation (#210)", "investigations": investigations,
              "candidate_calls": calls, "candidate_budget": budget.record(),
              "candidate_summary": summary,
              "scope": {"cases": "frozen 8 cases", "spcas9_head": contract["models"]
                        ["spcas9_non_nt"]["verified_head"]["name"],
                        "cas12a_path": "current dual-input path", "defaults_unchanged": True,
                        "not_approved_here": "all-head/path quality (#199/#200), NT (#201)"}}
    if budget.exhausted:
        report["termination"] = "planned_budget"
    return report, raw


def real_measure(contract, windows):
    """Fresh calls on the adopted runtime; disk/memory caches are bypassed."""
    sys.path.insert(0, str(ROOT))
    import non_nt_runtime as rt
    from xai import explain

    handles = {}
    for model, model_key in MODELS.items():
        spec = contract["models"][model_key]
        runtime = rt.NonNTRuntime.load(model)
        derive = rt.cas12a_derive if model == "cas12a" else None
        handles[model] = rt.explain_handle(runtime, head=spec["verified_head"]["index"],
                                           window_length=spec["inputs"][0]["shape"][0],
                                           derive=derive)
        # Registered untimed warm-up: one forward and one gradient per model.
        first = windows[model][next(iter(windows[model]))]
        handles[model].raw_fn(rt.one_hot(first)[None])
        handles[model].ig_grad_fn([x[None] for x in handles[model].ig_inputs(first)])

    def measure(call):
        handle, window = handles[call["model"]], windows[call["model"]][call["case"]]
        if call["kind"] == "ig":
            result = explain.integrated_gradients(handle, window, steps=call["steps"])
            return {"attribution": result.attribution,
                    "terms": [result.pred, result.baseline_pred, result.completeness_error]}
        result = explain.kernel_shap(handle, window, nsamples=call["nsamples"],
                                     n_background=call["n_background"],
                                     background_seed=call["background_seed"],
                                     subset_seed=call["subset_seed"], use_disk_cache=False)
        return {"values": result.values,
                "terms": [result.base_value, result.pred, result.additivity_error]}

    return measure, rt.runtime_versions()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        work = Path(argv[0]).resolve()
        contract = json.loads((work / "snapshot/contract.json").read_text(encoding="utf-8"))
        protocol = json.loads((work / "snapshot/protocol.json").read_text(encoding="utf-8"))
        windows = {model: {case["id"]: case["window"] for case in contract["cases"][model]}
                   for model in policy.MODEL_ORDER}
        measure, versions = real_measure(contract, windows)
        report, raw = collect(protocol, contract, measure)
        report["runtime"] = versions
        np.savez_compressed(work / "investigation-raw.npz", **raw)
        (work / "investigations.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0
    except BaseException as error:  # noqa: BLE001 - any abort must not look like a result
        print(f"candidate investigation aborted: {error!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
