"""Explicit real-model #219 check; run separately from regressions.

python -X utf8 tests/stall_limit_probe.py OUTPUT_DIRECTORY

With the default limits, each family's model load must end inside the load
limit, and prediction, IG, SHAP (screen defaults, Cas12a head where there is
one) and NT Rollout must never go a no-progress limit without reporting: no
`restarting` event, every request delivered. SHAP also runs at the screen
maximums (1024 samples, 32 background), the longest normal job (#220). The
largest gap (from submission, between reports, and to the result) is recorded.

Then a real non-NT child is frozen mid-SHAP with a short injected no-progress
limit: it must be replaced in the same family, reload its model, not re-run the
stalled request, and serve a new prediction. No child may outlive close.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from cancel_grace_probe import Options, variant


def timed_dispatcher(family, **limits):
    from desktop.api import DesktopApi
    from desktop.computation import ProcessDispatcher

    events = []
    dispatcher = ProcessDispatcher(lambda e: events.append((time.monotonic(), e)),
                                   family=family, **limits)
    return dispatcher, DesktopApi(dispatcher.renderer, dispatcher), events


def pick_site(events, rid):
    fragments = {k: v for _, e in events if e["request_id"] == rid
                 for k, v in e.get("fragments", {}).items()}
    sites = {}
    for fragment in fragments.values():
        sites.update(Options(fragment).options)
    usable = [(key, heads) for key, heads in sites.items() if heads]
    site, heads = next(((k, h) for k, h in usable if "Cas12a" in h), usable[0])
    return site, ("Cas12a" if "Cas12a" in heads else heads[0])


def measure(dispatcher, events, reply, timeout=1800):
    """Run one request to its end; return its duration and largest silent gap."""
    began = time.monotonic()
    rid = reply["request_id"]
    assert dispatcher.wait_idle(timeout), f"request {rid} did not finish"
    ended = time.monotonic()
    stamps = [began, *(t for t, e in events if e["request_id"] == rid
                       and e["type"] == "progress"), ended]
    return rid, {
        "seconds": round(ended - began, 3),
        "max_gap": round(max(b - a for a, b in pairwise(stamps)), 3),
        "progress_events": len(stamps) - 2,
        "outcome": dispatcher.outcome(rid),
    }


def check_family(family, dna, record):
    import psutil
    from desktop.api import XAI_DEFAULTS
    from desktop.computation import LOAD_LIMIT, NO_PROGRESS_LIMIT

    dispatcher, api, events = timed_dispatcher(family)
    try:
        rid, record["load"] = measure(dispatcher, events, api.prepare())
        assert record["load"]["outcome"] == "delivered", record["load"]
        assert record["load"]["seconds"] < LOAD_LIMIT
        rid, record["predict"] = measure(dispatcher, events, api.predict(variant(dna, 0)))
        site, head = pick_site(events, rid)
        record["site"], record["head"] = site, head
        jobs = {method: (lambda m=method: api.explain(dna, site, head, m, XAI_DEFAULTS[m]))
                for method in ("ig", "shap")}
        # The longest normal job (#220): SHAP at the screen maximums.
        jobs["shap_max"] = lambda: api.explain(dna, site, head, "shap",
                                               {"nsamples": 1024, "n_background": 32})
        if family == "nt":
            jobs["rollout"] = lambda: api.explain(dna, site, head, "rollout")
        for name, call in jobs.items():
            _, record[name] = measure(dispatcher, events, call())
        requests = [record[name] for name in ("predict", *jobs)]
        assert all(r["outcome"] == "delivered" for r in requests), record
        assert all(r["max_gap"] < NO_PROGRESS_LIMIT for r in requests), record
        assert not [e for _, e in events if e["type"] == "restarting"]
        child = psutil.Process(dispatcher.pid)
    finally:
        dispatcher.close()
    record["child_closed"] = dispatcher.pid is None and not child.is_running()
    assert record["child_closed"]


def check_stall(dna, record, limit=5.0):
    import psutil
    from desktop.computation import NO_PROGRESS_LIMIT

    dispatcher, api, events = timed_dispatcher("non_nt")
    try:
        api.prepare()
        assert dispatcher.wait_idle(1800)
        rid = api.predict(dna)["request_id"]
        assert dispatcher.wait_idle(1800)
        assert dispatcher.outcome(rid) == "delivered"
        site, head = pick_site(events, rid)
        # non-NT loads its models lazily inside the first prediction, a silent
        # stretch the default limit allows; shorten the limit only after it.
        dispatcher.no_progress_limit = limit
        stuck = api.explain(dna, site, head, "shap",
                            {"nsamples": 1024, "n_background": 32})["request_id"]
        deadline = time.monotonic() + 600
        while not any(e["type"] == "progress" and e["request_id"] == stuck for _, e in events):
            assert time.monotonic() < deadline, "SHAP never reported"
            time.sleep(0.01)
        old = psutil.Process(dispatcher.pid)
        old.suspend()
        frozen_at = time.monotonic()
        assert dispatcher.wait_idle(1800)
        restarting = [(t, e) for t, e in events if e["type"] == "restarting"]
        assert len(restarting) == 1 and restarting[0][1]["reason"] == "no_progress", restarting
        ready = [(t, e) for t, e in events if e["type"] == "ready"
                 and e["request_id"] == restarting[0][1]["request_id"]]
        assert ready, "restarted process never became ready"
        record["seconds_to_restart"] = round(restarting[0][0] - frozen_at, 3)
        record["seconds_to_ready"] = round(ready[0][0] - frozen_at, 3)
        record["stuck_outcome"] = dispatcher.outcome(stuck)
        record["old_child_gone"] = not old.is_running()
        record["family"] = restarting[0][1]["family"]
        # Nothing replays the stalled request after the new process is ready.
        record["requests_after_ready"] = sorted({
            e["request_id"] for t, e in events if t > ready[0][0]})
        # The new child loads lazily again inside this prediction.
        dispatcher.no_progress_limit = NO_PROGRESS_LIMIT
        _, record["predict_after"] = measure(dispatcher, events, api.predict(variant(dna, 1)))
        assert record["stuck_outcome"] == "cancelled"
        assert record["old_child_gone"] and record["family"] == "non_nt"
        assert record["predict_after"]["outcome"] == "delivered"
        child = psutil.Process(dispatcher.pid)
    finally:
        dispatcher.close()
    record["child_closed"] = dispatcher.pid is None and not child.is_running()
    assert record["child_closed"]


def run(output):
    import psutil
    from bundle_runtime import configure_offline
    from desktop.computation import LOAD_LIMIT, NO_PROGRESS_LIMIT

    configure_offline()
    output.mkdir(parents=True, exist_ok=True)
    dna = json.loads((ROOT / "docs/equivalence-baseline/capture.json")
                     .read_text(encoding="utf-8"))["input_seq"]
    report = {"issue": 219, "status": "fail", "no_progress_limit": NO_PROGRESS_LIMIT,
              "load_limit": LOAD_LIMIT, "families": {}}
    try:
        for family in ("non_nt", "nt"):
            check_family(family, dna, report["families"].setdefault(family, {}))
        check_stall(dna, report.setdefault("stall", {}))
        report["remaining_children"] = len(psutil.Process().children(recursive=True))
        assert report["remaining_children"] == 0
        report["status"] = "pass"
    except Exception:  # noqa: BLE001 -- persist the failure in the report
        report["error"] = traceback.format_exc()
    finally:
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]).resolve()))
