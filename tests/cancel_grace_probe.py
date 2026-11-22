"""Explicit real-model #218 check; run separately from regressions.

python -X utf8 tests/cancel_grace_probe.py OUTPUT_DIRECTORY

For each family, a cooperative cancel of prediction, IG, SHAP and (NT only)
Rollout must be answered inside the default cancel grace: same generation,
same child, no `restarting` event, and an explanation computed before the
cancels is still served from the process-lifetime memory cache. The time from
cancel to the request's outcome is recorded for the grace measurement ticket.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Options(HTMLParser):
    def __init__(self, fragment):
        super().__init__()
        self.options = {}
        self.feed(fragment)

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == "option" and "data-heads" in data:
            self.options[data["value"]] = json.loads(data["data-heads"])


def variant(dna, index):
    """Same length and alphabet, different bytes: an uncached prediction."""
    swap = {"A": "C", "C": "G", "G": "T", "T": "A"}
    return dna[:index] + swap[dna[index]] + dna[index + 1:]


def check_family(family, dna, record):
    import psutil

    from desktop.api import DesktopApi
    from desktop.computation import CANCEL_GRACE, ProcessDispatcher

    events = []
    dispatcher = ProcessDispatcher(events.append, family=family)
    api = DesktopApi(dispatcher.renderer, dispatcher)

    def run(reply, timeout=900):
        rid = reply["request_id"]
        assert dispatcher.wait_idle(timeout), f"{family}: request {rid} did not finish"
        return rid

    def first_progress(rid, timeout=600):
        deadline = time.monotonic() + timeout
        while not any(e["type"] == "progress" and e["request_id"] == rid for e in events):
            if dispatcher.outcome(rid) is not None:
                return False
            assert time.monotonic() < deadline, f"{family}: no progress for {rid}"
            time.sleep(0.01)
        return True

    try:
        run(api.prepare())
        pid, generation = dispatcher.pid, dispatcher.generation
        rid = run(api.predict(dna))
        fragments = {k: v for e in events if e["request_id"] == rid
                     for k, v in e.get("fragments", {}).items()}
        sites = Options(fragments.get("xai", "") + fragments.get("detail", "")).options
        if not sites:
            sites = {}
            for fragment in fragments.values():
                sites.update(Options(fragment).options)
        site, heads = next((key, value) for key, value in sites.items() if value)
        head = heads[0]
        record["site"], record["head"] = site, head
        seeded = run(api.explain(dna, site, head, "ig"))
        assert dispatcher.outcome(seeded) == "delivered"

        cancels = {
            "predict": lambda: api.predict(variant(dna, 0)),
            "ig": lambda: api.explain(dna, site, head, "ig", {"steps": 256}),
            "shap": lambda: api.explain(dna, site, head, "shap",
                                        {"nsamples": 1024, "n_background": 32}),
        }
        if family == "nt":
            cancels["rollout"] = lambda: api.explain(dna, site, head, "rollout")
        record["cancels"] = {}
        for name, call in cancels.items():
            rid = call()["request_id"]
            reported = first_progress(rid) if name != "rollout" else (time.sleep(0.5) or None)
            began = time.monotonic()
            api.cancel()
            assert dispatcher.wait_idle(CANCEL_GRACE * 4)
            record["cancels"][name] = {
                "progress_before_cancel": reported,
                "seconds_to_outcome": round(time.monotonic() - began, 3),
                "outcome": dispatcher.outcome(rid),
                "same_child": dispatcher.pid == pid,
                "same_generation": dispatcher.generation == generation,
            }
        assert not [e for e in events if e["type"] == "restarting"], "cooperative cancel restarted"
        assert all(c["same_child"] and c["same_generation"] for c in record["cancels"].values())
        assert all(c["seconds_to_outcome"] < CANCEL_GRACE for c in record["cancels"].values())

        began = time.monotonic()
        again = run(api.explain(dna, site, head, "ig"))
        record["cache_hit"] = {
            "seconds": round(time.monotonic() - began, 3),
            "progress_events": sum(1 for e in events
                                   if e["type"] == "progress" and e["request_id"] == again),
            "outcome": dispatcher.outcome(again),
        }
        assert record["cache_hit"]["outcome"] == "delivered"
        assert record["cache_hit"]["progress_events"] == 0, "IG was recomputed after cancels"
        child = psutil.Process(dispatcher.pid)
    finally:
        dispatcher.close()
    record["errors"] = [e["html"] for e in events if e["type"] == "error"]
    record["child_closed"] = dispatcher.pid is None and not child.is_running()
    assert record["child_closed"]


def run(output):
    import psutil

    from bundle_runtime import configure_offline
    from desktop.computation import CANCEL_GRACE

    configure_offline()
    output.mkdir(parents=True, exist_ok=True)
    dna = json.loads((ROOT / "docs/equivalence-baseline/capture.json")
                     .read_text(encoding="utf-8"))["input_seq"]
    report = {"issue": 218, "status": "fail", "cancel_grace": CANCEL_GRACE, "families": {}}
    try:
        for family in ("non_nt", "nt"):
            record = report["families"][family] = {}
            check_family(family, dna, record)
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
