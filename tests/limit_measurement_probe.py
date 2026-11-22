"""Explicit real-model #220 measurement; run separately from regressions.

python -X utf8 tests/limit_measurement_probe.py OUTPUT_DIRECTORY [ROUNDS]

Measures, offline on this PC, what the three computation-process limits must
cover. The limits are injected far out of reach so nothing is cut short:

* load: the whole `warm_up` (prepare) of each family, two fresh children per
  round. The load limit is about 3x the largest.
* silent gaps: every cancellable job (prediction cold and warm, a site-dense
  prediction, CSV, IG and SHAP at the screen defaults and at the screen
  maximums for a Cas12a and an SpCas9 head, NT Rollout) is timed from
  submission, between progress reports and to its result. The no-progress
  limit is about 3x the largest gap.
* batches: a cancel can only be honoured at a progress report, so the longest
  uninterruptible stretch is the largest gap above (the first stretch of a
  cold prediction included). Each job is also cancelled after its first
  report, and a cold prediction is cancelled at once, recording the time to
  the `cancelled` outcome. The cancel grace is about 1.5x the largest batch.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from cancel_grace_probe import Options, variant

UNREACHABLE = 1e6
#: 81 bp with a TTTV and an NGG every six bases: as many sites as an input gets.
DENSE = ("TTTAGG" * 14)[:81]
MAXIMUMS = {"ig": {"steps": 256}, "shap": {"nsamples": 1024, "n_background": 32}}


class Session:
    def __init__(self, family, save_dir):
        from desktop.api import DesktopApi
        from desktop.computation import ProcessDispatcher

        self.events = []
        self.dispatcher = ProcessDispatcher(
            lambda e: self.events.append((time.monotonic(), e)), family=family,
            cancel_grace=UNREACHABLE, no_progress_limit=UNREACHABLE,
            load_limit=UNREACHABLE)
        self.save_dir = save_dir
        self._saves = 0
        self.api = DesktopApi(self.dispatcher.renderer, self.dispatcher,
                              ask_save_path=self._save_path)

    def _save_path(self, name):
        self._saves += 1
        return str(self.save_dir / f"{self._saves}-{name}")

    def progress(self, rid):
        return [t for t, e in self.events if e["request_id"] == rid and e["type"] == "progress"]

    def measure(self, reply, timeout=4 * 3600):
        began = time.monotonic()
        rid = reply["request_id"]
        assert rid is not None, reply
        assert self.dispatcher.wait_idle(timeout), f"request {rid} did not finish"
        ended = time.monotonic()
        reports = self.progress(rid)
        stamps = [began, *reports, ended]
        return rid, {
            "seconds": round(ended - began, 3),
            "max_gap": round(max(b - a for a, b in pairwise(stamps)), 3),
            "first_gap": round(stamps[1] - began, 3),
            "max_between_reports": round(max((b - a for a, b in pairwise(reports)),
                                             default=0.0), 3),
            "progress_events": len(reports),
            "outcome": self.dispatcher.outcome(rid),
        }

    def cancel_after_first_report(self, reply, timeout=3600):
        rid = reply["request_id"]
        deadline = time.monotonic() + timeout
        while not self.progress(rid):
            if self.dispatcher.outcome(rid) is not None:
                return {"progress_before_cancel": False,
                        "outcome": self.dispatcher.outcome(rid)}
            assert time.monotonic() < deadline, f"no progress for {rid}"
            time.sleep(0.005)
        return self.cancel_now(rid, reported=True)

    def cancel_now(self, rid, reported=False):
        began = time.monotonic()
        self.api.cancel()
        assert self.dispatcher.wait_idle(3600)
        return {"progress_before_cancel": reported,
                "seconds_to_outcome": round(time.monotonic() - began, 3),
                "outcome": self.dispatcher.outcome(rid)}

    def sites(self, rid):
        fragments = {k: v for _, e in self.events if e["request_id"] == rid
                     for k, v in e.get("fragments", {}).items()}
        found = {}
        for fragment in fragments.values():
            found.update(Options(fragment).options)
        return {k: v for k, v in found.items() if v}

    def close(self):
        import psutil

        child = psutil.Process(self.dispatcher.pid) if self.dispatcher.pid else None
        self.dispatcher.close()
        return self.dispatcher.pid is None and (child is None or not child.is_running())


def heads_to_explain(sites):
    """One Cas12a and one SpCas9-family (site, head), the first of each."""
    chosen = {}
    for site, heads in sites.items():
        for head in heads:
            kind = "cas12a" if "Cas12a" in head else "spcas9"
            chosen.setdefault(kind, (site, head))
    return chosen


def measure_family(family, dna, save_dir, record):
    from desktop.api import XAI_DEFAULTS

    session = Session(family, save_dir)
    try:
        _, record["load"] = session.measure(session.api.prepare())
        jobs = record["jobs"] = {}
        rid, jobs["predict_cold"] = session.measure(session.api.predict(dna))
        targets = heads_to_explain(session.sites(rid))
        record["targets"] = {k: list(v) for k, v in targets.items()}
        _, jobs["predict_warm"] = session.measure(session.api.predict(variant(dna, 0)))
        _, jobs["predict_dense"] = session.measure(session.api.predict(DENSE))
        _, jobs["csv_uncached"] = session.measure(session.api.save_csv(variant(dna, 1)))
        _, jobs["csv_cached"] = session.measure(session.api.save_csv(dna))
        for kind, (site, head) in targets.items():
            for method in ("ig", "shap"):
                for setting, params in (("default", XAI_DEFAULTS[method]),
                                        ("max", MAXIMUMS[method])):
                    _, jobs[f"{method}_{setting}_{kind}"] = session.measure(
                        session.api.explain(dna, site, head, method, params))
            if family == "nt":
                _, jobs[f"rollout_{kind}"] = session.measure(
                    session.api.explain(dna, site, head, "rollout"))

        cancels = record["cancels"] = {}
        site, head = targets.get("cas12a") or next(iter(targets.values()))
        calls = {
            "predict": lambda: session.api.predict(variant(dna, 2)),
            "csv": lambda: session.api.save_csv(variant(dna, 3)),
            "ig": lambda: session.api.explain(dna, site, head, "ig", {"steps": 255}),
            "shap": lambda: session.api.explain(dna, site, head, "shap",
                                                {"nsamples": 1023, "n_background": 32}),
        }
        for name, call in calls.items():
            cancels[name] = session.cancel_after_first_report(call())
        if family == "nt":
            # Rollout never reports; cancel it midway and wait for its end.
            rid = session.api.explain(variant(dna, 0), site, head, "rollout")["request_id"]
            time.sleep(0.3)
            cancels["rollout"] = session.cancel_now(rid)
        record["restarting_events"] = sum(e["type"] == "restarting" for _, e in session.events)
        record["errors"] = [e["html"] for _, e in session.events if e["type"] == "error"]
    finally:
        record["child_closed"] = session.close()

    # A cold child cancelled at once: the first stretch cannot be interrupted.
    session = Session(family, save_dir)
    try:
        _, record["load_second"] = session.measure(session.api.prepare())
        rid = session.api.predict(dna)["request_id"]
        record["cancels"]["predict_cold_immediate"] = session.cancel_now(rid)
    finally:
        record["child_closed_second"] = session.close()


def summarise(rounds):
    loads, gaps, batches, cancels = [], [], [], []
    for families in rounds:
        for family, record in families.items():
            for key in ("load", "load_second"):
                if key in record:
                    loads.append((record[key]["seconds"], f"{family}.{key}"))
            for name, job in record.get("jobs", {}).items():
                gaps.append((job["max_gap"], f"{family}.{name}"))
                batches.append((job["max_gap"], f"{family}.{name}"))
            for name, cancel in record.get("cancels", {}).items():
                if "seconds_to_outcome" in cancel:
                    cancels.append((cancel["seconds_to_outcome"], f"{family}.cancel.{name}"))
                    batches.append((cancel["seconds_to_outcome"], f"{family}.cancel.{name}"))
    top = lambda items: dict(zip(("seconds", "where"), max(items))) if items else None
    return {"max_load": top(loads), "max_gap": top(gaps),
            "max_batch": top(batches), "max_cancel_to_outcome": top(cancels)}


def run(output, rounds):
    import psutil
    from bundle_runtime import configure_offline

    configure_offline()
    output.mkdir(parents=True, exist_ok=True)
    dna = json.loads((ROOT / "docs/equivalence-baseline/capture.json")
                     .read_text(encoding="utf-8"))["input_seq"]
    report = {"issue": 220, "status": "fail", "rounds": []}
    try:
        with tempfile.TemporaryDirectory() as saves:
            for _ in range(rounds):
                families = {}
                report["rounds"].append(families)
                for family in ("non_nt", "nt"):
                    measure_family(family, dna, Path(saves),
                                   families.setdefault(family, {}))
                (output / "report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["summary"] = summarise(report["rounds"])
        report["remaining_children"] = len(psutil.Process().children(recursive=True))
        delivered = all(job["outcome"] == "delivered" for families in report["rounds"]
                        for record in families.values() for job in record["jobs"].values())
        assert delivered and report["remaining_children"] == 0
        report["status"] = "pass"
    except Exception:  # noqa: BLE001 -- persist the failure in the report
        report["error"] = traceback.format_exc()
    finally:
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]).resolve(),
                         int(sys.argv[2]) if len(sys.argv) > 2 else 3))
