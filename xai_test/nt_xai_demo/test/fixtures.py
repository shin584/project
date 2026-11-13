"""Small synthetic stand-ins for the Cas12a ZIP and the SpCas9 artifact tree.

Every pinned hash is computed from the synthetic bytes, so a fixture built
without `corrupt` passes all pin checks; `corrupt` breaks exactly one pin.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from xai_test.nt_xai_demo.prepare import CAS12A_METHODS, Sources, population_digest

PREFIX = "reports/abc123/"
IDS = [3, 7, 11]

# What every IG shard records about the sequence it tokenised (the real ZIP's
# shape, shortened: two 6-mers and a single base over a 13bp sequence).
SHARD_LAYOUT = {
    "coordinates": "0-based half-open",
    "dna_spans": [[0, 6], [6, 12], [12, 13]],
    "layout": {
        "coordinate_system": "0-based half-open",
        "left_flank": [0, 2],
        "pam": [2, 6],
        "protospacer": [6, 12],
        "right_flank": [12, 13],
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def npz_bytes(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def dump(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def build_cas12a(root: Path, *, corrupt: str | None = None) -> tuple[Path, Path]:
    plan = dump({"statistics_ids": IDS, "kind": "synthetic"})
    plan_sha = sha(plan)
    members: dict[str, bytes] = {
        "plan.json": plan,
        "plan.sha256": plan_sha.encode(),
        "execution.json": dump({"backend": "eager"}),
        "cases.json": dump({"plan_sha256": plan_sha, "cases": []}),
        "sessions/s1/start.json": dump({}),
    }
    for source_id in IDS:
        for method in CAS12A_METHODS:
            members[f"shards/{method}/{source_id}.json"] = dump({})
        # Only IG shards record the layout; the other keys stand for the
        # per-row fields that differ between shards and must not be compared.
        members[f"shards/ig/{source_id}.json"] = dump(
            {**SHARD_LAYOUT, "source_row": source_id, "method": "ig"}
        )
        members[f"shards/ig/{source_id}-row.npz"] = b"not json"
    if corrupt == "ig_missing_shard":
        del members[f"shards/ig/{IDS[-1]}.json"]
    if corrupt == "ig_layout":
        moved = {**SHARD_LAYOUT["layout"], "pam": [3, 7]}
        members[f"shards/ig/{IDS[-1]}.json"] = dump(
            {**SHARD_LAYOUT, "layout": moved, "source_row": IDS[-1]}
        )
    for method in CAS12A_METHODS:
        array = npz_bytes(source_row=np.array(IDS, dtype=np.int64))
        members[PREFIX + f"{method}.npz"] = array
        members[PREFIX + f"{method}.sidecar.json"] = dump(
            {"npz_sha256": sha(array), "plan_sha256": plan_sha, "details": ["x"] * 50}
        )
    members[PREFIX + "predictions.sidecar.json"] = dump({"plan_sha256": plan_sha})
    members[PREFIX + "agreement.json"] = dump(
        {
            "case_only_ids": [],
            "group_diagnostics": {"length": {"median": 0.125}},
            "ig_mode": "full",
            "populations": {"all": {"rho": 0.1 + 0.2}},
            "protocol": {"statistic": "spearman"},
            "rows": [{"source_row": i} for i in IDS],
            "statistics_ids": IDS,
        }
    )
    members[PREFIX + "cases_annotated.json"] = dump(
        {"selection": {"plan_sha256": plan_sha}}
    )
    members[PREFIX + "rollout_diagnostics.json"] = dump([{}] * 3)
    for name in (
        "summary.json",
        "metrics.json",
        "position_profiles.json",
        "token_profiles.json",
        "rollout_summary.json",
    ):
        members[PREFIX + name] = dump({"name": name})
    members[PREFIX + "predictions.csv"] = b"source_row,label\n3,0.5\n"
    members[PREFIX + "cases.csv"] = b"source_row,role\n3,low_error\n"
    for name in (
        "position_profiles.svg",
        "token_profiles.svg",
        "agreement_distribution.svg",
    ):
        members[PREFIX + name] = b"<svg/>"
    members[PREFIX + "report.ko.md"] = "# 보고서\n".encode()

    if corrupt == "ism_npz":
        members[PREFIX + "ism.npz"] = npz_bytes(
            source_row=np.array([1], dtype=np.int64)
        )
    if corrupt == "plan":
        members["plan.sha256"] = b"0" * 64

    archive = root / "cas12a.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name, data in members.items():
            handle.writestr(name, data)
    archive_sha = sha(archive.read_bytes())
    audit = root / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "archive_sha256": "f" * 64 if corrupt == "archive" else archive_sha,
            }
        ),
        encoding="utf-8",
    )
    return archive, audit


def build_spcas9(root: Path, *, corrupt: str | None = None) -> tuple[Path, Path]:
    artifacts = root / "artifacts"
    split = root / "published_split"
    for directory in (
        "v3_testset_predictions",
        "v3_ism",
        "v3_ig",
        "v3_rollout",
        "v3_ig_subset",
        "v3_cases",
        "v3_head_agreement",
        "v3_report",
    ):
        (artifacts / directory).mkdir(parents=True)
    split.mkdir(parents=True)

    def put(path: Path, data: bytes) -> str:
        path.write_bytes(data)
        return sha(data)

    split_sha = {
        name: put(split / f"{name}.csv", f"row_index,{name}\n1,2\n".encode())
        for name in ("train", "val", "test")
    }
    put(
        split / "split_checksum.json",
        dump({"files": {f"{k}.csv": v for k, v in split_sha.items()}}),
    )
    predictions_sha = put(
        artifacts / "v3_testset_predictions/predictions.csv",
        b"row_index,SpCas9\n5,0.1\n",
    )
    put(
        artifacts / "v3_testset_predictions/predictions_checksum.json",
        dump(
            {
                "predictions_sha256": predictions_sha,
                "split": {"sha256": split_sha["test"]},
            }
        ),
    )

    ism_rows = np.array([5, 9, 17], dtype=np.int64)
    ig_rows = np.array([9, 17], dtype=np.int64)
    ism_sha = put(artifacts / "v3_ism/ism.npz", npz_bytes(row_index=ism_rows))
    ig_sha = put(artifacts / "v3_ig/ig.npz", npz_bytes(row_index=ig_rows))
    rollout_sha = put(
        artifacts / "v3_rollout/rollout.npz", npz_bytes(row_index=ism_rows)
    )
    ism_population = population_digest(ism_rows)
    membership = population_digest(ig_rows)
    if corrupt == "population":
        ism_population = "0" * 64
    put(
        artifacts / "v3_ism/ism.sidecar.json",
        dump({"delta_sha256": ism_sha, "population_digest": ism_population}),
    )
    put(
        artifacts / "v3_ig/ig.sidecar.json",
        dump(
            {"attribution_sha256": ig_sha, "subset": {"membership_digest": membership}}
        ),
    )
    put(
        artifacts / "v3_rollout/rollout.sidecar.json",
        dump(
            {
                "rollout_sha256": rollout_sha,
                "population_digest": population_digest(ism_rows),
            }
        ),
    )
    subset_sha = put(
        artifacts / "v3_ig_subset/ig_subset.json",
        dump(
            {
                "membership_digest": membership,
                "entries": [{"row_index": 17}, {"row_index": 9}],
            }
        ),
    )
    put(
        artifacts / "v3_ig_subset/ig_subset_checksum.json",
        dump({"files": {"ig_subset.json": subset_sha}}),
    )
    cases_sha = put(artifacts / "v3_cases/cases.json", dump({"cases": []}))
    put(
        artifacts / "v3_cases/cases_checksum.json",
        dump({"files": {"cases.json": cases_sha}}),
    )
    noise_sha = put(artifacts / "v3_noise_floor.json", dump({"by_head": {}}))
    put(
        artifacts / "v3_noise_floor_checksum.json",
        dump({"files": {"v3_noise_floor.json": noise_sha}}),
    )
    put(
        artifacts / "v3_cases/cases_annotated.json",
        dump(
            {
                "consumed": {
                    "cases": {"sha256": cases_sha},
                    "ism": {"sha256": ism_sha},
                    "ig": {"sha256": ig_sha},
                    "noise_floor": {"sha256": noise_sha},
                }
            }
        ),
    )
    put(
        artifacts / "v3_head_agreement/head_agreement.json",
        dump(
            {
                "population_digest": population_digest(ism_rows),
                "consumed": {"ism": {"sha256": ism_sha}},
            }
        ),
    )
    put(
        artifacts / "v3_report/ig_ism_validation.json",
        dump({"consumed": {"ism": {"sha256": ism_sha}, "ig": {"sha256": ig_sha}}}),
    )
    put(
        artifacts / "v3_report/head_order_reversal.json",
        dump(
            {
                "consumed": {
                    "digests": {
                        "test.csv": split_sha["test"],
                        "predictions.csv": predictions_sha,
                        "v3_noise_floor.json": noise_sha,
                    }
                }
            }
        ),
    )
    return artifacts, split


def build_sources(
    root: Path, *, corrupt: str | None = None, spcas9: bool = True
) -> Sources:
    root.mkdir(parents=True, exist_ok=True)
    archive, audit = build_cas12a(root, corrupt=corrupt)
    artifacts, split = build_spcas9(root / "spcas9", corrupt=corrupt)
    if not spcas9:
        (artifacts / "v3_ism/ism.npz").unlink()
        (split / "train.csv").unlink()
    return Sources(
        cas12a_zip=archive,
        cas12a_audit=audit,
        spcas9_artifacts=artifacts,
        spcas9_split=split,
        cas12a_report_prefix=PREFIX,
        repo_root=root,
    )
