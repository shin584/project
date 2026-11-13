"""`prepare`: gather the NT XAI 데모's saved artifacts into one checksummed directory.

Two sources, one destination (`xai_test/nt_xai_demo/data/`), one manifest:

* Cas12a — a light subset of the Test1500 result ZIP. Per-row shards and the
  large `*.sidecar.json` files are read in memory to check their pins but are
  never extracted. `agreement.json` is sliced by key selection only.
* SpCas9 (9-Hydra v3) — copied from the in-place artifacts and the published
  split. The bulk part of it is not in Git; when it is absent, only Cas12a is
  prepared and the absence is reported, not hidden. Whatever `spcas9/` files an
  earlier manifest certified and are still on disk (on a clone, the Git-tracked
  light JSON) are carried over with that manifest's entries, not re-hashed:
  without the bundle there is no source pin to certify a new hash against.

Order matters: every hash the sources already pin (archive audit, sidecar
digests, checksum files, `consumed` digests) is compared against the source
bytes *before* any file is written or any new checksum is generated. A
manifest built over an already-broken file would certify it.

The demo never loads a model and does not import the analysis packages; the
population digest below restates `v3_predict.population_digest`'s contract.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
XAI_DIR = PACKAGE_DIR.parent
REPO_ROOT = XAI_DIR.parent
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"

CAS12A_RESULT = XAI_DIR / "cas12a_NT_analysis" / "result"
CAS12A_ZIP = CAS12A_RESULT / "cas12a-test1500-results-a9e50fa7f668.zip"
CAS12A_AUDIT = CAS12A_RESULT / "test1500-a9e50fa7f668-final-audit.json"
CAS12A_REPORT_PREFIX = "reports/745133f240e1/"
SPCAS9_ARTIFACTS = XAI_DIR / "SpCas9_NT_analysis" / "artifacts"
SPCAS9_SPLIT = XAI_DIR / "SpCas9" / "published_split"

MANIFEST = "manifest.json"
SCHEMA = 1

# Top-level ZIP members, then report members. Everything else in the ZIP —
# `shards/` (10,500 per-row files), `sessions/`, every `*.sidecar.json` and the
# per-row `rollout_diagnostics.json` — stays in the ZIP.
CAS12A_TOP = ("plan.json", "plan.sha256", "execution.json", "cases.json")
CAS12A_REPORT = (
    "predictions.csv",
    "metrics.json",
    "summary.json",
    "position_profiles.json",
    "token_profiles.json",
    "rollout_summary.json",
    "cases_annotated.json",
    "cases.csv",
    "position_profiles.svg",
    "token_profiles.svg",
    "agreement_distribution.svg",
    "report.ko.md",
    "ism.npz",
    "ig.npz",
    "rollout.npz",
)
CAS12A_METHODS = ("ism", "ig", "rollout")
AGREEMENT_KEPT = ("populations", "group_diagnostics", "protocol")
AGREEMENT_SLICE = "cas12a/agreement.sliced.json"

# Cas12a's DNA token spans and sequence layout are recorded per IG shard, and
# the shards stay in the ZIP, so the screen that draws a bar per position and a
# row per token would have nowhere to read them from. They are lifted into one
# small file here instead of being re-derived: 34bp splits into 6-mers with a
# 4-base remainder, which is guessable, and a guess is exactly what this demo
# must not put under a number. `pam` is what makes the headline finding legible
# (the most sensitive region overlaps the PAM), so it is carried with them.
CAS12A_LAYOUT = "cas12a/layout.json"
CAS12A_LAYOUT_KEYS = ("coordinates", "dna_spans", "layout")
CAS12A_IG_SHARDS = "shards/ig/"

# Destination under `spcas9/` -> (source root key, path under that root).
SPCAS9_FILES = {
    **{
        f"published_split/{name}": ("split", name)
        for name in ("train.csv", "val.csv", "test.csv", "split_checksum.json")
    },
    **{
        name: ("artifacts", name)
        for name in (
            "v3_testset_predictions/predictions.csv",
            "v3_testset_predictions/predictions_checksum.json",
            "v3_ism/ism.npz",
            "v3_ism/ism.sidecar.json",
            "v3_ig/ig.npz",
            "v3_ig/ig.sidecar.json",
            "v3_rollout/rollout.npz",
            "v3_rollout/rollout.sidecar.json",
            "v3_ig_subset/ig_subset.json",
            "v3_ig_subset/ig_subset_checksum.json",
            "v3_cases/cases.json",
            "v3_cases/cases_annotated.json",
            "v3_cases/cases_checksum.json",
            "v3_head_agreement/head_agreement.json",
            "v3_report/head_order_reversal.json",
            "v3_report/ig_ism_validation.json",
            "v3_noise_floor.json",
            "v3_noise_floor_checksum.json",
        )
    },
}

# The one small CSV kept in Git; every other CSV here is bulk per-row data.
TRACKED_CSV = frozenset({"cas12a/cases.csv"})


def is_bulk(name: str) -> bool:
    """Whether a data-relative path is left out of Git (`.npz`, bulk CSV)."""
    return name.endswith(".npz") or (name.endswith(".csv") and name not in TRACKED_CSV)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def population_digest(row_index: Sequence[int] | np.ndarray) -> str:
    """SHA-256 of the row IDs as big-endian int64, order-sensitive (v3 contract)."""
    return sha256_bytes(
        np.asarray(row_index, dtype=np.int64).astype(">i8").tobytes(order="C")
    )


@dataclass(frozen=True)
class Sources:
    cas12a_zip: Path = CAS12A_ZIP
    cas12a_audit: Path = CAS12A_AUDIT
    spcas9_artifacts: Path = SPCAS9_ARTIFACTS
    spcas9_split: Path = SPCAS9_SPLIT
    cas12a_report_prefix: str = CAS12A_REPORT_PREFIX
    repo_root: Path = REPO_ROOT

    def spcas9_source(self, name: str) -> Path:
        root, relative = SPCAS9_FILES[name]
        return (
            self.spcas9_split if root == "split" else self.spcas9_artifacts
        ) / relative

    def display(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


@dataclass(frozen=True)
class Pin:
    """A hash a source already records, and the value recomputed from the bytes."""

    name: str
    expected: Any
    actual: Any

    @property
    def ok(self) -> bool:
        return self.expected == self.actual


class PrepareRefused(RuntimeError):
    """`prepare` stopped before touching `data/`; the message says why (Korean)."""


class MissingSource(PrepareRefused):
    pass


class SpCas9WouldBeLost(PrepareRefused):
    """Bulk under `spcas9/` that no manifest lists, with no bundle to rebuild it.

    Listed files are carried over instead, and Git-tracked light files can be
    restored from Git; only unlisted bulk would be lost for good.
    """


class PinMismatch(PrepareRefused):
    def __init__(self, failed: Sequence[Pin]):
        self.failed = list(failed)
        lines = [
            f"  - {p.name}: 기록 {p.expected!r} / 실제 {p.actual!r}"
            for p in self.failed
        ]
        super().__init__(
            "원본의 고정 해시와 일치하지 않아 manifest를 만들지 않았습니다:\n"
            + "\n".join(lines)
        )


@dataclass
class PrepareReport:
    data_dir: Path
    file_count: int
    total_bytes: int
    cas12a_pins: int
    spcas9_pins: int
    spcas9_missing: list[str] = field(default_factory=list)
    spcas9_carried: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Cas12a: 준비됨 (원본 고정 해시 {self.cas12a_pins}개 대조 통과)"]
        if self.spcas9_missing:
            lines.append(
                "SpCas9: 원본 묶음 부재 — 새로 준비하지 않았습니다. 없는 원본:"
            )
            lines += [f"  - {name}" for name in self.spcas9_missing]
            if self.spcas9_carried:
                lines.append(
                    f"  기존 SpCas9 자료 {len(self.spcas9_carried)}개를 유지함 "
                    "(해시는 이전 manifest의 기록 그대로; 원본 고정 해시 대조는 하지 않음)"
                )
            lines.append(
                "  복원 조건: Cas12a ZIP + SpCas9 원본 묶음 (xai_test/nt_xai_demo/DATA.md)"
            )
        else:
            lines.append(
                f"SpCas9: 준비됨 (원본 고정 해시 {self.spcas9_pins}개 대조 통과)"
            )
        lines.append(
            f"manifest: {self.data_dir / MANIFEST} "
            f"(파일 {self.file_count}개, {self.total_bytes / 1e6:.1f} MB)"
        )
        return "\n".join(lines)


def _json(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


def _npz_ids(data: bytes, key: str) -> list[int]:
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        return arrays[key].astype(np.int64).tolist()


def _cas12a(
    sources: Sources,
) -> tuple[list[Pin], dict[str, bytes], dict[str, dict[str, Any]], str]:
    """Pins, files to write (`cas12a/...` -> bytes), the derived-file records
    (agreement slice, layout) and the archive's sha256."""
    prefix = sources.cas12a_report_prefix
    audit = json.loads(sources.cas12a_audit.read_text(encoding="utf-8"))
    archive_sha = sha256_file(sources.cas12a_zip)
    pins = [
        Pin("Cas12a 최종 감사 status", "PASS", audit.get("status")),
        Pin("Cas12a ZIP sha256 (최종 감사)", audit.get("archive_sha256"), archive_sha),
    ]
    with zipfile.ZipFile(sources.cas12a_zip) as archive:
        read = archive.read
        plan = read("plan.json")
        plan_pin = read("plan.sha256").decode("ascii").strip()
        plan_ids = _json(plan)["statistics_ids"]
        pins.append(Pin("plan.json sha256 (plan.sha256)", plan_pin, sha256_bytes(plan)))
        for method in CAS12A_METHODS:
            npz = read(prefix + f"{method}.npz")
            sidecar = _json(read(prefix + f"{method}.sidecar.json"))
            pins += [
                Pin(
                    f"{method}.npz sha256 (sidecar npz_sha256)",
                    sidecar.get("npz_sha256"),
                    sha256_bytes(npz),
                ),
                Pin(
                    f"{method}.sidecar plan_sha256",
                    plan_pin,
                    sidecar.get("plan_sha256"),
                ),
                Pin(
                    f"{method}.npz source_row (plan statistics_ids)",
                    plan_ids,
                    _npz_ids(npz, "source_row"),
                ),
            ]
        pins += [
            Pin(
                "predictions.sidecar plan_sha256",
                plan_pin,
                _json(read(prefix + "predictions.sidecar.json")).get("plan_sha256"),
            ),
            Pin(
                "cases.json plan_sha256",
                plan_pin,
                _json(read("cases.json")).get("plan_sha256"),
            ),
            Pin(
                "cases_annotated selection.plan_sha256",
                plan_pin,
                _json(read(prefix + "cases_annotated.json"))
                .get("selection", {})
                .get("plan_sha256"),
            ),
        ]
        agreement_raw = read(prefix + "agreement.json")
        agreement = _json(agreement_raw)
        pins.append(
            Pin(
                "agreement statistics_ids (plan)",
                plan_ids,
                agreement.get("statistics_ids"),
            )
        )

        files = {f"cas12a/{name}": read(name) for name in CAS12A_TOP}
        files.update({f"cas12a/{name}": read(prefix + name) for name in CAS12A_REPORT})

        # Every IG shard records the same spans and layout; reading all of them
        # turns that into a checked fact rather than an assumption drawn from
        # one row. A single disagreeing shard fails the pin and nothing is
        # written, which is the point: the screen draws tokens at these spans.
        shards = sorted(
            name
            for name in archive.namelist()
            if name.startswith(CAS12A_IG_SHARDS) and name.endswith(".json")
        )
        layouts = {
            json.dumps(
                {key: _json(read(name)).get(key) for key in CAS12A_LAYOUT_KEYS},
                sort_keys=True,
                ensure_ascii=False,
            )
            for name in shards
        }
        # Agreement among the shards present says nothing about a missing one,
        # so the count is pinned to the plan's rows as well.
        pins += [
            Pin("IG shard 수 (plan statistics_ids)", len(plan_ids), len(shards)),
            Pin(
                f"IG shard {len(shards):,}개의 dna_spans·layout 일치",
                1,
                len(layouts),
            ),
        ]
        # Without exactly one layout the pin above fails and nothing is
        # written; no placeholder is staged in its place.
        if len(layouts) == 1:
            files[CAS12A_LAYOUT] = layouts.pop().encode("utf-8")
    sliced = {key: agreement[key] for key in AGREEMENT_KEPT}
    files[AGREEMENT_SLICE] = json.dumps(sliced, ensure_ascii=False).encode("utf-8")
    derived = {
        AGREEMENT_SLICE: {
            "source": f"zip:{prefix}agreement.json",
            "source_sha256": sha256_bytes(agreement_raw),
            "operation": "최상위 키 선택만 함; 값은 원본 그대로이며 산술 없음",
            "kept": list(AGREEMENT_KEPT),
            "dropped": sorted(set(agreement) - set(AGREEMENT_KEPT)),
        },
        CAS12A_LAYOUT: {
            "source": f"zip:{CAS12A_IG_SHARDS}*.json",
            "shards": len(shards),
            "operation": (
                "모든 IG shard에서 같은 키를 골라 일치를 확인한 뒤 그 한 벌을 씀; "
                "값은 원본 그대로이며 산술 없음"
            ),
            "kept": list(CAS12A_LAYOUT_KEYS),
        },
    }
    return pins, files, derived, archive_sha


def _spcas9_missing(sources: Sources) -> list[str]:
    return [
        sources.display(sources.spcas9_source(name))
        for name in SPCAS9_FILES
        if not sources.spcas9_source(name).is_file()
    ]


def _spcas9(sources: Sources) -> tuple[list[Pin], dict[str, Path], dict[str, str]]:
    """Pins, files to copy (`spcas9/...` -> source) and their checked sha256."""
    path = sources.spcas9_source
    digest = {name: sha256_file(path(name)) for name in SPCAS9_FILES}

    def load(name: str) -> Any:
        return json.loads(path(name).read_text(encoding="utf-8"))

    def ids(name: str) -> list[int]:
        return _npz_ids(path(name).read_bytes(), "row_index")

    split = load("published_split/split_checksum.json")["files"]
    predictions = load("v3_testset_predictions/predictions_checksum.json")
    ism = load("v3_ism/ism.sidecar.json")
    ig = load("v3_ig/ig.sidecar.json")
    rollout = load("v3_rollout/rollout.sidecar.json")
    subset = load("v3_ig_subset/ig_subset.json")
    annotated = load("v3_cases/cases_annotated.json")["consumed"]
    agreement = load("v3_head_agreement/head_agreement.json")
    validation = load("v3_report/ig_ism_validation.json")["consumed"]
    reversal = load("v3_report/head_order_reversal.json")["consumed"]["digests"]
    ism_digest = population_digest(ids("v3_ism/ism.npz"))

    def checksum(name: str, pin_file: str) -> Pin:
        return Pin(
            f"{name} sha256 ({pin_file})",
            load(pin_file)["files"][Path(name).name],
            digest[name],
        )

    pins = [
        *(
            Pin(
                f"{name}.csv sha256 (split_checksum)",
                split[f"{name}.csv"],
                digest[f"published_split/{name}.csv"],
            )
            for name in ("train", "val", "test")
        ),
        Pin(
            "predictions.csv sha256 (predictions_checksum)",
            predictions["predictions_sha256"],
            digest["v3_testset_predictions/predictions.csv"],
        ),
        Pin(
            "predictions_checksum split.sha256",
            predictions["split"]["sha256"],
            digest["published_split/test.csv"],
        ),
        Pin(
            "ism.npz sha256 (sidecar delta_sha256)",
            ism["delta_sha256"],
            digest["v3_ism/ism.npz"],
        ),
        Pin(
            "ism.npz population_digest (sidecar)", ism["population_digest"], ism_digest
        ),
        Pin(
            "ig.npz sha256 (sidecar attribution_sha256)",
            ig["attribution_sha256"],
            digest["v3_ig/ig.npz"],
        ),
        Pin(
            "ig.npz row_index (sidecar membership_digest)",
            ig["subset"]["membership_digest"],
            population_digest(ids("v3_ig/ig.npz")),
        ),
        Pin(
            "ig_subset membership_digest (ig sidecar)",
            ig["subset"]["membership_digest"],
            subset["membership_digest"],
        ),
        Pin(
            "ig_subset entries (membership_digest)",
            subset["membership_digest"],
            population_digest(sorted(e["row_index"] for e in subset["entries"])),
        ),
        Pin(
            "rollout.npz sha256 (sidecar rollout_sha256)",
            rollout["rollout_sha256"],
            digest["v3_rollout/rollout.npz"],
        ),
        Pin(
            "rollout.npz population_digest (sidecar)",
            rollout["population_digest"],
            population_digest(ids("v3_rollout/rollout.npz")),
        ),
        checksum("v3_ig_subset/ig_subset.json", "v3_ig_subset/ig_subset_checksum.json"),
        checksum("v3_cases/cases.json", "v3_cases/cases_checksum.json"),
        checksum("v3_noise_floor.json", "v3_noise_floor_checksum.json"),
        Pin(
            "cases_annotated consumed.cases",
            annotated["cases"]["sha256"],
            digest["v3_cases/cases.json"],
        ),
        Pin(
            "cases_annotated consumed.ism",
            annotated["ism"]["sha256"],
            digest["v3_ism/ism.npz"],
        ),
        Pin(
            "cases_annotated consumed.ig",
            annotated["ig"]["sha256"],
            digest["v3_ig/ig.npz"],
        ),
        Pin(
            "cases_annotated consumed.noise_floor",
            annotated["noise_floor"]["sha256"],
            digest["v3_noise_floor.json"],
        ),
        Pin(
            "head_agreement population_digest (ism)",
            agreement["population_digest"],
            ism_digest,
        ),
        Pin(
            "head_agreement consumed.ism",
            agreement["consumed"]["ism"]["sha256"],
            digest["v3_ism/ism.npz"],
        ),
        Pin(
            "ig_ism_validation consumed.ism",
            validation["ism"]["sha256"],
            digest["v3_ism/ism.npz"],
        ),
        Pin(
            "ig_ism_validation consumed.ig",
            validation["ig"]["sha256"],
            digest["v3_ig/ig.npz"],
        ),
        Pin(
            "head_order_reversal consumed test.csv",
            reversal["test.csv"],
            digest["published_split/test.csv"],
        ),
        Pin(
            "head_order_reversal consumed predictions.csv",
            reversal["predictions.csv"],
            digest["v3_testset_predictions/predictions.csv"],
        ),
        Pin(
            "head_order_reversal consumed v3_noise_floor.json",
            reversal["v3_noise_floor.json"],
            digest["v3_noise_floor.json"],
        ),
    ]
    copies = {f"spcas9/{name}": path(name) for name in SPCAS9_FILES}
    checked = {f"spcas9/{name}": value for name, value in digest.items()}
    return pins, copies, checked


def _previous_manifest(data_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads((data_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _carry_over(
    data_dir: Path, previous: dict[str, Any]
) -> tuple[list[Pin], dict[str, dict[str, Any]]]:
    """The `spcas9/` files to keep when the bundle is absent, and their pins.

    A file is kept, with its previous manifest entry, when that entry lists it
    and it is on disk; a listed file that is not on disk (bulk on a clone) is not
    there to keep. Each kept file is compared with its recorded sha256 — the only
    pin left — so a broken file is refused, never re-certified. Bulk on disk that
    no entry lists can be neither kept nor rebuilt, so `prepare` refuses.
    """
    listed = {
        name: entry
        for name, entry in previous.get("files", {}).items()
        if name.startswith("spcas9/")
    }
    kept = {name: e for name, e in listed.items() if (data_dir / name).is_file()}
    root = data_dir / "spcas9"
    on_disk = (
        path.relative_to(data_dir).as_posix()
        for path in (root.rglob("*") if root.is_dir() else ())
        if path.is_file()
    )
    unlisted = sorted(n for n in on_disk if is_bulk(n) and n not in listed)
    if unlisted:
        raise SpCas9WouldBeLost(
            "SpCas9 원본 묶음이 없고, data/spcas9/에 manifest에 없는 bulk 자료가 있어 "
            "덮어쓰지 않았습니다 (유지할 근거 해시도, 다시 만들 원본도 없음). "
            "묶음을 복원하거나 이 파일을 옮긴 뒤 다시 실행하십시오 "
            "(xai_test/nt_xai_demo/DATA.md):\n"
            + "\n".join(f"  - {name}" for name in unlisted)
        )
    pins = [
        Pin(
            f"{name} sha256 (이전 manifest)",
            entry.get("sha256"),
            sha256_file(data_dir / name),
        )
        for name, entry in sorted(kept.items())
    ]
    return pins, kept


def _absent_spcas9(
    missing: list[str], previous: dict[str, Any], carried: dict[str, Any]
) -> dict[str, Any]:
    """`nucleases.spcas9` without the bundle; `carried_over` only when files were kept.

    `verify` reads only `files` and the reader never reads `nucleases`, so
    `status` stays `"absent"` (the bundle was not checked this run) and the kept
    files are named beside it instead of under a third status value.
    """
    record: dict[str, Any] = {"status": "absent", "missing": missing}
    if carried:
        record["carried_over"] = {
            "from": "이전 manifest (이 실행 전의 data/manifest.json)",
            "previous_status": previous.get("nucleases", {})
            .get("spcas9", {})
            .get("status"),
            "note": (
                "원본 묶음이 없어 새로 준비하지 않았다. 아래 파일은 이전 manifest의 "
                "항목(sha256/bytes/git/source)을 그대로 이어받았고, 그 sha256과의 "
                "일치만 확인했다 (원본 고정 해시 대조는 이번 실행에서 하지 않음)."
            ),
            "files": sorted(carried),
        }
    return record


def _write_tree(
    stage: Path,
    written: dict[str, bytes],
    copied: dict[str, Path],
    origin: Callable[[str], str],
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for name in [*written, *copied]:
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in written:
            target.write_bytes(written[name])
        else:
            shutil.copyfile(copied[name], target)
    for name in sorted([*written, *copied]):
        target = stage / name
        entries[name] = {
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "git": "ignored" if is_bulk(name) else "tracked",
            "source": origin(name),
        }
    return entries


def prepare(
    sources: Sources | None = None, data_dir: Path = DEFAULT_DATA_DIR
) -> PrepareReport:
    """Check every source pin, then (only then) write the data and its manifest.

    Without the SpCas9 bundle, the `spcas9/` files an earlier manifest lists and
    that are on disk are carried over with their recorded entries.

    Raises a `PrepareRefused` without touching `data_dir` when a Cas12a source
    is missing, when any pin disagrees (a carried-over file against its previous
    entry included), or when an absent SpCas9 bundle would delete `spcas9/`
    bulk that no manifest lists.
    """
    sources = sources or Sources()
    data_dir = Path(data_dir)
    absent = [
        sources.display(path)
        for path in (sources.cas12a_zip, sources.cas12a_audit)
        if not path.is_file()
    ]
    if absent:
        raise MissingSource(
            "Cas12a 원본이 없어 준비하지 않았습니다 (복원 조건: xai_test/nt_xai_demo/DATA.md):\n"
            + "\n".join(f"  - {name}" for name in absent)
        )
    cas12a_pins, cas12a_files, derived, archive_sha = _cas12a(sources)
    missing = _spcas9_missing(sources)
    spcas9_pins, spcas9_files, spcas9_checked = (
        ([], {}, {}) if missing else _spcas9(sources)
    )
    previous = _previous_manifest(data_dir) if missing else {}
    carry_pins, carried = _carry_over(data_dir, previous) if missing else ([], {})
    failed = [p for p in [*cas12a_pins, *spcas9_pins, *carry_pins] if not p.ok]
    if failed:
        raise PinMismatch(failed)

    prefix = sources.cas12a_report_prefix

    def origin(name: str) -> str:
        if name == AGREEMENT_SLICE:
            return f"zip:{prefix}agreement.json (sliced)"
        if name == CAS12A_LAYOUT:
            return f"zip:{CAS12A_IG_SHARDS}*.json (모든 shard 일치, 키 선택)"
        if name.startswith("cas12a/"):
            member = name.removeprefix("cas12a/")
            return f"zip:{member if member in CAS12A_TOP else prefix + member}"
        return sources.display(spcas9_files[name])

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prepare-", dir=data_dir.parent))
    try:
        files = _write_tree(stage, cas12a_files, spcas9_files, origin)
        for name, entry in carried.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data_dir / name, target)
            files[name] = entry
        files = dict(sorted(files.items()))
        # The copies are hashed again; a source that changed after its pins
        # were checked must not be certified by the new manifest.
        changed = [
            Pin(f"{name} 복사본 sha256 (대조 시점)", value, files[name]["sha256"])
            for name, value in spcas9_checked.items()
        ] + [
            Pin(
                f"{name} 복사본 sha256 (이전 manifest)",
                entry["sha256"],
                sha256_file(stage / name),
            )
            for name, entry in carried.items()
        ]
        if any(not pin.ok for pin in changed):
            raise PinMismatch([pin for pin in changed if not pin.ok])
        payload = {
            "schema": SCHEMA,
            "restore": "Cas12a ZIP + SpCas9 원본 묶음 (xai_test/nt_xai_demo/DATA.md)",
            "nucleases": {
                "cas12a": {
                    "status": "prepared",
                    "archive": sources.display(sources.cas12a_zip),
                    "archive_sha256": archive_sha,
                    "audit": sources.display(sources.cas12a_audit),
                    "pins_checked": [p.name for p in cas12a_pins],
                },
                "spcas9": (
                    _absent_spcas9(missing, previous, carried)
                    if missing
                    else {
                        "status": "prepared",
                        "pins_checked": [p.name for p in spcas9_pins],
                    }
                ),
            },
            "derived": derived,
            "files": files,
        }
        (stage / MANIFEST).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if data_dir.exists():
            shutil.rmtree(data_dir)
        stage.rename(data_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return PrepareReport(
        data_dir=data_dir,
        file_count=len(files),
        total_bytes=sum(entry["bytes"] for entry in files.values()),
        cas12a_pins=len(cas12a_pins),
        spcas9_pins=len(spcas9_pins),
        spcas9_missing=missing,
        spcas9_carried=sorted(carried),
    )


def recorded_pins(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, list[str]]:
    """The pins `prepare` checked, per prepared nuclease, as its manifest records.

    `verify` compares files with the manifest only; it does not re-run these.
    They are listed so the check behind a file (the layout pin above all: the
    screen draws tokens at those spans) is visible without opening the JSON.
    """
    payload = _previous_manifest(Path(data_dir))
    return {
        nuclease: list(record.get("pins_checked", []))
        for nuclease, record in payload.get("nucleases", {}).items()
        if record.get("pins_checked")
    }


def render_verified(data_dir: Path = DEFAULT_DATA_DIR) -> str:
    """The success text of `verify`: where, and which prepare-time pins stand behind it."""
    data_dir = Path(data_dir)
    lines = [f"온전성 확인 통과: {data_dir / MANIFEST}"]
    for nuclease, pins in recorded_pins(data_dir).items():
        lines.append(f"{nuclease}: prepare 시점에 대조한 고정 핀 {len(pins)}개")
        lines += [f"  - {name}" for name in pins]
    return "\n".join(lines)


def verify(data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    """Every disagreement between `data_dir` and its manifest; empty means intact."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / MANIFEST
    if not manifest_path.is_file():
        return [f"manifest 없음: {manifest_path}"]
    files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    problems = []
    for name, entry in files.items():
        path = data_dir / name
        if not path.is_file():
            problems.append(f"없음: {name}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"sha256 불일치: {name}")
    for path in sorted(data_dir.rglob("*")):
        name = path.relative_to(data_dir).as_posix()
        if path.is_file() and name != MANIFEST and name not in files:
            problems.append(f"manifest에 없는 파일: {name}")
    return problems
