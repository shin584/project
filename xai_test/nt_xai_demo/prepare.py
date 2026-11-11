"""`prepare`: gather the NT XAI 데모's saved artifacts into one checksummed directory.

Two sources, one destination (`xai_test/nt_xai_demo/data/`), one manifest:

* Cas12a — a light subset of the Test1500 result ZIP. Per-row shards and the
  large `*.sidecar.json` files are read in memory to check their pins but are
  never extracted. `agreement.json` is sliced by key selection only.
* SpCas9 (9-Hydra v3) — copied from the in-place artifacts and the published
  split. The bulk part of it is not in Git; when it is absent, only Cas12a is
  prepared and the absence is reported, not hidden.

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
    pass


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

    def render(self) -> str:
        lines = [f"Cas12a: 준비됨 (원본 고정 해시 {self.cas12a_pins}개 대조 통과)"]
        if self.spcas9_missing:
            lines.append(
                "SpCas9: 부재 — 원본 묶음이 없어 준비하지 않았습니다. 없는 원본:"
            )
            lines += [f"  - {name}" for name in self.spcas9_missing]
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
) -> tuple[list[Pin], dict[str, bytes], dict[str, Any], str]:
    """Pins, files to write (`cas12a/...` -> bytes), the agreement-slice record
    and the archive's sha256."""
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
    sliced = {key: agreement[key] for key in AGREEMENT_KEPT}
    files[AGREEMENT_SLICE] = json.dumps(sliced, ensure_ascii=False).encode("utf-8")
    derived = {
        "source": f"zip:{prefix}agreement.json",
        "source_sha256": sha256_bytes(agreement_raw),
        "operation": "최상위 키 선택만 함; 값은 원본 그대로이며 산술 없음",
        "kept": list(AGREEMENT_KEPT),
        "dropped": sorted(set(agreement) - set(AGREEMENT_KEPT)),
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

    Raises a `PrepareRefused` without touching `data_dir` when a Cas12a source
    is missing, when any pin disagrees, or when an absent SpCas9 bundle would
    replace SpCas9 data an earlier run prepared (on a clone, the committed
    `data/spcas9/` copies).
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
    failed = [p for p in [*cas12a_pins, *spcas9_pins] if not p.ok]
    if failed:
        raise PinMismatch(failed)
    if missing and _previous_spcas9_status(data_dir) == "prepared":
        raise SpCas9WouldBeLost(
            "SpCas9 원본 묶음이 없는데 기존 data/에 SpCas9 자료가 있어 덮어쓰지 않았습니다. "
            "묶음을 복원한 뒤 다시 실행하십시오 (xai_test/nt_xai_demo/DATA.md). 없는 원본:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    prefix = sources.cas12a_report_prefix

    def origin(name: str) -> str:
        if name == AGREEMENT_SLICE:
            return f"zip:{prefix}agreement.json (sliced)"
        if name.startswith("cas12a/"):
            member = name.removeprefix("cas12a/")
            return f"zip:{member if member in CAS12A_TOP else prefix + member}"
        return sources.display(spcas9_files[name])

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prepare-", dir=data_dir.parent))
    try:
        files = _write_tree(stage, cas12a_files, spcas9_files, origin)
        # The copies are hashed again; a source that changed after its pins
        # were checked must not be certified by the new manifest.
        changed = [
            Pin(f"{name} 복사본 sha256 (대조 시점)", value, files[name]["sha256"])
            for name, value in spcas9_checked.items()
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
                    {"status": "absent", "missing": missing}
                    if missing
                    else {
                        "status": "prepared",
                        "pins_checked": [p.name for p in spcas9_pins],
                    }
                ),
            },
            "derived": {AGREEMENT_SLICE: derived},
            "files": files,
        }
        (stage / MANIFEST).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
    )


def _previous_spcas9_status(data_dir: Path) -> str | None:
    try:
        payload = json.loads((data_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload.get("nucleases", {}).get("spcas9", {}).get("status")


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
