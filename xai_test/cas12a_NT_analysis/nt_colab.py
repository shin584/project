"""Build a self-contained Colab handoff; audit returned evidence without a model."""

import argparse
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path

from cas12a.nt_data import canonical_bytes

from .nt_run import DirectoryStore, file_hash
from .nt_validation import audit_evidence
from .nt_validation_protocol import code_snapshot

REQUIREMENTS = """numpy==2.3.2
pandas==2.3.3
scipy==1.16.1
openpyxl==3.1.5
safetensors==0.7.0
transformers==5.8.1
"""


def notebook(bundle_sha256):
    cells = []

    def add(kind, source):
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n",
        }
        cell["id"] = hashlib.sha256(cell["source"].encode()).hexdigest()[:12]
        cell["metadata"]["id"] = cell["id"]
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add(
        "markdown",
        """
    # Cas12a NT #146 — validation probe와 IG 범위 동결
    이 노트북은 기존 NT 모델과 partition을 읽고 **validation만** 분석합니다.
    GPU 런타임을 선택한 뒤 순서대로 실행하세요. Test 분석과 재학습은 없습니다.
    상세 안내는 코드 ZIP의 `xai_test/cas12a_NT_analysis/docs/colab-validation.md`에 있습니다.
    동시에 다른 런타임을 실행하지 마세요. 기존 study가 있으면 동일 경로와 receipt로 재개합니다.
    """,
    )
    add(
        "code",
        """
    # 1. 의존성 설치. Colab의 CUDA PyTorch는 교체하지 않습니다.
    import subprocess, sys
    requirements = __REQUIREMENTS__
    subprocess.run([sys.executable, "-m", "pip", "install", *requirements.splitlines()], check=True)
    import importlib, importlib.metadata
    for package in ("numpy", "pandas", "scipy", "transformers", "safetensors"):
        if package in sys.modules and getattr(sys.modules[package], "__version__", None) != importlib.metadata.version(package):
            raise RuntimeError("설치된 버전과 로드된 버전이 다릅니다. 런타임을 다시 시작하고 처음부터 실행하세요.")
    """.replace("__REQUIREMENTS__", repr(REQUIREMENTS)),
    )
    add(
        "code",
        """
    # 2. 전달받은 source ZIP 업로드 및 정확한 bytes 검증
    from google.colab import files, drive
    import hashlib, json, zipfile
    from pathlib import Path, PurePosixPath
    EXPECTED_SOURCE_SHA256 = __HASH__
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("cas12a-validation-source.zip 하나만 업로드하세요.")
    archive_path = Path(next(iter(uploaded)))
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError("이 노트북과 같은 빌드의 ZIP이 아닙니다.")
    CODE_ROOT = Path("/content/cas12a-validation146")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("source-manifest.json"))
        if len(set(names)) != len(names) or set(names) != {"source-manifest.json", *manifest["files"]}:
            raise ValueError("source ZIP file set mismatch")
        for name in names:
            if "\\\\" in name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError("unsafe source archive path")
            data = archive.read(name)
            if name != "source-manifest.json" and hashlib.sha256(data).hexdigest() != manifest["files"][name]:
                raise ValueError("source member hash mismatch")
            target = CODE_ROOT / name
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError("기존 코드 폴더가 다릅니다. 같은 빌드를 사용하세요.")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(data)
    sys.path.insert(0, str(CODE_ROOT))
    drive.mount("/content/drive")
    """.replace("__HASH__", repr(bundle_sha256)),
    )
    add(
        "markdown",
        """
    ## 경로와 최초 생성/재개
    아래 경로를 본인의 Drive에 맞게 설정하세요. 원래 모델과 split을 그대로 사용합니다.
    `FIRST_EVER_STUDY=True`는 이 연구의 첫 생성일 때만 선택합니다. 이전 실행 기록이
    사라졌거나 비용을 이미 사용했다면 새 root를 만들지 말고 기존 ledger/receipt를 복구하세요.
    """,
    )
    add(
        "code",
        """
    # 3. 경로 설정 — 이 셀은 모델을 로드하지 않습니다.
    from xai_test.cas12a_NT_analysis.nt_inputs import verify_inputs
    from xai_test.cas12a_NT_analysis.nt_validation_protocol import prepare_protocol
    from xai_test.cas12a_NT_analysis.nt_validation import (
        open_study, freeze_protocol, run_phase, freeze_ig_coverage, export_evidence,
    )
    from cas12a.nt_data import digest
    BASE = Path("/content/drive/MyDrive/cas12a-nt")
    SOURCE = BASE / "source.xlsx"
    PARTITION = BASE / "partition.json"
    MODEL_RUN = BASE / "runs/20261101-nt500m-full"
    EXPORT = MODEL_RUN / "selected_export"
    EXPORT_RECEIPT = MODEL_RUN / "durable_export.json"
    STUDY_ROOT = Path("/content/drive/MyDrive/cas12a-xai/study-138")
    EVIDENCE_ROOT = Path("/content/drive/MyDrive/cas12a-xai/evidence-146")
    FIRST_EVER_STUDY = False  # 이 연구의 기존 지출이 없는 최초 생성일 때만 True
    inputs = verify_inputs(SOURCE, PARTITION, EXPORT, EXPORT_RECEIPT)
    protocol = prepare_protocol(PARTITION)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    print("승인할 protocol SHA256:", digest(protocol))
    """,
    )
    add(
        "markdown",
        """
    ## 결과 관측 전 실행 기준 확인
    바로 위 protocol의 validation ID·수치 기준·forecast/예산 정책을 확인하고 hash를
    아래에 붙여 넣으세요. 독립 설계 리뷰는 ZIP에 포함돼 있습니다. 품질 기준은 결과를
    본 뒤 바꾸지 않습니다. 아래 확인은 **새 FP32 Test를 아직 보지 않았고 이 연구를
    실행하는 런타임이 하나뿐**임을 뜻합니다. 과거 FP16 Test는 알려진 상태로 기록합니다.
    """,
    )
    add(
        "code",
        """
    # 4. 명시적 확인 후 protocol 동결. 아직 모델 추론은 시작하지 않습니다.
    APPROVED_PROTOCOL_SHA256 = ""  # 위에 표시된 hash 붙여 넣기
    SINGLE_EXECUTOR = False        # 이 연구의 실행자가 본 런타임 하나인지 확인
    NO_NEW_FP32_TEST = False       # 새 FP32 Test 결과 미관측인지 확인
    if APPROVED_PROTOCOL_SHA256 != digest(protocol) or not SINGLE_EXECUTOR or not NO_NEW_FP32_TEST:
        raise ValueError("protocol hash와 두 확인 항목을 먼저 설정하세요.")
    study = open_study(STUDY_ROOT, EVIDENCE_ROOT, initialize=FIRST_EVER_STUDY)
    frozen = freeze_protocol(inputs, study, PARTITION, protocol, EVIDENCE_ROOT,
        approved_sha256=APPROVED_PROTOCOL_SHA256,
        single_executor=SINGLE_EXECUTOR, no_new_fp32_test=NO_NEW_FP32_TEST)
    print("동결 완료:", frozen["protocol_sha256"], "이미 차감된 시간:", study.status()["charged_seconds"])
    """,
    )
    add(
        "markdown",
        """
    ## 검증 실행
    정확성 8행과 별도 warmup 2행, 실제 배치 32행씩 3그룹을 실행합니다.
    단계마다 같은 ledger에 예약 시간을 차감하며 receipt를 별도 Drive 폴더에 자동 보관합니다.
    수치 실패는 후속 실행을 차단합니다. 중단됐으면 아래 수동 정산 절차를 먼저 사용하세요.
    """,
    )
    add(
        "code",
        """
    # 5. 정확성 probe. 중단 후 수동 정산했을 때만 해당 attempt를 명시적으로 올립니다.
    ATTEMPTS = {name: 0 for name in protocol["phase_order"]}
    report = run_phase(inputs, study, frozen, "correctness", EVIDENCE_ROOT,
        attempt=ATTEMPTS["correctness"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["accepted"]:
        raise RuntimeError("정확성 실패: 후속 실행을 멈추고 반환 ZIP을 만드세요.")
    """,
    )
    add(
        "code",
        """
    # 6. 별도 warmup과 production 비용 반복 측정
    for phase in protocol["phase_order"][1:]:
        report = run_phase(inputs, study, frozen, phase, EVIDENCE_ROOT, attempt=ATTEMPTS[phase])
        print(phase, "accepted=", report["accepted"], "rates=", report.get("rates"))
        if not report["accepted"]:
            raise RuntimeError("수치 실패: 더 실행하지 말고 반환 ZIP을 만드세요.")
    """,
    )
    add(
        "code",
        """
    # 7. 저장 값 품질 재계산 → 전수 우선 forecast → IG membership 동결
    freeze = freeze_ig_coverage(study, frozen, PARTITION, EVIDENCE_ROOT)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    if not freeze["feasible"]:
        print("최소 400행도 예산 내 불가능합니다. Test를 실행하지 말고 결과를 반환하세요.")
    else:
        print("coverage 동결 완료. 이 노트북에는 Test 실행 단계가 없습니다. 결과를 반환하세요.")
    """,
    )
    add(
        "markdown",
        """
    ## 반환 ZIP — 성공·실패 모두 전달
    아래 셀은 별도로 실행할 수 있습니다. 저장 실패로 history를 검증할 수 없다면
    진단 자료로 표시하며 완료로 처리하지 않습니다. ZIP과 출력 SHA256을 함께 전달하세요.
    """,
    )
    add(
        "code",
        """
    # 8. 결과 묶음 저장/다운로드. 기존 파일은 덮어쓰지 않습니다.
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("/content/drive/MyDrive/cas12a-xai/returns") / f"validation146-{stamp}.zip"
    from xai_test.cas12a_NT_analysis.nt_run import Study
    # 재접속/저장 실패로 in-memory 객체가 없어도 원시 진단 자료를 보존합니다.
    frozen = json.loads((EVIDENCE_ROOT / "protocols" / f"{digest(protocol)}.json").read_bytes())
    receipt = export_evidence(Study(STUDY_ROOT), frozen, EVIDENCE_ROOT, output)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    files.download(str(output))
    """,
    )
    add(
        "markdown",
        """
    ## 수동 복구 — 정상 실행이면 건너뜁니다
    먼저 **이전 런타임/프로세스가 실제로 종료됐는지 독립적으로 확인**하세요.
    heartbeat 만료나 새 노트북 열기는 종료 증거가 아닙니다. 종료 증거 파일과 UTC Unix
    seconds, 시작부터 종료까지를 포함하는 보수적 상한을 제공해야 합니다. 불확실하면 중단합니다.
    저장 실패나 receipt 유실은 이 셀로 자동 복구하지 않습니다.
    """,
    )
    add(
        "code",
        """
    # 평소에는 False. 중단된 런타임 종료를 확인한 경우에만 설정합니다.
    RECOVER_INTERRUPTED_JOB = False
    if RECOVER_INTERRUPTED_JOB:
        TERMINATION_EVIDENCE = Path("/content/drive/MyDrive/cas12a-xai/termination-evidence.txt")
        TERMINATED_UTC = None  # 독립 확인한 종료 UTC Unix seconds
        UPPER_SECONDS = None  # 시작~종료 구간의 보수적 시간 상한
        study = open_study(STUDY_ROOT, EVIDENCE_ROOT)
        study.settle(TERMINATION_EVIDENCE, terminated_utc=TERMINATED_UTC, upper_seconds=UPPER_SECONDS)
        print("정산 완료:", study.receipt)
        print("최초 생성 모드를 끄고 4번 동결 셀부터 재개하세요. 미완료 phase의 attempt만 올립니다.")
    """,
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_bundle(directory):
    """Package exact local source, never real assets or unrelated working changes."""
    store = DirectoryStore(directory)
    if store.root.exists() and any(store.root.iterdir()):
        raise ValueError("bundle output requires a new empty directory")
    source = code_snapshot()
    analysis = Path(__file__).resolve().parent
    source["xai_test/cas12a_NT_analysis/docs/colab-validation.md"] = (
        analysis / "docs/colab-validation.md"
    ).read_bytes()
    source["requirements-colab.txt"] = REQUIREMENTS.encode()
    manifest = {
        "schema": 1,
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in source.items()
        },
    }
    store.root.mkdir(parents=True, exist_ok=True)
    archive_path = store.path("cas12a-validation-source.zip")
    with (
        archive_path.open("xb") as target,
        zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        for name, data in sorted(source.items()):
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
        archive.writestr(
            zipfile.ZipInfo("source-manifest.json", (1980, 1, 1, 0, 0, 0)),
            canonical_bytes(manifest),
        )
    source_sha = file_hash(archive_path)
    store.put(
        "Cas12a_NT_validation_colab.ipynb",
        json.dumps(notebook(source_sha), ensure_ascii=False, indent=2).encode(),
    )
    result = {
        "schema": 1,
        "files": {p.name: file_hash(p) for p in sorted(store.root.iterdir())},
    }
    store.put("bundle.json", canonical_bytes(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--archive", required=True)
    audit.add_argument("--sha256", required=True)
    audit.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = (
        build_bundle(args.output)
        if args.command == "build"
        else audit_evidence(args.archive, checksum=args.sha256, directory=args.output)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
