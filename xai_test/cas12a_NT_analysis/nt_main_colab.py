"""Build the uniquely named Test-1500 Colab notebook and matching source ZIP."""

import argparse
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

from cas12a.nt_data import canonical_bytes

from .nt_colab import REQUIREMENTS, notebook
from .nt_run import DirectoryStore, file_hash
from .nt_validation_protocol import code_snapshot


def main_notebook(checksum, archive_name):
    # Keep the proven dependency and checked extraction cells, using a short,
    # build-specific root and a source filename that cannot be confused with a pilot.
    book = notebook(checksum)
    book["cells"] = book["cells"][:3]
    book["cells"][0]["source"] = """# Cas12a NT · Test 1,500행 본 분석
기존 Test membership과 학습 완료 NT를 사용해 전체 예측·ISM·rollout, 고정 IG와 모든
기술 사례를 저장합니다. 모델을 한 번 로드하고 완료된 소스는 재개 시 재사용합니다.
검증 시연용 노트북이 아닙니다. 전체/TTTV/TTTT 성능과 설명·사례 보고서는 저장 결과에서
GPU 없이 다시 만들 수 있습니다. 같은 빌드의 아래 ZIP 하나를 업로드하세요.

입력 경로·IG 범위·실행 reservation은 경로 설정 셀 한 곳에 있습니다. IG 기본 범위는
전수 우선이며 affordability가 확인됐다는 뜻은 아닙니다. 자동 400행 fallback은 없습니다.
기존 초과 study의 정책 결정은 미확정입니다. `RESERVED_SECONDS=None` 상태에서는 GPU를
실행하지 않습니다. 새 output 폴더로 예산을 초기화하지 않으며 기존 ledger/receipt를
사용합니다. 기존 study가 차단돼 있으면 그대로 중단합니다. 정책 변경은 별도 결정 사항입니다.
동일 study는 한 런타임에서 실행하세요. 이 전달물 자체는 실제 Test GPU 완료 증거가 아닙니다.
"""
    upload = book["cells"][2]["source"]
    upload = upload.replace("cas12a-validation-source.zip", archive_name)
    upload = upload.replace(
        "/content/cas12a-validation146", "/content/c12main-" + checksum[:12]
    )
    book["cells"][2]["source"] = upload

    def add(kind, source):
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n",
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        book["cells"].append(cell)

    add(
        "code",
        """
    # 3. 기존 Drive 경로와 본 분석 설정 — 모델은 아직 로드하지 않습니다.
    from xai_test.cas12a_NT_analysis.nt_inputs import verify_test_inputs
    from xai_test.cas12a_NT_analysis.nt_main import prepare_main, run_main
    from xai_test.cas12a_NT_analysis.nt_main_report import report_main
    BASE = Path("/content/drive/MyDrive/cas12a-nt")
    SOURCE = BASE / "source.xlsx"
    PARTITION = BASE / "partition.json"
    MODEL_RUN = BASE / "runs/20261101-nt500m-full"
    OUTPUT = Path("/content/drive/MyDrive/cas12a-xai/main-test1500") / __BUILD__
    STUDY_ROOT = Path("/content/drive/MyDrive/cas12a-xai/study-138")
    EVIDENCE_ROOT = Path("/content/drive/MyDrive/cas12a-xai/evidence-146")
    RESERVED_SECONDS = None  # 기존 누적 예산 안에서 결정된 reservation만 입력. 추가 예산 승인이 아닙니다.
    COVERAGE_DIRECTORY = None  # 사전에 고정한 subset이 있으면 기존 coverage manifest 폴더
    COVERAGE_CHECKSUM = None   # 선택 사항; 해당 manifest의 저장된 checksum.json에서 자동 읽기
    if COVERAGE_DIRECTORY is not None:
        COVERAGE_CHECKSUM = json.loads((Path(COVERAGE_DIRECTORY) / "checksum.json").read_bytes())["sha256"]
    inputs = verify_test_inputs(SOURCE, PARTITION, MODEL_RUN / "selected_export", MODEL_RUN / "durable_export.json")
    plan = prepare_main(inputs, OUTPUT, coverage_directory=COVERAGE_DIRECTORY, coverage_checksum=COVERAGE_CHECKSUM)
    print("Test 행 수:", len(inputs.rows), "IG 통계 행 수:", len(plan["statistics_ids"]))
    print("완료 소스 재사용/새 결과 저장 위치:", OUTPUT)
    """.replace("__BUILD__", repr(checksum[:12])),
    )
    add(
        "code",
        """
    # 4. 본 분석. 과거 study를 재생성하거나 미정산 시간을 지우지 않습니다.
    from xai_test.cas12a_NT_analysis.nt_validation import open_study
    if RESERVED_SECONDS is None:
        raise ValueError("GPU 실행 정책 미확정: 기존 초과 study와 reservation 결정을 먼저 해결해야 합니다. 준비·보고는 계속 사용할 수 있습니다.")
    study = open_study(STUDY_ROOT, EVIDENCE_ROOT, initialize=False)
    def show_progress(event):
        print(f"{event['method']} {event['completed']}/{event['requested']} · 저장: {event['output_dir']}", flush=True)
    summary = run_main(inputs, OUTPUT, device="cuda:0", study=study,
                       reserved_seconds=RESERVED_SECONDS, on_progress=show_progress)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    """,
    )
    add(
        "code",
        """
    # 5. 저장 결과 → CSV/NPZ/JSON/정적 그림/한국어 보고서. 중단됐어도 실행 가능합니다.
    # 이 셀은 모델 추론을 실행하지 않습니다. GPU 없이 재접속해 1~3, 5셀을 실행해도 됩니다.
    from IPython.display import Markdown, display
    report = report_main(OUTPUT)
    REPORT_DIR = Path(report["output_dir"])
    print("저장 보고서:", REPORT_DIR, "전체 완료:", report["complete"])
    display(Markdown((REPORT_DIR / "report.ko.md").read_text(encoding="utf-8")))
    for image_name in ("position_profiles.svg", "token_profiles.svg", "agreement_distribution.svg"):
        from IPython.display import SVG
        display(SVG(filename=str(REPORT_DIR / image_name)))
    """,
    )
    add(
        "code",
        """
    # 6. GPU 추론 없이 전체 원시 shard와 현재 보고서를 다운로드합니다.
    import shutil
    archive = shutil.make_archive("/content/cas12a-test1500-results-" + __BUILD__, "zip", OUTPUT)
    files.download(archive)
    """.replace("__BUILD__", repr(checksum[:12])),
    )
    for cell in book["cells"]:
        cell["id"] = hashlib.sha256(cell["source"].encode()).hexdigest()[:12]
        cell["metadata"]["id"] = cell["id"]
    return book


def build_main_bundle(directory):
    store = DirectoryStore(directory)
    if store.root.exists() and any(store.root.iterdir()):
        raise ValueError("main bundle requires a new empty directory")
    source = code_snapshot()
    doc = Path(__file__).resolve().parent / "docs/colab-main-analysis.md"
    source["xai_test/cas12a_NT_analysis/docs/colab-main-analysis.md"] = doc.read_bytes()
    source["requirements-colab.txt"] = REQUIREMENTS.encode()
    manifest = {
        "schema": 1,
        "files": {n: hashlib.sha256(v).hexdigest() for n, v in source.items()},
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(
            {**source, "source-manifest.json": canonical_bytes(manifest)}.items()
        ):
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    data = buffer.getvalue()
    checksum = hashlib.sha256(data).hexdigest()
    zip_name = f"cas12a-nt-test1500-main-source-{checksum[:12]}.zip"
    notebook_name = f"Cas12a_NT_Test1500_main_{checksum[:12]}_colab.ipynb"
    store.put(zip_name, data)
    store.put(
        notebook_name,
        json.dumps(
            main_notebook(checksum, zip_name), ensure_ascii=False, indent=2
        ).encode(),
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
    for name in ("build", "report"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True)
        if name == "report":
            command.add_argument("--results", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_main_bundle(args.output)
    else:
        from .nt_main_report import report_main

        result = report_main(args.results, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
