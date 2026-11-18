"""#193: TensorFlow 없는 로컬 스테이징을 만들고, 네트워크를 막은 채 최소 실행한다.

채택 런타임이 **배포되는 모습**(TensorFlow 없음)에서 실제로 도는지 보는 유일한
근거다. 이 PC에는 TensorFlow가 설치돼 있어서 `Bidirectional` 역직렬화가 그것을
끌어온다 - 개발 환경에서 잰 값은 배포 구성을 대표하지 못한다.

site-packages에서 필요한 패키지만 복사한다 (네트워크 사용 안 함). 실행은 `-S`로
site-packages를 경로에서 제외하므로 TensorFlow는 import 자체가 불가능하다.
`WANTED` 목록은 추측이 아니라 실패하는 import를 따라가며 확정한 것이다.

실행:

    python docs/runtime/stage_offline.py <스테이징 경로>

스테이징 트리는 일회용이다. 복구 의존 대상이 아니며 이 명령으로 다시 만든다.
네트워크 차단은 이 프로세스 안에서 socket을 막는 방식이다. OS 수준 차단이나
깨끗한 오프라인 PC는 #204의 몫이며 여기서 입증하지 않는다.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

from verification_bundle import stage_execution, read_json
from verification_policy import runtime_version_failures

SITE = Path(sysconfig.get_paths()["purelib"])
STAGE = None
REPO = Path(__file__).resolve().parents[2]

# 채택 런타임이 실제로 쓰는 것만. TensorFlow와 그 전용 의존성은 일부러 제외한다.
WANTED = [
    "keras", "torch", "numpy", "h5py", "shap", "ml_dtypes", "namex", "optree",
    "typing_extensions.py", "sympy", "mpmath", "networkx", "filelock",
    "fsspec", "jinja2", "markupsafe", "packaging", "rich", "pygments",
    "markdown_it", "mdurl", "colorama", "six.py", "setuptools", "pkg_resources",
    # 확장 모듈이 의존하는 재배포 DLL 묶음. 빠지면 numpy C 확장이 로드되지 않는다.
    "numpy.libs", "torchgen", "yaml", "_yaml", "absl", "google", "psutil",
]
# runtime_versions reads SHAP distribution metadata without importing its optional
# plotting/scientific dependencies during the minimum forward/gradient run.
WANTED += [path.name for path in SITE.glob("shap-*.dist-info")]

RUNNER = r'''
import json, os, sys

staging, repo, raw_output = sys.argv[1:4]
# 표준 라이브러리는 남기고 site-packages만 경로에서 뺀다. -S로 이미 추가되지 않지만
# 혹시 섞여 들어와도 스테이징만 보도록 명시한다.
sys.path[:] = [staging, repo] + [
    p for p in sys.path if "site-packages" not in p.replace("\\", "/")
]

# 네트워크 차단: 준비와 실행 모두 네트워크 없이 된다는 것을 실제로 보장한다.
import socket
# socket.socket은 ssl이 상속하므로 클래스로 남겨 두고, 실제 연결만 막는다.
def _blocked(*args, **kwargs):
    raise OSError("network is disabled for this offline verification run")
class _BlockedSocket(socket.socket):
    connect = _blocked
    connect_ex = _blocked
    sendto = _blocked
socket.socket = _BlockedSocket
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
socket.gethostbyname = _blocked

import importlib.util
report = {
    "site_packages_on_path": any("site-packages" in p for p in sys.path),
    "tensorflow_importable": importlib.util.find_spec("tensorflow") is not None,
    "keras_importable": importlib.util.find_spec("keras") is not None,
}
try:
    import tensorflow  # noqa: F401
    report["tensorflow_import_attempt"] = "succeeded"
except Exception as error:
    report["tensorflow_import_attempt"] = f"{type(error).__name__}: {error}"

probes = []
for attempt in (lambda: socket.socket().connect(("1.1.1.1", 80)),
                lambda: socket.getaddrinfo("huggingface.co", 443),
                lambda: socket.create_connection(("1.1.1.1", 80))):
    try:
        attempt()
        probes.append("reached network")
    except OSError as error:
        probes.append(f"blocked: {error}")
report["network_probes"] = probes
report["network_blocked"] = all(p.startswith("blocked") for p in probes)

import psutil
process = psutil.Process()
def rss():
    return int(process.memory_info().rss)

report["rss"] = {"interpreter": rss()}
import numpy as np
import non_nt_runtime as rt

rt.import_keras()
report["rss"]["runtime_import"] = rss()
report["versions"] = rt.runtime_versions()
report["models"] = {}
raw = {}
held_models = {}
for key, head, window in (
    ("spcas9", 1, "CCGGTCGCTTTAAGAATAAATAAAAGCGCC"),
    ("cas12a", 0, "TTTAAGAATAAATAAAAGCGCCACAGGATTGACT"),
):
    runtime = rt.NonNTRuntime.load(key)
    held_models[key] = runtime
    report["rss"][f"{key}_model_loaded"] = rss()
    inputs = [rt.one_hot(window)[None]]
    if key == "cas12a":
        inputs.append(rt.cas12a_derive(window)[None])
    value = runtime.forward(inputs, head=head)
    output, grads = runtime.gradients(inputs, head=head)
    raw[f"{key}/output"] = value
    for index, grad in enumerate(grads):
        raw[f"{key}/gradient/{index}"] = grad
    report["models"][key] = {
        "raw_output": float(value[0]),
        "n_outputs": runtime.n_outputs,
        "gradient_abs_sum": [float(np.abs(g).sum()) for g in grads],
        "all_finite": bool(np.isfinite(value).all()
                           and all(np.isfinite(g).all() for g in grads)),
        "ops": sorted({type(layer).__name__ for layer in runtime._model.layers}),
        "held_models": list(held_models),
    }
    report["rss"][f"{key}_after_gradients"] = rss()
report["tensorflow_in_sys_modules_after_load"] = "tensorflow" in sys.modules
report["rss"]["peak_working_set"] = int(process.memory_info().peak_wset)
report["rss"]["note"] = ("one process, both models loaded in sequence and both kept "
                         "alive; the shipped design loads only the selected family")
report["status"] = "pass" if (
    report["network_blocked"]
    and not report["tensorflow_importable"]
    and not report["tensorflow_in_sys_modules_after_load"]
    and all(m["all_finite"] and all(s > 0 for s in m["gradient_abs_sum"])
            for m in report["models"].values())
) else "fail"
print("STAGEDJSON" + json.dumps(report))
np.savez_compressed(raw_output, **raw)
'''


def stage() -> dict:
    STAGE.mkdir(parents=True, exist_ok=True)
    copied, missing, total_bytes, total_files = [], [], 0, 0
    started = time.time()
    for name in WANTED:
        source = SITE / name
        if not source.exists():
            missing.append(name)
            continue
        target = STAGE / name
        if source.is_dir():
            if not target.exists():
                shutil.copytree(source, target, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            if not target.exists():
                shutil.copy2(source, target)
        for path in ([target] if target.is_file() else target.rglob("*")):
            if path.is_file():
                total_files += 1
                total_bytes += path.stat().st_size
        copied.append(name)
    return {"staged": copied, "missing_from_site_packages": missing,
            "files": total_files, "bytes": total_bytes,
            "seconds": round(time.time() - started, 1),
            "source": str(SITE), "staging": str(STAGE), "network_used": False}


def run(work: Path) -> int:
    info = stage()
    runner = STAGE.parent / "staged_runner.py"
    runner.write_text(RUNNER, encoding="utf-8")
    finished = subprocess.run(
        [sys.executable, "-S", str(runner), str(STAGE), str(REPO), str(work / "offline-raw.npz")],
        capture_output=True, text=True, cwd=str(REPO),
        env={k: v for k, v in os.environ.items()
             if k not in ("PYTHONPATH", "KERAS_BACKEND")},
    )
    lines = [l for l in finished.stdout.splitlines() if l.startswith("STAGEDJSON")]
    out = {"staging": info, "command": finished.args, "exit_code": finished.returncode,
           "executable": sys.executable}
    if lines:
        out["run"] = json.loads(lines[0][len("STAGEDJSON"):])
    else:
        out["run"] = {"status": "fail", "stderr": finished.stderr[-4000:],
                      "stdout": finished.stdout[-2000:]}
    version_errors = runtime_version_failures(out["run"].get("versions", {}),
                                              read_json(work / "snapshot/contract.json"))
    if version_errors:
        out["run"].update(status="fail", version_errors=version_errors)
    (work / "staged.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not (work / "offline-raw.npz").exists():
        # Preserve a completed failure report; missing arrays fail required inventory.
        import numpy as np
        np.savez_compressed(work / "offline-raw.npz")
    print(json.dumps(out, indent=2))
    return 0 if finished.returncode == 0 and out["run"].get("status") == "pass" else 1


def main() -> int:
    global STAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    STAGE = args.staging.resolve()
    if args.output.resolve() != work / "staged.json":
        parser.error("--output must be <work>/staged.json to retain stage provenance")
    try:
        with stage_execution(work, "offline") as result:
            result["exit_code"] = run(work)
        return result["exit_code"]
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
