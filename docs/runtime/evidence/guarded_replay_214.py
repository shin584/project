"""Run the archived reporting CLI with original inputs and measurements denied."""

import json
import os
from pathlib import Path
import runpy
import sys


def main():
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output, restored = Path(config["output"]), Path(config["restored"])
    code = restored / "bundle/snapshot/code"
    sys.path[:0] = [str(code / "docs/runtime"), str(code)]
    originals = [os.path.normcase(os.path.realpath(p)) for p in config["original_roots"]]
    report = {"probes": [], "violations": [], "scope": "Python audit guards; not OS isolation"}

    def audit(event, args):
        if event in ("open", "os.listdir", "os.scandir", "ctypes.dlopen") and args:
            value = args[0]
            if isinstance(value, (str, bytes, os.PathLike)):
                path = os.path.normcase(os.path.realpath(os.fsdecode(value)))
                if any(path == root or path.startswith(root + os.sep) for root in originals):
                    report["violations"].append({"event": event, "path": path})
                    raise PermissionError("Original measurement/source path denied")
        if (event.startswith(("subprocess.", "os.exec", "os.spawn", "socket."))
                or event == "os.system"):
            report["violations"].append({"event": event})
            raise PermissionError("Replay cannot start processes or use network")
        if event == "import" and args[0].split(".")[0] in (
                "keras", "torch", "tensorflow", "shap", "non_nt_runtime"):
            report["violations"].append({"event": event, "module": args[0]})
            raise PermissionError("Model measurement imports disabled in report replay")

    sys.addaudithook(audit)
    for path in config["probe_files"]:
        try:
            with open(path, "rb"):
                pass
            blocked = False
        except PermissionError:
            blocked = True
        report["probes"].append({"path": path, "blocked": blocked})
    report["violations"].clear()
    script = code / "docs/runtime/assemble_verification.py"
    sys.argv = [str(script), "--work", str(restored / "bundle"),
                "--output", str(output / "summary-regenerated.json")]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        report["module_origins"] = {name: str(Path(module.__file__).resolve())
                                    for name, module in list(sys.modules.items())
                                    if getattr(module, "__file__", None)}
        report["executable"] = sys.executable
        report["python"] = sys.version
        (output / "guard-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
