"""Restored minimum-run process. Python audit guards are not OS isolation."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig


def main():
    root = Path(sys.argv[1]).resolve()
    config = json.loads((root / "runner-config.json").read_text(encoding="utf-8"))
    payload, dependencies = root / "payload", root / "dependencies"
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    sys.path[:] = [str(payload), str(dependencies)] + [
        p for p in sys.path if "site-packages" not in p.lower()]
    os.chdir(root)
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
    originals = [os.path.normcase(os.path.realpath(p)) for p in config["original_roots"]]
    violations = []

    def forbidden(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        path = os.path.normcase(os.path.realpath(os.fsdecode(value)))
        return any(path == p or path.startswith(p + os.sep) for p in originals)

    def audit(event, args):
        if event in ("open", "os.listdir", "os.scandir", "ctypes.dlopen") and args and forbidden(args[0]):
            violations.append({"event": event, "path": os.fsdecode(args[0])})
            raise PermissionError("original path access denied")
        if event.startswith(("subprocess.", "os.exec", "os.spawn", "os.posix_spawn")) or event == "os.system":
            raise PermissionError("child process disabled in recovery target")
        if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.gethostbyname",
                     "socket.gethostbyaddr", "socket.bind", "socket.sendto", "socket.sendmsg"):
            raise PermissionError("network disabled in recovery target")

    sys.addaudithook(audit)
    report = {"status": "fail", "models": {}, "held_models": [],
              "external_prerequisites": {"executable": sys.executable,
                  "python": platform.python_version(), "stdlib": str(stdlib), "os": platform.platform()},
              "path_guard": {"scope": "Python audit open/listdir/scandir/dlopen and child-process denial; not OS isolation",
                             "original_roots": config["original_roots"], "probes": [], "violations": []}}
    try:
        for path in config["original_files"]:
            try:
                with open(path, "rb"):
                    pass
                blocked = False
            except PermissionError:
                blocked = True
            report["path_guard"]["probes"].append({"path": path, "blocked": blocked})
        violations.clear()
        import socket
        report["network_scope"] = "in-process Python socket audit events; no OS firewall or clean-PC claim"
        report["network_probes"] = []
        def connect():
            with socket.socket() as connection:
                connection.connect(("127.0.0.1", 9))
        def udp():
            with socket.socket(type=socket.SOCK_DGRAM) as connection:
                connection.sendto(b"probe", ("127.0.0.1", 9))
        for name, attempt in (("tcp", connect), ("dns", lambda: socket.getaddrinfo("example.com", 443)),
                              ("udp", udp)):
            try:
                attempt()
                blocked = False
            except PermissionError:
                blocked = True
            report["network_probes"].append({"kind": name, "blocked": blocked})
        report["network_blocked"] = all(p["blocked"] for p in report["network_probes"])
        report["tensorflow_importable"] = importlib.util.find_spec("tensorflow") is not None
        try:
            import tensorflow  # noqa: F401
            report["tensorflow_import_attempt"] = "succeeded"
        except ModuleNotFoundError:
            report["tensorflow_import_attempt"] = "ModuleNotFoundError"

        import psutil
        process = psutil.Process()
        def rss():
            return int(process.memory_info().rss)
        report["rss"] = {"interpreter": rss()}
        import numpy as np
        import non_nt_runtime as rt
        import model_source
        rt.import_keras()
        report["rss"]["runtime_import"] = rss()
        report["versions"] = rt.runtime_versions()
        contract = json.loads((payload / "recovery/contract.json").read_text(encoding="utf-8"))
        with np.load(payload / "recovery/inputs.npz", allow_pickle=False) as frozen:
            frozen_inputs = dict(frozen)
        raw, held = {}, {}
        for key in ("spcas9", "cas12a"):
            spec = contract["models"][key + "_non_nt"]
            window = contract["cases"][key][0]["window"]
            runtime = rt.NonNTRuntime.load(key)
            held[key] = runtime
            report["held_models"] = list(held)
            report["rss"][key + "_model_loaded"] = rss()
            case = contract["cases"][key][0]["id"]
            inputs = [frozen_inputs[f"{key}/{case}/input/{index}"].copy()
                      for index in range(len(spec["inputs"]))]
            value = runtime.forward(inputs, head=spec["verified_head"]["index"])
            output, gradients = runtime.gradients(inputs, head=spec["verified_head"]["index"])
            raw[key + "/output"] = value
            raw[key + "/gradient_output"] = output
            for index, grad in enumerate(gradients):
                raw[f"{key}/gradient/{index}"] = grad
            checkpoint = Path(model_source.resolve(spec["path"]))
            with checkpoint.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            report["models"][key] = {
                "checkpoint": str(checkpoint), "sha256": checksum,
                "head": spec["verified_head"], "window": window, "n_outputs": runtime.n_outputs,
                "raw_output": float(value[0]), "gradient_output": float(output[0]),
                "gradient_abs_sum": [float(np.abs(g).sum()) for g in gradients],
                "all_finite": bool(np.isfinite(value).all() and np.isfinite(output).all()
                                   and all(np.isfinite(g).all() for g in gradients)),
                "held_models": list(held), "ops": runtime.ops_present()}
            report["rss"][key + "_after_gradients"] = rss()
        np.savez_compressed(root / "offline-raw.npz", **raw)
        report["tensorflow_in_sys_modules_after_load"] = any(
            name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules)
        report["module_origins"] = {name: str(Path(module.__file__).resolve())
            for name, module in list(sys.modules.items()) if getattr(module, "__file__", None)
            and not str(module.__file__).startswith("<")}
        report["unexpected_module_origins"] = {name: path for name, path in report["module_origins"].items()
            if not Path(path).is_relative_to(root) and (not Path(path).is_relative_to(Path(sys.base_prefix))
                or "site-packages" in path.lower())}
        report["rss"]["peak_working_set"] = int(getattr(process.memory_info(), "peak_wset", rss()))
        report["rss"]["scope"] = "minimum forward/gradient, both non-NT models retained; excludes full IG/SHAP/UI peak"
        report["status"] = "pass" if (
            report["network_blocked"] and not report["tensorflow_importable"]
            and report["tensorflow_import_attempt"] == "ModuleNotFoundError"
            and not report["tensorflow_in_sys_modules_after_load"] and not report["unexpected_module_origins"]
            and not violations and all(p["blocked"] for p in report["path_guard"]["probes"])
            and all(m["all_finite"] and all(s > 0 for s in m["gradient_abs_sum"])
                    for m in report["models"].values())) else "fail"
    except Exception as error:
        import traceback
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report["path_guard"]["violations"] = violations
    (root / "runner-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
