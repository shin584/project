"""Independent recovery evidence required by the existing report boundary."""

from pathlib import Path


def restored_name(name):
    for prefix in ("code/", "assets/"):
        if name.startswith(prefix):
            return "payload/" + name[len(prefix):]
    return "payload/recovery/" + name


def recovery_failures(document, manifest, manifest_sha256, contract):
    """Validate recorded provenance without consulting the original machine."""
    recovery = document.get("recovery", {})
    errors = []
    if recovery.get("bundle_manifest_sha256") != manifest_sha256:
        errors.append("missing/mixed recovery preparation")
    files = recovery.get("files", {})
    for name, source in manifest["sources"].items():
        if files.get(restored_name(name)) != source["sha256"]:
            errors.append(f"missing/mismatched restored checksum: {name}")
    if files.get("payload/recovery/inputs.npz") != manifest["inputs_sha256"]:
        errors.append("missing/mismatched canonical inputs")
    if not recovery.get("wheels") or recovery.get("verified_files") != files:
        errors.append("unverified restored inputs")
    wheels = recovery.get("wheels", {})
    if not manifest.get("offline_dependencies") or set(wheels) != set(manifest["offline_dependencies"]):
        errors.append("declared wheel inventory mismatch")
    for name, version in manifest.get("offline_dependencies", {}).items():
        wheel = wheels.get(name, {})
        if (wheel.get("version", "").split("+")[0] != version
                or wheel.get("file") not in files or not wheel.get("file", "").startswith("wheels/")):
            errors.append(f"declared wheel version/checksum missing: {name}")
    if not recovery.get("installed_files"):
        errors.append("missing restored dependency checksums")
    if not recovery.get("imported_dependency_files") or not recovery.get("dependency_checksum_scope"):
        errors.append("missing imported dependency checksums/scope")
    if recovery.get("install", {}).get("exit_code") != 0:
        errors.append("dependency installation failed")
    if not recovery.get("install", {}).get("command"):
        errors.append("missing installation command")
    root_text = recovery.get("restored_root", "")
    root = Path(root_text)
    guard = document.get("run", {}).get("path_guard", {})
    originals = guard.get("original_roots", [])
    probes = guard.get("probes", [])
    if (not root.is_absolute() or not originals or not guard.get("scope") or guard.get("violations") != []
            or not probes or any(p.get("blocked") is not True for p in probes)
            or any(root.is_relative_to(Path(p)) or Path(p).is_relative_to(root) for p in originals)):
        errors.append("original path isolation missing/violated")
    sources = {item["source"] for item in manifest["sources"].values()}
    if not sources.issubset({p.get("path") for p in probes}):
        errors.append("original source probes missing")
    run = document.get("run", {})
    origins = run.get("module_origins", {})
    for module in ("non_nt_runtime", "model_source", "predict", "numpy", "keras", "torch", "h5py", "psutil"):
        if not origins.get(module) or not Path(origins[module]).is_relative_to(root):
            errors.append(f"module outside restored tree: {module}")
    if run.get("unexpected_module_origins") != {}:
        errors.append("unexpected module origins")
    if run.get("held_models") != ["spcas9", "cas12a"]:
        errors.append("missing held model inventory")
    for key in ("spcas9", "cas12a"):
        item = run.get("models", {}).get(key, {})
        checkpoint = item.get("checkpoint", "")
        spec = contract["models"][key + "_non_nt"]
        if not checkpoint or Path(checkpoint) != root / "payload" / spec["path"]:
            errors.append(f"model outside restored tree: {key}")
        if item.get("sha256") != spec["sha256"]:
            errors.append(f"model checksum mismatch: {key}")
        if item.get("head") != spec["verified_head"] or item.get("window") != contract["cases"][key][0]["window"]:
            errors.append(f"model case/head mismatch: {key}")
    rss = run.get("rss", {})
    if any(not isinstance(rss.get(stage), (int, float)) or rss[stage] <= 0 for stage in (
        "interpreter", "runtime_import", "spcas9_model_loaded", "spcas9_after_gradients",
        "cas12a_model_loaded", "cas12a_after_gradients", "peak_working_set")):
        errors.append("minimum-run RSS missing/invalid")
    network = run.get("network_probes", [])
    if (len(network) != 4 or {p.get("kind") for p in network} != {"tcp", "dns", "reverse_dns", "udp"}
            or any(p.get("blocked") is not True for p in network)):
        errors.append("network block probes missing/failed")
    if not run.get("network_scope"):
        errors.append("network scope missing")
    if run.get("tensorflow_import_attempt") != "ModuleNotFoundError":
        errors.append("TensorFlow import exclusion unverified")
    if not run.get("external_prerequisites") or not document.get("command") or document.get("exit_code") != 0:
        errors.append("missing/failed execution metadata")
    return errors
