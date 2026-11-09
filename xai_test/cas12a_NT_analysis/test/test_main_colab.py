import hashlib
import json
import zipfile

from xai_test.cas12a_NT_analysis.nt_main_colab import build_main_bundle


def test_main_notebook_has_exact_short_source_bundle_and_one_test_entry(tmp_path):
    result = build_main_bundle(tmp_path / "bundle")
    notebook_name = next(n for n in result["files"] if n.endswith(".ipynb"))
    zip_name = next(n for n in result["files"] if n.endswith(".zip"))
    assert "Test1500_main" in notebook_name
    assert "test1500-main-source" in zip_name
    book = json.loads((tmp_path / "bundle" / notebook_name).read_bytes())
    sources = [c["source"] for c in book["cells"] if c["cell_type"] == "code"]
    for source in sources:
        compile(source, "main-cell", "exec")
    joined = "\n".join(sources)
    assert zip_name in joined
    assert result["files"][zip_name] in joined
    assert "verify_test_inputs(" in joined
    assert "run_main(" in joined
    assert "report_main(" in joined
    assert "Study.create" not in joined
    assert "APPROVED_PROTOCOL_SHA256" not in joined
    assert "run_phase(" not in joined
    assert "N_TARGETS" not in joined
    assert "RESERVED_SECONDS" not in joined
    assert "open_study" not in joined
    assert "STUDY_ROOT" not in joined
    assert "EVIDENCE_ROOT" not in joined
    assert "/content/c12main-" in joined
    with zipfile.ZipFile(tmp_path / "bundle" / zip_name) as archive:
        manifest = json.loads(archive.read("source-manifest.json"))
        assert set(archive.namelist()) == {"source-manifest.json", *manifest["files"]}
        for name, checksum in manifest["files"].items():
            assert len(name) < 150
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum
        assert (
            "xai_test/cas12a_NT_analysis/docs/colab-main-analysis.md"
            in manifest["files"]
        )
    assert build_main_bundle(tmp_path / "same") == result


def test_packaged_analysis_cell_runs_without_legacy_study_or_reservation(
    inputs, tmp_path
):
    from xai_test.cas12a_NT_analysis.nt_inputs import verify_test_inputs
    from xai_test.cas12a_NT_analysis.nt_main import prepare_main, run_main
    from xai_test.cas12a_NT_analysis.nt_main_artifacts import read_main

    root = tmp_path / "results"
    verified = verify_test_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    prepare_main(verified, root)
    bundle = tmp_path / "bundle"
    build_main_bundle(bundle)
    book = json.loads(next(bundle.glob("*.ipynb")).read_bytes())
    cell = next(c["source"] for c in book["cells"] if c["source"].startswith("# 4."))
    # The delivery cell itself runs on the real tiny NT; only the external
    # runtime selection changes from Colab CUDA to this CPU fixture.
    namespace = {"inputs": verified, "OUTPUT": root, "run_main": run_main, "json": json}
    exec(  # noqa: S102 - execute our locally generated notebook cell in its test runtime
        compile(
            cell.replace('device="cuda:0"', 'device="cpu"'),
            "delivered-analysis-cell",
            "exec",
        ),
        namespace,
    )
    assert namespace["summary"]["complete"]
    assert read_main(root)["coverage"]["ism"]["available"] == [1, 8]
