import json
import shutil
import subprocess
import zipfile

import pytest

from xai_test.nt_xai_demo import __main__ as entry
from xai_test.nt_xai_demo import prepare as demo
from xai_test.nt_xai_demo.test.fixtures import (
    IDS,
    PREFIX,
    SHARD_LAYOUT,
    build_sources,
    sha,
)

LAYOUT_PIN = f"IG shard {len(IDS)}개의 dna_spans·layout 일치"


def manifest(data_dir):
    return json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))


def test_prepare_extracts_light_bundle_and_writes_one_verifiable_manifest(tmp_path):
    sources = build_sources(tmp_path / "src")
    data = tmp_path / "data"

    report = demo.prepare(sources, data)

    files = manifest(data)["files"]
    assert "cas12a/ism.npz" in files and "cas12a/metrics.json" in files
    assert "spcas9/published_split/train.csv" in files
    assert "spcas9/v3_testset_predictions/predictions.csv" in files
    assert not any(
        "shards/" in name
        or name.endswith(".sidecar.json")
        and name.startswith("cas12a/")
        for name in files
    )
    assert "cas12a/rollout_diagnostics.json" not in files
    on_disk = {p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()}
    assert on_disk == set(files) | {"manifest.json"}
    assert demo.verify(data) == []
    assert report.spcas9_missing == []


def test_agreement_is_sliced_by_key_selection_only_and_source_hash_recorded(tmp_path):
    sources = build_sources(tmp_path / "src")
    data = tmp_path / "data"
    demo.prepare(sources, data)

    sliced = json.loads(
        (data / "cas12a/agreement.sliced.json").read_text(encoding="utf-8")
    )
    with zipfile.ZipFile(sources.cas12a_zip) as archive:
        raw = archive.read(PREFIX + "agreement.json")
    original = json.loads(raw)
    assert set(sliced) == {"populations", "group_diagnostics", "protocol"}
    for key in sliced:
        assert sliced[key] == original[key]
    derived = manifest(data)["derived"]["cas12a/agreement.sliced.json"]
    assert derived["source_sha256"] == sha(raw)
    assert set(derived["dropped"]) >= {"rows", "statistics_ids"}


def test_layout_shared_by_every_ig_shard_is_lifted_into_one_pinned_file(tmp_path):
    sources = build_sources(tmp_path / "src")
    data = tmp_path / "data"

    demo.prepare(sources, data)

    layout = json.loads((data / "cas12a/layout.json").read_text(encoding="utf-8"))
    assert layout == SHARD_LAYOUT  # the recorded keys only, values untouched
    payload = manifest(data)
    assert LAYOUT_PIN in payload["nucleases"]["cas12a"]["pins_checked"]
    entry = payload["files"]["cas12a/layout.json"]
    assert entry["git"] == "tracked"
    # It is lifted out of the shards, not copied from a report member.
    assert "shards/ig/" in entry["source"] and PREFIX not in entry["source"]
    derived = payload["derived"]["cas12a/layout.json"]
    assert derived["kept"] == ["coordinates", "dna_spans", "layout"]
    assert derived["shards"] == len(IDS)
    assert demo.verify(data) == []


def test_one_ig_shard_with_another_layout_fails_the_pin_and_writes_nothing(tmp_path):
    sources = build_sources(tmp_path / "src", corrupt="ig_layout")
    data = tmp_path / "data"

    with pytest.raises(demo.PinMismatch, match="dna_spans·layout") as refused:
        demo.prepare(sources, data)

    assert [pin.actual for pin in refused.value.failed] == [2]  # two layouts seen
    assert not data.exists()


def test_a_missing_ig_shard_fails_the_shard_count_pin_and_writes_nothing(tmp_path):
    """Agreement among the shards present is not enough: every plan row has one."""
    sources = build_sources(tmp_path / "src", corrupt="ig_missing_shard")
    data = tmp_path / "data"

    with pytest.raises(demo.PinMismatch, match="IG shard 수") as refused:
        demo.prepare(sources, data)

    assert [(pin.expected, pin.actual) for pin in refused.value.failed] == [
        (len(IDS), len(IDS) - 1)
    ]
    assert not data.exists()


def test_verify_prints_the_pins_prepare_checked_including_the_layout_pin(
    tmp_path, monkeypatch, capsys
):
    data = tmp_path / "data"
    demo.prepare(build_sources(tmp_path / "src"), data)
    monkeypatch.setattr(demo, "DEFAULT_DATA_DIR", data)

    assert entry.main(["verify"]) == 0

    out = capsys.readouterr().out
    assert "온전성 확인 통과" in out
    assert LAYOUT_PIN in out
    assert "plan.json sha256 (plan.sha256)" in out


@pytest.mark.parametrize(
    "corrupt",
    ["archive", "plan", "ism_npz", "population", "ig_layout", "ig_missing_shard"],
)
def test_pin_mismatch_writes_nothing(tmp_path, corrupt):
    sources = build_sources(tmp_path / "src", corrupt=corrupt)
    data = tmp_path / "data"

    with pytest.raises(demo.PinMismatch):
        demo.prepare(sources, data)

    assert not data.exists()


def test_pin_mismatch_leaves_previous_data_untouched(tmp_path):
    data = tmp_path / "data"
    demo.prepare(build_sources(tmp_path / "good"), data)
    before = (data / "manifest.json").read_bytes()

    with pytest.raises(demo.PinMismatch):
        demo.prepare(build_sources(tmp_path / "bad", corrupt="ism_npz"), data)

    assert (data / "manifest.json").read_bytes() == before
    assert demo.verify(data) == []


def test_missing_spcas9_bundle_prepares_cas12a_only_and_reports_absence(tmp_path):
    sources = build_sources(tmp_path / "src", spcas9=False)
    data = tmp_path / "data"

    report = demo.prepare(sources, data)

    payload = manifest(data)
    assert payload["nucleases"]["spcas9"]["status"] == "absent"
    assert "v3_ism/ism.npz" in " ".join(payload["nucleases"]["spcas9"]["missing"])
    assert not any(name.startswith("spcas9/") for name in payload["files"])
    assert "carried_over" not in payload["nucleases"]["spcas9"]  # nothing to carry
    assert payload["nucleases"]["cas12a"]["status"] == "prepared"
    assert report.spcas9_missing
    text = report.render()
    assert "SpCas9" in text and "부재" in text


def _spcas9_entries(payload):
    return {k: v for k, v in payload["files"].items() if k.startswith("spcas9/")}


def _as_fresh_clone(data):
    """What a clone has: every Git-tracked file, no bulk (`.npz`, bulk csv)."""
    for name in manifest(data)["files"]:
        if demo.is_bulk(name):
            (data / name).unlink()


def test_fresh_clone_without_spcas9_bundle_prepares_cas12a_and_keeps_spcas9_light_files(
    tmp_path,
):
    data = tmp_path / "data"
    demo.prepare(build_sources(tmp_path / "full"), data)
    _as_fresh_clone(data)
    before = manifest(data)
    assert before["nucleases"]["spcas9"]["status"] == "prepared"
    light = {
        name: entry
        for name, entry in _spcas9_entries(before).items()
        if not demo.is_bulk(name)
    }
    light_bytes = {name: (data / name).read_bytes() for name in light}

    report = demo.prepare(build_sources(tmp_path / "partial", spcas9=False), data)

    after = manifest(data)
    # Cas12a was written anew, bulk included.
    assert (data / "cas12a/ism.npz").is_file()
    assert after["nucleases"]["cas12a"]["status"] == "prepared"
    # The SpCas9 light files survive byte for byte, their entries carried over
    # from the previous manifest (not re-hashed), and nothing else is listed.
    assert {n: (data / n).read_bytes() for n in light} == light_bytes
    assert _spcas9_entries(after) == light
    spcas9 = after["nucleases"]["spcas9"]
    assert spcas9["status"] == "absent"
    assert "v3_ism/ism.npz" in " ".join(spcas9["missing"])
    carried = spcas9["carried_over"]
    assert carried["files"] == sorted(light)
    assert carried["previous_status"] == "prepared"
    assert "manifest" in carried["from"]
    assert demo.verify(data) == []
    text = report.render()
    assert "부재" in text and "유지" in text
    assert report.spcas9_carried == sorted(light)


def test_carried_over_hashes_come_from_the_previous_manifest_not_a_new_hash(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    demo.prepare(build_sources(tmp_path / "full"), data)
    _as_fresh_clone(data)
    previous = _spcas9_entries(manifest(data))
    name = "spcas9/v3_ism/ism.sidecar.json"
    sentinel = dict(previous[name], source="이전 manifest의 원래 source")
    payload = manifest(data)
    payload["files"][name] = sentinel
    (data / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    demo.prepare(build_sources(tmp_path / "partial", spcas9=False), data)

    assert manifest(data)["files"][name] == sentinel


def test_carried_over_file_that_disagrees_with_previous_manifest_is_refused(tmp_path):
    data = tmp_path / "data"
    demo.prepare(build_sources(tmp_path / "full"), data)
    _as_fresh_clone(data)
    (data / "spcas9/v3_ism/ism.sidecar.json").write_text("{}", encoding="utf-8")
    before = (data / "manifest.json").read_bytes()

    with pytest.raises(demo.PinMismatch, match="ism.sidecar.json"):
        demo.prepare(build_sources(tmp_path / "partial", spcas9=False), data)

    assert (data / "manifest.json").read_bytes() == before


def test_absent_spcas9_bundle_preserves_previously_prepared_spcas9_bulk(tmp_path):
    data = tmp_path / "data"
    demo.prepare(build_sources(tmp_path / "full"), data)
    previous = _spcas9_entries(manifest(data))
    bulk = (data / "spcas9/v3_ism/ism.npz").read_bytes()

    demo.prepare(build_sources(tmp_path / "partial", spcas9=False), data)

    assert _spcas9_entries(manifest(data)) == previous
    assert (data / "spcas9/v3_ism/ism.npz").read_bytes() == bulk
    assert demo.verify(data) == []


def test_unlisted_spcas9_bulk_without_its_bundle_is_never_deleted(tmp_path):
    data = tmp_path / "data"
    demo.prepare(build_sources(tmp_path / "full"), data)
    payload = manifest(data)
    del payload["files"]["spcas9/v3_ism/ism.npz"]
    (data / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    before = (data / "manifest.json").read_bytes()

    with pytest.raises(demo.SpCas9WouldBeLost, match="v3_ism/ism.npz"):
        demo.prepare(build_sources(tmp_path / "partial", spcas9=False), data)

    assert (data / "manifest.json").read_bytes() == before
    assert (data / "spcas9/v3_ism/ism.npz").exists()


def test_missing_cas12a_source_is_reported_not_a_traceback(tmp_path):
    sources = build_sources(tmp_path / "src")
    sources.cas12a_audit.unlink()

    with pytest.raises(demo.MissingSource, match="audit.json"):
        demo.prepare(sources, tmp_path / "data")

    assert not (tmp_path / "data").exists()


def test_source_changed_after_pin_check_is_not_certified(tmp_path, monkeypatch):
    sources = build_sources(tmp_path / "src")
    data = tmp_path / "data"
    real_copy = demo.shutil.copyfile

    def tampering_copy(source, target):
        real_copy(source, target)
        if str(target).endswith("ism.npz"):
            with open(target, "ab") as handle:
                handle.write(b"x")

    monkeypatch.setattr(demo.shutil, "copyfile", tampering_copy)

    with pytest.raises(demo.PinMismatch, match="복사본"):
        demo.prepare(sources, data)

    assert not data.exists()
    assert not list(tmp_path.glob(".prepare-*"))


def test_verify_reports_modified_missing_and_unlisted_files(tmp_path):
    data = tmp_path / "data"
    demo.prepare(build_sources(tmp_path / "src"), data)
    (data / "cas12a/metrics.json").write_text("{}", encoding="utf-8")
    (data / "cas12a/ism.npz").unlink()
    (data / "stray.txt").write_text("x", encoding="utf-8")

    problems = demo.verify(data)

    assert any("cas12a/metrics.json" in p for p in problems)
    assert any("cas12a/ism.npz" in p for p in problems)
    assert any("stray.txt" in p for p in problems)


def test_manifest_git_column_matches_repository_gitignore():
    if shutil.which("git") is None:
        pytest.skip("git 없음")
    names = [
        "cas12a/ism.npz",
        "cas12a/predictions.csv",
        "cas12a/cases.csv",
        "cas12a/metrics.json",
        "spcas9/published_split/train.csv",
        "spcas9/v3_ism/ism.npz",
        "spcas9/v3_ism/ism.sidecar.json",
    ]
    for name in names:
        path = (demo.DEFAULT_DATA_DIR / name).relative_to(demo.REPO_ROOT).as_posix()
        ignored = (
            subprocess.run(
                ["git", "check-ignore", "-q", "--no-index", path],
                cwd=demo.REPO_ROOT,
                check=False,
            ).returncode
            == 0
        )
        assert ignored == demo.is_bulk(name), name


def test_real_data_matches_its_manifest():
    """File-change detection on the real prepared data; skipped where it was never prepared."""
    data = demo.DEFAULT_DATA_DIR
    if not (data / "manifest.json").exists():
        pytest.skip("실제 데모 자료 없음: `prepare`를 먼저 실행")
    files = manifest(data)["files"]
    bulk = [name for name in files if demo.is_bulk(name)]
    if bulk and not any((data / name).exists() for name in bulk):
        pytest.skip(
            "git 밖 bulk 자료(.npz/csv) 없음: 복원 조건은 Cas12a ZIP + SpCas9 원본 묶음"
        )
    assert demo.verify(data) == []
