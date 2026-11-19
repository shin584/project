"""non-NT SpCas9 예측기의 수치·식별 계약 (#196).

**실제 채택 checkpoint**(396KB)를 쓴다. CPU로 충분히 빠르다.

여기서 재는 것
  · 30bp 윈도우 인코딩과 9개 출력의 변종/헤드 순서가 원 모델과 같은지.
  · 원출력·효율 변환·표시 점수가 고정 사례에서 원 모델과 같은지.
  · 원출력 **단위의 출처**가 보관된 산출물로 확인되는지. 확인된 단위만 효율로 표시한다.
  · 사이트가 없으면 checkpoint를 읽지 않는지, NT 가중치·토크나이저를 건드리지 않는지.

원 모델의 기준값은 보관된 산출물에서 읽는다 — 이 테스트가 다시 계산한 값이 아니다.
  · `export-manifest.json`의 `validation.batches` — 학습 원본(.keras, TensorFlow
    backend)과 export가 같은 값을 냈다는 #98의 기록.
  · `test_eval/predictions.csv` — 같은 모델로 뜬 테스트셋 예측. `*_true`와 같은
    척도("published percentage / 100")에서 직접 비교된 값이다.

허용 오차는 `docs/runtime/contract.json`의 동결된 `criteria.forward_parity`
(abs 1e-4 / rel 1e-3)를 쓴다. 이 티켓이 새로 느슨하게 정한 값이 아니다.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import model_source
import non_nt_predict as nnp
import predict
from pam_scanner import SPCAS9_VARIANTS, CandidateSite

ROOT = Path(__file__).resolve().parent.parent
ASSET = ROOT / "models/non_nt/spcas9-9hydra-v2"
SPEC = predict.NON_NT_REGISTRY[("A", "SpCas9")]

#: 보관된 예측 CSV의 열 접두사. export manifest의 헤드 순서와 같은 순서여야 한다.
PREDICTION_COLUMNS = ("SpCas9", "SpCas9-NG", "VRQR variant", "xCas", "Sniper-Cas9",
                      "SpCas9-HF.1", "eSpCas9(1.1)", "HypaCas9", "evoCas9")


@pytest.fixture(scope="module")
def tolerance():
    contract = json.loads((ROOT / "docs/runtime/contract.json").read_text(encoding="utf-8"))
    parity = contract["criteria"]["forward_parity"]
    return {"abs": parity["max_abs_diff"], "rel": parity["max_rel_diff"]}


@pytest.fixture(scope="module")
def manifest():
    return json.loads((ASSET / "export-manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def predictor():
    return nnp.NonNTSpCas9Predictor(SPEC)


def site_for(window: str) -> CandidateSite:
    """고정 윈도우 하나를 그 윈도우를 그대로 쓰는 사이트로 감싼다.

    PAM은 윈도우 안의 자리(24-26)에서 읽는다 — `pam_scanner.WINDOW_LAYOUT`의
    SpCas9 배치다. 그래서 어느 변종이 이 사이트를 인식하는지도 실제 규칙으로 갈린다.
    """
    return CandidateSite(
        cas_type="SpCas9", strand="+", pam_start=24, cut_pos=40, distance=0,
        guide_seq=window[4:24], pam_seq=window[24:27], model_input_seq=window)


# ── 구조와 식별 ───────────────────────────────────────────────

def test_head_layer_order_matches_the_export_manifest(manifest):
    """코드에 적어 둔 헤드 순서가 보관된 manifest와 같은지 (출처 대조)."""
    heads = sorted(manifest["heads"], key=lambda h: h["index"])
    assert [h["layer"] for h in heads] == list(nnp.SPCAS9_HEAD_LAYERS)
    assert [h["protein"] for h in heads] == SPCAS9_VARIANTS


def test_window_length_and_channels_match_the_export_manifest(manifest):
    assert manifest["input"]["shape"][1:] == [nnp.SPCAS9_WINDOW_LENGTH, 4]
    assert manifest["input"]["channels"] == ["A", "C", "G", "T"]
    config = next(c for c in predict.CAS_CONFIGS if c.name == "SpCas9")
    assert config.model_input_len == nnp.SPCAS9_WINDOW_LENGTH


def test_loaded_graph_matches_the_assumed_shape_and_head_order(predictor):
    runtime = predictor.runtime
    assert runtime.input_shapes == [(nnp.SPCAS9_WINDOW_LENGTH, 4)]
    assert tuple(runtime.head_names) == nnp.SPCAS9_HEAD_LAYERS


def test_identity_names_the_pinned_checkpoint_without_loading_it():
    """예측 캐시 키를 만들 때 모델을 올리지 않는다 (#191의 지연 로딩)."""
    fresh = nnp.NonNTSpCas9Predictor(SPEC)
    identity = fresh.model_identity()
    assert fresh._runtime is None
    pinned = model_source.NON_NT_FILES[SPEC.paths[0]]["sha256"]
    assert identity.checkpoint_sha256 == pinned
    assert identity.family == "non_nt" and identity.cas == "SpCas9"
    assert identity.name == "9-Hydra v2"
    assert identity.version == f"spcas9-9hydra-v2 · sha256 {pinned[:12]}"


def test_loading_rejects_a_checkpoint_that_is_not_the_pinned_one(monkeypatch):
    """고정 해시와 어긋난 파일로는 추론하지 않는다 (#192의 보관 계약)."""
    monkeypatch.setattr(
        model_source, "resolve",
        lambda rel: (_ for _ in ()).throw(ValueError(f"Pinned model checksum mismatch: {rel}")))
    with pytest.raises(ValueError, match="checksum mismatch"):
        nnp.NonNTSpCas9Predictor(SPEC).runtime


# ── 원 모델과의 고정 사례 비교 ─────────────────────────────────

def test_raw_scores_match_the_reference_export_values(predictor, manifest, tolerance):
    """export manifest가 기록한 원 모델 값과 예측 행의 원출력이 같은지."""
    windows = manifest["fixtures"]["windows"]
    batch = next(b for b in manifest["validation"]["batches"]
                 if b["fixture"] == "distinct_batch")
    assert batch["n"] == len(windows)
    reference = {head["protein"]: [float(v) for v in head["source_values"]]
                 for head in batch["heads"]}

    sites = [site_for(window) for window in windows]
    rows = predictor.predict_batch(sites)
    # 고정 사례 세 개는 같은 자리(24)를 PAM으로 쓰므로 site_key가 겹친다. 어느 행이
    # 어느 윈도우의 것인지는 행이 들고 있는 입력 윈도우로 가린다.
    by_window = {(row.window_seq, row.cas_name): row for row in rows}

    checked = 0
    for index, site in enumerate(sites):
        for variant in predictor.explain_heads(site):
            row = by_window[(site.model_input_seq, variant)]
            assert row.raw_score == pytest.approx(reference[variant][index], **tolerance)
            checked += 1
    assert checked >= len(windows), "비교한 고정 사례가 없다"


def test_all_nine_heads_are_compared_across_the_fixture_windows(
        predictor, manifest, tolerance):
    """헤드 9개를 하나도 빠뜨리지 않고 원 모델 값과 비교한다.

    한 윈도우로는 9개를 다 못 덮는다 — VRQR는 NGA를, 나머지 여섯은 NGG를 인식하므로
    한 PAM이 둘을 동시에 만족할 수 없다. 고정 윈도우들의 합집합으로 덮는다.
    """
    windows = manifest["fixtures"]["windows"]
    batch = next(b for b in manifest["validation"]["batches"]
                 if b["fixture"] == "distinct_batch")

    compared = {}
    for index, window in enumerate(windows):
        rows = predictor.predict_batch([site_for(window)])
        for head in batch["heads"]:
            row = next((r for r in rows if r.cas_name == head["protein"]), None)
            if row is None:        # 이 변종이 인식하지 않는 PAM — 예측 행 자체가 없다
                continue
            assert row.raw_score == pytest.approx(
                float(head["source_values"][index]), **tolerance)
            compared.setdefault(head["protein"], 0)
            compared[head["protein"]] += 1
    assert sorted(compared) == sorted(SPCAS9_VARIANTS)


def test_raw_unit_provenance_is_the_published_efficiency_fraction(predictor, tolerance):
    """보관된 테스트셋 예측을 다시 내 단위의 출처를 확인한다.

    `test_eval/results.json`이 그 파일의 척도를 "published percentage / 100"이라고
    적고, 같은 척도의 `*_true`와 직접 비교했다. 그 예측을 이 런타임이 재현하면
    원출력의 단위도 그 척도다 — 그래서 효율(%)로 표시할 수 있다.
    """
    results = json.loads((ASSET / "test_eval/results.json").read_text(encoding="utf-8"))
    assert results["scale"] == "published percentage / 100 (the scale 9-Hydra v2 was trained on)"
    assert SPEC.output_mode == "raw"

    with (ASSET / "test_eval/predictions.csv").open(encoding="utf-8", newline="") as stream:
        # held-out test 행만 쓴다. 그 행들이 results.json의 지표가 나온 행이고,
        # 그 파일이 척도를 선언한 대상이다.
        held_out = (r for r in csv.DictReader(stream) if r["split"] == "test")
        sample = [record for record, _ in zip(held_out, range(8))]
    assert len(sample) == 8
    windows = [record["sequence"] for record in sample]
    assert all(len(window) == nnp.SPCAS9_WINDOW_LENGTH for window in windows)

    from non_nt_runtime import one_hot
    raw = predictor.runtime.forward_all([np.stack([one_hot(w) for w in windows])])
    expected = np.array([[float(record[f"{column}_pred"]) for column in PREDICTION_COLUMNS]
                         for record in sample])
    assert raw == pytest.approx(expected, **tolerance)


def test_display_score_is_the_clipped_fraction_as_efficiency(predictor, manifest):
    """표시 점수는 0~1로 자른 원출력 ×100이고, 그 의미는 '절단 효율'이다."""
    rows = predictor.predict_batch([site_for(manifest["fixtures"]["windows"][0])])
    assert rows
    for row in rows:
        assert row.score_kind == "efficiency"
        assert row.efficiency_pct == pytest.approx(min(max(row.raw_score, 0.0), 1.0) * 100)
        assert row.identity is not None and row.identity.family == "non_nt"
        assert row.identity.raw_unit.startswith("공개 효율(%)")


def test_negative_raw_outputs_are_clipped_not_dropped():
    """linear 출력이라 음수가 나올 수 있다. 0%로 눌러 보여주고 원출력은 보존한다.

    `test_predict.py`가 같은 변환 함수를 NT spec으로 재는 것과 같은 자리를, 이
    계열의 spec으로 짚는다.
    """
    assert predict._to_display_score(SPEC, -0.0123) == (0.0, "efficiency")


# ── 지연 로딩 ─────────────────────────────────────────────────

def test_no_sites_loads_no_checkpoint():
    fresh = nnp.NonNTSpCas9Predictor(SPEC)
    assert fresh.predict_batch([]) == []
    assert fresh._runtime is None


def test_progress_counts_sequences_once_per_batch(predictor, manifest):
    """진행률 단위가 NT 예측기와 같다 — 헤드 9개가 9배로 세지 않는다."""
    windows = manifest["fixtures"]["windows"]
    seen = []
    predictor.predict_batch([site_for(w) for w in windows], on_batch=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1] == (len(windows), len(windows))
    assert all(total == len(windows) for _, total in seen)


def test_progress_callback_exception_cancels_the_prediction(predictor, manifest):
    class Stop(Exception):
        pass

    def refuse(done, total):
        raise Stop

    with pytest.raises(Stop):
        predictor.predict_batch([site_for(manifest["fixtures"]["windows"][0])], on_batch=refuse)


# ── 설명은 아직 없다 ──────────────────────────────────────────

def test_explain_heads_match_the_prediction_rows(predictor, manifest):
    """XAI의 헤드 선택이 이 계열의 예측 행과 같은 집합을 가리킨다."""
    for window in manifest["fixtures"]["windows"]:
        site = site_for(window)
        rows = predictor.predict_batch([site])
        assert sorted(predictor.explain_heads(site)) == sorted(r.cas_name for r in rows)


def test_explanation_is_reported_as_not_implemented(predictor):
    """구현되지 않은 기법을 성공 결과로 내놓지 않는다 (#199가 붙인다)."""
    with pytest.raises(NotImplementedError, match="#199"):
        predictor.explain_handle("SpCas9-NG")
