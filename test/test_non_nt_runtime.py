"""non_nt_runtime: 채택 런타임(Keras 3 + torch backend)의 전방·gradient 계약 (#193).

실제 채택 checkpoint 두 개를 쓴다 (396KB/1.2MB, CPU로 충분히 빠르다).
#193의 수치 기준은 `docs/runtime/contract.json`에 동결돼 있고, 여기서는 그 계약이
의존하는 **코드 수준 성질**을 검증한다 — 런타임 선택, 추론 모드 결정성, head 선택,
배치 독립성, checksum 고정, 그리고 기존 xai/explain.py 루프와의 접합.

backend 차이 parity와 허용 오차 판정은 별도 프로세스가 필요하므로
`docs/runtime/verify_contract.py`가 담당한다.
"""

import non_nt_runtime as rt
import numpy as np
import pytest

SPCAS9_WINDOW = "CCGGTCGCTTTAAGAATAAATAAAAGCGCC"
CAS12A_WINDOW = "TTTAAGAATAAATAAAAGCGCCACAGGATTGACT"


#: 테스트도 런타임과 같은 원핫 규약을 쓴다 (사본을 늘리지 않는다).
_onehot = rt.one_hot


@pytest.fixture(scope="module")
def spcas9():
    return rt.NonNTRuntime.load("spcas9")


@pytest.fixture(scope="module")
def cas12a():
    return rt.NonNTRuntime.load("cas12a")


# ── 런타임 선택 ────────────────────────────────────────────────

def test_backend_is_torch_regardless_of_environment(monkeypatch):
    """사용자 셸의 KERAS_BACKEND가 앱이 쓰는 backend를 바꾸지 못한다."""
    monkeypatch.setenv("KERAS_BACKEND", "tensorflow")
    keras = rt.import_keras()
    assert keras.backend.backend() == "torch"
    assert rt.BACKEND == "torch"


def test_contract_versions_match_the_running_runtime():
    """동결한 계약의 버전이 실제 실행 중인 버전과 다르면 즉시 드러난다."""
    recorded = rt.runtime_versions()
    assert recorded["keras"] == "3.14.1"
    assert recorded["torch"].startswith("2.12.0")
    assert recorded["python"] == "3.13.7"
    assert recorded["numpy"] == "2.3.2"
    assert recorded["shap"] == "0.52.0"
    assert recorded["h5py"] == "3.14.0"
    assert recorded["backend"] == "torch"


# ── 모델 적재와 모양 ───────────────────────────────────────────

def test_spcas9_shape_contract(spcas9):
    assert spcas9.input_shapes == [(30, 4)]
    assert spcas9.n_outputs == 9
    assert spcas9.head_names[1] == "Output_SpCas9_NG"


def test_cas12a_shape_contract(cas12a):
    assert cas12a.input_shapes == [(34, 4), (354,)]
    assert cas12a.n_outputs == 1


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError):
        rt.NonNTRuntime.load("sacas9")


def test_tampered_checkpoint_is_rejected(tmp_path, monkeypatch):
    """checksum이 어긋난 파일로는 절대 추론하지 않는다 (#192의 고정 보관 계약)."""
    import model_source

    def fake_resolve(rel_path):
        raise ValueError(f"Pinned model checksum mismatch: {rel_path}")

    monkeypatch.setattr(model_source, "resolve", fake_resolve)
    with pytest.raises(ValueError, match="checksum mismatch"):
        rt.NonNTRuntime.load("spcas9")


# ── 추론 모드 ─────────────────────────────────────────────────

def test_forward_is_deterministic_so_dropout_is_off(spcas9):
    x = _onehot(SPCAS9_WINDOW)[None]
    first = spcas9.forward([x], head=1)
    second = spcas9.forward([x], head=1)
    assert first.shape == (1,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_forward_selects_the_requested_head(spcas9):
    x = _onehot(SPCAS9_WINDOW)[None]
    heads = [float(spcas9.forward([x], head=i)[0]) for i in range(9)]
    assert len(set(heads)) > 1, "9개 head가 모두 같은 값이면 head 선택이 동작하지 않는다"
    assert float(spcas9.forward([x], head=1)[0]) == heads[1]


def test_forward_batches_within_float32_noise(spcas9):
    """배치 크기는 값을 바꾸지 않는다 — float32 마지막 자리까지는 아니다.

    torch의 배치 커널이 누적 순서를 바꾸므로 비트 동일성은 성립하지 않는다.
    실측 차이는 1e-7 수준이고 동결한 허용 오차(1e-4)보다 세 자리 작다.
    SHAP이 수천 서열을 큰 배치로 넣어도 판정에 영향을 주지 않는다는 뜻이다.
    """
    xs = np.stack([_onehot(SPCAS9_WINDOW), _onehot("A" * 30), _onehot("GC" * 15)])
    one = spcas9.forward([xs], head=1, batch_size=1)
    big = spcas9.forward([xs], head=1, batch_size=8)
    assert np.abs(one - big).max() <= 1e-6


def test_forward_reports_progress_in_sequence_counts(spcas9):
    xs = np.stack([_onehot(SPCAS9_WINDOW)] * 5)
    seen = []
    spcas9.forward([xs], head=1, batch_size=2, on_batch=lambda d, t: seen.append((d, t)))
    assert seen == [(2, 5), (4, 5), (5, 5)]


def test_forward_propagates_callback_cancellation(spcas9):
    xs = np.stack([_onehot(SPCAS9_WINDOW)] * 4)

    def cancel(done, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        spcas9.forward([xs], head=1, batch_size=1, on_batch=cancel)


def test_cas12a_forward_is_a_probability(cas12a):
    seq = _onehot(CAS12A_WINDOW)[None]
    feat = rt.cas12a_derive(CAS12A_WINDOW)[None]
    y = cas12a.forward([seq, feat], head=0)
    assert y.shape == (1,)
    assert 0.0 <= float(y[0]) <= 1.0


# ── gradient ─────────────────────────────────────────────────

def test_gradients_match_the_forward_value(spcas9):
    """IG 보간 경로의 함수가 예측 전방 함수와 같다 (contract: function_match)."""
    x = _onehot(SPCAS9_WINDOW)[None]
    y_grad, grads = spcas9.gradients([x], head=1)
    y_fwd = spcas9.forward([x], head=1)
    assert abs(float(y_grad[0]) - float(y_fwd[0])) <= 1e-6
    assert len(grads) == 1
    assert grads[0].shape == x.shape
    assert np.isfinite(grads[0]).all()
    assert np.abs(grads[0]).sum() > 0


def test_gradients_are_independent_per_batch_row(spcas9):
    """explain.py는 배치 합의 기울기를 쓴다. 행끼리 섞이면 IG가 조용히 틀린다."""
    a = _onehot(SPCAS9_WINDOW)
    b = _onehot("GC" * 15)
    together = spcas9.gradients([np.stack([a, b])], head=1)[1][0]
    alone_a = spcas9.gradients([a[None]], head=1)[1][0][0]
    alone_b = spcas9.gradients([b[None]], head=1)[1][0][0]
    assert np.allclose(together[0], alone_a, atol=1e-6)
    assert np.allclose(together[1], alone_b, atol=1e-6)


def test_gradients_cover_every_cas12a_input(cas12a):
    seq = _onehot(CAS12A_WINDOW)[None]
    feat = rt.cas12a_derive(CAS12A_WINDOW)[None]
    _, grads = cas12a.gradients([seq, feat], head=0)
    assert [g.shape for g in grads] == [seq.shape, feat.shape]
    assert all(np.isfinite(g).all() for g in grads)
    assert np.abs(grads[0]).sum() > 0
    assert np.abs(grads[1]).sum() > 0


def test_gradients_at_the_zero_baseline_stay_finite(spcas9):
    """IG 기준 입력은 영벡터다. 거기서 NaN이 나오면 완결성 계산이 무의미해진다."""
    zero = np.zeros((1, 30, 4), dtype=np.float32)
    y, grads = spcas9.gradients([zero], head=1)
    assert np.isfinite(y).all()
    assert np.isfinite(grads[0]).all()


# ── 기존 XAI 루프와의 접합 ─────────────────────────────────────

def test_integrated_gradients_runs_through_the_existing_loop(spcas9):
    """xai/explain.py의 기존 IG 루프를 그대로 재사용할 수 있다."""
    from xai import explain

    handle = rt.explain_handle(spcas9, head=1, window_length=30)
    result = explain.integrated_gradients(handle, SPCAS9_WINDOW, steps=8)
    assert result.attribution.shape == (30,)
    assert np.isfinite(result.attribution).all()
    assert np.isfinite([result.pred, result.baseline_pred]).all()
    assert result.completeness_error < 5e-3


# ── Cas12a 파생 특징 ──────────────────────────────────────────

def test_cas12a_derive_matches_the_training_layout():
    """354 = GC 2개 + dinucleotide 352개, 그리고 그 순서.

    학습 코드가 `concatenate([gc, dinuc])`였으므로 순서가 뒤집히면 모델은
    그럴듯한 값을 내면서 조용히 틀린다 (#91).
    """
    from cas12a import features

    feat = rt.cas12a_derive(CAS12A_WINDOW)
    gc_feat, dinuc_feat = features.extract_seq_features(CAS12A_WINDOW)
    assert feat.shape == (354,)
    assert feat.dtype == np.float32
    assert np.array_equal(feat[:2], gc_feat)
    assert np.array_equal(feat[2:], dinuc_feat)
    assert feat[2:].sum() == 22.0


def test_shap_path_recomputes_features_from_the_mixed_onehot(cas12a):
    """SHAP이 서열을 섞으면 파생 특징도 함께 다시 계산된다 (#191)."""
    handle = rt.explain_handle(cas12a, head=0, window_length=34, derive=rt.cas12a_derive)
    real = _onehot(CAS12A_WINDOW)
    other = _onehot("GC" * 17)
    y = handle.raw_fn(np.stack([real, other]))
    assert y.shape == (2,)
    assert np.isfinite(y).all()
    direct = cas12a.forward(
        [np.stack([real, other]),
         np.stack([rt.cas12a_derive(CAS12A_WINDOW), rt.cas12a_derive("GC" * 17)])],
        head=0,
    )
    assert np.allclose(y, direct, atol=1e-6)


def test_cas12a_handle_keeps_both_input_totals(cas12a):
    """파생 특징 쪽 기여도를 염기에 배분하지도, 조용히 버리지도 않는다."""
    from xai import explain

    handle = rt.explain_handle(cas12a, head=0, window_length=34, derive=rt.cas12a_derive)
    result = explain.integrated_gradients(handle, CAS12A_WINDOW, steps=8)
    assert result.attribution.shape == (34,)
    assert set(handle.input_totals) == {"Input_Seq", "Input_Feat"}
    assert all(np.isfinite(v) for v in handle.input_totals.values())
    combined = sum(handle.input_totals.values())
    assert abs(combined - (result.pred - result.baseline_pred)) < 1e-2
