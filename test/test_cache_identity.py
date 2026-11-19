"""계산 식별이 완전히 같을 때만 설명 캐시를 재사용한다 (#208).

공개 경계만 쓴다: XAI 메모리 캐시(`ExplainCache`), 사전계산 SHAP 캐시
(`shap_cache.shap_cache_key/load/save`), 그 캐시를 조회하는 `explain.kernel_shap`.
SHAP은 실제 `shap.KernelExplainer`를 돌리고, 모델만 작은 numpy 함수로 바꾼다.

확인하는 것
  1) 식별 항목(checkpoint·계열·프로토콜·런타임 설정·헤드·윈도우·기법별 설정·두
     seed) 하나만 달라도 재사용하지 않는다 — 양 계열(nt, non_nt) 모두
  2) 식별을 입증할 수 없으면(이름 문자열뿐, 해시 형식 위반, 모르는 계열, 빈
     프로토콜) 적중으로 처리하지 않는다
  3) 프로토콜·출처 없는 legacy 산출물을 거부한다 (저장소의 실제 4개 포함)
  4) 거부 뒤에는 정상 fresh 계산으로 이어지고, 허용된 적중은 fresh 계산과 수치가 같다
  5) #207의 두 seed와 단일 seed 대응이 키에 그대로 반영된다
"""

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest
from predict import ExplainHandle
from xai import explain, explain_cache, shap_cache
from xai.explain import _onehot
from xai.identity import ExplanationIdentity

ROOT = Path(__file__).resolve().parent.parent
SEQ = "ACGT"
HEAD = "Cas12a"
NSAMPLES, N_BACKGROUND = 8, 2


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _identity(family="nt", **changes):
    base = ExplanationIdentity(family=family, model=f"{family}/Cas12a",
                               checkpoints=(_sha(f"{family} checkpoint"),),
                               protocol=f"{family}-protocol-v1",
                               configuration='{"dtype": "float32"}')
    return dataclasses.replace(base, **changes)


#: 식별의 모델 쪽 항목을 하나씩 바꾼 것. 양 계열에 같은 변화를 준다.
MODEL_CHANGES = {
    "checkpoint": dict(checkpoints=(_sha("retrained"),)),
    "protocol": dict(protocol="protocol-v2-units-corrected"),
    "configuration": dict(configuration='{"dtype": "bfloat16"}'),
    "model": dict(model="other/Cas12a"),
}


class _Model:
    """위치끼리 상호작용하는 작은 모델. 추론 호출 수를 센다."""

    def __init__(self, scale=1.0):
        self.calls = 0
        self.weights = np.arange(1, 5, dtype=np.float64) * scale

    def raw_fn(self, X, on_batch=None):
        self.calls += 1
        X = np.asarray(X, dtype=np.float64)
        linear = (X * self.weights).sum(axis=(1, 2))
        return linear + 0.5 * linear ** 2

    def handle(self, model_key=None, head=HEAD):
        def grad(batch):
            x = np.asarray(batch[0], dtype=np.float64)
            linear = (x * self.weights).sum(axis=(1, 2))
            g = (1.0 + linear)[:, None, None] * self.weights[None, :, None]
            self.calls += 1
            return linear + 0.5 * linear ** 2, [np.broadcast_to(g, x.shape).copy()]

        return ExplainHandle(raw_fn=self.raw_fn, ig_inputs=lambda seq: [_onehot(seq)],
                             ig_grad_fn=grad, project=lambda attrs: attrs[0].sum(axis=-1),
                             model_key=model_key, head=head)


class _Predictor:
    """explain_handle 호출 수를 센다. identity가 있으면 자기 식별을 댄다."""

    def __init__(self, model=None, identity=None):
        self.model = model or _Model()
        self.identity = identity
        self.handles = 0

    def explain_handle(self, head):
        self.handles += 1
        return self.model.handle(self.identity, head)

    def __getattr__(self, name):
        if name == "explanation_identity" and self.__dict__.get("identity") is not None:
            return lambda: self.identity
        raise AttributeError(name)


def _fresh_shap(model=None, **kwargs):
    """캐시를 거치지 않은 계산. 비교 기준."""
    kwargs = {"nsamples": NSAMPLES, "n_background": N_BACKGROUND, **kwargs}
    return explain.kernel_shap((model or _Model()).handle(), SEQ, use_disk_cache=False, **kwargs)


def _same_shap(a, b):
    np.testing.assert_array_equal(a.values, b.values)
    assert (a.base_value, a.pred, a.additivity_error) == (b.base_value, b.pred, b.additivity_error)
    assert (a.background_seed, a.subset_seed) == (b.background_seed, b.subset_seed)


# ══ XAI 메모리 캐시 ═════════════════════════════════════════════════
@pytest.mark.parametrize("family", ["nt", "non_nt"])
@pytest.mark.parametrize("change", sorted(MODEL_CHANGES))
def test_memory_cache_rejects_each_model_identity_difference(family, change):
    cache = explain_cache.ExplainCache(use_disk_cache=False)
    predictor = _Predictor()
    old = _identity(family)
    cache.kernel_shap(predictor, old, HEAD, SEQ, NSAMPLES, N_BACKGROUND)

    new = _identity(family, **MODEL_CHANGES[change])
    got = cache.kernel_shap(predictor, new, HEAD, SEQ, NSAMPLES, N_BACKGROUND)

    assert predictor.handles == 2, f"{change}가 다른데 예전 결과를 재사용했다"
    _same_shap(got, _fresh_shap())


def test_memory_cache_does_not_share_results_between_families():
    cache = explain_cache.ExplainCache(use_disk_cache=False)
    predictor = _Predictor()
    shared = dict(model="A/Cas12a", checkpoints=(_sha("same"),), protocol="p")
    cache.integrated_gradients(predictor, _identity("nt", **shared), HEAD, SEQ, 8)
    cache.integrated_gradients(predictor, _identity("non_nt", **shared), HEAD, SEQ, 8)

    assert predictor.handles == 2


@pytest.mark.parametrize("change", [
    pytest.param(dict(head="SpCas9-NG"), id="head"),
    pytest.param(dict(window_seq="TGCA"), id="window"),
    pytest.param(dict(nsamples=16), id="nsamples"),
    pytest.param(dict(n_background=3), id="n_background"),
    pytest.param(dict(background_seed=1), id="background_seed"),
    pytest.param(dict(subset_seed=1), id="subset_seed"),
    pytest.param(dict(seed=1), id="single_seed"),
])
def test_memory_cache_rejects_each_request_difference(change):
    cache = explain_cache.ExplainCache(use_disk_cache=False)
    predictor = _Predictor()
    base = dict(head=HEAD, window_seq=SEQ, nsamples=NSAMPLES, n_background=N_BACKGROUND)
    cache.kernel_shap(predictor, _identity(), **base)

    request = {**base, **change}
    got = cache.kernel_shap(predictor, _identity(), **request)

    assert predictor.handles == 2
    window = request.pop("window_seq")
    request.pop("head")
    _same_shap(got, explain.kernel_shap(_Model().handle(), window, use_disk_cache=False,
                                        **request))


def test_memory_cache_rejects_a_different_ig_step_count():
    cache = explain_cache.ExplainCache()
    predictor = _Predictor()
    cache.integrated_gradients(predictor, _identity(), HEAD, SEQ, 8)
    got = cache.integrated_gradients(predictor, _identity(), HEAD, SEQ, 16)

    assert predictor.handles == 2
    fresh = explain.integrated_gradients(_Model().handle(), SEQ, 16)
    np.testing.assert_array_equal(got.attribution, fresh.attribution)


def test_single_seed_and_equal_explicit_seeds_are_the_same_computation():
    cache = explain_cache.ExplainCache(use_disk_cache=False)
    predictor = _Predictor()
    first = cache.kernel_shap(predictor, _identity(), HEAD, SEQ, NSAMPLES, N_BACKGROUND, seed=3)
    second = cache.kernel_shap(predictor, _identity(), HEAD, SEQ, NSAMPLES, N_BACKGROUND,
                               background_seed=3, subset_seed=3)

    assert second is first
    assert predictor.handles == 1
    assert (first.background_seed, first.subset_seed) == (3, 3)


def test_accepted_memory_hits_equal_fresh_computation():
    cache = explain_cache.ExplainCache(use_disk_cache=False)
    predictor = _Predictor()
    args = (predictor, _identity(), HEAD, SEQ)
    cache.kernel_shap(*args, NSAMPLES, N_BACKGROUND, background_seed=1, subset_seed=2)
    cache.integrated_gradients(*args, 16)
    calls = predictor.model.calls

    shap_hit = cache.kernel_shap(*args, NSAMPLES, N_BACKGROUND, background_seed=1, subset_seed=2)
    ig_hit = cache.integrated_gradients(*args, 16)

    assert predictor.model.calls == calls and predictor.handles == 2, "적중인데 계산했다"
    _same_shap(shap_hit, _fresh_shap(background_seed=1, subset_seed=2))
    ig_fresh = explain.integrated_gradients(_Model().handle(), SEQ, 16)
    np.testing.assert_array_equal(ig_hit.attribution, ig_fresh.attribution)
    assert (ig_hit.pred, ig_hit.baseline_pred) == (ig_fresh.pred, ig_fresh.baseline_pred)


# ── 입증할 수 없는 식별 ─────────────────────────────────────────────
UNPROVEN = {
    "name_only": ("A/Cas12a", None),
    "short_checksum": (_identity(checkpoints=("abc",)), None),
    "no_checkpoint": (_identity(checkpoints=()), None),
    "unknown_family": (_identity(family="cnn"), None),
    "empty_protocol": (_identity(protocol=""), None),
    "predictor_names_other_model": ("A/Cas12a", _identity(model="A/SpCas9")),
    "predictor_identity_unverified": ("nt/Cas12a", _identity(protocol="")),
}


@pytest.mark.parametrize("case", sorted(UNPROVEN))
def test_unproven_identity_is_never_a_hit(case):
    model_key, predictor_identity = UNPROVEN[case]
    cache = explain_cache.ExplainCache(use_disk_cache=False)
    predictor = _Predictor(identity=predictor_identity)

    first = cache.kernel_shap(predictor, model_key, HEAD, SEQ, NSAMPLES, N_BACKGROUND)
    second = cache.kernel_shap(predictor, model_key, HEAD, SEQ, NSAMPLES, N_BACKGROUND)

    assert predictor.handles == 2 and len(cache) == 0
    _same_shap(second, _fresh_shap())
    _same_shap(first, second)


def test_a_name_key_is_reused_when_the_predictor_proves_its_identity():
    """Streamlit 경로: model_key는 이름이지만 예측기가 입증된 식별을 댄다."""
    cache = explain_cache.ExplainCache(use_disk_cache=False)
    predictor = _Predictor(identity=_identity(model="A/Cas12a"))
    first = cache.kernel_shap(predictor, "A/Cas12a", HEAD, SEQ, NSAMPLES, N_BACKGROUND)
    second = cache.kernel_shap(predictor, "A/Cas12a", HEAD, SEQ, NSAMPLES, N_BACKGROUND)

    assert second is first and predictor.handles == 1

    predictor.identity = _identity(model="A/Cas12a", checkpoints=(_sha("swapped"),))
    cache.kernel_shap(predictor, "A/Cas12a", HEAD, SEQ, NSAMPLES, N_BACKGROUND)
    assert predictor.handles == 2, "checkpoint가 바뀌었는데 예전 결과를 재사용했다"


# ══ 사전계산 SHAP 캐시 ══════════════════════════════════════════════
@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(shap_cache, "DEFAULT_DIR", tmp_path)
    return tmp_path


def _key(identity=None, **changes):
    request = {**dict(head=HEAD, window_seq=SEQ, nsamples=NSAMPLES, n_background=N_BACKGROUND,
                      background_seed=0, subset_seed=0), **changes}
    return shap_cache.shap_cache_key(identity or _identity(), **request)


def test_every_identity_field_changes_the_disk_key():
    base = _key()
    variants = [_key(_identity(family, **c)) for family in ("nt", "non_nt")
                for c in MODEL_CHANGES.values()]
    variants += [_key(_identity("non_nt"))]
    variants += [_key(**{k: v}) for k, v in dict(
        head="SpCas9-NG", window_seq="TGCA", nsamples=16, n_background=3,
        background_seed=1, subset_seed=1).items()]
    assert len({base, *variants}) == len(variants) + 1


@pytest.mark.parametrize("family", ["nt", "non_nt"])
def test_an_accepted_disk_hit_equals_fresh_and_skips_inference(store, family):
    ident = _identity(family)
    shap_cache.save(_key(ident, background_seed=1, subset_seed=2),
                    _fresh_shap(background_seed=1, subset_seed=2))

    model = _Model()
    hit = explain.kernel_shap(model.handle(ident), SEQ, NSAMPLES, N_BACKGROUND,
                              background_seed=1, subset_seed=2)

    assert model.calls == 0, "적중했는데 추론이 돌았다"
    _same_shap(hit, _fresh_shap(background_seed=1, subset_seed=2))


@pytest.mark.parametrize("family", ["nt", "non_nt"])
@pytest.mark.parametrize("change", sorted(MODEL_CHANGES))
def test_disk_results_of_another_model_identity_are_rejected(store, family, change):
    stale_model = _Model(scale=2.0)            # 예전 checkpoint/프로토콜이 낸 다른 값
    shap_cache.save(_key(_identity(family)), _fresh_shap(stale_model))

    current = _identity(family, **MODEL_CHANGES[change])
    model = _Model()
    got = explain.kernel_shap(model.handle(current), SEQ, NSAMPLES, N_BACKGROUND)

    assert model.calls > 0, "다른 식별의 디스크 결과를 재사용했다"
    _same_shap(got, _fresh_shap())


@pytest.mark.parametrize("seeds", [(1, 0), (0, 1)])
def test_disk_results_of_other_seeds_are_rejected(store, seeds):
    shap_cache.save(_key(), _fresh_shap(_Model(scale=2.0)))
    model = _Model()
    got = explain.kernel_shap(model.handle(_identity()), SEQ, NSAMPLES, N_BACKGROUND,
                              background_seed=seeds[0], subset_seed=seeds[1])

    assert model.calls > 0
    _same_shap(got, _fresh_shap(background_seed=seeds[0], subset_seed=seeds[1]))


def test_a_legacy_artifact_without_provenance_is_rejected(store):
    """#152 형식(schema 없음)으로 정확히 그 경로에 놓인 파일도 거부한다."""
    key = _key()
    stale = _fresh_shap(_Model(scale=2.0))
    path = shap_cache.save(key, stale)
    np.savez(path, key=np.array(key), values=stale.values, base_value=stale.base_value,
             pred=stale.pred, additivity_error=stale.additivity_error)

    assert shap_cache.load(key) is None
    model = _Model()
    got = explain.kernel_shap(model.handle(_identity()), SEQ, NSAMPLES, N_BACKGROUND)
    assert model.calls > 0
    _same_shap(got, _fresh_shap())


@pytest.mark.parametrize("bad", ["nan", "shape"])
def test_a_malformed_artifact_is_rejected(store, bad):
    stale = _fresh_shap()
    stale.values = np.append(stale.values, 0.0) if bad == "shape" else stale.values * np.nan
    shap_cache.save(_key(), stale)

    model = _Model()
    explain.kernel_shap(model.handle(_identity()), SEQ, NSAMPLES, N_BACKGROUND)
    assert model.calls > 0


def test_unproven_keys_are_never_looked_up_or_saved(store):
    name_key = shap_cache.shap_cache_key("A/Cas12a", HEAD, SEQ, NSAMPLES, N_BACKGROUND)
    with pytest.raises(ValueError):
        shap_cache.save(name_key, _fresh_shap())
    with pytest.raises(ValueError):
        shap_cache.save(_key(_identity(protocol="")), _fresh_shap())
    with pytest.raises(ValueError, match="seed"):
        shap_cache.save(_key(), _fresh_shap(seed=1))
    assert shap_cache.load(name_key) is None
    assert not list(store.iterdir())

    model = _Model()
    explain.kernel_shap(model.handle("A/Cas12a"), SEQ, NSAMPLES, N_BACKGROUND)
    assert model.calls > 0


def test_the_four_repository_legacy_artifacts_are_rejected_and_left_untouched():
    """저장소의 실제 legacy 파일. 예전 키로도, 같은 윈도우의 입증된 키로도 적중하지 않는다."""
    files = sorted((ROOT / "data" / "precomputed_shap").glob("*.npz"))
    assert len(files) == 4
    before = {p: p.read_bytes() for p in files}

    for path in files:
        with np.load(path) as f:
            assert "schema" not in f.files
            model_key, head, window, nsamples, n_background = str(f["key"]).split("|")
        assert shap_cache.load(shap_cache.shap_cache_key(
            model_key, head, window, nsamples, n_background)) is None
        nt = _identity("nt", model=model_key)
        assert shap_cache.load(shap_cache.shap_cache_key(
            nt, head, window, nsamples, n_background)) is None

    assert {p: p.read_bytes() for p in files} == before, "legacy 산출물을 바꿨다"


# ══ NT 계열의 실제 식별 ════════════════════════════════════════════
def test_nt_identity_hashes_checkpoint_bytes_and_follows_the_protocol(tmp_path, monkeypatch):
    import predict

    weights = tmp_path / "Cas12a.safetensors"
    weights.write_bytes(b"checkpoint one")
    spec = dataclasses.replace(predict.MODEL_REGISTRY[("A", "Cas12a")], paths=(str(weights),))

    first = predict.nt_explanation_identity(spec)
    assert first.verified and first.family == "nt" and first.model == "A/Cas12a"
    assert first.checkpoints == (_sha("checkpoint one"),)
    assert first.protocol == predict.NT_EXPLANATION_PROTOCOL

    weights.write_bytes(b"checkpoint two")
    assert predict.nt_explanation_identity(spec) != first
    weights.write_bytes(b"checkpoint one")
    monkeypatch.setattr(predict, "NT_EXPLANATION_PROTOCOL", "nt-units-corrected-v2")
    assert predict.nt_explanation_identity(spec) != first


def test_display_only_spec_fields_do_not_change_the_explanation_identity(tmp_path):
    """#196이 더한 표시용 이름이 기존 설명 캐시를 전부 미스로 만들지 않는지.

    `name`·`asset_id`·`raw_unit`은 화면이 읽는 이름이고 `family`는 식별이 따로 들고
    있다. 계산은 그대로이므로 식별도 그대로여야 한다 — 그렇지 않으면 열 이름을
    고치는 것만으로 사전계산 SHAP 캐시가 버려진다.
    """
    import predict

    weights = tmp_path / "Cas12a.safetensors"
    weights.write_bytes(b"checkpoint one")
    spec = dataclasses.replace(predict.MODEL_REGISTRY[("A", "Cas12a")], paths=(str(weights),))
    renamed = dataclasses.replace(
        spec, name="다른 이름", asset_id="other-asset", raw_unit="다른 단위 설명")
    assert predict.nt_explanation_identity(renamed) == predict.nt_explanation_identity(spec)

    # 반대로 계산을 바꾸는 항목은 식별을 바꾼다.
    rescaled = dataclasses.replace(spec, output_mode="sigmoid")
    assert predict.nt_explanation_identity(rescaled) != predict.nt_explanation_identity(spec)


def test_the_streamlit_name_path_follows_protocol_and_checkpoint_changes(tmp_path, monkeypatch):
    """예측기가 대는 식별은 프로세스 수명 동안 굳지 않는다 — #201이 프로토콜을 올리거나
    checkpoint 파일이 바뀌면 이름 키("A/Cas12a") 경로도 예전 결과를 재사용하지 않는다."""
    import os

    import predict

    weights = tmp_path / "Cas12a.safetensors"
    weights.write_bytes(b"checkpoint one")
    spec = dataclasses.replace(predict.MODEL_REGISTRY[("A", "Cas12a")], paths=(str(weights),))
    nt = predict.NTClassifierPredictor(spec)
    model = _Model()
    monkeypatch.setattr(nt, "explain_handle", lambda head: model.handle(None, head))
    cache = explain_cache.ExplainCache(use_disk_cache=False)

    def run():
        return cache.kernel_shap(nt, "A/Cas12a", HEAD, SEQ, NSAMPLES, N_BACKGROUND)

    first = run()
    calls = model.calls
    assert run() is first and model.calls == calls

    monkeypatch.setattr(predict, "NT_EXPLANATION_PROTOCOL", "nt-units-corrected-v2")
    second = run()
    assert model.calls > calls, "프로토콜이 바뀌었는데 예전 결과를 재사용했다"
    _same_shap(second, first)

    calls = model.calls
    weights.write_bytes(b"checkpoint two")
    os.utime(weights, ns=(1, 1))
    run()
    assert model.calls > calls, "checkpoint가 바뀌었는데 예전 결과를 재사용했다"
