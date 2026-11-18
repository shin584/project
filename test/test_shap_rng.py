"""공용 kernel_shap의 RNG 직렬화·복원과 두 seed 독립 제어 (#207).

실제 shap.KernelExplainer를 그대로 쓰고, 모델만 작은 numpy 함수로 바꾼다.
"ACGT"(4 위치)에 nsamples=8이면 2^4=16 부분집합 중 일부를 표본추출하므로
부분집합 seed가 값에 그대로 남는다.
"""

import threading

import numpy as np
import pytest
from predict import ExplainHandle
from xai import explain
from xai.explain import _onehot, kernel_shap

SEQ = "ACGT"
WAIT = 10.0   # 교착 시 테스트가 영원히 멈추지 않게 하는 상한 (초)


class _RecordingHandle:
    """위치끼리 상호작용하는 작은 모델. raw_fn에 들어온 배치를 모두 기록한다."""

    def __init__(self):
        self.calls = []
        weights = np.arange(1, 5, dtype=np.float64)

        def raw_fn(X, on_batch=None):
            X = np.asarray(X, dtype=np.float64)
            self.calls.append(X.copy())
            linear = (X * weights).sum(axis=(1, 2))
            if on_batch is not None:
                on_batch(len(X), len(X))
            return linear + 0.5 * linear ** 2

        self._handle = ExplainHandle(
            raw_fn=raw_fn,
            ig_inputs=lambda seq: [_onehot(seq)],
            ig_grad_fn=lambda batch: (raw_fn(batch[0]), [np.zeros_like(batch[0])]),
            project=lambda attrs: attrs[0].sum(axis=-1),
        )

    def __getattr__(self, name):
        return getattr(self._handle, name)


def _shap(handle=None, **kwargs):
    kwargs = {"nsamples": 8, "n_background": 2, "use_disk_cache": False, **kwargs}
    return kernel_shap(handle or _RecordingHandle(), SEQ, **kwargs)


def _assert_same_shap(a, b):
    """값·기준값·두 seed provenance가 모두 같다."""
    np.testing.assert_array_equal(a.values, b.values)
    assert a.base_value == b.base_value
    assert (a.background_seed, a.subset_seed) == (b.background_seed, b.subset_seed)


def _assert_state_equal(got, expected):
    assert got[0] == expected[0]
    np.testing.assert_array_equal(got[1], expected[1])
    assert got[2:] == expected[2:]


# ── 동시 호출 ────────────────────────────────────────────────────
class _ObservedLock:
    """운영 잠금을 그대로 감싸 '획득 시도'와 획득·해제 순서를 밖에서 볼 수 있게 한다.

    잠금은 바꾸지 않는다. 획득·해제는 원래 잠금(explain._SHAP_RNG_LOCK)에 위임한다.
    """

    def __init__(self, lock):
        self._lock = lock
        self.attempted = {}
        self.log = []
        self._guard = threading.Lock()

    def _event(self, name):
        with self._guard:
            return self.attempted.setdefault(name, threading.Event())

    def __enter__(self):
        name = threading.current_thread().name
        self._event(name).set()
        self._lock.acquire()
        with self._guard:
            self.log.append(("acquired", name))
        return self

    def __exit__(self, *exc):
        with self._guard:
            self.log.append(("released", threading.current_thread().name))
        self._lock.release()
        return False

    def wait_attempt(self, name):
        return self._event(name).wait(WAIT)


@pytest.mark.parametrize("seeds", [
    pytest.param(({"seed": 3}, {"seed": 3}), id="same-seed"),
    pytest.param(({"background_seed": 1, "subset_seed": 2},
                  {"background_seed": 4, "subset_seed": 5}), id="different-seeds"),
])
def test_overlapping_calls_match_serial_calls(monkeypatch, seeds):
    """두 호출이 동시에 진행 중이어도 직렬 호출과 같은 값을 받는다.

    겹침은 직렬화 구간 **밖**에서 정한다. 테스트가 운영 잠금을 먼저 쥔 채 두 호출을
    출발시키고, 두 호출이 모두 explainer 초기화(잠금 밖 모델 평가)를 끝내고 잠금
    획득을 시도한 것을 확인한 뒤에야 잠금을 놓는다. 잠금 안의 계산 코드에는 어떤
    대기도 두지 않는다. 잠금이 빠지면 획득 시도 신호가 오지 않아 실패한다.
    """
    seeds_a, seeds_b = seeds
    serial_a = _shap(**seeds_a)
    serial_b = _shap(**seeds_b)

    real = explain._SHAP_RNG_LOCK
    lock = _ObservedLock(real)
    monkeypatch.setattr(explain, "_SHAP_RNG_LOCK", lock)
    results, errors = {}, []

    def run(name, kwargs):
        try:
            results[name] = _shap(**kwargs)
        except BaseException as exc:   # noqa: BLE001 — 스레드 예외를 본 스레드로 넘긴다
            errors.append(exc)

    np.random.seed(99)
    caller_state = np.random.get_state()
    threads = [threading.Thread(target=run, name=n, args=(n, kw))
               for n, kw in (("A", seeds_a), ("B", seeds_b))]
    real.acquire()
    try:
        for t in threads:
            t.start()
        attempted = [lock.wait_attempt(t.name) for t in threads]
    finally:
        real.release()
    for t in threads:
        t.join(WAIT)
    assert attempted == [True, True], "두 호출이 잠금 앞에서 겹치지 않았다"
    assert not any(t.is_alive() for t in threads), "교착"
    assert not errors, errors

    # 한 호출의 획득~해제 사이에 다른 호출이 끼지 않는다.
    names = [n for _, n in lock.log]
    assert [e for e, _ in lock.log] == ["acquired", "released"] * 2
    assert names[0] == names[1] and names[2] == names[3] and names[0] != names[2]
    _assert_same_shap(results["A"], serial_a)
    _assert_same_shap(results["B"], serial_b)
    # 참여한 호출만 전역 RNG를 썼으므로 호출 전 상태로 돌아온다.
    _assert_state_equal(np.random.get_state(), caller_state)


# ── RNG 복원 ────────────────────────────────────────────────────
class Boom(Exception):
    pass


class Cancelled(Exception):
    """협력 취소: 진행률 콜백이 던지는 신호 (desktop.api.Cancelled와 같은 계약)."""


def _raise_after_subset_sampling(exc, n_background=2):
    """부분집합 표본추출을 마친 뒤 shap이 표본을 평가하는 구간에서 exc를 던지는 on_batch.

    모델 평가 순서 (서열 수 기준 누적 done):
      explainer 초기화(잠금 밖)  : 배경 기준값       → done = n_background
      shap_values 시작(seed 이후): f(x) 단독 평가    → done = 2·n_background
      표본추출 뒤 run()          : 표본 부분집합 평가 → done > 2·n_background
    던지는 시점의 전역 RNG 상태도 기록한다.
    """
    seen = {}

    def on_batch(done, total):
        if done > 2 * n_background:
            seen["rng_at_raise"] = np.random.get_state()
            raise exc

    return on_batch, seen


@pytest.mark.parametrize("exc", [
    pytest.param(Boom("model failed"), id="exception"),
    pytest.param(Cancelled("new request"), id="cooperative-cancel"),
])
def test_rng_is_restored_when_computation_stops_after_sampling(exc):
    np.random.seed(7)
    freshly_seeded = np.random.get_state()
    np.random.seed(1234)
    before = np.random.get_state()
    on_batch, seen = _raise_after_subset_sampling(exc)

    with pytest.raises(type(exc)) as raised:
        _shap(seed=7, on_batch=on_batch)

    assert raised.value is exc
    assert "rng_at_raise" in seen, "표본 평가 구간에 닿기 전에 멈췄다"
    at_raise = seen["rng_at_raise"]
    # seed(7)이 걸렸고 표본추출이 난수를 이미 소비한 뒤였다.
    assert not np.array_equal(at_raise[1], before[1])
    assert (not np.array_equal(at_raise[1], freshly_seeded[1])
            or at_raise[2] != freshly_seeded[2]), "표본추출 전에 멈췄다"
    _assert_state_equal(np.random.get_state(), before)
    # 잠금이 풀려 다음 호출이 그대로 돈다.
    assert not explain._SHAP_RNG_LOCK.locked()
    _assert_same_shap(_shap(seed=7), _shap(seed=7))


def test_rng_is_restored_after_normal_completion():
    np.random.seed(1234)
    before = np.random.get_state()
    _shap(seed=7)
    _assert_state_equal(np.random.get_state(), before)
    assert not explain._SHAP_RNG_LOCK.locked()


# ── 두 seed ──────────────────────────────────────────────────────
def _backgrounds(handle):
    """초기화는 기준 입력(전부 꺼짐)으로 f를 한 번 부른다 → 첫 배치가 배경 그대로다."""
    return handle.calls[0]


def _subset_masks(handle, n_background):
    """표본 평가 배치에서 부분집합 마스크를 되살린다.

    켜진 위치는 모든 배경에서 입력 염기와 같다. 꺼진 위치가 모든 배경에서 우연히
    입력과 같을 확률은 배경 8개면 4^-8이고, seed가 고정이라 결과도 고정이다.
    calls[0]=배경 기준값, calls[1]=f(x) 단독, calls[-1]=pred 계산이므로 그 사이가 표본이다.
    """
    onehot = _onehot(SEQ)
    sampled = np.concatenate(handle.calls[2:-1])
    rows = sampled.reshape(-1, n_background, len(SEQ), 4)
    return (rows == onehot[None, None]).all(axis=-1).all(axis=1)


def test_single_seed_maps_to_both_seeds():
    single = _shap(seed=5)
    split = _shap(background_seed=5, subset_seed=5)
    _assert_same_shap(single, split)
    assert (single.background_seed, single.subset_seed) == (5, 5)


def test_default_call_records_seed_zero_for_both_axes():
    result = _shap()
    assert (result.background_seed, result.subset_seed) == (0, 0)


def test_one_explicit_axis_overrides_only_that_axis():
    result = _shap(seed=5, subset_seed=9)
    assert (result.background_seed, result.subset_seed) == (5, 9)
    _assert_same_shap(result, _shap(background_seed=5, subset_seed=9))


def test_subset_seed_leaves_backgrounds_unchanged():
    n = 8
    h1, h2 = _RecordingHandle(), _RecordingHandle()
    a = _shap(h1, n_background=n, background_seed=3, subset_seed=0)
    b = _shap(h2, n_background=n, background_seed=3, subset_seed=1)
    np.testing.assert_array_equal(_backgrounds(h1), _backgrounds(h2))
    assert a.base_value == b.base_value
    assert not np.array_equal(_subset_masks(h1, n), _subset_masks(h2, n))
    assert not np.array_equal(a.values, b.values)


def test_background_seed_leaves_subset_sampling_unchanged():
    n = 8
    h1, h2 = _RecordingHandle(), _RecordingHandle()
    a = _shap(h1, n_background=n, background_seed=0, subset_seed=4)
    b = _shap(h2, n_background=n, background_seed=1, subset_seed=4)
    assert not np.array_equal(_backgrounds(h1), _backgrounds(h2))
    assert a.base_value != b.base_value
    np.testing.assert_array_equal(_subset_masks(h1, n), _subset_masks(h2, n))


@pytest.mark.parametrize("small,large", [(1, 4), (2, 16), (4, 8)])
def test_more_backgrounds_extend_the_same_random_prefix(small, large):
    h_small, h_large = _RecordingHandle(), _RecordingHandle()
    _shap(h_small, n_background=small, background_seed=11)
    _shap(h_large, n_background=large, background_seed=11)
    np.testing.assert_array_equal(_backgrounds(h_large)[:small], _backgrounds(h_small))


def test_disk_cache_keys_carry_both_seeds(monkeypatch):
    """#208: 사전계산 조회 키에 두 seed가 들어간다 (예전에는 0/0일 때만 조회했다).
    이름 문자열 model_key는 식별을 입증하지 못하므로 아예 조회하지 않는다."""
    import json

    from xai import shap_cache
    from xai.identity import ExplanationIdentity

    looked_up = []
    monkeypatch.setattr(shap_cache, "load", lambda key, root=None: looked_up.append(key))

    class Named(_RecordingHandle):
        model_key, head = "A/Cas12a", "Cas12a"

    kernel_shap(Named(), SEQ, nsamples=8, n_background=2, seed=0)
    assert looked_up == []

    class Keyed(_RecordingHandle):
        model_key = ExplanationIdentity("nt", "A/Cas12a", ("d" * 64,), "p")
        head = "Cas12a"

    for kwargs in ({"subset_seed": 1}, {"background_seed": 1}, {"seed": 0}):
        kernel_shap(Keyed(), SEQ, nsamples=8, n_background=2, **kwargs)
    seeds = [(json.loads(k)["background_seed"], json.loads(k)["subset_seed"])
             for k in looked_up]
    assert seeds == [(0, 1), (1, 0), (0, 0)]
