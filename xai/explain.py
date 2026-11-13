"""연결된 실제 모델에서 XAI 값(IG, SHAP)을 계산한다. 화면 코드와 분리된 순수 계산부.

모델 접근은 predict.ExplainHandle로만 한다. 여기서는 TensorFlow/PyTorch를 직접 쓰지 않는다.
모든 값은 모델 원출력(0~1 비율, clip 전) 기준이고, 화면에서 ×100(%p)으로 바꿔 보여준다.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class IGResult:
    attribution: np.ndarray    # [L] 위치별 기여도 (원출력 단위)
    pred: float                # f(x)
    baseline_pred: float       # f(기준 입력)
    completeness_error: float  # |Σ기여도 − (f(x) − f(기준))|


@dataclass
class SHAPResult:
    values: np.ndarray         # [L] 위치별 SHAP 값 (원출력 단위)
    base_value: float          # E[f] (배경 서열 평균 예측)
    pred: float                # f(x)
    additivity_error: float    # |base + Σ SHAP − f(x)|


#: IG 적분 단계를 한 번에 몇 개씩 역전파할지. 메모리가 부족하면 줄인다.
#: 값에는 영향이 없고 (아래 설명 참고) 최대 메모리 사용량만 달라진다.
IG_CHUNK_SIZE = 8


def integrated_gradients(handle, window_seq: str, steps: int = 64,
                         chunk_size: int = IG_CHUNK_SIZE) -> IGResult:
    """Integrated Gradients. 기준 입력은 전부 0(빈 서열), 구간 적분은 중점법.

    적분 단계를 한 배치에 모두 올리면 역전파 그래프가 steps배로 커져 메모리를 많이 쓴다.
    청크로 나눠 기울기 '합'을 누적하고 마지막에 steps로 나누면
    평균을 한 번에 구한 것과 수학적으로 같은 값이 나온다 (mean = sum / steps).
    """
    inputs = handle.ig_inputs(window_seq)
    alphas = (np.arange(steps) + 0.5) / steps

    chunk = max(1, int(chunk_size))
    sums = [np.zeros(np.shape(x), dtype=np.float64) for x in inputs]
    for i in range(0, steps, chunk):
        a = alphas[i:i + chunk]
        batch = [a.reshape((-1,) + (1,) * x.ndim) * x[None] for x in inputs]
        _, grads = handle.ig_grad_fn(batch)
        for s, g in zip(sums, grads):
            s += np.asarray(g, dtype=np.float64).sum(axis=0)

    attrs = [x * (s / steps) for x, s in zip(inputs, sums)]

    f_x = float(handle.ig_grad_fn([x[None] for x in inputs])[0][0])
    f_0 = float(handle.ig_grad_fn([np.zeros_like(x)[None] for x in inputs])[0][0])
    total = float(sum(a.sum() for a in attrs))
    return IGResult(
        attribution=np.asarray(handle.project(attrs), dtype=np.float64),
        pred=f_x, baseline_pred=f_0,
        completeness_error=abs(total - (f_x - f_0)),
    )


def _onehot(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    out = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        if b in mapping:
            out[i, mapping[b]] = 1.0
    return out


def _random_backgrounds(length: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(n, length))
    out = np.zeros((n, length, 4), dtype=np.float32)
    np.put_along_axis(out, idx[..., None], 1.0, axis=2)
    return out


def kernel_shap(handle, window_seq: str, nsamples: int = 512,
                n_background: int = 16, seed: int = 0,
                on_batch=None) -> SHAPResult:
    """염기 위치를 특징으로 하는 KernelSHAP.

    '꺼진' 위치는 무작위 배경 서열의 염기로 바꿔 모델에 넣고, 배경 n개의 평균을 쓴다.

    on_batch(done, total) : 진행률 콜백 (ADR 0001). done·total은 모델에 넣은 서열 수다.
        훅은 f(masks) 안이 아니라 handle.raw_fn 안쪽 배치 루프에 걸린다.
        f는 몇 번밖에 불리지 않아 여기에 걸면 진행률이 몇 단계로 끝난다.
        코어는 숫자만 넘기고 표시 문구는 화면이 정한다.
    """
    import shap

    onehot = _onehot(window_seq)
    length = onehot.shape[0]
    backgrounds = _random_backgrounds(length, n_background, seed)

    # shap은 f를 여러 번 부른다(배경 기준값 1회 + 샘플 1회). 호출마다 0부터 다시 세면
    # 진행률이 뒤로 돌아가므로, 누적 서열 수를 전체 예상치에 대고 보고한다.
    expected = max(1, n_background * (nsamples + 1))
    seen = 0

    def f(masks):
        nonlocal seen
        masks = np.asarray(masks)
        n = masks.shape[0]
        mixed = np.where(masks[:, None, :, None] > 0.5, onehot[None, None], backgrounds[None])
        X = mixed.reshape(n * n_background, length, 4)
        if on_batch is None:
            y = handle.raw_fn(X)
        else:
            base = seen
            y = handle.raw_fn(
                X,
                on_batch=lambda done, _total: on_batch(min(base + done, expected), expected),
            )
            seen = base + X.shape[0]
        return np.asarray(y).reshape(n, n_background).mean(axis=1)

    explainer = shap.KernelExplainer(f, np.zeros((1, length)))
    values = explainer.shap_values(np.ones((1, length)), nsamples=nsamples, l1_reg=0, silent=True)
    values = np.asarray(values, dtype=np.float64).reshape(-1)[:length]
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    pred = float(handle.raw_fn(onehot[None])[0])
    return SHAPResult(values=values, base_value=base, pred=pred,
                      additivity_error=abs(base + values.sum() - pred))

@dataclass
class RolloutResult:
    """Attention Rollout 결과.

    token_matrix    : [T, T] 토큰 단위 정보 의존도 (행이 열을 참조하는 정도)
    token_labels    : 토큰 T개의 이름 ('<cls>' 또는 '0-5' 같은 구간)
    position_profile: [L] 위치별로 '얼마나 참조되는지' 합산한 값
    n_layers        : 누적에 쓴 층 수
    """
    token_matrix:     np.ndarray
    token_labels:     list
    position_profile: np.ndarray
    n_layers:         int


def attention_rollout(handle, window_seq: str) -> RolloutResult:
    """Abnar & Zuidema(2020) 방식의 Attention Rollout.

    층마다 헤드를 평균한 뒤 잔차 연결을 단위행렬로 더하고 행 합을 1로 맞춘 다음,
    아래층부터 위층까지 곱해 누적한다. 토큰 기여도는 그 토큰이 덮는 염기에 균등 배분한다.
    """
    if handle.attention_fn is None:
        raise NotImplementedError("이 모델은 attention이 없어 Rollout을 계산할 수 없습니다")

    layers, spans, length = handle.attention_fn(window_seq)
    if not layers:
        raise RuntimeError(
            "모델이 attention 값을 돌려주지 않았습니다. "
            "predict.py의 EsmConfig에 attn_implementation=\"eager\"가 있는지 확인하세요."
        )

    rollout = None
    for att in layers:
        a = np.asarray(att, dtype=np.float64).mean(axis=0)      # 헤드 평균 [T, T]
        a = a + np.eye(a.shape[0])                              # 잔차 연결
        a = a / a.sum(axis=-1, keepdims=True)                   # 행 합 = 1
        rollout = a if rollout is None else a @ rollout

    # 각 토큰이 다른 토큰들로부터 얼마나 참조되는지 (열 방향 합)
    per_token = rollout.sum(axis=0)
    profile = np.zeros(length, dtype=np.float64)
    for value, (start, end) in zip(per_token, spans):
        if end > start:
            stop = min(end, length)
            if stop > start:
                profile[start:stop] += value / (end - start)

    labels = [f"{a}-{b - 1}" if b > a else "special" for a, b in spans]
    return RolloutResult(token_matrix=rollout, token_labels=labels,
                         position_profile=profile, n_layers=len(layers))
