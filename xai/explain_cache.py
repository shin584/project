"""IG·SHAP·Rollout 결과를 메모리에 들고 있는 캐시. 코어에 있고 Streamlit을 모른다.

왜 코어에 있나
  원래 이 캐시는 `@st.cache_data`로 xai/view.py에 붙어 있었다. 코어를 재사용해도
  데코레이터는 따라오지 않으므로, 데스크톱 화면에서는 보기를 전환하거나 그래프
  크기만 바꿔도 SHAP이 처음부터 다시 돌았다. 성능이 아니라 기능 문제다 (#182).

캐시 키 계약 (#208)
  키는 (기법, 모델 식별, head, window_seq, 기법별 설정)이다.
    IG       : steps
    SHAP     : nsamples, n_background, background_seed, subset_seed
    Rollout  : (없음)
  **모델 식별**은 이름이 아니라 입증된 `xai.identity.ExplanationIdentity`다 —
  계열, checkpoint sha256, 설명 프로토콜, 런타임 설정을 담는다. 두 가지로 얻는다.
    1) model_key 자리에 식별을 직접 넘긴다 (데스크톱 계산 프로세스).
    2) model_key가 이름 문자열("A/Cas12a")이면 예측기에게 묻는다
       (`predictor.explanation_identity()`). 이름과 식별의 model이 같아야 한다.
  둘 다 안 되면 식별을 입증할 수 없다. 그 요청은 **적중으로 처리하지 않고** 매번
  새로 계산하며, 결과를 저장하지도 않는다.

  예측기 객체와 진행률 콜백은 키에 들어가지 않는다. 콜백은 호출마다 새 함수
  객체라 키에 넣으면 캐시가 매번 빗나간다. 예측기가 무엇을 계산하는지는 식별이
  정한다. 그래프 크기는 애초에 키 재료가 아니다 (화면이 그릴 때만 쓴다).

  SHAP seed는 explain.kernel_shap과 같은 규칙으로 정한다 (#207): 단일 seed는 두
  축에 같은 값으로 대응하고, background_seed·subset_seed를 주면 그 축만 바뀐다.

handle은 미적중일 때만 만든다
  `predictor.explain_handle(head)`는 모델을 올리게 할 수 있다. 적중이면 부르지
  않는다 — 캐시가 있다는 말은 모델을 건드리지 않는다는 뜻이어야 한다.

디스크 기반 사전계산 SHAP 캐시(xai/shap_cache.py)와는 별개다. 그쪽은
`explain.kernel_shap` 안에서 조회하며 여기서 손대지 않는다. 미적중이면 이
캐시가 kernel_shap을 부르고, kernel_shap이 다시 디스크를 본다.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from xai import explain, identity
from xai.quality import record_quality

#: 입증된 계산 식별, 또는 예측기가 식별을 대야 하는 이름 문자열 ("A/Cas12a").
ModelKey = identity.ExplanationIdentity | str

#: 들고 있을 결과 개수. 하나는 서열 길이만 한 배열 몇 개라 작다. 화면은 사이트와
#: 설정을 오가므로 몇 개만 두면 금방 밀려나고, 무제한이면 오래 켜 둔 데스크톱
#: 앱에서 계속 쌓인다. 넘치면 가장 오래 쓰지 않은 것부터 버린다.
DEFAULT_MAX_ENTRIES = 64


class ExplainCache:
    """기법별 XAI 결과 캐시. 적중하면 계산 함수도 handle도 부르지 않는다.

    스레드 안전: 데스크톱은 작업 스레드에서 계산한다. 자료구조를 만지는 동안만
    잠그고 **계산 중에는 잠그지 않는다** (긴 SHAP이 화면 스레드를 막으면 안 된다).
    그래서 같은 키를 두 스레드가 동시에 미적중으로 만나면 두 번 계산될 수 있다.
    값이 같으므로 결과는 같고, 낭비는 드물다 — 잠금을 들고 계산하는 쪽이 나쁘다.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES, *, use_disk_cache: bool = True) -> None:
        self._entries: OrderedDict[tuple, Any] = OrderedDict()
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._use_disk_cache = use_disk_cache

    # ── 기법별 입구 ──────────────────────────────────────────────────
    def integrated_gradients(self, predictor, model_key: ModelKey, head: str, window_seq: str,
                             steps: int, on_batch: Callable | None = None) -> explain.IGResult:
        return self._get_or_compute(
            model_key, predictor,
            ("ig", str(head), window_seq.upper(), int(steps)),
            lambda handle: explain.integrated_gradients(handle, window_seq, int(steps),
                                                        on_batch=on_batch),
            head,
        )

    def kernel_shap(self, predictor, model_key: ModelKey, head: str, window_seq: str,
                    nsamples: int, n_background: int,
                    on_batch: Callable | None = None, seed: int = 0, *,
                    background_seed: int | None = None,
                    subset_seed: int | None = None) -> explain.SHAPResult:
        background_seed, subset_seed = explain.resolve_seeds(seed, background_seed, subset_seed)
        return self._get_or_compute(
            model_key, predictor,
            ("shap", str(head), window_seq.upper(), int(nsamples), int(n_background),
             background_seed, subset_seed),
            lambda handle: explain.kernel_shap(handle, window_seq, int(nsamples),
                                               int(n_background), on_batch=on_batch,
                                               background_seed=background_seed,
                                               subset_seed=subset_seed,
                                               use_disk_cache=self._use_disk_cache),
            head,
        )

    def attention_rollout(self, predictor, model_key: ModelKey, head: str,
                          window_seq: str) -> explain.RolloutResult:
        return self._get_or_compute(
            model_key, predictor,
            ("rollout", str(head), window_seq.upper()),
            lambda handle: explain.attention_rollout(handle, window_seq),
            head,
        )

    # ── 살림 ─────────────────────────────────────────────────────────
    def clear(self) -> None:
        """전부 버린다. 모델 파일을 바꿨거나 테스트가 깨끗한 상태를 원할 때."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ── 내부 ─────────────────────────────────────────────────────────
    @staticmethod
    def _identity(model_key, predictor) -> identity.ExplanationIdentity | None:
        """입증된 모델 식별, 또는 입증할 수 없으면 None."""
        direct = identity.verified(model_key)
        if direct is not None or isinstance(model_key, identity.ExplanationIdentity):
            return direct
        ask = getattr(predictor, "explanation_identity", None)
        if ask is None:
            return None
        found = identity.verified(ask())
        # 이름이 가리키는 모델과 예측기가 실제로 든 모델이 다르면 입증이 아니다.
        return found if found is not None and found.model == str(model_key) else None

    def _get_or_compute(self, model_key, predictor, request: tuple,
                        compute: Callable[[Any], Any], head: str):
        model_identity = self._identity(model_key, predictor)
        if model_identity is None:
            # 식별을 입증할 수 없다: 적중으로 처리하지도, 저장하지도 않는다.
            result = compute(predictor.explain_handle(head))
            record_quality(result, None, request)
            return result

        key = (request[0], model_identity) + request[1:]
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]

        # 미적중일 때만 handle을 만든다 (모델을 올릴 수 있다).
        result = compute(predictor.explain_handle(head))
        record_quality(result, model_identity, request)

        with self._lock:
            self._entries[key] = result
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return result


#: 프로세스 하나에 캐시 하나. Streamlit은 상호작용마다 스크립트를 다시 돌리지만
#: `sys.modules`는 유지되므로 이 객체는 살아남는다 (ADR 0001의 모듈 수준 싱글턴).
CACHE = ExplainCache()
