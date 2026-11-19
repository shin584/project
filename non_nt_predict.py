"""non-NT 계열의 예측기 (#196). 채택 런타임은 `non_nt_runtime`이 정한다.

이 모듈은 **예측**만 한다. 설명(IG·SHAP) 계산을 기존 XAI 루프에 꽂는 일은 #199이고,
Cas12a non-NT 예측은 #197이다. 아직 없는 것을 성공한 결과로 내놓지 않기 위해,
`explain_handle`은 구현되지 않았다고 분명히 말하고 멈춘다.

왜 `predict.py`가 아니라 따로 있는가
    `predict.py`를 import하면 NT 설정·레지스트리까지 따라오지만 torch/transformers는
    따라오지 않는다. 반대로 이 모듈은 Keras 런타임을 끌어온다. 계열을 고르지 않은
    프로세스가 두 런타임을 모두 import하지 않도록, `predict.FAMILY_PREDICTOR_CLASSES`는
    이 클래스를 **이름으로** 들고 있다 (#191의 RAM 설계).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

import model_source
from pam_scanner import SPCAS9_VARIANTS, CandidateSite, VariantResult, variant_compatible
from predict import (
    BasePredictor, ModelIdentity, ModelSpec, variant_result, version_label,
)

logger = logging.getLogger(__name__)

#: 9-Hydra v2 export의 출력 층 이름을 **변종 순서대로** 적은 것.
#: 출처는 `models/non_nt/spcas9-9hydra-v2/export-manifest.json`의 `heads`
#: (index → protein → layer)이며, 그 index 순서가 `SPCAS9_VARIANTS`와 같다.
#: 적재한 모델의 `output_names`를 이것과 대조한다. 순서가 어긋나면 예측은 그럴듯한
#: 숫자를 내면서 변종만 뒤바뀌므로, 조용히 틀리는 대신 멈춘다.
SPCAS9_HEAD_LAYERS: tuple[str, ...] = (
    "Output_SpCas9",          # SpCas9(WT)
    "Output_SpCas9_NG",       # SpCas9-NG
    "Output_VRQR_variant",    # VRQR
    "Output_xCas",            # xCas9
    "Output_Sniper_Cas9",     # Sniper
    "Output_SpCas9_HF_1",     # SpCas9-HF1
    "Output_eSpCas9_1_1_",    # eSpCas9(1.1)
    "Output_HypaCas9",        # HypaCas9
    "Output_evoCas9",         # evoCas9
)

#: 모델이 받는 윈도우 길이. `pam_scanner.CAS_CONFIGS`의 SpCas9 `model_input_len`과,
#: export manifest의 입력 모양 [None, 30, 4]과 같아야 한다.
SPCAS9_WINDOW_LENGTH = 30


class NonNTSpCas9Predictor(BasePredictor):
    """9-Hydra v2 CNN+LSTM export. 30bp 윈도우 하나 → 변종 9개의 원출력.

    `model_input_seq`를 그대로 원핫 [30, 4]로 넣는다. 채널 순서는 A·C·G·T이고
    (`non_nt_runtime.one_hot`), export manifest의 `input.channels`와 같다.

    원출력은 **공개 효율(%)÷100 비율**이다 (출처는 `predict.NON_NT_REGISTRY`의 주석).
    그래서 spec의 output_mode는 "raw"이고, 화면 점수는 0~1로 자른 뒤 ×100한 효율(%)이다.
    변환은 NT와 같은 `predict._to_display_score` 하나를 지난다.

    적재는 지연된다. 이 Cas의 사이트가 하나도 없으면 checkpoint를 읽지 않는다.
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self._runtime = None

    # ── 식별 ─────────────────────────────────────────────────────
    @property
    def _checkpoint(self) -> str:
        return self.spec.paths[0]

    def model_identity(self) -> ModelIdentity:
        """고정된 sha256으로 식별한다. 모델을 올리지 않는다.

        `model_source.NON_NT_FILES`가 이 경로의 sha256을 고정하고,
        `model_source.resolve`가 적재 때 실제 바이트를 그 값과 대조한다. 그래서
        해시를 다시 계산하지 않아도 식별이 입증된다 — 어긋난 파일로는 추론이 안 된다.
        """
        pinned = model_source.NON_NT_FILES.get(self._checkpoint, {})
        return ModelIdentity(
            family=self.spec.family, cas=self.spec.cas, name=self.spec.label,
            version=version_label(self.spec, pinned.get("sha256", "")),
            raw_unit=self.spec.raw_unit, checkpoint_sha256=pinned.get("sha256", ""))

    # ── 적재 ─────────────────────────────────────────────────────
    def _ensure_loaded(self):
        if self._runtime is not None:
            return self._runtime
        from non_nt_runtime import CHECKPOINTS, NonNTRuntime

        key = next((k for k, path in CHECKPOINTS.items() if path == self._checkpoint), None)
        if key is None:
            raise KeyError(
                f"채택 checkpoint 목록에 없는 경로입니다: {self._checkpoint} "
                "(non_nt_runtime.CHECKPOINTS)")
        logger.info("Loading non-NT weights: %s/%s", self.spec.family, self.spec.cas)
        runtime = NonNTRuntime.load(key)
        self._check_structure(runtime)
        self._runtime = runtime
        return runtime

    def _check_structure(self, runtime) -> None:
        """적재한 그래프가 이 예측기가 가정한 모양·헤드 순서인지.

        입력 모양과 헤드 순서는 예측값의 **의미**를 정한다. 어긋난 채로 돌리면
        오류 없이 변종만 뒤바뀐 표가 나온다.
        """
        expected_shape = [(SPCAS9_WINDOW_LENGTH, 4)]
        if runtime.input_shapes != expected_shape:
            raise RuntimeError(
                f"{self.spec.label} 입력 모양이 {expected_shape}가 아니라 "
                f"{runtime.input_shapes}입니다")
        if tuple(runtime.head_names) != SPCAS9_HEAD_LAYERS:
            raise RuntimeError(
                f"{self.spec.label} 출력 순서가 변종 순서와 다릅니다.\n"
                f"  기대: {list(SPCAS9_HEAD_LAYERS)}\n"
                f"  실제: {list(runtime.head_names)}")
        identity = self.model_identity()
        if identity.checkpoint_sha256 and runtime.checkpoint_sha256 != identity.checkpoint_sha256:
            raise RuntimeError(
                f"{self.spec.label} checkpoint 해시가 고정값과 다릅니다: "
                f"{runtime.checkpoint_sha256}")

    @property
    def runtime(self):
        return self._ensure_loaded()

    # ── 예측 ─────────────────────────────────────────────────────
    def predict_batch(self, sites: Sequence[CandidateSite],
                      on_batch: Callable | None = None) -> list[VariantResult]:
        """사이트마다 **인식하는 변종**의 행을 만든다 (NT 다중헤드와 같은 규칙).

        사이트가 없으면 checkpoint를 읽지 않고 바로 끝낸다 — 서열에 SpCas9 PAM이
        없으면 이 모델은 메모리에 올라가지 않는다 (#196).
        """
        if not sites:
            return []
        from non_nt_runtime import one_hot

        windows = np.stack([one_hot(site.model_input_seq) for site in sites])
        raw = self.runtime.forward_all([windows], on_batch=on_batch)
        identity = self.model_identity()

        results = []
        for index, variant in enumerate(SPCAS9_VARIANTS):
            for site, row in zip(sites, raw):
                if variant_compatible(variant, site.pam_seq):
                    results.append(
                        variant_result(self.spec, site, variant, float(row[index]), identity))
        return results

    # ── 설명 ─────────────────────────────────────────────────────
    def explain_heads(self, site: CandidateSite) -> list[str]:
        """이 사이트에서 설명 대상이 될 수 있는 변종. NT 다중헤드와 같은 목록이다.

        예측 행이 있는 변종과 같은 집합이라, XAI의 사이트·헤드 선택이 이 계열의
        예측 행을 그대로 가리킨다 (#196). 실제 설명 계산은 #199가 붙인다.
        """
        return [v for v in SPCAS9_VARIANTS if variant_compatible(v, site.pam_seq)]

    def explain_handle(self, head: str):
        raise NotImplementedError(
            "non-NT SpCas9의 IG·SHAP 계산은 아직 연결되지 않았습니다 (#199). "
            "#196은 예측 결과와 기존 보기·CSV의 연결까지입니다.")
