"""코어의 공개 표면. 화면(Streamlit·데스크톱)과 장래의 HTTP API는 여기 있는 것만 쓴다.

파일은 제자리에 두고 이름만 재노출한다 (ADR 0001: 파일을 옮기면 팀원이 작성한
화면 파일의 import 경로가 바뀐다). 여기에 없는 이름은 코어 내부 사정이다.

이름은 처음 쓰일 때 import한다. 차트 모듈은 shap·matplotlib 때문에 import에만
수 초가 걸리므로, 탐색·예측만 하는 쪽이 그 비용을 치르지 않게 한다.
`import core`는 UI 프레임워크(streamlit)도 모델 라이브러리(torch 등)도 끌어오지 않는다.

이름을 추가할 때는 `_EXPORTS`와 맨 아래 TYPE_CHECKING 블록을 함께 고친다
(둘이 어긋나면 test_core_facade가 실패한다).

일부러 빼 둔 것:
- "Model B(SaCas9)" 사전계산 산출물(`xai/data_source.py`, `xai/charts_modelb.py`).
  화면에서 내려가 있고 코드와 데이터만 남아 있다 (CONTEXT.md).
- `xai/view.py`. streamlit 화면이며 코어가 아니다.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# 공개 이름 → (원래 모듈, 그 모듈 안의 이름)
_EXPORTS: dict[str, tuple[str, str]] = {
    # 탐색: 81bp 서열에서 사이트를 찾는다
    "CasConfig": ("pam_scanner", "CasConfig"),
    "CAS_CONFIGS": ("pam_scanner", "CAS_CONFIGS"),
    "SPCAS9_VARIANTS": ("pam_scanner", "SPCAS9_VARIANTS"),
    "WINDOW_LAYOUT": ("pam_scanner", "WINDOW_LAYOUT"),
    "CandidateSite": ("pam_scanner", "CandidateSite"),
    "VariantResult": ("pam_scanner", "VariantResult"),
    "ScanResult": ("pam_scanner", "ScanResult"),
    "validate_sequence": ("pam_scanner", "validate_sequence"),
    "scan_all": ("pam_scanner", "scan_all"),
    "dna_to_rna": ("pam_scanner", "dna_to_rna"),
    "reverse_complement": ("pam_scanner", "reverse_complement"),
    "print_scan_report": ("pam_scanner", "print_scan_report"),
    # 예측: 연동된 모델 A로 사이트를 점수화한다
    "ModelSpec": ("predict", "ModelSpec"),
    "MODEL_REGISTRY": ("predict", "MODEL_REGISTRY"),
    "BasePredictor": ("predict", "BasePredictor"),
    "ExplainHandle": ("predict", "ExplainHandle"),
    "load_predictors": ("predict", "load_predictors"),
    "connected_models": ("predict", "connected_models"),
    "run_prediction": ("predict", "run_prediction"),
    # 설명(XAI): 윈도우 좌표계의 기여도
    "IGResult": ("xai.explain", "IGResult"),
    "SHAPResult": ("xai.explain", "SHAPResult"),
    "RolloutResult": ("xai.explain", "RolloutResult"),
    "integrated_gradients": ("xai.explain", "integrated_gradients"),
    "kernel_shap": ("xai.explain", "kernel_shap"),
    "attention_rollout": ("xai.explain", "attention_rollout"),
    # 사전계산 SHAP 캐시 (키는 설정 전체를 포함한다 — ADR 0001).
    # 원래 이름 load/save는 facade 밖에서는 무엇을 읽고 쓰는지 드러나지 않아 이름을 붙인다.
    "shap_cache_key": ("xai.shap_cache", "shap_cache_key"),
    "load_cached_shap": ("xai.shap_cache", "load"),
    "save_cached_shap": ("xai.shap_cache", "save"),
    # 차트: 위치별 배열 → matplotlib Figure
    "SEGMENT_COLORS": ("xai.charts", "SEGMENT_COLORS"),
    "segment_of": ("xai.charts", "segment_of"),
    "attribution_logo": ("xai.charts", "attribution_logo"),
    "shap_bars": ("xai.charts", "shap_bars"),
    "shap_waterfall": ("xai.charts", "shap_waterfall"),
    "rollout_profile": ("xai.charts", "rollout_profile"),
    "rollout_token_heatmap": ("xai.charts", "rollout_token_heatmap"),
}

__all__ = list(_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'core' has no public name {name!r}") from None
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # 다음부터는 __getattr__을 거치지 않는다
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from pam_scanner import (
        CAS_CONFIGS, SPCAS9_VARIANTS, WINDOW_LAYOUT, CandidateSite, CasConfig,
        ScanResult, VariantResult, dna_to_rna, print_scan_report, reverse_complement,
        scan_all, validate_sequence,
    )
    from predict import (
        MODEL_REGISTRY, BasePredictor, ExplainHandle, ModelSpec, connected_models,
        load_predictors, run_prediction,
    )
    from xai.charts import (
        SEGMENT_COLORS, attribution_logo, rollout_profile, rollout_token_heatmap,
        segment_of, shap_bars, shap_waterfall,
    )
    from xai.explain import (
        IGResult, RolloutResult, SHAPResult, attention_rollout, integrated_gradients,
        kernel_shap,
    )
    from xai.shap_cache import load as load_cached_shap
    from xai.shap_cache import save as save_cached_shap
    from xai.shap_cache import shap_cache_key
