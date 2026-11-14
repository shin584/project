"""SHAP 사전계산 저장소. kernel_shap이 계산 전에 먼저 조회한다.

화면은 사전계산 값을 받았는지 모른다. 조회 계층은 코어(xai/explain.py) 안쪽에만 있다.

캐시 키 계약
  키는 (model_key, head, window_seq, nsamples, n_background) 다섯 개 전부다.
  SHAP 값은 샘플 수·배경 수에 따라 달라지므로, 서열만으로 키를 잡으면
  사용자가 샘플 수를 바꿔도 예전 값이 나온다. xai/view.py의 compute_shap
  (@st.cache_data)과 같은 다섯 값을 쓴다.

저장 형식은 data_source.py가 Model B를 읽는 방식과 같은 npz다.
  data/precomputed_shap/<키 해시>.npz
"""

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "precomputed_shap"


def shap_cache_key(model_key: str, head: str, window_seq: str,
                   nsamples: int, n_background: int) -> str:
    """사전계산 SHAP의 캐시 키. 순수 함수 — 다섯 값 중 하나만 달라도 다른 키가 나온다."""
    return "|".join([str(model_key), str(head), str(window_seq).upper(),
                     str(int(nsamples)), str(int(n_background))])


def _path(key: str, root: Path) -> Path:
    return Path(root) / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def load(key: str, root: Path | str | None = None):
    """적중하면 SHAPResult, 미적중이면 None. 파일이 깨져 있어도 None(계산으로 넘어간다)."""
    from xai.explain import SHAPResult

    path = _path(key, DEFAULT_DIR if root is None else root)
    if not path.exists():
        return None
    try:
        with np.load(path) as f:
            if str(f["key"]) != key:          # 해시 충돌 방지
                return None
            return SHAPResult(values=np.asarray(f["values"], dtype=np.float64),
                              base_value=float(f["base_value"]),
                              pred=float(f["pred"]),
                              additivity_error=float(f["additivity_error"]))
    except Exception:
        return None


def save(key: str, result, root: Path | str | None = None) -> Path:
    path = _path(key, DEFAULT_DIR if root is None else root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, key=np.array(key), values=np.asarray(result.values, dtype=np.float64),
             base_value=result.base_value, pred=result.pred,
             additivity_error=result.additivity_error)
    return path
