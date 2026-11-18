"""SHAP 사전계산 저장소. kernel_shap이 계산 전에 먼저 조회한다.

화면은 사전계산 값을 받았는지 모른다. 조회 계층은 코어(xai/explain.py) 안쪽에만 있다.

캐시 키 계약 (#208)
  키는 계산 식별 전부다: 모델 식별(계열·모델·checkpoint sha256·설명 프로토콜·런타임
  설정, `xai.identity.ExplanationIdentity`) + head + window_seq + nsamples +
  n_background + background_seed + subset_seed. 정규 JSON 문자열로 만든다.

  model_key로 문자열("A/Cas12a")을 주면 키는 만들어지지만 **입증되지 않은 키**다.
  그런 키는 조회해도 항상 미적중이고, 저장은 거부한다. 이름만으로는 어느
  checkpoint·프로토콜의 결과인지 알 수 없기 때문이다.

파일 형식
  data/precomputed_shap/<키 해시>.npz. `schema`·`key` 필드로 출처를 남긴다.
  `schema`가 없거나 다른 파일(#152 시절의 legacy 산출물 포함), 저장된 키가 요청과
  다른 파일, 값의 모양·유한성이 맞지 않는 파일은 거부한다 — 거부는 미적중이므로
  호출자는 새로 계산한다. legacy 파일을 지우거나 다시 만들지는 않는다
  (NT 단위 교정과 재생성은 #201 소관).
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from xai import identity as _identity

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "precomputed_shap"

#: 출처를 갖춘 저장 형식의 판. 이 값이 없는 파일은 출처 없는 legacy 산출물이다.
SCHEMA = "shap-precomputed-v2"


def shap_cache_key(model_key, head: str, window_seq: str,
                   nsamples: int, n_background: int,
                   background_seed: int = 0, subset_seed: int = 0) -> str:
    """사전계산 SHAP의 캐시 키. 순수 함수 — 식별 항목 하나만 달라도 다른 키가 나온다.

    model_key가 입증된 `ExplanationIdentity`가 아니면 키의 identity는 null이고,
    그 키는 조회·저장에 쓰일 수 없다.
    """
    ident = _identity.verified(model_key)
    record = {
        "identity": ident.as_dict() if ident else None,
        "model_key": None if ident else str(model_key),
        "head": str(head),
        "window_seq": str(window_seq).upper(),
        "nsamples": int(nsamples),
        "n_background": int(n_background),
        "background_seed": int(background_seed),
        "subset_seed": int(subset_seed),
    }
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def _parse(key) -> dict | None:
    """입증된 키면 그 레코드, 아니면 None."""
    try:
        record = json.loads(key)
        ident = _identity.ExplanationIdentity.from_dict(record["identity"])
    except Exception:
        return None
    return record if ident.verified else None


def _path(key: str, root: Path | str) -> Path:
    return Path(root) / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def load(key, root: Path | str | None = None):
    """출처가 요청과 완전히 일치하면 SHAPResult, 아니면 None(새로 계산한다)."""
    from xai.explain import SHAPResult

    record = _parse(key)
    if record is None:
        return None
    path = _path(key, DEFAULT_DIR if root is None else root)
    if not path.exists():
        return None
    try:
        with np.load(path) as f:
            if "schema" not in f.files or str(f["schema"]) != SCHEMA:
                return None                   # 출처 없는 legacy 산출물
            if str(f["key"]) != key:          # 다른 계산의 결과 (또는 해시 충돌)
                return None
            values = np.asarray(f["values"], dtype=np.float64)
            scalars = [float(f[name]) for name in ("base_value", "pred", "additivity_error")]
    except Exception:
        return None
    if values.shape != (len(record["window_seq"]),) or not (
            np.isfinite(values).all() and np.isfinite(scalars).all()):
        return None
    base_value, pred, additivity_error = scalars
    return SHAPResult(values=values, base_value=base_value, pred=pred,
                      additivity_error=additivity_error,
                      background_seed=record["background_seed"],
                      subset_seed=record["subset_seed"])


def save(key, result, root: Path | str | None = None) -> Path:
    """입증된 키로만 저장한다. 출처 없는 산출물을 새로 만들지 않기 위해서다."""
    record = _parse(key)
    if record is None:
        raise ValueError("계산 식별(checkpoint·계열·프로토콜)을 입증할 수 없는 키로는 "
                         "사전계산 SHAP을 저장하지 않습니다.")
    seeds = (getattr(result, "background_seed", None), getattr(result, "subset_seed", None))
    if seeds != (record["background_seed"], record["subset_seed"]):
        raise ValueError(f"결과의 seed {seeds}가 키의 seed와 다릅니다.")
    path = _path(key, DEFAULT_DIR if root is None else root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, schema=np.array(SCHEMA), key=np.array(key),
             values=np.asarray(result.values, dtype=np.float64),
             base_value=result.base_value, pred=result.pred,
             additivity_error=result.additivity_error)
    return path
