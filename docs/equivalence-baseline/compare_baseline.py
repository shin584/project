"""#181 동등성 비교 — 데스크톱 화면의 산출물을 Streamlit 기준과 대조한다. (게이트 2에서 쓴다)

기준은 `capture_baseline.py`가 뜬 이 폴더의 파일들이다. 데스크톱 쪽에서 같은 고정 입력으로
같은 이름의 파일을 만들어 한 폴더에 넣고 그 폴더를 인자로 넘긴다.

    python docs/equivalence-baseline/compare_baseline.py <데스크톱_산출물_폴더>

비교 방식
  XAI 배열  np.allclose(rtol=1e-5, atol=1e-8). SHAP은 사전계산 캐시를 공유하므로 완전히 같아야 한다
  CSV       바이트 단위로 먼저 보고, 다르면 행 단위 diff를 보여준다
  후보 목록  candidates.json을 재귀 비교 (정렬 순서와 반올림 전 값까지 본다)

종료 코드 0이면 전부 일치, 1이면 차이가 있다. 차이는 모두 출력한다 (첫 차이에서 멈추지 않는다).
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

RTOL, ATOL = 1e-5, 1e-8

# npz 안에서 수치로 비교할 배열들
ARRAY_KEYS = ("ig_attribution", "shap_values",
              "rollout_position_profile", "rollout_token_matrix")


def _fail(msgs: list[str], text: str) -> None:
    msgs.append(text)
    print(f"  [다름] {text}")


def compare_csv(base: Path, other: Path, msgs: list[str]) -> None:
    name = base.name
    print(f"CSV {name}")
    if not other.exists():
        _fail(msgs, f"{name}: 데스크톱 쪽 파일이 없다")
        return
    b, o = base.read_bytes(), other.read_bytes()
    if b == o:
        print(f"  같음 ({len(b)} bytes)")
        return
    bl = b.decode("utf-8-sig").splitlines()
    ol = o.decode("utf-8-sig").splitlines()
    diff = list(difflib.unified_diff(bl, ol, "streamlit", "desktop", lineterm="", n=0))
    _fail(msgs, f"{name}: 내용이 다르다 ({len(b)} vs {len(o)} bytes)")
    for line in diff[:40]:
        print(f"    {line}")
    if len(diff) > 40:
        print(f"    ... diff {len(diff)}줄 중 40줄만 표시")


def compare_json(base: Path, other: Path, msgs: list[str]) -> None:
    name = base.name
    print(f"JSON {name}")
    if not other.exists():
        _fail(msgs, f"{name}: 데스크톱 쪽 파일이 없다")
        return
    a = json.loads(base.read_text(encoding="utf-8"))
    b = json.loads(other.read_text(encoding="utf-8"))

    def walk(x, y, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for k in dict.fromkeys(list(x) + list(y)):
                if k not in x:
                    _fail(msgs, f"{name}{path}.{k}: 기준에 없는 항목이 생겼다")
                elif k not in y:
                    _fail(msgs, f"{name}{path}.{k}: 데스크톱에 없다")
                else:
                    walk(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                _fail(msgs, f"{name}{path}: 길이가 다르다 ({len(x)} vs {len(y)})")
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{path}[{i}]")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                and not isinstance(x, bool) and not isinstance(y, bool):
            if not np.isclose(x, y, rtol=RTOL, atol=ATOL):
                _fail(msgs, f"{name}{path}: {x} vs {y}")
        elif x != y:
            _fail(msgs, f"{name}{path}: {x!r} vs {y!r}")

    walk(a, b, "")
    print("  비교 끝")


def compare_npz(base: Path, other: Path, msgs: list[str]) -> None:
    name = base.name
    print(f"XAI {name}")
    if not other.exists():
        _fail(msgs, f"{name}: 데스크톱 쪽 파일이 없다")
        return
    with np.load(base) as a, np.load(other) as b:
        if str(a["window_seq"]) != str(b["window_seq"]):
            _fail(msgs, f"{name}: 설명 대상 윈도우 서열이 다르다")
            return
        for key in ARRAY_KEYS:
            if key not in b:
                _fail(msgs, f"{name}:{key} 배열이 없다")
                continue
            x, y = np.asarray(a[key], float), np.asarray(b[key], float)
            if x.shape != y.shape:
                _fail(msgs, f"{name}:{key} 모양이 다르다 {x.shape} vs {y.shape}")
                continue
            if np.allclose(x, y, rtol=RTOL, atol=ATOL):
                print(f"  {key}: allclose (최대 차 {np.max(np.abs(x - y)):.3e})")
            else:
                _fail(msgs, f"{name}:{key} allclose 실패 "
                            f"(최대 차 {np.max(np.abs(x - y)):.3e})")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    other_dir = Path(argv[1]).resolve()
    if not other_dir.is_dir():
        print(f"폴더가 없다: {other_dir}")
        return 2

    msgs: list[str] = []
    compare_json(HERE / "candidates.json", other_dir / "candidates.json", msgs)
    compare_csv(HERE / "crispr_prediction_results.csv",
                other_dir / "crispr_prediction_results.csv", msgs)
    for npz in sorted(HERE.glob("xai-*.npz")):
        compare_npz(npz, other_dir / npz.name, msgs)

    print()
    if msgs:
        print(f"차이 {len(msgs)}건 — 동등성 미달")
        return 1
    print("전부 일치")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
