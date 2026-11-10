"""
read_header.py
──────────────
safetensors 파일의 '헤더'만 읽어 모델 구조를 파악한다.

safetensors는 파일 맨 앞에 텐서 이름·자료형·모양이 JSON으로 들어 있다.
그 부분만 읽으므로 파일이 1.9GB여도 몇 KB만 읽고 바로 끝난다.
추가 패키지도, 관리자 권한도 필요 없다.

사용법:
    python read_header.py models\\SpCas9.safetensors models\\Cas12a.safetensors

인자를 안 주면 models/ 아래 .safetensors를 모두 찾아 읽는다:
    python read_header.py
"""

import glob
import json
import os
import struct
import sys

PREVIEW = 15   # 앞뒤로 보여줄 텐서 개수


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def describe(path: str) -> None:
    print("=" * 72)
    print(path)
    size = os.path.getsize(path)
    print(f"  파일 크기: {size:,} 바이트 ({size / 1024**3:.2f} GB)")

    try:
        header = read_header(path)
    except Exception as e:
        print(f"  !! 헤더를 읽지 못했습니다: {type(e).__name__}: {e}")
        return

    tensors = []
    for name, info in header.items():
        if name == "__metadata__":
            print(f"  [metadata] {info}")
            continue
        tensors.append((name, info.get("dtype"), info.get("shape", [])))

    total = 0
    for _, _, shape in tensors:
        cnt = 1
        for d in shape:
            cnt *= d
        total += cnt

    print(f"  텐서 {len(tensors)}개, 파라미터 약 {total:,}개")

    # 이름 앞부분(접두사)별 개수 — 백본 종류를 가장 빠르게 알려준다
    prefixes = {}
    for name, _, _ in tensors:
        head = name.split(".")[0]
        prefixes[head] = prefixes.get(head, 0) + 1
    print("  최상위 접두사별 텐서 수:")
    for k, v in sorted(prefixes.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v}개")

    print(f"  --- 앞 {PREVIEW}개 ---")
    for name, dtype, shape in tensors[:PREVIEW]:
        print(f"    {name}: {dtype} {shape}")
    print(f"  --- 뒤 {PREVIEW}개 ---")
    for name, dtype, shape in tensors[-PREVIEW:]:
        print(f"    {name}: {dtype} {shape}")


paths = sys.argv[1:]
if not paths:
    base = os.path.dirname(os.path.abspath(__file__))
    paths = sorted(glob.glob(os.path.join(base, "models", "*.safetensors")))
    if not paths:
        print("models/ 아래에 .safetensors 파일이 없습니다. 경로를 인자로 주세요.")
        sys.exit(1)

for p in paths:
    describe(p)
