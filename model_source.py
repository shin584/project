"""
model_source.py
───────────────
모델 파일을 어디서 가져올지 한 곳에서 관리한다.

동작 순서
  1. `models/` 아래에 파일이 이미 있으면 그걸 쓴다. (한 번 받으면 다시 안 받는다)
  2. 없으면 DRIVE_FILES에 등록된 구글 드라이브 링크에서 내려받아 같은 경로에 저장한다.
  3. 둘 다 안 되면 어떤 파일이 없는지 알려주며 멈춘다.

바꿀 곳은 아래 DRIVE_FILES 하나뿐이다. predict.py는 수정할 필요가 없다.

필요한 패키지 (관리자 권한 필요 없음):
    pip install gdown
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════
# 여기만 채우면 된다
# ════════════════════════════════════════════════
#
# 형식)  "저장할 상대 경로": "구글 드라이브 공유 링크 또는 파일 ID"
#
# · 왼쪽은 predict.py의 MODEL_REGISTRY에 적은 경로와 **똑같이** 맞춘다.
# · 오른쪽은 아래 셋 중 아무 형태나 된다.
#     https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing
#     https://drive.google.com/open?id=1AbCdEfGhIjKlMnOp
#     1AbCdEfGhIjKlMnOp
# · 공유 설정은 "링크가 있는 모든 사용자"로 바꿔 둬야 받아진다.
#
DRIVE_FILES: dict[str, str] = {
    # 예시 (실제 링크로 바꿔 주세요)
    # "models/drive/SpCas9.keras": "https://drive.google.com/file/d/.../view?usp=sharing",
    # "models/drive/Cas12a.keras": "https://drive.google.com/file/d/.../view?usp=sharing",
}

# 폴더 하나에 모델이 모여 있다면 폴더 링크를 여기에 적어도 된다.
# 폴더 안의 파일 이름이 MODEL_REGISTRY 경로의 파일 이름과 같아야 한다.
DRIVE_FOLDER: str | None = None
DRIVE_FOLDER_DEST = "models/drive"   # 폴더를 받을 위치


# ════════════════════════════════════════════════
# 내부 구현
# ════════════════════════════════════════════════

def _abs(rel_path: str) -> str:
    return os.path.join(BASE_DIR, rel_path)


def _require_gdown():
    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "구글 드라이브에서 모델을 받으려면 gdown이 필요합니다.\n"
            "  pip install gdown\n"
            "(관리자 권한은 필요하지 않습니다.)"
        ) from e
    return gdown


def _download_file(ref: str, dest: str) -> None:
    gdown = _require_gdown()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # id만 준 경우에도 받아지도록 fuzzy=True로 둔다.
    url = ref if ref.startswith("http") else f"https://drive.google.com/uc?id={ref}"
    print(f"[model_source] 구글 드라이브에서 내려받는 중 → {dest}")
    gdown.download(url, dest, quiet=False, fuzzy=True)
    if not os.path.exists(dest):
        raise RuntimeError(
            f"다운로드가 끝났는데 파일이 없습니다: {dest}\n"
            "공유 설정이 '링크가 있는 모든 사용자'인지, 링크가 맞는지 확인해 주세요."
        )


def _download_folder() -> None:
    gdown = _require_gdown()
    dest = _abs(DRIVE_FOLDER_DEST)
    os.makedirs(dest, exist_ok=True)
    print(f"[model_source] 구글 드라이브 폴더를 내려받는 중 → {dest}")
    gdown.download_folder(DRIVE_FOLDER, output=dest, quiet=False, use_cookies=False)


def resolve(rel_path: str) -> str:
    """모델 파일의 로컬 절대 경로를 돌려준다. 없으면 드라이브에서 받아 온다."""
    local = _abs(rel_path)
    if os.path.exists(local):
        return local

    if rel_path in DRIVE_FILES:
        _download_file(DRIVE_FILES[rel_path], local)
        return local

    if DRIVE_FOLDER:
        # 폴더를 한 번 받아 둔 뒤, 그 안에서 같은 파일 이름을 찾는다.
        folder_dest = _abs(DRIVE_FOLDER_DEST)
        name = os.path.basename(rel_path)
        if not os.path.exists(os.path.join(folder_dest, name)):
            _download_folder()
        for root, _, files in os.walk(folder_dest):
            if name in files:
                return os.path.join(root, name)

    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없습니다: {rel_path}\n"
        f"  · 기대한 위치: {local}\n"
        "  · model_source.py의 DRIVE_FILES에 이 경로와 구글 드라이브 링크를 등록하거나,\n"
        "    해당 위치에 파일을 직접 넣어 주세요."
    )


def status() -> list[tuple[str, bool, str]]:
    """(상대경로, 로컬에 있는지, 출처) 목록. 화면에서 연동 상태를 보여줄 때 쓴다."""
    from predict import MODEL_REGISTRY

    out = []
    for spec in MODEL_REGISTRY.values():
        for p in spec.paths:
            exists = os.path.exists(_abs(p))
            if exists:
                origin = "로컬 파일"
            elif p in DRIVE_FILES:
                origin = "구글 드라이브 (아직 안 받음)"
            elif DRIVE_FOLDER:
                origin = "구글 드라이브 폴더 (아직 안 받음)"
            else:
                origin = "미등록"
            out.append((p, exists, origin))
    return out
