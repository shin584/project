"""설명 결과의 계산 식별 (#208). XAI 메모리 캐시·사전계산 SHAP 캐시의 재사용은
이 식별이 완전히 같을 때만 허용한다.

왜 필요한가
  두 캐시의 예전 키는 ("A/Cas12a", head, 서열, 설정)뿐이었다. 같은 이름 아래 checkpoint를
  바꾸거나, 다른 계열(NT/non-NT) 모델이 같은 이름을 쓰거나, 설명 프로토콜(단위·기준
  입력 정의)이 바뀌어도 예전 결과가 그대로 나왔다. 이름만으로는 어느 계산인지
  입증할 수 없다.

무엇이 식별인가
  모델 쪽 (이 객체)   : 계열, 모델 이름, checkpoint 바이트의 sha256, 설명 프로토콜,
                        결과에 영향을 주는 런타임 설정(dtype 등).
  요청 쪽 (캐시 키)   : 기법, 헤드, 윈도우, 기법별 설정, 배경·부분집합 seed.
  두 쪽을 함께 맞춰야 적중이다. 모델 쪽 식별은 `verified`일 때만 입증된 것으로 본다.
  입증할 수 없으면 두 캐시는 적중으로 처리하지 않고 새로 계산한다.

프로토콜 문자열은 계열이 소유한다. 계산 정의(단위 교정 등)가 바뀌면 그 계열이
프로토콜을 올리고, 그러면 이전 결과는 메모리·디스크 어디서도 재사용되지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

#: 식별이 알려진 계열. 새 계열은 여기에 더해야 입증된 식별로 인정된다.
FAMILIES = ("nt", "non_nt")

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ExplanationIdentity:
    """한 모델·한 설명 프로토콜의 계산 식별. 해시 가능하므로 캐시 키에 그대로 넣는다."""

    family: str                       # "nt" | "non_nt"
    model: str                        # 사람이 읽는 이름 ("A/Cas12a" 등)
    checkpoints: tuple[str, ...]      # 모델 파일 바이트의 sha256 (소문자 hex)
    protocol: str                     # 설명 프로토콜 식별자
    configuration: str = ""           # 결과에 영향을 주는 런타임 설정 (정규 JSON)

    @property
    def verified(self) -> bool:
        """식별을 입증할 재료가 모두 있는가. 하나라도 비면 캐시 적중을 허용하지 않는다."""
        return (self.family in FAMILIES
                and bool(self.model) and bool(self.protocol)
                and bool(self.checkpoints)
                and all(isinstance(c, str) and _SHA256.fullmatch(c) for c in self.checkpoints))

    def as_dict(self) -> dict:
        """사전계산 SHAP 캐시 키·파일에 남길 정규 표현."""
        data = asdict(self)
        data["checkpoints"] = list(self.checkpoints)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ExplanationIdentity:
        """as_dict의 역. 저장된 키를 읽을 때 쓴다 (항목이 빠지면 KeyError)."""
        return cls(family=str(data["family"]), model=str(data["model"]),
                   checkpoints=tuple(str(c) for c in data["checkpoints"]),
                   protocol=str(data["protocol"]),
                   configuration=str(data.get("configuration", "")))


def verified(candidate) -> ExplanationIdentity | None:
    """입증된 식별이면 그것을, 아니면 None. 문자열 model_key 같은 이름은 입증이 아니다."""
    if isinstance(candidate, ExplanationIdentity) and candidate.verified:
        return candidate
    return None


def file_sha256(path) -> str:
    """파일 바이트의 sha256. 경로가 같아도 내용이 다르면 다른 checkpoint다."""
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def from_files(family: str, model: str, paths: Iterable, protocol: str,
               configuration: dict) -> ExplanationIdentity:
    """실제 파일 바이트를 해시해 식별을 만든다. 경로 이름이 아니라 내용이 식별이다.
    configuration은 정규 JSON으로 고정해 같은 설정이 늘 같은 식별이 되게 한다."""
    return ExplanationIdentity(family=family, model=model,
                               checkpoints=tuple(file_sha256(p) for p in paths),
                               protocol=protocol,
                               configuration=json.dumps(configuration, sort_keys=True))
