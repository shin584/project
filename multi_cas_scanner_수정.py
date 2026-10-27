"""
[추가된 사항 - v2]
  ① has_spcas9_pam / has_sacas9_pam / has_cas12a_pam  개별 탐색 함수
  ② build_score_dict()  결과 통합 파이프라인 (딕셔너리 형태, ×100 퍼센트)
  ③ app.py(Streamlit)와 연동되는 구조로 정리
"""

import math  # 거리패널티
import os
import re  # 정규표현식
from dataclasses import (  # 메모리 효율과 가독성을 위해 정형화된 데이터 저장소 생성
    dataclass,
    field,
)

import numpy as np  # 데이터 행렬화(One-Hot Encoding)
import tensorflow as tf  # 모델 파일을 불러오고 입력된 서열 데이터에 대한 절단 효율 점수 예측

# ════════════════════════════════════════════════
# 섹션 1. 설정 (PAM 추가/수정은 여기만 건드리면 됨)
# ════════════════════════════════════════════════


@dataclass(frozen=True)  # 객체 읽기 전용, 해시 가능
class CasConfig:
    """
    Cas 단백질 한 종류의 PAM/절단 규칙 정의.
    새 단백질 추가 시 이 클래스 인스턴스만 하나 더 만들면 됨.
    """

    name: str  # 유전자 가위 이름
    pam_iupac: str  # PAM 서열 규칙
    guide_len: int  # 모델이 인식할 서열의 길이 결정
    pam_position: str  # '3prime' | '5prime'/ PAM 위치
    cut_offset: int  # DNA가 잘리는 위치
    model_path: str  # 효율 예측할 때 어떤 모델 파일을 사용할 것인지 정해주는 경로


# 3가지 유전자 가위의 구체적인 명세서
CAS_CONFIGS: list[CasConfig] = [
    CasConfig(
        name="SpCas9",
        pam_iupac="NGG",
        guide_len=20,
        pam_position="3prime",
        cut_offset=-3,
        model_path="path_A",
    ),
    CasConfig(
        name="SaCas9",
        pam_iupac="NNGRRT",
        guide_len=21,
        pam_position="3prime",
        cut_offset=-3,
        model_path="path_C",
    ),
    CasConfig(
        name="Cas12a",
        pam_iupac="TTTV",
        guide_len=23,
        pam_position="5prime",
        cut_offset=20,
        model_path="path_B",
    ),
]

# 이후에 추가 모델의 PAM 서열을 인식해야할 경우를 대비해 모든 IUPAC 넣어놓음
_IUPAC_TABLE: dict[str, str] = {
    "A": "A",
    "T": "T",
    "G": "G",
    "C": "C",
    "N": "[ATGC]",
    "R": "[AG]",
    "Y": "[CT]",
    "S": "[GC]",
    "W": "[AT]",
    "K": "[GT]",
    "M": "[AC]",
    "B": "[CGT]",
    "D": "[AGT]",
    "H": "[ACT]",
    "V": "[ACG]",
}

# Explicit v2 output contract; dictionary insertion order has no meaning.
HYDRA_OUTPUT_LAYERS: dict[str, str] = {
    "SpCas9": "Output_SpCas9",
    "SpCas9-NG": "Output_SpCas9_NG",
    "VRQR variant": "Output_VRQR_variant",
    "xCas": "Output_xCas",
    "Sniper-Cas9": "Output_Sniper_Cas9",
    "SpCas9-HF.1": "Output_SpCas9_HF_1",
    "eSpCas9(1.1)": "Output_eSpCas9_1_1_",
    "HypaCas9": "Output_HypaCas9",
    "evoCas9": "Output_evoCas9",
}


# ════════════════════════════════════════════════
# 섹션 2. 데이터 구조
# ════════════════════════════════════════════════


@dataclass(frozen=True)
class PredictionResult:
    """Finite efficiency on the model's original scale, or a failure reason."""

    efficiency: float | None = None
    reason: str | None = None
    efficiencies_by_protein: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.efficiency is None:
            if not self.reason:
                raise ValueError("Unavailable predictions require a reason")
            if self.efficiencies_by_protein:
                raise ValueError("Unavailable predictions cannot contain efficiencies")
        elif not math.isfinite(self.efficiency):
            raise ValueError("Non-finite efficiency")
        elif self.reason is not None:
            raise ValueError("Available predictions cannot have a failure reason")

    @property
    def available(self) -> bool:
        return self.efficiency is not None


@dataclass
class CandidateSite:
    """유효한 절단 후보 부위 하나의 정보"""

    cas_type: str  # 가위 종류
    strand: str  # 가닥 방향(정방향 +, 역방향 -)
    pam_start: int  # PAM 시작점(인덱스 형태)
    cut_pos: int  # 절단 위치
    distance: int  # 변이와의 거리
    guide_seq: str  # gRNA protospacer; distinct from the model input window
    pam_seq: str  # 발견된 PAM 서열
    prediction: PredictionResult = field(
        default_factory=lambda: PredictionResult(reason="Prediction not requested")
    )
    final_score: float | None = None  # 거리패널티 적용 후 % 값
    model_input_seq: str | None = None  # SpCas9: 4 + 20 + 3 + 3 in model orientation
    model_input_reason: str | None = None

    @property
    def raw_score(self) -> float | None:
        return self.prediction.efficiency


@dataclass
class ScanResult:
    """전체 스캔 결과 컨테이너"""

    input_seq: str  # 사용자가 입력한 81bp DNA 서열
    center_idx: int = 40  # 타겟 변위의 위치
    sites: list[CandidateSite] = field(
        default_factory=list
    )  # 발견된 모든 Candidatesite들이 담기는 리스트, 빈 리스트로 시작

    """필터링-특정 가위 종류만 골라냄"""

    def by_cas(self, cas_type: str) -> list[CandidateSite]:
        return [s for s in self.sites if s.cas_type == cas_type]

    """정렬-모든 후보의 점수를 내림차순으로 정렬"""

    def sorted_by_score(self) -> list[CandidateSite]:
        scored = [
            (s.final_score, s)
            for s in self.sites
            if s.prediction.available and s.final_score is not None
        ]
        return [
            site for _, site in sorted(scored, key=lambda pair: pair[0], reverse=True)
        ]

    """요약 출력-가위별로 찾은 PAM 개수"""

    def __repr__(self) -> str:
        counts = {cfg.name: len(self.by_cas(cfg.name)) for cfg in CAS_CONFIGS}
        parts = ", ".join(f"{k}={v}" for k, v in counts.items())
        return f"ScanResult({parts}, total={len(self.sites)})"


# ════════════════════════════════════════════════
# 섹션 3. 유틸리티 함수
# ════════════════════════════════════════════════


def iupac_to_regex(pam: str) -> str:
    """IUPAC PAM 서열 → 파이썬 정규표현식 문자열"""
    try:
        """PAM 서열에서 글자(b)를 하나씩 꺼내서 사전에 대조하여 변환(N->[ATGC]) 후 공백없이 이어붙임"""
        return "".join(_IUPAC_TABLE[b] for b in pam.upper())

    except KeyError as e:
        raise ValueError(f"알 수 없는 IUPAC 코드: {e}")  # 사전에 없는 단어 오류 처리


"""DNA 반대쪽 가닥의 서열을 계산, 유전자 가위가 반대쪽에 붙을 경우 고려"""


def reverse_complement(seq: str) -> str:
    """DNA 서열의 역상보(Reverse Complement) 반환"""
    table = str.maketrans("ACGT", "TGCA")
    return seq.translate(table)[::-1]  # 역순 정렬


def validate_sequence(seq: str) -> str:
    """입력 서열 유효성 검사 (81bp 고정, ATGCN 허용)"""
    seq = seq.upper().strip()  # 대문자 통일, 공백 제거
    if len(seq) != 81:
        raise ValueError(
            f"입력 서열은 81bp여야 합니다. (현재: {len(seq)}bp)\n"
            "변이 위치 기준 앞뒤 40bp씩 총 81bp를 입력해 주세요."
        )
    invalid = set(seq) - set("ATGCN")
    if invalid:
        raise ValueError(f"허용되지 않는 문자 포함: {invalid}")
    return seq


def gaussian_penalty(distance: int, sigma: float = 10.0) -> float:
    """거리 기반 가우시안 가중치 (거리 멀수록 0 수렴)"""
    """sigma 값 줄이면 감점 폭 커짐"""
    return math.exp(-(distance**2) / (2 * sigma**2))


# ════════════════════════════════════════════════
# 섹션 4-1. 개별 PAM 존재 여부 탐색 함수  ★ 신규
# ════════════════════════════════════════════════


def has_spcas9_pam(seq: str) -> bool:
    """
    서열 내 SpCas9 PAM(NGG) 존재 여부 확인.
    양쪽 가닥(정방향 NGG / 역방향에서 NGG) 모두 탐색.

    Parameters
    ----------
    seq : str  검사할 DNA 서열 (길이 무관)

    Returns
    -------
    bool  PAM이 하나라도 있으면 True
    """
    seq = seq.upper()  # 대문자로 통일
    pam_re = re.compile(
        iupac_to_regex("NGG")
    )  # 정방향 탐색(5'->3'), 정규표현식 패턴([ATGC]GG으로 변환
    return bool(pam_re.search(seq)) or bool(
        pam_re.search(reverse_complement(seq))
    )  # 정방향, 역방향 중 한군데라도 발견되면 True 반환


def has_sacas9_pam(seq: str) -> bool:
    """
    서열 내 SaCas9 PAM(NNGRRT) 존재 여부 확인.
    양쪽 가닥 모두 탐색.

    Parameters
    ----------
    seq : str  검사할 DNA 서열

    Returns
    -------
    bool  PAM이 하나라도 있으면 True
    """
    seq = seq.upper()
    pam_re = re.compile(iupac_to_regex("NNGRRT"))
    return bool(pam_re.search(seq)) or bool(pam_re.search(reverse_complement(seq)))


def has_cas12a_pam(seq: str) -> bool:
    """
    서열 내 Cas12a PAM(TTTV, V=A/C/G) 존재 여부 확인.
    양쪽 가닥 모두 탐색.

    Parameters
    ----------
    seq : str  검사할 DNA 서열

    Returns
    -------
    bool  PAM이 하나라도 있으면 True
    """
    seq = seq.upper()
    pam_re = re.compile(iupac_to_regex("TTTV"))
    return bool(pam_re.search(seq)) or bool(pam_re.search(reverse_complement(seq)))


def check_all_pams(seq: str) -> dict[str, bool]:
    """
    세 가지 Cas 단백질의 PAM 존재 여부를 한 번에 반환.

    Returns
    -------
    dict  예: {"SpCas9": True, "SaCas9": False, "Cas12a": True}
    """
    return {
        "SpCas9": has_spcas9_pam(seq),
        "SaCas9": has_sacas9_pam(seq),
        "Cas12a": has_cas12a_pam(seq),
    }


# ════════════════════════════════════════════════
# 섹션 4-2. PAM 탐색 (공통 스캔 로직)
# ════════════════════════════════════════════════


def _calc_cut_pos(
    cfg: CasConfig, pam_start: int, pam_end: int, strand: str, seq_len: int
) -> int:
    """절단 위치(원본 서열 기준) 계산. 역가닥은 좌표 변환."""
    if cfg.pam_position == "3prime":  # SpCas9, SaCas9는 PAM이 가이드 서열 뒤에 존재
        raw_cut = pam_start + cfg.cut_offset
    else:  # Cas12a는 PAM이 가이드 서열 앞에 존재
        raw_cut = pam_end + cfg.cut_offset
    return (
        raw_cut if strand == "+" else seq_len - raw_cut - 1
    )  # 역가닥 좌표를 정방향 서열 기준의 좌표로 복구


def scan_pam(
    seq: str, cfg: CasConfig, center: int = 40, max_dist: int = 15
) -> list[CandidateSite]:
    """단일 CasConfig 기준으로 양쪽 가닥에서 PAM을 탐색."""
    sites: list[CandidateSite] = []
    pam_re = re.compile(iupac_to_regex(cfg.pam_iupac))
    seq_len = len(seq)

    for strand, search_seq in [("+", seq), ("-", reverse_complement(seq))]:
        for m in pam_re.finditer(search_seq):
            pam_start = m.start()
            pam_end = m.end()

            if cfg.pam_position == "3prime":  # SpCas9, SaCas9
                guide_start = (
                    pam_start - cfg.guide_len
                )  # PAM 앞쪽으로 가이드 서열이 붙을 수 있을 만큼의 서열이 존재하는지 확인
                guide_end = pam_start  # 가이드 서열의 끝=PAM 서열의 시작
                if guide_start < 0:
                    continue
            else:  # Cas12a
                guide_start = pam_end
                guide_end = pam_end + cfg.guide_len
                if guide_end > seq_len:
                    continue

            cut_pos = _calc_cut_pos(cfg, pam_start, pam_end, strand, seq_len)
            distance = abs(cut_pos - center)
            if (
                distance > max_dist
            ):  # 절단위치와 변이 위치 사이의 거리 계산해서 유효 서열만 추림
                continue

            site = CandidateSite(
                cas_type=cfg.name,
                strand=strand,
                pam_start=pam_start,
                cut_pos=cut_pos,
                distance=distance,
                guide_seq=search_seq[guide_start:guide_end],
                pam_seq=m.group(),
            )
            if cfg.name == "SpCas9":
                window_start, window_end = guide_start - 4, pam_end + 3
                if window_start < 0 or window_end > seq_len:
                    site.model_input_reason = "SpCas9: insufficient flanking context"
                else:
                    window = search_seq[window_start:window_end]
                    if len(window) != 30 or set(window) - set("ACGT"):
                        site.model_input_reason = (
                            "SpCas9: invalid base or length in 30bp window"
                        )
                    else:
                        site.model_input_seq = window
            sites.append(site)

    return sites


# ════════════════════════════════════════════════
# 섹션 5. 모델 관련
# ════════════════════════════════════════════════

"""텍스트 염기 하나를 4자리의 숫자 리스트에 대흥시킴"""
_ONE_HOT: dict[str, list[int]] = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}


def _one_hot_encode(
    seq: str, target_len: int
) -> np.ndarray | None:  # 결과로 숫재 배열 return, 오류는 None return
    """
    서열 → One-Hot 인코딩 (shape: (1, target_len, 4))

    무결성 방어 ①: 허용 문자 검사
        A/T/G/C/N 이외 문자가 포함되면 인코딩을 거부하고 None 반환.
        (알 수 없는 문자를 [0,0,0,0]으로 조용히 통과시키지 않음)

    무결성 방어 ②: 길이 불일치 검사
        모델이 요구하는 guide_len과 실제 서열 길이가 다르면 None 반환.
    """
    seq = seq.upper()

    # ── 방어 ① 허용 문자 검사 ───────────────────────────────────
    _ALLOWED = set("ATGCN")
    invalid_chars = set(seq) - _ALLOWED
    if invalid_chars:
        print(
            f"  ⚠ [무결성] 허용되지 않는 문자 포함 → 인코딩 거부: {invalid_chars}"
            f"  (서열: {seq[:10]}...)"
        )
        return None

    # ── 방어 ② 길이 불일치 검사 ─────────────────────────────────
    if len(seq) != target_len:  # 가이드 서열 길이 확인
        print(
            f"  ⚠ [무결성] gRNA 길이 불일치 → 인코딩 거부: "
            f"모델 요구={target_len}bp, 실제={len(seq)}bp "
            f"(서열: {seq[:10]}...)"
        )
        return None

    arr = np.array([_ONE_HOT.get(b, [0, 0, 0, 0]) for b in seq], dtype=np.float32)
    """숫자리스트로 바꾼 DNA 서열을 Numpy 행렬로 변환"""
    return arr[np.newaxis, :]


class ModelBundle:
    """세 가지 모델 경로(A/B/C)를 관리하는 번들."""

    def __init__(
        self,
        path_A: str,  # SpCas9 모델 파일 위치
        path_B: str,  # Cas12a 모델 파일 위치
        path_C_dir: str,  # Sacas9는 모델 여러개이므로 파일이 들어있는 폴더 경로 받음
        sacas9_files: list[str] | None = None,
    ):
        if sacas9_files is None:
            sacas9_files = ["SaCas9_1.keras", "SaCas9_2.keras", "SaCas9_3.keras"]
        self._load_errors: dict[str, list[str]] = {}
        self.model_A = self._load_single(path_A, "SpCas9")
        self.model_B = self._load_single(path_B, "Cas12a")
        self.models_C = self._load_ensemble(path_C_dir, sacas9_files, "SaCas9")

        # ── 방어 ③ 모델 로드 후 상태 요약 경고 ─────────────────
        self._warn_missing_models()

    def _warn_missing_models(self) -> None:
        """Summarize load failures; affected families remain unavailable."""
        if self._load_errors:
            print("\n  ══ [무결성 경고] 미로드 모델 목록 ══")
            for family, errors in self._load_errors.items():
                print(f"    ❌ {family}: unavailable — {'; '.join(errors)}")
            print("  ══════════════════════════════════\n")

    def _load_single(self, path: str, label: str) -> tf.keras.Model | None:
        try:
            m = tf.keras.models.load_model(path, compile=False)
            print(f"  ✅ {label} 로드 완료: {path}")
            return m
        except Exception as e:  # noqa: BLE001 — third-party loader errors become unavailable
            self._load_errors.setdefault(label, []).append(
                f"{path}: {type(e).__name__}: {e}"
            )
            print(f"  ❌ {label} 로드 실패: {e}")
            return None

    """여러개의 AI 모델을 앙상블로 불러오기"""

    def _load_ensemble(
        self, directory: str, files: list[str], label: str
    ) -> list[tf.keras.Model]:
        models = []
        for fname in files:
            fpath = os.path.join(directory, fname)
            model = self._load_single(fpath, label)
            if model is not None:
                models.append(model)
        return models

    def predict(self, site: CandidateSite) -> PredictionResult:
        """Return an available efficiency or an explicit family-specific failure."""
        # ── 방어 ① CasConfig 존재 확인 ──────────────────────────
        cfg = next((c for c in CAS_CONFIGS if c.name == site.cas_type), None)
        if cfg is None:
            print(f"  ⚠ [무결성] 알 수 없는 Cas 종류: '{site.cas_type}' → 예측 건너뜀")
            return PredictionResult(reason=f"Unknown Cas family: {site.cas_type}")

        # ── 방어 ② One-Hot 인코딩 (문자 + 길이 검사 포함) ────────
        if not isinstance(site.guide_seq, str):
            return PredictionResult(
                reason=f"{site.cas_type}: invalid guide sequence type"
            )
        x = _one_hot_encode(site.guide_seq, cfg.guide_len)
        if x is None:
            # _one_hot_encode 내부에서 이미 경고 출력됨
            return PredictionResult(
                reason=f"{site.cas_type}: invalid guide sequence or length"
            )

        if site.cas_type == "SpCas9":
            if site.model_input_reason:
                return PredictionResult(reason=site.model_input_reason)
            window = site.model_input_seq
            if not isinstance(window, str) or len(window) != 30:
                return PredictionResult(
                    reason="SpCas9: complete 30bp model window required"
                )
            if set(window) - set("ACGT"):
                return PredictionResult(reason="SpCas9: invalid base in 30bp window")
            if window[4:24] != site.guide_seq or window[24:27] != site.pam_seq:
                return PredictionResult(
                    reason="SpCas9: model window disagrees with guide/PAM"
                )
            x = _one_hot_encode(window, 30)
            if x is None:
                return PredictionResult(reason="SpCas9: invalid model window")

        if site.cas_type in self._load_errors:
            return PredictionResult(
                reason=f"{site.cas_type}: model unavailable: {'; '.join(self._load_errors[site.cas_type])}"
            )

        # ── 방어 ③④ 모델 분기 예측 ──────────────────────────────
        try:
            if site.cas_type == "SpCas9":
                efficiencies = self._predict_hydra(x)
                return PredictionResult(
                    efficiency=efficiencies["SpCas9"],
                    efficiencies_by_protein=efficiencies,
                )
            elif site.cas_type == "Cas12a":
                score = self._predict_single(x, self.model_B)
            else:
                score = self._predict_ensemble(x, self.models_C)
            return PredictionResult(efficiency=score)
        except Exception as e:  # noqa: BLE001 — third-party inference errors become unavailable
            return PredictionResult(reason=f"{site.cas_type}: {type(e).__name__}: {e}")

    """SpCas9의 모델의 결과값 처리 함수"""

    def _predict_hydra(self, x: np.ndarray) -> dict[str, float]:
        # ── 방어 ③ 모델 None 명시적 경고 ────────────────────────
        if self.model_A is None:
            raise ValueError("SpCas9 Hydra model unavailable")
        preds = self.model_A.predict(x, verbose=0)
        if not isinstance(preds, dict):
            raise TypeError("Hydra v2 requires a dict keyed by output layer name")
        efficiencies = {}
        for protein, layer in HYDRA_OUTPUT_LAYERS.items():
            if layer not in preds:
                raise ValueError(f"Missing Hydra head {protein} ({layer})")
            values = np.asarray(preds[layer])
            if values.shape != (1, 1):
                raise ValueError(
                    f"Malformed Hydra head {protein} ({layer}): shape {values.shape}, expected (1, 1)"
                )
            if values.dtype.kind not in "fiu":
                raise ValueError(
                    f"Malformed Hydra head {protein} ({layer}): expected real numeric values"
                )
            value = float(values[0, 0])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite Hydra head {protein} ({layer})")
            efficiencies[protein] = value
        return efficiencies

    """Cas12a의 모델의 결과값 처리 함수"""

    @staticmethod
    def _predict_single(x: np.ndarray, model: tf.keras.Model | None) -> float:
        # ── 방어 ③ 모델 None 명시적 경고 ────────────────────────
        if model is None:
            raise ValueError("Cas12a model unavailable")
        return float(model.predict(x, verbose=0)[0, 0])

    """SaCas9의 모델의 결과값 처리 함수"""

    @staticmethod
    def _predict_ensemble(x: np.ndarray, models: list[tf.keras.Model]) -> float:
        # ── 방어 ③ 앙상블 모델 전부 누락 경고 ───────────────────
        if not models:
            raise ValueError("SaCas9 ensemble unavailable")
        scores = [float(m.predict(x, verbose=0)[0, 0]) for m in models]
        return sum(scores) / len(scores)


# ════════════════════════════════════════════════
# 섹션 6. 메인 스캐너 클래스
# ════════════════════════════════════════════════


class MultiCasScanner:
    """
    Multi-Cas 앙상블 스캐너.

    사용 흐름
    ---------
    1. scanner.scan(seq)                → ScanResult
    2. scanner.predict(result, bundle)  → 점수 채우기 (in-place)
    3. scanner.build_score_dict(result) → 딕셔너리 결과표  ★ 신규
    4. scanner.print_report(result)     → 터미널 출력
    """

    CENTER_IDX: int = 40
    MAX_DIST: int = 15
    SIGMA: float = 10.0

    """설정 및 초기화"""

    def __init__(self, max_dist: int = 15, sigma: float = 10.0):
        self.MAX_DIST = max_dist
        self.SIGMA = sigma

    """PAM 탐색"""

    def scan(self, raw_seq: str) -> ScanResult:
        """81bp 서열에서 전체 PAM 탐색."""
        seq = validate_sequence(raw_seq)
        result = ScanResult(input_seq=seq, center_idx=self.CENTER_IDX)
        for cfg in CAS_CONFIGS:
            result.sites.extend(
                scan_pam(
                    seq, cfg, self.CENTER_IDX, self.MAX_DIST
                )  # PAM의 위치를 모두 찾아냄
            )
        return result

    """예측 점수 산출"""

    def predict(self, result: ScanResult, bundle: ModelBundle) -> None:
        """각 후보에 모델 예측 점수를 채움 (in-place)."""
        for site in result.sites:
            site.prediction = bundle.predict(site)
            site.final_score = None
            raw = site.prediction.efficiency
            if not site.prediction.available or raw is None:
                continue
            penalty = gaussian_penalty(site.distance, self.SIGMA)  # 거리 패널티
            site.final_score = round(raw * penalty * 100, 2)  # %로 환산

    def build_score_dict(self, result: ScanResult) -> dict[str, dict]:
        """
        스캔 결과를 사용자 친화적인 딕셔너리로 변환.

        Available 후보는 기존 raw × gaussian × 100 점수와 계열별 순위를 보존한다.
        Unavailable 후보는 사유를 보존하며 점수와 rank가 None이다.
        선형 출력의 0과 음수도 유효하며 0~100 범위로 제한하지 않는다.

        Returns
        -------
        dict  형식 예시:
        {
          "SpCas9_+_cut50_rank1": {
              "cas_type":  "SpCas9",
              "strand":    "+",
              "cut_pos":   50,
              "distance":  10,
              "pam":       "CGG",
              "guide_seq": "ATCG....(20bp)",
              "available": True,
              "reason": None,
              "rank": 1,
              "raw_score": 0.872,    ← 모델 원본 scale
              "score_pct": 72.45,    ← raw × gaussian × 100
          },
          ...
        }
        """
        entries: list[tuple[str, CandidateSite, int | None]] = []
        ranked = result.sorted_by_score()
        for cfg in CAS_CONFIGS:
            sites = [s for s in ranked if s.cas_type == cfg.name]
            for rank, site in enumerate(sites, 1):
                key = f"{cfg.name}_{site.strand}_cut{site.cut_pos}_rank{rank}"
                entries.append((key, site, rank))
        for index, site in enumerate(result.sites, 1):
            if not site.prediction.available:
                key = f"{site.cas_type}_{site.strand}_cut{site.cut_pos}_unavailable{index}"
                entries.append((key, site, None))
        return {
            key: {
                "cas_type": site.cas_type,
                "strand": site.strand,
                "cut_pos": site.cut_pos,
                "distance": site.distance,
                "pam": site.pam_seq,
                "guide_seq": site.guide_seq,
                "model_input_seq": site.model_input_seq,
                "efficiencies_by_protein": dict(
                    site.prediction.efficiencies_by_protein
                ),
                "available": site.prediction.available,
                "reason": site.prediction.reason,
                "rank": rank,
                "raw_score": site.raw_score,
                "score_pct": site.final_score,  # ×100 변환 완료
            }
            for key, site, rank in entries
        }

    def route_to_models(self, result: ScanResult) -> dict[str, list[CandidateSite]]:
        """후보 목록을 모델 경로(path_A/B/C)별로 분류."""
        routes: dict[str, list[CandidateSite]] = {
            cfg.model_path: [] for cfg in CAS_CONFIGS
        }
        for site in result.sites:  # 각 후보의 가위 종류에 맞는 설정(cfg)을 찾음
            cfg = next(c for c in CAS_CONFIGS if c.name == site.cas_type)
            routes[cfg.model_path].append(site)
        return routes

    def print_report(self, result: ScanResult) -> None:
        """터미널용 결과 리포트 출력 (텍스트 기반 UI)."""
        SEP = "=" * 75
        SEP2 = "─" * 75

        print(f"\n{SEP}")
        print("  Multi-Cas Ensemble Scanner  |  스캔 결과 보고서")
        print(SEP)

        s, c = result.input_seq, result.center_idx
        print(f"  입력 서열 ({len(s)}bp)")
        print(f"  5'-{s[: c - 1]}[{s[c - 1 : c + 2]}]{s[c + 2 :]}-3'")
        print(f"     {'':>{c - 1}}↑ 변이 중심 (index {c})")
        print(f"  {SEP2}")

        # PAM 존재 여부 (has_xxx_pam 활용)
        print("  [PAM 존재 여부]")
        for name, found in check_all_pams(result.input_seq).items():
            print(f"    {'✓' if found else '✗'} {name}")
        print(f"  {SEP2}")

        sites = result.sorted_by_score()
        if not result.sites:
            print("  ⚠  유효한 절단 후보가 없습니다.")
            print(f"     (PAM 없음 또는 변이로부터 >{self.MAX_DIST}bp 초과)")
            print(SEP)
            return

        print(
            f"  {'순위':<5} {'Cas 종류':<10} {'가닥':<5} {'절단위치':<9}"
            f" {'거리':<6} {'원본점수':<10} {'최종점수(%)':<12} gRNA 서열"
        )
        print(f"  {SEP2}")

        for rank, site in enumerate(sites, 1):
            raw_str = f"{site.raw_score:.3f}"
            final_str = f"{site.final_score:.2f}%"
            print(
                f"  {rank:<5} {site.cas_type:<10} {site.strand:<5}"
                f" {site.cut_pos:<9} {site.distance:<6}"
                f" {raw_str:<10} {final_str:<12} {site.guide_seq}"
            )

        unavailable = [s for s in result.sites if not s.prediction.available]
        for site in unavailable:
            print(
                f"  unavailable {site.cas_type} {site.strand} cut{site.cut_pos}: "
                f"{site.prediction.reason}"
            )

        print(f"  {SEP2}")
        counts = {cfg.name: len(result.by_cas(cfg.name)) for cfg in CAS_CONFIGS}
        count_str = ", ".join(f"{k}: {v}개" for k, v in counts.items())
        print(f"  ✓ 총 {len(result.sites)}개 후보  ({count_str})")
        print(f"  ✓ 순위 포함: {len(sites)}개, unavailable: {len(unavailable)}개")
        print(f"  ✓ 거리 기준: 변이 중심 ±{self.MAX_DIST}bp 이내")
        print(f"  ✓ 페널티: 가우시안 (σ={self.SIGMA})")
        print(SEP)


# ════════════════════════════════════════════════
# 섹션 7. 터미널 단독 실행 예시
# ════════════════════════════════════════════════

if __name__ == "__main__":
    import random

    test_seq = (
        "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        "A"
        "TTTACGATCGATCGGGNGGCGATCGATCGATCGATCGATT"
    )

    scanner = MultiCasScanner(max_dist=15, sigma=10.0)

    print("\n[1] PAM 사전 확인")
    for name, found in check_all_pams(test_seq).items():
        print(f"  {name}: {'있음 ✓' if found else '없음 ✗'}")

    print("\n[2] 전체 스캔")
    result = scanner.scan(test_seq)
    print(f"  {result}")

    print("\n[3] 모의 점수 주입")
    for site in result.sites:
        raw = random.uniform(0.3, 0.95)
        site.prediction = PredictionResult(efficiency=raw)
        site.final_score = round(raw * gaussian_penalty(site.distance, 10.0) * 100, 2)

    print("\n[4] 딕셔너리 결과표")
    score_dict = scanner.build_score_dict(result)
    for key, val in score_dict.items():
        print(f"  {key}")
        print(
            f"    score_pct={val['score_pct']}%  raw={val['raw_score']}"
            f"  거리={val['distance']}bp  PAM={val['pam']}"
        )

    print("\n[5] 터미널 리포트")
    scanner.print_report(result)
