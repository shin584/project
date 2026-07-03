"""
[추가된 사항 - v2]
  ① has_spcas9_pam / has_sacas9_pam / has_cas12a_pam  개별 탐색 함수
  ② build_score_dict()  결과 통합 파이프라인 (딕셔너리 형태, ×100 퍼센트)
  ③ app.py(Streamlit)와 연동되는 구조로 정리
"""

import os
import re  # 정규표현식
import math # 거리패널티
import numpy as np # 데이터 행렬화(One-Hot Encoding)
import tensorflow as tf # 모델 파일을 불러오고 입력된 서열 데이터에 대한 절단 효율 점수 예측
from dataclasses import dataclass, field # 메모리 효율과 가독성을 위해 정형화된 데이터 저장소 생성
from typing import Optional


# ════════════════════════════════════════════════
# 섹션 1. 설정 (PAM 추가/수정은 여기만 건드리면 됨)
# ════════════════════════════════════════════════

@dataclass(frozen=True) # 객체 읽기 전용, 해시 가능
class CasConfig:
    """
    Cas 단백질 한 종류의 PAM/절단 규칙 정의.
    새 단백질 추가 시 이 클래스 인스턴스만 하나 더 만들면 됨.
    """
    name: str           # 유전자 가위 이름
    pam_iupac: str      # PAM 서열 규칙
    guide_len: int      # 모델이 인식할 서열의 길이 결정
    pam_position: str   # '3prime' | '5prime'/ PAM 위치
    cut_offset: int     # DNA가 잘리는 위치
    model_path: str     # 효율 예측할 때 어떤 모델 파일을 사용할 것인지 정해주는 경로

# 3가지 유전자 가위의 구체적인 명세서
CAS_CONFIGS: list[CasConfig] = [
    CasConfig(
        name         = "SpCas9",
        pam_iupac    = "NGG",
        guide_len    = 20,
        pam_position = "3prime",
        cut_offset   = -3,
        model_path   = "path_A",
    ),
    CasConfig(
        name         = "SaCas9",
        pam_iupac    = "NNGRRT",
        guide_len    = 21,
        pam_position = "3prime",
        cut_offset   = -3,
        model_path   = "path_C",
    ),
    CasConfig(
        name         = "Cas12a",
        pam_iupac    = "TTTV",
        guide_len    = 23,
        pam_position = "5prime",
        cut_offset   = 20,
        model_path   = "path_B",
    ),
]

# 이후에 추가 모델의 PAM 서열을 인식해야할 경우를 대비해 모든 IUPAC 넣어놓음
_IUPAC_TABLE: dict[str, str] = {
    'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C',
    'N': '[ATGC]',
    'R': '[AG]',
    'Y': '[CT]',
    'S': '[GC]',
    'W': '[AT]',
    'K': '[GT]',
    'M': '[AC]',
    'B': '[CGT]',
    'D': '[AGT]',
    'H': '[ACT]',
    'V': '[ACG]',
}


# ════════════════════════════════════════════════
# 섹션 2. 데이터 구조
# ════════════════════════════════════════════════

@dataclass
class CandidateSite:
    """유효한 절단 후보 부위 하나의 정보"""
    cas_type:    str            # 가위 종류
    strand:      str            # 가닥 방향(정방향 +, 역방향 -)
    pam_start:   int            # PAM 시작점(인덱스 형태)
    cut_pos:     int            #절단 위치
    distance:    int            # 변이와의 거리
    guide_seq:   str            # gRNA 서열(AI 모델에 입력값으로 넣을 가이드 서열)
    pam_seq:     str            # 발견된 PAM 서열
    raw_score:   float = 0.0    # 순수 절단 효율(0~1)
    final_score: float = 0.0    # 거리패널티 적용 후 %로 변환한 최종 값


@dataclass
class ScanResult:
    """전체 스캔 결과 컨테이너"""
    input_seq:  str                                                 # 사용자가 입력한 81bp DNA 서열
    center_idx: int = 40                                            # 타겟 변위의 위치
    sites:      list[CandidateSite] = field(default_factory=list)   # 발견된 모든 Candidatesite들이 담기는 리스트, 빈 리스트로 시작

    """필터링-특정 가위 종류만 골라냄"""
    def by_cas(self, cas_type: str) -> list[CandidateSite]:
        return [s for s in self.sites if s.cas_type == cas_type]

    """정렬-모든 후보의 점수를 내림차순으로 정렬"""
    def sorted_by_score(self) -> list[CandidateSite]:
        return sorted(self.sites, key=lambda s: s.final_score, reverse=True)

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
        return ''.join(_IUPAC_TABLE[b] for b in pam.upper())
    
    except KeyError as e: 
        raise ValueError(f"알 수 없는 IUPAC 코드: {e}") # 사전에 없는 단어 오류 처리

"""DNA 반대쪽 가닥의 서열을 계산, 유전자 가위가 반대쪽에 붙을 경우 고려"""
def reverse_complement(seq: str) -> str:
    """DNA 서열의 역상보(Reverse Complement) 반환"""
    table = str.maketrans("ACGT", "TGCA")
    return seq.translate(table)[::-1]   # 역순 정렬


def validate_sequence(seq: str) -> str:
    """입력 서열 유효성 검사 (81bp 고정, ATGCN 허용)"""
    seq = seq.upper().strip()   # 대문자 통일, 공백 제거
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
    return math.exp(-(distance ** 2) / (2 * sigma ** 2))


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
    seq = seq.upper()                           # 대문자로 통일
    pam_re = re.compile(iupac_to_regex("NGG"))  # 정방향 탐색(5'->3'), 정규표현식 패턴([ATGC]GG으로 변환
    return bool(pam_re.search(seq)) or bool(pam_re.search(reverse_complement(seq)))     # 정방향, 역방향 중 한군데라도 발견되면 True 반환


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

def _calc_cut_pos(cfg: CasConfig, pam_start: int, pam_end: int,
                  strand: str, seq_len: int) -> int:
    """절단 위치(원본 서열 기준) 계산. 역가닥은 좌표 변환."""
    if cfg.pam_position == "3prime":    # SpCas9, SaCas9는 PAM이 가이드 서열 뒤에 존재
        raw_cut = pam_start + cfg.cut_offset
    else:                               # Cas12a는 PAM이 가이드 서열 앞에 존재
        raw_cut = pam_end + cfg.cut_offset
    return raw_cut if strand == '+' else seq_len - raw_cut - 1  #역가닥 좌표를 정방향 서열 기준의 좌표로 복구


def scan_pam(seq: str, cfg: CasConfig,
             center: int = 40, max_dist: int = 15) -> list[CandidateSite]:
    """단일 CasConfig 기준으로 양쪽 가닥에서 PAM을 탐색."""
    sites: list[CandidateSite] = []
    pam_re  = re.compile(iupac_to_regex(cfg.pam_iupac))
    seq_len = len(seq)

    for strand, search_seq in [('+', seq), ('-', reverse_complement(seq))]:
        for m in pam_re.finditer(search_seq):
            pam_start = m.start()
            pam_end   = m.end()

            if cfg.pam_position == "3prime":            # SpCas9, SaCas9
                guide_start = pam_start - cfg.guide_len # PAM 앞쪽으로 가이드 서열이 붙을 수 있을 만큼의 서열이 존재하는지 확인
                guide_end   = pam_start                 # 가이드 서열의 끝=PAM 서열의 시작
                if guide_start < 0:
                    continue
            else:   # Cas12a
                guide_start = pam_end
                guide_end   = pam_end + cfg.guide_len
                if guide_end > seq_len:
                    continue

            cut_pos  = _calc_cut_pos(cfg, pam_start, pam_end, strand, seq_len)
            distance = abs(cut_pos - center)
            if distance > max_dist: # 절단위치와 변이 위치 사이의 거리 계산해서 유효 서열만 추림
                continue

            sites.append(CandidateSite(
                cas_type  = cfg.name,
                strand    = strand,
                pam_start = pam_start,
                cut_pos   = cut_pos,
                distance  = distance,
                guide_seq = search_seq[guide_start:guide_end],
                pam_seq   = m.group(),
            ))

    return sites


# ════════════════════════════════════════════════
# 섹션 5. 모델 관련
# ════════════════════════════════════════════════

"""텍스트 염기 하나를 4자리의 숫자 리스트에 대흥시킴"""
_ONE_HOT: dict[str, list[int]] = {
    'A': [1,0,0,0], 'C': [0,1,0,0],
    'G': [0,0,1,0], 'T': [0,0,0,1],
    'N': [0,0,0,0],
}


def _one_hot_encode(seq: str, target_len: int) -> Optional[np.ndarray]: # 결과로 숫재 배열 return, 오류는 None return
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
        print(f"  ⚠ [무결성] 허용되지 않는 문자 포함 → 인코딩 거부: {invalid_chars}"
              f"  (서열: {seq[:10]}...)")
        return None

    # ── 방어 ② 길이 불일치 검사 ─────────────────────────────────
    if len(seq) != target_len:  #가이드 서열 길이 확인
        print(f"  ⚠ [무결성] gRNA 길이 불일치 → 인코딩 거부: "
              f"모델 요구={target_len}bp, 실제={len(seq)}bp "
              f"(서열: {seq[:10]}...)")
        return None

    arr = np.array([_ONE_HOT.get(b, [0,0,0,0]) for b in seq], dtype=np.float32)
    """숫자리스트로 바꾼 DNA 서열을 Numpy 행렬로 변환"""
    return arr[np.newaxis, :]


class ModelBundle:
    """세 가지 모델 경로(A/B/C)를 관리하는 번들."""

    def __init__(self,
                 path_A: str,       # SpCas9 모델 파일 위치
                 path_B: str,       # Cas12a 모델 파일 위치
                 path_C_dir: str,   # Sacas9는 모델 여러개이므로 파일이 들어있는 폴더 경로 받음
                 sacas9_files: Optional[list[str]] = None,
                 hydra_head_idx: int = 0):
        if sacas9_files is None:
            sacas9_files = ['SaCas9_1.keras', 'SaCas9_2.keras', 'SaCas9_3.keras']
        self.hydra_head_idx = hydra_head_idx
        self.model_A  = self._load_single(path_A,  "SpCas9 Hydra")
        self.model_B  = self._load_single(path_B,  "Cas12a")
        self.models_C = self._load_ensemble(path_C_dir, sacas9_files, "SaCas9")

        # ── 방어 ③ 모델 로드 후 상태 요약 경고 ─────────────────
        self._warn_missing_models()

    def _warn_missing_models(self) -> None:
        """
        무결성 방어 ③: 모델 로드 실패 여부를 한 번에 정리해서 출력.
        로드되지 않은 모델로 예측이 들어오면 0.0을 반환하는데,
        사용자가 이 사실을 모를 수 있으므로 초기화 시점에 명시적으로 경고.
        """
        missing = []
        if self.model_A is None:
            missing.append("SpCas9 Hydra (path_A)  → SpCas9 예측값 전부 0.0 처리됨")
        if self.model_B is None:
            missing.append("Cas12a (path_B)         → Cas12a 예측값 전부 0.0 처리됨")
        if not self.models_C:
            missing.append("SaCas9 앙상블 (path_C)  → SaCas9 예측값 전부 0.0 처리됨")
        elif len(self.models_C) < 3:
            missing.append(
                f"SaCas9 앙상블 일부 누락   → {len(self.models_C)}/3개 모델로 평균 산출"
            )
        if missing:
            print("\n  ══ [무결성 경고] 미로드 모델 목록 ══")
            for msg in missing:
                print(f"    ❌ {msg}")
            print("  ══════════════════════════════════\n")

    @staticmethod
    def _load_single(path: str, label: str) -> Optional[tf.keras.Model]:    #path: .keras의 경로/ label: "SpCas9"와 같은 로그 남길 때 사용할 이름
        try:
            m = tf.keras.models.load_model(path)
            print(f"  ✅ {label} 로드 완료: {path}")
            return m
        except Exception as e:
            print(f"  ❌ {label} 로드 실패: {e}")
            return None

    """여러개의 AI 모델을 앙상블로 불러오기"""
    @staticmethod
    def _load_ensemble(directory: str, files: list[str],
                       label: str) -> list[tf.keras.Model]:
        models = []
        for fname in files:
            fpath = os.path.join(directory, fname)
            try:
                models.append(tf.keras.models.load_model(fpath))
                print(f"  ✅ {label} 앙상블 로드: {fname}")
            except Exception as e:
                print(f"  ❌ {label} 앙상블 로드 실패 ({fname}): {e}")
        return models

    def predict(self, site: CandidateSite) -> float:
        """
        CandidateSite → 원본 예측값(0~1) 반환.

        무결성 방어 흐름:
          ① CasConfig 조회 실패  → 0.0 (알 수 없는 Cas 종류)
          ② One-Hot 인코딩 실패  → 0.0 (문자 오류 or 길이 불일치)
          ③ 모델이 None          → 0.0 + 경고 메시지 출력
          ④ 예측 중 예외 발생    → 0.0 + 예외 메시지 출력
        """
        # ── 방어 ① CasConfig 존재 확인 ──────────────────────────
        cfg = next((c for c in CAS_CONFIGS if c.name == site.cas_type), None)
        if cfg is None:
            print(f"  ⚠ [무결성] 알 수 없는 Cas 종류: '{site.cas_type}' → 예측 건너뜀")
            return 0.0

        # ── 방어 ② One-Hot 인코딩 (문자 + 길이 검사 포함) ────────
        x = _one_hot_encode(site.guide_seq, cfg.guide_len)
        if x is None:
            # _one_hot_encode 내부에서 이미 경고 출력됨
            return 0.0

        # ── 방어 ③④ 모델 분기 예측 ──────────────────────────────
        try:
            if site.cas_type == "SpCas9":
                return self._predict_hydra(x)
            elif site.cas_type == "Cas12a":
                return self._predict_single(x, self.model_B)
            elif site.cas_type == "SaCas9":
                return self._predict_ensemble(x, self.models_C)
        except Exception as e:
            print(f"  ⚠ [무결성] 예측 중 예외 발생 ({site.cas_type}): {e}")
        return 0.0

    """SpCas9의 모델의 결과값 처리 함수"""
    def _predict_hydra(self, x: np.ndarray) -> float:
        # ── 방어 ③ 모델 None 명시적 경고 ────────────────────────
        if self.model_A is None:
            print("  ⚠ [무결성] SpCas9 Hydra 모델 미로드 → 0.0 반환")
            return 0.0
        preds = self.model_A.predict(x, verbose=0)
        if preds.ndim == 3:
            return float(preds[0, self.hydra_head_idx, 0])
        elif preds.ndim == 2 and preds.shape[1] > 1:
            return float(preds[0, self.hydra_head_idx])
        return float(preds[0, 0])

    """Cas12a의 모델의 결과값 처리 함수"""
    @staticmethod
    def _predict_single(x: np.ndarray,
                        model: Optional[tf.keras.Model]) -> float:
        # ── 방어 ③ 모델 None 명시적 경고 ────────────────────────
        if model is None:
            print("  ⚠ [무결성] Cas12a 모델 미로드 → 0.0 반환")
            return 0.0
        return float(model.predict(x, verbose=0)[0, 0])

    """SaCas9의 모델의 결과값 처리 함수"""
    @staticmethod
    def _predict_ensemble(x: np.ndarray,
                          models: list[tf.keras.Model]) -> float:
        # ── 방어 ③ 앙상블 모델 전부 누락 경고 ───────────────────
        if not models:
            print("  ⚠ [무결성] SaCas9 앙상블 모델 전부 미로드 → 0.0 반환")
            return 0.0
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

    CENTER_IDX: int   = 40
    MAX_DIST:   int   = 15
    SIGMA:      float = 10.0

    """설정 및 초기화"""
    def __init__(self, max_dist: int = 15, sigma: float = 10.0):
        self.MAX_DIST = max_dist
        self.SIGMA    = sigma

    """PAM 탐색"""
    def scan(self, raw_seq: str) -> ScanResult:
        """81bp 서열에서 전체 PAM 탐색."""
        seq    = validate_sequence(raw_seq)
        result = ScanResult(input_seq=seq, center_idx=self.CENTER_IDX)
        for cfg in CAS_CONFIGS:
            result.sites.extend(
                scan_pam(seq, cfg, self.CENTER_IDX, self.MAX_DIST)  #PAM의 위치를 모두 찾아냄
            )
        return result

    """예측 점수 산출"""
    def predict(self, result: ScanResult, bundle: ModelBundle) -> None:
        """각 후보에 모델 예측 점수를 채움 (in-place)."""
        for site in result.sites:
            raw              = bundle.predict(site)                         # AI 모델의 원본 예측값(0~1)
            penalty          = gaussian_penalty(site.distance, self.SIGMA)  # 거리 패널티
            site.raw_score   = round(raw, 4)
            site.final_score = round(raw * penalty * 100, 2)                # %로 환산

    def build_score_dict(self, result: ScanResult) -> dict[str, dict]:
        """
        스캔 결과를 사용자 친화적인 딕셔너리로 변환.  

        각 후보의 예측치(Sigmoid 결과값 혹은 스칼라)에 ×100을 적용하여
        0~100% 퍼센트 수치로 정규화하고 딕셔너리에 저장.

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
              "raw_score": 0.872,    ← 모델 원본값 (0~1)
              "score_pct": 72.45,    ← raw × gaussian × 100
          },
          ...
        }
        """
        score_dict: dict[str, dict] = {}
        for cfg in CAS_CONFIGS:
            sites = sorted(
                result.by_cas(cfg.name),
                key=lambda s: s.final_score, reverse=True
            )
            for rank, site in enumerate(sites, 1):
                key = f"{cfg.name}_{site.strand}_cut{site.cut_pos}_rank{rank}"
                score_dict[key] = {
                    "cas_type":  site.cas_type,
                    "strand":    site.strand,
                    "cut_pos":   site.cut_pos,
                    "distance":  site.distance,
                    "pam":       site.pam_seq,
                    "guide_seq": site.guide_seq,
                    "raw_score": site.raw_score,
                    "score_pct": site.final_score,   # ×100 변환 완료
                }
        return score_dict

    def route_to_models(self, result: ScanResult) -> dict[str, list[CandidateSite]]:
        """후보 목록을 모델 경로(path_A/B/C)별로 분류."""
        routes: dict[str, list[CandidateSite]] = {
            cfg.model_path: [] for cfg in CAS_CONFIGS
        }
        for site in result.sites:   # 각 후보의 가위 종류에 맞는 설정(cfg)을 찾음
            cfg = next(c for c in CAS_CONFIGS if c.name == site.cas_type)
            routes[cfg.model_path].append(site)
        return routes

    
    def print_report(self, result: ScanResult) -> None:
        """터미널용 결과 리포트 출력 (텍스트 기반 UI)."""
        SEP  = "=" * 75
        SEP2 = "─" * 75

        print(f"\n{SEP}")
        print(f"  Multi-Cas Ensemble Scanner  |  스캔 결과 보고서")
        print(SEP)

        s, c = result.input_seq, result.center_idx
        print(f"  입력 서열 ({len(s)}bp)")
        print(f"  5'-{s[:c-1]}[{s[c-1:c+2]}]{s[c+2:]}-3'")
        print(f"     {'':>{c-1}}↑ 변이 중심 (index {c})")
        print(f"  {SEP2}")

        # PAM 존재 여부 (has_xxx_pam 활용)
        print("  [PAM 존재 여부]")
        for name, found in check_all_pams(result.input_seq).items():
            print(f"    {'✓' if found else '✗'} {name}")
        print(f"  {SEP2}")

        sites = result.sorted_by_score()
        if not sites:
            print("  ⚠  유효한 절단 후보가 없습니다.")
            print(f"     (PAM 없음 또는 변이로부터 >{self.MAX_DIST}bp 초과)")
            print(SEP)
            return

        print(f"  {'순위':<5} {'Cas 종류':<10} {'가닥':<5} {'절단위치':<9}"
              f" {'거리':<6} {'원본점수':<10} {'최종점수(%)':<12} gRNA 서열")
        print(f"  {SEP2}")

        for rank, site in enumerate(sites, 1):
            raw_str   = f"{site.raw_score:.3f}" if site.raw_score else "  -  "
            final_str = f"{site.final_score:.2f}%" if site.final_score else "  -  "
            print(f"  {rank:<5} {site.cas_type:<10} {site.strand:<5}"
                  f" {site.cut_pos:<9} {site.distance:<6}"
                  f" {raw_str:<10} {final_str:<12} {site.guide_seq}")

        print(f"  {SEP2}")
        counts    = {cfg.name: len(result.by_cas(cfg.name)) for cfg in CAS_CONFIGS}
        count_str = ", ".join(f"{k}: {v}개" for k, v in counts.items())
        print(f"  ✓ 총 {len(sites)}개 후보  ({count_str})")
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
        raw              = random.uniform(0.3, 0.95)
        site.raw_score   = round(raw, 4)
        site.final_score = round(
            raw * gaussian_penalty(site.distance, 10.0) * 100, 2
        )

    print("\n[4] 딕셔너리 결과표")
    score_dict = scanner.build_score_dict(result)
    for key, val in score_dict.items():
        print(f"  {key}")
        print(f"    score_pct={val['score_pct']}%  raw={val['raw_score']}"
              f"  거리={val['distance']}bp  PAM={val['pam']}")

    print("\n[5] 터미널 리포트")
    scanner.print_report(result)
