"""
pam_scanner.py
──────────────
PAM 서열 탐색 전담 모듈. (모델·UI 코드와 분리)

[2학기 병합] 연동 대상은 SpCas9 · Cas12a 2종이다.
SaCas9는 모델을 연동하지 않기로 했으므로 탐색 대상에서도 제외했다.
(되살리려면 CAS_CONFIGS와 WINDOW_LAYOUT에 SaCas9 항목을 다시 넣고
 predict.py의 MODEL_REGISTRY에 경로를 등록하면 된다.)
"""

import math
from dataclasses import dataclass, field


# ════════════════════════════════════════════════
# 섹션 1. 설정
# ════════════════════════════════════════════════

@dataclass(frozen=True)
class CasConfig:
    name:            str
    pam_iupac:       str        # 탐색용 PAM 패턴
    guide_len:       int
    pam_position:    str        # '3prime' | '5prime'
    cut_offset:      int
    model_input_len: int
    pam_report_len:  int = 0    # 저장·표시할 PAM 길이 (0이면 pam_iupac 길이)


# SpCas9는 SpCas9-NG(2bp NG)까지 잡기 위해 NG로 탐색하되,
# 변종별 호환성 검사를 위해 3bp(NGN)까지 저장한다 (pam_report_len=3).
CAS_CONFIGS: list[CasConfig] = [
    CasConfig(name="SpCas9", pam_iupac="NG",   guide_len=20, pam_position="3prime",
              cut_offset=-3, model_input_len=30, pam_report_len=3),
    CasConfig(name="Cas12a", pam_iupac="TTTV", guide_len=23, pam_position="5prime",
              cut_offset=20, model_input_len=34),
]

# 모델 입력 윈도우: 3' PAM Cas는 가이드 시작 4bp 앞에서부터 자른다.
#   SpCas9: 4 + 가이드20 + PAM(NGN)3 + 뒤 3 = 30
WINDOW_UPSTREAM = 4

#   scan_pam이 만드는 윈도우 구성과 반드시 같아야 한다.
WINDOW_LAYOUT: dict[str, list[tuple[str, int, int]]] = {
    "SpCas9": [("flank", 0, 3), ("guide", 4, 23), ("PAM", 24, 26), ("flank", 27, 29)],
    "Cas12a": [("flank", 0, 3), ("PAM", 4, 7), ("guide", 8, 30), ("flank", 31, 33)],
}

# SpCas9 변종별 인식 PAM (모델 출력 인덱스 순서와 동일).
# ⚠ 아래 표는 일반적으로 알려진 PAM이므로, 학습 데이터/논문 기준과 맞는지 확인할 것.
SPCAS9_VARIANT_PAMS: dict[str, str] = {
    'SpCas9(WT)':   'NGG',
    'SpCas9-NG':    'NG',
    'VRQR':         'NGA',
    'xCas9':        'NG',
    'Sniper':       'NGG',
    'SpCas9-HF1':   'NGG',
    'eSpCas9(1.1)': 'NGG',
    'HypaCas9':     'NGG',
    'evoCas9':      'NGG',
}
SPCAS9_VARIANTS: list[str] = list(SPCAS9_VARIANT_PAMS)

_IUPAC_BASES: dict[str, set[str]] = {
    'A': {'A'}, 'T': {'T'}, 'G': {'G'}, 'C': {'C'},
    'N': {'A', 'T', 'G', 'C'},
    'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'S': {'G', 'C'},
    'W': {'A', 'T'}, 'K': {'G', 'T'}, 'M': {'A', 'C'},
    'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'},
    'H': {'A', 'C', 'T'}, 'V': {'A', 'C', 'G'},
}


# ════════════════════════════════════════════════
# 섹션 2. 데이터 구조
# ════════════════════════════════════════════════

@dataclass
class CandidateSite:
    """유효한 절단 후보 부위 하나"""
    cas_type:        str
    strand:          str
    pam_start:       int
    cut_pos:         int
    distance:        int
    guide_seq:       str
    pam_seq:         str
    model_input_seq: str
    raw_score:       float = 0.0
    final_score:     float = 0.0
    efficiency_pct:  float = 0.0
    score_kind:      str   = "efficiency"   # efficiency | percentile

    @property
    def site_key(self) -> str:
        """사이트 고유 키 (Cas 종류 + 가닥 + PAM 위치)"""
        return f"{self.cas_type}|{self.strand}|{self.pam_start}"


@dataclass
class VariantResult:
    """모델 하나가 사이트 하나(변종 하나)에 낸 예측 결과"""
    cas_name:    str      # 'SpCas9(WT)', ..., 'Cas12a'
    pam_seq:     str
    guide_seq:   str
    distance:    int
    strand:      str
    raw_score:   float
    final_score: float            # raw × 거리 패널티 × 스케일 (화면에는 쓰지 않음)
    efficiency_pct: float = 0.0   # 화면에 띄우는 점수 (%) — 의미는 score_kind가 정한다
    #: "efficiency" = 예측 절단 효율(%),  "percentile" = 학습 분포 내 백분위(%)
    #: 모델이 z-score로 학습된 경우, 평균·표준편차를 모르면 효율로 되돌릴 수 없어 백분위로 보여준다.
    score_kind:  str = "efficiency"
    site_key:    str = ""
    cas_type:    str = ""         # SpCas9 | Cas12a
    model:       str = "A"        # 예측한 모델 (A | B)
    window_seq:  str = ""         # 모델에 들어간 입력 윈도우


@dataclass
class ScanResult:
    input_seq:       str
    center_idx:      int = 40
    sites:           list[CandidateSite] = field(default_factory=list)
    variant_results: list[VariantResult] = field(default_factory=list)

    def by_cas(self, cas_type: str) -> list[CandidateSite]:
        return [s for s in self.sites if s.cas_type == cas_type]

    def sorted_by_score(self) -> list[CandidateSite]:
        return sorted(self.sites, key=lambda s: (s.efficiency_pct, s.raw_score),
                      reverse=True)

    def sorted_variant_results(self) -> list[VariantResult]:
        # 효율이 1순위. 0%로 잘려 동점이 생길 때만 원출력으로 가른다.
        # (변종마다 표준화 통계가 달라 원출력만으로 정렬하면 순서가 뒤바뀐다)
        return sorted(self.variant_results,
                      key=lambda v: (v.efficiency_pct, v.raw_score), reverse=True)


# ════════════════════════════════════════════════
# 섹션 3. 유틸리티
# ════════════════════════════════════════════════

def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn",
                          "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")
    return seq.translate(table)[::-1]


def iupac_match(seq_char: str, pam_char: str) -> bool:
    return bool(_IUPAC_BASES[seq_char] & _IUPAC_BASES[pam_char])


def iupac_pam_search(seq: str, pam: str) -> list[tuple[int, int, str]]:
    matches = []
    pam_len = len(pam)
    for i in range(len(seq) - pam_len + 1):
        window = seq[i:i + pam_len]
        if all(iupac_match(window[j], pam[j]) for j in range(pam_len)):
            matches.append((i, i + pam_len, window))
    return matches


def variant_compatible(variant: str, pam_seq: str) -> bool:
    """이 SpCas9 변종이 해당 PAM 서열을 인식하는지"""
    pattern = SPCAS9_VARIANT_PAMS[variant]
    if len(pam_seq) < len(pattern):
        return False
    return all(iupac_match(pam_seq[i], pattern[i]) for i in range(len(pattern)))


def validate_sequence(seq: str) -> str:
    seq = "".join(seq.upper().split())
    if len(seq) != 81:
        raise ValueError(
            f"입력 서열은 81bp여야 합니다. (현재: {len(seq)}bp)\n"
            "변이 위치 기준 앞뒤 40bp씩 총 81bp를 입력해 주세요."
        )
    invalid = set(seq) - set("ATGCNRYSWKMBDHV")
    if invalid:
        raise ValueError(f"허용되지 않는 문자 포함: {invalid}")
    return seq


def gaussian_penalty(distance: int, sigma: float = 10.0) -> float:
    return math.exp(-(distance ** 2) / (2 * sigma ** 2))


def dna_to_rna(seq: str) -> str:
    """출력 전용 T→U 변환. 내부 연산에는 사용하지 않는다."""
    return seq.replace('T', 'U').replace('t', 'u')


# ════════════════════════════════════════════════
# 섹션 4. PAM 탐색
# ════════════════════════════════════════════════

def _calc_cut_pos(cfg: CasConfig, pam_start: int, pam_end: int,
                  strand: str, seq_len: int) -> int:
    raw_cut = pam_start + cfg.cut_offset if cfg.pam_position == "3prime" \
        else pam_end + cfg.cut_offset
    return raw_cut if strand == '+' else seq_len - raw_cut - 1


def extract_model_input_window(search_seq: str, pam_start: int,
                               cfg: CasConfig) -> str:
    """모델 입력 윈도우. 서열 범위를 벗어나면 빈 문자열."""
    if cfg.pam_position == "3prime":
        window_start = pam_start - cfg.guide_len - WINDOW_UPSTREAM
    else:
        window_start = pam_start
    window_end = window_start + cfg.model_input_len
    if window_start < 0 or window_end > len(search_seq):
        return ""
    return search_seq[window_start:window_end]


def scan_pam(seq: str, cfg: CasConfig,
             center: int = 40, max_dist: int = 15) -> list[CandidateSite]:
    sites: list[CandidateSite] = []
    seq_len = len(seq)

    for strand, search_seq in [('+', seq), ('-', reverse_complement(seq))]:
        for pam_start, pam_end, matched in iupac_pam_search(search_seq, cfg.pam_iupac):

            if cfg.pam_position == "3prime":
                guide_start, guide_end = pam_start - cfg.guide_len, pam_start
                if guide_start < 0:
                    continue
            else:
                guide_start, guide_end = pam_end, pam_end + cfg.guide_len
                if guide_end > seq_len:
                    continue

            cut_pos  = _calc_cut_pos(cfg, pam_start, pam_end, strand, seq_len)
            distance = abs(cut_pos - center)
            if distance > max_dist:
                continue

            # 모델 입력 윈도우를 못 만들면 예측이 불가능하므로 후보에서 제외한다.
            # (발견 개수 = 예측 가능한 사이트 개수)
            model_seq = extract_model_input_window(search_seq, pam_start, cfg)
            if not model_seq:
                continue

            report_len = max(cfg.pam_report_len, len(matched))
            sites.append(CandidateSite(
                cas_type        = cfg.name,
                strand          = strand,
                pam_start       = pam_start,
                cut_pos         = cut_pos,
                distance        = distance,
                guide_seq       = search_seq[guide_start:guide_end],
                pam_seq         = search_seq[pam_start:pam_start + report_len],
                model_input_seq = model_seq,
            ))
    return sites


def scan_all(input_dna: str) -> ScanResult:
    """검증 → 모든 Cas 설정에 대해 PAM 탐색 (출력 없음)"""
    clean  = validate_sequence(input_dna)
    result = ScanResult(input_seq=clean)
    for cfg in CAS_CONFIGS:
        result.sites.extend(scan_pam(clean, cfg, center=result.center_idx))
    return result


def print_scan_report(result: ScanResult) -> None:
    """콘솔 로그용 요약 출력"""
    print(f"\n🔍 DNA 서열 분석 (길이: {len(result.input_seq)}bp)")
    print("-" * 60)
    for cfg in CAS_CONFIGS:
        sites = result.by_cas(cfg.name)
        print(f"[{cfg.name}] {len(sites)}개")
        for s in sites:
            print(f"  · {s.strand} | PAM {s.pam_seq} (RNA {dna_to_rna(s.pam_seq)}) "
                  f"| 위치 {s.pam_start} | 거리 {s.distance} | 가이드 {dna_to_rna(s.guide_seq)}")
    print("-" * 60)
    print(f"합계: {len(result.sites)}개")
