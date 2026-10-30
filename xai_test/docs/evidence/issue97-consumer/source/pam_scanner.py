"""
pam_scanner.py
──────────────
PAM 서열 탐색 전담 모듈.
"""

import re
import math
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ════════════════════════════════════════════════
# 섹션 1. 설정
# ════════════════════════════════════════════════

@dataclass(frozen=True)
class CasConfig:
    name:               str
    pam_iupac:          str
    guide_len:          int
    pam_position:       str   # '3prime' | '5prime'
    cut_offset:         int
    model_path:         str
    model_input_len:    int

# [수정] SpCas9 PAM을 NGN(3bp) → NG(2bp)로 변경
# 이유: SpCas9-NG 변종의 PAM이 2bp 'NG'이므로, 3bp NGN으로 탐색하면
#       SpCas9-NG 전용 사이트가 누락될 수 있음.
#       NG는 NGN의 부분집합이 아니라 독립적인 2bp PAM이므로 별도 처리 필요.
#       단, 나머지 8개 변종(WT, VRQR 등)은 NGG/NGA 등 NGN에 해당하므로
#       NG(2bp)로 탐색해도 모두 포함됨 (NGG → NG + G, 탐색시 NG 패턴 매칭 성공).
CAS_CONFIGS: list[CasConfig] = [
    CasConfig(
        name            = "SpCas9",
        pam_iupac       = "NG",       # 변경: NGN → NG (2bp, 모든 변종 커버)
        guide_len       = 20,
        pam_position    = "3prime",
        cut_offset      = -3,
        model_path      = "models/Multi-Cas_1IN_9Hydra_Divide_Testset.keras",
        model_input_len = 30,
    ),
    CasConfig(
        name            = "SaCas9",
        pam_iupac       = "NNGRRT",
        guide_len       = 21,
        pam_position    = "3prime",
        cut_offset      = -3,
        model_path      = "",
        model_input_len = 36,
    ),
    CasConfig(
        name            = "Cas12a",
        pam_iupac       = "TTTV",
        guide_len       = 23,
        pam_position    = "5prime",
        cut_offset      = 20,
        model_path      = "models/cas12a_model_fold_4.keras",
        model_input_len = 34,
    ),
]

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

# SpCas9 9개 변종 이름 (predict.py와 동기화)
SPCAS9_VARIANTS = [
    'SpCas9(WT)', 'SpCas9-NG', 'VRQR', 'xCas9',
    'Sniper', 'SpCas9-HF1', 'eSpCas9(1.1)', 'HypaCas9', 'evoCas9'
]


# ════════════════════════════════════════════════
# 섹션 2. 데이터 구조
# ════════════════════════════════════════════════

@dataclass
class CandidateSite:
    """유효한 절단 후보 부위 하나의 정보"""
    cas_type:        str
    strand:          str
    pam_start:       int
    cut_pos:         int
    distance:        int
    guide_seq:       str
    pam_seq:         str
    model_input_seq: str   = ""
    encoded_seq:     list  = field(default_factory=list)
    raw_score:       float = 0.0
    final_score:     float = 0.0


@dataclass
class VariantResult:
    """
    [신규] SpCas9 변종 개별 예측 결과.
    하나의 CandidateSite에서 9개 변종 각각에 대해 생성됨.
    """
    cas_name:    str    # 예: 'SpCas9(WT)', 'SpCas9-NG', ...
    pam_seq:     str
    guide_seq:   str
    distance:    int
    strand:      str
    raw_score:   float
    final_score: float


@dataclass
class ScanResult:
    """전체 스캔 결과 컨테이너"""
    input_seq:      str
    center_idx:     int                       = 40
    sites:          list[CandidateSite]       = field(default_factory=list)
    # [신규] 변종별 예측 결과를 담는 flat 리스트 (predict.py에서 채워짐)
    variant_results: list[VariantResult]      = field(default_factory=list)

    def by_cas(self, cas_type: str) -> list[CandidateSite]:
        return [s for s in self.sites if s.cas_type == cas_type]

    def sorted_by_score(self) -> list[CandidateSite]:
        return sorted(self.sites, key=lambda s: s.final_score, reverse=True)

    def sorted_variant_results(self) -> list[VariantResult]:
        """[신규] 변종 결과를 final_score 내림차순으로 정렬"""
        return sorted(self.variant_results, key=lambda v: v.final_score, reverse=True)

    def __repr__(self) -> str:
        counts = {cfg.name: len(self.by_cas(cfg.name)) for cfg in CAS_CONFIGS}
        parts  = ", ".join(f"{k}={v}" for k, v in counts.items())
        return f"ScanResult({parts}, total={len(self.sites)})"


# ════════════════════════════════════════════════
# 섹션 3. 유틸리티 함수
# ════════════════════════════════════════════════

def one_hot_encode(seq: str) -> list:
    mapping = {
        'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
        'R': [0.5, 0, 0.5, 0],  'Y': [0, 0.5, 0, 0.5],
        'S': [0, 0.5, 0.5, 0],  'W': [0.5, 0, 0, 0.5],
        'K': [0, 0, 0.5, 0.5],  'M': [0.5, 0.5, 0, 0],
        'B': [0, 0.33, 0.33, 0.33], 'D': [0.33, 0, 0.33, 0.33],
        'H': [0.33, 0.33, 0, 0.33], 'V': [0.33, 0.33, 0.33, 0],
        'N': [0.25, 0.25, 0.25, 0.25]
    }
    return [mapping.get(b, [0,0,0,0]) for b in seq.upper()]

def iupac_to_regex(pam: str) -> str:
    try:
        return ''.join(_IUPAC_TABLE[b] for b in pam.upper())
    except KeyError as e:
        raise ValueError(f"알 수 없는 IUPAC 코드: {e}")

def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn",
                           "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")
    return seq.translate(table)[::-1]

_IUPAC_BASES: dict[str, set[str]] = {
    'A': {'A'}, 'T': {'T'}, 'G': {'G'}, 'C': {'C'},
    'N': {'A','T','G','C'},
    'R': {'A','G'}, 'Y': {'C','T'}, 'S': {'G','C'},
    'W': {'A','T'}, 'K': {'G','T'}, 'M': {'A','C'},
    'B': {'C','G','T'}, 'D': {'A','G','T'},
    'H': {'A','C','T'}, 'V': {'A','C','G'},
}

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

def validate_sequence(seq: str) -> str:
    seq = "".join(seq.upper().split())
    if len(seq) != 81:
        raise ValueError(
            f"입력 서열은 81bp여야 합니다. (현재: {len(seq)}bp)\n"
            "변이 위치 기준 앞뒤 40bp씩 총 81bp를 입력해 주세요."
        )
    VALID_IUPAC = set("ATGCNRYSWKMBDHV")
    invalid = set(seq) - VALID_IUPAC
    if invalid:
        raise ValueError(f"허용되지 않는 문자 포함: {invalid}")
    return seq

def gaussian_penalty(distance: int, sigma: float = 10.0) -> float:
    return math.exp(-(distance ** 2) / (2 * sigma ** 2))

def _wcswidth(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)

def _ljust_wide(s: str, width: int) -> str:
    return s + ' ' * max(width - _wcswidth(s), 0)

def dna_to_rna(seq: str) -> str:
    """
    출력 전용 T→U 변환 함수.
    내부 연산(PAM 탐색, 인코딩 등)에는 절대 사용하지 않음.
    """
    return seq.replace('T', 'U').replace('t', 'u')


# ════════════════════════════════════════════════
# 섹션 4-1. 개별 PAM 존재 여부 탐색 함수
# ════════════════════════════════════════════════

def has_spcas9_pam(seq: str) -> bool:
    """
    SpCas9 및 9가지 변종 PAM 존재 여부 확인.
    [수정] 2bp NG 패턴으로 탐색 (SpCas9-NG 포함)
    """
    seq = seq.upper()
    pam_re = re.compile(iupac_to_regex("NG"))
    return bool(pam_re.search(seq)) or bool(pam_re.search(reverse_complement(seq)))

def has_sacas9_pam(seq: str) -> bool:
    seq    = seq.upper()
    pam_re = re.compile(iupac_to_regex("NNGRRT"))
    return bool(pam_re.search(seq)) or bool(pam_re.search(reverse_complement(seq)))

def has_cas12a_pam(seq: str) -> bool:
    seq    = seq.upper()
    pam_re = re.compile(iupac_to_regex("TTTV"))
    return bool(pam_re.search(seq)) or bool(pam_re.search(reverse_complement(seq)))

def check_all_pams(seq: str) -> dict[str, bool]:
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
    if cfg.pam_position == "3prime":
        raw_cut = pam_start + cfg.cut_offset
    else:
        raw_cut = pam_end + cfg.cut_offset
    return raw_cut if strand == '+' else seq_len - raw_cut - 1

def extract_model_input_window(search_seq: str, pam_start: int, pam_end: int,
                                cfg: CasConfig) -> str:
    seq_len    = len(search_seq)
    window_len = cfg.model_input_len

    if cfg.pam_position == "3prime":
        # [수정] PAM이 2bp(NG)로 바뀌었으므로 윈도우 시작점도 그에 맞게 계산
        # model_input_len=30: guide(20bp) + PAM(2bp) + upstream(4bp) + downstream(4bp) 구성
        window_start = pam_start - cfg.guide_len - 4
        window_end   = window_start + window_len
    else:
        window_start = pam_start
        window_end   = window_start + window_len

    if window_start < 0 or window_end > seq_len:
        return ""

    return search_seq[window_start:window_end]

def scan_pam(seq: str, cfg: CasConfig,
             center: int = 40, max_dist: int = 15) -> list[CandidateSite]:
    sites:   list[CandidateSite] = []
    seq_len = len(seq)

    for strand, search_seq in [('+', seq), ('-', reverse_complement(seq))]:
        for pam_start, pam_end, matched_window in iupac_pam_search(search_seq, cfg.pam_iupac):

            if cfg.pam_position == "3prime":
                guide_start = pam_start - cfg.guide_len
                guide_end   = pam_start
                if guide_start < 0:
                    continue
            else:
                guide_start = pam_end
                guide_end   = pam_end + cfg.guide_len
                if guide_end > seq_len:
                    continue

            cut_pos  = _calc_cut_pos(cfg, pam_start, pam_end, strand, seq_len)
            distance = abs(cut_pos - center)
            if distance > max_dist:
                continue

            model_seq = extract_model_input_window(search_seq, pam_start, pam_end, cfg)

            sites.append(CandidateSite(
                cas_type        = cfg.name,
                strand          = strand,
                pam_start       = pam_start,
                cut_pos         = cut_pos,
                distance        = distance,
                guide_seq       = search_seq[guide_start:guide_end],
                pam_seq         = matched_window,
                model_input_seq = model_seq,
                encoded_seq     = one_hot_encode(model_seq) if model_seq else [],
            ))

    return sites


def print_pam_analysis(input_dna: str) -> ScanResult:
    clean_seq   = validate_sequence(input_dna)
    all_results = ScanResult(input_seq=clean_seq)

    print(f"\n🔍 DNA 서열 분석 시작 (길이: {len(clean_seq)}bp)")
    print("-" * 60)

    for cfg in CAS_CONFIGS:
        found_sites = scan_pam(clean_seq, cfg)
        all_results.sites.extend(found_sites)

        print(f"[{cfg.name}] 탐색 결과:")
        if not found_sites:
            print("  - 발견된 PAM 서열이 없습니다.")
        for site in found_sites:
            # ── DNA 표기 (기존 출력 유지) ──────────────────────────────────────
            print(f"  · 가닥: {site.strand} | PAM(DNA): {site.pam_seq} | 위치: {site.pam_start} | 가이드(DNA): {site.guide_seq}")
            # ── RNA 표기 (T→U 변환, 신규 추가) ────────────────────────────────
            print(f"           PAM(RNA): {dna_to_rna(site.pam_seq)}        가이드(RNA): {dna_to_rna(site.guide_seq)}")
        print("-" * 60)

    print("\n📊 [PAM 탐색 최종 요약]")
    print("+" + "-"*15 + "+" + "-"*15 + "+")
    print(f"| {'Cas 모델':^13} | {'발견된 개수':^11} |")
    print("+" + "-"*15 + "+" + "-"*15 + "+")
    for cfg in CAS_CONFIGS:
        count = len(all_results.by_cas(cfg.name))
        print(f"| {cfg.name:<13} | {count:^13} |")
    print("+" + "-"*15 + "+" + "-"*15 + "+")
    print(f"| {'합계':<13} | {len(all_results.sites):^13} |")
    print("+" + "-"*15 + "+" + "-"*15 + "+")

    return all_results
