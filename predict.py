"""
predict.py
──────────
모델 로딩 및 절단 효율 예측 전담 모듈.

연동 모델은 (모델 이름, Cas 종류) 키의 MODEL_REGISTRY 한 곳에서 관리한다.
  - 모델 파일 이름이 바뀌면 → MODEL_REGISTRY의 경로만 수정
  - 모델을 어디서 가져올지(로컬/구글 드라이브)는 → model_source.py에서 결정
  - 새 모델 종류를 연동하면 → 예측기 클래스를 PREDICTOR_CLASSES에 등록

[현재 연동 모델] Nucleotide Transformer(ESM) 기반 safetensors 2종
  · SpCas9 : models/SpCas9.safetensors — 출력 9개 (SPCAS9_VARIANTS 순서)
  · Cas12a : models/Cas12a.safetensors — 출력 1개

  두 파일 모두 EsmForSequenceClassification이고, 설정은 safetensors 헤더에서
  자동으로 읽어 낸다 (층 수·hidden·vocab·출력 개수). 헤더로 확인된 값:
      vocab 4105 · hidden 1280 · 24층 · intermediate 5120 · 절대위치 1002
      attention heads 20  (contact_head [1, 480] = 24층 × 20헤드로 확정)

필요한 패키지 (관리자 권한 불필요):
    pip install torch transformers safetensors
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
import streamlit as st

import model_source
from pam_scanner import (
    SPCAS9_VARIANTS, CandidateSite, ScanResult, VariantResult,
    gaussian_penalty, variant_compatible,
)


# ════════════════════════════════════════════════
# 0. NT(Nucleotide Transformer) 설정 — 바꿀 일이 생기면 여기만
# ════════════════════════════════════════════════

#: 토크나이저를 받아올 HuggingFace 저장소.
#: NT v1 계열은 토크나이저가 동일하므로 500m 저장소 아무거나 써도 된다.
#: 처음 실행할 때 한 번 내려받아 캐시에 저장된다 (인터넷 필요).
NT_TOKENIZER_REPO = "InstaDeepAI/nucleotide-transformer-500m-human-ref"

#: 출력 해석 방식은 모델마다 다르므로 MODEL_REGISTRY의 ModelSpec에서 지정한다.
#: (아래 OUTPUT_MODES 설명 참고). 이 전역값은 더 이상 쓰지 않는다.

#: attention head 수. 가중치에는 안 들어 있어 헤더에서 역산한 값이다.
NT_NUM_ATTENTION_HEADS = 20

#: 한 번에 모델에 넣을 서열 수. 메모리가 부족하면 줄인다.
NT_BATCH_SIZE = 8

#: 가중치를 메모리에 올릴 때의 자료형.
#:   "float32"  — 파일 그대로. 모델 하나당 약 1.9GB. 기본값.
#:   "bfloat16" — 절반 크기(약 1.0GB). 메모리가 부족하면 이걸로 바꾼다.
#:                예측값이 소수점 아래에서 조금 달라질 수 있다.
NT_DTYPE = "float32"


# ════════════════════════════════════════════════
# 1. 레지스트리
# ════════════════════════════════════════════════

#: 모델이 내놓는 값을 어떻게 읽어야 하는지.
#:   "raw"     — 이미 0~1 비율. clip해서 그대로 효율(%)로 쓴다.
#:   "sigmoid" — 로짓. sigmoid를 거쳐 0~1로 만든다.
#:   "percent" — 0~100 퍼센트. 100으로 나눈다.
#:   "zscore"  — 표준화된 값. 0이 학습 데이터의 평균 효율이다.
#:               target_mean·target_std를 알면 실제 효율로 되돌리고,
#:               모르면 정규분포 백분위로 바꿔 보여준다.
OUTPUT_MODES = ("raw", "sigmoid", "percent", "zscore")


@dataclass(frozen=True)
class ModelSpec:
    model:       str               # 'A' | 'B'
    cas:         str               # 'SpCas9' | 'Cas12a'
    paths:       tuple[str, ...]   # 모델 파일 (미연동이면 빈 튜플)
    score_scale: float             # 모델 출력 → 거리 보정 점수 환산 배율
    output_mode: str = "raw"       # 위 OUTPUT_MODES 중 하나
    #: output_mode가 "zscore"일 때만 쓴다. **학습 때 타깃을 표준화하며 쓴** 평균·표준편차.
    #: 평가 데이터로 다시 구한 값이 아니라, 학습 파이프라인이 적용한 바로 그 값이어야 한다.
    #: 반드시 **0~1 척도(fraction)** 로 넣는다. 퍼센트(49.6)가 아니라 0.496이다.
    #: 출력이 여러 개인 멀티헤드 모델은 **헤드 순서대로** 튜플을 넣는다
    #: (SpCas9은 SPCAS9_VARIANTS 순서). 값 하나만 넣으면 모든 헤드에 같이 쓴다.
    #: 둘 다 채우면 효율(%)로 역변환하고, 비워두면 백분위로 표시한다.
    target_mean: float | tuple[float, ...] | None = None
    target_std:  float | tuple[float, ...] | None = None


# 경로는 프로젝트 폴더 기준 상대 경로다. 실제로 어디서 가져오는지는 model_source가 정한다.
#
# 출력 해석 (학습 담당자 확인 결과)
#   · SpCas9 (nt_v3) : z-score로 학습됨 → "zscore"
#   · Cas12a         : raw 0~1 값       → "raw" (clip해서 사용)
#
# SpCas9 학습 노트북(SpCas9_v3.ipynb cell 2)의 표준화 통계.
#   Ytr = train 타깃(published 퍼센트) / 100  → fraction
#   mu  = np.nanmean(Ytr, 0),  sd = np.nanstd(Ytr, 0)   (ddof=0, 모표준편차)
#   train.csv에서만 계산하고 val/test에 같은 값을 적용 → 누수 없음
# 순서는 SPCAS9_VARIANTS와 같고, 모델 파일의 출력 이름 순서와도 일치한다.
_SPCAS9_TARGET_MEAN = (
    0.2488796,   # SpCas9(WT)
    0.2669363,   # SpCas9-NG
    0.1934095,   # VRQR
    0.1924410,   # xCas9
    0.2381671,   # Sniper
    0.1364213,   # SpCas9-HF1
    0.1665104,   # eSpCas9(1.1)
    0.1417134,   # HypaCas9
    0.0529859,   # evoCas9
)
_SPCAS9_TARGET_STD = (
    0.2255415, 0.1500599, 0.1622795, 0.1627697, 0.2208438,
    0.1912237, 0.2013223, 0.1848161, 0.1168224,
)

MODEL_REGISTRY: dict[tuple[str, str], ModelSpec] = {
    ("A", "SpCas9"): ModelSpec(
        "A", "SpCas9", ("models/SpCas9.safetensors",), 100,
        output_mode="zscore",
        target_mean=_SPCAS9_TARGET_MEAN, target_std=_SPCAS9_TARGET_STD),
    ("A", "Cas12a"): ModelSpec(
        "A", "Cas12a", ("models/Cas12a.safetensors",), 1,
        output_mode="raw"),
}


# ════════════════════════════════════════════════
# 2. 공용 유틸
# ════════════════════════════════════════════════

BASES = "ACGT"


def one_hot_encode_dna(sequence: str) -> np.ndarray:
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([mapping.get(b, [0, 0, 0, 0]) for b in sequence.upper()])


def onehot_to_seq(row: np.ndarray) -> str:
    """[L, 4] 원핫 → 서열 문자열. 전부 0인 자리는 'N'."""
    return "".join(BASES[int(np.argmax(p))] if p.sum() > 0 else "N" for p in row)


def _normal_cdf(z: float) -> float:
    """표준정규분포 누적확률. z-score → 백분위(0~1). scipy 없이 math.erf로 계산한다."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _head_stat(value, head_index: int):
    """스칼라면 그대로, 헤드별 튜플이면 해당 헤드 값을 꺼낸다."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(value[head_index])


def _to_display_score(spec: ModelSpec, raw: float, head_index: int = 0) -> tuple[float, str]:
    """모델 원출력 → (화면에 띄울 0~100 값, 그 값의 의미).

    zscore 모델은 평균·표준편차를 알면 실제 효율로 되돌리고,
    모르면 학습 분포 안에서의 백분위로 바꾼다. 어느 쪽이든 순위는 보존된다.
    멀티헤드 모델은 헤드마다 통계가 다르므로 head_index로 골라 쓴다.
    """
    if spec.output_mode == "zscore":
        mean = _head_stat(spec.target_mean, head_index)
        std = _head_stat(spec.target_std, head_index)
        if mean is not None and std is not None:
            # 0~100 척도로 잘못 넣으면 전부 100%로 눌려 조용히 틀린다. 미리 막는다.
            if not (0.0 <= mean <= 1.5) or not (0.0 < std <= 1.5):
                raise ValueError(
                    f"{spec.cas} 헤드 {head_index}: mean={mean}, std={std} "
                    "— 0~1 척도(fraction)의 값이 아닙니다. 퍼센트라면 "
                    "100으로 나눠서 넣어 주세요 (예: 49.6% → 0.496)."
                )
            eff = float(raw) * std + mean
            return min(max(eff, 0.0), 1.0) * 100.0, "efficiency"
        return _normal_cdf(float(raw)) * 100.0, "percentile"
    return min(max(float(raw), 0.0), 1.0) * 100.0, "efficiency"


def _result(spec: ModelSpec, site: CandidateSite, cas_name: str, raw: float) -> VariantResult:
    """모델 원출력 하나를 VariantResult로 만든다.

    efficiency_pct(화면 표시용)의 의미는 score_kind가 정한다.
      · "efficiency" — 예측 절단 효율 (%)
      · "percentile" — 학습 분포 안에서의 백분위 (%)   ← z-score 모델에서 평균·표준편차가 없을 때
    final_score(거리 보정 점수) = 표시값 × 가우시안 거리 패널티 × 스케일 — 화면에는 쓰지 않는다.
    """
    head_index = SPCAS9_VARIANTS.index(cas_name) if cas_name in SPCAS9_VARIANTS else 0
    display, kind = _to_display_score(spec, raw, head_index)
    clipped = display / 100.0
    return VariantResult(
        cas_name=cas_name, pam_seq=site.pam_seq, guide_seq=site.guide_seq,
        distance=site.distance, strand=site.strand,
        raw_score=float(raw),
        final_score=clipped * gaussian_penalty(site.distance) * spec.score_scale,
        efficiency_pct=display,
        score_kind=kind,
        site_key=site.site_key, cas_type=site.cas_type,
        model=spec.model, window_seq=site.model_input_seq,
    )


@dataclass
class ExplainHandle:
    """XAI(IG·SHAP) 계산에 필요한 모델 접근 함수 묶음. 모델 종류별로 predictor가 만든다.

    raw_fn      : [N, L, 4] 원핫 → [N] 모델 원출력.
                  옵셔널 키워드 on_batch(done, total)로 배치 진행률을 받을 수 있다.
    ig_inputs   : 윈도우 서열 → IG 입력 배열 목록 (배치 차원 없음)
    ig_grad_fn  : 입력 배열 목록(각각 [N, ...]) → (출력 [N], 입력별 기울기 목록)
    project     : 입력별 기여도 목록 → 위치별 기여도 [L]
    """
    raw_fn:     Callable
    ig_inputs:  Callable
    ig_grad_fn: Callable
    project:    Callable
    #: 서열 → (층별 attention [H,T,T] 목록, 토큰별 서열구간, 서열 길이).
    #: attention이 없는 모델(CNN 등)은 None으로 둔다.
    attention_fn: Callable | None = None


class BasePredictor(ABC):
    connected = True

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @abstractmethod
    def predict_batch(self, sites: list[CandidateSite]) -> list[VariantResult]:
        ...

    def explain_heads(self, site: CandidateSite) -> list[str]:
        """이 사이트에서 설명할 수 있는 출력(변종) 이름 목록"""
        return [site.cas_type]

    def explain_handle(self, head: str) -> ExplainHandle:
        raise NotImplementedError(f"{self.spec.model}/{self.spec.cas}는 XAI를 지원하지 않습니다")


class UnconnectedPredictor(BasePredictor):
    """아직 연동되지 않은 모델. 예측 결과를 만들지 않는다."""
    connected = False

    def predict_batch(self, sites):
        return []


# ════════════════════════════════════════════════
# 3. NT 예측기
# ════════════════════════════════════════════════

@st.cache_resource(show_spinner="토크나이저 준비 중...")
def load_nt_tokenizer(repo: str = NT_TOKENIZER_REPO):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo)


def _apply_output_mode(logits, torch, mode: str):
    """모델 원출력을 공통 척도로 옮긴다.

    "raw"와 "zscore"는 그대로 둔다. zscore는 0~1로 누를 수 없는 값이라
    화면에 올릴 때(_result) 백분위나 역변환으로 처리한다.
    """
    if mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode 값이 잘못되었습니다: {mode!r} "
                         f"({' | '.join(OUTPUT_MODES)} 중 하나)")
    if mode == "sigmoid":
        return torch.sigmoid(logits)
    if mode == "percent":
        return logits / 100.0
    return logits


def _config_from_state(state: dict, num_labels: int):
    """safetensors 가중치 모양에서 EsmConfig를 복원한다."""
    from transformers import EsmConfig

    vocab, hidden = state["esm.embeddings.word_embeddings.weight"].shape
    max_pos = state["esm.embeddings.position_embeddings.weight"].shape[0]
    inter = state["esm.encoder.layer.0.intermediate.dense.weight"].shape[0]
    layers = 1 + max(int(k.split(".")[3]) for k in state
                     if k.startswith("esm.encoder.layer."))
    return EsmConfig(
        vocab_size=int(vocab), hidden_size=int(hidden),
        num_hidden_layers=int(layers),
        num_attention_heads=NT_NUM_ATTENTION_HEADS,
        intermediate_size=int(inter),
        max_position_embeddings=int(max_pos),
        num_labels=int(num_labels),
        position_embedding_type="absolute",
        emb_layer_norm_before=False,
        token_dropout=True,
        pad_token_id=1, mask_token_id=2,
        # 기본값(sdpa)은 attention 값을 돌려주지 않아 Attention Rollout을 못 쓴다.
        # 입력이 토큰 10개 안팎이라 eager로 바꿔도 속도 차이는 거의 없다.
        attn_implementation="eager",
    )


class NTClassifierPredictor(BasePredictor):
    """Nucleotide Transformer(ESM) 기반 EsmForSequenceClassification.

    출력 개수(num_labels)로 역할이 갈린다.
      · 9개 → SpCas9 변종 9종 (SPCAS9_VARIANTS 순서)
      · 1개 → 단일 Cas (Cas12a)
    """

    def __init__(self, spec):
        super().__init__(spec)
        # 지연 로딩: 이 Cas의 사이트가 실제로 나왔을 때만 모델을 올린다.
        # (예: 서열에 Cas12a PAM이 없으면 Cas12a 모델은 메모리에 올라가지 않는다)
        self._model = None
        self._num_labels = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import gc

        import torch
        from safetensors import safe_open
        from transformers import EsmForSequenceClassification

        dtype = getattr(torch, NT_DTYPE)
        path = model_source.resolve(self.spec.paths[0])

        # 텐서를 하나씩 읽어 바로 목표 자료형으로 바꾼다.
        # 파일 전체를 통째로 올린 뒤 변환하면 순간 사용량이 두 배가 된다.
        with safe_open(path, framework="pt") as f:
            state = {k: f.get_tensor(k).to(dtype) for k in f.keys()}

        num_labels = int(state["classifier.out_proj.weight"].shape[0])
        cfg = _config_from_state(state, num_labels)

        # meta 장치에 뼈대만 만든 뒤 assign=True로 가중치를 그대로 꽂는다.
        # 무작위 초기화 메모리를 쓰지 않아 최대 사용량이 파일 크기 수준에서 끝난다.
        with torch.device("meta"):
            model = EsmForSequenceClassification(cfg)
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                f"{self.spec.cas} 가중치가 모델 구조와 맞지 않습니다.\n"
                f"  채워지지 않은 가중치: {list(missing)[:5]}\n"
                f"  파일에만 있는 가중치: {list(unexpected)[:5]}"
            )

        # 가중치 파일에 없는 버퍼(position_ids 등)는 meta로 남으므로 직접 채운다.
        # 이걸 빼먹으면 오류 없이 예측값만 조용히 틀어진다.
        for name, buf in list(model.named_buffers()):
            if not buf.is_meta:
                continue
            parent, attr = name.rsplit(".", 1)
            value = (torch.arange(cfg.max_position_embeddings).expand((1, -1))
                     if attr == "position_ids"
                     else torch.zeros(buf.shape, dtype=buf.dtype))
            model.get_submodule(parent).register_buffer(attr, value, persistent=False)

        left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if t.is_meta]
        if left:
            raise RuntimeError(f"초기화되지 않은 텐서가 남았습니다: {left[:5]}")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self._model = model
        self._dtype = dtype
        self._num_labels = num_labels
        self._tokenizer = load_nt_tokenizer()
        del state
        gc.collect()

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def num_labels(self) -> int:
        self._ensure_loaded()
        return self._num_labels

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def is_multihead(self) -> bool:
        return self.num_labels == len(SPCAS9_VARIANTS)

    # ── 추론 ─────────────────────────────────────────────────────
    def _encode(self, seqs: list[str]):
        return self.tokenizer(list(seqs), return_tensors="pt", padding=True)

    def _logits(self, seqs: list[str], on_batch: Callable | None = None) -> np.ndarray:
        """서열 목록 → [N, num_labels] 출력 (설정에 따라 sigmoid 적용)

        on_batch(done, total) : 배치 하나를 끝낼 때마다 부르는 옵셔널 콜백 (ADR 0001).
            done·total은 '모델에 넣은 서열 수'다. 코어는 숫자만 넘기고,
            표시 문구와 진행률 바는 화면이 정한다. 넘기지 않으면 동작이 전과 같다.
        """
        import torch
        total = len(seqs)
        out = []
        for i in range(0, total, NT_BATCH_SIZE):
            enc = self._encode(seqs[i:i + NT_BATCH_SIZE])
            with torch.no_grad():
                logits = _apply_output_mode(self.model(**enc).logits, torch,
                                            self.spec.output_mode)
            out.append(logits.detach().float().numpy())   # bfloat16은 numpy가 못 받는다
            if on_batch is not None:
                on_batch(min(i + NT_BATCH_SIZE, total), total)
        return np.concatenate(out, axis=0)

    def predict_batch(self, sites):
        if not sites:
            return []
        preds = self._logits([s.model_input_seq for s in sites])

        results = []
        if self.is_multihead:
            for j, variant in enumerate(SPCAS9_VARIANTS):
                for site, row in zip(sites, preds):
                    if variant_compatible(variant, site.pam_seq):   # 인식하는 PAM만
                        results.append(_result(self.spec, site, variant, float(row[j])))
        else:
            for site, row in zip(sites, preds):
                results.append(_result(self.spec, site, self.spec.cas, float(row[0])))
        return results

    def explain_heads(self, site):
        if self.is_multihead:
            return [v for v in SPCAS9_VARIANTS if variant_compatible(v, site.pam_seq)]
        return [site.cas_type]

    # ── XAI ──────────────────────────────────────────────────────
    #
    # NT는 염기 원핫이 아니라 **토큰**을 입력으로 받는다 (6-mer 하나가 토큰 하나).
    # 그래서 기울기는 토큰 임베딩에 대해 구하고, 토큰 기여도를 그 토큰이 덮는
    # 염기들에 균등하게 나눠 위치별 값으로 되돌린다.
    # (2학기 Model B가 쓴 Deterministic Projection Rule과 같은 방식)

    def _token_spans(self, seq: str) -> list[tuple[int, int]]:
        """토큰 i가 덮는 서열 구간 [start, end). 특수토큰은 (0, 0)."""
        ids = self.tokenizer(seq)["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        special = set(self.tokenizer.all_special_tokens)
        spans, pos = [], 0
        for t in tokens:
            if t in special:
                spans.append((0, 0))
                continue
            n = len(t)                      # 6-mer면 6, 남은 염기면 1
            spans.append((pos, pos + n))
            pos += n
        return spans

    def _head_index(self, head: str) -> int:
        return SPCAS9_VARIANTS.index(head) if self.is_multihead else 0

    def explain_handle(self, head: str) -> ExplainHandle:
        import torch
        model = self.model
        idx = self._head_index(head)

        def raw_fn(X, on_batch: Callable | None = None):
            """[N, L, 4] 원핫 → [N]. SHAP이 섞어 만든 원핫을 서열로 되돌려 넣는다.

            on_batch는 _logits의 배치 루프로 그대로 흘려보낸다 (진행률용).
            """
            seqs = [onehot_to_seq(row) for row in np.asarray(X)]
            return self._logits(seqs, on_batch=on_batch)[:, idx]

        def ig_inputs(seq: str):
            """IG 입력 = 토큰 임베딩 [T, H]. 기준 입력은 explain.py가 0으로 만든다."""
            enc = self._encode([seq])
            with torch.no_grad():
                emb = model.esm.embeddings.word_embeddings(enc["input_ids"])
            self._ig_mask = enc["attention_mask"]
            self._ig_spans = self._token_spans(seq)
            self._ig_len = len(seq)
            return [emb[0].detach().float().numpy()]

        def ig_grad_fn(batch):
            """임베딩 [N, T, H] → (출력 [N], 임베딩 기울기 [N, T, H])"""
            x = torch.tensor(np.asarray(batch[0], dtype=np.float32),
                             dtype=self._dtype).requires_grad_(True)
            mask = self._ig_mask.expand(x.shape[0], -1)
            y = _apply_output_mode(
                model(inputs_embeds=x, attention_mask=mask).logits, torch,
                self.spec.output_mode)[:, idx]
            y.sum().backward()      # 샘플끼리 독립이라 합의 기울기 = 샘플별 기울기
            return (y.detach().float().numpy(),
                    [x.grad.detach().float().numpy()])

        def project(attrs):
            """토큰별 기여도 [T, H] → 위치별 [L]. 토큰 값을 덮는 염기 수로 나눠 배분."""
            per_token = np.asarray(attrs[0]).sum(axis=-1)      # [T]
            pos = np.zeros(self._ig_len, dtype=np.float64)
            for value, (start, end) in zip(per_token, self._ig_spans):
                if end > start:
                    stop = min(end, self._ig_len)
                    if stop > start:
                        pos[start:stop] += value / (end - start)
            return pos

        def attention_fn(seq: str):
            """층별 attention 행렬을 그대로 돌려준다. 추론 1회면 끝난다."""
            enc = self._encode([seq])
            with torch.no_grad():
                out = model(**enc, output_attentions=True)
            layers = [a[0].detach().float().numpy() for a in out.attentions]  # [H,T,T]
            return layers, self._token_spans(seq), len(seq)

        return ExplainHandle(raw_fn, ig_inputs, ig_grad_fn, project, attention_fn)


# 연동된 (모델, Cas) 조합 → 예측기 클래스
PREDICTOR_CLASSES: dict[tuple[str, str], type[BasePredictor]] = {
    ("A", "SpCas9"): NTClassifierPredictor,
    ("A", "Cas12a"): NTClassifierPredictor,
}


@st.cache_resource(show_spinner="모델 불러오는 중... (처음 한 번은 오래 걸립니다)")
def load_predictors() -> dict[tuple[str, str], BasePredictor]:
    predictors = {}
    for key, spec in MODEL_REGISTRY.items():
        cls = PREDICTOR_CLASSES.get(key)
        predictors[key] = cls(spec) if (cls and spec.paths) else UnconnectedPredictor(spec)
    return predictors


def connected_models(predictors: dict) -> set[str]:
    return {model for (model, _), p in predictors.items() if p.connected}


# ════════════════════════════════════════════════
# 4. 실행
# ════════════════════════════════════════════════

def run_prediction(all_results: ScanResult, predictors: dict) -> ScanResult:
    """연동된 모든 모델로 사이트들을 배치 예측한다."""
    all_results.variant_results = []

    for (_, cas), predictor in predictors.items():
        sites = [s for s in all_results.sites if s.cas_type == cas and s.model_input_seq]
        all_results.variant_results.extend(predictor.predict_batch(sites))

    # 사이트별 대표 값(효율이 가장 높은 변종)을 CandidateSite에 기록.
    # 변종마다 표준화 통계가 달라 원출력(z)이 커도 효율이 낮을 수 있으므로,
    # 반드시 효율 기준으로 고르고 동점일 때만 원출력으로 가른다.
    def _key(vr):
        return (vr.efficiency_pct, vr.raw_score)

    best: dict[str, VariantResult] = {}
    for vr in all_results.variant_results:
        if vr.model == "A" and (vr.site_key not in best
                                or _key(vr) > _key(best[vr.site_key])):
            best[vr.site_key] = vr
    for site in all_results.sites:
        if site.site_key in best:
            b = best[site.site_key]
            site.raw_score, site.final_score, site.efficiency_pct = \
                b.raw_score, b.final_score, b.efficiency_pct
            site.score_kind = b.score_kind

    return all_results
