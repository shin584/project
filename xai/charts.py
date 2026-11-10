"""IG / SHAP 그래프 (Matplotlib). 입력은 모두 위치별 배열이라 Cas 종류와 무관하게 쓴다.

값 단위는 '효율 %p'(모델 원출력 × 100)로 받는다. 그래프 글자는 영어로 둔다(한글 폰트 문제 회피).
"""

import matplotlib

matplotlib.use("Agg")

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

BASES = ("A", "C", "G", "T")
FIG_WIDTH = 11
SEGMENT_COLORS = {"flank": "#9aa0a6", "guide": "#4c78a8", "PAM": "#54a24b"}


def segment_of(pos: int, layout) -> str:
    for name, start, end in layout:
        if start <= pos <= end:
            return name
    return "flank"


def _margins(length: int) -> dict:
    return {"left": 0.07, "right": 0.98, "bottom": 0.22, "top": 0.85}


def _mark_segments(ax, layout, shade: bool = True) -> None:
    """flank / guide / PAM 구간을 그래프 위에 표시"""
    for name, start, end in layout:
        color = SEGMENT_COLORS[name]
        if shade:
            ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.08, zorder=0)
        ax.annotate("", xy=(start - 0.4, 1.02), xytext=(end + 0.4, 1.02),
                    xycoords=("data", "axes fraction"),
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 3})
        ax.text((start + end) / 2, 1.05, f"{name} ({start}-{end})",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                color=color, fontsize=9, fontweight="bold")


def _position_axis(ax, sequence: str) -> None:
    n = len(sequence)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{i}\n{b}" for i, b in enumerate(sequence)], fontsize=7, family="monospace")
    ax.set_xlabel("Position / base in the model input window")


def attribution_logo(ig_pct: np.ndarray, sequence: str, layout):
    """실제 염기만 그리고, 글자 높이 = 부호 있는 IG 기여도 (효율 %p)"""
    n = len(sequence)
    matrix = pd.DataFrame(0.0, index=range(n), columns=list(BASES))
    for pos, base in enumerate(sequence.upper()):
        if base in BASES:
            matrix.loc[pos, base] = ig_pct[pos]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
    logomaker.Logo(matrix, ax=ax, color_scheme="classic", flip_below=False)
    ax.axhline(0, color="#333", lw=0.8)
    lo, hi = min(0.0, float(ig_pct.min())), max(0.0, float(ig_pct.max()))
    pad = 0.1 * ((hi - lo) or 1.0)
    ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
    _mark_segments(ax, layout)
    _position_axis(ax, sequence)
    ax.set_ylabel("IG attribution (%p)\n(up = raises efficiency,\ndown = lowers it)", fontsize=8)
    fig.subplots_adjust(**_margins(n))
    return fig


def shap_bars(values_pct: np.ndarray, sequence: str, layout):
    """위치별 SHAP 값 (빨강 = 효율 상승, 파랑 = 하락)"""
    n = len(sequence)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in values_pct]
    ax.bar(range(n), values_pct, color=colors, width=0.8)
    ax.axhline(0, color="#333", lw=0.8)
    _mark_segments(ax, layout)
    _position_axis(ax, sequence)
    ax.set_ylabel("SHAP value (%p)", fontsize=8)
    fig.subplots_adjust(**_margins(n))
    return fig


def shap_waterfall(values_pct: np.ndarray, base_pct: float, sequence: str, layout, max_display: int = 12):
    """shap.plots.waterfall: 기준값에서 시작해 위치별 기여를 더해 예측값에 도달 (상위만 표시)"""
    names = [f"{i} {b} ({segment_of(i, layout)})" for i, b in enumerate(sequence.upper())]
    explanation = shap.Explanation(values=np.asarray(values_pct), base_values=float(base_pct),
                                   feature_names=names)
    plt.figure()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(FIG_WIDTH, 4.8)
    fig.tight_layout()
    return fig

def rollout_profile(profile: np.ndarray, sequence: str, layout):
    """위치별 정보 의존도. 값이 클수록 모델이 그 자리를 많이 참조했다."""
    n = len(sequence)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
    ax.bar(range(n), profile, color="#72b7b2", width=0.8)
    _mark_segments(ax, layout)
    _position_axis(ax, sequence)
    ax.set_ylabel("Information dependency\n(higher = referenced more)", fontsize=8)
    fig.subplots_adjust(**_margins(n))
    return fig


def rollout_token_heatmap(matrix: np.ndarray, labels: list):
    """토큰 × 토큰 의존도 행렬. 행이 열을 얼마나 참조하는지."""
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Depended-on token (column)", fontsize=8)
    ax.set_ylabel("Querying token (row)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
