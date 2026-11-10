"""Model B(SaCas9) 사전계산 결과용 Matplotlib 그래프/표. 2학기 charts.py를 그대로 옮겼다.

각 함수는 `SampleBundle`의 평범한 배열을 받아 Figure를 돌려주므로, Streamlit 쪽은
배치만 한다. 위치 기준 그래프들은 `POSITION_XLIM`과 `FIG_WIDTH`를 공유해 세로로
쌓았을 때 x축이 맞는다. 그래프 글자는 CJK 폰트 문제를 피하려고 영어로 둔다.

[2학기 병합] 실제 연동 모델(SpCas9 · Cas12a)의 IG·SHAP 그래프는 `xai/charts.py`에
따로 있다. 이름이 겹치는 함수(attribution_logo 등)가 있으므로 두 모듈을 섞지 말 것.
"""

import matplotlib

matplotlib.use("Agg")

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.ticker import PercentFormatter

from xai.data_source import (
    BASES,
    DISTAL_REGION,
    PAM_REGION,
    PHYSICAL_FEATURE_KEYS,
    PHYSICAL_FEATURE_LABELS,
    SEED_REGION,
    SEQ_LEN,
    ComplexScenario,
    SampleBundle,
    ShapGroup,
)

FIG_WIDTH = 11
POSITION_XLIM = (-0.5, SEQ_LEN - 0.5)
REGIONS = (
    ("Distal", DISTAL_REGION, "#4c78a8"),
    ("Seed", SEED_REGION, "#f58518"),
    ("PAM", PAM_REGION, "#54a24b"),
)
# Complex mismatch scenario regions -> bar colour.
SCENARIO_REGION_COLORS = {
    "Seed": "#f58518",
    "Distal": "#4c78a8",
    "Intermittent": "#9d755d",
}
# Fixed margins (instead of tight_layout) so stacked position charts share
# exactly the same x-axis placement.
POSITION_MARGINS = {"left": 0.07, "right": 0.98, "bottom": 0.22, "top": 0.85}


def _mark_regions(ax, *, shade: bool = True) -> None:
    """Label Distal/Seed/PAM above the axes, and optionally shade them."""
    for name, (start, end), color in REGIONS:
        if shade:
            ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.08, zorder=0)
        ax.annotate(
            "",
            xy=(start - 0.4, 1.02),
            xytext=(end + 0.4, 1.02),
            xycoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "-", "color": color, "lw": 3},
        )
        ax.text(
            (start + end) / 2,
            1.05,
            f"{name} ({start}-{end})",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=color,
            fontsize=9,
            fontweight="bold",
        )


def _position_axis(ax, sequence: str) -> None:
    ax.set_xlim(*POSITION_XLIM)
    ax.set_xticks(range(SEQ_LEN))
    ax.set_xticklabels(
        [f"{i}\n{b}" for i, b in enumerate(sequence)], fontsize=7, family="monospace"
    )
    ax.set_xlabel("Position / reference base")


def ism_heatmap(
    ism_delta: np.ndarray, alt_bases: tuple[tuple[str, str, str], ...], sequence: str
):
    """36x3 ISM relative-delta heatmap, each cell labelled with its substituted base."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.8))
    values = ism_delta.T  # (3, 36)
    vmax = float(np.nanmax(np.abs(values))) or 1.0
    im = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=(-0.5, SEQ_LEN - 0.5, 2.5, -0.5),
    )
    for pos, alts in enumerate(alt_bases):
        for slot, base in enumerate(alts):
            dark = abs(values[slot, pos]) > 0.6 * vmax
            ax.text(
                pos,
                slot,
                base,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if dark else "#333",
            )
    _mark_regions(ax, shade=False)
    for _, (start, end), color in REGIONS:
        ax.axvline(start - 0.5, color=color, lw=1.2)
        ax.axvline(end + 0.5, color=color, lw=1.2)
    _position_axis(ax, sequence)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["alt 1", "alt 2", "alt 3"], fontsize=8)
    ax.set_ylabel("Substituted base", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    cbar.set_label("Score change (red = up, blue = down)", fontsize=8)
    fig.tight_layout()
    return fig


def complex_mismatch_bars(delta: np.ndarray, scenarios: list[ComplexScenario]):
    """Score change for each of the 15 complex mismatch scenarios, coloured by region."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))
    colors = [SCENARIO_REGION_COLORS[s.region] for s in scenarios]
    y = np.arange(len(scenarios))
    bars = ax.barh(y, delta, color=colors)
    ax.bar_label(bars, labels=[f"{v:+.0%}" for v in delta], fontsize=7, padding=2)
    ax.margins(x=0.12)
    ax.set_yticks(y)
    ax.set_yticklabels([s.mutation_id for s in scenarios], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", lw=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Score change vs. original (left = worse, right = better)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in SCENARIO_REGION_COLORS.values()
    ]
    ax.legend(
        handles, list(SCENARIO_REGION_COLORS), fontsize=8, loc="best", title="Region"
    )
    fig.tight_layout()
    return fig


def attribution_logo(ig_attribution: np.ndarray, sequence: str):
    """Attribution Logo: only the actual base at each position, height = signed IG."""
    matrix = pd.DataFrame(0.0, index=range(SEQ_LEN), columns=list(BASES))
    for pos, base in enumerate(sequence):
        matrix.loc[pos, base] = ig_attribution[pos]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
    logomaker.Logo(matrix, ax=ax, color_scheme="classic", flip_below=False)
    ax.axhline(0, color="#333", lw=0.8)
    lo, hi = min(0.0, ig_attribution.min()), max(0.0, ig_attribution.max())
    pad = 0.1 * ((hi - lo) or 1.0)
    ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
    _mark_regions(ax)
    _position_axis(ax, sequence)
    ax.set_ylabel(
        "IG attribution\n(up = raises score,\ndown = lowers score)", fontsize=8
    )
    fig.subplots_adjust(**POSITION_MARGINS)
    return fig


def rollout_profile(attention_rollout: np.ndarray, sequence: str):
    """Per-position Information Dependency: how much every position draws on each one."""
    profile = attention_rollout.sum(axis=0)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
    ax.bar(range(SEQ_LEN), profile, color="#72b7b2", width=0.8)
    _mark_regions(ax)
    _position_axis(ax, sequence)
    ax.set_ylabel("Summed dependency\n(higher = referenced more)", fontsize=8)
    fig.subplots_adjust(**POSITION_MARGINS)
    return fig


def rollout_heatmap(attention_rollout: np.ndarray):
    """The raw 36x36 Information Dependency matrix (row draws on column)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attention_rollout, cmap="viridis")
    ticks = range(0, SEQ_LEN, 3)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Depended-on position (column)")
    ax.set_ylabel("Querying position (row)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


PREDICTION_DETAIL_HINTS = {
    "sample_id": "Testset 안에서의 샘플 번호",
    "서열": "분석 대상 36염기 DNA 서열",
    "Model B 예측값": "모델이 예측한 절단 효율 점수 (높을수록 잘 잘림)",
    "실측값": "실험으로 측정된 실제 절단 효율 점수",
    "오차": "예측과 실측의 차이 (작을수록 정확)",
    "MFE": "gRNA가 스스로 접힐 때의 에너지 (더 음수일수록 단단히 접힘)",
    "ΔG": "gRNA–DNA 결합 에너지 (더 음수일수록 결합이 안정적)",
    "Tm": "결합이 풀리기 시작하는 온도 (높을수록 안정적)",
    "GC": "G·C 염기의 비율(%)",
}


def prediction_detail_table(bundle: SampleBundle) -> pd.DataFrame:
    """Model B 샘플 하나의 상세 표; 실측/오차 행은 값이 있을 때만."""
    rows = {"sample_id": str(bundle.sample_id), "서열": bundle.sequence}
    rows["Model B 예측값"] = f"{bundle.prediction:.4f}"
    if bundle.true_score is not None:
        rows["실측값"] = f"{bundle.true_score:.4f}"
        rows["오차"] = f"{bundle.error:.4f}"
    for key in PHYSICAL_FEATURE_KEYS:
        rows[PHYSICAL_FEATURE_LABELS[key]] = f"{bundle.physical_features[key]:.3f}"
    return pd.DataFrame(
        {
            "값": rows,
            "의미": {k: PREDICTION_DETAIL_HINTS.get(k, "") for k in rows},
        }
    )


def shap_waterfall(groups: tuple[ShapGroup, ...], base_value: float):
    """Token-grouped SHAP rendered with `shap.plots.waterfall` (11 signed groups)."""
    explanation = shap.Explanation(
        values=np.array([g.shap_value for g in groups]),
        base_values=base_value,
        feature_names=[g.label for g in groups],
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=len(groups), show=False)
    fig = plt.gcf()
    fig.set_size_inches(FIG_WIDTH, 6)
    fig.tight_layout()
    return fig
