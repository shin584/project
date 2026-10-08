"""Matplotlib figure / table builders for the Model B explanation dashboard (issue #20).

Each function takes plain arrays from a `SampleBundle` and returns a Figure,
so the Streamlit layer only arranges them. Position-indexed charts share
`POSITION_XLIM` and `FIG_WIDTH` so they line up when stacked vertically.
Chart text is English to avoid missing-glyph issues with CJK fonts.
"""

import matplotlib

matplotlib.use("Agg")

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from data_source import (
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
# Complex mismatch scenario regions (mismatch_profiling.py) -> bar colour.
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
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("(mutant - WT) / |WT|", fontsize=8)
    fig.tight_layout()
    return fig


def complex_mismatch_bars(delta: np.ndarray, scenarios: list[ComplexScenario]):
    """Score change for each of the 15 complex mismatch scenarios, coloured by region."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))
    colors = [SCENARIO_REGION_COLORS[s.region] for s in scenarios]
    y = np.arange(len(scenarios))
    ax.barh(y, delta, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([s.mutation_id for s in scenarios], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Relative score change (mutant - WT) / |WT|")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in SCENARIO_REGION_COLORS.values()
    ]
    ax.legend(handles, list(SCENARIO_REGION_COLORS), fontsize=8, loc="best")
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
    ax.set_ylabel("IG attribution")
    fig.subplots_adjust(**POSITION_MARGINS)
    return fig


def rollout_profile(attention_rollout: np.ndarray, sequence: str):
    """Per-position Information Dependency: how much every position draws on each one."""
    profile = attention_rollout.sum(axis=0)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
    ax.bar(range(SEQ_LEN), profile, color="#72b7b2", width=0.8)
    _mark_regions(ax)
    _position_axis(ax, sequence)
    ax.set_ylabel("Summed dependency")
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


def prediction_detail_table(bundle: SampleBundle) -> pd.DataFrame:
    """Tab 3's one-column table; truth/error rows only when the sample has them."""
    rows = {"sample_id": str(bundle.sample_id), "서열": bundle.sequence}
    rows["Model B 예측값"] = f"{bundle.prediction:.4f}"
    if bundle.true_score is not None:
        rows["실측값"] = f"{bundle.true_score:.4f}"
        rows["오차"] = f"{bundle.error:.4f}"
    for key in PHYSICAL_FEATURE_KEYS:
        rows[PHYSICAL_FEATURE_LABELS[key]] = f"{bundle.physical_features[key]:.3f}"
    return pd.DataFrame({"값": rows})


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
