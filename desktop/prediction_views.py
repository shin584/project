"""Model-independent Best Candidates and SpCas9 heatmap fragments (#184).

The reference is app.py's first two views. Its heatmap is a coloured table,
not a separate image. HTML, styles and all data stay inside the fragment.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

from core import CAS_CONFIGS, SPCAS9_VARIANTS, VariantResult, dna_to_rna

SCORE_LABELS = {
    "efficiency": "예측 절단 효율 (%)",
    "percentile": "학습 분포 백분위 (%)",
}
MAX_ROWS = 20


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _guide_label(row: VariantResult) -> str:
    return (
        f"{dna_to_rna(row.guide_seq)}  dist:{row.distance}  "
        f"PAM:{row.pam_seq}  strand:{row.strand}"
    )


def score_cell(value: float, precision: int) -> str:
    if math.isnan(value):
        return '<td style="background-color:transparent;color:inherit">None</td>'

    # Same fixed normalization, colour map and text contrast as pandas Styler
    # background_gradient in app.py (including its .408 luminance threshold).
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    rgba = colormaps["RdYlGn"](value / 100.0)
    rgb = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]
    ]
    luminance = sum(c * w for c, w in zip(rgb, (0.2126, 0.7152, 0.0722)))
    foreground = "#f1f1f1" if luminance < 0.408 else "#000000"
    return (
        f'<td style="background-color:{to_hex(rgba)};color:{foreground}">'
        f"{value:.{precision}f}%</td>"
    )


def _table(headers: list[str], body: list[str]) -> str:
    return (
        '<table class="grid"><thead><tr>'
        + "".join(f'<th scope="col">{_esc(h)}</th>' for h in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _limit_caption(total: int) -> str:
    if total <= MAX_ROWS:
        return ""
    return (
        f'<p class="caption">상위 {MAX_ROWS}개만 표시합니다. (전체 {total}개) '
        "전체 결과는 '전체 상세 보기' 탭을 이용하세요.</p>"
    )


def _score_groups(
    rows: Sequence[VariantResult],
) -> list[tuple[str, list[VariantResult]]]:
    """Keep the reference's Cas order; also split mixed kinds within one Cas."""
    kinds = {r.score_kind for r in rows}
    if len(kinds) <= 1:
        return [("", list(rows))] if rows else []
    groups = []
    for config in CAS_CONFIGS:
        for kind in SCORE_LABELS:
            items = [
                r for r in rows if r.cas_type == config.name and r.score_kind == kind
            ]
            if items:
                groups.append((config.name, items))
    return groups


def best_candidates_fragment(rows: Sequence[VariantResult]) -> str:
    """One model A representative per site, ranked by display score then raw."""
    best: dict[tuple[str, str], VariantResult] = {}
    for row in rows:
        if row.model != "A":
            continue
        # A kind is a unit: never choose between efficiency and percentile.
        key = (row.site_key, row.score_kind)
        current = best.get(key)
        if current is None or (row.efficiency_pct, row.raw_score) > (
            current.efficiency_pct,
            current.raw_score,
        ):
            best[key] = row

    parts = [
        (
            '<section id="best"><p>각 사이트에서 <strong>점수가 가장 높은 변종 1개</strong>만 '
            "대표로 표시합니다. 변종별 비교는 '변종 비교' 탭을 확인하세요.</p>"
        )
    ]
    if not best:
        return "".join(parts) + '<p class="empty">예측 결과가 없습니다.</p></section>'
    groups = _score_groups(list(best.values()))
    if len(groups) > 1:
        parts.append(
            '<p class="caption">Cas 종류마다 점수의 단위가 달라 표를 나눴습니다. '
            "<strong>서로 다른 표의 숫자는 직접 비교할 수 없습니다.</strong></p>"
        )
    for cas, items in groups:
        items = sorted(
            items, key=lambda r: (r.efficiency_pct, r.raw_score), reverse=True
        )
        label = SCORE_LABELS[items[0].score_kind]
        if cas:
            parts.append(f"<h3>{_esc(cas)}  ·  {len(items)}개 사이트</h3>")
        if items[0].score_kind == "percentile":
            parts.append(
                '<p class="caption">이 값은 절단 효율이 아니라 <strong>학습 분포 안에서의 '
                "백분위</strong>입니다. 높을수록 학습 데이터의 가이드들보다 우수하다는 뜻입니다.</p>"
            )
        parts.append(_limit_caption(len(items)))
        body = []
        for rank, row in enumerate(items[:MAX_ROWS], start=1):
            cells = [
                rank,
                row.cas_name,
                row.pam_seq,
                dna_to_rna(row.guide_seq),
                row.strand,
                row.distance,
            ]
            body.append(
                f'<tr data-site-key="{_esc(row.site_key)}" data-head="{_esc(row.cas_name)}">'
                + "".join(f"<td>{_esc(c)}</td>" for c in cells)
                + score_cell(round(row.efficiency_pct, 2), 2)
                + f"<td>{row.raw_score:.4f}</td></tr>"
            )
        parts.append(
            _table(
                [
                    "순위",
                    "최적 Cas 변종",
                    "PAM",
                    "가이드 서열 (RNA)",
                    "가닥",
                    "거리 (bp)",
                    label,
                    "원출력",
                ],
                body,
            )
        )
    parts.append(
        "<p class=\"caption\">'원출력'은 모델이 낸 값 그대로입니다. SpCas9은 z-score라 "
        "변종마다 평균·표준편차가 달라서, 원출력이 낮아도 효율이 더 높을 수 있습니다.</p></section>"
    )
    return "".join(parts)


def variant_comparison_fragment(rows: Sequence[VariantResult]) -> str:
    """Guide × present SpCas9 variants, max aggregation and row-max ordering."""
    import pandas as pd

    sp = [r for r in rows if r.cas_type == "SpCas9" and r.cas_name in SPCAS9_VARIANTS]
    if not sp:
        return (
            '<section id="variants"><p class="empty">'
            "SpCas9 PAM 사이트가 발견되지 않았습니다.</p></section>"
        )
    parts = ['<section id="variants">']
    for _, items in _score_groups(sp):
        label = SCORE_LABELS[items[0].score_kind]
        parts.append(
            f"<p>행: 가이드 서열 / 열: SpCas9 변종. 값은 <strong>{label}</strong> 이며, "
            "<strong>초록에 가까울수록 높고, 빨강에 가까울수록 낮습니다.</strong> "
            "색이 칠해지지 않고 <strong>'None'</strong> 으로 표시된 칸은 "
            "그 변종이 인식하지 않는 PAM이라 예측 자체가 없는 자리입니다.</p>"
            '<p class="caption">Cas12a는 변종이 없어 비교할 대상이 없습니다. '
            "결과는 'Best Candidates'와 '전체 상세 보기' 탭에서 확인하세요.</p>"
        )
        data = pd.DataFrame(
            [
                {
                    "guide_label": _guide_label(r),
                    "variant": r.cas_name,
                    "score": round(r.efficiency_pct, 2),
                }
                for r in items
            ]
        )
        site_keys = {_guide_label(r): r.site_key for r in items}
        present = data["variant"].unique()
        pivot = (
            data.pivot_table(
                index="guide_label", columns="variant", values="score", aggfunc="max"
            )
            .reindex(columns=[v for v in SPCAS9_VARIANTS if v in present])
            .assign(_max=lambda d: d.max(axis=1))
            .sort_values("_max", ascending=False)
            .drop(columns="_max")
        )
        parts.append(_limit_caption(len(pivot)))
        body = [
            f'<tr data-site-key="{_esc(site_keys[guide])}"><th scope="row">{_esc(guide)}</th>'
            + "".join(score_cell(float(value), 1) for value in values)
            + "</tr>"
            for guide, *values in pivot.head(MAX_ROWS).itertuples(name=None)
        ]
        parts.append(_table(["guide_label", *pivot.columns.tolist()], body))
    return "".join(parts) + "</section>"
