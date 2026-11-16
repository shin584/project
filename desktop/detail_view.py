"""Model-independent '전체 상세 보기' table, filter and CSV (#185).

The reference is app.py's third view (`app.py:294-391`). Streamlit filters on
the search button and offers the *filtered* table as the CSV, so this module
holds the one filter predicate Python uses for the CSV. The screen repeats it
on the `data-*` values each row carries; those are the unrounded values used
here, so both sides draw the same boundaries.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core import VariantResult, dna_to_rna

from desktop.prediction_views import score_cell

EFF = "예측 점수 (%)"
DETAIL_COLUMNS = [
    "순위",
    "Cas 변종",
    "PAM",
    "가이드 서열 (RNA)",
    "가닥",
    "거리 (bp)",
    "원출력",
    EFF,
    "점수 기준",
]
_SCORE_KIND_LABEL = {"efficiency": "절단 효율", "percentile": "학습 분포 백분위"}

#: Streamlit slider limits (`app.py:327-335`).
EFF_LIMITS = (0.0, 100.0)
DIST_LIMITS = (0, 15)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


@dataclass(frozen=True)
class DetailFilter:
    """Streamlit's three detail filters. `cas_names=None` means all selected."""

    cas_names: frozenset[str] | None = None
    eff_range: tuple[float, float] = EFF_LIMITS
    dist_range: tuple[int, int] = DIST_LIMITS

    @classmethod
    def parse(cls, raw: dict[str, Any] | None) -> DetailFilter:
        """Read the screen's `filters` value, rejecting what the controls can't send.

        Everything here arrives from JS across the bridge, so a wrong shape is
        bad data rather than a programming error: one `ValueError` for all of
        it, which the dispatcher turns into one `error` event for the screen.
        """
        if raw is None:
            return cls()
        # `not raw` would also swallow [] and 0 as "no filter", quietly saving
        # every row when the screen sent something malformed.
        if not isinstance(raw, dict):
            raise ValueError(f"필터 형식이 잘못됐습니다: {raw!r}")
        unknown = set(raw) - {"cas_names", "eff_range", "dist_range"}
        if unknown:
            raise ValueError(f"모르는 필터: {', '.join(sorted(unknown))}")
        names = raw.get("cas_names")
        if names is not None and (
            not isinstance(names, list) or not all(isinstance(n, str) for n in names)
        ):
            raise ValueError("cas_names는 변종 이름 목록이어야 합니다")
        return cls(
            cas_names=None if names is None else frozenset(names),
            eff_range=_range(raw.get("eff_range", EFF_LIMITS), EFF_LIMITS, "점수"),
            dist_range=_whole(
                _range(raw.get("dist_range", DIST_LIMITS), DIST_LIMITS, "거리"), "거리"
            ),
        )

    def matches(self, row: VariantResult) -> bool:
        # Same predicate as app.py:341-346, on the unrounded score.
        return (
            (self.cas_names is None or row.cas_name in self.cas_names)
            and self.eff_range[0] <= row.efficiency_pct <= self.eff_range[1]
            and self.dist_range[0] <= row.distance <= self.dist_range[1]
        )

    def apply(self, rows: Sequence[VariantResult]) -> list[VariantResult]:
        return [r for r in rows if self.matches(r)]


def _range(value: Any, limits: tuple[float, float], label: str) -> tuple[float, float]:
    """One slider's `[low, high]`, rejecting what the control cannot produce."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
    ):
        raise ValueError(f"{label} 범위는 [최소, 최대] 두 숫자여야 합니다: {value!r}")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{label} 범위의 최소가 최대보다 큽니다: {low} > {high}")
    if low < limits[0] or high > limits[1]:
        raise ValueError(
            f"{label} 범위는 {limits[0]}~{limits[1]} 안이어야 합니다: {value!r}"
        )
    return low, high


def _whole(value: tuple[float, float], label: str) -> tuple[int, int]:
    if any(not v.is_integer() for v in value):
        raise ValueError(f"{label} 범위는 정수여야 합니다: {value!r}")
    return int(value[0]), int(value[1])


# ════════════════════════════════════════════════
# 표와 CSV
# ════════════════════════════════════════════════


def _cells(row: VariantResult, rank: int) -> list[Any]:
    """One row's values, in `DETAIL_COLUMNS` order (`app.py:353-364`).

    The saved CSV and the drawn table both come from here. '저장된 CSV의 열·순서·
    값이 Streamlit의 CSV와 일치한다'는 #185의 완료 기준이라, 두 쪽이 열을 따로
    세면 조용히 어긋날 수 있다.
    """
    return [
        rank,
        row.cas_name,
        row.pam_seq,
        dna_to_rna(row.guide_seq),
        row.strand,
        row.distance,
        round(row.raw_score, 4),
        round(row.efficiency_pct, 2),
        _SCORE_KIND_LABEL[row.score_kind],
    ]


def detail_table(rows: Sequence[VariantResult]) -> list[list[Any]]:
    """Header plus one list per row, ranks renumbered from 1.

    The caller filters first; Streamlit numbers the rows it kept, so a CSV of a
    filtered table counts 1..N rather than carrying the full list's ranks.
    """
    return [list(DETAIL_COLUMNS)] + [
        _cells(row, rank) for rank, row in enumerate(rows, start=1)
    ]


# ════════════════════════════════════════════════
# 조각
# ════════════════════════════════════════════════


def _rows_html(rows: Sequence[VariantResult]) -> str:
    """One `<tr>` per row, carrying what the screen filters and highlights on.

    `data-eff` is the **반올림 전** score, the value Streamlit's filter reads
    (`app.py:341-346`). Filtering on the displayed 51.99 instead would keep rows
    the Streamlit screen drops at a 52.0 lower bound.

    `data-cas-name`은 필터가 보는 값이고 `data-head`는 XAI가 보는 값이다. 지금은
    두 값이 같지만 **같은 것이 아니다** — 헤드는 모델 출력의 인덱스, `cas_name`은
    그 예측 행의 이름이다 (`CONTEXT.md`의 '헤드'·'예측 행'). 한 속성을 둘로 겸해
    쓰면 둘이 갈라지는 날 조용히 틀린다.
    """
    body = []
    for rank, row in enumerate(rows, start=1):
        cells = _cells(row, rank)
        display = [*cells[:6], f"{row.raw_score:.4f}"]
        body.append(
            f'<tr data-site-key="{_esc(row.site_key)}" data-head="{_esc(row.cas_name)}"'
            f' data-cas-name="{_esc(row.cas_name)}"'
            f' data-eff="{_esc(row.efficiency_pct)}" data-dist="{_esc(row.distance)}">'
            + "".join(f"<td>{_esc(cell)}</td>" for cell in display)
            + score_cell(cells[7], 2)
            + f"<td>{_esc(cells[8])}</td></tr>"
        )
    return "".join(body)


def _filters_html(rows: Sequence[VariantResult]) -> str:
    """Streamlit's filter block (`app.py:303-340`) as plain controls.

    Streamlit's two-handle slider has no native HTML equivalent, so each range
    has lower/upper sliders with the same limits and step. The values a user can
    reach are the same, and `DetailFilter` rejects anything outside them.
    """
    boxes = "".join(
        f'<label><input type="checkbox" name="detail-cas" value="{_esc(name)}" checked>'
        f"{_esc(name)}</label>"
        for name in sorted({row.cas_name for row in rows})
    )
    eff_low, eff_high = EFF_LIMITS
    dist_low, dist_high = DIST_LIMITS
    return (
        '<form id="detail-filters"><h3>필터 설정</h3><div class="filter-columns">'
        "<fieldset><legend>Cas 변종 선택</legend>"
        '<p><button type="button" id="detail-all">전체 선택</button> '
        '<button type="button" id="detail-none">전체 해제</button></p>'
        f'<div class="checklist">{boxes}</div></fieldset>'
        "<fieldset><legend>효율 범위 및 거리 설정</legend>"
        '<label for="detail-eff-low">점수 범위 (%)</label>'
        f'<p><input id="detail-eff-low" type="range" aria-label="최소 점수 (%)" min="{eff_low}" max="{eff_high}"'
        f' step="0.5" value="{eff_low}"> ~ '
        f'<input id="detail-eff-high" type="range" aria-label="최대 점수 (%)" min="{eff_low}" max="{eff_high}"'
        f' step="0.5" value="{eff_high}"></p>'
        f'<p class="caption" id="detail-eff-caption">선택 범위: {eff_low:.1f}% ~ '
        f"{eff_high:.1f}%</p>"
        '<label for="detail-dist-low">거리 범위 (bp)</label>'
        f'<p><input id="detail-dist-low" type="range" aria-label="최소 거리 (bp)" min="{dist_low}"'
        f' max="{dist_high}" step="1" value="{dist_low}"> ~ '
        f'<input id="detail-dist-high" type="range" aria-label="최대 거리 (bp)" min="{dist_low}"'
        f' max="{dist_high}" step="1" value="{dist_high}"></p>'
        f'<p class="caption" id="detail-dist-caption">변이로부터 {dist_low} ~ '
        f"{dist_high} bp 이내</p></fieldset></div>"
        '<p><button type="submit" id="detail-search">검색</button> '
        # Streamlit은 검색을 누른 뒤에야 다운로드 버튼을 그린다 (`app.py:385`).
        '<button type="button" id="detail-csv" disabled>CSV 다운로드</button></p>'
        '<p class="caption">CSV는 지금 검색해 둔 결과를 그대로 저장합니다.</p></form>'
    )


def detail_fragment(rows: Sequence[VariantResult]) -> str:
    """The whole view: filters, then the table the search button reveals.

    Every row travels to the screen once. The screen hides the rows its filters
    drop and renumbers what is left, so searching never asks Python to predict
    again — only '저장' does, and it re-applies the same filter there.
    """
    parts = [
        '<section id="detail"><h2>전체 상세 보기</h2>',
        "<p>원하는 조건을 설정한 뒤 <strong>검색 버튼</strong>을 눌러 결과를 확인하세요.</p>",
    ]
    if not rows:
        return "".join(parts) + '<p class="empty">예측 결과가 없습니다.</p></section>'
    head = "".join(f'<th scope="col">{_esc(name)}</th>' for name in DETAIL_COLUMNS)
    parts.append(_filters_html(rows))
    parts.append(
        f'<div id="detail-results" hidden data-total="{len(rows)}">'
        '<p class="caption" id="detail-count"></p>'
        # Hidden with the table when nothing matches: Streamlit draws this
        # caption inside `if filtered:` and never beside the warning.
        '<p class="caption" id="detail-highlight">노란 줄(<span id="detail-selected">0</span>개)은 '
        "'XAI 설명' 탭에서 선택한 사이트입니다. 한 사이트를 여러 변종이 인식하면 "
        "변종마다 한 줄씩 나오므로 여러 줄이 칠해집니다. "
        "<strong>'점수 기준' 열이 다른 행끼리는 점수를 직접 비교할 수 없습니다.</strong></p>"
        f'<table class="grid"><thead><tr>{head}</tr></thead>'
        f"<tbody>{_rows_html(rows)}</tbody></table>"
        # st.warning in the reference (`app.py:392`), so it is styled as one.
        '<p class="warning" id="detail-no-match" hidden>필터 조건에 맞는 결과가 없습니다.</p>'
        "</div></section>"
    )
    return "".join(parts)
