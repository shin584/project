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

#: Streamlit 화면이 내려주던 열. **순서와 의미를 바꾸지 않는다** — `app.py:353-364`의
#: 기준이고, `docs/equivalence-baseline/`의 보존된 CSV가 이 열들로 떠 있다.
BASE_COLUMNS = [
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

#: #196에서 **맨 끝에** 더한 열. 값은 예측 행이 들고 온 모델 식별에서 나온다
#: (`CONTEXT.md`의 '모델 식별'). 모델을 바꾸면 예측값과 순위가 달라지므로, 저장된
#: 표가 어느 모델·어느 checkpoint의 결과인지 그 파일만 보고 알 수 있어야 한다.
MODEL_COLUMNS = ["모델", "모델 버전"]

DETAIL_COLUMNS = BASE_COLUMNS + MODEL_COLUMNS

#: 예측기를 거치지 않은 행(미리보기 등)의 모델 열. 모델 이름을 꾸며 넣지 않는다.
UNKNOWN_MODEL = "—"
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
    identity = row.identity
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
        identity.name if identity else UNKNOWN_MODEL,
        identity.version if identity else UNKNOWN_MODEL,
    ]


def detail_table(rows: Sequence[VariantResult]) -> list[list[Any]]:
    """Header plus one list per row, ranks renumbered from 1.

    The caller filters first; Streamlit numbers the rows it kept, so a CSV of a
    filtered table counts 1..N rather than carrying the full list's ranks.
    """
    return [list(DETAIL_COLUMNS)] + [
        _cells(row, rank) for rank, row in enumerate(rows, start=1)
    ]


def base_columns_only(table: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """`BASE_COLUMNS`만 남긴 표. 보존된 기준 CSV와 대조할 때 쓴다 (#196).

    #196이 모델·버전 열을 끝에 더하면서 저장 파일이 기준 CSV와 바이트로 같지 않게
    됐다. 기준이 증거로 남긴 것은 **기존 열의 값과 순서**이므로, 그 부분만 떼어
    비교한다. 양쪽이 같은 자리에서 자르도록 함수는 하나만 둔다.
    """
    width = len(BASE_COLUMNS)
    return [list(row[:width]) for row in table]


def csv_rows(data: bytes) -> list[list[str]]:
    """저장된 CSV 바이트 → 셀 목록. `desktop.api.csv_bytes`의 역이다."""
    import csv
    import io

    return list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def csv_against_baseline(saved: bytes, expected: bytes) -> dict:
    """저장된 CSV를 보존된 기준과 대조한다 (#196).

    모델·버전 열이 끝에 붙었으므로 파일 전체가 바이트로 같지는 않다. 기준이 증거로
    남긴 것은 기존 열의 값과 순서이니 그 부분을 바이트로 비교하고, 더해진 열은 따로
    보고한다. `desktop/validation.py`, `docs/equivalence-baseline/compare_baseline.py`,
    `tests/test_nt_process.py`가 **이 함수 하나**를 함께 쓴다.

    **기준 쪽 바이트는 건드리지 않는다.** 양쪽을 다시 직렬화하면 인용 규칙이나 줄
    끝이 달라도 통과해 버려서, 바이트 비교가 아니라 왕복 비교가 된다. 열을 떼어내야
    하는 저장 파일만 다시 직렬화하고, 보존된 기준의 원래 바이트와 맞춘다.
    """
    from desktop.api import csv_bytes

    saved_rows = csv_rows(saved)
    return {
        "csv_base_columns_byte_equal": csv_bytes(base_columns_only(saved_rows)) == expected,
        "csv_whole_file_byte_equal": saved == expected,
        "csv_model_columns": [row[-len(MODEL_COLUMNS):] for row in saved_rows[:2]],
    }


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
    raw = BASE_COLUMNS.index("원출력")
    score = BASE_COLUMNS.index(EFF)
    body = []
    for rank, row in enumerate(rows, start=1):
        cells = _cells(row, rank)
        # 원출력은 반올림 전 값을 그대로 보여주고, 점수 칸만 색을 입힌다.
        display = [*cells[:raw], f"{row.raw_score:.4f}"]
        body.append(
            f'<tr data-site-key="{_esc(row.site_key)}" data-head="{_esc(row.cas_name)}"'
            f' data-cas-name="{_esc(row.cas_name)}"'
            f' data-raw="{_esc(row.raw_score)}"'
            f' data-eff="{_esc(row.efficiency_pct)}" data-dist="{_esc(row.distance)}">'
            + "".join(f"<td>{_esc(cell)}</td>" for cell in display)
            + score_cell(cells[score], 2)
            + "".join(f"<td>{_esc(cell)}</td>" for cell in cells[score + 1:])
            + "</tr>"
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
        '<label for="detail-eff-low">최소 점수 (%)</label>'
        f'<input id="detail-eff-low" type="range" min="{eff_low:g}" max="{eff_high:g}"'
        f' step="0.5" value="{eff_low:g}" aria-describedby="detail-eff-caption">'
        f'<div class="caption" id="detail-eff-caption">{eff_low:g}% 이상</div>'
        '<label for="detail-dist-high">최대 거리 (bp)</label>'
        f'<input id="detail-dist-high" type="range" min="{dist_low}" max="{dist_high}"'
        f' step="1" value="{dist_high}" aria-describedby="detail-dist-caption">'
        f'<div class="caption" id="detail-dist-caption">{dist_high} bp 이하</div></fieldset></div>'
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
    parts = ['<section id="detail">']
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
