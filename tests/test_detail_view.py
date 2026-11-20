"""#185: 전체 상세 보기 · 필터 · CSV. 모델 가중치도 WebView도 없이 돈다.

기준은 Streamlit 화면의 '전체 상세 보기'(`app.py:294-391`)와 #181이 뜬 CSV다.
"""

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest
from core import VariantResult
from desktop.api import csv_bytes
from core import ModelIdentity
from desktop.detail_view import (
    BASE_COLUMNS,
    DETAIL_COLUMNS,
    MODEL_COLUMNS,
    UNKNOWN_MODEL,
    DetailFilter,
    base_columns_only,
    csv_against_baseline,
    csv_rows,
    detail_fragment,
    detail_table,
)

MODEL = ModelIdentity(family="nt", cas="SpCas9", name="NT SpCas9",
                      version="sha256 0123456789ab")
CAS12A = ModelIdentity(family="nt", cas="Cas12a", name="NT Cas12a",
                       version="sha256 0123456789ab")

BASELINE = Path(__file__).resolve().parent.parent / "docs" / "equivalence-baseline"


def row(name, eff, *, dist=3, raw=0.0, kind="efficiency", site="SpCas9|+|40",
        cas="SpCas9", guide="ATCGATCGATCGATCGATCG", pam="AGG", strand="+",
        identity=None):
    return VariantResult(
        cas_name=name, pam_seq=pam, guide_seq=guide, distance=dist, strand=strand,
        raw_score=raw, efficiency_pct=eff, score_kind=kind, site_key=site,
        cas_type=cas, model="A", identity=identity,
    )


class Rows(HTMLParser):
    """조각의 상세 표에서 각 행의 속성과 칸 글자를 모은다."""

    def __init__(self, fragment):
        super().__init__()
        self.head: list[str] = []
        self.rows: list[dict] = []
        self.cell: list[str] | None = None
        self.in_head = False
        self.feed(fragment)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "thead":
            self.in_head = True
        elif tag == "tr" and not self.in_head:
            self.rows.append({"attrs": attrs, "cells": [], "styles": []})
        elif tag in {"td", "th"}:
            self.cell = []
            if not self.in_head:
                self.rows[-1]["styles"].append(attrs.get("style", ""))

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "thead":
            self.in_head = False
        elif tag in {"td", "th"} and self.cell is not None:
            text = "".join(self.cell)
            (self.head if self.in_head else self.rows[-1]["cells"]).append(text)
            self.cell = None


# ── 표 ─────────────────────────────────────────────


def test_table_uses_the_streamlit_columns_values_and_ranks():
    rows = [row("SpCas9(WT)", 51.98895989939809, raw=1.6856821775436401, identity=MODEL)]
    assert detail_table(rows) == [
        DETAIL_COLUMNS,
        [1, "SpCas9(WT)", "AGG", "AUCGAUCGAUCGAUCGAUCG", "+", 3, 1.6857, 51.99, "절단 효율",
         "NT SpCas9", "sha256 0123456789ab"],
    ]
    # #196은 기존 열을 그 순서대로 두고 모델·버전만 **맨 끝에** 더한다.
    assert BASE_COLUMNS == [
        "순위", "Cas 변종", "PAM", "가이드 서열 (RNA)", "가닥", "거리 (bp)",
        "원출력", "예측 점수 (%)", "점수 기준",
    ]
    assert DETAIL_COLUMNS == BASE_COLUMNS + MODEL_COLUMNS
    assert MODEL_COLUMNS == ["모델", "모델 버전"]


def test_rows_without_a_predictor_claim_no_model():
    """미리보기·가짜 행이 모델 이름을 꾸며 내지 않는다 (#196)."""
    cells = detail_table([row("SpCas9(WT)", 10.0)])[1]
    assert cells[-2:] == [UNKNOWN_MODEL, UNKNOWN_MODEL]


def test_each_row_names_the_model_that_produced_it():
    """한 CSV에 Cas가 섞여도 모델 열은 행마다 그 행을 낸 모델을 가리킨다."""
    other = ModelIdentity(family="nt", cas="Cas12a", name="NT Cas12a", version="sha256 cafe")
    rows = [row("SpCas9(WT)", 60.0, identity=MODEL),
            row("Cas12a", 70.0, cas="Cas12a", site="Cas12a|+|16", identity=other)]
    assert [cells[-2:] for cells in detail_table(rows)[1:]] == [
        ["NT SpCas9", "sha256 0123456789ab"], ["NT Cas12a", "sha256 cafe"]]


def test_each_row_keeps_its_own_score_kind():
    """점수 단위가 다른 행이 한 표에 있어도 각 행의 '점수 기준'이 제 단위를 말한다."""
    rows = [
        row("Cas12a", 70.0, kind="efficiency", cas="Cas12a", site="Cas12a|+|16"),
        row("SpCas9(WT)", 60.0, kind="percentile"),
    ]
    kind_column = DETAIL_COLUMNS.index("점수 기준")
    kinds = [cells[kind_column] for cells in detail_table(rows)[1:]]
    assert kinds == ["절단 효율", "학습 분포 백분위"]
    fragment = detail_fragment(rows)
    assert [r["cells"][kind_column] for r in Rows(fragment).rows] == kinds
    assert "'점수 기준' 열이 다른 행끼리는 점수를 직접 비교할 수 없습니다." in fragment


def test_csv_is_byte_identical_to_the_streamlit_baseline():
    """#181 기준 CSV의 행을 다시 만들어 쓰면 바이트까지 같아야 한다 (게이트 2)."""
    captured = json.loads((BASELINE / "candidates.json").read_text(encoding="utf-8"))
    rows = [
        row(
            r["Cas 변종"], r["예측 점수 (%)"], dist=r["거리 (bp)"], raw=r["원출력"],
            kind="efficiency" if r["점수 기준"] == "절단 효율" else "percentile",
            guide=r["가이드 서열 (RNA)"].replace("U", "T"), pam=r["PAM"],
            strand=r["가닥"],
        )
        for r in captured["detail"]["rows"]
    ]
    expected = (BASELINE / "crispr_prediction_results.csv").read_bytes()
    # #196이 모델·버전 열을 끝에 더했다. 기준이 증거로 남긴 것은 기존 열의 값과
    # 순서이므로 그 부분을 바이트로 비교한다.
    assert csv_bytes(base_columns_only(detail_table(rows))) == expected
    # 기준을 뜰 때의 기본 필터는 한 줄도 떨어뜨리지 않았다.
    assert captured["detail"]["dropped_by_default_filter"] == 0
    assert len(DetailFilter.parse(None).apply(rows)) == len(rows)


def test_csv_rounds_and_writes_exactly_like_the_pandas_download():
    """반올림까지 포함해 `df.to_csv(index=False)`와 맞춘다.

    기준 CSV는 이미 반올림된 값으로 떠 있어서 `round()` 자체를 재지 못한다.
    여기서는 반올림 전 값을 넣고 `app.py:353-389`의 식을 그대로 다시 써서 비교한다.
    """
    import pandas as pd

    rows = [
        row("SpCas9-NG", 51.98895989939809, raw=1.6856821775436401, dist=11),
        row("Sniper", 49.055555555, raw=1.14314159, dist=1, strand="-", pam="TGG"),
        row("Cas12a", 7.004999999, raw=0.070049999, cas="Cas12a", kind="percentile"),
    ]
    expected = pd.DataFrame(
        [
            {
                "순위": i,
                "Cas 변종": r.cas_name,
                "PAM": r.pam_seq,
                "가이드 서열 (RNA)": "".join("U" if b == "T" else b for b in r.guide_seq),
                "가닥": r.strand,
                "거리 (bp)": r.distance,
                "원출력": round(r.raw_score, 4),
                "예측 점수 (%)": round(r.efficiency_pct, 2),
                "점수 기준": (
                    "절단 효율" if r.score_kind == "efficiency" else "학습 분포 백분위"
                ),
            }
            for i, r in enumerate(rows, start=1)
        ]
    ).to_csv(index=False, lineterminator="\r\n")
    assert csv_bytes(base_columns_only(detail_table(rows))) == expected.encode("utf-8-sig")


# ── 필터 ───────────────────────────────────────────


def test_default_filter_passes_everything_streamlit_shows():
    rows = [row("A", 0.0, dist=0), row("B", 100.0, dist=15), row("C", 42.5, dist=7)]
    assert DetailFilter.parse(None).apply(rows) == rows
    assert DetailFilter.parse({}).apply(rows) == rows


@pytest.mark.parametrize("raw", [[], 0, "", "cas_names"])
def test_falsy_junk_is_an_error_not_a_silent_save_everything(raw):
    """빈 목록이 '필터 없음'으로 통하면 거른 줄 알고 전체를 저장하게 된다."""
    with pytest.raises(ValueError):
        DetailFilter.parse(raw)


def test_filter_bounds_are_inclusive_like_streamlit():
    rows = [row("A", 10.0, dist=2), row("B", 10.5, dist=3), row("C", 20.0, dist=5),
            row("D", 20.01, dist=4)]
    chosen = DetailFilter.parse({"eff_range": [10.5, 20.0], "dist_range": [3, 5]})
    assert [r.cas_name for r in chosen.apply(rows)] == ["B", "C"]


def test_filter_by_cas_names_keeps_order_and_renumbers_ranks():
    rows = [row("SpCas9(WT)", 90.0), row("Cas12a", 80.0, cas="Cas12a"),
            row("SpCas9-NG", 70.0)]
    chosen = DetailFilter.parse({"cas_names": ["SpCas9-NG", "SpCas9(WT)"]})
    table = detail_table(chosen.apply(rows))
    assert [cells[:2] for cells in table[1:]] == [[1, "SpCas9(WT)"], [2, "SpCas9-NG"]]


def test_no_cas_selected_filters_out_everything():
    assert DetailFilter.parse({"cas_names": []}).apply([row("A", 50.0)]) == []


def test_filter_uses_the_unrounded_score():
    """Streamlit은 반올림 전 값으로 거른다. 화면에 50.00으로 보여도 49.999는 빠진다."""
    rows = [row("A", 49.999), row("B", 50.0)]
    chosen = DetailFilter.parse({"eff_range": [50.0, 100.0]})
    assert [r.cas_name for r in chosen.apply(rows)] == ["B"]


@pytest.mark.parametrize(
    "raw",
    [
        {"eff_range": [60, 50]},
        {"dist_range": [5, 1]},
        {"eff_range": [-1, 50]},
        {"dist_range": [0, 16]},
        {"eff_range": [0]},
        {"cas_names": "SpCas9(WT)"},
        {"unknown": 1},
    ],
)
def test_filter_rejects_what_the_streamlit_controls_cannot_produce(raw):
    with pytest.raises(ValueError):
        DetailFilter.parse(raw)


# ── 조각 ───────────────────────────────────────────


def test_fragment_lists_every_row_with_the_values_the_screen_filters_on():
    rows = [row("SpCas9(WT)", 51.98895989939809, dist=11, site="SpCas9|+|32",
                identity=MODEL),
            row("Cas12a", 7.25, dist=1, cas="Cas12a", site="Cas12a|-|60", identity=CAS12A)]
    parsed = Rows(detail_fragment(rows))
    assert parsed.head == DETAIL_COLUMNS
    assert [r["cells"] for r in parsed.rows] == [
        ["1", "SpCas9(WT)", "AGG", "AUCGAUCGAUCGAUCGAUCG", "+", "11", "0.0000",
         "51.99%", "절단 효율", "NT SpCas9", "sha256 0123456789ab"],
        ["2", "Cas12a", "AGG", "AUCGAUCGAUCGAUCGAUCG", "+", "1", "0.0000",
         "7.25%", "절단 효율", "NT Cas12a", "sha256 0123456789ab"],
    ]
    first = parsed.rows[0]["attrs"]
    assert first["data-site-key"] == "SpCas9|+|32"
    assert first["data-head"] == "SpCas9(WT)"
    # 필터가 보는 값은 헤드가 아니라 예측 행의 이름이다 (CONTEXT.md의 '헤드').
    assert first["data-cas-name"] == "SpCas9(WT)"
    # 화면의 필터는 반올림 전 값을 그대로 받아 Streamlit과 같은 경계에서 거른다.
    assert float(first["data-eff"]) == 51.98895989939809
    assert first["data-dist"] == "11"


def test_fragment_colours_only_the_score_column_on_the_fixed_scale():
    parsed = Rows(detail_fragment([row("A", 0.0), row("B", 100.0)]))
    score = DETAIL_COLUMNS.index("예측 점수 (%)")
    for item in parsed.rows:
        styles = item["styles"]
        assert "background-color:#" in styles[score]
        assert all(not s for i, s in enumerate(styles) if i != score)
    low, high = (r["styles"][score] for r in parsed.rows)
    assert low != high  # vmin 0 · vmax 100의 RdYlGn이라 양끝 색이 다르다


def test_fragment_offers_a_checkbox_per_variant_in_sorted_order():
    rows = [row("SpCas9-NG", 9.0), row("Cas12a", 8.0, cas="Cas12a"),
            row("SpCas9(WT)", 7.0), row("SpCas9-NG", 6.0)]
    fragment = detail_fragment(rows)
    for text in ("필터 설정", "Cas 변종 선택", "전체 선택", "전체 해제",
                 "효율 범위 및 거리 설정", "최소 점수 (%)", "최대 거리 (bp)",
                 "검색", "CSV 다운로드", "0% 이상", "15 bp 이하"):
        assert text in fragment
    assert "원하는 조건을 설정한 뒤" not in fragment and "<h2>전체 상세 보기</h2>" not in fragment
    assert fragment.count('type="range"') == 2
    boxes = [part.split('"', 1)[0] for part in fragment.split('name="detail-cas" value="')[1:]]
    assert boxes == ["Cas12a", "SpCas9(WT)", "SpCas9-NG"]


def test_fragment_hides_results_until_the_search_button_like_streamlit():
    fragment = detail_fragment([row("A", 50.0)])
    results = fragment.split('<div id="detail-results"', 1)[1].split(">", 1)[0]
    assert "hidden" in results


def test_fragment_without_rows_says_so():
    fragment = detail_fragment([])
    assert "예측 결과가 없습니다." in fragment
    assert "<table" not in fragment
    assert "detail-filters" not in fragment


def test_fragment_is_self_contained():
    fragment = detail_fragment([row("<b>", 50.0)])
    assert "<script" not in fragment
    assert "<b>" not in fragment


# ── 보존된 기준과의 대조 (#196) ────────────────────────────────


def test_baseline_comparison_keeps_the_preserved_bytes_untouched():
    """더해진 열을 뗀 저장 파일이 보존된 기준의 **원래 바이트**와 같아야 한다.

    양쪽을 다시 직렬화하면 인용 규칙이나 줄 끝이 달라도 통과해, 바이트 비교가
    아니라 왕복 비교가 된다. 기준 쪽은 건드리지 않는다.
    """
    expected = (BASELINE / "crispr_prediction_results.csv").read_bytes()
    rows = csv_rows(expected)
    saved = csv_bytes([[*rows[0], *MODEL_COLUMNS]]
                      + [[*row, "NT SpCas9", "sha256 0123456789ab"] for row in rows[1:]])
    report = csv_against_baseline(saved, expected)
    assert report["csv_base_columns_byte_equal"]
    assert not report["csv_whole_file_byte_equal"]
    assert report["csv_model_columns"] == [
        ["모델", "모델 버전"], ["NT SpCas9", "sha256 0123456789ab"]]


def test_baseline_comparison_rejects_a_changed_existing_column():
    expected = (BASELINE / "crispr_prediction_results.csv").read_bytes()
    rows = csv_rows(expected)
    rows[1][1] = "다른 변종"
    saved = csv_bytes([[*row, "m", "v"] for row in rows])
    assert not csv_against_baseline(saved, expected)["csv_base_columns_byte_equal"]
