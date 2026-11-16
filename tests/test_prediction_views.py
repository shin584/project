"""#184: observable table contents without models or a WebView."""

from html.parser import HTMLParser

import pytest
from core import VariantResult
from desktop.api import Cancelled
from desktop.prediction_views import (
    best_candidates_fragment,
    variant_comparison_fragment,
)
from desktop.renderer import PredictionRenderer


class Tables(HTMLParser):
    def __init__(self, fragment):
        super().__init__()
        self.tables = []
        self.cell: list[str] | None = None
        self.feed(fragment)

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self.tables[-1].append([])
        elif tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            assert self.cell is not None
            self.tables[-1][-1].append("".join(self.cell))
            self.cell = None


def row(
    site,
    name,
    score,
    raw=0.0,
    *,
    kind="efficiency",
    cas="SpCas9",
    guide="ATCG",
    model="A",
):
    return VariantResult(
        cas_name=name,
        pam_seq="AGG",
        guide_seq=guide,
        distance=3,
        strand="+",
        raw_score=raw,
        efficiency_pct=score,
        score_kind=kind,
        site_key=site,
        cas_type=cas,
        model=model,
    )


def test_best_selects_one_per_site_by_display_score_then_raw_and_ranks_sites():
    fragment = best_candidates_fragment(
        [
            row("site1", "SpCas9(WT)", 40, raw=9),
            row("site1", "SpCas9-NG", 80, raw=-1),
            row("site2", "SpCas9(WT)", 80, raw=0),
            row("site2", "xCas9", 80, raw=1),
            row("ignored", "Model B", 100, model="B"),
        ]
    )
    (table,) = Tables(fragment).tables
    assert table[0] == [
        "순위",
        "최적 Cas 변종",
        "PAM",
        "가이드 서열 (RNA)",
        "가닥",
        "거리 (bp)",
        "예측 절단 효율 (%)",
        "원출력",
    ]
    assert table[1:] == [
        ["1", "xCas9", "AGG", "AUCG", "+", "3", "80.00%", "1.0000"],
        ["2", "SpCas9-NG", "AGG", "AUCG", "+", "3", "80.00%", "-1.0000"],
    ]


def test_best_separates_score_kinds_and_keeps_cas12a_prediction():
    fragment = best_candidates_fragment(
        [
            row("cas12a", "Cas12a", 90, cas="Cas12a"),
            row("sp", "SpCas9(WT)", 95, kind="percentile"),
        ]
    )
    sp, cas12a = Tables(fragment).tables
    assert sp[0][-2] == "학습 분포 백분위 (%)"
    assert sp[1][1] == "SpCas9(WT)"
    assert cas12a[0][-2] == "예측 절단 효율 (%)"
    assert cas12a[1][1] == "Cas12a"
    assert "서로 다른 표의 숫자는 직접 비교할 수 없습니다." in fragment


def test_variants_pivot_present_columns_aggregate_max_sort_rows_and_mark_missing():
    fragment = variant_comparison_fragment(
        [
            row("low", "xCas9", 0, guide="AAAA"),
            row("high", "SpCas9-NG", 90, guide="TTTT"),
            row("high", "SpCas9-NG", 100, guide="TTTT"),
            row("cas12a", "Cas12a", 99, cas="Cas12a"),
        ]
    )
    (table,) = Tables(fragment).tables
    assert table == [
        ["guide_label", "SpCas9-NG", "xCas9"],
        ["UUUU  dist:3  PAM:AGG  strand:+", "100.0%", "None"],
        ["AAAA  dist:3  PAM:AGG  strand:+", "None", "0.0%"],
    ]
    assert "background-color:#006837" in fragment
    assert "background-color:#a50026" in fragment
    assert 'background-color:transparent;color:inherit">None' in fragment
    assert "<script" not in fragment and "https://" not in fragment


def test_variants_separate_score_kinds_even_within_spcas9():
    tables = Tables(
        variant_comparison_fragment(
            [
                row("eff", "SpCas9-NG", 60),
                row("pct", "xCas9", 80, kind="percentile"),
            ]
        )
    ).tables
    assert len(tables) == 2
    assert tables[0][0] == ["guide_label", "SpCas9-NG"]
    assert tables[1][0] == ["guide_label", "xCas9"]


def test_each_view_limits_to_twenty_after_sorting():
    rows = [row(str(i), "SpCas9-NG", i, guide="A" * (i + 1)) for i in range(21)]
    for render in (best_candidates_fragment, variant_comparison_fragment):
        fragment = render(rows)
        (table,) = Tables(fragment).tables
        assert len(table) == 21  # header plus 20 displayed sites
        assert "상위 20개만 표시합니다. (전체 21개)" in fragment
        assert "20.0" in table[1][-2 if render is best_candidates_fragment else -1]


def test_empty_views_and_cas12a_only_comparison_have_reference_messages():
    assert "예측 결과가 없습니다." in best_candidates_fragment([])
    fragment = variant_comparison_fragment([row("cas", "Cas12a", 50, cas="Cas12a")])
    assert "SpCas9 PAM 사이트가 발견되지 않았습니다." in fragment
    assert Tables(fragment).tables == []


DNA = (
    "GATCCAGTCAGCATGC"
    + "TTTA"
    + "CGATCGTACGATCGAATCGT"
    + "AGG"
    + "CTAGCATCGATCGATGCATCGTAGCTAGCTAGCATCGA"
)


class FixedPredictor:
    """Stand in for model inference while using the real scan/prediction core."""

    connected = True

    def predict_batch(self, sites, on_batch):
        results = []
        for done, site in enumerate(sites, start=1):
            results.append(
                VariantResult(
                    cas_name="Cas12a" if site.cas_type == "Cas12a" else "SpCas9-NG",
                    pam_seq=site.pam_seq,
                    guide_seq=site.guide_seq,
                    distance=site.distance,
                    strand=site.strand,
                    raw_score=0.5,
                    efficiency_pct=50,
                    cas_type=site.cas_type,
                    site_key=site.site_key,
                )
            )
            on_batch(done, len(sites))
        return results


def test_real_renderer_uses_core_with_monotonic_progress_across_both_models():
    renderer = PredictionRenderer(
        {("A", "SpCas9"): FixedPredictor(), ("A", "Cas12a"): FixedPredictor()}
    )
    progress = []
    fragments = renderer.prediction_fragments(DNA, lambda d, t: progress.append((d, t)))
    done = [d for d, _ in progress]
    assert done == sorted(done)
    assert len({t for _, t in progress}) == 1
    assert progress[-1][0] == progress[-1][1]
    (best,) = Tables(fragments["best"]).tables
    assert {cells[1] for cells in best[1:]} == {"SpCas9-NG", "Cas12a"}


def test_real_renderer_propagates_progress_cancellation():
    renderer = PredictionRenderer({("A", "SpCas9"): FixedPredictor()})

    def cancel(done, total):
        raise Cancelled()

    with pytest.raises(Cancelled):
        renderer.prediction_fragments(DNA, cancel)
