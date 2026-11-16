"""Real core predictions for the desktop's first two views (#184).

The weight-free renderer inherits the same tables with a fake row source.
Detail filtering and XAI controls belong to #185/#187; real mode never presents
fake explanations as real analysis.
"""

from __future__ import annotations

import html
from typing import Any

import core

from desktop.api import OnBatch
from desktop.prediction_views import (
    best_candidates_fragment,
    variant_comparison_fragment,
)

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


def _table(headers: list[str], rows: list[list[Any]], attrs: list[dict]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for cells, attr in zip(rows, attrs):
        marks = "".join(f' {k}="{esc(v)}"' for k, v in attr.items())
        body.append(
            f"<tr{marks}>" + "".join(f"<td>{esc(c)}</td>" for c in cells) + "</tr>"
        )
    return (
        f'<table class="grid"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _detail_cells(row: core.VariantResult, rank: int) -> list[Any]:
    return [
        rank,
        row.cas_name,
        row.pam_seq,
        core.dna_to_rna(row.guide_seq),
        row.strand,
        row.distance,
        round(row.raw_score, 4),
        round(row.efficiency_pct, 2),
        _SCORE_KIND_LABEL[row.score_kind],
    ]


class PredictionRenderer:
    def __init__(self, predictors: dict | None = None):
        self._predictors = predictors

    def warm_up(self, on_batch: OnBatch) -> str:
        if self._predictors is None:
            self._predictors = core.load_predictors()
        on_batch(1, 1)
        return '<p class="status">모델 준비 완료</p>'

    def _rows_with_progress(
        self, dna: str, on_batch: OnBatch
    ) -> list[core.VariantResult]:
        if self._predictors is None:
            self._predictors = core.load_predictors()
        scan = core.scan_all(dna)
        total_sites = sum(
            sum(s.cas_type == cas and bool(s.model_input_seq) for s in scan.sites)
            for (_, cas), predictor in self._predictors.items()
            if predictor.connected
        )
        completed = 0

        def report(done: int, total: int) -> None:
            nonlocal completed
            # The core restarts counters per predictor; the desktop exposes one
            # operation. Keep cancellation exceptions flowing to the core.
            on_batch(completed + done, total_sites)
            if done == total:
                completed += total

        result = core.run_prediction(scan, self._predictors, on_batch=report)
        return result.sorted_variant_results()

    def sites_fragment(self, dna: str, on_batch: OnBatch) -> str:
        sites = core.scan_all(dna).sites
        table = _table(
            ["Cas", "가닥", "PAM 위치", "PAM", "가이드 서열 (RNA)", "거리 (bp)"],
            [
                [
                    s.cas_type,
                    s.strand,
                    s.pam_start,
                    s.pam_seq,
                    core.dna_to_rna(s.guide_seq),
                    s.distance,
                ]
                for s in sites
            ],
            [{"data-site-key": s.site_key} for s in sites],
        )
        return (
            f'<section id="sites"><h2>후보 사이트 {len(sites)}개</h2>{table}</section>'
        )

    def prediction_fragments(self, dna: str, on_batch: OnBatch) -> dict[str, str]:
        rows = self._rows_with_progress(dna, on_batch)
        detail = _table(
            DETAIL_COLUMNS,
            [_detail_cells(r, i) for i, r in enumerate(rows, start=1)],
            [{"data-site-key": r.site_key, "data-head": r.cas_name} for r in rows],
        )
        return {
            "best": best_candidates_fragment(rows),
            "variants": variant_comparison_fragment(rows),
            "detail": f'<section id="detail"><h2>전체 상세 보기</h2>{detail}</section>',
        }

    def explanation_fragment(
        self,
        dna: str,
        site_key: str,
        head: str,
        method: str,
        params: dict[str, Any],
        size: int,
        on_batch: OnBatch,
    ) -> str:
        return '<p class="empty">XAI 설명 보기는 준비 중입니다.</p>'

    def csv_table(self, dna: str, on_batch: OnBatch) -> list[list[Any]]:
        rows = self._rows_with_progress(dna, on_batch)
        return [DETAIL_COLUMNS] + [
            _detail_cells(r, i) for i, r in enumerate(rows, start=1)
        ]
