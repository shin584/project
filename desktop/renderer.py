"""Core predictions and cached explanations for the desktop views."""

from __future__ import annotations

import html
from typing import Any

import core

from desktop.api import OnBatch
from desktop.detail_view import DetailFilter, detail_fragment, detail_table
from desktop.prediction_views import (
    best_candidates_fragment,
    variant_comparison_fragment,
)
from desktop.xai_view import controls_template, explanation_fragment


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
        metrics = '<div class="metrics">' + "".join(
            f'<div><span>{html.escape(config.name)}</span><strong>'
            f'{sum(s.cas_type == config.name for s in sites)}개 발견</strong></div>'
            for config in core.CAS_CONFIGS
        ) + '</div>'
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
            f'<section id="sites">{metrics}<details><summary>후보 사이트 {len(sites)}개</summary>'
            f'{table}</details></section>'
        )

    def prediction_fragments(self, dna: str, on_batch: OnBatch) -> dict[str, str]:
        rows = self._rows_with_progress(dna, on_batch)
        diagnostics = []
        for cas in sorted({row.cas_type for row in rows}):
            items = [row for row in rows if row.cas_type == cas]
            # Preview subclasses deliberately have no predictors and must not
            # import predict/torch just to describe their synthetic scores.
            spec = core.MODEL_REGISTRY.get(("A", cas)) if self._predictors is not None else None
            mode = spec.output_mode if spec else "?"
            what = "효율(%)" if items[0].score_kind == "efficiency" else "백분위(%)"
            diagnostics.append(
                f'<li><strong>{html.escape(cas)}</strong> · 원출력 '
                f'{min(r.raw_score for r in items):.3f} ~ {max(r.raw_score for r in items):.3f}'
                f' · 해석 <code>{html.escape(mode)}</code> → 화면 표시는 {what}</li>'
            )
        diagnostic_html = ('<details id="model-diagnostics"><summary>모델 출력 해석 상태</summary><ul>'
                           + ''.join(diagnostics) + '</ul></details>')
        return {
            "best": diagnostic_html + best_candidates_fragment(rows),
            "variants": variant_comparison_fragment(rows),
            "detail": (
                detail_fragment(rows)
                + controls_template(core.scan_all(dna).sites, rows, self._explain_heads)
            ),
        }

    def _explain_heads(self, site: core.CandidateSite) -> list[str]:
        predictor = (self._predictors or {}).get(("A", site.cas_type))
        if predictor is None or not predictor.connected:
            return []
        # Small prediction-only adapters may not expose explanation support.
        heads = getattr(predictor, "explain_heads", None)
        return heads(site) if heads else []

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
        site = next((s for s in core.scan_all(dna).sites if s.site_key == site_key), None)
        if site is None:
            raise LookupError(f"그런 사이트가 없습니다: {site_key}")
        if self._predictors is None:
            self._predictors = core.load_predictors()
        if head not in self._explain_heads(site):
            raise LookupError(f"이 사이트에 없는 헤드입니다: {head}")
        return explanation_fragment(
            self._predictors[("A", site.cas_type)], site, head, method, params, size, on_batch
        )

    def csv_table(
        self, dna: str, on_batch: OnBatch, filters: dict[str, Any] | None = None
    ) -> list[list[Any]]:
        # 거르고 나서 번호를 다시 매긴다 — Streamlit이 내려주는 표와 같다.
        chosen = DetailFilter.parse(filters)
        return detail_table(chosen.apply(self._rows_with_progress(dna, on_batch)))
