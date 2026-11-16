"""Desktop XAI fragments using the same window coordinates and charts as Streamlit."""

from __future__ import annotations

import base64
import html
import io
import json
from collections.abc import Callable, Sequence
from typing import Any

import core

from desktop.api import OnBatch


def controls_template(
    sites: Sequence[core.CandidateSite],
    rows: Sequence[core.VariantResult],
    heads_of: Callable[[core.CandidateSite], list[str]],
) -> str:
    """Travel with the detail fragment, without adding a slot or bridge call."""
    best = {}
    for row in rows:
        if row.model == "A" and row.site_key not in best:
            best[row.site_key] = row
    ranked = sorted(
        sites,
        key=lambda s: (best[s.site_key].efficiency_pct, best[s.site_key].raw_score)
        if s.site_key in best else (0.0, 0.0),
        reverse=True,
    )
    options = []
    for site in ranked:
        row = best.get(site.site_key)
        score = row.efficiency_pct if row else 0.0
        kind = row.score_kind if row else "efficiency"
        what = "효율" if kind == "efficiency" else "백분위"
        label = (f"{site.cas_type} · {site.strand}가닥 · PAM {site.pam_seq} · "
                 f"거리 {site.distance}bp · {what} {score:.1f}%")
        heads = html.escape(json.dumps(heads_of(site), ensure_ascii=False), quote=True)
        options.append(
            f'<option value="{html.escape(site.site_key, quote=True)}" '
            f'data-heads="{heads}">{html.escape(label)}</option>'
        )
    return (
        '<template id="xai-sites-template">' + "".join(options) + '</template>'
    )


def _figure(fig, caption: str, size: int) -> str:
    import matplotlib.pyplot as plt

    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        return (
            f'<figure class="xai" style="width:{size}%">'
            f'<img src="{uri}" alt="{html.escape(caption)}">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>'
        )
    finally:
        plt.close(fig)


def explanation_fragment(
    predictor,
    site: core.CandidateSite,
    head: str,
    method: str,
    params: dict[str, Any],
    size: int,
    on_batch: OnBatch,
) -> str:
    """Only computation parameters reach the shared XAI memory cache."""
    window = site.model_input_seq
    layout = core.WINDOW_LAYOUT[site.cas_type]
    model_key = f"A/{site.cas_type}"
    cache = core.EXPLAIN_CACHE
    # Compute before making any figures so cancellation leaves no open figures.
    if method == "ig":
        steps = _setting(params, "steps", 8, 256)
        result = cache.integrated_gradients(
            predictor, model_key, head, window, steps, on_batch=on_batch
        )
        body = '<h3>글자별 기여도 (Attribution Logo, IG)</h3>'
        body += '<p>글자가 위로 클수록 효율을 많이 올린 자리, 아래로 클수록 깎아먹은 자리입니다.</p>'
        body += _figure(core.attribution_logo(result.attribution * 100, window, layout),
                        "글자별 기여도 (Attribution Logo, IG)", size)
        body += (f'<p class="caption">예측 {result.pred * 100:.2f}% − '
                 f'기준 {result.baseline_pred * 100:.2f}% · '
                 f'기여도 합 {result.attribution.sum() * 100:+.3f}%p · '
                 f'완결성 오차 {result.completeness_error * 100:.4f}%p</p>')
    elif method == "shap":
        nsamples = _setting(params, "nsamples", 16, 1024)
        background = _setting(params, "n_background", 2, 32)
        result = cache.kernel_shap(
            predictor, model_key, head, window, nsamples, background, on_batch=on_batch
        )
        values = result.values * 100
        body = '<h3>위치별 기여분 (SHAP)</h3>'
        body += '<p>빨강은 효율을 올린 자리, 파랑은 내린 자리입니다. 워터폴은 기준점에서 각 기여를 더해 최종 예측값에 도달하는 과정입니다.</p>'
        body += _figure(core.shap_bars(values, window, layout), "위치별 기여분 (SHAP)", size)
        body += _figure(core.shap_waterfall(values, result.base_value * 100, window, layout),
                        "SHAP 워터폴", size)
        body += (f'<p class="caption">기준값 {result.base_value * 100:.2f}% + '
                 f'SHAP 합 {values.sum():+.2f}%p = '
                 f'{(result.base_value + result.values.sum()) * 100:.2f}% · '
                 f'실제 예측 {result.pred * 100:.2f}% · '
                 f'가법성 오차 {result.additivity_error * 100:.4f}%p</p>')
    elif method == "rollout":
        result = cache.attention_rollout(predictor, model_key, head, window)
        body = '<h3>위치별 정보 의존도 (Attention Rollout)</h3>'
        body += '<p>막대가 높을수록 모델이 그 자리를 많이 참고했습니다. 좋다·나쁘다의 방향이 없는 주목도입니다.</p>'
        body += _figure(core.rollout_profile(result.position_profile, window, layout),
                        "위치별 정보 의존도 (Attention Rollout)", size)
        body += (f'<details><summary>토큰 × 토큰 의존도 행렬 보기 '
                 f'(토큰 {len(result.token_labels)}개)</summary>')
        body += _figure(core.rollout_token_heatmap(result.token_matrix, result.token_labels),
                        "토큰 × 토큰 의존도 행렬", size) + '</details>'
        body += f'<p class="caption">{result.n_layers}개 층의 attention 누적 · 모델 내부의 정보 의존성이지 물리적 결합이 아닙니다.</p>'
    else:
        raise ValueError(f"모르는 XAI 기법: {method}")

    sequence = "".join(
        f'<span style="background-color:{core.SEGMENT_COLORS[core.segment_of(i, layout)]}55">'
        f'{html.escape(base)}</span>' for i, base in enumerate(window)
    )
    legend = " ".join(
        f'<span style="background-color:{core.SEGMENT_COLORS[name]}55">{name} {a}–{b}</span>'
        for name, a, b in layout
    )
    return (
        '<section><h3>모델이 실제로 읽은 구간</h3>'
        f'<div class="window-sequence">{sequence}</div><p>{legend}</p>'
        f'<p>설명 대상: {html.escape(head)} · PAM {html.escape(site.pam_seq)} · '
        f'변이와의 거리 {site.distance} bp · 윈도우 {len(window)}bp (0부터 시작)</p>'
        + body + '</section>'
    )


def _setting(params: dict[str, Any], name: str, low: int, high: int) -> int:
    value = int(params[name])
    if not low <= value <= high:
        raise ValueError(f"{name} 값은 {low}~{high} 사이여야 합니다.")
    return value
