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
        scores = html.escape(json.dumps({
            r.cas_name: [r.efficiency_pct, r.score_kind] for r in rows
            if r.model == "A" and r.site_key == site.site_key
        }, ensure_ascii=False), quote=True)
        options.append(
            f'<option value="{html.escape(site.site_key, quote=True)}" '
            f'data-heads="{heads}" data-scores="{scores}" '
            f'data-pam="{html.escape(site.pam_seq, quote=True)}" '
            f'data-distance="{site.distance}">{html.escape(label)}</option>'
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
    import numpy as np

    def position(index: int) -> str:
        return (f'{index}번째 글자 <code>{html.escape(window[index])}</code> '
                f'({core.segment_of(index, layout)} 구간)')

    def top_positions(title, values, signed=False):
        indices = np.argsort(-np.abs(values) if signed else -values)[:3]
        lines = []
        for i in indices:
            suffix = (f" → 효율을 {'올림' if values[i] > 0 else '깎음'} ({values[i]:+.2f}%p)"
                      if signed else "")
            lines.append(f'<li>{position(int(i))}{suffix}</li>')
        return f'<div class="success"><strong>{title}</strong><ul>' + ''.join(lines) + '</ul></div>'
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
        pct = result.attribution * 100
        up, down = int(np.argmax(pct)), int(np.argmin(pct))
        body += ('<div class="success"><strong>이 그래프가 말하는 것</strong>'
                 f'<p>효율을 <strong>가장 많이 올린 자리</strong>: {position(up)} → {pct[up]:+.2f}%p</p>'
                 f'<p>효율을 <strong>가장 많이 깎은 자리</strong>: {position(down)} → {pct[down]:+.2f}%p</p></div>')
        body += ('<details><summary>기술적 참고</summary>'
                 '<p>기준 입력은 빈 서열(모든 염기 값 0)이고, 기준 대비 변화를 위치별로 나눈 값입니다. '
                 '기여도 합이 (이 서열의 예측 − 기준의 예측)과 같아야 계산이 맞습니다 (완결성 검사).</p>'
                 f'<p class="caption">예측 {result.pred * 100:.2f}% − '
                 f'기준 {result.baseline_pred * 100:.2f}% · '
                 f'기여도 합 {result.attribution.sum() * 100:+.3f}%p · '
                 f'완결성 오차 {result.completeness_error * 100:.4f}%p</p></details>')
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
        body += top_positions("영향이 가장 컸던 자리 3곳", values, signed=True)
        body += ('<details><summary>기술적 참고 (기준값 · 합산값)</summary>'
                 '<p>위치별 KernelSHAP입니다. 어떤 자리를 끈다는 것은 무작위 배경 서열의 염기로 바꾼다는 뜻입니다. '
                 '배경에는 PAM 자리도 포함됩니다. 샘플 수를 바꾸면 추정값이 달라질 수 있습니다.</p>'
                 f'<p class="caption">기준값 {result.base_value * 100:.2f}% + '
                 f'SHAP 합 {values.sum():+.2f}%p = '
                 f'{(result.base_value + result.values.sum()) * 100:.2f}% · '
                 f'실제 예측 {result.pred * 100:.2f}% · '
                 f'가법성 오차 {result.additivity_error * 100:.4f}%p</p></details>')
    elif method == "rollout":
        result = cache.attention_rollout(predictor, model_key, head, window)
        body = '<h3>위치별 정보 의존도 (Attention Rollout)</h3>'
        body += '<p>막대가 높을수록 모델이 그 자리를 많이 참고했습니다. 좋다·나쁘다의 방향이 없는 주목도입니다.</p>'
        body += _figure(core.rollout_profile(result.position_profile, window, layout),
                        "위치별 정보 의존도 (Attention Rollout)", size)
        body += top_positions("모델이 가장 자주 쳐다본 자리 3곳", result.position_profile)
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
        '<section><div class="xai-window" '
        f'data-site-key="{html.escape(site.site_key, quote=True)}">'
        '<h3>모델이 실제로 읽은 구간</h3>'
        f'<div class="window-sequence">{sequence}</div><p>{legend}</p>'
        '<p class="caption">초록 PAM은 가위가 찾는 표지판, 파랑 guide는 자를 위치를 지정하는 주소, '
        '회색 flank는 주변 서열입니다. 아래 그래프의 가로축이 이 구간과 같습니다.</p></div>'
        f'<p>설명 대상: {html.escape(head)} · PAM {html.escape(site.pam_seq)} · '
        f'변이와의 거리 {site.distance} bp · 윈도우 {len(window)}bp (0부터 시작)</p>'
        + body + '</section>'
    )


def _setting(params: dict[str, Any], name: str, low: int, high: int) -> int:
    value = int(params[name])
    if not low <= value <= high:
        raise ValueError(f"{name} 값은 {low}~{high} 사이여야 합니다.")
    return value
