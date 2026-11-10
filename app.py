"""CRISPR PAM Scanner + XAI (1학기 + 2학기 통합)

탭 구성 — 두 학기 화면에서 겹치던 탭을 합쳤다.
  1. Best Candidates : 사이트별 최고 효율 후보       (1학기 '최고 후보군')
  2. 변종 비교        : SpCas9 변종 9종 히트맵             (Cas12a는 변종이 없어 제외)
  3. 전체 상세 보기   : 필터 · 검색 · CSV             (1학기 탭4 + 2학기 '예측 상세')
  4. XAI 설명         : 연동 모델에서 Rollout·IG·SHAP 직접 계산 (2학기 'XAI 분석')

연동 모델은 SpCas9 · Cas12a 2종이다 (predict.MODEL_REGISTRY 참고).
"""

import pandas as pd
import streamlit as st

from pam_scanner import (
    CAS_CONFIGS, SPCAS9_VARIANTS, WINDOW_LAYOUT, dna_to_rna, print_scan_report, scan_all,
)
from predict import load_predictors, run_prediction
from xai import view as xai_view

st.set_page_config(page_title="CRISPR PAM Scanner + XAI", layout="wide")

EFF       = "예측 점수 (%)"           # 의미는 결과의 score_kind가 정한다
SCORE_LABELS = {
    "efficiency": "예측 절단 효율 (%)",   # 모델이 0~1 비율로 학습된 경우
    "percentile": "학습 분포 백분위 (%)",  # 모델이 z-score로 학습된 경우
}
GRADIENT  = {"cmap": "RdYlGn", "vmin": 0, "vmax": 100}

# 변종 히트맵에는 '그 변종이 인식하지 않는 PAM' 칸이 결측으로 남는다.
# pandas의 background_gradient는 결측에 검정(#000000)을 직접 칠하고 밝은 글자를 얹는다.
# 그러면 라이트 모드에서 검은 배경 위 어두운 글자가 되어 아무것도 안 보인다.
# 배경을 투명으로, 글자색을 상속으로 덮으면 테마 배경과 글자색을 그대로 따라간다.
#   다크  모드 → 어두운 배경 + 밝은 글자
#   라이트 모드 → 밝은 배경 + 어두운 글자
MISSING_TEXT = "None"      # 결측 칸에 표시할 글자 ("–" 등으로 바꿔도 된다)

HIGHLIGHT = "background-color: #ffe8a3; color: #000"
MAX_ROWS  = 20


# ── 공용 헬퍼 ────────────────────────────────────────────────────────
def style_missing(value):
    """결측 칸만 테마 기본 배경·글자색으로 되돌린다 (히트맵 전용)."""
    if pd.isna(value):
        return "background-color: transparent; color: inherit"
    return ""

def show_df(data, **kwargs):
    """Streamlit 버전 차이(width="stretch" vs use_container_width) 흡수"""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:
        st.dataframe(data, use_container_width=True, **kwargs)


def _score_key(vr):
    """정렬·선택 기준. 화면에 보이는 효율이 1순위, 동점이면 원출력으로 가른다.

    0%로 잘린 값끼리는 효율이 모두 같아지므로 원출력이 동점 해소에 필요하다.
    """
    return (vr.efficiency_pct, vr.raw_score)


def best_per_site(results, model: str = "A") -> dict:
    """사이트별 효율이 가장 높은 결과 1개 (기본 모델 A)"""
    best = {}
    for vr in results:
        if vr.model != model:
            continue
        cur = best.get(vr.site_key)
        # 변종마다 표준화 통계(mu·sd)가 다르므로 원출력(z)이 커도 효율이 낮을 수 있다.
        # 대표 변종은 반드시 **효율** 기준으로 고른다. 효율이 같을 때만 원출력으로 가른다.
        if cur is None or _score_key(vr) > _score_key(cur):
            best[vr.site_key] = vr
    return best


def site_label(site) -> str:
    what = "효율" if getattr(site, "score_kind", "efficiency") == "efficiency" else "백분위"
    return (f"{site.cas_type} · {site.strand}가닥 · PAM {site.pam_seq} · "
            f"거리 {site.distance}bp · {what} {site.efficiency_pct:.1f}%")


# ── 모델 준비 ────────────────────────────────────────────────────────
predictors = load_predictors()


# ── 헤더 + 사이드바(입력) ────────────────────────────────────────────
st.title("CRISPR PAM Scanner + XAI")
st.markdown("81bp DNA 서열을 입력하면 PAM 탐색, 절단 효율 예측, 예측 근거(XAI)를 보여줍니다.")

if 'all_results' not in st.session_state:
    st.session_state.all_results = None
if 'detail_filtered' not in st.session_state:
    st.session_state.detail_filtered = None

with st.sidebar:
    st.header("서열 입력")
    dna_input = st.text_input("81bp DNA 서열", placeholder="ATCG...")
    run_btn   = st.button("분석 시작", type="primary")

if run_btn and dna_input:
    try:
        with st.spinner("PAM 탐색 중..."):
            res = scan_all(dna_input)
            print_scan_report(res)
        with st.spinner("모델 예측 중..."):
            res = run_prediction(res, predictors)
        st.session_state.all_results     = res
        st.session_state.detail_filtered = None
        st.session_state.pop("sel_site", None)      # 이전 분석의 사이트 선택 초기화
    except ValueError as e:
        st.error(f"오류: {e}")
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {type(e).__name__}: {e}")

all_results = st.session_state.all_results
if all_results is None:
    st.info("왼쪽 사이드바에 81bp DNA 서열을 입력하고 **분석 시작**을 눌러 주세요.")
    st.stop()

vrs = all_results.variant_results

# ── XAI 대상 사이트 (선택창은 XAI 탭 안에 있음) ──────────────────────
site_map = {s.site_key: s for s in all_results.sorted_by_score()}
sel_key  = st.session_state.get("sel_site")
if sel_key not in site_map:
    sel_key = next(iter(site_map), None)        # 기본값: 효율이 가장 높은 사이트
sel_site = site_map.get(sel_key)

with st.sidebar:
    st.divider()
    st.slider("XAI 그래프 크기", min_value=40, max_value=100, value=80, step=5,
              format="%d%%", key="fig_pct",
              help="XAI 탭의 그래프 가로 폭입니다. 줄이면 그래프 높이도 함께 줄어듭니다.")

# ── PAM 탐색 요약 ────────────────────────────────────────────────────
st.subheader("PAM 탐색 결과")
for col, cfg in zip(st.columns(len(CAS_CONFIGS)), CAS_CONFIGS):
    col.metric(label=cfg.name, value=f"{len(all_results.by_cas(cfg.name))}개 발견")
st.caption("모델 입력 윈도우가 서열 밖으로 나가는 사이트는 예측할 수 없어 제외됩니다.")

st.subheader("절단 효율 예측 결과")
st.caption("점수의 의미는 Cas 종류마다 다릅니다 — Cas12a는 **예측 절단 효율(%)**, "
           "SpCas9은 **학습 분포 백분위(%)** 입니다. 자세한 내용은 아래 "
           "'모델 출력 해석 상태'를 펼쳐 보세요. 거리 보정은 적용하지 않았고, "
           "변이와의 거리는 '거리 (bp)' 열에서 확인할 수 있습니다.")

# ── 진단: 모델별 원출력 범위와 해석 방식 ─────────────────────────────
# 모델마다 학습 방식이 달라 출력의 의미가 다르다. predict.MODEL_REGISTRY에서 지정한다.
if vrs:
    from predict import MODEL_REGISTRY

    _lines = []
    for _cas in sorted({v.cas_type for v in vrs}):
        _r = [v.raw_score for v in vrs if v.cas_type == _cas]
        _spec = MODEL_REGISTRY.get(("A", _cas))
        _mode = _spec.output_mode if _spec else "?"
        _kind = next((v.score_kind for v in vrs if v.cas_type == _cas), "efficiency")
        _what = "효율(%)" if _kind == "efficiency" else "백분위(%)"
        _lines.append(f"- **{_cas}** · 원출력 {min(_r):.3f} ~ {max(_r):.3f} · "
                      f"해석 `{_mode}` → 화면 표시는 {_what}")

    with st.expander("모델 출력 해석 상태", expanded=False):
        st.markdown("\n".join(_lines))
        if any(v.score_kind == "percentile" for v in vrs):
            st.info(
                "**SpCas9은 z-score로 학습된 모델입니다.** 0이 학습 데이터의 평균 효율이라, "
                "평균·표준편차 없이는 효율의 절댓값으로 되돌릴 수 없습니다. "
                "그래서 **학습 분포 안에서의 백분위**로 표시합니다 — "
                "'백분위 95%'는 '학습에 쓰인 가이드들 중 상위 5% 수준'이라는 뜻입니다. "
                "순위는 그대로 유효합니다.\n\n"
                "학습 목표값의 평균·표준편차를 구하시면 `predict.py`의 "
                "`MODEL_REGISTRY`에서 `target_mean`·`target_std`를 채워 주세요. "
                "그러면 자동으로 실제 효율(%) 표시로 바뀝니다.",
            )

tab_best, tab_cmp, tab_detail, tab_xai = st.tabs([
    "Best Candidates", "변종 비교", "전체 상세 보기", "XAI 설명",
])

# ════════════════════════════════════════════════════════
# 탭1: Best Candidates
# ════════════════════════════════════════════════════════
with tab_best:
    st.markdown("각 사이트에서 **점수가 가장 높은 변종 1개**만 대표로 표시합니다. "
                "변종별 비교는 '변종 비교' 탭을 확인하세요.")
    a_best = best_per_site(vrs, "A")

    if not a_best:
        st.info("예측 결과가 없습니다.")
    else:
        # 점수의 단위가 같으면(모두 효율 %) 한 표로, 섞여 있으면 Cas별로 나눈다.
        _kinds = {vr.score_kind for vr in a_best.values()}
        if len(_kinds) == 1:
            groups = {"": list(a_best.values())}
        else:
            groups = {}
            for vr in a_best.values():
                groups.setdefault(vr.cas_type, []).append(vr)
            st.caption("Cas 종류마다 점수의 단위가 달라 표를 나눴습니다. "
                       "**서로 다른 표의 숫자는 직접 비교할 수 없습니다.**")

        _order = [c.name for c in CAS_CONFIGS if c.name in groups] if groups.keys() != {""} else [""]
        for cas in _order:
            items = sorted(groups[cas], key=_score_key, reverse=True)
            label = SCORE_LABELS[items[0].score_kind]

            if cas:
                st.markdown(f"#### {cas}  ·  {len(items)}개 사이트")
            if items[0].score_kind == "percentile":
                st.caption("이 값은 절단 효율이 아니라 **학습 분포 안에서의 백분위**입니다. "
                           "높을수록 학습 데이터의 가이드들보다 우수하다는 뜻입니다.")
            if len(items) > MAX_ROWS:
                st.caption(f"상위 {MAX_ROWS}개만 표시합니다. (전체 {len(items)}개) "
                           "전체 결과는 '전체 상세 보기' 탭을 이용하세요.")

            rows = [{
                '순위':              rank,
                '최적 Cas 변종':      vr.cas_name,
                'PAM':               vr.pam_seq,
                '가이드 서열 (RNA)':  dna_to_rna(vr.guide_seq),
                '가닥':              vr.strand,
                '거리 (bp)':         vr.distance,
                label:               round(vr.efficiency_pct, 2),
                '원출력':             round(vr.raw_score, 4),
            } for rank, vr in enumerate(items[:MAX_ROWS], start=1)]

            df_best = pd.DataFrame(rows)
            show_df(
                df_best.style.background_gradient(subset=[label], **GRADIENT)
                             .format({label: "{:.2f}%", '원출력': "{:.4f}"}),
                hide_index=True, height=36 * (len(df_best) + 1) + 3,
            )
        st.caption("'원출력'은 모델이 낸 값 그대로입니다. SpCas9은 z-score라 "
                   "변종마다 평균·표준편차가 달라서, 원출력이 낮아도 효율이 더 높을 수 있습니다.")

# ════════════════════════════════════════════════════════
# 탭2: 변종 비교 — SpCas9 변종 9종 히트맵
#       (Cas12a는 변종이 없어 비교 대상이 아니므로 제외)
# ════════════════════════════════════════════════════════
with tab_cmp:
    sp = [vr for vr in vrs if vr.cas_name in SPCAS9_VARIANTS]
    if not sp:
        st.info("SpCas9 PAM 사이트가 발견되지 않았습니다.")
    else:
        _sp_label = SCORE_LABELS[sp[0].score_kind]
        st.markdown(
            f"행: 가이드 서열 / 열: SpCas9 변종. 값은 **{_sp_label}** 이며, "
            "**초록에 가까울수록 높고, 빨강에 가까울수록 낮습니다.** "
            "색이 칠해지지 않고 **'None'** 으로 표시된 칸은 "
            "그 변종이 인식하지 않는 PAM이라 예측 자체가 없는 자리입니다."
        )
        st.caption("Cas12a는 변종이 없어 비교할 대상이 없습니다. "
                   "결과는 'Best Candidates'와 '전체 상세 보기' 탭에서 확인하세요.")

        df_long = pd.DataFrame([{
            'guide_label': f"{dna_to_rna(vr.guide_seq)}  dist:{vr.distance}  "
                           f"PAM:{vr.pam_seq}  strand:{vr.strand}",
            'variant':     vr.cas_name,
            'score':       round(vr.efficiency_pct, 2),
        } for vr in sp])
        present  = df_long['variant'].unique()
        df_pivot = (
            df_long.pivot_table(index='guide_label', columns='variant',
                                values='score', aggfunc='max')
            .reindex(columns=[v for v in SPCAS9_VARIANTS if v in present])
            .assign(_max=lambda d: d.max(axis=1))
            .sort_values('_max', ascending=False)
            .drop(columns='_max')
        )
        total = len(df_pivot)
        if total > MAX_ROWS:
            st.caption(f"상위 {MAX_ROWS}개만 표시합니다. (전체 {total}개) "
                       "전체 결과는 '전체 상세 보기' 탭을 이용하세요.")
        df_pivot = df_pivot.head(MAX_ROWS)
        # column_config(NumberColumn)는 쓰지 않는다. 쓰면 Streamlit이 자체 렌더러로
        # 그려서 Styler의 서식(na_rep)과 결측 스타일이 모두 무시된다.
        _styler = df_pivot.style.background_gradient(**GRADIENT)
        # pandas가 결측에 칠한 검정을 덮어쓴다 (CSS는 뒤에 온 선언이 이긴다).
        _styler = (_styler.map(style_missing) if hasattr(_styler, "map")
                   else _styler.applymap(style_missing))
        show_df(
            _styler.format("{:.1f}%", na_rep=MISSING_TEXT),
            height=36 * (len(df_pivot) + 1) + 3,
        )

# ════════════════════════════════════════════════════════
# 탭3: 전체 상세 보기
# ════════════════════════════════════════════════════════
with tab_detail:
    st.markdown("원하는 조건을 설정한 뒤 **검색 버튼**을 눌러 결과를 확인하세요.")

    all_vr = all_results.sorted_variant_results()

    if not all_vr:
        st.info("예측 결과가 없습니다.")
    else:
        all_cas_names = sorted({vr.cas_name for vr in all_vr})

        for name in all_cas_names:
            if f"chk_{name}" not in st.session_state:
                st.session_state[f"chk_{name}"] = True

        st.markdown("#### 필터 설정")
        fc1, fc2 = st.columns([1, 1])

        with fc1:
            st.markdown("**Cas 변종 선택**")
            tc1, tc2 = st.columns(2)
            if tc1.button("전체 선택", key="sel_all"):
                for name in all_cas_names:
                    st.session_state[f"chk_{name}"] = True
                st.rerun()
            if tc2.button("전체 해제", key="desel_all"):
                for name in all_cas_names:
                    st.session_state[f"chk_{name}"] = False
                st.rerun()
            with st.container(height=280, border=True):
                for name in all_cas_names:
                    st.checkbox(name, key=f"chk_{name}")

        with fc2:
            st.markdown("**효율 범위 및 거리 설정**")
            eff_range = st.slider("점수 범위 (%)", min_value=0.0, max_value=100.0,
                                  value=(0.0, 100.0), step=0.5, format="%.1f%%",
                                  key="slider_eff")
            st.caption(f"선택 범위: {eff_range[0]:.1f}% ~ {eff_range[1]:.1f}%")

            dist_range = st.slider("거리 범위 (bp)", min_value=0, max_value=15,
                                   value=(0, 15), step=1, format="%d bp", key="slider_dist")
            st.caption(f"변이로부터 {dist_range[0]} ~ {dist_range[1]} bp 이내")

        st.markdown(" ")

        if st.button("검색", type="primary", key="detail_search"):
            sel_cas = [n for n in all_cas_names if st.session_state[f"chk_{n}"]]
            st.session_state.detail_filtered = [
                vr for vr in all_vr
                if vr.cas_name in sel_cas
                and eff_range[0] <= vr.efficiency_pct <= eff_range[1]
                and dist_range[0] <= vr.distance <= dist_range[1]
            ]

        if st.session_state.detail_filtered is not None:
            filtered = st.session_state.detail_filtered
            st.caption(f"검색 결과: {len(filtered)}개 / 전체 {len(all_vr)}개")

            if filtered:
                rows = [{
                    '순위':              i,
                    'Cas 변종':          vr.cas_name,
                    'PAM':               vr.pam_seq,
                    '가이드 서열 (RNA)':  dna_to_rna(vr.guide_seq),
                    '가닥':              vr.strand,
                    '거리 (bp)':         vr.distance,
                    '원출력':             round(vr.raw_score, 4),
                    EFF:                 round(vr.efficiency_pct, 2),
                    '점수 기준':          ("절단 효율" if vr.score_kind == "efficiency"
                                        else "학습 분포 백분위"),
                } for i, vr in enumerate(filtered, start=1)]
                df_detail = pd.DataFrame(rows)

                # 선택된 사이트 표시용 마스크. 사이트 키를 표에 열로 넣으면
                # Streamlit이 Styler의 열 숨김을 무시해 그대로 보이므로 밖에 둔다.
                _is_selected = [vr.site_key == sel_key for vr in filtered]

                def _highlight_selected(row):
                    return [HIGHLIGHT if _is_selected[row.name] else ""] * len(row)

                _n_sel = sum(_is_selected)
                st.caption(
                    f"노란 줄({_n_sel}개)은 'XAI 설명' 탭에서 선택한 사이트입니다. "
                    "한 사이트를 여러 변종이 인식하면 변종마다 한 줄씩 나오므로 여러 줄이 칠해집니다. "
                    "**'점수 기준' 열이 다른 행끼리는 점수를 직접 비교할 수 없습니다.**")
                show_df(
                    df_detail.style.apply(_highlight_selected, axis=1)
                             .background_gradient(subset=[EFF], **GRADIENT)
                             .format({EFF: "{:.2f}%", '원출력': "{:.4f}"}),
                    hide_index=True,
                )
                st.download_button(
                    label="CSV 다운로드",
                    data=df_detail.to_csv(index=False).encode('utf-8-sig'),
                    file_name="crispr_prediction_results.csv", mime="text/csv",
                    key="csv_download",
                )
            else:
                st.warning("필터 조건에 맞는 결과가 없습니다.")

# ════════════════════════════════════════════════════════
# 탭4: XAI 설명 — 연동된 실제 모델(SpCas9 · Cas12a)로 IG·SHAP·Rollout 계산
#
# 2학기 Model B(SaCas9) 사전계산 화면은 피드백에 따라 내렸다.
# 코드(xai/view.render_modelb, xai/data_source.py, xai/charts_modelb.py)와
# data/ 아래 데이터는 그대로 남아 있으므로, 필요하면 여기서 다시 불러 쓸 수 있다.
# ════════════════════════════════════════════════════════
with tab_xai:
    st.subheader("모델 예측 근거 상세 분석 (XAI)")
    xai_view.render_glossary()

    st.markdown(
        "예측 점수만 보면 **왜 그 점수가 나왔는지** 알 수 없습니다. "
        "여기서는 선택한 지점의 서열을 모델에 다시 넣어, "
        "**서열의 어느 글자가 그 점수를 만들었는지** 되짚어 봅니다.\n\n"
        "아래에서 분석할 지점과 가위 종류를 고르면 그래프가 나옵니다."
    )

    if sel_site is None:
        st.info("후보 사이트가 없어 XAI를 표시할 수 없습니다.")
    else:
        st.caption("**분석할 지점** — 앞 탭에서 찾은 절단 후보 중 하나를 고릅니다. "
                   "점수가 높은 순으로 정렬돼 있습니다.")
        st.selectbox(
            "분석 대상 사이트 (효율 높은 순)", list(site_map),
            format_func=lambda k: site_label(site_map[k]), key="sel_site",
        )
        predictor = predictors[("A", sel_site.cas_type)]
        heads     = predictor.explain_heads(sel_site)

        if not heads:
            st.warning("이 사이트의 PAM을 인식하는 변종이 없어 설명할 출력이 없습니다.")
        else:
            if len(heads) > 1:
                st.caption(
                    "**설명할 가위 종류** — 같은 지점이라도 가위 변종마다 성능이 다릅니다. "
                    f"이 지점의 표지판(PAM `{sel_site.pam_seq}`)을 인식하는 "
                    f"{len(heads)}종 중 하나를 고르세요. 고른 변종의 예측 근거만 계산합니다."
                )
            head = (st.selectbox("설명할 출력(변종)", heads, key="xai_head")
                    if len(heads) > 1 else heads[0])

            layout = WINDOW_LAYOUT[sel_site.cas_type]
            window = sel_site.model_input_seq
            _hit = next((v for v in vrs
                         if v.model == "A" and v.site_key == sel_key
                         and v.cas_name == head), None)
            eff = _hit.efficiency_pct if _hit else 0.0
            _kind = _hit.score_kind if _hit else "efficiency"

            st.divider()
            xai_view.render_header(sel_site, head, eff, layout, _kind)
            st.caption("아래 그래프의 숫자는 **효율 %p**(퍼센트 포인트) 단위입니다. "
                       "예를 들어 '+3.2%p'는 그 자리가 효율을 3.2퍼센트 포인트만큼 "
                       "끌어올렸다는 뜻입니다.")

            with st.expander("계산 설정"):
                st.caption(
                    "연동 모델이 4억 8천만 파라미터라 CPU에서는 추론 한 번도 "
                    "가볍지 않습니다. **IG는 단계 수만큼, SHAP은 (샘플 수 × 배경 수)만큼** "
                    "추론을 반복하니 값을 올리기 전에 소요 시간을 가늠해 보세요."
                )
                ig_steps = st.number_input("IG 적분 단계 수", 8, 256, 32, step=8,
                                           key="ig_steps",
                                           help="클수록 정확하지만 느립니다. 완결성 오차가 크면 늘리세요.")
                shap_n   = st.number_input("SHAP 샘플 수", 16, 1024, 64, step=16,
                                           key="shap_n")
                shap_bg  = st.number_input("SHAP 배경 서열 수", 2, 32, 4, step=2,
                                           key="shap_bg")
                st.caption(f"현재 설정 기준 추론 횟수 — IG {int(ig_steps) + 2}회, "
                           f"SHAP 약 {int(shap_n) * int(shap_bg):,}회")

            xai_method = st.radio(
                "XAI 분석 기법 선택",
                ["Attention Rollout", "Integrated Gradients (IG)",
                 "SHAP (Shapley Additive exPlanations)"],
                horizontal=True, key="xai_method_radio",
                help="Attention Rollout이 가장 빠릅니다(추론 1회). SHAP이 가장 느립니다.",
            )
            st.markdown(" ")

            model_key = f"A/{sel_site.cas_type}"
            try:
                if xai_method == "Attention Rollout":
                    xai_view.render_rollout(
                        xai_view.compute_rollout(predictor, model_key, head, window),
                        window, layout)
                elif xai_method == "Integrated Gradients (IG)":
                    xai_view.render_ig(
                        xai_view.compute_ig(predictor, model_key, head, window,
                                            int(ig_steps)),
                        window, layout)
                else:
                    xai_view.render_shap(
                        xai_view.compute_shap(predictor, model_key, head, window,
                                              int(shap_n), int(shap_bg)),
                        window, layout)
            except Exception as e:
                st.error(f"XAI 계산 중 오류가 발생했습니다: {type(e).__name__}: {e}")
