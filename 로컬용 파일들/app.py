import streamlit as st
import pandas as pd
from pam_scanner import print_pam_analysis, validate_sequence, CAS_CONFIGS, SPCAS9_VARIANTS, dna_to_rna
from predict import load_models, run_prediction

st.set_page_config(page_title="CRISPR PAM Scanner", layout="wide")
st.title("🧬 CRISPR PAM Scanner")
st.markdown("81bp DNA 서열을 입력하면 PAM 탐색 및 절단 효율을 예측합니다.")



def get_models():
    return load_models()

model, models_sa, model_cas12a = get_models()

if 'all_results' not in st.session_state:
    st.session_state.all_results = None
if 'detail_filtered' not in st.session_state:
    st.session_state.detail_filtered = None

dna_input = st.text_input("81bp DNA 서열 입력", placeholder="ATCG...")
run_btn   = st.button("🔍 분석 시작")

if run_btn and dna_input:
    try:
        validate_sequence(dna_input)

        with st.spinner("PAM 탐색 중..."):
            all_results = print_pam_analysis(dna_input)

        with st.spinner("모델 예측 중..."):
            all_results = run_prediction(all_results, model, models_sa, model_cas12a)

        st.session_state.all_results     = all_results
        st.session_state.detail_filtered = None

    except ValueError as e:
        st.error(f"❌ 오류: {e}")

if st.session_state.all_results is not None:
    all_results = st.session_state.all_results

    st.subheader("📊 PAM 탐색 결과")
    col1, col2, col3 = st.columns(3)
    for col, cfg in zip([col1, col2, col3], CAS_CONFIGS):
        sites = all_results.by_cas(cfg.name)
        col.metric(label=cfg.name, value=f"{len(sites)}개 발견")

    st.subheader("🤖 절단 효율 예측 결과")

    tab_best, tab_sp, tab_others, tab_detail = st.tabs([
        "🏆 Best Candidates",
        "🔬 SpCas9 변종 히트맵",
        "🧪 SaCas9 / Cas12a",
        "📋 전체 상세 보기",
    ])

    # ════════════════════════════════════════════════════════
    # 탭1: Best Candidates
    # ════════════════════════════════════════════════════════
    with tab_best:
        st.markdown(
            "각 가이드 서열에서 **가장 높은 효율을 보인 변종 1개**만 대표로 표시합니다. "
            "자세한 변종별 비교는 다른 탭을 확인하세요."
        )
        if not all_results.variant_results:
            st.info("예측 결과가 없습니다.")
        else:
            guide_best: dict[str, object] = {}
            for vr in all_results.variant_results:
                key = vr.guide_seq
                if key not in guide_best or vr.final_score > guide_best[key].final_score:
                    guide_best[key] = vr

            sorted_best = sorted(guide_best.values(), key=lambda v: v.final_score, reverse=True)
            total_best  = len(sorted_best)
            display_best = sorted_best[:20]   # 최대 20개

            if total_best > 20:
                st.caption(f"상위 20개만 표시합니다. (전체 {total_best}개) 전체 결과는 '전체 상세 보기' 탭을 이용하세요.")

            rows_best = []
            for rank, vr in enumerate(display_best, start=1):
                rows_best.append({
                    '순위':              rank,
                    '최적 Cas 변종':      vr.cas_name,
                    'PAM':               vr.pam_seq,
                    '가이드 서열 (DNA)':  vr.guide_seq,
                    '가이드 서열 (RNA)':  dna_to_rna(vr.guide_seq),
                    '가닥':              vr.strand,
                    '거리 (bp)':         vr.distance,
                    '최종 효율 (%)':      round(vr.final_score, 2),
                })

            df_best = pd.DataFrame(rows_best)
            st.dataframe(
                df_best.style.background_gradient(
                    subset=['최종 효율 (%)'], cmap='RdYlGn', vmin=0, vmax=100
                ),
                use_container_width=True,
                hide_index=True,
                height=36 * (len(df_best) + 1) + 3,
            )

    # ════════════════════════════════════════════════════════
    # 탭2: SpCas9 변종 히트맵
    # ════════════════════════════════════════════════════════
    with tab_sp:
        sp_results = [vr for vr in all_results.variant_results
                      if vr.cas_name in SPCAS9_VARIANTS]

        if not sp_results:
            st.info("SpCas9 PAM 사이트가 발견되지 않았습니다.")
        else:
            st.markdown(
                "행: 가이드 서열 / 열: SpCas9 변종. "
                "**색이 밝을수록(노랑/주황) 절단 효율이 높고, 진할수록(빨강) 낮습니다.**"
            )
            pivot_rows = []
            for vr in sp_results:
                # 옵션F: RNA 서열 + 라벨 명시 메타정보
                label = (
                    f"{dna_to_rna(vr.guide_seq)}  dist:{vr.distance}  PAM:{vr.pam_seq}  strand:{vr.strand}"
                )
                pivot_rows.append({
                    'guide_label': label,
                    'variant':     vr.cas_name,
                    'score':       round(vr.final_score, 2),
                })

            df_pivot = (
                pd.DataFrame(pivot_rows)
                .pivot_table(index='guide_label', columns='variant',
                             values='score', aggfunc='max')
                .reindex(columns=[v for v in SPCAS9_VARIANTS
                                  if v in pd.DataFrame(pivot_rows)['variant'].unique()])
                .assign(_max=lambda d: d.max(axis=1))
                .sort_values('_max', ascending=False)
                .drop(columns='_max')
            )

            total_sp = len(df_pivot)
            if total_sp > 20:
                st.caption(f"상위 20개만 표시합니다. (전체 {total_sp}개) 전체 결과는 '전체 상세 보기' 탭을 이용하세요.")
            df_pivot = df_pivot.head(20)

            # 모든 열(변종 점수 열) 너비를 100px으로 균등하게 설정
            col_config = {col: st.column_config.NumberColumn(col, width=100)
                          for col in df_pivot.columns}

            st.dataframe(
                df_pivot.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=100)
                        .format("{:.1f}"),
                use_container_width=True,
                height=36 * (len(df_pivot) + 1) + 3,
                column_config=col_config,
            )

    # ════════════════════════════════════════════════════════
    # 탭3: SaCas9 / Cas12a
    # ════════════════════════════════════════════════════════
    with tab_others:
        other_results = [vr for vr in all_results.variant_results
                         if vr.cas_name in ('SaCas9', 'Cas12a')]

        if not other_results:
            st.info("SaCas9 / Cas12a PAM 사이트가 발견되지 않았습니다.")
        else:
            rows_ot = []
            for i, vr in enumerate(
                sorted(other_results, key=lambda v: v.final_score, reverse=True), start=1
            ):
                rows_ot.append({
                    '순위':              i,
                    'Cas 종류':          vr.cas_name,
                    'PAM':               vr.pam_seq,
                    '가이드 서열 (DNA)':  vr.guide_seq,
                    '가이드 서열 (RNA)':  dna_to_rna(vr.guide_seq),  # RNA 추가
                    '가닥':              vr.strand,
                    '거리 (bp)':         vr.distance,
                    'Raw Score':         round(vr.raw_score, 4),
                    '최종 효율 (%)':      round(vr.final_score, 2),
                })

            st.dataframe(
                pd.DataFrame(rows_ot).style.background_gradient(
                    subset=['최종 효율 (%)'], cmap='RdYlGn', vmin=0, vmax=100
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ════════════════════════════════════════════════════════
    # 탭4: 전체 상세 보기
    # ════════════════════════════════════════════════════════
    with tab_detail:
        st.markdown("원하는 조건을 설정한 뒤 **검색 버튼**을 눌러 결과를 확인하세요.")

        all_vr = all_results.sorted_variant_results()

        if not all_vr:
            st.info("예측 결과가 없습니다.")
        else:
            all_cas_names = sorted(set(vr.cas_name for vr in all_vr))

            for name in all_cas_names:
                if f"chk_{name}" not in st.session_state:
                    st.session_state[f"chk_{name}"] = True

            st.markdown("#### 🔧 필터 설정")
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
                st.markdown(" ")

                eff_range = st.slider(
                    "최종 효율 범위 (%)",
                    min_value=0.0, max_value=100.0,
                    value=(0.0, 100.0), step=0.5,
                    format="%.1f%%",
                    key="slider_eff",
                )
                st.caption(f"선택 범위: {eff_range[0]:.1f}% ~ {eff_range[1]:.1f}%")

                st.markdown(" ")

                dist_range = st.slider(
                    "거리 범위 (bp)",
                    min_value=0, max_value=15,
                    value=(0, 15), step=1,
                    format="%d bp",
                    key="slider_dist",
                )
                st.caption(f"변이로부터 {dist_range[0]} ~ {dist_range[1]} bp 이내")

            st.markdown(" ")

            if st.button("🔍 검색", type="primary", key="detail_search"):
                sel_cas          = [n for n in all_cas_names if st.session_state[f"chk_{n}"]]
                min_eff, max_eff = eff_range
                min_dist, max_dist = dist_range

                st.session_state.detail_filtered = [
                    vr for vr in all_vr
                    if vr.cas_name in sel_cas
                    and min_eff <= vr.final_score <= max_eff
                    and min_dist <= vr.distance <= max_dist
                ]

            if st.session_state.detail_filtered is not None:
                filtered = st.session_state.detail_filtered
                st.caption(f"검색 결과: {len(filtered)}개 / 전체 {len(all_vr)}개")

                if filtered:
                    rows_detail = []
                    for i, vr in enumerate(filtered, start=1):
                        rows_detail.append({
                            '순위':              i,
                            'Cas 변종':          vr.cas_name,
                            'PAM':               vr.pam_seq,
                            '가이드 서열 (DNA)':  vr.guide_seq,
                            '가이드 서열 (RNA)':  dna_to_rna(vr.guide_seq),  # RNA 추가
                            '가닥':              vr.strand,
                            '거리 (bp)':         vr.distance,
                            'Raw Score':         round(vr.raw_score, 4),
                            '최종 효율 (%)':      round(vr.final_score, 2),
                        })

                    df_detail = pd.DataFrame(rows_detail)
                    st.dataframe(
                        df_detail.style.background_gradient(
                            subset=['최종 효율 (%)'], cmap='RdYlGn', vmin=0, vmax=100
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv = df_detail.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="⬇️ CSV 다운로드",
                        data=csv,
                        file_name="crispr_prediction_results.csv",
                        mime="text/csv",
                        key="csv_download",
                    )
                else:
                    st.warning("필터 조건에 맞는 결과가 없습니다.")
