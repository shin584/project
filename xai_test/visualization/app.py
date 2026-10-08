"""Model B explanation dashboard - stage-1 draft (issue #20).

Run from this directory: `streamlit run app.py`. Renders only the
Colab-precomputed Testset results through `XAIDataSource`; see
`plan_draft.md` and ADR 0002 (Model B only - no Model A, no
Concordant/Discordant labels).
"""

import charts
import matplotlib.pyplot as plt
import streamlit as st
from data_source import (
    DISTAL_REGION,
    PAM_REGION,
    SEED_REGION,
    PrecomputedDataSource,
    XAIDataSource,
)

st.set_page_config(page_title="Model B Explainer", layout="wide")


@st.cache_resource
def get_data_source() -> XAIDataSource:
    return PrecomputedDataSource()


def show_figure(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


source = get_data_source()
ig_ids = set(source.ig_sample_ids())

st.title("🧬 SaCas9 Model B 예측 설명 대시보드")
st.markdown(
    "Model B(NT 임베딩 + 물리 피처 → XGBoost)가 SaCas9 절단 효율 점수를 "
    "내린 근거를 시각화합니다. 현재는 Testset 514개에 대해 사전 계산된 결과만 표시합니다."
)

with st.sidebar:
    st.header("샘플 선택")
    sample_id = st.selectbox(
        "sample_id (Testset 0–513)",
        options=source.sample_ids(),
        format_func=lambda i: f"{i}  · IG 있음" if i in ig_ids else str(i),
        key="sample_id",
    )
    st.caption(
        f"IG(Integrated Gradients)는 {len(ig_ids)}개 샘플에만 계산되어 있습니다."
    )

bundle = source.get_sample(sample_id)

st.subheader(f"sample_id {bundle.sample_id}")
st.code(bundle.sequence, language=None)
cols = st.columns(3)
cols[0].metric("Model B 예측값", f"{bundle.prediction:.4f}")
if bundle.true_score is not None:
    cols[1].metric("실측값 (Testset)", f"{bundle.true_score:.4f}")
    cols[2].metric("절대 오차", f"{bundle.error:.4f}")

tab_rank, tab_heat, tab_detail, tab_xai = st.tabs(
    ["최고 후보군", "위치별 히트맵", "예측 상세", "XAI 분석"]
)

# ── Tab 1: ranking ───────────────────────────────────────────────────
with tab_rank:
    st.markdown("Testset 서열을 **Model B 예측 점수** 순으로 정렬한 랭킹입니다.")
    ranking = source.ranking()
    ranking.insert(0, "순위", range(1, len(ranking) + 1))
    selected_rank = int(ranking.index[ranking["sample_id"] == sample_id][0]) + 1
    st.caption(f"선택한 샘플의 순위: {selected_rank} / {len(ranking)}")
    st.dataframe(
        ranking.style.background_gradient(subset=["prediction"], cmap="RdYlGn").format(
            {"prediction": "{:.4f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

# ── Tab 2: ISM heatmap + complex mismatch ────────────────────────────
with tab_heat:
    st.markdown(
        "**ISM(In-silico Mutagenesis) 히트맵**: 각 위치의 염기를 다른 3개 염기로 바꿨을 때 "
        "예측 점수가 얼마나 변하는지(상대 변화량)입니다. 빨강 = 점수 상승, 파랑 = 점수 하락. "
        "칸 안의 글자는 바꿔 넣은 염기입니다."
    )
    st.caption(
        f"PAM {PAM_REGION[0]}–{PAM_REGION[1]}, Seed {SEED_REGION[0]}–{SEED_REGION[1]}, "
        f"Distal {DISTAL_REGION[0]}–{DISTAL_REGION[1]} (0-based 위치)"
    )
    show_figure(
        charts.ism_heatmap(bundle.ism_delta, bundle.ism_alt_bases, bundle.sequence)
    )
    st.markdown(
        "**복합 미스매치 시나리오 (15개)**: 여러 위치를 동시에 바꿨을 때의 점수 상대 변화량입니다."
    )
    show_figure(
        charts.complex_mismatch_bars(
            bundle.complex_mismatch_delta, source.complex_scenarios()
        )
    )

# ── Tab 3: prediction detail ─────────────────────────────────────────
with tab_detail:
    st.table(charts.prediction_detail_table(bundle))
    if bundle.true_score is not None:
        st.caption("실측값과 오차는 Testset 샘플에서만 제공됩니다.")
    st.caption(
        "MFE: gRNA 최소 자유 에너지 · ΔG: gRNA–DNA 결합 자유 에너지 · "
        "Tm: 녹는점 · GC: GC 함량(%)"
    )

# ── Tab 4: XAI view ──────────────────────────────────────────────────
with tab_xai:
    st.markdown("#### Attribution Logo (Integrated Gradients)")
    if bundle.ig_attribution is not None:
        st.caption(
            "각 위치의 실제 염기 글자 높이 = 부호 있는 IG 기여도 (위: 점수 상승, 아래: 점수 하락). "
            "**Deterministic Projection Rule** 가정: 6-mer 토큰의 기여도를 해당 6개 염기에 "
            "균등하게 나눈 것이며, 모델이 염기를 개별적으로 인식한다는 근거는 아닙니다."
        )
        show_figure(charts.attribution_logo(bundle.ig_attribution, bundle.sequence))
    else:
        st.info(
            "IG 미계산: 이 샘플에는 Integrated Gradients가 계산되어 있지 않습니다. "
            "사이드바에서 'IG 있음' 표시된 샘플을 선택하세요."
        )

    st.markdown("#### Attention Rollout 위치 프로파일")
    st.caption(
        "각 위치가 다른 위치들로부터 얼마나 참조되는지(정보 의존도)를 합산한 값입니다. "
        "모델 내부 정보 의존성이며 물리적 결합이 아님에 유의하세요."
    )
    show_figure(charts.rollout_profile(bundle.attention_rollout, bundle.sequence))
    with st.expander("원본 36×36 Attention Rollout 히트맵 보기"):
        show_figure(charts.rollout_heatmap(bundle.attention_rollout))

    st.markdown("#### Token-grouped SHAP")
    st.caption(
        "임베딩 SHAP을 7개 토큰(6-mer 6개 + [CLS] = 전역 서열 문맥)별로 부호 있게 합산하고, "
        "물리 피처 4개는 개별로 표시합니다. 빨강 = 점수를 올림, 파랑 = 점수를 내림. "
        "그래프의 E[f(X)]는 기준값으로, export에 저장되지 않아 "
        "Testset 전체의 (예측값 − SHAP 합) 평균으로 근사했습니다."
    )
    shap_output = bundle.shap_base_value + sum(g.shap_value for g in bundle.shap_groups)
    st.caption(
        f"SHAP 합산값 f(x) = {shap_output:.4f}, 실제 Model B 예측값 = {bundle.prediction:.4f}. "
        "SHAP은 별도 Colab 실행에서 계산되어 두 값이 조금 다를 수 있습니다."
    )
    show_figure(charts.shap_waterfall(bundle.shap_groups, bundle.shap_base_value))
