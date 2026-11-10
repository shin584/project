"""XAI 화면 조각들. app.py의 XAI 탭에서 호출한다.

모듈 레벨에서 st.* 화면 출력을 하지 않는다 (set_page_config 순서 보호).

두 갈래가 있다.
  1) 실제 연동 모델(SpCas9 · Cas12a) → xai/explain.py가 모델에서 IG·SHAP을 직접 계산,
     xai/charts.py로 그린다.
  2) Model B(SaCas9) 사전계산 결과 → xai/data_source.py가 읽고,
     xai/charts_modelb.py로 2학기와 같은 형태로 그린다. (기존 유지)
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from xai import charts, charts_modelb, data_source, explain


# ── 계산 (결과는 캐시. 모델 파일을 바꾼 뒤에는 앱을 다시 시작하거나 캐시를 비우세요) ──
@st.cache_data(show_spinner="IG 계산 중...")
def compute_ig(_predictor, model_key: str, head: str, window_seq: str, steps: int):
    return explain.integrated_gradients(_predictor.explain_handle(head), window_seq, steps)


@st.cache_data(show_spinner="SHAP 계산 중... (몇 초 걸릴 수 있어요)")
def compute_shap(_predictor, model_key: str, head: str, window_seq: str,
                 nsamples: int, n_background: int):
    return explain.kernel_shap(_predictor.explain_handle(head), window_seq,
                               nsamples, n_background)


@st.cache_data(show_spinner="Attention Rollout 계산 중...")
def compute_rollout(_predictor, model_key: str, head: str, window_seq: str):
    return explain.attention_rollout(_predictor.explain_handle(head), window_seq)


@st.cache_resource(show_spinner="Model B 사전계산 결과 불러오는 중...")
def get_modelb_source():
    """Model B(SaCas9) 사전계산 데이터. 파일이 없으면 None."""
    if not data_source.data_available():
        return None
    return data_source.PrecomputedDataSource()


# ── 공용 헬퍼 ────────────────────────────────────────────────────────
def show_figure(fig) -> None:
    """사이드바 '그래프 크기' 슬라이더 값(%)만큼 가운데 열에 그린다."""
    frac = st.session_state.get("fig_pct", 80) / 100
    if frac >= 0.99:
        st.pyplot(fig)
    else:
        pad = (1 - frac) / 2
        _, mid, _ = st.columns([pad, frac, pad])
        with mid:
            st.pyplot(fig)
    plt.close(fig)


def colored_sequence(sequence: str, layout) -> str:
    cells = []
    for pos, base in enumerate(sequence):
        color = charts.SEGMENT_COLORS[charts.segment_of(pos, layout)]
        cells.append(f"<span style='background-color:{color}55;padding:2px 1px;"
                     f"border-radius:3px'>{base}</span>")
    return ("<div style='font-family:monospace;font-size:1.4rem;letter-spacing:2px'>"
            + "".join(cells) + "</div>")


def segment_legend(layout) -> str:
    chips = []
    for name in dict.fromkeys(n for n, _, _ in layout):
        spans = ", ".join(f"{a}–{b}" for n, a, b in layout if n == name)
        chips.append(f"<span style='background-color:{charts.SEGMENT_COLORS[name]}55;padding:2px 8px;"
                     f"border-radius:3px;margin-right:6px'>{name} {spans}</span>")
    return "<div style='font-size:0.85rem'>" + "".join(chips) + "</div>"


def render_glossary() -> None:
    with st.expander("처음이신가요? 5분이면 되는 배경 설명", expanded=False):
        st.markdown(
            """
#### 이 도구가 하는 일

유전자를 고치려면 **DNA의 특정 지점을 잘라야** 합니다. 그 가위 역할을 하는 것이
CRISPR 단백질(SpCas9, Cas12a 등)입니다. 그런데 가위가 아무 데나 잘 드는 건 아니어서,
같은 유전자라도 **어느 지점을 노리느냐에 따라 잘 잘리기도 하고 거의 안 잘리기도 합니다.**

이 도구는 입력한 DNA 서열에서 자를 수 있는 후보 지점을 모두 찾고,
AI 모델로 **각 지점이 얼마나 잘 잘릴지** 예측합니다. XAI 탭은 거기서 한 걸음 더 나아가
**모델이 왜 그렇게 예측했는지**를 보여줍니다.

#### 서열의 세 구간

모델에 들어가는 30~34글자 구간은 역할이 셋으로 나뉩니다.

| 구간 | 쉽게 말하면 | 자세히 |
|---|---|---|
| **PAM** | 가위가 찾는 **표지판** | 이 짧은 신호가 없으면 가위가 아예 달라붙지 못합니다. 가위 종류마다 찾는 표지판이 다릅니다 |
| **guide** | **자를 위치를 지정하는 주소** | 가이드 RNA가 이 부분과 짝을 이뤄 가위를 데려옵니다 |
| **flank** | 그 **주변 환경** | 직접 관여하진 않지만 가위가 접근하기 쉬운지에 영향을 줍니다 |

그래프 위쪽의 색 막대가 이 구간들을 표시합니다.

#### 숫자 읽는 법

- **예측 절단 효율 (%)** — 이 지점이 얼마나 잘 잘릴지. **높을수록 좋습니다.**
- **%p (퍼센트 포인트)** — 효율의 변화량입니다. 30%에서 35%가 되면 "+5%p"입니다.
  "5% 증가"(31.5%)와 다른 개념이라 구분해서 씁니다.
- **변이와의 거리** — 고치려는 지점에서 자르는 위치까지의 거리. **가까울수록 유리합니다.**

#### 세 가지 분석 방법의 차이

같은 서열을 보고도 서로 다른 질문에 답합니다. **결과가 달라도 모순이 아닙니다.**

| 방법 | 답하는 질문 | 속도 |
|---|---|---|
| **Attention Rollout** | 모델이 어디를 **자주 쳐다봤나** | 빠름 |
| **IG** | 각 글자가 효율을 **올렸나 내렸나** | 보통 |
| **SHAP** | 평균적인 서열과 비교해 각 자리가 **얼마나 밀고 당겼나** | 느림 |
"""
        )


# ── 실제 모델 XAI 본문 ───────────────────────────────────────────────
def render_header(site, head: str, efficiency_pct: float, layout,
                  score_kind: str = "efficiency") -> None:
    st.markdown("**모델이 실제로 읽은 구간**")
    st.markdown(colored_sequence(site.model_input_seq, layout), unsafe_allow_html=True)
    st.markdown(segment_legend(layout), unsafe_allow_html=True)
    st.caption("초록 **PAM**은 가위가 찾는 표지판, 파랑 **guide**는 자를 위치를 지정하는 주소, "
               "회색 **flank**는 주변 서열입니다. 아래 그래프의 가로축이 이 구간과 같습니다.")
    cols = st.columns(4)
    cols[0].metric("설명 대상", head)
    cols[1].metric("예측 절단 효율" if score_kind == "efficiency" else "학습 분포 백분위",
                   f"{efficiency_pct:.1f}%")
    cols[2].metric("PAM", site.pam_seq)
    cols[3].metric("변이와의 거리", f"{site.distance} bp")


def render_ig(result, sequence: str, layout) -> None:
    st.markdown("#### 글자별 기여도 (Attribution Logo, IG)")
    st.markdown(
        "**한 줄 요약 — 어느 글자가 효율을 올렸고, 어느 글자가 깎아먹었는지 보여줍니다.**\n\n"
        "모델에게 *\"이 서열의 글자를 하나도 모르는 상태에서 시작해 지금 서열까지 "
        "조금씩 채워 넣으면, 점수가 어떻게 변하나?\"* 를 물어본 결과입니다. "
        "그 변화량을 글자마다 나눠 가진 것이 아래 그래프입니다.\n\n"
        "**읽는 법** · 각 자리에 그 자리의 실제 염기가 글자로 그려집니다. "
        "**글자가 위로 클수록 효율을 많이 올린 자리**, **아래로 클수록 깎아먹은 자리**입니다. "
        "글자가 거의 안 보이면 있으나 없으나 비슷했다는 뜻입니다."
    )
    pct = result.attribution * 100
    show_figure(charts.attribution_logo(pct, sequence, layout))
    up, down = int(np.argmax(pct)), int(np.argmin(pct))
    seg = charts.segment_of
    st.success(
        "**이 그래프가 말하는 것**\n\n"
        f"- 효율을 **가장 많이 올린 자리**: {up}번째 글자 `{sequence[up]}` "
        f"({seg(up, layout)} 구간) → {pct[up]:+.2f}%p\n"
        f"- 효율을 **가장 많이 깎은 자리**: {down}번째 글자 `{sequence[down]}` "
        f"({seg(down, layout)} 구간) → {pct[down]:+.2f}%p"
    )
    with st.expander("기술적 참고"):
        st.markdown(
            "- 기준 입력은 **빈 서열(모든 염기 값 0)** 이고, 기준 대비 변화를 위치별로 나눈 값입니다.\n"
            "- 기여도 합이 (이 서열의 예측 − 기준의 예측)과 같아야 계산이 맞습니다 (완결성 검사)."
        )
        st.caption(
            f"예측 {result.pred * 100:.2f}% − 기준 {result.baseline_pred * 100:.2f}% "
            f"= {(result.pred - result.baseline_pred) * 100:+.3f}%p, "
            f"기여도 합 {pct.sum():+.3f}%p, 완결성 오차 {result.completeness_error * 100:.4f}%p"
        )


def render_shap(result, sequence: str, layout) -> None:
    st.markdown("#### 위치별 기여분 (SHAP)")
    st.markdown(
        "**한 줄 요약 — 평범한 서열과 비교해서, 이 서열의 각 자리가 점수를 얼마나 올리고 내렸는지입니다.**\n\n"
        "IG가 \"아무것도 없는 상태\"와 비교한다면, SHAP은 **\"무작위로 만든 평범한 서열들의 평균\"** "
        "과 비교합니다. 기준점이 다르니 강조하는 자리도 다를 수 있습니다.\n\n"
        "**읽는 법** · 막대그래프에서 **빨강은 효율을 올린 자리, 파랑은 내린 자리**입니다. "
        "그 아래 워터폴 그래프는 기준점에서 출발해 각 자리의 기여를 하나씩 더해 "
        "최종 예측값에 도달하는 과정을 계단식으로 보여줍니다."
    )
    pct = result.values * 100
    show_figure(charts.shap_bars(pct, sequence, layout))
    show_figure(charts.shap_waterfall(pct, result.base_value * 100, sequence, layout))
    top = np.argsort(-np.abs(pct))[:3]
    st.success(
        "**영향이 가장 컸던 자리 3곳**\n\n"
        + "\n".join(
            f"- {i}번째 글자 `{sequence[i]}` ({charts.segment_of(int(i), layout)} 구간) → "
            f"효율을 {'올림' if pct[i] > 0 else '깎음'} ({pct[i]:+.2f}%p)" for i in top)
    )
    with st.expander("기술적 참고 (기준값 · 합산값)"):
        st.markdown(
            "- 위치별 KernelSHAP입니다. 어떤 자리를 '끈다'는 것은 그 자리를 **무작위 배경 서열의 염기**로 바꾼다는 뜻입니다.\n"
            "- 무작위 배경에는 PAM 자리도 포함되므로, PAM의 기여가 크게 나올 수 있습니다.\n"
            "- 추정에 샘플링을 쓰므로 계산 설정(샘플 수)을 바꾸면 값이 조금 달라질 수 있습니다."
        )
        st.caption(
            f"기준값 {result.base_value * 100:.2f}% + SHAP 합 {pct.sum():+.2f}%p "
            f"= {(result.base_value + result.values.sum()) * 100:.2f}% "
            f"(실제 예측 {result.pred * 100:.2f}%, 가법성 오차 {result.additivity_error * 100:.4f}%p)"
        )


def render_rollout(result, sequence: str, layout) -> None:
    st.markdown("#### 위치별 정보 의존도 (Attention Rollout)")
    st.markdown(
        "**한 줄 요약 — 모델이 예측할 때 어느 자리를 자주 쳐다봤는지 보여줍니다.**\n\n"
        "글을 읽을 때 중요해 보이는 단어에 눈이 더 오래 머무는 것과 비슷합니다. "
        "이 모델은 내부적으로 \"어느 글자를 참고할지\" 스스로 정하는데, "
        "그 시선을 24개 층에 걸쳐 따라가 합친 것이 아래 그래프입니다.\n\n"
        "**읽는 법** · 막대가 **높을수록 모델이 그 자리를 많이 참고**했다는 뜻입니다.\n\n"
        "**주의** · 많이 쳐다봤다고 해서 \"좋은 자리\"라는 뜻은 아닙니다. "
        "여기에는 좋다·나쁘다의 방향이 없고 **주목도만** 있습니다. "
        "방향까지 알고 싶으면 IG나 SHAP을 보세요."
    )
    profile = result.position_profile
    show_figure(charts.rollout_profile(profile, sequence, layout))

    top = np.argsort(-profile)[:3]
    seg = charts.segment_of
    st.success(
        "**모델이 가장 자주 쳐다본 자리 3곳**\n\n"
        + "\n".join(f"- {i}번째 글자 `{sequence[i]}` ({seg(int(i), layout)} 구간)" for i in top)
        + "\n\n이 자리들이 효율에 좋았는지 나빴는지는 IG나 SHAP에서 확인하세요."
    )

    with st.expander(f"토큰 × 토큰 의존도 행렬 보기 (토큰 {len(result.token_labels)}개)"):
        st.caption(
            "모델은 염기를 하나씩 보지 않고 **6개씩 묶은 토큰**으로 읽습니다. "
            "행이 열을 얼마나 참조하는지 나타내며, 'special'은 문장 시작 토큰입니다."
        )
        show_figure(charts.rollout_token_heatmap(result.token_matrix, result.token_labels))

    with st.expander("기술적 참고"):
        st.markdown(
            f"- {result.n_layers}개 층의 attention을 헤드 평균한 뒤, 잔차 연결을 단위행렬로 "
            "더하고 행 합을 1로 맞춰 아래층부터 곱해 누적했습니다 (Abnar & Zuidema, 2020).\n"
            "- 토큰 단위 값을 그 토큰이 덮는 염기에 균등하게 나눠 위치별로 폈습니다. "
            "모델이 염기를 개별로 인식한다는 뜻은 아닙니다.\n"
            "- **모델 내부의 정보 의존성이지 물리적 결합이 아닙니다.** "
            "DNA가 그 자리에서 실제로 상호작용한다는 근거로 쓰면 안 됩니다."
        )


# ── Model B(SaCas9) 사전계산 결과 — 2학기 화면 그대로 ────────────────
def render_modelb(source) -> None:
    """2학기 대시보드의 Model B 설명 화면. 사전계산된 Testset 결과를 그대로 보여준다."""
    ds = data_source
    ig_ids = set(source.ig_sample_ids())

    st.info(
        "이 섹션은 **Model B(SaCas9)** 를 Testset으로 돌려 Colab에서 미리 계산해 둔 결과입니다. "
        "위쪽 섹션의 실제 모델 XAI와 달리 입력 서열에 따라 다시 계산되지 않습니다.",
    )

    sample_id = st.selectbox(
        "sample_id (Testset 0–513)",
        options=source.sample_ids(),
        format_func=lambda i: f"{i}  · IG 있음" if i in ig_ids else str(i),
        key="modelb_sample_id",
    )
    st.caption(f"IG(Integrated Gradients)는 {len(ig_ids)}개 샘플에만 계산되어 있습니다.")

    bundle = source.get_sample(sample_id)

    st.markdown(f"**sample_id {bundle.sample_id}**")
    st.code(bundle.sequence, language=None)
    cols = st.columns(3)
    cols[0].metric("Model B 예측값", f"{bundle.prediction:.4f}")
    if bundle.true_score is not None:
        cols[1].metric("실측값 (Testset)", f"{bundle.true_score:.4f}")
        cols[2].metric("절대 오차", f"{bundle.error:.4f}")

    # ── 위치별 히트맵 (ISM + 복합 미스매치) ──────────────────────────
    with st.expander("위치별 히트맵 (ISM · 복합 미스매치)", expanded=True):
        st.markdown(
            "**ISM(In-silico Mutagenesis) 히트맵**: 각 위치의 염기를 다른 3개 염기로 바꿨을 때 "
            "예측 점수가 얼마나 변하는지(상대 변화량)입니다. 빨강 = 점수 상승, 파랑 = 점수 하락. "
            "칸 안의 글자는 바꿔 넣은 염기입니다."
        )
        st.caption(
            f"PAM {ds.PAM_REGION[0]}–{ds.PAM_REGION[1]}, "
            f"Seed {ds.SEED_REGION[0]}–{ds.SEED_REGION[1]}, "
            f"Distal {ds.DISTAL_REGION[0]}–{ds.DISTAL_REGION[1]} (0-based 위치)"
        )
        show_figure(charts_modelb.ism_heatmap(
            bundle.ism_delta, bundle.ism_alt_bases, bundle.sequence))

        st.markdown("**복합 미스매치 시나리오 (15개)**: 여러 위치를 동시에 바꿨을 때의 점수 상대 변화량입니다.")
        show_figure(charts_modelb.complex_mismatch_bars(
            bundle.complex_mismatch_delta, source.complex_scenarios()))

    # ── 예측 상세 ────────────────────────────────────────────────────
    with st.expander("예측 상세 (물리 피처 포함)"):
        st.table(charts_modelb.prediction_detail_table(bundle))
        if bundle.true_score is not None:
            st.caption("실측값과 오차는 Testset 샘플에서만 제공됩니다.")
        st.caption("MFE: gRNA 최소 자유 에너지 · ΔG: gRNA–DNA 결합 자유 에너지 · "
                   "Tm: 녹는점 · GC: GC 함량(%)")

    # ── Testset 랭킹 ─────────────────────────────────────────────────
    with st.expander("Testset 랭킹 (Model B 예측 점수 순)"):
        ranking = source.ranking()
        ranking.insert(0, "순위", range(1, len(ranking) + 1))
        selected_rank = int(ranking.index[ranking["sample_id"] == sample_id][0]) + 1
        st.caption(f"선택한 샘플의 순위: {selected_rank} / {len(ranking)}")
        st.dataframe(
            ranking.style.background_gradient(subset=["prediction"], cmap="RdYlGn")
                   .format({"prediction": "{:.4f}"}),
            hide_index=True,
        )

    # ── Attribution Logo (IG) ────────────────────────────────────────
    st.markdown("#### Attribution Logo (Integrated Gradients)")
    if bundle.ig_attribution is not None:
        st.caption(
            "각 위치의 실제 염기 글자 높이 = 부호 있는 IG 기여도 (위: 점수 상승, 아래: 점수 하락). "
            "**Deterministic Projection Rule** 가정: 6-mer 토큰의 기여도를 해당 6개 염기에 "
            "균등하게 나눈 것이며, 모델이 염기를 개별적으로 인식한다는 근거는 아닙니다."
        )
        show_figure(charts_modelb.attribution_logo(bundle.ig_attribution, bundle.sequence))
    else:
        st.info("IG 미계산: 이 샘플에는 Integrated Gradients가 계산되어 있지 않습니다. "
                "위 선택창에서 'IG 있음' 표시된 샘플을 선택하세요.")

    # ── Attention Rollout ────────────────────────────────────────────
    st.markdown("#### Attention Rollout 위치 프로파일")
    st.caption(
        "각 위치가 다른 위치들로부터 얼마나 참조되는지(정보 의존도)를 합산한 값입니다. "
        "모델 내부 정보 의존성이며 물리적 결합이 아님에 유의하세요."
    )
    show_figure(charts_modelb.rollout_profile(bundle.attention_rollout, bundle.sequence))
    with st.expander("원본 36×36 Attention Rollout 히트맵 보기"):
        show_figure(charts_modelb.rollout_heatmap(bundle.attention_rollout))

    # ── Token-grouped SHAP ───────────────────────────────────────────
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
    show_figure(charts_modelb.shap_waterfall(bundle.shap_groups, bundle.shap_base_value))
