"""#181 동등성 기준 — 고정 입력 1건의 Streamlit 화면 산출물을 기계 비교용으로 저장한다.

게이트 2는 "데스크톱 화면이 현재 Streamlit 화면과 같은가"를 판정한다. 스크린샷만으로는
점수 소수점과 정렬 동점 처리가 안 보이므로, 같은 입력에 대한 산출물을 수치로 떠 둔다.

여기서 만드는 것 (모두 `docs/equivalence-baseline/` 아래)
  candidates.json                     후보 목록·정렬·점수 (화면 표시값과 원출력 둘 다)
  crispr_prediction_results.csv       '전체 상세 보기'의 CSV 다운로드와 바이트 단위로 같은 파일
  xai-<cas>-<head>.npz                IG·SHAP·Rollout 배열 (np.allclose로 비교)
  capture.json                        입력·설정·환경 기록

화면 코드를 호출하지 않고 코어(`core`)를 직접 부른다. 행 구성과 정렬·반올림은
`app.py`의 표 생성 코드를 그대로 옮겼다. 화면과 어긋나면 그건 이 파일의 버그다.

사용법 (가중치와 토크나이저가 있어야 한다)
    python docs/equivalence-baseline/capture_baseline.py

SHAP 주의 — `shap.KernelExplainer`는 전역 `np.random`을 쓰고 seed를 받지 않아 실행마다
값이 조금 달라진다. 그래서 계산한 SHAP을 `data/precomputed_shap/`에도 저장한다
(`kernel_shap`이 계산 전에 조회하는 그 저장소다). 저장 후에는 두 화면이 같은 캐시를
읽으므로 `np.allclose`가 아니라 완전히 같은 값이 나온다. 캐시를 지우고 다시 계산하면
SHAP만 ~1e-3 수준으로 흔들릴 수 있다.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import core  # noqa: E402
from xai import shap_cache  # noqa: E402

# ── 고정 입력 ────────────────────────────────────────────────────────
# #179 게이트 1 스모크와 같은 서열이다. 이 서열을 고른 이유:
#   · NGG PAM(AGG·TGG)이 있어 SpCas9이 실제로 지연 로딩된다
#   · Cas12a TTTV 사이트가 양쪽 가닥에 하나씩 있다
#   · 두 Cas 모두 사이트가 있어 '점수 단위가 섞인 표'(Best Candidates의 2표 분기)를 덮는다
FIXED_INPUT = "CAAAGTAGCCGGTCGCTTTAAGAATAAATAAAAGCGCCACAGGATTGACTAAACCACAGCCGGCGCCCTGGCCCCCTACCG"

# 화면의 '계산 설정' 기본값. 바꾸면 SHAP 캐시 키가 달라져 시연 경로에서 미적중이 난다.
IG_STEPS = 32
SHAP_NSAMPLES = 64
SHAP_N_BACKGROUND = 4

MAX_ROWS = 20  # app.py의 MAX_ROWS
EFF = "예측 점수 (%)"
SCORE_LABELS = {
    "efficiency": "예측 절단 효율 (%)",
    "percentile": "학습 분포 백분위 (%)",
}


def _score_key(vr):
    """app.py._score_key — 화면 효율이 1순위, 동점이면 원출력."""
    return (vr.efficiency_pct, vr.raw_score)


def best_per_site(results, model: str = "A") -> dict:
    """app.py.best_per_site 그대로."""
    best = {}
    for vr in results:
        if vr.model != model:
            continue
        cur = best.get(vr.site_key)
        if cur is None or _score_key(vr) > _score_key(cur):
            best[vr.site_key] = vr
    return best


def best_candidates_tables(vrs) -> list[dict]:
    """'Best Candidates' 보기의 표들. 점수 단위가 섞이면 Cas별로 나뉜다."""
    a_best = best_per_site(vrs, "A")
    kinds = {vr.score_kind for vr in a_best.values()}
    if len(kinds) <= 1:
        groups = {"": list(a_best.values())}
    else:
        groups = {}
        for vr in a_best.values():
            groups.setdefault(vr.cas_type, []).append(vr)

    order = ([c.name for c in core.CAS_CONFIGS if c.name in groups]
             if set(groups) != {""} else [""])
    tables = []
    for cas in order:
        items = sorted(groups[cas], key=_score_key, reverse=True)
        label = SCORE_LABELS[items[0].score_kind]
        tables.append({
            "cas": cas,
            "score_label": label,
            "total_sites": len(items),
            "shown": min(len(items), MAX_ROWS),
            "rows": [{
                "순위": rank,
                "최적 Cas 변종": vr.cas_name,
                "PAM": vr.pam_seq,
                "가이드 서열 (RNA)": core.dna_to_rna(vr.guide_seq),
                "가닥": vr.strand,
                "거리 (bp)": vr.distance,
                label: round(vr.efficiency_pct, 2),
                "원출력": round(vr.raw_score, 4),
                # 화면에 없는 비교용 열 — 반올림 전 값으로 동점 처리를 검증한다
                "_site_key": vr.site_key,
                "_efficiency_pct_full": vr.efficiency_pct,
                "_raw_score_full": vr.raw_score,
            } for rank, vr in enumerate(items[:MAX_ROWS], start=1)],
        })
    return tables


def variant_heatmap(vrs) -> dict:
    """'변종 비교' 보기의 SpCas9 변종 히트맵 (피벗 결과)."""
    sp = [vr for vr in vrs if vr.cas_name in core.SPCAS9_VARIANTS]
    if not sp:
        return {"rows": [], "columns": [], "total": 0}

    df_long = pd.DataFrame([{
        "guide_label": f"{core.dna_to_rna(vr.guide_seq)}  dist:{vr.distance}  "
                       f"PAM:{vr.pam_seq}  strand:{vr.strand}",
        "variant": vr.cas_name,
        "score": round(vr.efficiency_pct, 2),
    } for vr in sp])
    present = df_long["variant"].unique()
    pivot = (
        df_long.pivot_table(index="guide_label", columns="variant",
                            values="score", aggfunc="max")
        .reindex(columns=[v for v in core.SPCAS9_VARIANTS if v in present])
        .assign(_max=lambda d: d.max(axis=1))
        .sort_values("_max", ascending=False)
        .drop(columns="_max")
    )
    total = len(pivot)
    pivot = pivot.head(MAX_ROWS)
    return {
        "columns": list(pivot.columns),
        "total": total,
        "shown": len(pivot),
        # 결측(그 변종이 인식하지 않는 PAM)은 화면에서 'None'으로 보인다 → JSON에서는 null
        "rows": [{"guide_label": idx,
                  **{col: (None if pd.isna(v) else float(v)) for col, v in row.items()}}
                 for idx, row in pivot.iterrows()],
    }


def detail_dataframe(all_results, sel_key: str) -> pd.DataFrame:
    """'전체 상세 보기'의 표. 필터는 기본값(= 전부 통과) 기준이다.

    기본 필터는 Cas 변종 전체 선택 · 점수 0~100% · 거리 0~15bp이므로 모든 행이 남는다.
    """
    all_vr = all_results.sorted_variant_results()
    filtered = [vr for vr in all_vr
                if 0.0 <= vr.efficiency_pct <= 100.0 and 0 <= vr.distance <= 15]
    rows = [{
        "순위": i,
        "Cas 변종": vr.cas_name,
        "PAM": vr.pam_seq,
        "가이드 서열 (RNA)": core.dna_to_rna(vr.guide_seq),
        "가닥": vr.strand,
        "거리 (bp)": vr.distance,
        "원출력": round(vr.raw_score, 4),
        EFF: round(vr.efficiency_pct, 2),
        "점수 기준": ("절단 효율" if vr.score_kind == "efficiency"
                   else "학습 분포 백분위"),
    } for i, vr in enumerate(filtered, start=1)]
    df = pd.DataFrame(rows)
    df.attrs["dropped_by_default_filter"] = len(all_vr) - len(filtered)
    df.attrs["highlighted_rows"] = [i for i, vr in enumerate(filtered, start=1)
                                    if vr.site_key == sel_key]
    return df


def capture_xai(predictors, site, head: str) -> dict:
    """한 사이트·한 변종의 XAI 3기법. 배열은 npz로, 스칼라는 JSON으로 돌려준다."""
    predictor = predictors[("A", site.cas_type)]
    handle = predictor.explain_handle(head)
    window = site.model_input_seq
    model_key = f"A/{site.cas_type}"

    print(f"    Rollout ...", flush=True)
    rollout = core.attention_rollout(handle, window)
    print(f"    IG ({IG_STEPS} steps) ...", flush=True)
    ig = core.integrated_gradients(handle, window, IG_STEPS)
    print(f"    SHAP ({SHAP_NSAMPLES}x{SHAP_N_BACKGROUND}) ...", flush=True)
    shap_res = core.kernel_shap(handle, window, SHAP_NSAMPLES, SHAP_N_BACKGROUND)

    # 두 화면이 같은 값을 보도록 사전계산 저장소에 넣는다 (KernelSHAP은 seed를 못 받는다).
    key = shap_cache.shap_cache_key(model_key, head, window,
                                    SHAP_NSAMPLES, SHAP_N_BACKGROUND)
    cache_path = shap_cache.save(key, shap_res)

    name = f"xai-{site.cas_type}-{head.replace('/', '_')}"
    npz_path = HERE / f"{name}.npz"
    np.savez(
        npz_path,
        window_seq=np.array(window),
        ig_attribution=ig.attribution,
        shap_values=shap_res.values,
        rollout_position_profile=rollout.position_profile,
        rollout_token_matrix=rollout.token_matrix,
        rollout_token_labels=np.array(rollout.token_labels),
    )
    return {
        "file": npz_path.name,
        "site_key": site.site_key,
        "cas_type": site.cas_type,
        "head": head,
        "strand": site.strand,
        "pam_seq": site.pam_seq,
        "distance": site.distance,
        "window_seq": window,
        "settings": {"ig_steps": IG_STEPS,
                     "shap_nsamples": SHAP_NSAMPLES,
                     "shap_n_background": SHAP_N_BACKGROUND},
        "ig": {"pred": ig.pred, "baseline_pred": ig.baseline_pred,
               "completeness_error": ig.completeness_error},
        "shap": {"base_value": shap_res.base_value, "pred": shap_res.pred,
                 "additivity_error": shap_res.additivity_error,
                 "cache_key": key,
                 "cache_file": str(cache_path.relative_to(ROOT)).replace("\\", "/")},
        "rollout": {"n_layers": rollout.n_layers,
                    "n_tokens": len(rollout.token_labels)},
    }


def main() -> int:
    print(f"고정 입력 ({len(FIXED_INPUT)}bp): {FIXED_INPUT}")
    print("PAM 탐색 ...", flush=True)
    scan = core.scan_all(FIXED_INPUT)
    print(f"  사이트 {len(scan.sites)}개 "
          + ", ".join(f"{c.name} {len(scan.by_cas(c.name))}" for c in core.CAS_CONFIGS))

    print("모델 로드 + 예측 ... (SpCas9은 NGG 사이트에서 지연 로딩된다)", flush=True)
    predictors = core.load_predictors()
    scan = core.run_prediction(scan, predictors)
    vrs = scan.variant_results
    print(f"  변종 예측 {len(vrs)}건")

    # 화면의 XAI 기본 선택: 점수가 가장 높은 사이트.
    site_map = {s.site_key: s for s in scan.sorted_by_score()}
    sel_key = next(iter(site_map))
    sel_site = site_map[sel_key]

    detail = detail_dataframe(scan, sel_key)
    csv_bytes = detail.to_csv(index=False).encode("utf-8-sig")
    csv_path = HERE / "crispr_prediction_results.csv"
    csv_path.write_bytes(csv_bytes)
    print(f"  CSV {csv_path.name} — {len(detail)}행 / {len(csv_bytes)} bytes")

    candidates = {
        "input_seq": FIXED_INPUT,
        "selected_site_key": sel_key,
        "sites": [{
            "site_key": s.site_key, "cas_type": s.cas_type, "strand": s.strand,
            "pam_seq": s.pam_seq, "pam_start": s.pam_start, "distance": s.distance,
            "guide_seq": s.guide_seq, "model_input_seq": s.model_input_seq,
            "raw_score": s.raw_score, "efficiency_pct": s.efficiency_pct,
            "score_kind": s.score_kind,
        } for s in scan.sorted_by_score()],
        "best_candidates": best_candidates_tables(vrs),
        "variant_heatmap": variant_heatmap(vrs),
        "detail": {
            "rows": json.loads(detail.to_json(orient="records", force_ascii=False)),
            "highlighted_rows": detail.attrs["highlighted_rows"],
            "dropped_by_default_filter": detail.attrs["dropped_by_default_filter"],
        },
    }
    (HERE / "candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  candidates.json — 사이트 {len(candidates['sites'])}개, "
          f"상세 {len(detail)}행")

    # XAI는 기본 선택 사이트 + 다른 Cas 종류의 최고 사이트 둘 다 뜬다 (두 모델 경로 모두 덮는다).
    targets = [sel_site]
    other = next((s for s in scan.sorted_by_score() if s.cas_type != sel_site.cas_type),
                 None)
    if other is not None:
        targets.append(other)

    xai = []
    for site in targets:
        predictor = predictors[("A", site.cas_type)]
        heads = predictor.explain_heads(site)
        if not heads:
            print(f"  {site.site_key}: 설명할 변종이 없어 건너뜀")
            continue
        head = heads[0]
        print(f"  XAI {site.cas_type} · {site.strand} · PAM {site.pam_seq} · head {head}")
        xai.append(capture_xai(predictors, site, head))

    meta = {
        "issue": 181,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_seq": FIXED_INPUT,
        "settings": {"ig_steps": IG_STEPS, "shap_nsamples": SHAP_NSAMPLES,
                     "shap_n_background": SHAP_N_BACKGROUND},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "csv": {"file": csv_path.name, "rows": len(detail), "bytes": len(csv_bytes),
                "encoding": "utf-8-sig"},
        "xai": xai,
    }
    try:
        import streamlit
        meta["environment"]["streamlit"] = streamlit.__version__
    except Exception:
        pass
    try:
        import torch
        meta["environment"]["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import shap
        meta["environment"]["shap"] = shap.__version__
    except Exception:
        pass

    (HERE / "capture.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("완료 — capture.json 기록")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
