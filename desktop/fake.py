"""경계의 가짜 구현 (#178). 모델 가중치 없이, 패키징 없이 돈다.

**사이트 탐색은 진짜다.** `pam_scanner`를 그대로 import해 쓴다 (#177: "계산·차트
코드(`pam_scanner`, `predict`, `xai/explain`, `xai/charts.py`)는 그대로 재사용한다").
그 모듈은 `math`와 `dataclasses`만 import하므로 가중치도 torch도 끌고 오지 않는다.
탐색 규칙을 베껴 두면 `pam_scanner`가 바뀔 때 조용히 어긋나고, 팀원들은 존재하지
않는 `site_key`에 화면을 묶게 된다 — #178이 막으려는 바로 그 충돌이다.

**가짜인 것은 점수와 기여도뿐이다.** 서열에서 결정적으로 만든 값이라 같은 입력이면
늘 같은 그림이 나오지만, **설명으로 읽으면 안 된다.**

표의 열 이름은 Streamlit 화면(`app.py`)을 따라간다. 여기서 어긋나면 #177 게이트 2의
동등성 점검(고정 입력 수치·CSV diff 비교)이 깨진다.
"""

from __future__ import annotations

import hashlib
import html
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# 공용 코어는 형제 체크아웃에 있다 (`packaging/README.md`: branch `dashboard_core`).
_CORE_PATH = Path(__file__).resolve().parents[2] / "xai_dashboard"
if str(_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(_CORE_PATH))

try:
    from pam_scanner import (
        SPCAS9_VARIANTS,
        CandidateSite,
        VariantResult,
        dna_to_rna,
        scan_all,
        variant_compatible,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - 체크아웃이 없을 때만
    raise ModuleNotFoundError(
        f"`pam_scanner`를 찾지 못했습니다. 형제 체크아웃 `xai_dashboard`가 "
        f"{_CORE_PATH}에 있어야 합니다 (branch `dashboard_core`)."
    ) from exc

BATCH = 8  # predict.NT_BATCH_SIZE
EFF = "예측 점수 (%)"  # app.py:23. 의미는 '점수 종류'가 정한다.

#: 전체 상세 보기의 열. `app.py:353-364`의 `df_detail`과 같다. CSV도 이것이다.
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

#: Best Candidates 보기의 열. `app.py:221-229`.
BEST_COLUMNS = [
    "순위",
    "최적 Cas 변종",
    "PAM",
    "가이드 서열 (RNA)",
    "가닥",
    "거리 (bp)",
    EFF,
    "원출력",
]

_SCORE_KIND_LABEL = {"efficiency": "절단 효율", "percentile": "학습 분포 백분위"}


# ════════════════════════════════════════════════
# 가짜 예측 행
# ════════════════════════════════════════════════


def _fake_score(*parts: str) -> float:
    """서열에서 만든 0~1 값. 난수가 아니라 해시라서 같은 입력이면 늘 같다."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def heads_of(site: CandidateSite) -> list[str]:
    """이 사이트에서 설명·예측 대상이 되는 헤드. Cas12a는 변종이 없어 하나뿐이다."""
    if site.cas_type == "Cas12a":
        return ["Cas12a"]
    return [v for v in SPCAS9_VARIANTS if variant_compatible(v, site.pam_seq)]


def rows_of(site: CandidateSite) -> list[VariantResult]:
    """예측 행. 변종이 없는 Cas12a도 행을 하나 만든다 (CONTEXT.md의 '예측 행')."""
    rows = []
    for head in heads_of(site):
        unit = _fake_score(site.model_input_seq, head)
        # Cas12a는 0~1 비율, SpCas9은 z-score라 원출력의 범위가 다르다.
        raw = unit if site.cas_type == "Cas12a" else (unit * 6.0) - 3.0
        rows.append(
            VariantResult(
                cas_name=head,
                pam_seq=site.pam_seq,
                guide_seq=site.guide_seq,
                distance=site.distance,
                strand=site.strand,
                raw_score=raw,
                final_score=0.0,  # 거리 보정 점수는 화면에 쓰지 않는다
                efficiency_pct=round(unit * 100, 2),
                score_kind="efficiency" if site.cas_type == "Cas12a" else "percentile",
                site_key=site.site_key,
                cas_type=site.cas_type,
                model="A",
                window_seq=site.model_input_seq,
            )
        )
    return rows


def _guide_label(row: VariantResult) -> str:
    """변종 비교 표의 행 이름. `app.py:261-263`과 같다."""
    return (
        f"{dna_to_rna(row.guide_seq)}  dist:{row.distance}  "
        f"PAM:{row.pam_seq}  strand:{row.strand}"
    )


# ════════════════════════════════════════════════
# 조각 만들기
# ════════════════════════════════════════════════


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[Any]], attrs: list[dict]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for cells, attr in zip(rows, attrs):
        marks = "".join(f' {k}="{_esc(v)}"' for k, v in attr.items())
        body.append(
            f"<tr{marks}>" + "".join(f"<td>{_esc(c)}</td>" for c in cells) + "</tr>"
        )
    return (
        f'<table class="grid"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _detail_cells(row: VariantResult, rank: int) -> list[Any]:
    return [
        rank,
        row.cas_name,
        row.pam_seq,
        dna_to_rna(row.guide_seq),
        row.strand,
        row.distance,
        round(row.raw_score, 4),
        round(row.efficiency_pct, 2),
        _SCORE_KIND_LABEL[row.score_kind],
    ]


def _best_cells(row: VariantResult, rank: int) -> list[Any]:
    return [
        rank,
        row.cas_name,
        row.pam_seq,
        dna_to_rna(row.guide_seq),
        row.strand,
        row.distance,
        round(row.efficiency_pct, 2),
        round(row.raw_score, 4),
    ]


def _marks(row: VariantResult) -> dict:
    return {"data-site-key": row.site_key, "data-head": row.cas_name}


def _bars_svg(values: list[float], caption: str) -> str:
    """막대 그림 하나. 조각에 바깥 주소가 없어야 해서 인라인 SVG로 그린다.

    `xmlns`를 넣지 않는 것은 실수가 아니다. HTML 안의 인라인 SVG에는 필요 없고,
    그 값(`http://www.w3.org/2000/svg`) 자체가 바깥 주소다.
    """
    width, height, pad = 640, 140, 10
    span = max(1e-9, max(abs(v) for v in values))
    step = (width - 2 * pad) / max(1, len(values))
    mid = height / 2
    bars = []
    for i, value in enumerate(values):
        tall = (value / span) * (mid - pad)
        bars.append(
            f'<rect x="{pad + i * step:.2f}" y="{mid - tall if tall >= 0 else mid:.2f}" '
            f'width="{max(1.0, step - 1):.2f}" height="{abs(tall):.2f}" '
            f'fill="{"#2b6cb0" if value >= 0 else "#c05621"}"></rect>'
        )
    return (
        f'<figure class="xai"><svg viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_esc(caption)}">'
        f'<line x1="{pad}" y1="{mid}" x2="{width - pad}" y2="{mid}" '
        f'stroke="#718096"></line>{"".join(bars)}</svg>'
        f"<figcaption>{_esc(caption)}</figcaption></figure>"
    )


# ── XAI 기법별 계획 (몇 단계로 보고할지 + 설명 문구) ──────────


def _ig_plan(params: dict[str, Any], head: str) -> tuple[int, str]:
    steps = int(params.get("steps", 32))
    return max(1, math.ceil(steps / 4)), f"IG · {head} · {steps}단계 (가짜 값)"


def _shap_plan(params: dict[str, Any], head: str) -> tuple[int, str]:
    nsamples = int(params.get("nsamples", 64))
    background = int(params.get("n_background", 4))
    return (
        max(1, math.ceil(nsamples * background / BATCH)),
        f"SHAP · {head} · 샘플 {nsamples} × 배경 {background} (가짜 값)",
    )


def _rollout_plan(params: dict[str, Any], head: str) -> tuple[int, str]:
    # 순전파 1회라 중간 취소하지 않는다. 그래서 보고할 단계가 없다 (#177).
    return 0, f"Attention Rollout · {head} (가짜 값)"


_XAI_PLANS: dict[str, Callable[[dict[str, Any], str], tuple[int, str]]] = {
    "ig": _ig_plan,
    "shap": _shap_plan,
    "rollout": _rollout_plan,
}


# ════════════════════════════════════════════════
# 가짜 Renderer
# ════════════════════════════════════════════════


class FakeRenderer:
    """경계의 `Renderer` 프로토콜을 가중치 없이 만족시키는 구현.

    `step_delay`는 한 단계에 쉬는 시간이다. 진행률과 취소가 눈에 보이게 하려고
    둔 것이고, 시험에서는 0에 가깝게 둔다.
    """

    def __init__(self, step_delay: float = 0.05):
        self.step_delay = step_delay

    def _report_step(self, on_batch, done: int, total: int) -> None:
        """진행률 한 단계. **지금 요청이 아니면 여기서 `Cancelled`가 난다** —
        가짜 구현에서 취소가 실제로 걸리는 지점은 여기뿐이다."""
        on_batch(done, total)
        if self.step_delay:
            time.sleep(self.step_delay)

    def _rows_with_progress(self, dna: str, on_batch) -> list[VariantResult]:
        """사이트마다 한 단계씩 보고하며 예측 행을 만든다. 예측과 CSV가 함께 쓴다."""
        sites = scan_all(dna).sites
        rows: list[VariantResult] = []
        for index, site in enumerate(sites, start=1):
            rows.extend(rows_of(site))
            self._report_step(on_batch, index, len(sites))
        rows.sort(key=lambda r: (r.efficiency_pct, r.raw_score), reverse=True)
        return rows

    # ── 경계가 부르는 네 가지 ──────────────────────

    def sites_fragment(self, dna: str, on_batch) -> str:
        sites = scan_all(dna).sites
        table = _table(
            ["Cas", "가닥", "PAM 위치", "PAM", "가이드 서열 (RNA)", "거리 (bp)"],
            [
                [
                    s.cas_type,
                    s.strand,
                    s.pam_start,
                    s.pam_seq,
                    dna_to_rna(s.guide_seq),
                    s.distance,
                ]
                for s in sites
            ],
            [{"data-site-key": s.site_key} for s in sites],
        )
        return (
            f'<section id="sites"><h2>후보 사이트 {len(sites)}개</h2>{table}</section>'
        )

    def prediction_fragments(self, dna: str, on_batch) -> dict[str, str]:
        rows = self._rows_with_progress(dna, on_batch)
        best = _table(
            BEST_COLUMNS,
            [_best_cells(r, i) for i, r in enumerate(rows[:20], start=1)],
            [_marks(r) for r in rows[:20]],
        )
        detail = _table(
            DETAIL_COLUMNS,
            [_detail_cells(r, i) for i, r in enumerate(rows, start=1)],
            [_marks(r) for r in rows],
        )
        return {
            "best": f'<section id="best"><h2>Best Candidates</h2>{best}</section>',
            "variants": (
                '<section id="variants"><h2>변종 비교</h2>'
                f"{self._variant_pivot(rows)}</section>"
            ),
            "detail": f'<section id="detail"><h2>전체 상세 보기</h2>{detail}</section>',
        }

    def _variant_pivot(self, rows: list[VariantResult]) -> str:
        """변종 비교는 평평한 목록이 아니라 피벗이다 (`app.py:260-270`).

        행은 가이드, 열은 그 입력에 실제로 나타난 변종(`SPCAS9_VARIANTS` 순서),
        칸은 표시 점수. Cas12a는 변종이 없어 이 표에 들어가지 않는다.
        """
        sp = [r for r in rows if r.cas_type == "SpCas9"]
        if not sp:
            return (
                '<p class="empty">SpCas9 사이트가 없어 비교할 변종이 없습니다. '
                "Cas12a는 변종이 없습니다.</p>"
            )
        present = [v for v in SPCAS9_VARIANTS if any(r.cas_name == v for r in sp)]
        labels: dict[str, VariantResult] = {}
        for row in sp:
            labels.setdefault(_guide_label(row), row)
        cells, attrs = [], []
        for label, sample in labels.items():
            scores = {
                r.cas_name: r.efficiency_pct for r in sp if _guide_label(r) == label
            }
            cells.append(
                [label] + [round(scores[v], 2) if v in scores else "-" for v in present]
            )
            attrs.append({"data-site-key": sample.site_key})
        return _table(["가이드"] + present, cells, attrs)

    def explanation_fragment(
        self,
        dna: str,
        site_key: str,
        head: str,
        method: str,
        params: dict[str, Any],
        on_batch,
    ) -> str:
        site = next((s for s in scan_all(dna).sites if s.site_key == site_key), None)
        if site is None:
            raise LookupError(f"그런 사이트가 없습니다: {site_key}")
        if head not in heads_of(site):
            raise LookupError(f"이 사이트에 없는 헤드입니다: {head}")
        plan = _XAI_PLANS.get(method)
        if plan is None:
            raise ValueError(f"모르는 XAI 기법: {method}")

        total, caption = plan(params, head)
        for index in range(1, total + 1):
            self._report_step(on_batch, index, total)
        if total == 0 and self.step_delay:
            time.sleep(self.step_delay)  # 취소 지점이 없는 짧은 계산

        window = site.model_input_seq
        values = [
            _fake_score(window, head, method, str(i)) - 0.5 for i in range(len(window))
        ]
        return _bars_svg(values, caption)

    def csv_table(self, dna: str, view: str, on_batch) -> list[list[Any]]:
        """CSV 한 장. Streamlit은 전체 상세만 내보내므로 열도 그것과 같다."""
        rows = self._rows_with_progress(dna, on_batch)
        if view == "variants":
            rows = [r for r in rows if r.cas_type == "SpCas9"]
        elif view == "best":
            rows = rows[:20]
        return [DETAIL_COLUMNS] + [
            _detail_cells(r, i) for i, r in enumerate(rows, start=1)
        ]
