"""NT-Cas12a 뷰. 이 티켓에서는 `report`.

Why the report opens with a list of questions: the two nucleases' analyses are
not symmetric. SpCas9 has a replicate noise floor and a head-order reversal
between three analysed heads; Cas12a has one regression output, so neither
question exists there. Without the questions stated up front, the shorter
report reads as a report with sections missing.

Every figure goes through `render.Figure`, which cannot be built without its
stratum, head, statistic kind and actual denominator. The numbers themselves
are read, never recomputed.

The `all` population pools the TTTV and TTTT groups, so `render.Stratum`
requires it to say what the pooling does - and what pooling does depends on
the statistic. For a rank correlation it folds in the mean efficiency
difference between the two groups (0.7007 pooled against 0.5423 / 0.2995
within); for a count it is a sum. So the note is built per displayed value
from that value's own per-group entries, never copied from another statistic's
note.
"""

from __future__ import annotations

from collections.abc import Callable

from . import render
from .reader import Cas12aReader

TITLE = "NT-Cas12a Test 본 분석 (저장된 산출물)"

QUESTIONS = (
    "Test 전수에서 예측이 라벨과 얼마나 맞는가 — PAM 집단(all/TTTV/TTTT)별로",
    "그 수치 중 어디까지가 군 내부 순위 능력이고, 어디부터가 군 사이 차이인가",
    "네 방법(예측·ISM·IG·rollout)이 실제로 몇 행에서 성공했는가",
    "attention rollout을 이 모집단에서 행별로 해석할 수 있는가",
)

# Stated, not implied: the asymmetry with the SpCas9 report is a property of
# the model's output, not an omission in this analysis.
NOT_ASKED = (
    "묻지 않은 것: replicate 노이즈 바닥과 head 간 순서 역전. 이 모델은 "
    "출력이 하나뿐이어서 두 질문이 성립하지 않는다 (SpCas9 report에만 있는 절이다)."
)

# raw is the view; clipped is the second line of the same table.
CLIPPINGS = (("raw", "기본"), ("clipped", "둘째 줄"))

PERFORMANCE_STATISTICS = (
    ("Spearman", "spearman", "spearman_rank_correlation"),
    ("MAE", "mae", "mean_absolute_error"),
)

COVERAGE_METHODS = (
    ("prediction", "예측"),
    ("ism", "ISM"),
    ("ig", "IG"),
    ("rollout", "rollout"),
)


def _count(ids: object, *, what: str) -> int:
    """How many rows a coverage entry holds.

    `summary.json` records each method's coverage as explicit row-ID lists
    (`available`, `failed`, `not_computed`, `requested`), so the count is the
    list's length. An entry that is not a list is refused rather than coerced:
    a bare number here would mean the artifact's shape changed, and guessing
    which it was is how a count turns into the wrong denominator.
    """
    if not isinstance(ids, list):
        raise render.RenderRefused(
            f"coverage의 {what}이(가) 행 ID 목록이 아니다: {type(ids).__name__}"
        )
    return len(ids)


def _groups(reader: Cas12aReader) -> tuple[str, ...]:
    """The PAM groups that are not pooled: the ones `all` is made of."""
    return tuple(group for group in reader.strata if not reader.pools_pam_groups(group))


def _stratum(
    reader: Cas12aReader,
    name: str,
    *,
    per_group: Callable[[str], str],
    effect: str,
) -> render.Stratum:
    """A PAM population; where it pools, what the pooling does to *this* value.

    `per_group` renders the same statistic for one group, so the pooled figure
    always carries the within-group numbers it is made of. `effect` says what
    pooling does to that kind of statistic - a rank correlation absorbs the
    between-group efficiency difference, a count is a sum - because the two
    mislead in different ways and one note cannot cover both.
    """
    if not reader.pools_pam_groups(name):
        return render.Stratum(name)
    within = " / ".join(f"{group} {per_group(group)}" for group in _groups(reader))
    return render.Stratum(
        name, pooled=True, pooling_note=f"{effect}, 군별 값은 {within}"
    )


def _performance(reader: Cas12aReader) -> render.Section:
    figures: list[render.Figure] = []
    for stratum in reader.strata:
        for label, key, statistic in PERFORMANCE_STATISTICS:
            for clipping, standing in CLIPPINGS:
                entry = reader.metrics_for(stratum, clipping)
                declared = _stratum(
                    reader,
                    stratum,
                    per_group=lambda group, key=key, clipping=clipping: (
                        render.format_value(reader.metrics_for(group, clipping)[key])
                    ),
                    effect=(
                        "TTTV·TTTT의 평균 효율 차이를 포함한다"
                        if key == "spearman"
                        else "TTTV·TTTT를 합친 행에서 계산했다"
                    ),
                )
                figures.append(
                    render.Figure(
                        name=f"{stratum} {label} ({clipping}, {standing})",
                        value=entry[key],
                        stratum=declared,
                        head=reader.head,
                        statistic=f"{statistic} ({clipping})",
                        denominator=render.Denominator(
                            rows=int(entry["count"]), unit="행"
                        ),
                        scale="fraction (native raw output)",
                    )
                )
    return render.Section(
        title="예측 성능 (PAM 집단 3개)",
        notes=(
            "세 집단을 모두 적는다. raw가 기본이고 clipped는 같은 표의 둘째 줄이다.",
            "TTTV/TTTT의 분모는 관측된 각 집단의 행 수다.",
        ),
        figures=tuple(figures),
    )


def _population(reader: Cas12aReader) -> render.Section:
    summary = reader.summary
    coverage = summary.get("coverage") or {}

    # The population itself pools, so say so once with the group sizes. The
    # coverage lists are not split by PAM group in `summary.json`, so nothing
    # here claims a per-group coverage number: these are the group sizes.
    pooled = _stratum(
        reader,
        "all",
        per_group=lambda group: f"{int(reader.metrics_for(group)['count']):,}행",
        effect="TTTV·TTTT 두 군을 합친 모집단이다",
    )

    figures = [
        render.Figure(
            name="분석 완료",
            value=bool(summary["complete"]),
            stratum=pooled,
            head=reader.head,
            statistic="summary.complete (저장된 상태)",
            denominator=render.Denominator(
                rows=int(summary["population_count"]), unit="행"
            ),
        )
    ]
    for method, label in COVERAGE_METHODS:
        entry = coverage.get(method)
        if not entry:
            continue
        available = _count(entry["available"], what=f"{method}.available")
        requested = _count(entry["requested"], what=f"{method}.requested")
        failed = _count(entry["failed"], what=f"{method}.failed")
        not_computed = _count(entry["not_computed"], what=f"{method}.not_computed")
        figures.append(
            render.Figure(
                name=f"{label} 성공 coverage (요구 {requested:,}행 중)",
                value=available,
                stratum=pooled,
                head=reader.head,
                statistic="실제 성공 coverage (요구 대비)",
                denominator=render.Denominator(rows=requested, unit="행"),
                note=f"실패 {failed:,}행, 미계산 {not_computed:,}행",
            )
        )
    return render.Section(
        title="모집단과 coverage",
        notes=(
            (
                f"partition={summary.get('partition', '기록 없음')}, "
                f"IG 통계 모집단={summary.get('ig_mode', '기록 없음')}"
            ),
        ),
        figures=tuple(figures),
    )


def _rollout(reader: Cas12aReader) -> render.Section:
    figures: list[render.Figure] = []
    for stratum in reader.strata:
        entry = reader.rollout_degeneracy(stratum)
        accepted = int(entry["n_accepted"])
        degenerate = int(entry["near_identical_query_count"])
        figures.append(
            render.Figure(
                name=(
                    f"{stratum} DNA query가 거의 동일한 행 "
                    f"(rollout 통과 {accepted:,}행 중)"
                ),
                value=degenerate,
                stratum=_stratum(
                    reader,
                    stratum,
                    per_group=lambda group: (
                        render.format_value(
                            int(
                                reader.rollout_degeneracy(group)[
                                    "near_identical_query_count"
                                ]
                            )
                        )
                        + "행"
                    ),
                    effect="TTTV·TTTT 두 군의 합이다",
                ),
                head=reader.head,
                statistic="near_identical_query_count (rollout 진단)",
                denominator=render.Denominator(rows=accepted, unit="행"),
            )
        )
    return render.Section(
        title="rollout 해석 제한",
        notes=(
            (
                "이 수치는 저장 실패도 수치 실패도 아니다: rollout 품질은 전 행 "
                "통과했고 값은 그대로 저장돼 있다. 해석 제한이다 — DNA query가 "
                "거의 동일해 행 사이의 rollout 차이를 행의 성질로 읽을 수 없다."
            ),
            "rollout은 encoder token attention flow이며 출력 attribution이 아니다.",
        ),
        figures=tuple(figures),
    )


def report(reader: Cas12aReader) -> str:
    """The Cas12a `report` view."""
    return render.render(
        [_population(reader), _performance(reader), _rollout(reader)],
        claim_limit=reader.claim_limit,
        title=TITLE,
        questions=(*QUESTIONS, NOT_ASKED),
        pointers=(render.CROSS_POINTER,),
    )
