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


# --------------------------------------------------------------------------- #
# cases and explain
# --------------------------------------------------------------------------- #

CASES_TITLE = "NT-Cas12a 기술 사례 (원 자료의 source_row + role 그대로)"
CASES_QUESTIONS = (
    "어떤 행을 기술 사례로 골랐고 어떤 역할로 골랐는가 — 역할별 모집단과 함께",
    "그 사례에서 네 방법이 각각 성공했는가",
)
CASES_NOTES = (
    (
        "식별자는 원 자료 그대로 `source_row`와 `role`이다. 데모가 ID를 새로 "
        "만들지 않는다 — `explain <source_row>:<role>`로 조회한다."
    ),
    (
        "기술 사례는 무작위 상세 부분집합이 아니다. 오차로 골라낸 행이므로 기술적 "
        "해석에만 쓰고 모집단 요약으로 읽지 않는다."
    ),
    (
        "역할마다 자기 후보 모집단이 있다. 선정 개수를 후보 수와 함께 적는다 — "
        "out_of_range 5건과 high_error 5건은 같은 크기의 주장이 아니다."
    ),
)
EXPLAIN_QUESTIONS = (
    "이 행에 어떤 방법이 있고 어떤 방법이 없는가 — 없는 것은 기록된 사유와 함께",
    "있는 방법이 이 행에서 무엇을 가리키는가",
)
ROLLOUT_ROW_LIMIT = (
    "이 행의 rollout은 저장돼 있으나 행의 성질로 읽을 수 없다: DNA query가 거의 "
    "동일해 행 사이 차이가 생기지 않는다. 저장·수치 실패가 아니라 해석 제한이다."
)


def _case_line(reader: Cas12aReader, case: dict) -> render.Figure:
    """One listed case: its row, role, PAM group, and which methods succeeded.

    No label, prediction or error here, although the case record holds them.
    They are this row's own statistics with their own kind and scale, and on
    this line they would ride under the selection figure's stratum ("this
    role's pool") and statistic ("case selection"), which are not theirs.
    `explain` gives each of them its own four fields.
    """
    identifier = f"{case['source_row']}:{case['role']}"
    pool = reader.case_pool(str(case["role"]))
    note = f"PAM 집단={case['pam_stratum']} / " + " / ".join(
        entry.text() for entry in reader.case_methods(identifier)
    )
    return render.Figure(
        name=f"{identifier} (이 역할의 순위 {case['rank_in_category']})",
        value=f"source_row {case['source_row']}",
        # A case's role is its own population, never pooled with another
        # role's or with the Test population.
        stratum=f"{case['role']} 후보 모집단",
        head=reader.head,
        statistic=f"사례 선정 ({case['selection_reason']})",
        denominator=render.Denominator(
            rows=pool["available"],
            unit="행",
            note=(
                f"이 역할이 실제로 뽑은 후보 행 (선정 {pool['selected']:,}건, "
                f"기준을 만족한 행 {pool['eligible']:,}행 중 앞 범주가 이미 가져간 "
                "행을 제외한 수)"
            ),
        ),
        note=note,
    )


def cases(reader: Cas12aReader) -> str:
    """The Cas12a `cases` view: every described case under its row and role."""
    records = [reader.case(identifier) for identifier in reader.case_ids()]
    section = render.Section(
        title="사례 목록",
        notes=(
            *CASES_NOTES,
            f"사례 {len(records):,}건. 행별 값은 `explain <source_row>:<role>`로 본다.",
        ),
        figures=tuple(_case_line(reader, case) for case in records),
    )
    return render.render(
        [section],
        claim_limit=reader.claim_limit,
        title=CASES_TITLE,
        questions=CASES_QUESTIONS,
    )


def _explain_row(reader: Cas12aReader, case: dict) -> render.Section:
    row_id = int(case["source_row"])
    stored = reader.prediction(row_id)
    group = str(case["pam_stratum"])
    figures = [
        render.Figure(
            name="라벨 / raw 예측 / clipped 예측",
            value=(
                f"{stored['label']:.4g} / {stored['raw_prediction']:.4g} / "
                f"{stored['clipped_prediction']:.4g}"
            ),
            stratum=group,
            head=reader.head,
            statistic="저장된 라벨과 예측 (raw 기본, clipped 병기)",
            denominator=render.single_row(),
            scale="fraction (native raw output)",
        ),
        render.Figure(
            name="raw 절대 오차",
            value=case["raw_absolute_error"],
            stratum=group,
            head=reader.head,
            statistic="raw absolute error (선정에 쓴 값)",
            denominator=render.single_row(),
            scale="fraction",
            note=f"선정 사유: {case['selection_reason']}",
        ),
    ]
    return render.Section(
        title=f"source_row {row_id} ({case['role']}) — 행과 선정",
        notes=(
            (
                "이 행이 속한 PAM 집단만 적는다. 집단을 합친 수치는 이 뷰에 없다 — "
                "합치면 군 사이 평균 효율 차이가 섞인다."
            ),
        ),
        figures=tuple(figures),
    )


def _explain_methods(reader: Cas12aReader, case: dict) -> render.Section:
    """Each method's recorded status for this row, a failure quoted as a failure.

    The statuses are the annotation's own, so a method that ran and failed
    shows its recorded `failure` text instead of leaving a blank that would
    read as the demo having lost the number.
    """
    identifier = f"{case['source_row']}:{case['role']}"
    notes = [entry.text() for entry in reader.case_methods(identifier)]
    return render.Section(title="이 행의 방법별 유무", notes=tuple(notes))


def _explain_values(reader: Cas12aReader, case: dict) -> render.Section:
    """The recorded per-row figures, each from the method that actually ran."""
    identifier = f"{case['source_row']}:{case['role']}"
    annotation = reader.case_annotation(identifier)
    methods = annotation.get("methods") or {}
    group = str(case["pam_stratum"])
    figures: list[render.Figure] = []
    notes: list[str] = []

    ism = methods.get("ism") or {}
    if ism.get("status") == "available":
        summary = ism["summary"]
        figures.append(
            render.Figure(
                name="최대 민감 위치",
                value=summary["top_position"],
                stratum=group,
                head=reader.head,
                statistic="ISM 위치 프로파일의 최대 위치 (WT 슬롯 제외)",
                denominator=render.single_row(),
                note="PAM은 5′ 쪽 4-7이다",
            )
        )
        figures.append(
            render.Figure(
                name="토큰별 평균 민감도",
                value=summary["token_mean_sensitivity"],
                stratum=group,
                head=reader.head,
                statistic="DNA 블록별 평균 |raw delta|",
                denominator=render.single_row(),
                scale="fraction",
            )
        )
        if summary.get("pam_share") is not None:
            figures.append(
                render.Figure(
                    name="PAM 구간이 차지하는 민감도 몫",
                    value=summary["pam_share"],
                    stratum=group,
                    head=reader.head,
                    statistic="PAM(4-7) 평균 ÷ 창 전체 평균",
                    denominator=render.single_row(),
                    scale="무차원 비율 (창의 몫이며 염기별 PAM attribution이 아니다)",
                )
            )
    else:
        notes.append(f"ISM 값 없음 — {ism.get('failure') or '사유 기록 없음'}")

    ig = methods.get("ig") or {}
    if ig.get("status") == "available":
        summary = ig["summary"]
        figures.append(
            render.Figure(
                name="DNA 블록별 signed mass",
                value=summary["dna_signed_mass"],
                stratum=group,
                head=reader.head,
                statistic="IG signed mass (DNA 블록)",
                denominator=render.single_row(),
                scale="모델 출력 단위 (fraction 척도의 출력 차이에 대한 분해)",
                note=(
                    "completeness 통과="
                    f"{render.format_value(summary['quality']['completeness_pass'])}, "
                    f"잔차={summary['quality']['absolute_residual']:.3g} "
                    f"(한계 {summary['quality']['completeness_limit']:.3g}), "
                    f"steps={summary['steps']}"
                ),
            )
        )
    else:
        notes.append(
            f"IG 값 없음 — status={ig.get('status', '기록 없음')}: "
            f"{ig.get('failure') or '사유 기록 없음'}"
        )

    rollout = methods.get("rollout") or {}
    if rollout.get("status") == "available":
        summary = rollout["summary"]
        diagnostic = summary.get("sequence_query_diagnostic") or {}
        figures.append(
            render.Figure(
                name="rollout query 진단",
                value=str(diagnostic.get("status", "기록 없음")),
                stratum=group,
                head=reader.head,
                statistic="sequence_query_diagnostic (rollout 해석 가능성)",
                denominator=render.single_row(),
                note=(
                    "거의 동일한 query 비율 "
                    f"{render.format_value(diagnostic.get('near_identical_fraction'))}"
                ),
            )
        )
        notes.extend(
            render.quoted(str(summary.get("interpretation")), ROLLOUT_ROW_LIMIT)
        )
    else:
        notes.append(
            f"rollout 값 없음 — status={rollout.get('status', '기록 없음')}: "
            f"{rollout.get('failure') or '사유 기록 없음'}"
        )

    return render.Section(title="방법별 값", notes=tuple(notes), figures=tuple(figures))


def explain(reader: Cas12aReader, identifier: str) -> str:
    """The Cas12a `explain <source_row>[:<role>]` view."""
    case = reader.case(identifier)
    return render.render(
        [
            _explain_row(reader, case),
            _explain_methods(reader, case),
            _explain_values(reader, case),
        ],
        claim_limit=reader.claim_limit,
        title=(f"source_row {case['source_row']} ({case['role']}) — 방법별 유무와 값"),
        questions=EXPLAIN_QUESTIONS,
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


# --------------------------------------------------------------------------- #
# agreement
# --------------------------------------------------------------------------- #

AGREEMENT_TITLE = "NT-Cas12a IG-ISM 일치도와 ISM 위치 프로파일 (PAM 집단별, 판정 없음)"
AGREEMENT_QUESTIONS = (
    "IG와 ISM이 t1–t5 토큰의 순위를 비슷하게 매기는가 — PAM 집단별 측정값으로",
    "집단 평균 프로파일끼리는 순위가 얼마나 비슷한가",
    "ISM 위치 프로파일에서 가장 민감한 위치가 PAM 안에 있는가 — 집단별로",
)
AGREEMENT_NOT_ASKED = (
    "묻지 않은 것: head별 일치도. 이 모델은 출력이 하나뿐이다. protocol의 primary "
    "외의 변형(9-token, PAM-breaking 제외, mass)은 agreement.json에 기록돼 있으나 "
    "이 뷰에 없다."
)
NO_VERDICT = render.NO_VERDICT

#: The TTTV PAM's four bases, 5′ of the protospacer. The analysis' own PAM
#: share is computed over this span (`nt_main_report`: `sensitivity[4:8]`).
PAM = (4, 7)


def _agreement_figures(reader: Cas12aReader) -> render.Section:
    figures: list[render.Figure] = []
    for stratum in reader.strata:
        entry = reader.agreement_primary(stratum)
        median = entry["median"]
        figures.append(
            render.Figure(
                name=f"{stratum} IG-ISM 순위 상관 (행별 median)",
                value=median
                if median is not None
                else f"정의되지 않음 — {entry['reason']}",
                stratum=_stratum(
                    reader,
                    stratum,
                    per_group=lambda group: render.format_value(
                        reader.agreement_primary(group)["median"]
                    ),
                    effect="TTTV·TTTT 두 군의 행을 합친 행별 분포의 median이다",
                ),
                head=reader.head,
                statistic=f"spearman_rank_correlation ({entry['primary']}, 행별 분포의 median)",
                denominator=render.Denominator(
                    rows=entry["n_valid"],
                    unit="행",
                    note=f"유효 {entry['n_valid']:,}행 / 전체 {entry['n_total']:,}행",
                ),
                scale="무차원 순위 상관",
            )
        )
        rho = entry["mean_profile_rho"]
        figures.append(
            render.Figure(
                name=f"{stratum} 집단 평균 프로파일끼리의 순위 상관",
                value=rho
                if rho is not None
                else f"정의되지 않음 — {entry['mean_profile_reason']}",
                stratum=_stratum(
                    reader,
                    stratum,
                    per_group=lambda group: render.format_value(
                        reader.agreement_primary(group)["mean_profile_rho"]
                    ),
                    effect="TTTV·TTTT 두 군의 행을 합친 평균 프로파일이다",
                ),
                head=reader.head,
                statistic=f"spearman_rank_correlation ({entry['primary']}, 집단 평균 ISM·IG 토큰 프로파일 사이)",
                denominator=render.Denominator(
                    rows=entry["mean_profile_tokens"],
                    unit="토큰",
                    note=(
                        f"{entry['mean_profile_rows']:,}행의 평균 프로파일 사이 — 행 수가 아니라 "
                        "토큰 수가 분모다"
                    ),
                ),
                scale="무차원 순위 상관",
            )
        )
    section = reader.agreement_section
    return render.Section(
        title="IG-ISM 일치도 (PAM 집단 3개, 판정 없음)",
        notes=(
            NO_VERDICT,
            *(
                f"원문(report.ko.md '{section['title']}' 절): {line}"
                for line in section["prose"]
            ),
            (
                "Cas12a 산출물에는 `mismatch_note` 키가 없다. 낮은 일치도의 뜻은 위 "
                "절이 그 자리를 대신해 적고 있어 그대로 인용한다."
            ),
        ),
        figures=tuple(figures),
    )


def _profile(reader: Cas12aReader, stratum: str) -> render.Section:
    entry = reader.position_profile(stratum)
    profile = [float(value) for value in entry["mean_abs_raw_delta"]]

    peak = render.profile_peak(profile)
    inside = PAM[0] <= peak <= PAM[1]
    figure = render.Figure(
        name="최대 민감 위치",
        value=f"{peak} ({'PAM 안' if inside else 'PAM 밖'}, PAM {PAM[0]}-{PAM[1]}, 5′)",
        stratum=_stratum(
            reader,
            stratum,
            per_group=lambda group: str(
                render.profile_peak(
                    reader.position_profile(group)["mean_abs_raw_delta"]
                )
            ),
            effect="TTTV·TTTT 두 군의 행을 합친 평균 프로파일이다",
        ),
        head=reader.head,
        statistic=(
            "ISM 위치 프로파일의 최댓값 위치 (위치별 평균 |raw delta|, WT 슬롯 제외 "
            "유효 대체 염기 평균, 집단 평균)"
        ),
        denominator=render.Denominator(
            rows=entry["n_available"],
            unit="행",
            note=f"요구 {entry['requested']:,}행 중 ISM 성공 행",
        ),
        scale="fraction (native raw output)",
    )
    return render.Section(
        title=f"ISM 위치 프로파일 — {stratum}",
        figures=(figure,),
        chart=tuple(render.profile_bars(profile, pam=PAM)),
    )


def _svgs(reader: Cas12aReader) -> render.Section:
    """The bundle's SVGs, by path. Not opened, not rendered."""
    lines = [
        f"{label}: {path}" + ("" if path.exists() else " (파일 없음)")
        for label, path in reader.agreement_section["images"]
    ]
    return render.Section(
        title="저장된 그림 (경로만, 렌더하지 않는다)",
        notes=(
            *lines,
            (
                "SpCas9에는 대응하는 SVG가 없어, 렌더하면 두 뉴클레아제의 표시가 "
                "비대칭이 된다. 그래서 두 뷰 모두 위 ASCII 막대로 보여준다."
            ),
        ),
    )


def agreement(reader: Cas12aReader, *arguments: str) -> str:
    """The Cas12a `agreement` view. There is no head to choose."""
    if arguments:
        raise KeyError(
            f"Cas12a는 출력이 하나뿐이라 head를 고를 수 없다: {' '.join(arguments)}"
        )
    return render.render(
        [
            _agreement_figures(reader),
            *(_profile(reader, stratum) for stratum in reader.strata),
            _svgs(reader),
        ],
        claim_limit=reader.claim_limit,
        title=AGREEMENT_TITLE,
        questions=(*AGREEMENT_QUESTIONS, AGREEMENT_NOT_ASKED),
    )


# --------------------------------------------------------------------------- #
# cross
# --------------------------------------------------------------------------- #

#: The population the cross finding is stated on: Test in full.
CROSS_STRATUM = "all"


def _top_two(profile: list[float]) -> tuple[int, int]:
    """The largest and second-largest positions, by `render.profile_peak`'s rule."""
    first = render.profile_peak(profile)
    rest = [float("nan") if i == first else v for i, v in enumerate(profile)]
    return first, render.profile_peak(rest)


def _in_pam(position: int) -> bool:
    return PAM[0] <= position <= PAM[1]


def _cross_ranked(
    reader: Cas12aReader,
) -> tuple[dict, list[float], tuple[int, int]]:
    """The profile entry, its values and its top two positions, read once.

    Refused if the most sensitive position is outside the PAM: the finding
    would not hold for these artifacts, and stating it anyway is the kind of
    claim this view exists to keep honest. The clause and the evidence section
    both come through here, so neither can be built from a profile the other
    would have refused.
    """
    entry = reader.position_profile(CROSS_STRATUM)
    profile = [float(value) for value in entry["mean_abs_raw_delta"]]
    ranked = _top_two(profile)
    if not _in_pam(ranked[0]):
        raise render.RenderRefused(
            f"NT-Cas12a {CROSS_STRATUM}의 최대 민감 위치 {ranked[0]}이 PAM "
            f"{PAM[0]}-{PAM[1]} 밖이다: 교차 소견이 이 자료에서 성립하지 않는다"
        )
    return entry, profile, ranked


def cross_sections(reader: Cas12aReader) -> tuple[render.Section, ...]:
    """Cas12a's half of the `cross` finding: its own section, never a shared table."""
    entry, profile, ranked = _cross_ranked(reader)
    stratum = _stratum(
        reader,
        CROSS_STRATUM,
        per_group=lambda group: str(
            render.profile_peak(reader.position_profile(group)["mean_abs_raw_delta"])
        ),
        effect="TTTV·TTTT 두 군의 행을 합친 평균 프로파일이다 (군별 최대 위치)",
    )
    denominator = render.Denominator(
        rows=entry["n_available"],
        unit="행",
        note=f"Test 전수 요구 {entry['requested']:,}행 중 ISM 성공 행",
    )
    figures = tuple(
        render.Figure(
            name=f"민감 위치 {rank}위",
            value=(
                f"{position} ({'PAM 안' if _in_pam(position) else 'PAM 밖'}, "
                f"값 {render.format_value(profile[position])})"
            ),
            stratum=stratum,
            head=reader.head,
            statistic=(
                "ISM 위치 프로파일 (위치별 평균 |raw delta|, WT 슬롯 제외 유효 대체 "
                "염기 평균, 집단 평균)"
            ),
            denominator=denominator,
            scale="fraction (native raw output)",
        )
        for rank, position in enumerate(ranked, start=1)
    )
    return (
        render.Section(
            title=f"근거: NT-Cas12a — {CROSS_STRATUM}",
            notes=(f"PAM {PAM[0]}-{PAM[1]} (TTTV)은 protospacer의 5′ 쪽이다.",),
            figures=figures,
        ),
    )


def cross_finding(reader: Cas12aReader) -> str:
    """The Cas12a clause of the finding, worded from the positions it found.

    Where the second position is outside the PAM, the clause says so for that
    position rather than naming only the one inside: a sentence that lists the
    inside positions and leaves the other out would read as if both were in
    the PAM while the evidence below it prints `PAM 밖`.
    """
    _, _, ranked = _cross_ranked(reader)
    if sorted(ranked) == [PAM[1] - 1, PAM[1]]:
        where = f"PAM {PAM[0]}-{PAM[1]}의 마지막 두 염기"
    else:
        where = f"PAM {PAM[0]}-{PAM[1]} 기준 " + ", ".join(
            f"{position}은 {'안' if _in_pam(position) else '밖'}" for position in ranked
        )
    return f"NT-Cas12a: 가장 민감한 두 위치 {ranked[0]}·{ranked[1]} ({where}). PAM은 5′ 쪽."


def cross_agreement_mention(reader: Cas12aReader) -> str:
    """Cas12a's IG-ISM agreement, named only to say what population it is over.

    Like SpCas9's, it carries its own provenance in prose - population, head,
    statistic kind and denominator - because this one is deliberately not a
    `render.Figure`.
    """
    entry = reader.agreement_primary(CROSS_STRATUM)
    return (
        f"NT-Cas12a {render.format_value(entry['median'])}는 Test 전수"
        f"({entry['n_valid']:,}행, TTTV·TTTT 풀링)의 t1-t5 토큰 기준"
        f"({entry['primary']}), head={reader.head}, IG-ISM "
        "spearman_rank_correlation의 행별 분포 median이다"
    )
