"""9-Hydra v3 (SpCas9) 뷰. 이 티켓에서는 `report`.

Only the three heads XAI was actually run on are shown (`SpCas9`,
`SpCas9-NG`, `evoCas9`; primary `SpCas9`). The prediction table and the ISM
archive store nine, because one forward pass returns them all; every head
named here comes from `reader.analysed_heads`, so the other six have no path
onto the screen.

The report opens with the questions this analysis answered, because its
sections are not the Cas12a report's: Cas12a has one output and so neither a
head-ordering reversal nor a replicate noise floor.

The replicate noise floor lives inside the head-ordering reversal section and
nowhere else. It is the spread of a repeated 30-mer's *measured* labels
(fraction scale), which is what the reversal tests an inverted pair's measured
label gap against. The IG-ISM agreement is a dimensionless rank correlation,
so the floor has neither its unit nor its object, and placing the two side by
side would invite reading 0.70 against 0.0637. The floor is shown per head, as
`single_constant_note` requires; the pooled median is never shown.

The IG-ISM agreement carries no verdict. It is shown as measured, beside the
artifact's `mismatch_note`: neither technique is the other's ground truth.

Both strata stay apart everywhere (`"pooling": "forbidden"`): every figure
names one stratum, and no count is the sum of the two.
"""

from __future__ import annotations

from . import render
from .reader import SpCas9Reader

TITLE = "9-Hydra v3 (SpCas9) XAI 분석 (저장된 산출물, 분석된 head 3개)"

QUESTIONS = (
    "IG와 ISM이 다섯 토큰의 순위를 비슷하게 매기는가 — 층·head별 측정값으로",
    "분석된 3개 head가 한 행 안에서 head 순서를 실측과 같게 매기는가 — head 순서 역전, 층별로",
    "역전된 head 쌍의 실측 효율 차이가 replicate 노이즈 바닥을 넘는 행은 몇 개인가 — head별 바닥으로",
    "각 수치가 어느 모집단에서 왔는가 — ISM sweep과 IG 고정 명단은 분모가 다르다",
)

NOT_ASKED = (
    "묻지 않은 것: attention rollout의 head별 해석. rollout은 공유 trunk의 attention이라 "
    "head 축이 없다 (ADR 0008). 예측 성능 표도 이 report에 없다: 저장 번들에 SpCas9 "
    "성능 산출물이 없고, 데모는 재계산하지 않는다."
)

#: The population every cross-head v3 figure is computed on (CONTEXT.md:
#: 9-Hydra v3 analysis intersection). It is not one of the two strata, and the
#: two strata are not its parts: it also holds rows in neither of them.
INTERSECTION = "analysis_intersection"


#: A Korean gloss for each artifact note the report quotes. The gloss is the
#: demo's own summary and is labelled as such: the quote above it stays the
#: authority, so a speaker can read either without the audience losing track of
#: which words the analysis recorded and which the demo added.
MISMATCH_GLOSS = (
    "상관이 낮은 것은 버그가 아니라 토큰 수준 attribution의 한계다. 마지막 토큰이 "
    "PAM(24-26)과 오른쪽 flank(27-29)를 함께 묶으므로, 염기 단위 ISM과 토큰 단위 IG가 "
    "거기서 일치할 이유가 없다 (ADR 0007). 불일치는 측정된 그대로 적고, 수치를 올리려고 "
    "다시 돌리거나 임계값을 바꾸거나 버리지 않는다."
)
REVERSAL_DEFINITION_GLOSS = (
    "한 행에서 예측한 head 순서가 실측 순서와 어긋난 행을 말한다. 세 head 쌍에 대한 "
    "Kendall tau가 음수이고, 어느 쪽에도 동률 쌍이 없는 행이다. 이 tau는 한 행 안에서 "
    "head들끼리를 비교한 값이다."
)
SELECTION_POOL_GLOSS = (
    "head 순서 역전은 보고된 결과이며 사례를 골라내는 후보 명단이 아니다. 역전된 쌍 "
    "전부가 노이즈 바닥을 넘어야 한다고 요구하면 단 1행만 남으므로, 명단을 채우려고 "
    "기준을 낮추면 측정 노이즈를 모델의 실패로 보고하게 된다. 그래서 이 절은 개수와 "
    "분포만 담고 고를 수 있는 행 목록은 담지 않는다."
)
CLEARING_FLOOR_GLOSS = (
    "역전된 쌍마다 그 쌍의 바닥을 각각 넘어야 한 행으로 센다. 쌍의 바닥은 그 쌍을 "
    "이루는 두 head의 바닥 중 더 큰 쪽이다."
)
QUALIFIER_GLOSS = (
    "이 층의 행은 타깃 5개의 PAM 변이체다. 그래서 n은 행 수가 아니라 타깃 5개이며, 이 "
    "행 수를 다른 층에 더하지 않는다. 이 층의 모든 풀링된 수치는 그 5개 타깃의 변이체 "
    "위에서 센 값이라서 타깃별로 한 번 더 적는다 — 한 타깃이 떠받쳐서 성립하는 비율과 "
    "다섯 타깃 모두에서 성립하는 비율은 다른 결과다."
)


def _heads_label(reader: SpCas9Reader) -> str:
    return "분석된 3개 head (" + "·".join(reader.analysed_heads) + ")"


def _percent_points(fraction: float) -> str:
    """A fraction-scale efficiency spread in percentage points, for the speaker."""
    return f"{fraction * 100:.2f}%p"


# --------------------------------------------------------------------------- #
# population
# --------------------------------------------------------------------------- #


def _population(reader: SpCas9Reader) -> render.Section:
    validation = reader.validation
    figures = [
        render.Figure(
            name="ISM sweep 행 수",
            value=reader.ism_rows.rows,
            stratum=INTERSECTION,
            head=_heads_label(reader),
            statistic="행 수 (ISM sweep 모집단)",
            denominator=render.Denominator(rows=reader.ism_rows.rows, unit="행"),
            note=(
                "두 층의 합이 아니다: 어느 층에도 속하지 않는 행(non-library, "
                "non-NGG)도 담는다"
            ),
        )
    ]
    for stratum in reader.strata:
        declared = validation["strata"][stratum]
        figures.append(
            render.Figure(
                name="IG 고정 명단 행 수",
                value=len(reader.ig_subset_rows(stratum)),
                stratum=stratum,
                head=_heads_label(reader),
                statistic="행 수 (IG 고정 명단, 이 층)",
                denominator=render.Denominator(
                    rows=int(declared["rows"]),
                    unit="행",
                    targets=int(declared["distinct_protospacers"]),
                ),
            )
        )
    heads = reader.analysed_heads
    return render.Section(
        title="모집단",
        notes=(
            (
                f"분석된 head: {', '.join(heads)} (primary: {reader.primary_head}). "
                "저장만 되고 분석되지 않은 head는 이 데모에 나오지 않는다."
            ),
            "두 층은 합치지 않는다 (pooling: forbidden). 층마다 따로 적는다.",
        ),
        figures=tuple(figures),
    )


# --------------------------------------------------------------------------- #
# IG-ISM agreement
# --------------------------------------------------------------------------- #


def _agreement(reader: SpCas9Reader) -> render.Section:
    figures: list[render.Figure] = []
    statistic = reader.validation["statistic"]
    for stratum in reader.strata:
        for head in reader.analysed_heads:
            entry = reader.validation_figures(stratum, head, "magnitude")
            per_row = entry["per_row"]
            undefined = int(per_row.get("undefined_rows", 0))
            figures.append(
                render.Figure(
                    name=f"{head} IG-ISM 순위 상관 (magnitude, 행별 median)",
                    value=per_row["median"],
                    stratum=stratum,
                    head=head,
                    statistic=f"{statistic} (magnitude, 행별 분포의 median)",
                    denominator=render.Denominator(
                        rows=int(per_row["rows"]),
                        unit="행",
                        targets=entry["distinct_protospacers"],
                    ),
                    scale="무차원 순위 상관",
                    note=f"정의되지 않는 행 {undefined:,}행" if undefined else None,
                )
            )
    return render.Section(
        title="IG-ISM 일치도 (층·head별, 판정 없음)",
        notes=(
            (
                "측정된 값으로 제시한다. 합격·불합격 판정을 붙이지 않는다 — 두 기법은 "
                "서로의 정답이 아니다."
            ),
            *render.quoted(reader.validation["mismatch_note"], MISMATCH_GLOSS),
            (
                "행별 분포의 median만 적는다. 층 평균 프로파일끼리의 상관과 위치 "
                "프로파일은 `agreement` 뷰의 것이다 (#161)."
            ),
            (
                "두 층을 합친 값은 없다 (pooling: forbidden). generalisation은 처음 보는 "
                "protospacer의 일반화, pam_library는 타깃 5개의 PAM 실험이다."
            ),
        ),
        figures=tuple(figures),
    )


# --------------------------------------------------------------------------- #
# head-ordering reversal, with the noise floor in its only context
# --------------------------------------------------------------------------- #


def _floor_notes(reader: SpCas9Reader) -> tuple[str, ...]:
    """What the floor is, in the one place it appears, and what it is not.

    The definition is the artifact's: the spread of one repeated 30-mer's
    measured labels on the fraction scale. Saying so with the percentage-point
    reading is what keeps it from being read as a correlation threshold.
    """
    return (
        (
            "replicate 노이즈 바닥: 반복 측정된 같은 30-mer의 **실측** 라벨 "
            "max − min (fraction 척도)의 head별 median. head마다 따로 적는다."
        ),
        (
            "head별 값을 쓴다 (single_constant_note). pooled 상수는 쓰지 않는다: "
            "evoCas9의 바닥이 다른 두 head보다 훨씬 낮아 pooled를 쓰면 evoCas9의 "
            "자기 노이즈 밖 차이까지 통과시킨다."
        ),
        (
            "더 큰 쪽을 쓰는 이유: 쌍의 실측 차이가 두 head 모두의 노이즈를 넘지 못하면 "
            "그 순서는 측정 노이즈 안에서 정해진 것이고, 모델이 틀렸다고 말할 근거가 "
            "되지 못한다."
        ),
        (
            "이 바닥은 **실측 효율의 변동 폭**이라 IG-ISM 일치도(무차원 순위 상관)와는 "
            "단위도 대상도 달라 연결할 수 없다. 여기, 실측 효율 차이를 해석하는 문맥에만 둔다."
        ),
        (
            # limit_note asks for both pairs of counts wherever the floor is
            # used: the cut is the sharpest edge of the limit.
            "바닥의 한계: 한 split의 반복 30-mer 중 분석 교집합에 남은 것만이 근거다 — "
            f"그룹 {int(reader.noise_floor['groups']):,}개·"
            f"{int(reader.noise_floor['rows']):,}행이 "
            f"분석 교집합 {int(reader.noise_floor['intersection_rows']):,}행에 대한 것이고, "
            "자르지 않은 split의 반복은 "
            f"{int(reader.noise_floor['repeated_groups_in_split']):,}그룹·"
            f"{int(reader.noise_floor['repeated_rows_in_split']):,}행이다. 대부분의 반복 "
            "30-mer는 분석된 3개 head 전부에서 측정되지 않았다."
        ),
    )


def _reversal(reader: SpCas9Reader) -> render.Section:
    reversal = reader.head_order_reversal
    floor = reader.reversal_floor()
    figures: list[render.Figure] = []

    for head in reader.analysed_heads:
        own = reader.noise_floor_by_head(head)
        # One source for the displayed number: the floor artifact's own
        # per-head median. `reversal_floor` has already refused the case where
        # the reversal quotes something else.
        value = float(own["median"])
        figures.append(
            render.Figure(
                name=f"{head} replicate 노이즈 바닥 (median spread)",
                value=value,
                stratum=INTERSECTION,
                head=head,
                statistic="replicate 노이즈 바닥 (실측 라벨 spread의 median)",
                denominator=render.Denominator(
                    rows=int(own["groups"]),
                    unit="그룹",
                    note="반복 30-mer 그룹, 그룹마다 spread 하나",
                ),
                scale=f"fraction = {_percent_points(value)}",
            )
        )

    for stratum in reader.strata:
        entry = reader.reversal_stratum(stratum)
        rows = int(entry["rows"])
        targets = int(entry["distinct_protospacers"])
        population = render.Denominator(rows=rows, unit="행", targets=targets)
        reversals = int(entry["reversal_rows"])

        figures.append(
            render.Figure(
                name="head 순서 역전 행",
                value=reversals,
                stratum=stratum,
                head=_heads_label(reader),
                statistic=f"reversal_rows ({reversal['statistic']} < 0, 동률 쌍 제외)",
                denominator=population,
                note=(
                    f"tau가 음수인 행 전체 {int(entry['tau_negative_rows']):,}행, "
                    f"그중 동률 쌍이 있어 제외한 행 "
                    f"{int(entry['tied_rows_with_negative_tau']):,}행 "
                    f"(동률 쌍이 있는 행은 모두 {int(entry['tied_rows']):,}행)"
                ),
            )
        )
        figures.append(
            render.Figure(
                name="그중 실측 차이가 쌍별 노이즈 바닥을 넘는 행",
                value=int(entry["clearing_floor"]["rows"]),
                stratum=stratum,
                head=_heads_label(reader),
                statistic="clearing_floor (쌍별 head 노이즈 바닥 적용)",
                denominator=render.Denominator(
                    rows=reversals, unit="행", note="역전 행"
                ),
                note=CLEARING_FLOOR_GLOSS,
            )
        )
        gap = entry["gap_over_floor"]
        figures.append(
            render.Figure(
                name="역전된 쌍의 실측 차이 / 그 쌍의 바닥 (median)",
                value=gap["median"],
                stratum=stratum,
                head=_heads_label(reader),
                statistic="gap_over_floor (실측 label gap ÷ 쌍별 바닥) median",
                denominator=render.Denominator(
                    rows=int(gap["rows"]), unit="쌍", note="역전된 head 쌍"
                ),
                scale="바닥의 배수 (1.0이면 바닥과 같다)",
                note="개수보다 이 분포가 소견을 지탱한다: 1.0 미만이면 그 쌍의 순서는 측정 노이즈 안에서 정해졌다",
            )
        )

        for target in entry.get("per_target", ()):
            target_rows = int(target["rows"])
            figures.append(
                render.Figure(
                    name=f"타깃 {target['protospacer']} 역전 행",
                    value=int(target["reversal_rows"]),
                    stratum=f"{stratum} / 타깃 1개",
                    head=_heads_label(reader),
                    statistic="reversal_rows (이 타깃의 PAM 변이체 안에서)",
                    denominator=render.Denominator(
                        rows=target_rows, unit="행", targets=1
                    ),
                    note=(
                        f"쌍별 바닥을 넘는 행 {int(target['rows_clearing_floor']):,}행"
                    ),
                )
            )

        for pair, pair_floor in floor["pair_floors"].items():
            inverted = entry["inverted_pairs"][pair]
            figures.append(
                render.Figure(
                    name=f"{pair.replace('_vs_', ' vs ')} 역전 쌍의 실측 효율 차이 median",
                    value=inverted["label_gap"]["median"],
                    stratum=stratum,
                    head=pair.replace("_vs_", " vs "),
                    statistic="역전된 쌍의 measured label gap median",
                    denominator=render.Denominator(
                        rows=int(inverted["rows"]), unit="행", note="이 쌍이 역전된 행"
                    ),
                    scale=(
                        "fraction "
                        f"({_percent_points(float(inverted['label_gap']['median']))}) "
                        f"— 이 쌍의 바닥 {float(pair_floor):.4g}"
                        f" ({_percent_points(float(pair_floor))})"
                    ),
                    note=(f"바닥을 넘는 행 {int(inverted['rows_clearing_floor']):,}행"),
                )
            )

    notes = (
        (
            "분석된 3개 head를 유지하는 이유가 이 절이다: 한 행 안에서 head 사이 순서가 "
            "역전되는 행이 실제로 있어서, head 하나만 보여주면 이 결과가 사라진다."
        ),
        *render.quoted(reversal["reversal_definition"], REVERSAL_DEFINITION_GLOSS),
        *render.quoted(reversal["selection_pool_note"], SELECTION_POOL_GLOSS),
        *render.quoted(floor["pair_floor_rule"], CLEARING_FLOOR_GLOSS),
        *_floor_notes(reader),
        *(
            line
            for note in (reader.reversal_stratum("pam_library").get("qualifier_note"),)
            if note
            for line in render.quoted(note, QUALIFIER_GLOSS)
        ),
    )
    return render.Section(title="head 순서 역전", notes=notes, figures=tuple(figures))


# --------------------------------------------------------------------------- #
# cases and explain
# --------------------------------------------------------------------------- #

CASES_TITLE = "9-Hydra v3 (SpCas9) 기술 사례 (원 자료의 case_id 그대로)"
CASES_QUESTIONS = (
    "어떤 행을 사례로 골랐고 무엇을 기준으로 골랐는가 — 층·기준별로",
    "그 사례 중 어느 행이 IG 고정 명단 안에 있는가",
)
CASES_NOTES = (
    "ID는 원 자료(cases.json)의 `case_id`다. 데모가 ID를 새로 만들지 않는다.",
    (
        "선정은 예측과 라벨만 읽고 attribution을 읽지 않는다. 설명할 attribution으로 "
        "사례를 고르면 순환이 된다."
    ),
    (
        "한 행은 한 사례다. 기준은 ERRLO → ERRHI → HEADGAP 순서로 적용되므로 HEADGAP은 "
        "오차 cell이 가져가지 않은 행을 받는다."
    ),
    "행별 값은 `explain <ID>`로 본다.",
)

EXPLAIN_QUESTIONS = (
    "이 행에 어떤 방법이 있고 어떤 방법이 없는가 — 없는 것은 사유와 함께",
    "있는 방법이 이 행에서 무엇을 가리키는가 — head별로",
)
SPLIT_SAFEGUARD_NOTE = (
    "split 역할을 항상 적는다. 실측으로 ISM 모집단과 IG 고정 명단은 전부 `test`이고 "
    "미매칭 ID가 없으므로 이 표시는 경고가 아니라 저가 안전장치다 — train·val 행의 "
    "오차를 설명하는 사고를 막고, 전제가 깨지면 바로 드러난다."
)
LIBRARY_QUALIFIER_GLOSS = (
    "이 사례는 타깃 5개 중 하나에서 관측된 것이다. 같은 cell의 사례 3건을 독립 관측 "
    "3건으로 읽지 않는다."
)
IG_COVERAGE_GLOSS = (
    "IG는 사례 선정 **전에** 자기 seed로 고정된 400행 명단에서 돌았다. 그래서 사례가 "
    "IG를 갖는 것은 두 명단이 우연히 겹친 경우뿐이고, IG가 없는 사례는 실패가 아니라 "
    "명단 밖이라는 뜻이다. 사례를 명단에 맞춰 고르거나 버리지 않았다."
)


def _selection_pool(reader: SpCas9Reader, stratum: str) -> render.Denominator:
    entry = reader.case_strata[stratum]
    return render.Denominator(
        rows=int(entry["rows"]),
        unit="행",
        targets=int(entry["distinct_protospacers"]),
        note="이 층의 선정 모집단",
    )


#: Which head a criterion ranked on. `ERRLO`/`ERRHI` rank the `SpCas9`
#: absolute error inside that head's own quartiles; `HEADGAP` is the spread of
#: *all three* analysed heads' absolute errors, so naming one head there would
#: misstate the criterion for a third of the cases.
HEADGAP = "HEADGAP"


def _criterion_head(reader: SpCas9Reader, criterion: str) -> str:
    if criterion == HEADGAP:
        return f"{_heads_label(reader)} — 세 head 절대오차의 max − min"
    return f"{reader.primary_head} (이 기준이 순위를 매긴 head)"


def _case_line(reader: SpCas9Reader, case: dict) -> render.Figure:
    """One listed case: its row, its split role, and which methods it carries."""
    case_id = str(case["case_id"])
    row_id = int(case["row_index"])
    methods = reader.case_methods(case_id)
    # Short here, with reasons once in the section's notes: the IG reason and
    # the rollout limit are the same sentence for every case, and repeating
    # either eighteen times buries the per-case facts a speaker is reading.
    # `explain <ID>` states them against the single row.
    carried = ", ".join(entry.method for entry in methods if entry.present)
    missing = ", ".join(entry.method for entry in methods if not entry.present)
    note = (
        f"row_index={row_id} / split={reader.split_role(row_id) or '미매칭'} / "
        f"보유: {carried}"
    )
    note += f" / 없음: {missing}" if missing else " / 없는 방법 없음"
    qualifier = str(case.get("reporting_qualifier") or "").strip()
    if qualifier:
        note += f" / 기록된 한정(원문): {qualifier}"
    return render.Figure(
        name=f"{case_id} ({case['criterion']})",
        value=f"row_index {row_id}",
        stratum=str(case["stratum"]),
        head=_criterion_head(reader, str(case["criterion"])),
        statistic=(
            f"사례 선정 ({case['criterion']}, 이 cell의 선정 순위 "
            f"{case['selection_rank']})"
        ),
        denominator=_selection_pool(reader, str(case["stratum"])),
        note=note,
    )


def cases(reader: SpCas9Reader) -> str:
    """The SpCas9 `cases` view: every case under the case table's own `case_id`."""
    records = [reader.case(case_id) for case_id in reader.case_ids()]
    with_ig = [
        str(case["case_id"])
        for case in records
        if reader.case_annotation(str(case["case_id"])).get("ig") is not None
    ]
    coverage = reader.cases_annotated.get("ig_coverage") or {}
    ig_note = (
        f"사례 {len(records):,}개 중 IG를 가진 사례 {len(with_ig):,}개"
        + (f": {', '.join(with_ig)}" if with_ig else "")
        + f". 나머지는 IG 고정 명단 {reader.ig_rows.rows:,}행에 없다."
    )
    rollout_limit = next(
        entry.limit
        for entry in reader.case_methods(records[0]["case_id"])
        if entry.method == "rollout"
    )
    notes = [
        *CASES_NOTES,
        ig_note,
        f"rollout은 모든 사례 행을 덮지만 {rollout_limit}",
    ]
    if coverage.get("note"):
        notes.extend(render.quoted(str(coverage["note"]), IG_COVERAGE_GLOSS))
    section = render.Section(
        title="사례 목록",
        notes=tuple(notes),
        figures=tuple(_case_line(reader, case) for case in records),
    )
    return render.render(
        [section],
        claim_limit=reader.claim_limit,
        title=CASES_TITLE,
        questions=CASES_QUESTIONS,
    )


def _explain_row(reader: SpCas9Reader, case: dict) -> render.Section:
    """The row itself: split role first, then this row's labels and predictions."""
    row_id = int(case["row_index"])
    stratum = str(case["stratum"])
    figures = [
        render.Figure(
            name="split 역할",
            value=reader.split_role(row_id) or "미매칭 (split 파일에 이 행이 없다)",
            stratum=stratum,
            head="해당 없음 (행의 성질)",
            statistic="published_split의 split 열",
            denominator=render.Denominator(
                rows=reader.split_rows, unit="행", note="split 파일 3개 합계"
            ),
        )
    ]
    for head in reader.analysed_heads:
        label = case.get(f"label_{head}")
        prediction = case.get(f"pred_{head}")
        figures.append(
            render.Figure(
                name=f"{head} 실측 / 예측",
                value=(
                    (render.format_value(label) if label is not None else "미측정")
                    + " / "
                    + (
                        render.format_value(prediction)
                        if prediction is not None
                        else "예측 없음"
                    )
                ),
                stratum=stratum,
                head=head,
                statistic="사례 표의 실측 라벨과 예측",
                denominator=render.single_row(),
                scale="fraction",
            )
        )
    notes = [SPLIT_SAFEGUARD_NOTE]
    qualifier = str(case.get("reporting_qualifier") or "").strip()
    if qualifier:
        notes.extend(render.quoted(qualifier, LIBRARY_QUALIFIER_GLOSS))
    return render.Section(
        title=f"{case['case_id']} — 행과 선정",
        notes=tuple(notes),
        figures=tuple(figures),
    )


def _explain_methods(reader: SpCas9Reader, case: dict) -> render.Section:
    """Which methods this row carries, and for each absence the recorded reason."""
    notes = [entry.text() for entry in reader.case_methods(str(case["case_id"]))]
    return render.Section(title="이 행의 방법별 유무", notes=tuple(notes))


def _explain_values(reader: SpCas9Reader, case: dict) -> render.Section:
    """The recorded per-head figures for the methods this row does carry."""
    annotation = reader.case_annotation(str(case["case_id"]))
    stratum = str(case["stratum"])
    ism = annotation["ism"]
    figures: list[render.Figure] = []
    for head in reader.analysed_heads:
        figures.append(
            render.Figure(
                name=f"{head} ISM 토큰 프로파일",
                value=ism["token_profile"][head],
                stratum=stratum,
                head=head,
                statistic="위치별 평균 |ISM delta|의 토큰 평균 (WT 슬롯 제외)",
                denominator=render.single_row(),
                scale="fraction",
                note=f"토큰 순서: {', '.join(reader.token_names)}",
            )
        )
        figures.append(
            render.Figure(
                name=f"{head} 최대 민감 위치 / 토큰",
                value=f"{ism['top_position'][head]} / {ism['top_token'][head]}",
                stratum=stratum,
                head=head,
                statistic="이 행 ISM 프로파일의 최대 위치와 그 토큰",
                denominator=render.single_row(),
            )
        )
        figures.append(
            render.Figure(
                name=f"{head} PAM 구간이 차지하는 민감도 몫",
                value=ism["pam_share"][head],
                stratum=stratum,
                head=head,
                statistic="PAM(24-26) 평균 ÷ 30bp 창 평균",
                denominator=render.single_row(),
                scale="무차원 비율 (창의 몫이며 염기별 PAM attribution이 아니다)",
                note=(
                    "이 층 안에서의 백분위 "
                    f"{float(ism['pam_share_percentile_in_stratum'][head]):.4g}, "
                    "attribution outlier 여부 "
                    f"{render.format_value(annotation['attribution_outlier'][head])} "
                    "(기술용이며 선정 기준이 아니다)"
                ),
            )
        )
    notes = [
        (
            "ISM은 염기 치환, IG는 토큰 attribution이다. 해상도가 달라 둘은 서로의 "
            "정답이 아니다 (ADR 0007)."
        )
    ]
    ig = annotation.get("ig")
    if ig is None:
        notes.append(
            f"IG 값이 이 절에 없는 이유: 이 행은 IG 고정 명단 "
            f"{reader.ig_rows.rows:,}행에 없다."
        )
    else:
        for head in reader.analysed_heads:
            figures.append(
                render.Figure(
                    name=f"{head} IG 토큰 attribution",
                    value=ig["token_attribution"][head],
                    stratum=stratum,
                    head=head,
                    statistic="토큰 5개의 IG attribution",
                    denominator=render.single_row(),
                    scale=reader.ig_scale,
                    note=f"최대 토큰: {ig['top_token'][head]}",
                )
            )
    return render.Section(
        title="방법별 값 (head별)", notes=tuple(notes), figures=tuple(figures)
    )


def _explain_head_order(reader: SpCas9Reader, case: dict) -> render.Section:
    """This row's own head ordering: a description of a row, never a selection.

    The `report`'s reversal section counts rows per stratum; this one reads the
    single row's recorded tau and its inverted pairs, so a reader looking at one
    case can see whether its heads disagree and whether that disagreement is
    larger than the replicate noise of the pair it belongs to. The section's own
    note points back at `report` for the population figures, because one row's
    ordering says nothing about how common that ordering is.
    """
    order = reader.case_annotation(str(case["case_id"]))["head_order"]
    stratum = str(case["stratum"])
    figures = [
        render.Figure(
            name="이 행의 head 순서 역전",
            value=bool(order["reversal"]),
            stratum=stratum,
            head=_heads_label(reader),
            statistic=f"{reader.head_order_reversal['statistic']} (이 행의 head 3쌍)",
            denominator=render.single_row(),
            note=(
                f"tau={float(order['tau']):.4g}, 동률 쌍="
                f"{render.format_value(order['tied_pair'])}"
            ),
        )
    ]
    for pair in order.get("inverted_pairs", ()):
        gap = float(pair["label_gap"])
        pair_floor = float(pair["floor"])
        heads = str(pair["pair"]).replace("_vs_", " vs ")
        figures.append(
            render.Figure(
                name=f"{heads} 역전 쌍의 실측 차이",
                value=gap,
                stratum=stratum,
                head=heads,
                statistic="역전된 쌍의 measured label gap",
                denominator=render.Denominator(rows=1, unit="쌍", note="이 행의 이 쌍"),
                scale=(
                    f"fraction ({_percent_points(gap)}) — 이 쌍의 바닥 "
                    f"{pair_floor:.4g} ({_percent_points(pair_floor)})"
                ),
                note=f"바닥을 넘는가: {render.format_value(pair['clears_floor'])}",
            )
        )
    return render.Section(
        title="head 순서 (이 행)",
        notes=(
            (
                "행별 head 순서는 기술이며 선정 기준이 아니다. 모집단 수치는 "
                "`report`의 head 순서 역전 절에 있다."
            ),
            *render.quoted(str(order["clears_noise_floor_rule"]), CLEARING_FLOOR_GLOSS),
        ),
        figures=tuple(figures),
    )


def explain(reader: SpCas9Reader, identifier: str) -> str:
    """The SpCas9 `explain <ID>` view: one row, method by method."""
    case = reader.case(identifier)
    return render.render(
        [
            _explain_row(reader, case),
            _explain_methods(reader, case),
            _explain_values(reader, case),
            _explain_head_order(reader, case),
        ],
        claim_limit=reader.claim_limit,
        title=f"{case['case_id']} (row_index {case['row_index']}) — 방법별 유무와 값",
        questions=EXPLAIN_QUESTIONS,
    )


def report(reader: SpCas9Reader) -> str:
    """The SpCas9 `report` view."""
    return render.render(
        [_population(reader), _agreement(reader), _reversal(reader)],
        claim_limit=reader.claim_limit,
        title=TITLE,
        questions=(*QUESTIONS, NOT_ASKED),
        pointers=(render.CROSS_POINTER,),
    )
