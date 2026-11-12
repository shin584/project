"""`cross`: the one qualitative finding the two analyses share, and nothing more.

Each model's most sensitive span overlaps its own PAM: SpCas9's
`pam_plus_right_flank` token (PAM 3′ of the protospacer), NT-Cas12a's last two
PAM bases (PAM 5′). That is a statement about each model separately.

The two analyses share no axis — different datasets, splits, label
definitions, token sets and populations — so their figures never go in one
table. Each nuclease's evidence is its own section, built by that nuclease's
own view module from that nuclease's reader, and the screen says out loud what
is *not* compared and why: performance, because the datasets and label
definitions differ, and the IG-ISM agreement numbers, because they are not the
same statistic counted on the same population.

The SpCas9 numbers are generalisation-stratum means over the IG fixed list for
the SpCas9 head — not the ISM sweep, not both IG strata — so the denominator
is printed with every one of them, and the PAM library stratum's counterparts
are a section apart.

This view is the one that stands outside a single analysis, so it takes both
readers and the shell offers it with no nuclease selected.
"""

from __future__ import annotations

from . import cas12a_views, render, spcas9_views
from .reader import Cas12aReader, SpCas9Reader

TITLE = "교차 소견 — 9-Hydra v3 (SpCas9)와 NT-Cas12a (정성적, 수치 비교 없음)"

QUESTIONS = ("두 모델 각각에서, 가장 민감한 구간이 그 모델 자신의 PAM과 겹치는가",)

NOT_ASKED = (
    "묻지 않은 것: 두 모델 중 어느 쪽이 더 민감한가, 더 잘 맞히는가. 두 분석은 "
    "공통 축이 없어 그런 질문이 성립하지 않는다."
)

FINDING_HEADING = "소견 (정성적, 하나)"

FINDING_LEAD = "두 모델 모두 최대 민감 구간이 자기 PAM과 겹친다."

FINDING_SHAPE = (
    "같은 방향의 소견이지만 같은 수치가 아니다. PAM의 방향도 서로 반대이고 "
    "(SpCas9 3′, Cas12a 5′), 한쪽은 토큰 5개 단위, 다른 쪽은 위치 단위다. "
    "그래서 두 뉴클레아제의 수치를 나란히 둔 비교표를 만들지 않는다."
)

NOT_COMPARED_HEADING = "비교하지 않는 것"

NOT_COMPARED_PERFORMANCE = (
    "성능 지표 (Spearman·MAE 등): 두 모델은 데이터셋·분할·라벨 정의가 서로 다르다. "
    "각 report에서 그 모델의 모집단과 함께 본다."
)

NOT_COMPARED_AGREEMENT_LEAD = (
    "IG-ISM 일치도 수치: 같은 통계를 같은 모집단에서 센 것이 아니다. 각각은 이렇다 —"
)

NOT_COMPARED_AGREEMENT_TAIL = (
    "모집단·토큰 정의·head 축이 모두 달라 두 수치의 차이는 모델의 차이로 읽을 수 "
    "없다. 각 뉴클레아제의 `agreement` 뷰에서 그 분석 안에서만 본다."
)

#: Printed first in the claim-limit frame, above both artifacts' own records.
#: The finding names both models, so the limit is stated for both; Cas12a
#: records its limits as prose sections with no equivalent single sentence, so
#: this line is labelled as the demo's own and each quote below it as an
#: artifact's.
CLAIM_LIMIT_BOTH = (
    "이 화면의 수치는 두 모델 모두에 대해, 그 모델이 예측에 쓰는 내부 정보 사용"
    "(어느 위치·토큰에 예측이 의존하는가)을 기술한다. 물리적 DNA/단백질 결합에 "
    "대한 주장이 아니다. 이 한 문장은 데모가 두 분석에 공통으로 적는 것이다 "
    "(NT-Cas12a의 산출물은 한계를 한 문장이 아니라 절 단위 산문으로 적는다). "
    "아래에 각 artifact가 기록한 한계를 그대로 인용한다: SpCas9의 문장, 그리고 "
    "이 화면에 Cas12a ISM 수치가 뜨므로 NT-Cas12a의 한계 절."
)

CAS12A_QUOTE_LABEL = "원문(NT-Cas12a artifact 기록, report.ko.md의 한계 절):"

CLAIM_LIMIT_GLOSS = (
    "여기 실린 모든 수치는 모델이 예측에 어느 위치·토큰을 쓰는지, 그리고 한 행의 "
    "head 순서를 어떻게 매기는지를 기술한다. 물리적 DNA/단백질 상호작용에 대한 "
    "주장은 하나도 없다."
)


def claim_limit(spcas9: SpCas9Reader, cas12a: Cas12aReader) -> str:
    """The demo's common sentence, then SpCas9's record, then Cas12a's sections.

    `cross` prints both analyses' figures, so both analyses' recorded limits
    are on screen. Each record is quoted as the artifact's and the sentence
    covering both models is labelled as the demo's own: restating a recorded
    limit *in place of* the quote would make the demo the author of a limit
    the analysis stated. SpCas9's record is English, so the demo's Korean
    reading is set beside it and labelled as a gloss; Cas12a's is already
    Korean and is quoted as it stands.
    """
    quote, gloss = render.quoted(spcas9.claim_limit, CLAIM_LIMIT_GLOSS)
    return (
        f"{CLAIM_LIMIT_BOTH}\n\n{quote}\n{gloss}\n\n"
        f"{CAS12A_QUOTE_LABEL}\n{cas12a.claim_limit}"
    )


def _finding(spcas9: SpCas9Reader, cas12a: Cas12aReader) -> render.Section:
    """The finding itself: one lead line, one clause per model, no figures.

    The clauses are built by each nuclease's own module from its own
    artifacts, so what is said here cannot drift from the evidence sections
    below it.
    """
    return render.Section(
        title=FINDING_HEADING,
        notes=(
            FINDING_LEAD,
            spcas9_views.cross_finding(spcas9),
            cas12a_views.cross_finding(cas12a),
            FINDING_SHAPE,
        ),
    )


def _not_compared(spcas9: SpCas9Reader, cas12a: Cas12aReader) -> render.Section:
    """What this screen does not compare, and why. Named numbers, no figures.

    The two agreement medians are *named* here — with the population each was
    counted on — precisely so nobody reads them off two `agreement` screens
    and subtracts them. They carry no `render.Figure`: a figure would put them
    in the shape of a displayed result, which is what this section refuses.
    """
    return render.Section(
        title=NOT_COMPARED_HEADING,
        notes=(
            NOT_COMPARED_PERFORMANCE,
            NOT_COMPARED_AGREEMENT_LEAD,
            f"  - {spcas9_views.cross_agreement_mention(spcas9)}",
            f"  - {cas12a_views.cross_agreement_mention(cas12a)}",
            NOT_COMPARED_AGREEMENT_TAIL,
        ),
    )


def cross(*, spcas9: SpCas9Reader, cas12a: Cas12aReader) -> str:
    """The `cross` view: the finding, each model's evidence apart, what is not compared."""
    sections = [
        _finding(spcas9, cas12a),
        *spcas9_views.cross_sections(spcas9),
        *cas12a_views.cross_sections(cas12a),
        _not_compared(spcas9, cas12a),
    ]
    return render.render(
        sections,
        claim_limit=claim_limit(spcas9, cas12a),
        title=TITLE,
        questions=(*QUESTIONS, NOT_ASKED),
    )
