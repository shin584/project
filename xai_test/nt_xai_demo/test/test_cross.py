"""`cross`: one qualitative finding across the two models, and what is not compared.

The finding is that each model's most sensitive span overlaps its own PAM -
SpCas9's `pam_plus_right_flank` token (3′), Cas12a's positions in the TTTV
PAM (5′). It is stated as a finding about each model separately: the two
nucleases' figures are never put side by side in one table, because nothing
about them is on a common axis.

The SpCas9 figures are generalisation-stratum means over the IG fixed list
(200 rows, 193 targets in the real data) for the SpCas9 head - not the ISM
sweep, not both IG strata - so the denominator is printed with them, and the
PAM library stratum's counterparts are shown apart.

The same screen says what is not compared (performance, IG-ISM agreement) and
why, and that every figure describes the model's internal information use,
not physical DNA/protein binding.
"""

from __future__ import annotations

import pytest

from xai_test.nt_xai_demo import cross_views, render
from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo import shell as sh
from xai_test.nt_xai_demo.test import reader_fixtures as fx


@pytest.fixture
def data_dir(tmp_path):
    return fx.build_data_dir(tmp_path / "data")


@pytest.fixture
def spcas9(data_dir):
    return rd.SpCas9Reader(data_dir)


@pytest.fixture
def cas12a(data_dir):
    return rd.Cas12aReader(data_dir)


@pytest.fixture
def text(spcas9, cas12a):
    return cross_views.cross(spcas9=spcas9, cas12a=cas12a)


def _sections(text: str) -> dict[str, list[str]]:
    """The rendered text split at its `[title]` lines."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], [])
        elif line.startswith("-- "):
            current = None
        elif current is not None:
            current.append(line)
    return sections


def _section(text: str, fragment: str) -> str:
    found = [
        "\n".join(lines)
        for title, lines in _sections(text).items()
        if fragment in title
    ]
    assert len(found) == 1, (fragment, list(_sections(text)))
    return found[0]


def _heads(section: str) -> set[str]:
    return {
        line.split("head=")[1].split(" / ")[0]
        for line in section.splitlines()
        if "head=" in line
    }


# --------------------------------------------------------------------------- #
# the shell
# --------------------------------------------------------------------------- #


def test_cross_runs_with_no_nuclease_selected(data_dir, spcas9, cas12a):
    shell = sh.Shell(data_dir=data_dir)

    outcome = shell.execute("cross")

    assert outcome.ok, outcome.text
    assert outcome.text == cross_views.cross(spcas9=spcas9, cas12a=cas12a)


def test_cross_prints_the_same_whichever_nuclease_is_selected(data_dir):
    shell = sh.Shell(data_dir=data_dir)
    unselected = shell.execute("cross").text

    for nuclease in ("cas12a", "spcas9"):
        shell.execute(nuclease)
        assert shell.execute("cross").text == unselected, nuclease


def test_cross_is_its_own_one_shot_command(data_dir):
    outcome = sh.run_once(["cross"], data_dir=data_dir)

    assert outcome.ok, outcome.text
    assert "PAM" in outcome.text


def test_cross_takes_no_argument(data_dir):
    outcome = sh.run_once(["cross", "SpCas9"], data_dir=data_dir)

    assert not outcome.ok


def test_each_report_carries_only_the_one_line_pointer(data_dir):
    """One pointer line per report, and no other line sending a reader to `cross`.

    Matched on the backticked view name, not on the word: an artifact's own
    prose says "cross-head" and "across heads", and a substring match would
    pass or fail on wording that has nothing to do with this pointer.
    """
    for nuclease in ("cas12a", "spcas9"):
        report = sh.run_once([nuclease, "report"], data_dir=data_dir).text
        mentions = [line for line in report.splitlines() if "`cross`" in line]
        assert mentions == [render.CROSS_POINTER], nuclease


# --------------------------------------------------------------------------- #
# the finding
# --------------------------------------------------------------------------- #


def test_it_states_one_qualitative_finding_for_both_models(text):
    finding = _section(text, "소견")

    assert "자기 PAM" in finding
    assert "3′" in finding and "5′" in finding


def test_no_section_holds_both_nucleases_figures(text, spcas9, cas12a):
    """No comparison table: each nuclease's figures sit in sections of their own."""
    for title, lines in _sections(text).items():
        heads = _heads("\n".join(lines))
        assert not (spcas9.primary_head in heads and cas12a.head in heads), title


def test_the_spcas9_figures_are_the_generalisation_tokens_of_the_spcas9_head(
    text, spcas9
):
    section = _section(text, "SpCas9 — generalisation")
    recorded = spcas9.validation_figures("generalisation", spcas9.primary_head)

    assert _heads(section) == {spcas9.primary_head}
    assert "층=generalisation" in section
    assert "pam_plus_right_flank" in section and "24-29" in section
    for profile in (recorded["ism_token_profile"], recorded["ig_token_profile"]):
        assert render.format_value(profile[-1]) in section
        others = profile[:-1]
        assert render.format_value(min(others)) in section
        assert render.format_value(max(others)) in section


def test_the_spcas9_denominator_is_the_strata_rows_and_targets_on_screen(text, spcas9):
    section = _section(text, "SpCas9 — generalisation")
    recorded = spcas9.validation_figures("generalisation", spcas9.primary_head)
    denominator = (
        f"분모={recorded['rows']:,}행·{recorded['distinct_protospacers']:,}타깃"
    )

    provenance = [line for line in section.splitlines() if "분모=" in line]
    assert provenance and all(denominator in line for line in provenance)
    # It says what the denominator is not: the ISM sweep, or all IG rows.
    assert f"{spcas9.ism_rows.rows:,}행" in section
    assert f"{spcas9.ig_rows.rows:,}행" in section


def test_the_pam_library_counterparts_are_shown_apart(text, spcas9):
    library = _section(text, "SpCas9 — pam_library")
    generalisation = _section(text, "SpCas9 — generalisation")
    recorded = spcas9.validation_figures("pam_library", spcas9.primary_head)

    assert "층=pam_library" in library
    assert "층=pam_library" not in generalisation
    assert render.format_value(recorded["ism_token_profile"][-1]) in library
    assert render.format_value(recorded["ig_token_profile"][-1]) in library
    assert (
        f"{recorded['rows']:,}행·{recorded['distinct_protospacers']:,}타깃" in library
    )


def test_the_cas12a_figures_are_its_peak_positions_inside_its_pam(text, cas12a):
    section = _section(text, "NT-Cas12a")
    profile = cas12a.position_profile("all")["mean_abs_raw_delta"]
    peak = render.profile_peak(profile)

    assert _heads(section) == {cas12a.head}
    assert "PAM 4-7" in section
    assert f"{peak} (PAM 안" in section
    assert render.format_value(float(profile[peak])) in section
    # `all` pools TTTV and TTTT, so the pooling is stated with each group's peak.
    assert "풀링:" in section
    for group in ("TTTV", "TTTT"):
        assert group in section


def test_a_second_position_outside_the_pam_is_said_to_be_outside(
    spcas9, cas12a, monkeypatch
):
    """The clause cannot read as if both positions were in the PAM.

    The finding rests on the most sensitive position; the second one is shown
    beside it, and where that one falls outside the PAM the sentence says so
    for that position rather than listing only the one inside.
    """
    recorded = cas12a.position_profile

    def runner_up_outside(stratum):
        entry = recorded(stratum)
        entry["mean_abs_raw_delta"][20] = float(entry["mean_abs_raw_delta"][7]) / 2
        return entry

    monkeypatch.setattr(cas12a, "position_profile", runner_up_outside)
    text = cross_views.cross(spcas9=spcas9, cas12a=cas12a)

    finding = _section(text, "소견")
    assert "20은 밖" in finding
    assert "마지막 두 염기" not in finding
    assert "민감 위치 2위: 20 (PAM 밖" in _section(text, "NT-Cas12a")


def test_the_finding_is_refused_when_spcas9s_peak_token_is_not_the_pam(
    spcas9, cas12a, monkeypatch
):
    recorded = spcas9.validation_figures

    def peak_elsewhere(stratum, head, variant="magnitude"):
        entry = recorded(stratum, head, variant)
        entry["ism_token_profile"] = list(reversed(entry["ism_token_profile"]))
        return entry

    monkeypatch.setattr(spcas9, "validation_figures", peak_elsewhere)

    with pytest.raises(render.RenderRefused, match="pam_plus_right_flank"):
        cross_views.cross(spcas9=spcas9, cas12a=cas12a)


def test_the_finding_is_refused_when_cas12as_peak_is_outside_its_pam(
    spcas9, cas12a, monkeypatch
):
    recorded = cas12a.position_profile

    def peak_elsewhere(stratum):
        entry = recorded(stratum)
        entry["mean_abs_raw_delta"][20] = 9.0
        return entry

    monkeypatch.setattr(cas12a, "position_profile", peak_elsewhere)

    with pytest.raises(render.RenderRefused, match="PAM"):
        cross_views.cross(spcas9=spcas9, cas12a=cas12a)


# --------------------------------------------------------------------------- #
# what is not compared, and the claim limit
# --------------------------------------------------------------------------- #


def test_it_says_what_is_not_compared_and_why(text, spcas9, cas12a):
    section = _section(text, "비교하지 않는 것")

    assert "성능 지표" in section
    for reason in ("데이터셋", "분할", "라벨"):
        assert reason in section
    assert "일치도" in section
    spcas9_agreement = spcas9.validation_figures("generalisation", spcas9.primary_head)
    cas12a_agreement = cas12a.agreement_primary("all")
    assert render.format_value(spcas9_agreement["per_row"]["median"]) in section
    assert render.format_value(cas12a_agreement["median"]) in section
    assert "generalisation" in section and "magnitude" in section
    assert "t1-t5" in section
    assert "같은 모집단" in section


def test_the_not_compared_section_holds_no_figures(text):
    """The agreement numbers are named to say why they are apart, not tabulated."""
    section = _section(text, "비교하지 않는 것")

    assert "층=" not in section


def test_the_claim_limit_says_internal_use_not_physical_binding(text, spcas9):
    claim = text.split(f"-- {render.CLAIM_LIMIT_HEADING}")[1]

    assert spcas9.claim_limit in claim  # SpCas9's own words, verbatim
    assert "내부 정보 사용" in claim
    assert "물리적" in claim
    assert "NT-Cas12a" in claim  # stated for both, not only the one that wrote it
