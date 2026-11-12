"""`agreement`: IG-ISM agreement per stratum, and the ISM position profile as bars.

SpCas9's `ig_ism_validation.json` declares `"pooling": "forbidden"`, so the
generalisation stratum (200 rows, 193 targets) and the PAM library stratum (200
rows, 5 targets) are shown apart, each with its own denominator, and no figure
anywhere is their sum. Cas12a's populations are its PAM groups; `all` pools
them and has to say so.

The agreement carries no pass/fail verdict: it is shown as measured, beside the
artifact's own note that neither technique is the other's ground truth.

The profile is drawn as console ASCII bars with the PAM bracketed, so whether
the most sensitive position sits inside the PAM is visible at a glance. The
SVGs in the Cas12a bundle are named by path only, never rendered - SpCas9 has
no counterpart, and rendering one side only would make the display asymmetric.
The ISM position profile is the mean |delta| over the three alternative bases,
the reference (WT) slot excluded.
"""

from __future__ import annotations

import json
import re

import pytest

from xai_test.nt_xai_demo import cas12a_views, render, spcas9_views
from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo import shell as sh
from xai_test.nt_xai_demo.test import reader_fixtures as fx

VERDICT_WORDS = ("합격", "불합격", "통과 판정", "우수", "충분히", "양호", "실패했다")
BAR_LINE = re.compile(r"^\s*([\[\]| ])\s+(\d+)\s+(#*)\s+(\S+)")


@pytest.fixture
def data_dir(tmp_path):
    return fx.build_data_dir(tmp_path / "data")


@pytest.fixture
def spcas9(data_dir):
    return rd.SpCas9Reader(data_dir)


@pytest.fixture
def cas12a(data_dir):
    return rd.Cas12aReader(data_dir)


def _provenance(text: str) -> list[str]:
    return [
        line.strip() for line in text.splitlines() if line.strip().startswith("층=")
    ]


def _bars(text: str) -> list[tuple[str, int, str, str]]:
    """Every profile bar line: (PAM marker, position, bar, value text)."""
    found = []
    for line in text.splitlines():
        match = BAR_LINE.match(line)
        if match and "#" in line:
            found.append(
                (match.group(1), int(match.group(2)), match.group(3), match.group(4))
            )
    return found


def _profiles(text: str) -> list[list[tuple[str, int, str, str]]]:
    """Bar lines grouped into one list per drawn profile (positions restart at 0)."""
    groups: list[list[tuple[str, int, str, str]]] = []
    for bar in _bars(text):
        if bar[1] == 0:
            groups.append([])
        groups[-1].append(bar)
    return groups


# --------------------------------------------------------------------------- #
# both nucleases
# --------------------------------------------------------------------------- #


def test_agreement_runs_for_both_nucleases_through_the_shell(data_dir):
    shell = sh.Shell(data_dir=data_dir)
    for nuclease in ("cas12a", "spcas9"):
        shell.execute(nuclease)
        outcome = shell.execute("agreement")

        assert outcome.ok, outcome.text
        assert render.CLAIM_LIMIT_HEADING in outcome.text


def test_every_figure_carries_stratum_head_statistic_and_denominator(spcas9, cas12a):
    for text in (spcas9_views.agreement(spcas9), cas12a_views.agreement(cas12a)):
        lines = _provenance(text)
        assert lines
        for line in lines:
            for field in ("층=", "head=", "통계=", "분모="):
                assert field in line, line


def test_the_claim_limit_closes_both_views(spcas9, cas12a):
    spcas9_text = spcas9_views.agreement(spcas9)
    cas12a_text = cas12a_views.agreement(cas12a)

    assert spcas9_text.rstrip().endswith(fx.CLAIM_LIMIT)
    assert fx.CAS12A_LIMIT_TEXT in cas12a_text.split(render.CLAIM_LIMIT_HEADING)[1]


def test_no_view_passes_a_verdict_on_the_agreement(spcas9, cas12a):
    for text in (spcas9_views.agreement(spcas9), cas12a_views.agreement(cas12a)):
        # The demo's own sentence saying it gives no verdict is the one place
        # those words may appear; everything else must be verdict-free.
        body = "\n".join(
            line for line in text.splitlines() if "판정을 붙이지 않는다" not in line
        )
        for word in VERDICT_WORDS:
            assert word not in body, word


# --------------------------------------------------------------------------- #
# SpCas9: two strata, never pooled
# --------------------------------------------------------------------------- #


def test_spcas9_shows_each_stratum_apart_with_its_own_figure(spcas9):
    text = spcas9_views.agreement(spcas9)

    for stratum, median in fx.VALIDATION_MEDIAN.items():
        lines = [
            line
            for line in text.splitlines()
            if "per_row" not in line and f"{median:.4g}" in line
        ]
        assert lines, stratum
    for line in _provenance(text):
        assert line.startswith(("층=generalisation", "층=pam_library")), line


def test_spcas9_never_shows_a_figure_over_both_strata(spcas9):
    text = spcas9_views.agreement(spcas9)
    pooled_rows = sum(
        int(spcas9.validation["strata"][stratum]["rows"]) for stratum in spcas9.strata
    )

    for line in _provenance(text):
        assert f"분모={pooled_rows:,}행" not in line, line
        assert "풀링:" not in line, line


def test_spcas9_shows_every_analysed_head_in_each_stratum(spcas9):
    text = spcas9_views.agreement(spcas9)

    for stratum in spcas9.strata:
        for head in fx.ANALYSED_HEADS:
            assert any(
                line.startswith(f"층={stratum} / head={head} ")
                for line in _provenance(text)
            ), (stratum, head)


def test_spcas9_quotes_the_mismatch_note_beside_the_agreement(spcas9):
    text = spcas9_views.agreement(spcas9)

    assert spcas9.validation["mismatch_note"] in text


def test_spcas9_profile_is_drawn_per_stratum_with_the_pam_bracketed(spcas9):
    text = spcas9_views.agreement(spcas9)
    profiles = _profiles(text)

    assert len(profiles) == len(spcas9.strata)
    for bars in profiles:
        assert [position for _, position, _, _ in bars] == list(range(fx.POSITIONS))
        markers = {position: marker for marker, position, _, _ in bars}
        assert markers[24] == "[" and markers[25] == "|" and markers[26] == "]"
        assert all(markers[p] == " " for p in range(fx.POSITIONS) if not 24 <= p <= 26)


def test_spcas9_profile_values_are_the_wt_excluded_mean(spcas9):
    text = spcas9_views.agreement(spcas9)
    profiles = dict(zip(spcas9.strata, _profiles(text)))

    for stratum, bars in profiles.items():
        expected = fx.spcas9_stratum_profile(stratum, spcas9.primary_head)
        shown = [float(value) for _, _, _, value in bars]
        assert shown == pytest.approx(expected, rel=1e-3), stratum
        # Averaging over all four slots would read three quarters of this.
        assert shown != pytest.approx([v * 0.75 for v in expected], rel=1e-3)


def test_spcas9_profile_head_is_chosen_by_name(spcas9):
    primary = _profiles(spcas9_views.agreement(spcas9))[0]
    evo = _profiles(spcas9_views.agreement(spcas9, "evoCas9"))[0]

    expected = fx.spcas9_stratum_profile("generalisation", "evoCas9")
    assert [float(v) for *_, v in evo] == pytest.approx(expected, rel=1e-3)
    assert [v for *_, v in evo] != [v for *_, v in primary]


def test_spcas9_profile_refuses_a_head_that_was_not_analysed(spcas9):
    with pytest.raises(KeyError):
        spcas9_views.agreement(spcas9, "xCas")


def test_spcas9_profile_that_disagrees_with_the_recorded_tokens_is_refused(
    data_dir,
):
    path = data_dir / "spcas9/v3_report/ig_ism_validation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    tokens = report["strata"]["generalisation"]["by_head"]["SpCas9"]["magnitude"][
        "mean_token_profile"
    ]
    tokens["ism"] = [value * 0.75 for value in tokens["ism"]]
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(rd.ArtifactMismatch):
        spcas9_views.agreement(rd.SpCas9Reader(data_dir))


def test_spcas9_states_where_the_peak_sits_against_the_pam(spcas9):
    text = spcas9_views.agreement(spcas9)

    # The fixture's profile rises with position, so its peak is position 29,
    # past the PAM: the view has to say so rather than assume the PAM wins.
    peak_lines = [line for line in text.splitlines() if "최대 민감 위치" in line]
    assert peak_lines
    assert all("29" in line and "PAM 밖" in line for line in peak_lines)


def test_spcas9_names_no_svg_because_it_has_none(spcas9):
    text = spcas9_views.agreement(spcas9)

    assert ".svg" not in text
    assert "SVG" in text  # the asymmetry is stated, not left silent


# --------------------------------------------------------------------------- #
# Cas12a: three PAM populations, `all` declared pooled
# --------------------------------------------------------------------------- #


def test_cas12a_reads_the_protocols_primary_statistic(cas12a):
    text = cas12a_views.agreement(cas12a)

    for population, (_, median, rho) in fx.CAS12A_AGREEMENT.items():
        assert any(
            line.strip().startswith(f"층={population}") for line in _provenance(text)
        ), population
        assert f"{median:.4g}" in text
        assert f"{rho:.4g}" in text
    assert fx.CAS12A_PRIMARY in text
    assert "-0.05" not in text  # a non-primary variant's value


def test_cas12a_all_says_it_pools_with_the_group_values(cas12a):
    text = cas12a_views.agreement(cas12a)

    pooled = [line for line in _provenance(text) if line.startswith("층=all")]
    assert pooled
    for line in pooled:
        assert "풀링:" in line and "TTTV" in line and "TTTT" in line, line


def test_cas12a_quotes_its_own_note_on_low_agreement(cas12a):
    text = cas12a_views.agreement(cas12a)

    for line in fx.CAS12A_AGREEMENT_TEXT.splitlines():
        assert line in text


def test_cas12a_profile_brackets_the_pam_and_finds_the_peak_inside_it(cas12a):
    text = cas12a_views.agreement(cas12a)
    profiles = _profiles(text)

    assert len(profiles) == len(fx.CAS12A_AGREEMENT)
    for bars, population in zip(profiles, cas12a.strata):
        assert len(bars) == fx.CAS12A_POSITIONS
        markers = {position: marker for marker, position, _, _ in bars}
        assert [markers[p] for p in range(4, 8)] == ["[", "|", "|", "]"]
        longest = max(bars, key=lambda bar: len(bar[2]))
        assert longest[1] == fx.CAS12A_PROFILE_PEAK[population]

    peak_lines = [line for line in text.splitlines() if "최대 민감 위치" in line]
    assert len(peak_lines) == len(fx.CAS12A_AGREEMENT)
    assert all("PAM 안" in line for line in peak_lines)


def test_cas12a_names_the_svgs_by_path_and_does_not_render_them(cas12a):
    text = cas12a_views.agreement(cas12a)

    for _, name in fx.CAS12A_SVGS:
        assert str(cas12a.root / name) in text
    assert "<svg" not in text


def test_cas12a_takes_no_head_argument(cas12a):
    with pytest.raises(KeyError):
        cas12a_views.agreement(cas12a, "SpCas9")


# --------------------------------------------------------------------------- #
# the bar renderer
# --------------------------------------------------------------------------- #


def test_profile_bars_scale_to_the_peak_and_bracket_the_pam():
    lines = render.profile_bars([0.1, 0.4, 0.2, 0.05], pam=(1, 2), width=8)
    bars = _bars("\n".join(lines))

    assert [marker for marker, *_ in bars] == [" ", "[", "]", " "]
    assert [len(bar) for _, _, bar, _ in bars] == [2, 8, 4, 1]
    peaks = [line for line in lines if render.PEAK_MARK in line]
    assert len(peaks) == 1 and " 1 " in peaks[0]


def test_profile_bars_show_a_missing_value_as_missing_not_zero():
    lines = render.profile_bars([0.1, float("nan"), 0.2], pam=(0, 0), width=4)

    missing = [line for line in lines if "값 없음" in line]
    assert len(missing) == 1
