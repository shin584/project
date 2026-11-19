"""Model-free site discovery, including the full model-window boundary."""

import pytest

from pam_scanner import CAS_CONFIGS, reverse_complement, scan_all, scan_pam


@pytest.mark.parametrize("reverse", [False, True])
def test_spcas9_sites_include_strand_pam_window_and_unique_key(reverse):
    seq = "A" * 43 + "AGG" + "A" * 35
    result = scan_all(reverse_complement(seq) if reverse else seq)
    strand = "-" if reverse else "+"
    sites = result.by_cas("SpCas9")

    assert {s.site_key for s in sites} == {
        f"SpCas9|{strand}|43", f"SpCas9|{strand}|44",
    }
    site = next(s for s in sites if s.pam_start == 43)
    assert site.pam_seq == "AGG"
    assert site.guide_seq == "A" * 20
    assert site.model_input_seq == "A" * 24 + "AGGAAA"
    assert site.cut_pos == 40
    assert site.distance == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_cas12a_sites_have_their_own_keys_and_34bp_windows(reverse):
    seq = "A" * 16 + "TTTA" + "A" * 61
    result = scan_all(reverse_complement(seq) if reverse else seq)
    [site] = result.by_cas("Cas12a")
    assert site.site_key == f"Cas12a|{'-' if reverse else '+'}|16"
    assert site.model_input_seq == "TTTA" + "A" * 30
    assert site.guide_seq == "A" * 23
    assert site.cut_pos == 40
    assert site.distance == 0


@pytest.mark.parametrize("pam_start,accepted", [(23, False), (24, True), (75, True), (76, False)])
def test_spcas9_rejects_sites_without_a_complete_model_window(pam_start, accepted):
    seq = "A" * pam_start + "AGG" + "A" * (78 - pam_start)
    sites = scan_pam(seq, CAS_CONFIGS[0], max_dist=81)
    assert any(s.strand == "+" and s.pam_start == pam_start for s in sites) == accepted


@pytest.mark.parametrize("pam_start,accepted", [(47, True), (48, False)])
def test_cas12a_rejects_sites_even_when_the_guide_fits(pam_start, accepted):
    seq = "A" * pam_start + "TTTA" + "A" * (77 - pam_start)
    sites = scan_pam(seq, CAS_CONFIGS[1], max_dist=81)
    assert any(s.strand == "+" and s.pam_start == pam_start for s in sites) == accepted


def test_sites_outside_the_cut_distance_are_excluded():
    seq = "A" * 26 + "AGG" + "A" * 52
    assert scan_all(seq).sites == []


# ── 계열별 Cas12a 윈도우와 WINDOW_LAYOUT (#196 + #197 통합) ──────


CAS12A_DEMO = (
    "CAAAGTAGCCGGTCGCTTTAAGAATAAATAAAAGCGCCACAGGATTGACT"
    "AAACCACAGCCGGCGCCCTGGCCCCCTACCG"
)


def _cas12a_windows(configs):
    from pam_scanner import scan_all

    return {s.site_key: s for s in scan_all(CAS12A_DEMO, configs=configs).sites
            if s.cas_type == "Cas12a"}


def test_the_two_families_scan_the_same_cas12a_sites_with_different_windows():
    """계열이 달라도 사이트는 같고 **윈도우만** 4bp 앞으로 밀린다 (#197).

    `site_key`가 같으므로 화면의 사이트 선택은 그대로 쓰이지만 `model_input_seq`는
    다르다 — 계열을 섞어 윈도우를 재사용하면 조용히 다른 구간을 설명하게 된다.
    """
    from pam_scanner import CAS_CONFIGS
    from non_nt_cas12a_predict import MANIFEST_CAS_CONFIGS

    nt = _cas12a_windows(CAS_CONFIGS)
    non_nt = _cas12a_windows(MANIFEST_CAS_CONFIGS)
    assert nt and set(nt) == set(non_nt)
    for key, site in nt.items():
        other = non_nt[key]
        assert len(site.model_input_seq) == len(other.model_input_seq) == 34
        assert site.pam_seq == other.pam_seq
        assert site.model_input_seq != other.model_input_seq
        # non-NT 윈도우는 NT 윈도우보다 정확히 4bp 앞에서 시작한다.
        assert other.model_input_seq[4:] .startswith(site.model_input_seq[:30])


def test_window_layout_matches_the_non_nt_cas12a_window_only():
    """`WINDOW_LAYOUT["Cas12a"]`의 PAM 구간이 어느 계열에 맞는지 고정한다.

    PAM 4–7은 non-NT 윈도우에서만 실제 PAM이고, NT 윈도우에서는 PAM이 0–3이다
    (#190이 기록한 불일치). `WINDOW_LAYOUT`은 `cas_type` 하나로만 갈려 이 차이를
    담지 못하므로, non-NT 설명을 붙이는 #199/#200이 계열별 layout을 정해야 한다.
    이 검사는 그 전제가 조용히 바뀌는 것을 막는다.
    """
    from pam_scanner import CAS_CONFIGS, WINDOW_LAYOUT
    from non_nt_cas12a_predict import MANIFEST_CAS_CONFIGS

    name, start, end = next(seg for seg in WINDOW_LAYOUT["Cas12a"] if seg[0] == "PAM")
    assert (name, start, end) == ("PAM", 4, 7)
    for key, site in _cas12a_windows(MANIFEST_CAS_CONFIGS).items():
        assert site.model_input_seq[start:end + 1] == site.pam_seq
    for key, site in _cas12a_windows(CAS_CONFIGS).items():
        assert site.model_input_seq[start:end + 1] != site.pam_seq
        assert site.model_input_seq[:4] == site.pam_seq
