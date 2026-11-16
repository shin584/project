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
