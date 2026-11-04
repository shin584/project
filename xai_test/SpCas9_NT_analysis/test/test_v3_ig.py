"""Tests for v3's Integrated Gradients and its tokenise contract (#130).

Two groups, and they fail for different reasons:

- **Token contract** — pure string work plus the checkpoint's *own* tokenizer
  (`result/vocab.txt` and `result/tokenizer_config.json` are tracked; only the
  weights are not). These pin ADR 0007: 30bp is exactly five 6-mer tokens, the
  last one carries PAM *and* right flank, and no IG artifact may name a
  base-level `pam` key.
- **IG implementation** — a small randomly-initialised 9-output ESM built
  inside the test. No download, no checkpoint, no network. These carry
  ADR 0006: attribution is taken at the encoder input embedding, so
  completeness holds and no position's attribution is identically zero.

`test_post_encoder_gradients_are_identically_zero` is the ADR's evidence
rather than a test of our code: it demonstrates on the toy model that the
attachment point ADR 0006 forbids really does zero every sequence position,
which is what makes the strict 1e-2 completeness assertion load-bearing.
"""

import json

import numpy as np
import pandas as pd
import pytest
import v3_ig
import v3_predict
import v3_selection
import v3_strata
from v3_ig import (
    PAM_TOKEN_INDEX,
    PAM_TOKEN_NAME,
    SEQUENCE_LENGTH,
    TOKEN_BOUNDARIES,
    TOKEN_COUNT,
    TOKEN_NAMES,
    TOKEN_SPAN,
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

CHECKPOINT_DIR = v3_ig.SUBTREE_ROOT / "result"

# Three heads, deliberately non-adjacent in `v3_predict.HEADS` (0, 1, 8), so a
# head-index off-by-one cannot pass by landing on a neighbour.
HEADS = v3_strata.ANALYSED_HEADS

SEQUENCES = [
    "ACGTACGTACGTACGTACGTACGTAGGTCA",
    "TTGGCATCAGATCCGATTACAAGGTTACGC",
    "GGGCCCAAATTTGGGCCCAAATTTAGGCCA",
]

#: AC11's pin, in the shape the writer demands: 40 hex characters.
CODE_REF = "0" * 40


# --- fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))


@pytest.fixture(scope="module")
def toy_model():
    """A small randomly-initialised 9-output ESM, same shape as v3.

    Keeps `token_dropout`, `position_embedding_type` and the special-token ids
    from v3's real `config.json`, because those are exactly the parts of the
    embedding path IG has to replay faithfully.
    """
    from transformers import EsmConfig, EsmForSequenceClassification

    real = json.loads((CHECKPOINT_DIR / "config.json").read_text(encoding="utf-8"))
    config = EsmConfig(
        vocab_size=real["vocab_size"],
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=real["pad_token_id"],
        mask_token_id=real["mask_token_id"],
        token_dropout=real["token_dropout"],
        position_embedding_type=real["position_embedding_type"],
        emb_layer_norm_before=real["emb_layer_norm_before"],
        num_labels=len(v3_predict.HEADS),
        problem_type=real["problem_type"],
    )
    torch.manual_seed(42)
    return EsmForSequenceClassification(config).eval()


@pytest.fixture(scope="module")
def ig_result(toy_model, tokenizer):
    return v3_ig.run_ig(
        toy_model,
        tokenizer,
        SEQUENCES,
        row_index=[7, 11, 23],
        heads=HEADS,
        steps=v3_ig.DEFAULT_IG_STEPS,
        batch_size=2,
    )


def provenance_for(tmp_path):
    """The split every sidecar has to name (AC11), as a file on disk.

    `write_ig_artifact` takes the split as a path and digests it, so a run
    cannot leave "which bytes was the GPU time spent on?" unanswered.
    """
    split = tmp_path / "provenance" / "test.csv"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("row_index,sequence", encoding="utf-8")
    return {"split_csv": split}


def subset_for(out_dir, stratum: str, rows=(7, 11, 23)):
    """Write a stand-in manifest naming `rows` as `stratum`'s IG subset.

    `write_ig_artifact` takes the manifest as a **path** and loads it through
    `load_ig_subset` (#131's AC7), so it has to exist on disk with its
    checksum file beside it. The real one is 200 rows per stratum drawn by
    `v3_selection`; what matters here is only that the writer checks the
    artifact against the pinned bytes.
    """
    entries = pd.DataFrame(
        {
            "row_index": list(rows),
            "stratum": stratum,
            "cell": "errq1",
            "selection_order": range(1, len(rows) + 1),
            "error_quartile": 1,
            "is_ngg": True,
            "protospacer": "A" * 20,
        }
    )
    manifest = v3_selection.SubsetManifest(
        entries=entries, seed=7, rows_per_stratum=len(rows)
    )
    return v3_selection.write_ig_subset(out_dir / stratum, manifest)


# --- token contract (ADR 0007) --------------------------------------------


def test_window_is_exactly_five_six_mer_tokens():
    tokens = v3_ig.tokenize_window(SEQUENCES[0])
    assert len(tokens) == TOKEN_COUNT == 5
    assert all(len(token) == TOKEN_SPAN == 6 for token in tokens)
    assert "".join(tokens) == SEQUENCES[0]
    assert SEQUENCE_LENGTH == TOKEN_COUNT * TOKEN_SPAN


def test_token_boundaries_are_fixed_and_tile_the_window():
    assert TOKEN_BOUNDARIES == ((0, 5), (6, 11), (12, 17), (18, 23), (24, 29))
    covered = [
        position
        for index in range(TOKEN_COUNT)
        for position in range(*v3_ig.token_slice(index).indices(SEQUENCE_LENGTH))
    ]
    assert covered == list(range(SEQUENCE_LENGTH))


def test_last_token_is_named_pam_plus_right_flank():
    assert PAM_TOKEN_INDEX == TOKEN_COUNT - 1 == 4
    assert TOKEN_NAMES[PAM_TOKEN_INDEX] == PAM_TOKEN_NAME == "pam_plus_right_flank"
    assert len(TOKEN_NAMES) == TOKEN_COUNT
    assert len(set(TOKEN_NAMES)) == TOKEN_COUNT


def test_no_token_is_named_pam():
    # The whole point of ADR 0007: a token named `pam` would be a
    # PAM-plus-flank average wearing a base-level label.
    assert all(name.lower() != "pam" for name in TOKEN_NAMES)


def test_the_last_token_covers_the_pam_and_the_right_flank_exactly():
    pam_token = set(range(*v3_ig.token_slice(PAM_TOKEN_INDEX).indices(SEQUENCE_LENGTH)))
    pam = set(range(*v3_strata.PAM.indices(SEQUENCE_LENGTH)))
    right_flank = set(range(*v3_strata.RIGHT_FLANK.indices(SEQUENCE_LENGTH)))
    assert pam_token == pam | right_flank
    # ...and it is the merge, not a clean PAM: both halves are non-empty.
    assert pam and right_flank


def test_the_real_tokenizer_matches_the_contract(tokenizer):
    for sequence in SEQUENCES:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer(sequence)["input_ids"])
        assert tokens == ["<cls>", *v3_ig.tokenize_window(sequence)]
        assert tokens[v3_ig.SEQUENCE_TOKEN_SLICE] == list(
            v3_ig.tokenize_window(sequence)
        )


def test_tokenize_window_rejects_a_sequence_it_cannot_split_cleanly():
    with pytest.raises(ValueError, match="length"):
        v3_ig.tokenize_window("ACGT" * 8)
    with pytest.raises(ValueError, match="non-ACGT"):
        v3_ig.tokenize_window("N" + SEQUENCES[0][1:])


def test_artifact_keys_may_not_name_a_base_level_pam():
    v3_ig.validate_artifact_keys(["attribution", "row_index", PAM_TOKEN_NAME])
    for forbidden in ("pam", "PAM", "pam_attribution", "pam_24_26"):
        with pytest.raises(ValueError, match="ADR 0007"):
            v3_ig.validate_artifact_keys(["attribution", forbidden])


# --- IG implementation (ADR 0006) -----------------------------------------


def test_attribution_is_token_resolution_with_the_hidden_axis_summed(ig_result):
    assert ig_result.attribution.shape == (len(SEQUENCES), len(HEADS), TOKEN_COUNT)
    assert ig_result.attribution.dtype == np.float32
    assert ig_result.heads == tuple(HEADS)
    assert ig_result.token_names == TOKEN_NAMES
    # 1280 (or the toy's 32) must not survive as an axis.
    assert 32 not in ig_result.attribution.shape


def test_completeness_holds_to_one_percent_relative(ig_result):
    delta = ig_result.f_x - ig_result.f_baseline
    # If the toy model barely moves off its baseline the relative assertion
    # below would be vacuous, so pin the signal first.
    assert np.all(np.abs(delta) > 1e-3)
    assert np.all(np.abs(ig_result.completeness_residual) <= 1e-2 * np.abs(delta))
    v3_ig.assert_completeness(ig_result)


def test_attribution_is_not_identically_zero_at_any_position(ig_result):
    assert np.all(np.abs(ig_result.attribution).sum(axis=(0, 1)) > 0)
    v3_ig.assert_nonzero_attribution(ig_result)


def test_post_encoder_gradients_are_identically_zero(toy_model, tokenizer):
    """ADR 0006's trap, demonstrated rather than asserted about our code.

    Attaching IG to the post-encoder hidden state gives every sequence
    position a gradient of exactly 0, because `EsmClassificationHead` pools
    `<cls>` alone. An all-zero attribution from that mistake looks like "the
    model ignores the sequence".
    """
    encoded = v3_ig.encode(tokenizer, SEQUENCES[:1])
    hidden = toy_model.esm(**encoded).last_hidden_state.detach().requires_grad_(True)
    toy_model.classifier(hidden)[:, 0].sum().backward()
    grad = hidden.grad[0]
    assert torch.all(grad[0] != 0), "the <cls> position must still carry gradient"
    assert torch.all(grad[1:] == 0), "non-<cls> gradients are zero by construction"
    # The input-embedding attachment point does not have this property.
    assert np.all(
        np.abs(
            v3_ig.run_ig(
                toy_model,
                tokenizer,
                SEQUENCES[:1],
                row_index=[0],
                heads=HEADS[:1],
                steps=8,
            ).attribution
        )
        > 0
    )


def test_the_attributed_function_is_the_deployed_forward_pass(
    ig_result, toy_model, tokenizer
):
    """`f_x` must be the model's own output, token dropout included.

    Passing `inputs_embeds` instead of replacing the word-embedding output
    silently skips ESM's `token_dropout` rescale (it is guarded on
    `input_ids is not None`), so IG would explain a function the deployed
    model never computes.
    """
    encoded = v3_ig.encode(tokenizer, SEQUENCES)
    columns = [v3_predict.HEADS.index(head) for head in HEADS]
    with torch.no_grad():
        deployed = toy_model(**encoded).logits.double().numpy()[:, columns]
    # Tolerance is float32 batching noise, not slack: the mistake this guards
    # against is ~1e4 times larger, as the next assertion shows.
    np.testing.assert_allclose(ig_result.f_x, deployed, rtol=1e-5, atol=1e-8)

    with torch.no_grad():
        embeddings = toy_model.get_input_embeddings()(encoded["input_ids"])
        shortcut = (
            toy_model(
                inputs_embeds=embeddings,
                attention_mask=encoded["attention_mask"],
            )
            .logits.double()
            .numpy()[:, columns]
        )
    assert np.max(np.abs(shortcut - deployed)) > 1e-4, (
        "the inputs_embeds shortcut is supposed to differ from the deployed "
        "forward pass (it skips ESM's token_dropout rescale) - if it no longer "
        "does, this test has stopped guarding anything"
    )


def test_the_baseline_is_the_zero_embedding(ig_result, toy_model, tokenizer):
    encoded = v3_ig.encode(tokenizer, SEQUENCES[:1])
    embeddings = toy_model.get_input_embeddings()
    handle = embeddings.register_forward_hook(
        lambda module, inputs, output: torch.zeros_like(output)
    )
    try:
        with torch.no_grad():
            logits = toy_model(**encoded).logits.double().numpy()[0]
    finally:
        handle.remove()
    expected = logits[[v3_predict.HEADS.index(head) for head in HEADS]]
    np.testing.assert_allclose(ig_result.f_baseline, expected, rtol=1e-6, atol=1e-9)


def test_row_index_is_carried_and_must_be_ascending(ig_result, toy_model, tokenizer):
    assert ig_result.row_index.tolist() == [7, 11, 23]
    assert ig_result.row_index.dtype == np.int64
    with pytest.raises(ValueError, match="ascending"):
        v3_ig.run_ig(
            toy_model, tokenizer, SEQUENCES, row_index=[23, 7, 11], heads=HEADS, steps=4
        )


def test_batching_does_not_change_the_attribution(toy_model, tokenizer, ig_result):
    single = v3_ig.run_ig(
        toy_model,
        tokenizer,
        SEQUENCES,
        row_index=[7, 11, 23],
        heads=HEADS,
        steps=v3_ig.DEFAULT_IG_STEPS,
        batch_size=3,
    )
    np.testing.assert_allclose(
        single.attribution, ig_result.attribution, rtol=1e-5, atol=1e-7
    )


def test_the_batch_is_moved_to_the_model_device(toy_model, tokenizer):
    """IG runs on Colab's GPU; the tokenizer always returns CPU tensors.

    Only the CPU leg is exercisable here, so this pins the plumbing rather
    than the CUDA behaviour: the device is read off the model, and `encode`
    moves the batch to whatever `device_of` reports.
    """
    assert v3_ig.device_of(toy_model) == next(toy_model.parameters()).device
    encoded = v3_ig.encode(tokenizer, SEQUENCES[:1], device=v3_ig.device_of(toy_model))
    assert encoded["input_ids"].device == v3_ig.device_of(toy_model)


def test_run_ig_encodes_every_batch_onto_the_model_device(
    toy_model, tokenizer, monkeypatch
):
    seen = []
    original = v3_ig.encode

    def record(*args, **kwargs):
        seen.append(kwargs.get("device"))
        return original(*args, **kwargs)

    monkeypatch.setattr(v3_ig, "encode", record)
    v3_ig.run_ig(
        toy_model,
        tokenizer,
        SEQUENCES,
        row_index=[0, 1, 2],
        heads=HEADS,
        steps=2,
        batch_size=1,
    )
    assert seen == [v3_ig.device_of(toy_model)] * len(SEQUENCES)


def test_run_ig_rejects_a_sequence_of_the_wrong_length(toy_model, tokenizer):
    with pytest.raises(ValueError, match="length"):
        v3_ig.run_ig(
            toy_model, tokenizer, ["ACGT" * 8], row_index=[0], heads=HEADS, steps=4
        )


# --- run-time tolerance: reported, never fatal ----------------------------


def test_run_tolerance_is_reported_as_a_pass_rate_and_does_not_raise():
    residual = np.array([[0.0, 0.5], [0.0, 0.0]])
    f_x = np.array([[1.0, 1.0], [1.0, 1.0]])
    f_baseline = np.array([0.0, 0.0])
    report = v3_ig.completeness_report(residual, f_x, f_baseline)
    assert report["pass_rate"] == pytest.approx(0.75)
    assert report["failing"] == 1
    assert report["per_head"]["pass_rate"] == [1.0, 0.5]
    assert report["tolerance"] == {"absolute_floor": 0.01, "relative": 0.05}
    assert report["max_abs_residual"] == pytest.approx(0.5)


def test_run_tolerance_floors_the_relative_term_at_one_percent():
    # |f(x) - f(0)| == 0, so a purely relative tolerance would fail every row
    # for a residual of any size. The 0.01 floor is what makes it meaningful.
    f_x = np.zeros((1, 1))
    f_baseline = np.zeros(1)
    assert v3_ig.completeness_report(np.array([[0.009]]), f_x, f_baseline)[
        "pass_rate"
    ] == pytest.approx(1.0)
    assert v3_ig.completeness_report(np.array([[0.02]]), f_x, f_baseline)[
        "pass_rate"
    ] == pytest.approx(0.0)


def test_the_relative_term_dominates_once_the_signal_is_large():
    f_x, f_baseline = np.array([[1.0]]), np.array([0.0])
    report = v3_ig.completeness_report(np.array([[0.04]]), f_x, f_baseline)
    assert report["pass_rate"] == pytest.approx(1.0)
    assert v3_ig.completeness_report(np.array([[0.06]]), f_x, f_baseline)[
        "pass_rate"
    ] == pytest.approx(0.0)


def test_the_unit_tolerance_is_strict_where_the_run_tolerance_is_loose(
    toy_model, tokenizer
):
    """A 5% residual passes the run report and fails the unit assertion.

    The two tolerances exist for different jobs: the loose one keeps a 400-row
    Colab run from aborting on near-zero-signal rows, the strict one is what
    actually catches the all-zero-attribution bug.
    """
    result = v3_ig.run_ig(
        toy_model, tokenizer, SEQUENCES[:1], row_index=[0], heads=HEADS[:1], steps=8
    )
    broken = result.replace(completeness_residual=0.04 * np.abs(result.delta))
    assert v3_ig.completeness_report(
        broken.completeness_residual, broken.f_x, broken.f_baseline
    )["pass_rate"] == pytest.approx(1.0)
    with pytest.raises(AssertionError, match="completeness"):
        v3_ig.assert_completeness(broken)


# --- artifact --------------------------------------------------------------


def test_write_ig_artifact_round_trips_and_names_no_pam_key(ig_result, tmp_path):
    path = v3_ig.write_ig_artifact(
        tmp_path,
        ig_result,
        code_ref=CODE_REF,
        **provenance_for(tmp_path),
        stratum=v3_strata.GENERALISATION,
        subset_manifest=subset_for(tmp_path, v3_strata.GENERALISATION),
    )
    with np.load(path) as archive:
        keys = set(archive.files)
        assert keys == {
            "attribution",
            "completeness_residual",
            "special_attribution",
            "row_index",
        }
        v3_ig.validate_artifact_keys(keys)
        assert archive["attribution"].dtype == np.float32
        assert archive["completeness_residual"].dtype == np.float64
        assert archive["special_attribution"].dtype == np.float64
        assert archive["row_index"].dtype == np.int64
        assert archive["row_index"].tolist() == [7, 11, 23]
        np.testing.assert_array_equal(archive["attribution"], ig_result.attribution)

    sidecar = json.loads((path.parent / "ig.sidecar.json").read_text(encoding="utf-8"))
    assert sidecar["heads"] == list(HEADS)
    assert sidecar["token_names"] == list(TOKEN_NAMES)
    assert sidecar["token_boundaries"] == [list(b) for b in TOKEN_BOUNDARIES]
    assert sidecar["stratum"] == v3_strata.GENERALISATION
    assert sidecar["steps"] == v3_ig.DEFAULT_IG_STEPS
    assert sidecar["attribution_point"] == "encoder_input_embedding"
    assert sidecar["baseline"] == "zero_embedding"
    # All nine, not just the three attributed: one forward returns them all,
    # and f(0) cannot be recovered once the Colab session ends.
    assert list(sidecar["f_baseline"]) == list(v3_predict.HEADS)
    for position, head in enumerate(HEADS):
        assert sidecar["f_baseline"][head] == pytest.approx(
            ig_result.f_baseline[position], rel=1e-6, abs=1e-9
        )
    assert sidecar["completeness"]["pass_rate"] == pytest.approx(1.0)
    assert sidecar["rows"] == len(SEQUENCES)


def test_the_completeness_check_is_re_derivable_from_the_artifact(ig_result, tmp_path):
    """An auditor holding only the written files can recompute f(x).

    This is what "the check cannot be re-run after the Colab session ends"
    asks for: f(x) = f(0) + attribution.sum() + special + residual. Without
    `special_attribution` the stored residual could only be re-read, since the
    <cls> and padding share of the sum is not in the five reported tokens.
    """
    path = v3_ig.write_ig_artifact(
        tmp_path,
        ig_result,
        code_ref=CODE_REF,
        **provenance_for(tmp_path),
        stratum=v3_strata.PAM_LIBRARY,
        subset_manifest=subset_for(tmp_path, v3_strata.PAM_LIBRARY),
    )
    sidecar = json.loads((path.parent / "ig.sidecar.json").read_text(encoding="utf-8"))
    with np.load(path) as archive:
        f_baseline = np.array(
            [sidecar["f_baseline"][head] for head in sidecar["heads"]]
        )
        recovered = (
            f_baseline
            + archive["attribution"].sum(axis=2)
            + archive["special_attribution"]
            + archive["completeness_residual"]
        )
    np.testing.assert_allclose(recovered, ig_result.f_x, rtol=1e-5, atol=1e-7)


def test_write_ig_artifact_refuses_an_all_zero_attribution(ig_result, tmp_path):
    """ADR 0006's failure must not reach disk, unlike a tolerance miss.

    The run tolerance is reported and never fatal, but an identically zero
    profile is the wiring bug itself, and downstream it reads as "the model
    ignores the sequence".
    """
    broken = ig_result.replace(attribution=np.zeros_like(ig_result.attribution))
    with pytest.raises(ValueError, match="identically zero"):
        v3_ig.write_ig_artifact(
            tmp_path,
            broken,
            code_ref=CODE_REF,
            **provenance_for(tmp_path),
            stratum=v3_strata.GENERALISATION,
            subset_manifest=subset_for(tmp_path, v3_strata.GENERALISATION),
        )


def test_write_ig_artifact_refuses_an_unknown_stratum(ig_result, tmp_path):
    with pytest.raises(ValueError, match="unknown stratum"):
        v3_ig.write_ig_artifact(
            tmp_path,
            ig_result,
            code_ref=CODE_REF,
            **provenance_for(tmp_path),
            stratum="generalization",
            subset_manifest=subset_for(tmp_path, v3_strata.GENERALISATION),
        )


def test_write_ig_artifact_refuses_a_non_finite_attribution(ig_result, tmp_path):
    broken = ig_result.replace(
        attribution=np.where(
            np.arange(ig_result.attribution.size).reshape(ig_result.attribution.shape)
            == 0,
            np.nan,
            ig_result.attribution,
        ).astype(np.float32)
    )
    with pytest.raises(ValueError, match="finite"):
        v3_ig.write_ig_artifact(
            tmp_path,
            broken,
            code_ref=CODE_REF,
            **provenance_for(tmp_path),
            stratum=v3_strata.PAM_LIBRARY,
            subset_manifest=subset_for(tmp_path, v3_strata.PAM_LIBRARY),
        )


def test_write_ig_artifact_names_the_subset_manifest_it_addresses(ig_result, tmp_path):
    """AC7: an IG artifact says which frozen manifest its rows came from.

    Two IG artifacts are the same subset's iff their `membership_digest`
    agrees, which is the only check that survives the Colab session the run
    happened in.
    """
    manifest_path = subset_for(tmp_path, v3_strata.GENERALISATION)
    subset = v3_selection.load_ig_subset(manifest_path)
    path = v3_ig.write_ig_artifact(
        tmp_path,
        ig_result,
        code_ref=CODE_REF,
        **provenance_for(tmp_path),
        stratum=v3_strata.GENERALISATION,
        subset_manifest=manifest_path,
    )
    sidecar = json.loads((path.parent / "ig.sidecar.json").read_text(encoding="utf-8"))
    assert sidecar["subset"]["manifest"] == v3_selection.IG_SUBSET_NAME
    assert sidecar["subset"]["membership_digest"] == subset.digest
    assert sidecar["subset"]["seed"] == 7
    assert sidecar["subset"]["stratum_digest"] == v3_predict.population_digest(
        subset.rows_for(v3_strata.GENERALISATION)
    )


def test_write_ig_artifact_refuses_rows_the_manifest_does_not_hold(ig_result, tmp_path):
    # The run drifted off the frozen subset — one row renumbered. Caught at
    # the artifact, not left for whoever tries to join it later.
    with pytest.raises(ValueError, match="frozen subset"):
        v3_ig.write_ig_artifact(
            tmp_path,
            ig_result,
            code_ref=CODE_REF,
            **provenance_for(tmp_path),
            stratum=v3_strata.GENERALISATION,
            subset_manifest=subset_for(
                tmp_path, v3_strata.GENERALISATION, rows=(7, 11, 24)
            ),
        )


def test_write_ig_artifact_refuses_a_different_strata_rows(ig_result, tmp_path):
    # The manifest holds both strata; writing a generalisation run against the
    # library stratum's rows is a mis-join that reads as a valid artifact.
    manifest_path = subset_for(tmp_path, v3_strata.PAM_LIBRARY)
    with pytest.raises(ValueError, match="frozen subset"):
        v3_ig.write_ig_artifact(
            tmp_path,
            ig_result,
            code_ref=CODE_REF,
            **provenance_for(tmp_path),
            stratum=v3_strata.GENERALISATION,
            subset_manifest=manifest_path,
        )


# --- the combined 400-row artifact (#133) ----------------------------------


def combined_subset_for(out_dir, rows=((7, 11), (23,))):
    """A stand-in manifest holding both strata, like the real 400-row one.

    The real manifest is 200 generalisation rows and 200 library rows under
    seed 129; what matters here is that one manifest holds both, so "the whole
    subset" and "one stratum's half" are different row sets and the writer has
    to tell them apart.
    """
    generalisation, library = rows
    entries = pd.DataFrame(
        {
            "row_index": [*generalisation, *library],
            "stratum": [
                *[v3_strata.GENERALISATION] * len(generalisation),
                *[v3_strata.PAM_LIBRARY] * len(library),
            ],
            "cell": "errq1",
            "selection_order": range(1, len(generalisation) + len(library) + 1),
            "error_quartile": 1,
            "is_ngg": True,
            "protospacer": "A" * 20,
        }
    )
    manifest = v3_selection.SubsetManifest(
        entries=entries, seed=129, rows_per_stratum=len(generalisation)
    )
    return v3_selection.write_ig_subset(out_dir / "combined", manifest)


def test_write_ig_artifact_writes_the_whole_subset_as_one_archive(ig_result, tmp_path):
    """#133: one `(rows, 3, 5)` archive over both strata, not one per stratum.

    `ALL_STRATA` is the default, so the driver's single call is also the
    documented shape: two half-artifacts would carry two sidecars that can
    only agree, and leave every downstream join to concatenate them in an
    order nothing pins.
    """
    manifest_path = combined_subset_for(tmp_path)
    subset = v3_selection.load_ig_subset(manifest_path)
    path = v3_ig.write_ig_artifact(
        tmp_path,
        ig_result,
        code_ref=CODE_REF,
        **provenance_for(tmp_path),
        subset_manifest=manifest_path,
    )
    with np.load(path) as archive:
        assert archive["attribution"].shape == (3, len(HEADS), TOKEN_COUNT)
        assert archive["row_index"].tolist() == [7, 11, 23]

    sidecar = json.loads((path.parent / "ig.sidecar.json").read_text(encoding="utf-8"))
    assert sidecar["stratum"] == v3_ig.ALL_STRATA == "all"
    assert sidecar["strata"] == list(v3_strata.STRATUM_NAMES)
    assert sidecar["rows"] == 3
    # The combined write addresses the whole manifest, so its own rows and the
    # manifest's are the same set and the two digests coincide. That equality
    # is what distinguishes a full-subset artifact from one stratum's half.
    assert sidecar["subset"]["membership_digest"] == subset.digest
    assert sidecar["subset"]["stratum_digest"] == subset.digest


def test_the_combined_write_refuses_one_stratums_rows(ig_result, tmp_path):
    # Half the subset, written as if it were all of it: a 200-row archive that
    # reads as the 400-row one is exactly the mis-join AC7 exists to stop.
    half = combined_subset_for(tmp_path, rows=((7, 11, 23), ()))
    manifest = v3_selection.load_ig_subset(half)
    assert manifest.rows_for(v3_strata.PAM_LIBRARY).size == 0
    with pytest.raises(ValueError, match="frozen subset"):
        v3_ig.write_ig_artifact(
            tmp_path,
            ig_result,
            code_ref=CODE_REF,
            **provenance_for(tmp_path),
            stratum=v3_strata.PAM_LIBRARY,
            subset_manifest=half,
        )


def test_a_per_stratum_write_refuses_the_whole_subsets_rows(ig_result, tmp_path):
    # The run covered both strata; claiming it is one stratum's artifact would
    # mislabel 400 rows as 200.
    with pytest.raises(ValueError, match="frozen subset"):
        v3_ig.write_ig_artifact(
            tmp_path,
            ig_result,
            code_ref=CODE_REF,
            **provenance_for(tmp_path),
            stratum=v3_strata.GENERALISATION,
            subset_manifest=combined_subset_for(tmp_path),
        )


def test_the_sidecar_pins_the_code_ref_the_split_and_the_versions(ig_result, tmp_path):
    """AC11: the driver's two pins land in the sidecar, not just in the cell.

    ISM (#132) and IG (#133) are separate Colab sessions, so their
    `transformers` versions can differ — which is why each analysis carries
    its own sidecar rather than sharing one.
    """
    provenance = provenance_for(tmp_path)
    path = v3_ig.write_ig_artifact(
        tmp_path,
        ig_result,
        code_ref=CODE_REF,
        **provenance,
        subset_manifest=combined_subset_for(tmp_path),
    )
    sidecar = json.loads((path.parent / "ig.sidecar.json").read_text(encoding="utf-8"))
    assert sidecar["code_ref"] == CODE_REF
    assert sidecar["versions"]["transformers"] == transformers.__version__
    assert sidecar["versions"]["torch"] == torch.__version__
    assert sidecar["split"]["sha256"] == v3_predict.sha256_of(provenance["split_csv"])
    assert sidecar["nan_policy"] == "forbidden"


@pytest.mark.parametrize("missing", [None, "", "model_analysis", "0" * 39, "0" * 41])
def test_write_ig_artifact_refuses_an_unpinned_code_ref(ig_result, tmp_path, missing):
    # A branch name moves, so it cannot address the code a run was made with.
    with pytest.raises(ValueError, match="CODE_REF"):
        v3_ig.write_ig_artifact(
            tmp_path,
            ig_result,
            code_ref=missing,
            **provenance_for(tmp_path),
            subset_manifest=combined_subset_for(tmp_path),
        )


def test_run_ig_reports_progress_per_batch(toy_model, tokenizer):
    """The 400-row run is one call with nothing on disk until it ends.

    A Colab session that prints nothing for 50 batches cannot be told apart
    from a stalled one, so the batch loop reports instead of the driver
    guessing.
    """
    seen = []
    v3_ig.run_ig(
        toy_model,
        tokenizer,
        SEQUENCES,
        row_index=[7, 11, 23],
        heads=HEADS,
        steps=2,
        batch_size=2,
        progress=seen.append,
    )
    assert seen == [
        {"rows_done": 2, "rows_total": 3},
        {"rows_done": 3, "rows_total": 3},
    ]
