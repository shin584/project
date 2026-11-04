"""Tests for v3's in-silico saturation mutagenesis sweep (#132).

Three groups, failing for different reasons:

- **Layout** — pure string and numpy work: 91 variants per row, ACGT in fixed
  order at every position, and the reference base's slot exactly 0. These pin
  the `(n, 30, 4, 9)` schema the design chose over SaCas9's alt-only
  `(n, 36, 3)`, and they need neither torch nor a checkpoint.
- **The sweep** — `run_ism` against an injected `forward_fn`, so batching,
  sharding, resume and the de-standardisation are tested exactly, with
  arithmetic a reader can check by hand.
- **The toy model** — a small randomly-initialised 9-output ESM built inside
  the test, reached through the same `v3_predict.model_forward_fn` the Colab
  driver uses. No download, no checkpoint, no network.

The artifact tests carry AC11: a sidecar without `code_ref` or without a
pinned `transformers` version is refused rather than written, because version
drift is what silently zeroed Model B's IG in the SaCas9 pipeline.
"""

import json

import numpy as np
import pytest
import v3_ism
import v3_predict
import v3_strata
from v3_ism import (
    BASES,
    POSITIONS,
    VARIANTS_PER_ROW,
)
from v3_predict import HEADS

CODE_REF = "0" * 40

SEQUENCES = [
    "ACGTACGTACGTACGTACGTACGTAGGTCA",
    "TTGGCATCAGATCCGATTACAAGGTTACGC",
    "GGGCCCAAATTTGGGCCCAAATTTAGGCCA",
]
ROW_INDEX = [5, 11, 30]

MU = np.arange(len(HEADS), dtype=np.float64) / 10 + 0.3
SD = np.full(len(HEADS), 0.25, dtype=np.float64)


# --- an injectable forward pass -------------------------------------------


def linear_forward_fn(record=None):
    """A deterministic stand-in for v3: one weight per (position, base, head).

    Linear in the one-hot sequence, so every ISM delta has a closed form
    (`weight[position, mutant] - weight[position, reference]`) and the test can
    assert the sweep's arithmetic rather than merely its shape.
    """
    generator = np.random.default_rng(0)
    weights = generator.normal(size=(POSITIONS, len(BASES), len(HEADS)))

    def forward_fn(sequences):
        sequences = list(sequences)
        if record is not None:
            record.append(len(sequences))
        one_hot = np.zeros((len(sequences), POSITIONS, len(BASES)))
        for row, sequence in enumerate(sequences):
            for position, base in enumerate(sequence):
                one_hot[row, position, BASES.index(base)] = 1.0
        return np.einsum("rpb,pbh->rh", one_hot, weights)

    forward_fn.weights = weights
    return forward_fn


def run(forward_fn, *, mu=MU, sd=SD, **kwargs):
    return v3_ism.run_ism(forward_fn, SEQUENCES, ROW_INDEX, mu=mu, sd=sd, **kwargs)


@pytest.fixture
def result():
    return run(linear_forward_fn())


# --- layout ----------------------------------------------------------------


def test_bases_are_acgt_in_a_fixed_order():
    # The axis means the same thing in every row, which is the whole reason
    # the schema is (n, 30, 4, 9) rather than SaCas9's alt-only (n, 36, 3).
    assert BASES == ("A", "C", "G", "T")
    assert v3_ism.BASE_INDEX == {"A": 0, "C": 1, "G": 2, "T": 3}


def test_a_row_costs_one_wild_type_plus_ninety_mutants():
    assert POSITIONS == v3_predict.SEQUENCE_LENGTH == 30
    assert VARIANTS_PER_ROW == 1 + POSITIONS * (len(BASES) - 1) == 91


def test_substitution_jobs_are_position_major_and_skip_the_reference():
    jobs = v3_ism.substitution_jobs(SEQUENCES[0])
    assert len(jobs) == POSITIONS * (len(BASES) - 1)
    assert jobs[:3] == ((0, 1), (0, 2), (0, 3))  # reference A at position 0
    positions = [position for position, _ in jobs]
    assert positions == sorted(positions)
    for position, base in jobs:
        assert BASES[base] != SEQUENCES[0][position]
        assert jobs.count((position, base)) == 1


def test_variants_put_the_wild_type_first_and_change_exactly_one_base():
    variants = v3_ism.variants(SEQUENCES[1])
    assert len(variants) == VARIANTS_PER_ROW
    assert variants[0] == SEQUENCES[1]
    assert len(set(variants)) == VARIANTS_PER_ROW
    for variant in variants[1:]:
        differing = [
            position
            for position, (left, right) in enumerate(zip(variant, SEQUENCES[1]))
            if left != right
        ]
        assert len(differing) == 1
        assert len(variant) == POSITIONS


def test_the_reference_slot_is_exactly_zero_and_the_others_are_the_delta():
    predictions = np.arange(VARIANTS_PER_ROW * len(HEADS), dtype=np.float64).reshape(
        VARIANTS_PER_ROW, len(HEADS)
    )
    delta = v3_ism.delta_block(SEQUENCES[0], predictions)
    assert delta.shape == (POSITIONS, len(BASES), len(HEADS))
    for position, base in enumerate(SEQUENCES[0]):
        # A true zero, not a sentinel: substituting the reference base changes
        # nothing, so the delta there is 0 by definition.
        assert np.all(delta[position, BASES.index(base)] == 0.0)
    for order, (position, base) in enumerate(v3_ism.substitution_jobs(SEQUENCES[0])):
        np.testing.assert_array_equal(
            delta[position, base], predictions[1 + order] - predictions[0]
        )


def test_the_base_axis_means_the_same_thing_whatever_the_reference_is():
    """SaCas9's alt-only layout is what this rules out.

    There, "slot 0" is a different base at each position, so indexing a PAM
    base directly is impossible — which is exactly what ADR 0007 hands ISM.
    """
    left, right = "A" + SEQUENCES[0][1:], "T" + SEQUENCES[0][1:]
    predictions = np.ones((VARIANTS_PER_ROW, len(HEADS)))
    predictions[1:] = 2.0
    for base in BASES:
        slot = BASES.index(base)
        for sequence in (left, right):
            delta = v3_ism.delta_block(sequence, predictions)
            expected = 0.0 if sequence[0] == base else 1.0
            assert np.all(delta[0, slot] == expected)


def test_delta_block_rejects_a_prediction_block_of_the_wrong_size():
    with pytest.raises(ValueError, match="91"):
        v3_ism.delta_block(SEQUENCES[0], np.zeros((90, len(HEADS))))


# --- the sweep -------------------------------------------------------------


def test_run_ism_has_the_designed_shapes_and_dtypes(result):
    assert result.delta.shape == (len(SEQUENCES), POSITIONS, len(BASES), len(HEADS))
    assert result.delta.dtype == np.float32
    assert result.wt_prediction.shape == (len(SEQUENCES), len(HEADS))
    assert result.wt_prediction.dtype == np.float32
    assert result.reference_base.shape == (len(SEQUENCES), POSITIONS)
    assert result.row_index.tolist() == ROW_INDEX
    assert result.row_index.dtype == np.int64
    assert result.heads == HEADS
    assert np.isfinite(result.delta).all()


def test_every_one_of_the_nine_heads_is_swept(result):
    # One forward returns all nine, so storing three would only force a re-run.
    assert len(result.heads) == 9
    assert set(v3_strata.ANALYSED_HEADS) < set(result.heads)
    assert np.all(np.abs(result.delta).sum(axis=(0, 1, 2)) > 0)


def test_the_delta_is_the_de_standardised_difference(result):
    forward_fn = linear_forward_fn()
    expected = run(forward_fn)
    weights = forward_fn.weights
    for row, sequence in enumerate(SEQUENCES):
        for position, base in enumerate(sequence):
            reference = weights[position, BASES.index(base)]
            for slot in range(len(BASES)):
                np.testing.assert_allclose(
                    expected.delta[row, position, slot],
                    (weights[position, slot] - reference) * SD,
                    rtol=1e-5,
                    atol=1e-7,
                )


def test_the_delta_ignores_mu_while_the_wild_type_prediction_does_not():
    """`mu` cancels in a difference; it does not cancel in `f(x)` itself.

    The sweep needs both anyway: `wt_prediction` is what lets a consumer join
    this artifact against `predictions.csv` and catch a mis-join, and that
    check is only meaningful on the same fraction scale.
    """
    forward_fn = linear_forward_fn()
    shifted = run(forward_fn, mu=MU + 7.0)
    plain = run(forward_fn)
    np.testing.assert_array_equal(shifted.delta, plain.delta)
    np.testing.assert_allclose(
        shifted.wt_prediction, plain.wt_prediction + 7.0, rtol=1e-5, atol=1e-5
    )


def test_the_wild_type_prediction_is_on_the_fraction_scale():
    forward_fn = linear_forward_fn()
    swept = run(forward_fn)
    direct = np.asarray(forward_fn(SEQUENCES)) * SD + MU
    np.testing.assert_allclose(swept.wt_prediction, direct, rtol=1e-5, atol=1e-6)


def test_the_batch_size_bounds_every_forward_call_without_changing_the_result():
    sizes: list[int] = []
    batched = run(linear_forward_fn(sizes), batch_size=16)
    assert max(sizes) <= 16
    assert sum(sizes) == len(SEQUENCES) * VARIANTS_PER_ROW
    np.testing.assert_array_equal(
        batched.delta, run(linear_forward_fn(), batch_size=VARIANTS_PER_ROW * 3).delta
    )
    assert batched.batch_size == 16


def test_the_shard_size_does_not_change_the_result(result):
    sharded = run(linear_forward_fn(), shard_rows=1)
    np.testing.assert_array_equal(sharded.delta, result.delta)
    np.testing.assert_array_equal(sharded.wt_prediction, result.wt_prediction)


def test_shard_bounds_tile_the_population():
    assert v3_ism.shard_bounds(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert v3_ism.shard_bounds(4, 4) == [(0, 4)]
    assert v3_ism.shard_bounds(0, 4) == []
    with pytest.raises(ValueError, match="shard_rows"):
        v3_ism.shard_bounds(4, 0)


def test_run_ism_requires_a_strictly_ascending_row_index():
    """Implicit positional identity is how a 3,851-row and a 400-row artifact
    silently mis-join, so the ordering is a precondition, not a fix-up."""
    with pytest.raises(ValueError, match="ascending"):
        v3_ism.run_ism(linear_forward_fn(), SEQUENCES, [30, 5, 11], mu=MU, sd=SD)
    with pytest.raises(ValueError, match="ascending"):
        v3_ism.run_ism(linear_forward_fn(), SEQUENCES, [5, 5, 30], mu=MU, sd=SD)


def test_run_ism_rejects_a_mismatched_row_index_length():
    with pytest.raises(ValueError, match="row_index"):
        v3_ism.run_ism(linear_forward_fn(), SEQUENCES, [5, 11], mu=MU, sd=SD)


def test_run_ism_rejects_a_sequence_it_cannot_sweep():
    with pytest.raises(ValueError, match="length"):
        v3_ism.run_ism(linear_forward_fn(), ["ACGT" * 8], [0], mu=MU, sd=SD)
    with pytest.raises(ValueError, match="non-ACGT"):
        v3_ism.run_ism(linear_forward_fn(), ["N" + SEQUENCES[0][1:]], [0], mu=MU, sd=SD)


def test_run_ism_rejects_a_mu_sd_of_the_wrong_width():
    with pytest.raises(ValueError, match="mu"):
        run(linear_forward_fn(), mu=MU[:3])
    with pytest.raises(ValueError, match="sd"):
        run(linear_forward_fn(), sd=SD[:3])


def test_run_ism_rejects_a_non_positive_sd():
    with pytest.raises(ValueError, match="sd"):
        run(linear_forward_fn(), sd=np.zeros(len(HEADS)))


def test_a_non_finite_forward_fails_at_the_batch_not_at_the_write():
    def broken(sequences):
        out = np.ones((len(sequences), len(HEADS)))
        out[0, 0] = np.nan
        return out

    with pytest.raises(ValueError, match="finite"):
        run(broken)


def test_a_forward_of_the_wrong_width_is_rejected():
    with pytest.raises(ValueError, match="9"):
        run(lambda sequences: np.zeros((len(sequences), 3)))


# --- resume ----------------------------------------------------------------


def test_a_second_run_reuses_the_shards_and_calls_no_forward(tmp_path):
    """350K forward passes must not be paid twice for one disconnect."""
    sizes: list[int] = []
    first = run(linear_forward_fn(sizes), shard_rows=1, checkpoint_dir=tmp_path)
    assert sizes and first.reused_shards == ()
    assert len(list(tmp_path.glob("ism_rows_*.npz"))) == len(SEQUENCES)

    again: list[int] = []
    second = run(linear_forward_fn(again), shard_rows=1, checkpoint_dir=tmp_path)
    assert again == []
    assert second.reused_shards == (0, 1, 2)
    np.testing.assert_array_equal(second.delta, first.delta)
    np.testing.assert_array_equal(second.wt_prediction, first.wt_prediction)
    np.testing.assert_array_equal(second.reference_base, first.reference_base)


def test_a_partial_run_resumes_from_where_it_stopped(tmp_path):
    whole = run(linear_forward_fn(), shard_rows=1, checkpoint_dir=tmp_path)
    for stale in sorted(tmp_path.glob("ism_rows_*.npz"))[1:]:
        stale.unlink()

    sizes: list[int] = []
    resumed = run(linear_forward_fn(sizes), shard_rows=1, checkpoint_dir=tmp_path)
    assert resumed.reused_shards == (0,)
    assert sum(sizes) == (len(SEQUENCES) - 1) * VARIANTS_PER_ROW
    np.testing.assert_array_equal(resumed.delta, whole.delta)


def test_a_shard_from_a_different_population_is_refused(tmp_path):
    run(linear_forward_fn(), shard_rows=1, checkpoint_dir=tmp_path)
    with pytest.raises(ValueError, match="different population"):
        v3_ism.run_ism(
            linear_forward_fn(),
            SEQUENCES,
            [6, 11, 30],
            mu=MU,
            sd=SD,
            shard_rows=1,
            checkpoint_dir=tmp_path,
        )


def test_a_shard_whose_sequences_changed_is_refused(tmp_path):
    """A reused shard has to match the sequences, not just their row ids.

    An edited split keeps its `row_index` column, so row ids alone would let a
    stale shard contaminate the sweep silently.
    """
    run(linear_forward_fn(), shard_rows=1, checkpoint_dir=tmp_path)
    changed = ["T" + SEQUENCES[0][1:], *SEQUENCES[1:]]
    with pytest.raises(ValueError, match="sequences"):
        v3_ism.run_ism(
            linear_forward_fn(),
            changed,
            ROW_INDEX,
            mu=MU,
            sd=SD,
            shard_rows=1,
            checkpoint_dir=tmp_path,
        )


def test_a_shard_appears_under_its_final_name_only_once_complete(tmp_path, monkeypatch):
    """A session killed mid-write must not leave a loadable-looking shard.

    The next run reads whatever sits at the shard's name, so a truncated
    archive there would fail the sweep on a corrupt file instead of simply
    recomputing that shard.
    """
    real = np.savez

    def die(handle, **arrays):
        real(handle, **arrays)
        raise KeyboardInterrupt("session killed mid-write")

    monkeypatch.setattr(np, "savez", die)
    with pytest.raises(KeyboardInterrupt):
        run(linear_forward_fn(), shard_rows=1, checkpoint_dir=tmp_path)
    assert list(tmp_path.glob("ism_rows_*.npz")) == []

    monkeypatch.undo()
    for leftover in tmp_path.glob("*.partial"):
        leftover.unlink()
    assert (
        run(linear_forward_fn(), shard_rows=1, checkpoint_dir=tmp_path).reused_shards
        == ()
    )


def test_progress_is_reported_per_shard(tmp_path):
    seen: list[dict] = []
    run(
        linear_forward_fn(),
        shard_rows=2,
        checkpoint_dir=tmp_path,
        progress=seen.append,
    )
    assert [entry["rows_done"] for entry in seen] == [2, 3]
    assert [entry["reused"] for entry in seen] == [False, False]
    assert all(entry["rows_total"] == len(SEQUENCES) for entry in seen)


# --- artifact --------------------------------------------------------------


def write(result, tmp_path, **kwargs):
    # AC11: the writer refuses a sidecar whose `torch` / `transformers`
    # versions it cannot record, so the write path needs both importable —
    # unlike the layout, sweep and resume tests above, which need neither.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    kwargs.setdefault("code_ref", CODE_REF)
    return v3_ism.write_ism_artifact(
        tmp_path, result, **provenance_for(tmp_path), **kwargs
    )


def provenance_for(tmp_path):
    """The split and checkpoint every sidecar has to name (AC11)."""
    checkpoint = tmp_path / "provenance" / "ckpt"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text(
        '{"problem_type": "regression"}', encoding="utf-8"
    )
    split = tmp_path / "provenance" / "test.csv"
    split.write_text("row_index,sequence", encoding="utf-8")
    return {"split_csv": split, "checkpoint_dir": checkpoint}


def sidecar_of(path):
    return json.loads((path.parent / "ism.sidecar.json").read_text(encoding="utf-8"))


def test_write_ism_artifact_round_trips(result, tmp_path):
    path = write(result, tmp_path)
    assert path.name == "ism.npz"
    with np.load(path) as archive:
        assert set(archive.files) == {
            "delta",
            "wt_prediction",
            "reference_base",
            "row_index",
        }
        assert archive["delta"].dtype == np.float32
        assert archive["delta"].shape == result.delta.shape
        assert archive["row_index"].dtype == np.int64
        assert archive["row_index"].tolist() == ROW_INDEX
        np.testing.assert_array_equal(archive["delta"], result.delta)
        np.testing.assert_array_equal(archive["wt_prediction"], result.wt_prediction)

        # The zero is identifiable as the reference base's, from the artifact
        # alone — ADR 0007 asks ISM for per-base PAM numbers, and "0" has to be
        # distinguishable from "no effect".
        reference = archive["reference_base"].astype(np.int64)
        taken = np.take_along_axis(
            archive["delta"], reference[:, :, None, None], axis=2
        )
        assert np.all(taken == 0.0)


def test_the_sidecar_records_every_head_and_names_the_six_unanalysed(result, tmp_path):
    sidecar = sidecar_of(write(result, tmp_path))
    assert sidecar["heads"] == list(HEADS)
    assert sidecar["analysed_heads"] == list(v3_strata.ANALYSED_HEADS)
    assert sidecar["stored_not_analysed"] == [
        head for head in HEADS if head not in v3_strata.ANALYSED_HEADS
    ]
    assert len(sidecar["stored_not_analysed"]) == 6
    assert "not analysed" in sidecar["stored_not_analysed_note"]


def test_the_sidecar_pins_the_code_ref_and_the_library_versions(result, tmp_path):
    sidecar = sidecar_of(write(result, tmp_path))
    assert sidecar["code_ref"] == CODE_REF
    assert sidecar["versions"]["transformers"]
    assert sidecar["versions"]["torch"]
    assert sidecar["batch_size"] == result.batch_size
    assert sidecar["shard_rows"] == result.shard_rows
    assert sidecar["rows"] == len(SEQUENCES)
    assert sidecar["forwards"] == len(SEQUENCES) * VARIANTS_PER_ROW
    assert sidecar["population"] == v3_ism.POPULATION
    assert sidecar["delta_scale"] == "fraction"
    assert sidecar["bases"] == list(BASES)
    assert sidecar["population_digest"] == v3_predict.population_digest(ROW_INDEX)
    assert sidecar["delta_sha256"] == v3_predict.sha256_of(tmp_path / "ism.npz")


def test_the_sidecar_records_the_consumed_split_and_checkpoint(result, tmp_path):
    """AC11 asks for both digests, so neither is optional on the write path."""
    provenance = provenance_for(tmp_path)
    sidecar = sidecar_of(write(result, tmp_path))
    assert sidecar["split"]["sha256"] == v3_predict.sha256_of(provenance["split_csv"])
    assert sidecar["checkpoint"]["files"]["config.json"]
    with pytest.raises(TypeError):
        v3_ism.write_ism_artifact(tmp_path, result, code_ref=CODE_REF)


def test_write_ism_artifact_refuses_an_unpinned_code_ref(result, tmp_path):
    # Not a sha at all, a branch name, a short sha, and 40 characters that are
    # not lowercase hex: a pin that cannot address a commit is not a pin.
    for missing in ("", None, "model_analysis", "cbdbc54", "z" * 40, "A" * 40):
        with pytest.raises(ValueError, match="CODE_REF"):
            v3_ism.write_ism_artifact(
                tmp_path, result, code_ref=missing, **provenance_for(tmp_path)
            )


def test_write_ism_artifact_refuses_an_unrecorded_transformers_version(
    result, tmp_path, monkeypatch
):
    """AC11: version drift is the mechanism, so a null version is not writable."""
    monkeypatch.setattr(
        v3_ism, "library_versions", lambda: {"torch": "2.12.0", "transformers": None}
    )
    with pytest.raises(ValueError, match="transformers"):
        write(result, tmp_path)


def test_write_ism_artifact_refuses_a_non_finite_delta(result, tmp_path):
    delta = result.delta.copy()
    # In a slot the reference check does not own, so this test fails on the
    # finiteness guard and not on "the reference slot must be 0".
    alternative = (result.reference_base[0, 0] + 1) % len(BASES)
    delta[0, 0, alternative, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        write(result.replace(delta=delta), tmp_path)


def test_write_ism_artifact_refuses_a_nan_in_the_reference_slot(result, tmp_path):
    delta = result.delta.copy()
    delta[0, 0, result.reference_base[0, 0], 0] = np.nan
    with pytest.raises(ValueError):
        write(result.replace(delta=delta), tmp_path)


def test_write_ism_artifact_refuses_a_non_zero_reference_slot(result, tmp_path):
    delta = result.delta.copy()
    delta[0, 0, result.reference_base[0, 0]] = 0.5
    with pytest.raises(ValueError, match="reference"):
        write(result.replace(delta=delta), tmp_path)


def test_write_ism_artifact_refuses_a_non_ascending_row_index(result, tmp_path):
    broken = result.replace(row_index=np.array([30, 11, 5], dtype=np.int64))
    with pytest.raises(ValueError, match="ascending"):
        write(broken, tmp_path)


# --- the toy model ---------------------------------------------------------
#
# `importorskip` lives inside these fixtures rather than at module scope. At
# module scope it raises on import and skips the whole file, including the ~45
# tests above that need no torch, no checkpoint and no GPU — which is the
# coverage AC12 asks for, so it must not vanish with an optional library.

CHECKPOINT_DIR = v3_ism.SUBTREE_ROOT / "result"


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))


@pytest.fixture(scope="module")
def toy_model():
    """A small randomly-initialised 9-output ESM, same shape as v3."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
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
        num_labels=len(HEADS),
        problem_type=real["problem_type"],
    )
    torch.manual_seed(42)
    return EsmForSequenceClassification(config).eval()


@pytest.fixture(scope="module")
def toy_forward_fn(toy_model, tokenizer):
    return v3_predict.model_forward_fn(toy_model, tokenizer, batch_size=32)


def test_the_toy_sweep_matches_a_direct_forward(toy_forward_fn):
    swept = v3_ism.run_ism(
        toy_forward_fn, SEQUENCES[:1], [5], mu=MU, sd=SD, batch_size=32
    )
    direct = np.asarray(toy_forward_fn(v3_ism.variants(SEQUENCES[0])))
    expected = v3_ism.delta_block(SEQUENCES[0], (direct - direct[0]) * SD)
    np.testing.assert_allclose(swept.delta[0], expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        swept.wt_prediction[0], direct[0] * SD + MU, rtol=1e-4, atol=1e-6
    )


def test_the_toy_sweep_is_indifferent_to_the_batch_size(
    toy_forward_fn, toy_model, tokenizer
):
    one_shot = v3_ism.run_ism(
        toy_forward_fn, SEQUENCES, ROW_INDEX, mu=MU, sd=SD, batch_size=VARIANTS_PER_ROW
    )
    trickle = v3_ism.run_ism(
        v3_predict.model_forward_fn(toy_model, tokenizer, batch_size=7),
        SEQUENCES,
        ROW_INDEX,
        mu=MU,
        sd=SD,
        batch_size=7,
    )
    np.testing.assert_allclose(trickle.delta, one_shot.delta, rtol=1e-4, atol=1e-6)


def test_the_pam_is_resolved_base_by_base(toy_forward_fn):
    """ADR 0007's half of the bargain.

    IG's last 6-mer token merges the PAM (24-26) with the right flank (27-29).
    A single-base substitution is indifferent to that boundary, so ISM gives
    each PAM base its own three alternatives — and this asserts they are
    actually populated rather than merely shaped.
    """
    swept = v3_ism.run_ism(
        toy_forward_fn, SEQUENCES, ROW_INDEX, mu=MU, sd=SD, batch_size=64
    )
    pam = swept.delta[:, v3_strata.PAM]
    assert pam.shape == (len(SEQUENCES), 3, len(BASES), len(HEADS))
    non_zero = np.count_nonzero(np.abs(pam).sum(axis=-1), axis=-1)
    # Three alternatives populated at each PAM base, the reference left at 0.
    assert np.all(non_zero == len(BASES) - 1)


def test_the_toy_sweep_writes_a_complete_artifact(toy_forward_fn, tmp_path):
    swept = v3_ism.run_ism(
        toy_forward_fn,
        SEQUENCES,
        ROW_INDEX,
        mu=MU,
        sd=SD,
        batch_size=64,
        shard_rows=2,
        checkpoint_dir=tmp_path / "shards",
    )
    path = write(swept, tmp_path / "out")
    with np.load(path) as archive:
        assert archive["delta"].shape == (3, POSITIONS, len(BASES), len(HEADS))
        assert np.isfinite(archive["delta"]).all()
    assert sidecar_of(path)["shard_rows"] == 2
