"""Tests for v3's attention rollout (#136).

Three groups, failing for different reasons:

- **Rollout arithmetic** — pure numpy against synthetic attention. These pin
  the operation *order* (`0.5*A + 0.5*I` first, row-normalise second), the
  cross-layer left-multiplication, and the head averaging. No torch, no
  checkpoint, no GPU.
- **Shape contract (ADR 0008, ADR 0007)** — the stored array is `(rows, 5, 5)`
  with **no head axis** and no 30x30 unfold, and the writer refuses both.
- **Rollout implementation** — the same small randomly-initialised 9-output ESM
  `test_v3_ig.py` builds, run through `run_rollout` with `output_attentions`.

The padding test is the one that would otherwise fail silently: if the
attention mask ever stopped reaching the encoder, the five sequence tokens
would distribute part of their mass over 26 padding positions and the stored
5x5 block would quietly lose it.
"""

import json

import numpy as np
import pytest
import v3_ig
import v3_predict
import v3_rollout

# The token contract is defined in `v3_ig` and imported by `v3_rollout` rather
# than restated there, so the tests take it from where it is defined.
from v3_ig import (
    CLS_TOKEN_COUNT,
    SEQUENCE_TOKEN_SLICE,
    TOKEN_COUNT,
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

CHECKPOINT_DIR = v3_rollout.SUBTREE_ROOT / "result"

SEQUENCES = [
    "ACGTACGTACGTACGTACGTACGTAGGTCA",
    "TTGGCATCAGATCCGATTACAAGGTTACGC",
    "GGGCCCAAATTTGGGCCCAAATTTAGGCCA",
]
ROW_INDEX = [7, 11, 23]

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

    Two layers and two attention heads rather than 24 and 20: the rollout is a
    product over layers and a mean over attention heads, so both axes only have
    to be >1 for the arithmetic under test to be exercised.
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
def rollout_result(toy_model, tokenizer):
    return v3_rollout.run_rollout(
        toy_model, tokenizer, SEQUENCES, row_index=ROW_INDEX, batch_size=2
    )


def provenance_for(tmp_path):
    """The split every sidecar has to name (AC11), as a file on disk."""
    split = tmp_path / "provenance" / "test.csv"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("row_index,sequence", encoding="utf-8")
    return {"split_csv": split}


def write_for(tmp_path, result, **kwargs):
    """Write `result` with the provenance the writer demands (AC11)."""
    return v3_rollout.write_rollout_artifact(
        tmp_path / "v3_rollout",
        result,
        code_ref=kwargs.pop("code_ref", CODE_REF),
        checkpoint_dir=CHECKPOINT_DIR,
        **provenance_for(tmp_path),
        **kwargs,
    )


def sidecar_of(path):
    return json.loads(
        (path.parent / "rollout.sidecar.json").read_text(encoding="utf-8")
    )


def row_stochastic(rows: int, size: int, *, seed: int) -> np.ndarray:
    """A `(rows, size, size)` stack of row-stochastic matrices."""
    generator = np.random.default_rng(seed)
    raw = generator.random((rows, size, size))
    return raw / raw.sum(axis=-1, keepdims=True)


# --- rollout arithmetic ----------------------------------------------------


def test_layer_rollout_rows_sum_to_one():
    attention = row_stochastic(2, 6, seed=1)
    rolled = v3_rollout.layer_rollout(attention)
    assert rolled.shape == attention.shape
    assert np.allclose(rolled.sum(axis=-1), 1.0)


def test_layer_rollout_adds_the_identity_before_normalising():
    # On an already row-stochastic A the two orders coincide — 0.5*A + 0.5*I is
    # already row-stochastic, so normalising it divides by 1 — which is exactly
    # why the order has to be pinned on an *unnormalised* A, where they differ.
    # Real attention probabilities are normalised; a future refactor that
    # reorders the two operations would not be caught by the stochastic case.
    attention = row_stochastic(1, 4, seed=2)
    assert np.allclose(
        v3_rollout.layer_rollout(attention), 0.5 * attention + 0.5 * np.eye(4)
    )

    unnormalised = np.array([[[2.0, 6.0], [3.0, 1.0]]])
    add_then_normalise = v3_rollout.layer_rollout(unnormalised)
    normalise_then_add = 0.5 * (
        unnormalised / unnormalised.sum(axis=-1, keepdims=True)
    ) + 0.5 * np.eye(2)
    # 0.5*A + 0.5*I = [[1.5, 3.0], [1.5, 1.0]], row sums 4.5 and 2.5.
    assert np.allclose(add_then_normalise, [[[1 / 3, 2 / 3], [0.6, 0.4]]])
    assert np.allclose(normalise_then_add, [[[0.625, 0.375], [0.375, 0.625]]])
    assert not np.allclose(add_then_normalise, normalise_then_add)


def test_layer_rollout_of_the_identity_is_the_identity():
    identity = np.stack([np.eye(5)])
    assert np.allclose(v3_rollout.layer_rollout(identity), identity)


def test_layer_rollout_refuses_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        v3_rollout.layer_rollout(np.ones((1, 3, 4)))


def test_rollout_averages_over_attention_heads():
    # Two attention heads, deliberately different: the rollout is a function of
    # their mean, so feeding the mean as a single head must give the same
    # matrix. This is the Abnar & Zuidema head average, and it is *also* why
    # there is no regression-head axis (ADR 0008) — see the shape tests.
    two_heads = row_stochastic(2, 6, seed=3).reshape(1, 2, 6, 6)
    averaged = two_heads.mean(axis=1, keepdims=True)
    assert np.allclose(
        v3_rollout.rollout_matrix([two_heads]),
        v3_rollout.rollout_matrix([averaged]),
    )


def test_rollout_left_multiplies_each_later_layer():
    first = row_stochastic(1, 4, seed=4)[:, None]
    second = row_stochastic(1, 4, seed=5)[:, None]
    expected = v3_rollout.layer_rollout(second[:, 0]) @ v3_rollout.layer_rollout(
        first[:, 0]
    )
    assert np.allclose(v3_rollout.rollout_matrix([first, second]), expected)
    # Order matters: the reversed product is a different matrix, which is what
    # makes "layers arrive first-to-last" part of the contract.
    assert not np.allclose(
        v3_rollout.rollout_matrix([second, first]),
        v3_rollout.rollout_matrix([first, second]),
    )


def test_rollout_of_one_layer_is_that_layer_rolled():
    only = row_stochastic(2, 6, seed=6)[:, None]
    assert np.allclose(
        v3_rollout.rollout_matrix([only]), v3_rollout.layer_rollout(only[:, 0])
    )


def test_rollout_rows_sum_to_one_before_the_block_is_taken():
    attentions = [row_stochastic(2, 8, seed=seed)[:, None] for seed in (7, 8, 9)]
    rolled = v3_rollout.rollout_matrix(attentions)
    assert np.allclose(rolled.sum(axis=-1), 1.0)


def test_rollout_refuses_an_empty_layer_stack():
    with pytest.raises(ValueError, match="at least one layer"):
        v3_rollout.rollout_matrix([])


# --- shape contract: ADR 0008 (no head axis), ADR 0007 (no unfold) ---------


def test_sequence_block_drops_cls_and_padding():
    full = row_stochastic(3, v3_predict.MAX_LENGTH, seed=10)
    block = v3_rollout.sequence_block(full)
    assert block.shape == (3, TOKEN_COUNT, TOKEN_COUNT) == (3, 5, 5)
    assert np.allclose(block, full[:, SEQUENCE_TOKEN_SLICE, SEQUENCE_TOKEN_SLICE])
    assert CLS_TOKEN_COUNT == 1
    assert SEQUENCE_TOKEN_SLICE == slice(1, 6)


def test_cls_mass_is_what_the_block_discards():
    full = row_stochastic(3, v3_predict.MAX_LENGTH, seed=11)
    mass = v3_rollout.cls_mass(full)
    assert mass.shape == (3, TOKEN_COUNT)
    assert np.allclose(mass, full[:, SEQUENCE_TOKEN_SLICE, 0])


def test_padding_guard_accepts_a_rollout_that_ignores_the_padding():
    full = np.zeros((1, v3_predict.MAX_LENGTH, v3_predict.MAX_LENGTH))
    # Every sequence token splits its mass between <cls> and the tokens.
    full[:, SEQUENCE_TOKEN_SLICE, : TOKEN_COUNT + 1] = 1.0 / (TOKEN_COUNT + 1)
    v3_rollout.assert_padding_is_unattended(full)


def test_padding_guard_refuses_mass_on_a_padding_column():
    full = np.zeros((1, v3_predict.MAX_LENGTH, v3_predict.MAX_LENGTH))
    full[:, SEQUENCE_TOKEN_SLICE, : TOKEN_COUNT + 1] = 1.0 / (TOKEN_COUNT + 1)
    full[0, 2, TOKEN_COUNT + 4] = 1e-3
    with pytest.raises(ValueError, match="padding"):
        v3_rollout.assert_padding_is_unattended(full)


def test_padding_guard_reports_the_offender_it_names():
    # The magnitude in the message has to belong to the coordinates in the
    # message: two violations of different sizes, and the worst one is the one
    # described. Reporting the global maximum beside the *first* offender's
    # coordinates would attribute 0.02 to the token holding 0.001.
    full = np.zeros((1, v3_predict.MAX_LENGTH, v3_predict.MAX_LENGTH))
    full[:, SEQUENCE_TOKEN_SLICE, : TOKEN_COUNT + 1] = 1.0 / (TOKEN_COUNT + 1)
    full[0, 1 + 0, TOKEN_COUNT + 4] = 1e-3
    full[0, 1 + 3, TOKEN_COUNT + 7] = 2e-2

    with pytest.raises(ValueError) as raised:
        v3_rollout.assert_padding_is_unattended(full)
    message = str(raised.value)
    assert "token 3" in message
    assert "0.02" in message
    assert "2 such positions" in message


def test_the_block_is_not_unfolded_to_base_coordinates():
    # ADR 0007: attention is a token-to-token quantity and the last token binds
    # the PAM to the right flank, so a 30x30 rollout would re-mix exactly what
    # that ADR separates. SaCas9 unfolded to 36x36; that is not inherited.
    assert not hasattr(v3_rollout, "project_rollout_to_nucleotides")
    assert v3_rollout.UNFOLD_TO_BASES is False
    assert TOKEN_COUNT == 5 != v3_predict.SEQUENCE_LENGTH


def test_token_names_come_from_the_ig_token_contract():
    assert v3_rollout.TOKEN_NAMES == v3_ig.TOKEN_NAMES
    assert v3_rollout.TOKEN_NAMES[-1] == "pam_plus_right_flank"
    assert all(name.lower() != "pam" for name in v3_rollout.TOKEN_NAMES)


# --- the implementation ----------------------------------------------------


def test_run_rollout_has_no_head_axis(rollout_result):
    # ADR 0008. Nine regression heads share one encoder trunk and the rollout is
    # a function of that trunk's attention alone, so there is one matrix per
    # row and the array is 3-dimensional.
    assert rollout_result.rollout.ndim == 3
    assert rollout_result.rollout.shape == (len(SEQUENCES), 5, 5)
    assert rollout_result.rollout.shape != (
        len(SEQUENCES),
        len(v3_predict.HEADS),
        5,
        5,
    )
    assert v3_rollout.HEAD_AXIS is False
    # Nor is the head count hiding anywhere in the shape.
    assert len(v3_predict.HEADS) not in rollout_result.rollout.shape


def test_run_rollout_is_finite_and_float32(rollout_result):
    assert rollout_result.rollout.dtype == np.float32
    assert np.isfinite(rollout_result.rollout).all()


def test_run_rollout_keeps_the_row_index_it_was_given(rollout_result):
    assert rollout_result.row_index.tolist() == ROW_INDEX
    assert rollout_result.row_index.dtype == np.int64


def test_run_rollout_row_mass_is_positive_and_at_most_one(rollout_result):
    mass = rollout_result.rollout.sum(axis=-1)
    assert (mass > 0).all()
    # <= 1 because the <cls> share is dropped after the rollout, not folded
    # back in: the block is not renormalised.
    assert (mass <= 1.0 + 1e-5).all()
    assert np.allclose(mass + rollout_result.cls_mass, 1.0, atol=1e-5)


def test_run_rollout_records_the_trunk_it_rolled(rollout_result, toy_model):
    assert rollout_result.attention_layers == toy_model.config.num_hidden_layers
    assert rollout_result.attention_heads == toy_model.config.num_attention_heads


def test_run_rollout_is_batch_size_invariant(toy_model, tokenizer):
    one = v3_rollout.run_rollout(
        toy_model, tokenizer, SEQUENCES, row_index=ROW_INDEX, batch_size=1
    )
    whole = v3_rollout.run_rollout(
        toy_model, tokenizer, SEQUENCES, row_index=ROW_INDEX, batch_size=len(SEQUENCES)
    )
    assert np.allclose(one.rollout, whole.rollout, atol=1e-5)


def test_run_rollout_reports_progress(toy_model, tokenizer):
    events = []
    v3_rollout.run_rollout(
        toy_model,
        tokenizer,
        SEQUENCES,
        row_index=ROW_INDEX,
        batch_size=2,
        progress=events.append,
    )
    assert [event["rows_done"] for event in events] == [2, 3]
    assert all(event["rows_total"] == 3 for event in events)


def test_run_rollout_forces_eager_attention_and_restores_it(toy_model, tokenizer):
    # A fused kernel (sdpa, the default on a freshly loaded checkpoint) never
    # forms the probability matrix, and transformers 5.x answers
    # `output_attentions=True` on such a model with an **empty tuple** rather
    # than an error — so an unforced rollout run has nothing to roll and says
    # nothing about it. The fixture's model is sdpa by default, so every
    # `run_rollout` test above already depends on this; what this pins is that
    # the implementation is put back afterwards.
    before = toy_model.config._attn_implementation
    assert before != v3_rollout.EAGER_ATTENTION

    result = v3_rollout.run_rollout(
        toy_model, tokenizer, SEQUENCES, row_index=ROW_INDEX, batch_size=2
    )
    assert result.attention_implementation == v3_rollout.EAGER_ATTENTION
    assert toy_model.config._attn_implementation == before


def test_eager_attention_restores_the_implementation_after_a_failure(toy_model):
    before = toy_model.config._attn_implementation
    with pytest.raises(RuntimeError), v3_rollout.eager_attention(toy_model) as forced:
        assert forced == v3_rollout.EAGER_ATTENTION
        assert toy_model.config._attn_implementation == "eager"
        raise RuntimeError("boom")
    assert toy_model.config._attn_implementation == before


def test_eager_attention_admits_when_it_could_not_force_the_path():
    # transformers < 5 has no setter: it warns and falls back to eager by
    # itself, so the probabilities are real — but this run has not verified
    # which path produced them, and a sidecar must not claim a pin it did not
    # set. Recording "eager" here would make the sidecar contradict its own
    # note.
    class Old:
        config = type("Config", (), {})()

    with v3_rollout.eager_attention(Old()) as forced:
        assert forced == v3_rollout.UNKNOWN_ATTENTION != v3_rollout.EAGER_ATTENTION


def test_an_empty_attention_tuple_is_refused(tokenizer, monkeypatch):
    # What an sdpa model does under transformers 5.x, in the shape the run
    # meets it: no exception, no attentions.
    class Fused:
        config = type("Config", (), {"_attn_implementation": "sdpa"})()

        def parameters(self):
            yield torch.zeros(1)

        def __call__(self, **_):
            return type("Output", (), {"attentions": ()})()

    with pytest.raises(ValueError, match="nothing to roll"):
        v3_rollout.run_rollout(
            Fused(), tokenizer, SEQUENCES[:1], row_index=[7], batch_size=1
        )


def test_run_rollout_refuses_an_unsorted_row_index(toy_model, tokenizer):
    with pytest.raises(ValueError, match="ascending"):
        v3_rollout.run_rollout(
            toy_model, tokenizer, SEQUENCES, row_index=[23, 7, 11], batch_size=2
        )


def test_run_rollout_refuses_a_row_index_of_the_wrong_length(toy_model, tokenizer):
    with pytest.raises(ValueError, match="row_index"):
        v3_rollout.run_rollout(
            toy_model, tokenizer, SEQUENCES, row_index=[7, 11], batch_size=2
        )


# --- the artifact ----------------------------------------------------------


def test_writer_stores_exactly_the_rollout_and_the_row_index(tmp_path, rollout_result):
    path = write_for(tmp_path, rollout_result)
    assert path.name == "rollout.npz"
    with np.load(path, allow_pickle=False) as archive:
        assert sorted(archive.files) == ["rollout", "row_index"]
        assert archive["rollout"].shape == (len(SEQUENCES), 5, 5)
        assert archive["rollout"].dtype == np.float32
        assert archive["row_index"].dtype == np.int64
        assert archive["row_index"].tolist() == ROW_INDEX


def test_sidecar_states_the_rollout_is_head_invariant(tmp_path, rollout_result):
    # AC: "head별 그림을 기대하는 리뷰가 반드시 오므로 사이드카가 head 불변임을
    # 명시한다." Nine identical matrices read as a bug or as "the heads attend
    # identically"; the sidecar has to say the quantity has no head to vary over.
    path = write_for(tmp_path, rollout_result)
    sidecar = sidecar_of(path)
    assert sidecar["head_axis"] is False
    assert sidecar["heads"] == list(v3_predict.HEADS)
    assert "ADR 0008" in sidecar["head_invariance_note"]
    assert sidecar["adr"] == ["0007", "0008"]


def test_sidecar_records_the_layer_operation_and_that_only_the_final_is_kept(
    tmp_path, rollout_result
):
    sidecar = sidecar_of(write_for(tmp_path, rollout_result))
    assert sidecar["layer_operation"] == "0.5*A + 0.5*I, then row-normalise"
    assert sidecar["layers_combined"] == rollout_result.attention_layers
    assert sidecar["attention_heads_averaged"] == rollout_result.attention_heads
    assert sidecar["stored"] == "final_rollout_only"
    assert sidecar["per_layer_stored"] is False


def test_sidecar_records_the_token_resolution_and_refuses_an_unfold(
    tmp_path, rollout_result
):
    sidecar = sidecar_of(write_for(tmp_path, rollout_result))
    assert sidecar["token_resolution"] == 5
    assert sidecar["token_names"] == list(v3_ig.TOKEN_NAMES)
    assert sidecar["token_boundaries"] == [
        [0, 5],
        [6, 11],
        [12, 17],
        [18, 23],
        [24, 29],
    ]
    assert sidecar["unfolded_to_bases"] is False
    assert "ADR 0007" in sidecar["token_resolution_note"]


def test_sidecar_names_the_quantity_information_dependency(tmp_path, rollout_result):
    sidecar = sidecar_of(write_for(tmp_path, rollout_result))
    assert sidecar["quantity"] == "information_dependency"
    assert "not a physical" in sidecar["quantity_note"]


def test_sidecar_records_the_discarded_cls_mass(tmp_path, rollout_result):
    # The block's rows sum to less than 1 and the sidecar has to say where the
    # rest went, or a reader meets a non-stochastic matrix with no explanation.
    sidecar = sidecar_of(write_for(tmp_path, rollout_result))
    assert sidecar["renormalised_after_cls_removal"] is False
    cls = sidecar["cls_mass"]
    assert 0.0 <= cls["min"] <= cls["median"] <= cls["max"] <= 1.0
    assert np.isclose(cls["max"], float(rollout_result.cls_mass.max()))


def test_sidecar_carries_the_population_pins(tmp_path, rollout_result):
    sidecar = sidecar_of(write_for(tmp_path, rollout_result))
    assert sidecar["population"] == "analysis_intersection"
    assert sidecar["population_digest"] == v3_predict.population_digest(ROW_INDEX)
    assert sidecar["rows"] == len(SEQUENCES)
    assert sidecar["forwards"] == len(SEQUENCES)
    assert sidecar["nan_policy"] == "forbidden"
    assert sidecar["rollout_sha256"]


def test_sidecar_pins_the_code_ref_and_the_library_versions(tmp_path, rollout_result):
    # AC11: the driver pins CODE_REF and the transformers version, and both are
    # recorded here. C1 is a different run, so the versions can legitimately
    # differ from the ISM sidecar's — which is why the sidecars are per analysis.
    sidecar = sidecar_of(write_for(tmp_path, rollout_result))
    assert sidecar["code_ref"] == CODE_REF
    assert sidecar["versions"]["transformers"] == transformers.__version__
    assert sidecar["versions"]["torch"] == torch.__version__
    assert sidecar["split"]["sha256"]
    assert sidecar["checkpoint"]["files"]["config.json"]


def test_writer_refuses_a_branch_name_as_the_code_ref(tmp_path, rollout_result):
    with pytest.raises(ValueError, match="CODE_REF"):
        write_for(tmp_path, rollout_result, code_ref="model_analysis")


def test_writer_refuses_a_missing_code_ref(tmp_path, rollout_result):
    with pytest.raises(ValueError, match="CODE_REF"):
        write_for(tmp_path, rollout_result, code_ref=None)


def test_writer_refuses_an_array_with_a_head_axis(tmp_path, rollout_result):
    # The ADR 0008 failure in artifact form: nine copies of one matrix.
    with_heads = rollout_result.replace(
        rollout=np.repeat(
            rollout_result.rollout[:, None], len(v3_predict.HEADS), axis=1
        )
    )
    with pytest.raises(ValueError, match="head axis"):
        write_for(tmp_path, with_heads)


def test_writer_refuses_a_base_coordinate_unfold(tmp_path, rollout_result):
    unfolded = rollout_result.replace(
        rollout=np.repeat(np.repeat(rollout_result.rollout, 6, axis=1), 6, axis=2)
    )
    with pytest.raises(ValueError, match="token resolution"):
        write_for(tmp_path, unfolded)


def test_writer_refuses_a_result_that_rolled_no_layers(tmp_path, rollout_result):
    with pytest.raises(ValueError, match="no attention layers"):
        write_for(tmp_path, rollout_result.replace(attention_layers=0))


def test_writer_requires_the_checkpoint_it_reports(tmp_path, rollout_result):
    # The design's sidecar rule: "Every artifact carries a metadata sidecar
    # recording the checkpoint digest, the consumed split's checksums...". A
    # rollout artifact that omits it cannot answer which weights produced the
    # attention, so the argument is required rather than optional.
    with pytest.raises(TypeError, match="checkpoint_dir"):
        v3_rollout.write_rollout_artifact(
            tmp_path / "v3_rollout",
            rollout_result,
            code_ref=CODE_REF,
            **provenance_for(tmp_path),
        )


def test_writer_refuses_a_non_finite_rollout(tmp_path, rollout_result):
    broken = rollout_result.rollout.copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        write_for(tmp_path, rollout_result.replace(rollout=broken))


def test_writer_refuses_a_descending_row_index(tmp_path, rollout_result):
    with pytest.raises(ValueError, match="ascending"):
        write_for(
            tmp_path,
            rollout_result.replace(row_index=np.array([23, 11, 7], dtype=np.int64)),
        )
