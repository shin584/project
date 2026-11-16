"""Display score semantics and configuration reconstruction without weights."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from predict import MODEL_REGISTRY, _config_from_state, _to_display_score


@pytest.mark.parametrize("raw,expected", [(-1.0, 15.865525393), (0.0, 50.0), (1.0, 84.134474607)])
def test_zscores_without_training_statistics_are_percentiles(raw, expected):
    spec = replace(MODEL_REGISTRY[("A", "SpCas9")], target_mean=None, target_std=None)
    value, kind = _to_display_score(spec, raw)
    assert value == pytest.approx(expected)
    assert kind == "percentile"


@pytest.mark.parametrize("head,raw,expected", [(0, 1.0, 30.0), (1, -1.0, 50.0), (0, -10.0, 0.0), (1, 10.0, 100.0)])
def test_zscores_with_training_statistics_restore_efficiency_per_head(head, raw, expected):
    spec = replace(MODEL_REGISTRY[("A", "SpCas9")], target_mean=(0.2, 0.8), target_std=(0.1, 0.3))
    value, kind = _to_display_score(spec, raw, head)
    assert value == pytest.approx(expected)
    assert kind == "efficiency"


def test_percent_training_statistics_are_rejected():
    spec = replace(MODEL_REGISTRY[("A", "SpCas9")], target_mean=49.6, target_std=20.0)
    with pytest.raises(ValueError, match="fraction"):
        _to_display_score(spec, 0.0)


@pytest.mark.parametrize("raw,expected", [(-0.5, 0.0), (0.25, 25.0), (1.5, 100.0)])
def test_fraction_outputs_are_clipped_efficiencies(raw, expected):
    assert _to_display_score(MODEL_REGISTRY[("A", "Cas12a")], raw) == (expected, "efficiency")


def test_esm_configuration_is_reconstructed_from_shapes_without_tensors():
    state = {
        "esm.embeddings.word_embeddings.weight": SimpleNamespace(shape=(4105, 1280)),
        "esm.embeddings.position_embeddings.weight": SimpleNamespace(shape=(1002, 1280)),
        "esm.encoder.layer.0.intermediate.dense.weight": SimpleNamespace(shape=(5120, 1280)),
        "esm.encoder.layer.23.intermediate.dense.weight": SimpleNamespace(shape=(5120, 1280)),
    }
    cfg = _config_from_state(state, num_labels=9)
    assert (cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers) == (4105, 1280, 24)
    assert (cfg.intermediate_size, cfg.max_position_embeddings, cfg.num_labels) == (5120, 1002, 9)
    assert cfg.num_attention_heads == 20
    assert cfg.position_embedding_type == "absolute"
    assert not cfg.emb_layer_norm_before
    assert cfg.token_dropout
    assert (cfg.pad_token_id, cfg.mask_token_id) == (1, 2)
    assert cfg._attn_implementation == "eager"
