"""#201 public explanation boundaries: units, ESM equivalence, reference and caches."""

from dataclasses import replace

import numpy as np
import pytest

from predict import MODEL_REGISTRY, ExplainHandle, NTClassifierPredictor, SPCAS9_VARIANTS
from xai.explain import IGResult, _onehot, integrated_gradients, kernel_shap
from xai.identity import ExplanationIdentity
from xai.units import ExplanationUnits, nt_units


def test_affine_display_changes_predictions_and_deltas_without_changing_raw_results():
    spec = replace(MODEL_REGISTRY['A', 'SpCas9'], target_mean=(.2, .8), target_std=(.1, .3))
    first, second = nt_units(spec, 0), nt_units(spec, 1)
    assert first.prediction(2) == pytest.approx(40)
    assert second.prediction(2) == pytest.approx(140)  # explanation is not clipped
    assert first.contribution(-2) == pytest.approx(-20)
    assert second.contribution(-2) == pytest.approx(-60)
    raw = IGResult(np.array([1., -2.]), 2., -1., 0., units=second)
    np.testing.assert_array_equal(raw.attribution, [1., -2.])
    assert raw.pred == 2
    assert raw.units.prediction(raw.pred) == pytest.approx(140)
    missing = nt_units(replace(spec, target_mean=None), 0)
    assert not missing.efficiency
    assert missing.prediction(2) == 2
    assert missing.delta_unit != '%p'


class TinyTokenizer:
    all_special_tokens = ['<cls>', '<eos>']

    def __call__(self, seqs, **kwargs):
        import torch
        single = isinstance(seqs, str)
        seqs = [seqs] if single else seqs
        ids = [[0, *[3 + 'ACGT'.index(b) for b in seq], 2] for seq in seqs]
        # ESM mask token is 7 in this fixture, so EOS is not accidentally masked.
        if single:
            return {'input_ids': ids[0]}
        return {'input_ids': torch.tensor(ids), 'attention_mask': torch.ones(len(ids), len(ids[0]), dtype=torch.long)}

    def convert_ids_to_tokens(self, ids):
        return ['<cls>' if i == 0 else '<eos>' if i == 2 else 'ACGT'[i - 3] for i in ids]


class TinyNT(NTClassifierPredictor):
    def __init__(self):
        import torch
        from transformers import EsmConfig, EsmForSequenceClassification

        super().__init__(MODEL_REGISTRY['A', 'SpCas9'])
        torch.manual_seed(123)
        self._model = EsmForSequenceClassification(EsmConfig(
            vocab_size=8, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
            num_attention_heads=2, num_labels=9, max_position_embeddings=64,
            token_dropout=True, mask_token_id=7, pad_token_id=1,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            attn_implementation='eager')).eval()
        self._tokenizer = TinyTokenizer()
        self._dtype = torch.float32
        self._num_labels = 9
        for p in self._model.parameters():
            p.requires_grad_(False)

    def explanation_identity(self):
        return ExplanationIdentity('nt', 'A/SpCas9', ('a' * 64,), 'tiny-201')


def test_nt_ig_matches_token_id_prediction_and_keeps_special_tokens_fixed():
    predictor = TinyNT()
    handle = predictor.explain_handle('SpCas9-NG')
    inputs = handle.ig_inputs('ACGT')
    baseline = handle.ig_baseline(inputs)
    np.testing.assert_array_equal(baseline[0][[0, -1]], inputs[0][[0, -1]])
    np.testing.assert_array_equal(baseline[0][1:-1], np.zeros((4, 16)))
    # Creating another handle/window must not replace the first handle's state.
    predictor.explain_handle('xCas9').ig_inputs('AAAAAA')
    pred, gradients = handle.ig_grad_fn([inputs[0][None]])
    expected = handle.raw_fn(_onehot('ACGT')[None])
    np.testing.assert_allclose(pred, expected, atol=1e-7, rtol=1e-5)
    assert np.isfinite(gradients[0]).all()
    result = integrated_gradients(handle, 'ACGT', 32)
    assert len(result.attribution) == 4
    assert result.completeness_error < 1e-5
    assert result.units.scale == pytest.approx(15.00599)


def test_quality_failure_stays_displayable_and_nonfinite_result_is_rejected(caplog):
    from xai.quality import record_quality

    result = IGResult(np.array([2.]), .1, 0., 1.9)
    with caplog.at_level('INFO', logger='xai.quality'):
        record = record_quality(result, None, ('ig', 'Cas12a', 'A', 32))
    assert record['numerical_status'] == 'fail'
    assert 'display finite result' in caplog.text
    with pytest.raises(ValueError, match='NaN/inf'):
        record_quality(replace(result, attribution=np.array([np.nan])), None, ())


def test_shap_disk_roundtrip_preserves_affine_units_and_refuses_old_protocol(tmp_path, monkeypatch):
    from xai import shap_cache

    ident = ExplanationIdentity('nt', 'A/SpCas9', ('b' * 64,), 'corrected-201')
    units = ExplanationUnits(10, 20, True, 'known training stats')
    handle = ExplainHandle(lambda x: np.asarray(x)[:, :, 0].sum(1), None, None, None,
                           model_key=ident, head=SPCAS9_VARIANTS[0], units=units)
    fresh = kernel_shap(handle, 'ACGT', 8, 2, use_disk_cache=False)
    key = shap_cache.shap_cache_key(ident, handle.head, 'ACGT', 8, 2)
    shap_cache.save(key, fresh, tmp_path)
    loaded = shap_cache.load(key, tmp_path)
    np.testing.assert_array_equal(loaded.values, fresh.values)
    assert loaded.units == units
    assert shap_cache.load(shap_cache.shap_cache_key(replace(ident, protocol='legacy'), handle.head,
                                                  'ACGT', 8, 2), tmp_path) is None
    monkeypatch.setattr(shap_cache, 'DEFAULT_DIR', tmp_path)
    handle.raw_fn = lambda *a, **k: pytest.fail('disk hit evaluated the model')
    hit = kernel_shap(handle, 'ACGT', 8, 2)
    assert hit.units == units
    np.testing.assert_array_equal(hit.values, fresh.values)
