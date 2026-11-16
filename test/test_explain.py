"""XAI input preparation and a known linear explanation without a model."""

import numpy as np

from predict import ExplainHandle
from xai.explain import _onehot, _random_backgrounds, integrated_gradients


def test_onehot_preserves_positions_and_leaves_unknown_bases_zero():
    np.testing.assert_array_equal(_onehot("acgTN"), [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0],
    ])
    assert _onehot("").shape == (0, 4)
    assert _onehot("A").dtype == np.float32


def test_random_backgrounds_are_seeded_onehot_sequences():
    backgrounds = _random_backgrounds(length=34, n=4, seed=7)
    assert backgrounds.shape == (4, 34, 4)
    assert backgrounds.dtype == np.float32
    assert set(np.unique(backgrounds)) == {0, 1}
    np.testing.assert_array_equal(backgrounds.sum(axis=-1), np.ones((4, 34)))
    np.testing.assert_array_equal(backgrounds, _random_backgrounds(34, 4, 7))
    assert not np.array_equal(backgrounds, _random_backgrounds(34, 4, 8))


def test_linear_handle_produces_known_ig_attributions():
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    def gradients(batch):
        [x] = batch
        return (x * weights).sum(axis=(1, 2)), [np.broadcast_to(weights, x.shape)]

    handle = ExplainHandle(
        raw_fn=lambda x: (x * weights).sum(axis=(1, 2)),
        ig_inputs=lambda seq: [_onehot(seq)],
        ig_grad_fn=gradients,
        project=lambda attrs: attrs[0].sum(axis=-1),
    )
    result = integrated_gradients(handle, "ACGT", steps=7, chunk_size=3)
    np.testing.assert_allclose(result.attribution, [1, 2, 3, 4])
    assert result.pred == 10.0
    assert result.baseline_pred == 0.0
    assert result.completeness_error == 0.0
