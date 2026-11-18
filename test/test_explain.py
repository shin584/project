"""XAI input preparation and a known linear explanation without a model."""

import numpy as np
from predict import ExplainHandle
from xai.explain import (
    _onehot,
    _random_backgrounds,
    integrated_gradients,
    kernel_shap,
)


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


def _nonlinear_handle():
    """위치끼리 상호작용하는 작은 함수. KernelSHAP이 부분집합을 실제로 표본추출한다."""
    weights = np.arange(1, 5, dtype=np.float64)

    def raw_fn(X, on_batch=None):
        X = np.asarray(X, dtype=np.float64)
        linear = (X * weights).sum(axis=(1, 2))
        return linear + 0.5 * linear ** 2

    return ExplainHandle(
        raw_fn=raw_fn,
        ig_inputs=lambda seq: [_onehot(seq)],
        ig_grad_fn=lambda batch: (raw_fn(batch[0]), [np.zeros_like(batch[0])]),
        project=lambda attrs: attrs[0].sum(axis=-1),
    )


def test_kernel_shap_repeats_exactly_for_the_same_seed():
    """같은 seed면 호출마다 같은 값이 나와야 한다 (#193).

    shap의 부분집합 표본추출은 numpy **전역** RNG를 쓴다. seed를 걸지 않으면
    같은 입력·같은 설정으로도 호출마다 값이 달라져, 사용자가 같은 화면을 두 번
    열었을 때 다른 기여도를 보게 된다.
    """
    handle = _nonlinear_handle()
    first = kernel_shap(handle, "ACGT", nsamples=8, n_background=2, seed=0,
                        use_disk_cache=False)
    second = kernel_shap(handle, "ACGT", nsamples=8, n_background=2, seed=0,
                         use_disk_cache=False)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.base_value == second.base_value


def test_kernel_shap_does_not_disturb_the_callers_global_rng():
    """재현성을 위해 전역 RNG를 건드리더라도 호출자 상태는 원래대로 되돌린다."""
    handle = _nonlinear_handle()
    np.random.seed(1234)
    expected = np.random.random(3)
    np.random.seed(1234)
    kernel_shap(handle, "ACGT", nsamples=8, n_background=2, seed=0,
                use_disk_cache=False)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_kernel_shap_seed_changes_the_explanation():
    """seed는 배경 서열과 표본추출을 함께 정한다. 바꾸면 값이 바뀌어야 한다."""
    handle = _nonlinear_handle()
    first = kernel_shap(handle, "ACGT", nsamples=8, n_background=2, seed=0,
                        use_disk_cache=False)
    other = kernel_shap(handle, "ACGT", nsamples=8, n_background=2, seed=5,
                        use_disk_cache=False)
    assert not np.array_equal(first.values, other.values)
