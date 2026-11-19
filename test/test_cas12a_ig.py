"""#195: continuous features and the adopted single-sequence explanation seam."""

import numpy as np
import pytest

import non_nt_runtime as rt


WINDOW = "TTTAAGAATAAATAAAAGCGCCACAGGATTGACT"


def test_continuous_features_preserve_all_training_features_and_unknown_bases():
    import torch
    from cas12a.continuous import derive_features

    sequences = [WINDOW, "A" * 34, "GC" * 17]
    sequences += [(a + b) * 17 for a in "ACGT" for b in "ACGT"]
    sequences += [WINDOW[:i] + "N" + WINDOW[i + 1:] for i in range(34)]
    x = torch.tensor(np.stack([rt.one_hot(s) for s in sequences]))
    actual = derive_features(x).numpy()
    expected = np.stack([rt.cas12a_derive(s) for s in sequences])
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(derive_features(torch.zeros_like(x)).numpy(), 0)


def test_soft_features_have_the_analytic_gc_and_adjacent_pair_gradient():
    import torch
    from cas12a.continuous import derive_features

    x = torch.full((34, 4), 0.25, dtype=torch.float64, requires_grad=True)
    features = derive_features(x)
    np.testing.assert_allclose(features.detach().numpy()[:2], [0.5, 0.5])
    np.testing.assert_allclose(features.detach().numpy()[2:], 0.0625)
    # 2*GC + 3*tail_GC + 5*(first A * second C).
    gradient = torch.autograd.grad(2*features[0] + 3*features[1] + 5*features[3], x)[0]
    expected = np.zeros((34, 4))
    expected[8:31, 1:3] = 2/23
    expected[23:31, 1:3] += 3/8
    expected[8, 0] += 1.25
    expected[9, 1] += 1.25
    np.testing.assert_allclose(gradient.numpy(), expected, atol=1e-8, rtol=0)


@pytest.fixture(scope="module")
def runtime():
    return rt.NonNTRuntime.load("cas12a")


def test_selected_handle_explains_the_composed_model_through_the_existing_loop(runtime):
    from xai.explain import integrated_gradients

    handle = rt.cas12a_explain_handle(runtime)
    assert len(handle.ig_inputs(WINDOW)) == 1
    result = integrated_gradients(handle, WINDOW, steps=32)
    direct = runtime.forward([rt.one_hot(WINDOW)[None], rt.cas12a_derive(WINDOW)[None]])
    np.testing.assert_allclose(result.pred, direct[0], atol=1e-6, rtol=0)
    assert result.attribution.shape == (34,)
    assert np.isfinite(result.attribution).all()
    assert sum(handle.input_totals.values()) == pytest.approx(result.attribution.sum())
    # Numerical quality is measured against the frozen contract by the verifier;
    # a default-step completeness miss must not prevent a computable result.
    assert np.isfinite(result.completeness_error)


def test_composite_raw_function_recomputes_soft_features_without_rounding(runtime):
    handle = rt.cas12a_explain_handle(runtime)
    x = 0.5*rt.one_hot(WINDOW)[None]
    z = rt.cas12a_derive(WINDOW)[None]
    expected_features = np.concatenate([0.5*z[:, :2], 0.25*z[:, 2:]], axis=1)
    expected = runtime.forward([x, expected_features])
    np.testing.assert_allclose(handle.raw_fn(x), expected, atol=1e-6, rtol=0)
    y, _ = handle.ig_grad_fn([x])
    np.testing.assert_allclose(y, expected, atol=1e-6, rtol=0)


def test_composite_gradients_are_independent_per_batch_row(runtime):
    a, b = rt.one_hot(WINDOW), 0.5*rt.one_hot("GC"*17)
    _, together = runtime.composite_gradients(np.stack([a, b]))
    for row, x in zip(together, (a, b)):
        _, single = runtime.composite_gradients(x[None])
        np.testing.assert_allclose(row, single[0], atol=1e-6, rtol=0)


def test_path_protocol_prevents_cross_path_memory_and_disk_cache_hits(runtime):
    from xai.explain_cache import ExplainCache
    from xai.shap_cache import shap_cache_key

    class Predictor:
        calls = 0

        def explain_handle(self, head):
            self.calls += 1
            return self.handle

    cache, predictor = ExplainCache(), Predictor()
    composite = rt.cas12a_explain_handle(runtime)
    dual = rt.cas12a_explain_handle(runtime, path="dual")
    assert composite.model_key.verified and dual.model_key.verified
    assert composite.model_key.checkpoints == dual.model_key.checkpoints
    assert composite.model_key.protocol != dual.model_key.protocol
    assert shap_cache_key(composite.model_key, composite.head, WINDOW, 64, 4, 0, 0) != \
        shap_cache_key(dual.model_key, dual.head, WINDOW, 64, 4, 0, 0)
    predictor.handle = composite
    first = cache.integrated_gradients(predictor, composite.model_key, composite.head, WINDOW, 8)
    predictor.handle = dual
    cache.integrated_gradients(predictor, dual.model_key, dual.head, WINDOW, 8)
    repeated = cache.integrated_gradients(predictor, composite.model_key, composite.head, WINDOW, 8)
    assert repeated is first
    assert predictor.calls == 2
    assert len(cache) == 2


def test_selected_path_keeps_progress_and_callback_cancellation(runtime):
    from xai.explain import integrated_gradients

    handle = rt.cas12a_explain_handle(runtime)
    progress = []
    integrated_gradients(handle, WINDOW, steps=4, chunk_size=2,
                         on_batch=lambda d, t: progress.append((d, t)))
    assert progress == [(2, 4), (4, 4)]

    def cancel(done, total):
        raise RuntimeError("cancelled by caller")

    with pytest.raises(RuntimeError, match="cancelled by caller"):
        integrated_gradients(handle, WINDOW, steps=32, on_batch=cancel)
    with pytest.raises(RuntimeError, match="cancelled by caller"):
        handle.raw_fn(rt.one_hot(WINDOW)[None], on_batch=cancel)


def test_path_contract_rejects_unknown_paths_and_other_models(runtime):
    with pytest.raises(ValueError, match="Unknown Cas12a IG path"):
        rt.cas12a_explain_handle(runtime, path="invented")
    with pytest.raises(ValueError, match="requires the Cas12a"):
        rt.cas12a_explain_handle(rt.NonNTRuntime("spcas9", None))
