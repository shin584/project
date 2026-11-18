"""#194 child renderer cache behavior, using the existing linear model seam."""

from dataclasses import replace

import core
import pytest
from desktop.worker_renderer import ComputationRenderer
from test_desktop_xai import DNA, LinearPredictor, pngs


@pytest.fixture
def renderer(tmp_path, monkeypatch):
    import predict

    specs = {}
    for cas in ("SpCas9", "Cas12a"):
        path = tmp_path / (cas + ".weights")
        path.write_bytes((cas + " checkpoint one").encode())
        specs[("A", cas)] = replace(predict.MODEL_REGISTRY[("A", cas)], paths=(str(path),))
    monkeypatch.setattr(core, "MODEL_REGISTRY", specs)
    predictors = {("A", cas): LinearPredictor(cas) for cas in ("SpCas9", "Cas12a")}
    return ComputationRenderer(predictors), predictors


def test_prediction_csv_and_chart_resize_reuse_calculations(renderer):
    view, predictors = renderer
    progress = []
    report = lambda d, t: progress.append((d, t))
    view.prediction_fragments(DNA, report)
    assert progress
    progress.clear()
    view.csv_table(DNA, report)
    view.prediction_fragments(DNA, report)
    assert not progress
    first = view.explanation_fragment(DNA, "Cas12a|+|16", "Cas12a", "ig", {"steps": 8}, 40, report)
    progress.clear()
    # Another view/technique must not flush the previous display or arrays.
    view.explanation_fragment(DNA, "Cas12a|+|16", "Cas12a", "rollout", {}, 80, report)
    handles = predictors[("A", "Cas12a")].handles
    resized = view.explanation_fragment(DNA, "Cas12a|+|16", "Cas12a", "ig", {"steps": 8}, 100, report)
    assert pngs(first) == pngs(resized)
    assert predictors[("A", "Cas12a")].handles == handles
    assert not progress


def test_new_checkpoint_and_settings_do_not_reuse_old_explanations(renderer, tmp_path, monkeypatch):
    view, predictors = renderer
    predictor = predictors[("A", "Cas12a")]

    def render(steps=8):
        return view.explanation_fragment(DNA, "Cas12a|+|16", "Cas12a", "ig", {"steps": steps}, 80, lambda d, t: None)

    render()
    assert predictor.handles == 1
    render(16)
    assert predictor.handles == 2
    checkpoint = tmp_path / "replacement.weights"
    checkpoint.write_bytes(b"different checkpoint")
    specs = dict(core.MODEL_REGISTRY)
    specs[("A", "Cas12a")] = replace(specs[("A", "Cas12a")], paths=(str(checkpoint),))
    monkeypatch.setattr(core, "MODEL_REGISTRY", specs)
    render()
    assert predictor.handles == 3


def test_a_new_explanation_protocol_does_not_reuse_old_explanations(renderer, monkeypatch):
    """#208: when #201 advances the NT protocol, earlier explanations are not reused."""
    import predict

    view, predictors = renderer
    predictor = predictors[("A", "Cas12a")]

    def render():
        return view.explanation_fragment(DNA, "Cas12a|+|16", "Cas12a", "shap",
                                         {"nsamples": 16, "n_background": 2}, 80, lambda d, t: None)

    first = render()
    render()
    assert predictor.handles == 1
    monkeypatch.setattr(predict, "NT_EXPLANATION_PROTOCOL", "nt-units-corrected-v2")
    again = render()
    assert predictor.handles == 2
    assert pngs(again) == pngs(first), "fresh recomputation must match the earlier result"


def test_child_shap_ignores_unverified_disk_results(renderer, monkeypatch):
    from xai import shap_cache

    def reject_disk(*args, **kwargs):
        pytest.fail("v3 must not query old disk results without checkpoint/protocol provenance")

    monkeypatch.setattr(shap_cache, "load", reject_disk)
    view, predictors = renderer
    predictor = predictors[("A", "Cas12a")]
    original = predictor.explain_handle

    def handle_with_disk_identity(head):
        handle = original(head)
        handle.model_key = "A/Cas12a"
        handle.head = head
        return handle

    monkeypatch.setattr(predictor, "explain_handle", handle_with_disk_identity)
    fragment = view.explanation_fragment(DNA, "Cas12a|+|16", "Cas12a", "shap",
                                         {"nsamples": 16, "n_background": 2}, 80, lambda d, t: None)
    assert len(pngs(fragment)) == 2
