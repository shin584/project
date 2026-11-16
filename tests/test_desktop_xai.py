"""#187: real desktop calculation/cache/charts with weight-free model handles."""

import base64
import io
import re
import threading
from html.parser import HTMLParser

import core
import numpy as np
import pytest
from desktop.api import DesktopApi, Dispatcher
from desktop.renderer import PredictionRenderer

DNA = (
    "GATCCAGTCAGCATGCTTTACGATCGTACGATCGAATCGTAGG"
    "CTAGCATCGATCGATGCATCGTAGCTAGCTAGCATCGA"
)


class LinearPredictor:
    """Only the model boundary is substituted; XAI and rendering stay real."""

    connected = True

    def __init__(self, cas):
        self.cas = cas
        self.handles = 0
        self.windows = []
        self.batch_started = threading.Event()
        self.release_batch = threading.Event()
        self.release_batch.set()

    def explain_heads(self, site):
        return ["Cas12a"] if self.cas == "Cas12a" else ["SpCas9-NG", "xCas9"]

    def predict_batch(self, sites, on_batch=None):
        rows = []
        for done, site in enumerate(sites, 1):
            for head in self.explain_heads(site):
                rows.append(core.VariantResult(
                    cas_name=head, pam_seq=site.pam_seq, guide_seq=site.guide_seq,
                    distance=site.distance, strand=site.strand, raw_score=0.5,
                    efficiency_pct=50, site_key=site.site_key, cas_type=self.cas,
                ))
            if on_batch:
                on_batch(done, len(sites))
        return rows

    def explain_handle(self, head):
        self.handles += 1
        weights = np.array([0.01, 0.02, -0.01, -0.02])

        def encode(window):
            self.windows.append(window)
            return [np.eye(4)[["ACGT".index(b) for b in window]]]

        def gradients(batch):
            x = np.asarray(batch[0])
            return (x * weights).sum(axis=(1, 2)), [np.broadcast_to(weights, x.shape)]

        def raw_fn(x, on_batch=None):
            outputs = []
            for start in range(0, len(x), 8):
                chunk = np.asarray(x[start:start + 8])
                outputs.extend((chunk * weights).sum(axis=(1, 2)))
                if on_batch:
                    self.batch_started.set()
                    if not self.release_batch.wait(10):
                        raise TimeoutError("Test did not release model batch")
                    on_batch(min(start + 8, len(x)), len(x))
            return np.asarray(outputs)

        def attention(window):
            self.windows.append(window)
            midpoint = len(window) // 2
            return ([np.array([[[0.8, 0.2], [0.4, 0.6]]])],
                    [(0, midpoint), (midpoint, len(window))], len(window))

        # No model_key: do not use the real model's disk SHAP results.
        return core.ExplainHandle(
            raw_fn=raw_fn, ig_inputs=encode, ig_grad_fn=gradients,
            project=lambda attrs: attrs[0].sum(axis=-1), attention_fn=attention,
        )


@pytest.fixture
def rig():
    core.EXPLAIN_CACHE.clear()
    predictors = {("A", cas): LinearPredictor(cas) for cas in ("SpCas9", "Cas12a")}
    renderer = PredictionRenderer(predictors)
    events = []
    dispatcher = Dispatcher(events.append)
    try:
        yield DesktopApi(renderer, dispatcher), dispatcher, events, predictors
    finally:
        for predictor in predictors.values():
            predictor.release_batch.set()
        dispatcher.close()
        core.EXPLAIN_CACHE.clear()


def explain(rig, method, *, cas="Cas12a", head=None, params=None, size=80):
    api, dispatcher, events, _ = rig
    site = "Cas12a|+|16" if cas == "Cas12a" else "SpCas9|+|40"
    rid = api.explain(DNA, site, head or cas, method, params, size)["request_id"]
    assert dispatcher.wait_idle(30)
    received = [e for e in events if e["request_id"] == rid]
    assert dispatcher.outcome(rid) == "delivered", received
    return received[-1]["fragments"]["xai"]


def pngs(fragment):
    return [base64.b64decode(s) for s in re.findall(r'src="data:image/png;base64,([^"]+)"', fragment)]


@pytest.mark.parametrize("cas,head,length", [("Cas12a", "Cas12a", 34), ("SpCas9", "SpCas9-NG", 30)])
@pytest.mark.parametrize("method,count", [("rollout", 2), ("ig", 1), ("shap", 2)])
def test_methods_render_real_pngs_in_model_window_coordinates(rig, cas, head, length, method, count):
    import matplotlib.pyplot as plt
    from PIL import Image

    fragment = explain(rig, method, cas=cas, head=head)
    images = pngs(fragment)
    assert len(images) == count
    for data in images:
        image = Image.open(io.BytesIO(data))
        assert image.format == "PNG"
        assert image.width >= 500 and image.height >= 200
    assert f"윈도우 {length}bp" in fragment
    assert head in fragment
    assert "PAM" in fragment and "guide" in fragment and "flank" in fragment
    assert not re.search(r'<script|https?://|src="(?!data:)', fragment)
    assert plt.get_fignums() == []


def test_ig_matches_streamlit_chart_with_known_signed_attributions(rig):
    import matplotlib.pyplot as plt

    fragment = explain(rig, "ig")
    # Independent linear model: each actual base contributes its own weight.
    sequence = "TTTACGATCGTACGATCGAATCGTAGGCTAGCAT"
    assert len(sequence) == 34
    values_pct = np.array([{"A": 1, "C": 2, "G": -1, "T": -2}[b] for b in sequence])
    figure = core.attribution_logo(values_pct, sequence, core.WINDOW_LAYOUT["Cas12a"])
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=100)
    finally:
        plt.close(figure)
    assert pngs(fragment) == [buffer.getvalue()]


@pytest.mark.parametrize("method", ["rollout", "ig", "shap"])
def test_size_and_reselecting_an_explanation_reuse_the_shared_cache(rig, method):
    first = explain(rig, method, size=40)
    predictor = rig[3][("A", "Cas12a")]
    assert predictor.handles == 1
    explain(rig, "rollout" if method != "rollout" else "ig")
    handles = predictor.handles
    second = explain(rig, method, size=100)
    assert predictor.handles == handles
    assert pngs(first) == pngs(second)
    assert 'style="width:40%"' in first
    assert 'style="width:100%"' in second


@pytest.mark.parametrize("method,changed", [
    ("ig", {"steps": 40}),
    ("shap", {"nsamples": 80}),
    ("shap", {"n_background": 6}),
])
def test_calculation_settings_invalidate_the_cached_result(rig, method, changed):
    explain(rig, method)
    before = rig[3][("A", "Cas12a")].handles
    explain(rig, method, params=changed)
    assert rig[3][("A", "Cas12a")].handles == before + 1


def test_real_shap_progress_increases_to_completion(rig):
    explain(rig, "shap")
    progress = [e["done"] / e["total"] for e in rig[2] if e["type"] == "progress"]
    assert len(progress) > 2
    assert 0 < progress[0] < progress[-1] == 1
    assert progress == sorted(progress)


def test_new_request_cancels_real_shap_and_cancelled_result_is_not_cached(rig):
    api, dispatcher, events, predictors = rig
    predictor = predictors[("A", "Cas12a")]
    predictor.release_batch.clear()
    old = api.explain(DNA, "Cas12a|+|16", "Cas12a", "shap")["request_id"]
    assert predictor.batch_started.wait(10)
    new = api.explain(DNA, "Cas12a|+|16", "Cas12a", "rollout")["request_id"]
    predictor.release_batch.set()
    assert dispatcher.wait_idle(30)
    assert dispatcher.outcome(old) == "cancelled"
    assert dispatcher.outcome(new) == "delivered"
    assert not any(e["request_id"] == old and e["type"] in {"result", "error"} for e in events)
    before = predictor.handles
    explain(rig, "shap")
    assert predictor.handles == before + 1


class SiteOptions(HTMLParser):
    def __init__(self, fragment):
        super().__init__()
        self.options = []
        self.feed(fragment)

    def handle_starttag(self, tag, attrs):
        if tag == "option":
            self.options.append(dict(attrs))


def test_prediction_provides_all_sites_and_model_heads_without_model_b(rig):
    import json

    api, dispatcher, events, _ = rig
    rid = api.predict(DNA)["request_id"]
    assert dispatcher.wait_idle(30)
    fragment = next(e for e in events if e["request_id"] == rid and e["type"] == "result")["fragments"]["detail"]
    options = SiteOptions(fragment).options
    assert {o["value"] for o in options} == {s.site_key for s in core.scan_all(DNA).sites}
    for option in options:
        assert json.loads(option["data-heads"]) == (
            ["Cas12a"] if option["value"].startswith("Cas12a|") else ["SpCas9-NG", "xCas9"]
        )
    assert "SaCas9" not in fragment and "Model B" not in fragment
