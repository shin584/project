"""NT unit labels and exhibition policy at the public HTML rendering boundary."""

from types import SimpleNamespace

import core
import numpy as np
import pytest

from desktop.xai_view import explanation_fragment
from xai.explain import IGResult, SHAPResult
from xai.units import ExplanationUnits

DNA = 'CAAAGTAGCCGGTCGCTTTAAGAATAAATAAAAGCGCCACAGGATTGACTAAACCACAGCCGGCGCCCTGGCCCCCTACCG'


@pytest.mark.parametrize('efficiency', [True, False])
@pytest.mark.parametrize('method', ['ig', 'shap'])
def test_nt_keeps_finite_graphs_with_failed_quality_and_labels_the_actual_units(method, efficiency):
    site = next(s for s in core.scan_all(DNA).sites if s.site_key == 'SpCas9|+|32')
    units = ExplanationUnits(30, 80, True, 'known stats') if efficiency else ExplanationUnits(1, 0, False, 'unverified')
    values = np.zeros(30)
    values[0] = 2
    # Intentionally incomplete; the display policy keeps this finite graph.
    ig = IGResult(values, 2, -1, 1., units)
    shap = SHAPResult(values, -1, 2, 1., 0, 0, units)
    cache = SimpleNamespace(integrated_gradients=lambda *a, **k: ig,
                            kernel_shap=lambda *a, **k: shap)
    fragment = explanation_fragment(SimpleNamespace(), site, 'SpCas9-NG', method,
                                    {'steps': 32, 'nsamples': 64, 'n_background': 4}, 80, lambda *a: None,
                                    cache=cache)
    assert fragment.count('src="data:image/png;base64,') == (1 if method == 'ig' else 2)
    assert 'class="warning"' not in fragment and 'class="error"' not in fragment
    assert 'raw 예측 2' in fragment
    if efficiency:
        assert '140.00%' in fragment
        assert '50.00%' in fragment  # baseline uses offset exactly once
        assert '+60.00%p' in fragment or '+60.000%p' in fragment
        assert '0~100%로 제한' in fragment
    else:
        assert '%p' not in fragment
        assert '2.00 (raw)' in fragment
        assert '효율이나 백분위 기여도가 아닙니다' in fragment
