"""Child-only NT renderer with provenance-aware, process-lifetime caches."""

import json
from collections import OrderedDict

import core
import predict
from xai.explain_cache import ExplainCache

from desktop.renderer import PredictionRenderer


class ComputationRenderer(PredictionRenderer):
    def __init__(self, predictors=None):
        super().__init__(predictors, explanation_cache=ExplainCache(use_disk_cache=False))
        self._predictions = OrderedDict()
        self._charts = OrderedDict()

    def _model_key(self, cas):
        # A verified identity (checkpoint bytes, family, protocol) is what lets
        # the explanation cache reuse anything at all (#208). predict memoizes the
        # hash per file state and protocol, so registry or protocol changes show.
        return predict.nt_explanation_identity(core.MODEL_REGISTRY[("A", cas)])

    @staticmethod
    def _remember(cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        # Chart fragments can be large; keep at most eight complete displays.
        while len(cache) > 8:
            cache.popitem(last=False)
        return value

    def _rows_with_progress(self, dna, on_batch):
        dna = core.validate_sequence(dna)
        key = (tuple(self._model_key(cas) for _, cas in core.MODEL_REGISTRY), dna)
        if key in self._predictions:
            self._predictions.move_to_end(key)
            return self._predictions[key]
        return self._remember(self._predictions, key, super()._rows_with_progress(dna, on_batch))

    def explanation_fragment(self, dna, site_key, head, method, params, size, on_batch):
        dna = core.validate_sequence(dna)
        cas = site_key.split("|")[0]
        key = (self._model_key(cas), dna, site_key, head, method,
               json.dumps(params, sort_keys=True))
        if key in self._charts:
            self._charts.move_to_end(key)
            fragment = self._charts[key]
        else:
            fragment = super().explanation_fragment(dna, site_key, head, method, params, 80, on_batch)
            self._remember(self._charts, key, fragment)
        return fragment.replace('style="width:80%"', f'style="width:{size}%"')
