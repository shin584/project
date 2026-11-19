"""Child-only renderers with provenance-aware, process-lifetime caches.

One class per model family. The computation process is chosen by factory name
(`desktop.computation.ProcessDispatcher`), so a process holds exactly one
family's weights and a family change replaces the process (ADR 0007).
"""

import json
from collections import OrderedDict

import core
import predict
from xai.explain_cache import ExplainCache

from desktop.renderer import PredictionRenderer


class ComputationRenderer(PredictionRenderer):
    """NT family (the existing behaviour)."""

    FAMILY = "nt"

    def __init__(self, predictors=None):
        super().__init__(predictors, explanation_cache=ExplainCache(use_disk_cache=False))
        self._predictions = OrderedDict()
        self._charts = OrderedDict()

    def _model_key(self, cas):
        # A verified identity (checkpoint bytes, family, protocol) is what lets
        # the explanation cache reuse anything at all (#208). predict memoizes the
        # hash per file state and protocol, so registry or protocol changes show.
        return predict.nt_explanation_identity(core.MODEL_REGISTRY[("A", cas)])

    def _prediction_key(self, cas):
        """Which model and checkpoint the cached prediction rows came from.

        Separate from `_model_key` because a family can predict before its
        explanations are connected, and because predictions do not depend on an
        explanation protocol. For NT the two coincide, and keeping them equal
        here leaves the #194 cache behaviour unchanged.
        """
        return self._model_key(cas)

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
        key = (self.FAMILY,
               tuple(self._prediction_key(cas) for _, cas in sorted(self._load())), dna)
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


class NonNTComputationRenderer(ComputationRenderer):
    """non-NT family: the adopted Keras-3-on-torch runtime (#193, #196).

    Predictions, the four views and the CSV come from this family's checkpoints
    only. Explanations are not connected yet (#199), so asking for one raises
    instead of returning a chart of something else.
    """

    FAMILY = "non_nt"

    def _scan(self, dna):
        from non_nt_cas12a_predict import MANIFEST_CAS_CONFIGS

        return core.scan_all(dna, configs=MANIFEST_CAS_CONFIGS)

    def _prediction_key(self, cas):
        """The adopted checkpoint's pinned identity, without loading the model.

        `model_source.resolve` compares the real bytes against that pinned
        sha256 when the model is loaded, so an unexpected file cannot predict
        under this key.
        """
        return self._load()[("A", cas)].model_identity()

    def _model_key(self, cas):
        ticket = "#200" if cas == "Cas12a" else "#199"
        raise NotImplementedError(
            f"non-NT {cas} 설명 계산은 {ticket}에서 제공합니다.")
