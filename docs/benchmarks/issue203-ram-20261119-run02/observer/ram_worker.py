"""Measurement-only renderer observers; all production calls run unchanged."""

import functools
import json
import os
import sys
import time
from pathlib import Path

from desktop.renderer import PredictionRenderer
from xai import explain, shap_cache

COUNTS = {"integrated_gradients": 0, "kernel_shap": 0, "attention_rollout": 0,
          "disk_shap_lookups": 0, "disk_shap_hits": 0}


def count_call(function, name):
    @functools.wraps(function)
    def invoke(*args, **kwargs):
        COUNTS[name] += 1
        result = function(*args, **kwargs)
        if name == "disk_shap_lookups" and result is not None:
            COUNTS["disk_shap_hits"] += 1
        return result
    return invoke


for method in ("integrated_gradients", "kernel_shap", "attention_rollout"):
    setattr(explain, method, count_call(getattr(explain, method), method))
shap_cache.load = count_call(shap_cache.load, "disk_shap_lookups")

def snapshot(renderer, operation):
    predictors = []
    for key, predictor in (getattr(renderer, "_predictors", None) or {}).items():
        predictors.append({
            "key": list(key), "class": type(predictor).__name__,
            "paths": list(predictor.spec.paths),
            "nt_model_loaded": getattr(predictor, "_model", None) is not None,
            "nt_tokenizer_loaded": getattr(predictor, "_tokenizer", None) is not None,
            "non_nt_runtime_loaded": getattr(predictor, "_runtime", None) is not None,
        })
    modules = {name: getattr(sys.modules[name], "__version__", None)
               for name in ("torch", "keras", "tensorflow", "transformers", "tokenizers",
                            "numpy", "shap", "matplotlib") if name in sys.modules}
    record = {"time": time.time(), "pid": os.getpid(), "operation": operation,
              "predictors": predictors, "modules": modules, "compute_counts": dict(COUNTS),
              "torch_threads": sys.modules["torch"].get_num_threads() if "torch" in sys.modules else None}
    path = Path(os.environ["DASHBOARD_RAM_OUTPUT"]) / f"runtime-{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


class Observer(PredictionRenderer):
    def warm_up(self, *args, **kwargs):
        value = super().warm_up(*args, **kwargs)
        snapshot(self, "prepare")
        return value

    def prediction_fragments(self, *args, **kwargs):
        value = super().prediction_fragments(*args, **kwargs)
        snapshot(self, "predict")
        return value

    def explanation_fragment(self, *args, **kwargs):
        value = super().explanation_fragment(*args, **kwargs)
        snapshot(self, "explain:" + str(kwargs.get("method", args[3] if len(args) > 3 else "")))
        return value


if os.environ.get("DASHBOARD_RAM_MODE") == "legacy":
    class ObservedLegacy(Observer):
        pass
else:
    from desktop.worker_renderer import ComputationRenderer, NonNTComputationRenderer

    class ObservedNT(Observer, ComputationRenderer):
        pass

    class ObservedNonNT(Observer, NonNTComputationRenderer):
        pass
