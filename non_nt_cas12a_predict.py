"""Manifest CNN+BiLSTM Cas12a prediction from a scanned 34bp window (#197).

The adopted runtime derives the training-time 354 features from the window;
callers supply no feature values. IG follows #195's composite sequence path;
SHAP recomputes those same features for every perturbed sequence (#200).
"""

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

import model_source
from pam_scanner import CAS_CONFIGS, CandidateSite, ModelIdentity, VariantResult
from predict import BasePredictor, ModelSpec, variant_result, version_label


# The NT scanner's historic Cas12a window starts at the PAM. The manifest CNN
# was trained on 4 flank + 4 PAM + 23 guide + 3 flank. Keep NT unchanged.
MANIFEST_CAS_CONFIGS = [
    replace(cfg, window_upstream=4) if cfg.name == "Cas12a" else cfg
    for cfg in CAS_CONFIGS
]


class NonNTCas12aPredictor(BasePredictor):
    """One Cas12a row per site, with lazy, process-owned checkpoint loading."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self._runtime = None

    def model_identity(self) -> ModelIdentity:
        pinned = model_source.NON_NT_FILES[self.spec.paths[0]]
        digest = pinned["sha256"]
        return ModelIdentity(
            family=self.spec.family, cas=self.spec.cas, name=self.spec.label,
            version=version_label(self.spec, digest), raw_unit=self.spec.raw_unit,
            checkpoint_sha256=digest)

    @property
    def runtime(self):
        if self._runtime is None:
            from non_nt_runtime import CHECKPOINTS, NonNTRuntime

            if self.spec.paths != (CHECKPOINTS["cas12a"],):
                raise ValueError("Cas12a requires the adopted manifest checkpoint")
            runtime = NonNTRuntime.load("cas12a")
            if (runtime.input_shapes != [(34, 4), (354,)]
                    or runtime.input_names() != ["Input_Seq", "Input_Feat"]
                    or runtime.head_names != ["dense_2"]):
                raise RuntimeError("Cas12a checkpoint input/output contract differs")
            if runtime.checkpoint_sha256 != self.model_identity().checkpoint_sha256:
                raise RuntimeError("Cas12a checkpoint SHA256 differs from the adopted model")
            self._runtime = runtime
        return self._runtime

    def predict_batch(self, sites: Sequence[CandidateSite],
                      on_batch: Callable | None = None) -> list[VariantResult]:
        if not sites:
            return []
        # Reject an unrelated window before loading weights or deriving features.
        if any(s.cas_type != "Cas12a" or len(s.model_input_seq) != 34
               or s.model_input_seq[4:8].upper() != s.pam_seq.upper()
               or s.model_input_seq[8:31].upper() != s.guide_seq.upper()
               for s in sites):
            raise ValueError("Cas12a requires 34bp windows: 4 flank + PAM + 23bp guide + 3 flank")
        from non_nt_runtime import cas12a_derive, one_hot

        windows = [site.model_input_seq.upper() for site in sites]
        sequence = np.stack([one_hot(window) for window in windows])
        features = np.stack([cas12a_derive(window) for window in windows])
        raw = self.runtime.forward_all([sequence, features], on_batch=on_batch)[:, 0]
        if not np.isfinite(raw).all():
            raise RuntimeError("Cas12a prediction contains NaN or infinity")
        identity = self.model_identity()
        return [variant_result(self.spec, site, "Cas12a", float(value), identity)
                for site, value in zip(sites, raw)]

    def explain_heads(self, site: CandidateSite) -> list[str]:
        return ["Cas12a"] if site.cas_type == "Cas12a" else []

    def explanation_identity(self):
        # The adopted handle owns the path protocol and actual loaded SHA.
        from xai.identity import verified

        identity = verified(self.explain_handle("Cas12a").model_key)
        if identity is None:
            raise RuntimeError("Cas12a 설명 checkpoint 식별을 확인할 수 없습니다.")
        return identity

    def explain_handle(self, head: str):
        from non_nt_runtime import cas12a_explain_handle

        if head != "Cas12a":
            raise ValueError(f"모르는 Cas12a 헤드: {head}")
        handle = cas12a_explain_handle(self.runtime)
        # Use the same name as the prediction row in both memory/disk caches.
        handle.head = head
        return handle
