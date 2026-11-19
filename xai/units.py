"""Affine display units for explanations; raw model results are never overwritten."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExplanationUnits:
    scale: float = 100.0
    offset: float = 0.0
    efficiency: bool = True
    source: str = "fraction output"

    def prediction(self, raw):
        return np.asarray(raw) * self.scale + self.offset

    def contribution(self, raw):
        return np.asarray(raw) * self.scale

    @property
    def target(self):
        return "효율" if self.efficiency else "원출력"

    @property
    def value_unit(self):
        return "%" if self.efficiency else " (raw)"

    @property
    def delta_unit(self):
        return "%p" if self.efficiency else " (raw)"


def nt_units(spec, head_index):
    """Use the same registered training statistics as prediction, without clipping."""
    from predict import _head_stat

    if spec.output_mode == "zscore":
        mean = _head_stat(spec.target_mean, head_index)
        std = _head_stat(spec.target_std, head_index)
        if mean is None or std is None:
            return ExplanationUnits(1.0, 0.0, False, "missing training statistics")
        if not (np.isfinite(mean) and 0 <= mean <= 1.5
                and np.isfinite(std) and 0 < std <= 1.5):
            raise ValueError("Invalid fraction-scale training statistics")
        return ExplanationUnits(std * 100, mean * 100, True,
                                "SpCas9_v3.ipynb: train targets / 100; nanmean/nanstd ddof=0")
    if spec.output_mode in ("raw", "sigmoid", "percent"):
        return ExplanationUnits(source="NT registered fraction output after output_mode")
    return ExplanationUnits(1.0, 0.0, False, "unverified output units")
