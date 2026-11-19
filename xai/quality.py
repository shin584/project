"""Internal numerical diagnostics. Finite results remain displayable (#201)."""

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

from xai.explain import IGResult, SHAPResult, RolloutResult

logger = logging.getLogger(__name__)
_LOG_LOCK = threading.Lock()
_FILE_LOGGER_READY = False

# Frozen before real-model measurement in docs/nt-xai/contract.json.
RESIDUAL_ATOL_PP = 0.5
RESIDUAL_RTOL = 0.05


def _enable_internal_log():
    global _FILE_LOGGER_READY
    with _LOG_LOCK:
        if not _FILE_LOGGER_READY:
            folder = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir())) / 'dashboard-v3/logs'
            folder.mkdir(parents=True, exist_ok=True)
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(folder / 'nt-xai.jsonl', maxBytes=2_000_000,
                                          backupCount=2, encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            _FILE_LOGGER_READY = True


def record_quality(result, model_identity, request):
    """Reject nonfinite calculations; log numerical quality without UI warnings."""
    if isinstance(result, IGResult):
        arrays = [result.attribution, result.pred, result.baseline_pred, result.completeness_error]
        error = float(result.units.contribution(result.completeness_error))
        signal = abs(float(result.units.contribution(result.pred - result.baseline_pred)))
    elif isinstance(result, SHAPResult):
        arrays = [result.values, result.pred, result.base_value, result.additivity_error]
        error = float(result.units.contribution(result.additivity_error))
        signal = abs(float(result.units.contribution(result.pred - result.base_value)))
    elif isinstance(result, RolloutResult):
        arrays = [result.token_matrix, result.position_profile]
        error, signal = 0.0, 0.0
    else:
        return None  # Test doubles do not carry numerical arrays.
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("XAI result contains NaN/inf")
    limit = RESIDUAL_ATOL_PP + RESIDUAL_RTOL * signal
    record = dict(identity=model_identity.as_dict() if model_identity else None,
                  request=request, finite=True, residual=error, limit=limit,
                  residual_unit=result.units.delta_unit if hasattr(result, "units") else "attention",
                  numerical_status="pass" if error <= limit else "fail",
                  display_policy="display finite result; no quality banner")
    if model_identity is not None and model_identity.family == 'nt':
        try:
            _enable_internal_log()
        except OSError:
            logger.exception('Cannot open internal NT XAI diagnostic file')
    logger.info(json.dumps(record, ensure_ascii=False))
    return record
