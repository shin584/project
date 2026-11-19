"""Regenerate #197's original-model reference in a fresh TensorFlow process.

Run from the dashboard root:
    python -X utf8 docs/verify_issue197_reference.py --output <reference.json>

Uses the archived training feature code and the original .keras bytes, without
importing the dashboard's predictor, feature adapter or torch runtime.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import keras
    import numpy as np
    import tensorflow as tf

    root = Path(__file__).resolve().parents[1]
    asset = root / "models/non_nt/cas12a-cnn-20261102"
    checkpoint = asset / "cas12a_cnn_manifest.keras"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected = "94ac977865efa1cdad734ae05e033cbaf40b38e109013a6d2fe2b9361aa8b28f"
    if digest != expected:
        raise ValueError("Original Cas12a checkpoint differs")
    with zipfile.ZipFile(asset / "source_snapshot.zip") as source:
        feature_source = source.read("cas12a/features.py")
    namespace = {}
    exec(compile(feature_source, "training/features.py", "exec"), namespace)
    # Actual +/- scanned windows, plus the four pre-existing #193 probes.
    contract = json.loads((root / "docs/runtime/contract.json").read_text(encoding="utf-8"))
    windows = ["TCGCTTTAAGAATAAATAAAAGCGCCACAGGATT",
               "GTGGTTTAGTCAATCCTGTGGCGCTTTTATTTAT",
               "AAAATTTA" + "A" * 26]
    windows += [case["window"] for case in contract["cases"]["cas12a"]]
    seq = np.stack([namespace["one_hot_encode"](w) for w in windows]).astype(np.float32)
    feat = np.stack([np.concatenate(namespace["extract_seq_features"](w)) for w in windows])
    model = keras.saving.load_model(checkpoint, compile=False)
    values = np.asarray(model([seq, feat], training=False)).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Reference contains nonfinite output")
    report = {
        "issue": 197, "backend": keras.backend.backend(),
        "versions": {"keras": keras.__version__, "tensorflow": tf.__version__, "numpy": np.__version__},
        "checkpoint_sha256": digest,
        "training_feature_source_sha256": hashlib.sha256(feature_source).hexdigest(),
        "source_snapshot_sha256": hashlib.sha256((asset / "source_snapshot.zip").read_bytes()).hexdigest(),
        "tolerance": contract["criteria"]["forward_parity"],
        "cases": [{"window": w, "sequence_input_sha256": hashlib.sha256(s.tobytes()).hexdigest(),
                   "feature_input_sha256": hashlib.sha256(f.tobytes()).hexdigest(),
                   "raw_score": float(y)} for w, s, f, y in zip(windows, seq, feat, values)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Original-model reference: {len(values)} cases -> {args.output}")


if __name__ == "__main__":
    main()
