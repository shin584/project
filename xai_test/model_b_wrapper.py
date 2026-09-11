import os
import pickle
import re
import subprocess

import joblib
import numpy as np
import torch
import xgboost as xgb
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import gc_fraction
from transformers import AutoModelForMaskedLM, AutoTokenizer


class Model_B_Predictor:
    def __init__(self, nt_model_dir: str, xgb_model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading NT backbone from {nt_model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(nt_model_dir)
        self.nt_model = AutoModelForMaskedLM.from_pretrained(nt_model_dir).to(
            self.device
        )
        self.nt_model.eval()

        print(f"Loading XGBoost model from {xgb_model_path}...")
        self.xgb_model = self._load_xgb_model(xgb_model_path)

    def _load_xgb_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"XGBoost model file not found: {path}")

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:  # noqa: BLE001 - pre-existing multi-format load fallback
            try:
                return joblib.load(path)
            except Exception:  # noqa: BLE001 - pre-existing multi-format load fallback
                model = xgb.XGBRegressor()
                model.load_model(path)
                return model

    def extract_nt_embeddings(self, sequences: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            sequences, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.nt_model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

            batch_size, seq_len, hidden_dim = last_hidden_state.shape
            assert seq_len == 7, f"Expected 7 tokens, got {seq_len}"
            assert hidden_dim == 1280, f"Expected 1280 hidden dim, got {hidden_dim}"

            flattened_embeddings = last_hidden_state.view(batch_size, -1).cpu().numpy()
        return flattened_embeddings

    def _get_guide_rna(self, dna_seq: str) -> str:
        complement_map = str.maketrans("ATCGatcg", "UAGCuagc")
        return dna_seq.translate(complement_map)[::-1]

    # RNAfold/RNAduplex are external processes: each subprocess.run() call pays
    # OS process-spawn overhead regardless of how little work it does. Both CLIs
    # accept many records over a single stdin stream, so batching sequences into
    # one call (chunked, to keep any single call's stdin/stdout bounded) turns an
    # O(n) process-spawn cost into O(n / batch_size) - the dominant cost at
    # ISM-sweep scale (tens of thousands of mutants). Deliberate scope override of
    # issue #9's "RNAfold/RNAduplex call batching" Out-of-Scope note, requested
    # directly by the user; verified to reproduce the exact same per-sequence
    # energy values as the prior one-process-per-sequence implementation.
    PHYSICAL_FEATURE_BATCH_SIZE = 500

    @staticmethod
    def _chunked(items: list, size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _calc_mfe_batch(self, seqs: list[str]) -> list[float]:
        energies: list[float] = []
        for chunk in self._chunked(seqs, self.PHYSICAL_FEATURE_BATCH_SIZE):
            # RNAfold CLI 호출 (--noPS: 이미지 생성 방지)
            proc = subprocess.run(
                ["RNAfold", "--noPS"],
                input="".join(f"{seq}\n" for seq in chunk),
                capture_output=True,
                text=True,
                check=True,
            )
            # Each record prints "<sequence>\n<dot-bracket structure> (<energy>)\n".
            # The structure diagram's own parentheses never have a digit directly
            # inside them, so this regex matches exactly one energy per record,
            # in input order. 출력 예시: ... ( -4.20)
            matches = re.findall(r"\(\s*(-?\d+(?:\.\d+)?)\)", proc.stdout)
            assert len(matches) == len(chunk), (
                f"RNAfold batch output mismatch: expected {len(chunk)} results, got {len(matches)}"
            )
            energies.extend(float(m) for m in matches)
        return energies

    def _calc_duplex_dg_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        energies: list[float] = []
        for chunk in self._chunked(pairs, self.PHYSICAL_FEATURE_BATCH_SIZE):
            # RNAduplex CLI 호출
            proc = subprocess.run(
                ["RNAduplex"],
                input="".join(f"{guide}\n{target}\n" for guide, target in chunk),
                capture_output=True,
                text=True,
                check=True,
            )
            # 출력 예시: ... :  1,36 :   1,36 (-32.50)
            matches = re.findall(r"\(\s*(-?\d+(?:\.\d+)?)\)", proc.stdout)
            assert len(matches) == len(chunk), (
                f"RNAduplex batch output mismatch: expected {len(chunk)} results, got {len(matches)}"
            )
            energies.extend(float(m) for m in matches)
        return energies

    def compute_physical_features(self, sequences: list[str]) -> np.ndarray:
        # 1. MFE (RNAfold.exe), batched across all sequences
        mfe_values = self._calc_mfe_batch(sequences)

        # 2. Duplex ΔG (RNAduplex.exe), batched across all sequences
        guide_rnas = [self._get_guide_rna(seq) for seq in sequences]
        dg_values = self._calc_duplex_dg_batch(list(zip(guide_rnas, sequences)))

        features = []
        for seq, mfe, dg in zip(sequences, mfe_values, dg_values):
            # 3. Tm (must match training-time nn_table=DNA_NN4 in nt_feature_stacking.py)
            tm = mt.Tm_NN(seq, nn_table=mt.DNA_NN4)

            # 4. GC Content
            gc = gc_fraction(seq) * 100.0

            features.append([mfe, dg, tm, gc])
        return np.array(features, dtype=np.float32)

    def predict(self, sequences: list[str]) -> np.ndarray:
        embeddings = self.extract_nt_embeddings(sequences)
        physical_feats = self.compute_physical_features(sequences)

        final_features = np.hstack([embeddings, physical_feats])
        assert final_features.shape[1] == 8964, (
            f"Dimension mismatch: Expected 8964, got {final_features.shape[1]}"
        )

        preds = self.xgb_model.predict(final_features)
        return np.array(preds).flatten()
