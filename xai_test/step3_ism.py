import os

import numpy as np
import pandas as pd
from ism_sweep import run_ism_sweep
from model_a_wrapper import Model_A_Predictor
from model_b_wrapper import Model_B_Predictor


def main():
    print("=== [Step 3] In-silico Saturation Mutagenesis (ISM) Sweep ===")

    metadata_path = "test_metadata.csv"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} not found. test_dataset_verifier.py를 먼저 실행해 주세요."
        )

    meta_df = pd.read_csv(metadata_path)
    sequences = meta_df["sequence"].tolist()
    print(f"Loaded testset: {len(sequences)} samples.")

    model_a_weight = "best_model_fold1.pth"
    nt_backbone_dir = "./NT_sacas9_fintuned_model"
    xgb_weight_path = "hybrid_xgb_model.json"

    print("\nInitializing Model A Predictor (CNN+RNN)...")
    predictor_a = Model_A_Predictor(weight_path=model_a_weight)

    print("\nInitializing Model B Predictor (NT 500M + 4 Phys + XGBoost)...")
    predictor_b = Model_B_Predictor(
        nt_model_dir=nt_backbone_dir, xgb_model_path=xgb_weight_path
    )

    print(
        f"\nRunning ISM sweep: {len(sequences)} sequences x 36 positions x 3 alt bases "
        f"({len(sequences) * 36 * 3} mutants per model)..."
    )
    ism_result = run_ism_sweep(sequences, predictor_a, predictor_b)

    for name, ism_delta in ism_result.items():
        print(
            f" - {name}: shape={ism_delta.shape}, NaN-masked={np.isnan(ism_delta).any(axis=(1, 2)).sum()} samples"
        )

    output_path = "ism_delta.npz"
    np.savez(output_path, **ism_result)
    print(f"\nSaved ISM sensitivity arrays to {output_path}")
    print(
        "(Final combined export into model_analysis_arrays.npz / _summary.json is issue #17's job.)"
    )

    print("\n=== Step 3 (ISM) Execution Completed ===")


if __name__ == "__main__":
    main()
