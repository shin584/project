import json
import os

import numpy as np
import pandas as pd
from ism_sweep import DEFAULT_PREDICT_BATCH_SIZE
from mismatch_profiling import (
    generate_mismatch_complex_scenarios,
    run_mismatch_profiling,
    scenario_to_metadata,
)
from model_a_wrapper import Model_A_Predictor
from model_b_wrapper import Model_B_Predictor

PREDICT_BATCH_SIZE = DEFAULT_PREDICT_BATCH_SIZE


def main():
    print("=== [Step 3] Mismatch Tolerance Profiling (Single + Complex) ===")

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

    scenarios = generate_mismatch_complex_scenarios()
    print(
        f"\nGenerated {len(scenarios)} fixed complex mismatch scenarios "
        f"(random_seed=42): {sum(s.region == 'Seed' for s in scenarios)} Seed, "
        f"{sum(s.region == 'Distal' for s in scenarios)} Distal, "
        f"{sum(s.region == 'Intermittent' for s in scenarios)} Intermittent."
    )

    print(
        f"\nRunning mismatch profiling: {len(sequences)} sequences x 36 positions x 3 "
        f"alt bases (single) + {len(scenarios)} complex scenarios, per model..."
    )
    result = run_mismatch_profiling(
        sequences,
        predictor_a,
        predictor_b,
        scenarios=scenarios,
        predict_batch_size=PREDICT_BATCH_SIZE,
    )

    for name, array in result.items():
        print(f" - {name}: shape={array.shape}")

    output_path = "mismatch_delta.npz"
    np.savez(output_path, **result)
    print(f"\nSaved mismatch profiling arrays to {output_path}")

    example_sequence = sequences[0]
    complex_metadata = [
        scenario_to_metadata(
            scenario, scenario_index=i, example_sequence=example_sequence
        )
        for i, scenario in enumerate(scenarios)
    ]
    metadata_output_path = "mismatch_complex_metadata.json"
    with open(metadata_output_path, "w") as f:
        json.dump(complex_metadata, f, indent=2)
    print(f"Saved complex scenario metadata to {metadata_output_path}")
    print(
        "(Final combined export into model_analysis_arrays.npz / _summary.json is issue #17's job.)"
    )

    print("\n=== Step 3 (Mismatch Profiling) Execution Completed ===")


if __name__ == "__main__":
    main()
