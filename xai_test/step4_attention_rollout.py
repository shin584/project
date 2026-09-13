import json
import os

import numpy as np
import pandas as pd
from attention_rollout import run_attention_rollout
from model_b_xai_wrapper import Model_B_XAIPredictor

TEST_METADATA_PATH = "test_metadata.csv"
NT_MODEL_DIR = "./NT_sacas9_fintuned_model"
OUTPUT_NPZ_PATH = "attention_rollout_matrix.npz"
OUTPUT_JSON_PATH = "attention_rollout_summary.json"


def main():
    print("=== [Step 4-4] Attention Rollout (Information Dependency Matrix) ===")

    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"{TEST_METADATA_PATH} not found. test_dataset_verifier.py를 먼저 실행해 주세요."
        )

    meta_df = pd.read_csv(TEST_METADATA_PATH)
    sequences = meta_df["sequence"].tolist()
    print(f"Loaded testset: {len(sequences)} samples.")

    print("\nInitializing Model B XAI Predictor (NT sequence-classification head)...")
    xai_predictor = Model_B_XAIPredictor(nt_model_dir=NT_MODEL_DIR)

    print(
        f"\nRunning Attention Rollout over {len(sequences)} sequences "
        f"(fixed order: 0.5*A + 0.5*I, then row-normalize, per layer)..."
    )
    result = run_attention_rollout(xai_predictor, sequences)
    rollout_matrix = result["attention_rollout"]
    assert rollout_matrix.shape == (len(sequences), 36, 36), (
        f"Expected ({len(sequences)}, 36, 36), got {rollout_matrix.shape}"
    )

    np.savez(OUTPUT_NPZ_PATH, attention_rollout=rollout_matrix)
    print(f"\nSaved per-sample Information Dependency matrices to {OUTPUT_NPZ_PATH}")

    summary = {
        "token_matrix_dim": result["token_matrix_dim"],
        "projected_nucleotide_matrix_dim": result["projected_nucleotide_matrix_dim"],
        "num_samples": len(sequences),
        "note": result["note"],
    }
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {OUTPUT_JSON_PATH}")
    print(
        "(Final combined export into model_analysis_arrays.npz / _summary.json is issue #17's job.)"
    )

    print("\n=== Step 4-4 (Attention Rollout) Execution Completed ===")


if __name__ == "__main__":
    main()
