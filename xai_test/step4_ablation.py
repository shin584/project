import json
import os

import numpy as np
import pandas as pd
from ablation import run_ablation
from dev_split import build_dev_split
from ism_sweep import DEFAULT_PREDICT_BATCH_SIZE, dispatch_in_batches
from model_b_wrapper import Model_B_Predictor

FULL_DATASET_PATH = "SaCas9_v4.xlsx"
TEST_METADATA_PATH = "test_metadata.csv"
NT_MODEL_DIR = "./NT_sacas9_fintuned_model"
XGB_MODEL_PATH = "hybrid_xgb_model.json"


def _extract_nt_embeddings_in_batches(
    predictor: Model_B_Predictor, sequences: list[str], batch_size: int
) -> np.ndarray:
    # extract_nt_embeddings() runs the whole list through the NT transformer
    # in one forward pass - same unbounded-memory shape as Model_B_Predictor's
    # predict(), which is why the ISM sweep (issue #11) had to chunk its own
    # calls externally. train.csv-scale inputs here (~4600 sequences) hit the
    # same risk, so batch here rather than inside the wrapper (reusing
    # ism_sweep's shared chunking helper rather than a second hand-rolled loop).
    if not sequences:
        return np.empty((0, 0))
    return dispatch_in_batches(
        predictor.extract_nt_embeddings, sequences, batch_size, stack=np.vstack
    )


def _build_features(predictor: Model_B_Predictor, sequences: list[str]) -> np.ndarray:
    embeddings = _extract_nt_embeddings_in_batches(
        predictor, sequences, DEFAULT_PREDICT_BATCH_SIZE
    )
    physical = predictor.compute_physical_features(sequences)
    return np.hstack([embeddings, physical])


def main():
    print("=== [Step 4-1] Physical Feature Ablation + Paired Bootstrap ===")

    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"{TEST_METADATA_PATH} not found. test_dataset_verifier.py를 먼저 실행해 주세요."
        )
    if not os.path.exists(FULL_DATASET_PATH):
        raise FileNotFoundError(f"{FULL_DATASET_PATH} not found.")

    print(
        "\nBuilding dev split (train/val) from the seed=42 permutation, disjoint from the Testset..."
    )
    train_df, val_df = build_dev_split(FULL_DATASET_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    print("\nInitializing Model B feature extractor (NT 500M + 4 Phys)...")
    predictor = Model_B_Predictor(
        nt_model_dir=NT_MODEL_DIR, xgb_model_path=XGB_MODEL_PATH
    )

    print("\nExtracting features for train/val/test (this can take a while on CPU)...")
    X_train = _build_features(predictor, train_df["sequence"].tolist())
    X_val = _build_features(predictor, val_df["sequence"].tolist())
    X_test = _build_features(predictor, test_df["sequence"].tolist())
    y_train = train_df["true_score"].to_numpy()
    y_val = val_df["true_score"].to_numpy()
    y_test = test_df["true_score"].to_numpy()

    print(
        f"\nTraining 3 ablation variants (Embedding/Physical/Full) under the shared "
        f"XGBoost protocol, evaluating on the {len(test_df)}-sample Testset..."
    )
    result = run_ablation(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n=== Ablation Results (Testset) ===")
    for variant, metrics in result["ablation_results"].items():
        print(f" - {variant}: {metrics}")
    print("\n=== Paired Bootstrap 95% CI (Full vs each variant) ===")
    for key, ci in result["paired_bootstrap_95ci"].items():
        print(f" - {key}: [{ci[0]:.4f}, {ci[1]:.4f}]")

    output_path = "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved ablation results to {output_path}")
    print(
        "(Final combined export into model_analysis_summary.json is issue #17's job.)"
    )

    print("\n=== Step 4-1 (Ablation) Execution Completed ===")


if __name__ == "__main__":
    main()
