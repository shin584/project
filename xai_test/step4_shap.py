import json
import os

import numpy as np
import pandas as pd
from ism_sweep import DEFAULT_PREDICT_BATCH_SIZE, dispatch_in_batches
from model_b_wrapper import Model_B_Predictor
from shap_analysis import run_shap_analysis

TEST_METADATA_PATH = "test_metadata.csv"
NT_MODEL_DIR = "./NT_sacas9_fintuned_model"
XGB_MODEL_PATH = "hybrid_xgb_model.json"
DEPENDENCE_PLOT_DIR = "shap_dependence_plots"
OUTPUT_JSON_PATH = "shap_results.json"
OUTPUT_NPZ_PATH = "shap_values.npz"


def _build_features(predictor: Model_B_Predictor, sequences: list[str]) -> np.ndarray:
    # Neither extract_nt_embeddings() nor compute_physical_features() batches
    # internally - same chunking precedent as step3_ism.py / step4_ablation.py,
    # here bounding a single 514-sample Testset forward pass instead of a
    # multi-thousand-sequence ablation training set.
    embeddings = dispatch_in_batches(
        predictor.extract_nt_embeddings,
        sequences,
        DEFAULT_PREDICT_BATCH_SIZE,
        stack=np.vstack,
    )
    physical = predictor.compute_physical_features(sequences)
    return np.hstack([embeddings, physical])


def main():
    print("=== [Step 4-2] Tree SHAP + Physical-Feature Dependence Plots ===")

    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"{TEST_METADATA_PATH} not found. test_dataset_verifier.py를 먼저 실행해 주세요."
        )

    meta_df = pd.read_csv(TEST_METADATA_PATH)
    sequences = meta_df["sequence"].tolist()
    print(f"Loaded testset: {len(sequences)} samples.")

    print("\nInitializing Model B Predictor (production NT 500M + 4 Phys + XGBoost)...")
    predictor = Model_B_Predictor(
        nt_model_dir=NT_MODEL_DIR, xgb_model_path=XGB_MODEL_PATH
    )

    print("\nBuilding the full 8,964-feature Testset input matrix...")
    X_test = _build_features(predictor, sequences)
    assert X_test.shape == (len(sequences), 8964), (
        f"Expected ({len(sequences)}, 8964), got {X_test.shape}"
    )

    print(
        "\nComputing Tree SHAP values and dependence plots over the full feature matrix..."
    )
    result = run_shap_analysis(
        predictor.xgb_model, X_test, dependence_plot_dir=DEPENDENCE_PLOT_DIR
    )

    print("\n=== Top-20 Sequence-Embedding Features (mean |SHAP|) ===")
    for entry in result["top_embedding_features"]:
        print(f" - feature[{entry['feature_index']}]: {entry['mean_abs_shap']:.6f}")
    print("\n=== Physical Feature SHAP (mean |SHAP|) ===")
    for entry in result["physical_feature_shap"]:
        print(f" - {entry['feature_name']}: {entry['mean_abs_shap']:.6f}")
    print(
        f"\nAggregate physical SHAP contribution ratio: {result['physical_contribution_ratio']:.6f}"
    )
    print(f"\nDependence plots saved to {DEPENDENCE_PLOT_DIR}/:")
    for name, path in result["dependence_plot_paths"].items():
        print(f" - {name}: {path}")

    np.savez(OUTPUT_NPZ_PATH, shap_values=result["shap_values"])
    print(f"\nSaved full SHAP value matrix to {OUTPUT_NPZ_PATH}")

    summary = {
        "top_embedding_features": result["top_embedding_features"],
        "physical_feature_shap": result["physical_feature_shap"],
        "physical_contribution_ratio": result["physical_contribution_ratio"],
        "dependence_plot_paths": result["dependence_plot_paths"],
    }
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved SHAP summary to {OUTPUT_JSON_PATH}")
    print(
        "(Final combined export into model_analysis_summary.json is issue #17's job.)"
    )

    print("\n=== Step 4-2 (Tree SHAP) Execution Completed ===")


if __name__ == "__main__":
    main()
