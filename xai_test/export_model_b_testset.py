"""Runner: Model B predictions + physical features for all 514 Testset samples (issue #19).

Run on Colab GPU (`colab_model_b_testset_export.ipynb`) - the NT forward pass
over the whole Testset does not fit this machine's RAM. Requires (in the
working directory):

- `test_metadata.csv` (issue #1, `test_dataset_verifier.py`)
- `NT_sacas9_fintuned_model/` + `hybrid_xgb_model.json` (Model B)
- `final_analysis_result/model_analysis_summary.json` - the committed copy of
  `step5_export.py`'s output (issue #17; step5 itself writes it to the
  working directory)

Writes `final_analysis_result/model_b_testset_predictions.csv` only after its
15 Case Study rows are confirmed to match the values already in
`model_analysis_summary.json`.
"""

import json
import os

import pandas as pd
from model_b_testset_export import (
    build_model_b_testset_table,
    check_case_studies_match,
    write_model_b_testset_table,
)
from model_b_wrapper import Model_B_Predictor

TEST_METADATA_PATH = "test_metadata.csv"
NT_MODEL_DIR = "./NT_sacas9_fintuned_model"
XGB_MODEL_PATH = "hybrid_xgb_model.json"
SUMMARY_PATH = os.path.join("final_analysis_result", "model_analysis_summary.json")
OUTPUT_PATH = os.path.join("final_analysis_result", "model_b_testset_predictions.csv")


def _require(path: str, produced_by: str) -> str:
    # Same phrasing as step5_export.py's `_require`.
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. {produced_by}를 먼저 실행해 주세요."
        )
    return path


def main():
    print("=== Model B Testset Export (predictions + physical features) ===")

    meta_df = pd.read_csv(_require(TEST_METADATA_PATH, "test_dataset_verifier.py"))
    with open(_require(SUMMARY_PATH, "step5_export.py"), encoding="utf-8") as f:
        case_studies = json.load(f)["case_studies"]
    sequences = meta_df["sequence"].tolist()
    print(f"Loaded testset: {len(sequences)} samples.")

    predictor_b = Model_B_Predictor(
        nt_model_dir=NT_MODEL_DIR, xgb_model_path=XGB_MODEL_PATH
    )

    print("\nScoring the Testset with Model B...")
    pred_b = predictor_b.predict(sequences)
    print("Computing physical features (MFE, dG, Tm, GC)...")
    physical = predictor_b.compute_physical_features(sequences)

    table = build_model_b_testset_table(meta_df, pred_b, physical)

    print(
        f"\nChecking the {len(case_studies)} Case Study rows against {SUMMARY_PATH}..."
    )
    check_case_studies_match(table, case_studies)
    print(" - All Case Study rows match.")

    print(f"\nWriting {OUTPUT_PATH}...")
    write_model_b_testset_table(table, OUTPUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
