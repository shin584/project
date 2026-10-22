"""Build `colab_grouped_model_b.ipynb` (issue #37).

Embeds the committed `NOTEBOOK_MODULES` with `%%writefile` cells, so the
notebook runs exactly the repo's code on Colab without fetching anything.
Each embedded module is read back from the written notebook and checked
byte-identical (LF line endings, as git stores them) to the repo file.

Usage: python build_colab_grouped_model_b.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "colab_grouped_model_b.ipynb"
# Committed modules the notebook embeds with `%%writefile`, in dependency order.
NOTEBOOK_MODULES = (
    "test_dataset_verifier.py",
    "grouped_split.py",
    "grouped_testset.py",
    "model_b_wrapper.py",
    "ism_sweep.py",
    "paired_bootstrap.py",
    "ablation.py",
    "grouped_model_b.py",
)


def _repo_source(name: str) -> str:
    # core.autocrlf may check files out with CRLF; git stores LF.
    return (BASE_DIR / name).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def _cell(cell_type: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def md(source: str) -> dict:
    return _cell("markdown", source.strip("\n"))


def code(source: str) -> dict:
    return _cell("code", source.strip("\n"))


INTRO = """
# Leak-free Model B on the grouped split (issue #37)

Fine-tunes NT and trains Model B's XGBoost heads on the sequence-grouped
split in `grouped_split/` (#34/#35): 1,445 train sequences (4,118 rows), 180
val (513 rows), 180 test (514 rows), with no sequence shared between splits.
Production Model B's backbone was fine-tuned on the stale
`train.csv`/`val.csv`, which contain almost all Testset sequences.

- **Fine-tune:** `nt_pretrain.py`'s recipe unchanged (`grouped_model_b.FINETUNE_ARGS`),
  on grouped train rows, early-stopped on grouped val rows.
- **XGBoost heads** (`ablation.XGB_HYPERPARAMS`, early-stopped on grouped val),
  on the `Model_B_Predictor` features (8,960-d flattened last hidden state + 4
  physical features):
  - **all-rows:** one sample per train row (as in production);
  - **seq-mean:** one sample per train sequence, labels averaged across replicates.

**One-time setup:** upload to a Google Drive folder, keeping the layout:
`SaCas9_v4.xlsx`, `hybrid_xgb_model.json` (only to construct
`Model_B_Predictor`, whose constructor needs an XGBoost file; its head is not
used here), and `grouped_split/assignments.csv`,
`grouped_split/test_metadata.csv`, `grouped_split/test_checksum.json`. Set
`DRIVE_DIR` below to that folder.

**Outputs:**
- `NT_sacas9_grouped_model/` on Drive (~2 GB): the best fine-tuned checkpoint.
  Download it to an untracked local `xai_test/NT_sacas9_grouped_model/`; it
  loads with `Model_B_Predictor` like `NT_sacas9_fintuned_model/`.
- `grouped_model_b/` (also on Drive, and downloaded as a zip):
  `xgb_all_rows.json`, `xgb_seq_mean.json`, `test_predictions.csv`,
  `training_log.json`. Copy it into your local `xai_test/` and commit it.
  Embeddings are not saved.
"""

SETUP = [
    md("## 1. Mount Google Drive"),
    code(
        """
from google.colab import drive

drive.mount("/content/drive")
"""
    ),
    md(
        """
## 2. Point at your uploaded assets

Edit `DRIVE_DIR` to the Drive folder holding the files listed above.
"""
    ),
    code(
        """
import os

DRIVE_DIR = "/content/drive/MyDrive/xai_test"  # <-- EDIT ME

required = [
    "SaCas9_v4.xlsx",
    "hybrid_xgb_model.json",
    "grouped_split/assignments.csv",
    "grouped_split/test_metadata.csv",
    "grouped_split/test_checksum.json",
]
missing = [
    name for name in required if not os.path.exists(os.path.join(DRIVE_DIR, name))
]
if missing:
    raise FileNotFoundError(
        f"Missing in {DRIVE_DIR}: {missing}. "
        "Upload these to Drive first, or fix DRIVE_DIR."
    )

os.chdir(DRIVE_DIR)
print("Working directory:", os.getcwd())
"""
    ),
    md(
        """
## 3. Install ViennaRNA CLI (RNAfold / RNAduplex) and Python deps

`Model_B_Predictor.compute_physical_features` needs both binaries on `PATH`.
`nt_pretrain.py` upgraded `transformers`/`accelerate` before fine-tuning; so does this.
"""
    ),
    code(
        """
!apt-get -qq update && apt-get -qq install -y vienna-rna
!which RNAfold && which RNAduplex
"""
    ),
    code(
        "!pip install -q --upgrade transformers accelerate\n"
        "!pip install -q biopython xgboost openpyxl"
    ),
    md(
        """
## 4. Write the repo modules

Embedded from the committed `xai_test/` repo at notebook-build time
(`build_colab_grouped_model_b.py` checks them byte-identical), not fetched at runtime.
"""
    ),
]

RUN = [
    md("## 5. Confirm GPU is available"),
    code(
        """
import torch

assert torch.cuda.is_available(), (
    "Switch the runtime to GPU (Runtime > Change runtime type)."
)
print(torch.cuda.get_device_name(0))
"""
    ),
    md(
        """
## 6. Load the grouped split and verify it

Joins `SaCas9_v4.xlsx` to `grouped_split/assignments.csv` on `original_id`,
then refuses to continue unless train/val/test share no sequence and the test
rows are exactly `grouped_split/test_metadata.csv` (checksum from
`grouped_split/test_checksum.json`).
"""
    ),
    code(
        """
import json

import pandas as pd
from grouped_model_b import (
    join_grouped_splits,
    verify_grouped_testset,
    verify_no_sequence_overlap,
)
from grouped_split import load_full_dataset

full_df = load_full_dataset("SaCas9_v4.xlsx")
assignments = pd.read_csv("grouped_split/assignments.csv")
splits = join_grouped_splits(full_df, assignments)
verify_no_sequence_overlap(splits)

testset_metadata = pd.read_csv("grouped_split/test_metadata.csv")
with open("grouped_split/test_checksum.json", encoding="utf-8") as f:
    checksum_record = json.load(f)
verify_grouped_testset(splits["test"], testset_metadata, checksum_record)

train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
n_rows = {name: len(df) for name, df in splits.items()}
assert n_rows == {"train": 4118, "val": 513, "test": 514}, n_rows
print("Rows:", n_rows)
n_sequences = {name: df["normalized_sequence"].nunique() for name, df in splits.items()}
print("Sequences:", n_sequences)
print("Zero sequence overlap; Testset matches grouped_split/test_metadata.csv.")
"""
    ),
    md(
        """
## 7. Fine-tune NT (`nt_pretrain.py` recipe, grouped data)

Up to 10 epochs with early stopping (patience 3) on val loss; the best
checkpoint is reloaded at the end and saved to `DRIVE_DIR/NT_sacas9_grouped_model/`
with its tokenizer. Per-epoch checkpoints go to local Colab disk and are deleted
afterwards. The Trainer's `log_history` is also saved to Drive right away.

If the runtime disconnects after this step, rerun steps 1-6 with
`REUSE_SAVED_CHECKPOINT = True` to reuse the saved checkpoint and log instead
of fine-tuning again. Leave it `False` for a fresh run.
"""
    ),
    code(
        """
import shutil
import time

from grouped_model_b import (
    CHECKPOINT_DIRNAME,
    SEED,
    finetune_nt,
    summarize_finetune_history,
)

REUSE_SAVED_CHECKPOINT = False
LOCAL_CHECKPOINTS = "/content/nt_grouped_checkpoints"
checkpoint_dir = os.path.join(DRIVE_DIR, CHECKPOINT_DIRNAME)
log_history_path = os.path.join(DRIVE_DIR, "grouped_finetune_log_history.json")

if REUSE_SAVED_CHECKPOINT:
    for path in (log_history_path, os.path.join(checkpoint_dir, "config.json")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nothing to reuse: {path} is missing")
    print(f"Reusing {checkpoint_dir} and {log_history_path}")
    with open(log_history_path, encoding="utf-8") as f:
        finetune_log_history = json.load(f)
else:
    t0 = time.time()
    trainer, tokenizer = finetune_nt(train_df, val_df, LOCAL_CHECKPOINTS, seed=SEED)
    print(f"Fine-tune completed in {(time.time() - t0) / 60:.1f} min")

    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    finetune_log_history = trainer.state.log_history
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(finetune_log_history, f, indent=2)
    print(f"Best checkpoint saved to {checkpoint_dir}")

    del trainer
    torch.cuda.empty_cache()
    shutil.rmtree(LOCAL_CHECKPOINTS, ignore_errors=True)

finetune_summary = summarize_finetune_history(finetune_log_history)
for epoch in finetune_summary["epochs"]:
    print(epoch)
print(
    f"Best epoch {finetune_summary['best_epoch']}, "
    f"val loss {finetune_summary['best_eval_loss']:.4f}"
)
"""
    ),
    md(
        """
## 8. Extract features with `Model_B_Predictor`

Loads the saved checkpoint exactly as the local `Model_B_Predictor` will, and
builds features with its own `extract_nt_embeddings` + `compute_physical_features`
(batched with `ism_sweep.dispatch_in_batches`, as in `colab_ablation.ipynb`).
Features depend only on the sequence, so each distinct sequence is embedded
once and shared by its replicate rows. Lower `PREDICT_BATCH_SIZE` on CUDA OOM.
"""
    ),
    code(
        """
import numpy as np
from ism_sweep import dispatch_in_batches
from model_b_wrapper import Model_B_Predictor

predictor = Model_B_Predictor(
    nt_model_dir=checkpoint_dir, xgb_model_path="hybrid_xgb_model.json"
)

PREDICT_BATCH_SIZE = 128  # lower (e.g. 32) if you hit CUDA OOM


def build_features(model_b, sequences):
    embeddings = dispatch_in_batches(
        model_b.extract_nt_embeddings, sequences, PREDICT_BATCH_SIZE, stack=np.vstack
    )
    physical = model_b.compute_physical_features(sequences)
    return np.hstack([embeddings, physical])


unique_sequences = sorted(
    set(train_df["sequence"])
    | set(val_df["sequence"])
    | set(testset_metadata["sequence"])
)
t0 = time.time()
X_unique = build_features(predictor, unique_sequences)
print(f"Feature extraction completed in {(time.time() - t0) / 60:.1f} min")
row_of = {seq: i for i, seq in enumerate(unique_sequences)}


def features_for(df):
    return X_unique[[row_of[seq] for seq in df["sequence"]]]


X_train, y_train = features_for(train_df), train_df["true_score"].to_numpy()
X_val, y_val = features_for(val_df), val_df["true_score"].to_numpy()
X_test = features_for(testset_metadata)
assert X_unique.shape == (len(unique_sequences), 8964), X_unique.shape
print(f"X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

del predictor
torch.cuda.empty_cache()
"""
    ),
    md(
        """
## 9. Train the all-rows and seq-mean XGBoost heads
"""
    ),
    code(
        """
from grouped_model_b import train_heads
from paired_bootstrap import compute_metrics

heads = train_heads(
    X_train,
    y_train,
    train_df["normalized_sequence"],
    X_val,
    y_val,
    val_df["normalized_sequence"],
)
pred_all_rows = heads["all_rows"].model.predict(X_test)
pred_seq_mean = heads["seq_mean"].model.predict(X_test)

y_test = testset_metadata["true_score"].to_numpy()
for name, fit in heads.items():
    print(
        f"{name}: {fit.n_train_samples} train samples, "
        f"best_iteration={fit.model.best_iteration}"
    )
# Quick look only; the full evaluation is a separate step (issue #38).
print("Row-level Testset metrics, all-rows:", compute_metrics(y_test, pred_all_rows))
print("Row-level Testset metrics, seq-mean:", compute_metrics(y_test, pred_seq_mean))
"""
    ),
    md(
        """
## 10. Save `grouped_model_b/`

Written to `DRIVE_DIR/grouped_model_b/` and downloaded as a zip.
"""
    ),
    code(
        """
from grouped_model_b import build_test_predictions, build_training_log

os.makedirs("grouped_model_b", exist_ok=True)
heads["all_rows"].model.save_model("grouped_model_b/xgb_all_rows.json")
heads["seq_mean"].model.save_model("grouped_model_b/xgb_seq_mean.json")

predictions = build_test_predictions(testset_metadata, pred_all_rows, pred_seq_mean)
assert len(predictions) == 514
predictions.to_csv(
    "grouped_model_b/test_predictions.csv", index=False, lineterminator="\\n"
)

training_log = build_training_log(finetune_log_history, heads, seed=SEED, n_rows=n_rows)
with open("grouped_model_b/training_log.json", "w", encoding="utf-8") as f:
    json.dump(training_log, f, indent=2)

print(sorted(os.listdir("grouped_model_b")))
"""
    ),
    md(
        """
## 11. Check the checkpoint loads in `Model_B_Predictor`

Loads the Drive checkpoint with the all-rows head the way the local code will,
and predicts the Testset end to end. Embeddings are batched differently from
step 8, so GPU float noise can move a few predictions slightly (an XGBoost split
can tip); the check is on rank agreement, with the max difference printed.
"""
    ),
    code(
        """
from ism_sweep import predict_in_batches

reloaded = Model_B_Predictor(
    nt_model_dir=checkpoint_dir, xgb_model_path="grouped_model_b/xgb_all_rows.json"
)
reloaded_preds = predict_in_batches(
    reloaded, testset_metadata["sequence"].tolist(), PREDICT_BATCH_SIZE
)
from scipy.stats import spearmanr

max_diff = float(np.max(np.abs(reloaded_preds - pred_all_rows)))
agreement = spearmanr(reloaded_preds, pred_all_rows)[0]
print(f"Max |reloaded - pred_all_rows| = {max_diff:.2e}, Spearman {agreement:.6f}")
assert agreement > 0.999, "Reloaded Model_B_Predictor disagrees with pred_all_rows"
"""
    ),
    md("## 12. Download the results"),
    code(
        """
from google.colab import files

shutil.make_archive("/content/grouped_model_b", "zip", DRIVE_DIR, "grouped_model_b")
files.download("/content/grouped_model_b.zip")
"""
    ),
    md(
        """
Unzip into your local `xai_test/` (giving `xai_test/grouped_model_b/`) and commit
it. Download `NT_sacas9_grouped_model/` from Drive into `xai_test/` separately;
don't commit it.
"""
    ),
]


def build_notebook() -> dict:
    # Not `code()`: it strips the trailing newline the module bytes end with.
    module_cells = [
        _cell("code", f"%%writefile {name}\n{_repo_source(name)}")
        for name in NOTEBOOK_MODULES
    ]
    return {
        "cells": [md(INTRO), *SETUP, *module_cells, *RUN],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def check_embedded_modules(notebook: dict) -> list[str]:
    """Names of the `%%writefile` modules, each checked identical to the repo file."""
    embedded = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if not source.startswith("%%writefile "):
            continue
        header, _, body = source.partition("\n")
        name = header.removeprefix("%%writefile ")
        if body != _repo_source(name):
            raise AssertionError(f"Embedded {name} differs from the repo file")
        embedded.append(name)
    return embedded


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        check_embedded_modules(json.load(f))
    print(f"Wrote {NOTEBOOK_PATH} ({len(NOTEBOOK_MODULES)} embedded modules, verified)")


if __name__ == "__main__":
    main()
