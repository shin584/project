import pandas as pd
import pytest
from model_b_wrapper import Model_B_Predictor
from model_b_xai_wrapper import Model_B_XAIPredictor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

NT_MODEL_DIR = "./NT_sacas9_fintuned_model"
XGB_MODEL_PATH = "hybrid_xgb_model.json"

# First 3 rows of test_metadata.csv (sample_id 0-2), 36bp SaCas9 sequences.
TESTSET_SEQUENCES = [
    "CCCGGCTGACCTGCCCAAGCTGGTGGAGGGGCTGAA",
    "TTGAAGGAAGGGAATCCAGGTGTGTAAGGGTCACCT",
    "GGAGGAGGAAGAAGAACTGGAAGAGGTGGAAGACCT",
]

# Model_B_Predictor's aggregate metrics over the full 514-sample Testset
# (test_metadata.csv), per step1_verify.py's own methodology, captured before
# Model_B_XAIPredictor existed. AC4 requires predict() stay pinned to these.
EXPECTED_MODEL_B_SPEARMAN = 0.8963391736653834
EXPECTED_MODEL_B_PEARSON = 0.9046887504030652
EXPECTED_MODEL_B_MAE = 0.08033443220798515
EXPECTED_MODEL_B_MSE = 0.010395696335958235


@pytest.fixture(scope="module")
def xai_predictor():
    return Model_B_XAIPredictor(nt_model_dir=NT_MODEL_DIR)


def test_token_embeddings_shape_and_requires_grad(xai_predictor):
    embeddings = xai_predictor.get_token_embeddings(TESTSET_SEQUENCES)

    assert embeddings.shape == (len(TESTSET_SEQUENCES), 7, 1280)
    assert embeddings.requires_grad


def test_token_embeddings_malformed_input_raises(xai_predictor):
    # 12bp sequence tokenizes to 3 tokens (<cls> + two 6-mers), not 7.
    with pytest.raises(AssertionError):
        xai_predictor.get_token_embeddings(["ATCGATCGATCG"])


def test_attentions_per_layer_with_cls_present(xai_predictor):
    attentions = xai_predictor.get_attentions(TESTSET_SEQUENCES)

    assert isinstance(attentions, tuple)
    assert len(attentions) == 24  # num_hidden_layers, per config.json
    for layer_attention in attentions:
        # (batch, num_attention_heads, seq_len, seq_len); seq_len includes <cls>.
        assert layer_attention.shape == (len(TESTSET_SEQUENCES), 20, 7, 7)


def test_model_b_predictor_predict_unchanged():
    meta_df = pd.read_csv("test_metadata.csv")
    sequences = meta_df["sequence"].tolist()
    true_scores = meta_df["true_score"].values

    predictor = Model_B_Predictor(
        nt_model_dir=NT_MODEL_DIR, xgb_model_path=XGB_MODEL_PATH
    )
    preds = predictor.predict(sequences)

    assert spearmanr(true_scores, preds)[0] == pytest.approx(
        EXPECTED_MODEL_B_SPEARMAN, abs=1e-6
    )
    assert pearsonr(true_scores, preds)[0] == pytest.approx(
        EXPECTED_MODEL_B_PEARSON, abs=1e-6
    )
    assert mean_absolute_error(true_scores, preds) == pytest.approx(
        EXPECTED_MODEL_B_MAE, abs=1e-6
    )
    assert mean_squared_error(true_scores, preds) == pytest.approx(
        EXPECTED_MODEL_B_MSE, abs=1e-6
    )
