"""Cancellation through real predictor handles, using tiny in-memory torch tensors."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predict
from xai import explain


class Cancelled(Exception):
    pass


@pytest.fixture(params=["Cas12a", "SpCas9"])
def predictor(request):
    torch = pytest.importorskip("torch")
    cas = request.param
    labels = 1 if cas == "Cas12a" else len(predict.SPCAS9_VARIANTS)

    class Tokenizer:
        all_special_tokens = []

        def __call__(self, seqs, **kwargs):
            if isinstance(seqs, str):
                return {"input_ids": ["ACGT".index(base) for base in seqs]}
            ids = torch.tensor([["ACGT".index(base) for base in seq] for seq in seqs])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def convert_ids_to_tokens(self, ids):
            return ["ACGT"[i] for i in ids]

    class Model:
        def __init__(self):
            embedding = torch.nn.Embedding.from_pretrained(
                torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8,
            )
            self.esm = SimpleNamespace(embeddings=SimpleNamespace(word_embeddings=embedding))

        def __call__(self, input_ids=None, inputs_embeds=None, attention_mask=None):
            x = self.esm.embeddings.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
            output = (x.square().sum(dim=-1) * attention_mask).sum(dim=-1) / 100
            return SimpleNamespace(logits=output[:, None].repeat(1, labels))

    # Substitute only pretrained resources; encoding, batching, gradients and projection stay real.
    p = predict.NTClassifierPredictor(predict.MODEL_REGISTRY[("A", cas)])
    p._model = Model()
    p._tokenizer = Tokenizer()
    p._num_labels = labels
    p._dtype = torch.float32
    return p


@pytest.mark.parametrize("method", ["ig", "shap"])
def test_cancelled_explanation_leaves_predictor_reusable(predictor, method):
    head = "Cas12a" if predictor.spec.cas == "Cas12a" else "SpCas9(WT)"
    handle = predictor.explain_handle(head)

    def compute(seq, on_batch=None):
        if method == "ig":
            return explain.integrated_gradients(
                handle, seq, steps=10, chunk_size=4, on_batch=on_batch).attribution
        # A non-default seed bypasses the precomputed SHAP cache.
        return explain.kernel_shap(
            handle, seq, nsamples=16, n_background=2, seed=7, on_batch=on_batch).values

    expected = compute("ACGT")
    cancelled = Cancelled("new request")

    def cancel(done, total):
        assert 0 < done < total
        raise cancelled

    with pytest.raises(Cancelled) as exc:
        compute("ACGTAC", on_batch=cancel)
    assert exc.value is cancelled
    np.testing.assert_allclose(compute("ACGT"), expected)
    assert predictor.explain_handle(head).raw_fn(
        np.array([predict.one_hot_encode_dna("ACGT")]))[0] == pytest.approx(0.021875)
