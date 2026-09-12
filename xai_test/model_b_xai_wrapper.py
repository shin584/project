import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Model_B_XAIPredictor:
    """XAI-only counterpart to `Model_B_Predictor` (see ADR-0001).

    Loads the same NT checkpoint via `AutoModelForSequenceClassification`
    (not `AutoModelForMaskedLM`) so gradients flow through the checkpoint's
    own regression head and `output_attentions=True` is available. Has no
    analysis behavior of its own; Integrated Gradients and Attention Rollout
    build on top of it. Must never modify or risk `Model_B_Predictor`'s
    production inference path.
    """

    EXPECTED_SEQ_LEN = 7
    EXPECTED_HIDDEN_DIM = 1280

    def __init__(self, nt_model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(
            f"Loading NT backbone (XAI, sequence-classification head) from {nt_model_dir}..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(nt_model_dir)
        # "eager" attention is required for output_attentions=True; the default
        # "sdpa" backend silently returns no attention weights instead.
        self.nt_model = AutoModelForSequenceClassification.from_pretrained(
            nt_model_dir, attn_implementation="eager"
        ).to(self.device)
        self.nt_model.eval()

    def _tokenize(self, sequences: list[str]):
        return self.tokenizer(
            sequences, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

    def get_token_embeddings(self, sequences: list[str]) -> torch.Tensor:
        """Return the last-layer token embeddings, gradients enabled.

        Shape is asserted here, before any consumer (e.g. Integrated
        Gradients) uses the tensor, so a malformed input raises immediately
        instead of silently propagating a wrong shape downstream.
        """
        inputs = self._tokenize(sequences)
        outputs = self.nt_model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]

        _batch_size, seq_len, hidden_dim = embeddings.shape
        assert seq_len == self.EXPECTED_SEQ_LEN, (
            f"Expected {self.EXPECTED_SEQ_LEN} tokens, got {seq_len}"
        )
        assert hidden_dim == self.EXPECTED_HIDDEN_DIM, (
            f"Expected {self.EXPECTED_HIDDEN_DIM} hidden dim, got {hidden_dim}"
        )
        assert embeddings.requires_grad, (
            "Embeddings must carry gradients for Integrated Gradients to run"
        )

        return embeddings

    def classify_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Run only the classification head on a `(batch, 7, 1280)` embedding tensor.

        Lets a caller (e.g. Integrated Gradients) backprop from a chosen
        embedding tensor through just the head, instead of replaying the
        full encoder - or reaching into `nt_model`'s internals itself - for
        every interpolation step.
        """
        return self.nt_model.classifier(embeddings).squeeze(-1)

    def get_attentions(self, sequences: list[str]) -> tuple[torch.Tensor, ...]:
        """Return per-layer attention matrices, `<cls>` still included.

        Stripping `<cls>` and mapping to nucleotide coordinates happens
        downstream, in Attention Rollout.
        """
        inputs = self._tokenize(sequences)
        outputs = self.nt_model(**inputs, output_attentions=True)
        return outputs.attentions
