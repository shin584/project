# Separate Model_B_XAIPredictor instead of extending Model_B_Predictor

`Model_B_Predictor` is the production inference wrapper: it loads the NT checkpoint only for its frozen encoder hidden states and calls XGBoost for the final score. The XAI analysis pipeline (Integrated Gradients, Attention Rollout) needs different things from the same checkpoint: gradients through the checkpoint's own regression head, and `output_attentions=True` transformer internals — neither of which production inference ever needs.

We considered adding these as optional flags/methods on `Model_B_Predictor` itself, but decided to add a separate `Model_B_XAIPredictor` class instead. Reasoning: production inference stays simple and fast with no XAI-only branches to reason about, and the two classes can diverge (e.g. different loading classes — `AutoModel` vs `AutoModelForSequenceClassification` — for the same checkpoint path) without one's changes risking the other's correctness.
