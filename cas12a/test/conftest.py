"""Shared offline NT model; heavyweight imports stay inside the fixture."""

import itertools

import pytest


@pytest.fixture
def tiny_nt(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    path = tmp_path / "generic-tiny-nt"
    path.mkdir()
    vocabulary = ["<cls>", "<pad>", "<mask>", "<unk>"]
    vocabulary += ["".join(bases) for bases in itertools.product("ACGT", repeat=6)]
    vocabulary += list("ACGTN")
    (path / "vocab.txt").write_text("\n".join(vocabulary) + "\n")
    tokenizer = transformers.EsmTokenizer(str(path / "vocab.txt"), eos_token=None)
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    config = transformers.EsmConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
        pad_token_id=1,
        mask_token_id=2,
        token_dropout=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    transformers.EsmForMaskedLM(config).save_pretrained(path)
    return path
