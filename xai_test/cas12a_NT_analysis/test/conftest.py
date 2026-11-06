import itertools
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from cas12a.data import SEQUENCE_COLUMN, TARGET_COLUMN
from cas12a.nt_data import SplitConfig, canonical_bytes, partition_library, read_library
from xai_test.cas12a_NT_analysis.nt_run import file_hash


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "synthetic.csv"
    sequences = [
        "ACGT"
        + ("TTTA" if i % 2 else "TTTT")
        + "".join("ACGT"[(i >> shift) & 3] for shift in (4, 2, 0))
        + "A" * 20
        + "CGA"
        for i in range(10)
    ]
    pd.DataFrame(
        {SEQUENCE_COLUMN: sequences, TARGET_COLUMN: [i * 10 for i in range(10)]}
    ).to_csv(source, index=False)
    manifest = tmp_path / "partition.json"
    data = read_library(source, SplitConfig.fixture(4, 4, 2))
    partition = partition_library(data, SplitConfig.fixture(4, 4, 2))
    manifest.write_bytes(canonical_bytes(partition))
    export = tmp_path / "selected_export"
    export.mkdir()
    # A saved tiny scalar ESM uses the real vocabulary-size/embedding contract.
    import torch
    from transformers import EsmConfig, EsmForSequenceClassification, EsmTokenizer

    vocab = ["<unk>", "<pad>", "<mask>", "<cls>", "N"]
    vocab += ["".join(bases) for bases in itertools.product("ACGT", repeat=6)]
    vocab += list("ACGT") + ["<eos>", "<bos>"]
    (export / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = EsmTokenizer(vocab_file=str(export / "vocab.txt"), eos_token=None)
    tokenizer.save_pretrained(export)
    torch.manual_seed(42)
    model = EsmForSequenceClassification(
        EsmConfig(
            vocab_size=4105,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
            num_labels=1,
            problem_type="regression",
            pad_token_id=1,
            mask_token_id=2,
            token_dropout=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )
    model.save_pretrained(export)
    receipt = tmp_path / "fixture_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "validated": True,
                "files": {
                    f"selected_export/{p.name}": file_hash(p)
                    for p in export.iterdir()
                    if p.is_file()
                },
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        source=source,
        manifest=manifest,
        export=export,
        receipt=receipt,
        partition=partition,
        rows=data["rows"],
    )
