import pytest

from xai_test.cas12a_NT_analysis.nt_inputs import verify_inputs


def test_verification_returns_only_existing_validation_membership(inputs):
    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    assert [row["source_row"] for row in verified.rows] == sorted(
        inputs.partition["partitions"]["validation"]
    )
    assert {row["source_row"] for row in verified.rows}.isdisjoint(
        inputs.partition["partitions"]["test"]
    )
    with pytest.raises(ValueError, match="fixture"):
        verify_inputs(inputs.source, inputs.manifest, inputs.export, inputs.receipt)
    (inputs.export / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="export hash"):
        verify_inputs(
            inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
        )


def test_actual_batch_rejects_out_of_embedding_special_and_preserves_spans(inputs):
    from transformers import AutoTokenizer

    from xai_test.cas12a_NT_analysis.nt_inputs import (
        encode_sequences,
        validate_vocabulary,
    )

    tokenizer = AutoTokenizer.from_pretrained(inputs.export, local_files_only=True)
    validate_vocabulary(tokenizer, 4105)
    sequence = inputs.rows[0]["sequence"]
    encoded = encode_sequences(tokenizer, [sequence], 4105)[0]
    assert encoded["dna_spans"] == [
        [0, 6],
        [6, 12],
        [12, 18],
        [18, 24],
        [24, 30],
        [30, 31],
        [31, 32],
        [32, 33],
        [33, 34],
    ]
    assert (
        "".join(tokenizer.convert_ids_to_tokens(encoded["input_ids"][1:10])) == sequence
    )
    assert encoded["attention_mask"] == [1] * 10 + [0, 0]
    tokenizer.eos_token = "<eos>"
    with pytest.raises(ValueError, match="span/special/padding"):
        encode_sequences(tokenizer, [sequence], 4105)
