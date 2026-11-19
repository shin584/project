"""Display score semantics and configuration reconstruction without weights."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from predict import MODEL_REGISTRY, _config_from_state, _to_display_score


@pytest.mark.parametrize("raw,expected", [(-1.0, 15.865525393), (0.0, 50.0), (1.0, 84.134474607)])
def test_zscores_without_training_statistics_are_percentiles(raw, expected):
    spec = replace(MODEL_REGISTRY[("A", "SpCas9")], target_mean=None, target_std=None)
    value, kind = _to_display_score(spec, raw)
    assert value == pytest.approx(expected)
    assert kind == "percentile"


@pytest.mark.parametrize("head,raw,expected", [(0, 1.0, 30.0), (1, -1.0, 50.0), (0, -10.0, 0.0), (1, 10.0, 100.0)])
def test_zscores_with_training_statistics_restore_efficiency_per_head(head, raw, expected):
    spec = replace(MODEL_REGISTRY[("A", "SpCas9")], target_mean=(0.2, 0.8), target_std=(0.1, 0.3))
    value, kind = _to_display_score(spec, raw, head)
    assert value == pytest.approx(expected)
    assert kind == "efficiency"


def test_percent_training_statistics_are_rejected():
    spec = replace(MODEL_REGISTRY[("A", "SpCas9")], target_mean=49.6, target_std=20.0)
    with pytest.raises(ValueError, match="fraction"):
        _to_display_score(spec, 0.0)


@pytest.mark.parametrize("raw,expected", [(-0.5, 0.0), (0.25, 25.0), (1.5, 100.0)])
def test_fraction_outputs_are_clipped_efficiencies(raw, expected):
    assert _to_display_score(MODEL_REGISTRY[("A", "Cas12a")], raw) == (expected, "efficiency")


def test_esm_configuration_is_reconstructed_from_shapes_without_tensors():
    state = {
        "esm.embeddings.word_embeddings.weight": SimpleNamespace(shape=(4105, 1280)),
        "esm.embeddings.position_embeddings.weight": SimpleNamespace(shape=(1002, 1280)),
        "esm.encoder.layer.0.intermediate.dense.weight": SimpleNamespace(shape=(5120, 1280)),
        "esm.encoder.layer.23.intermediate.dense.weight": SimpleNamespace(shape=(5120, 1280)),
    }
    cfg = _config_from_state(state, num_labels=9)
    assert (cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers) == (4105, 1280, 24)
    assert (cfg.intermediate_size, cfg.max_position_embeddings, cfg.num_labels) == (5120, 1002, 9)
    assert cfg.num_attention_heads == 20
    assert cfg.position_embedding_type == "absolute"
    assert not cfg.emb_layer_norm_before
    assert cfg.token_dropout
    assert (cfg.pad_token_id, cfg.mask_token_id) == (1, 2)
    assert cfg._attn_implementation == "eager"


# ── 모델 계열 ──────────────────────────────────────────────────


def test_each_family_loads_only_its_own_registry():
    """계열 하나만 만든다. 다른 계열의 예측기는 그 프로세스에 없다 (#191, #196)."""
    import predict

    nt = predict.load_predictors(warmup=False, family="nt")
    non_nt = predict.load_predictors(warmup=False, family="non_nt")
    assert set(nt) == set(non_nt) == {("A", "SpCas9"), ("A", "Cas12a")}
    assert all(p.spec.family == "nt" for p in nt.values())
    assert all(p.spec.family == "non_nt" for p in non_nt.values())
    assert type(nt[("A", "SpCas9")]).__name__ == "NTClassifierPredictor"
    assert type(non_nt[("A", "SpCas9")]).__name__ == "NonNTSpCas9Predictor"


def test_the_non_nt_family_warms_nothing_up():
    """해당 Cas의 사이트가 나왔을 때만 적재한다 — 시작 때 데우지 않는다 (#196)."""
    import predict

    assert predict.FAMILY_WARMUP["non_nt"] == ()
    assert predict.FAMILY_WARMUP["nt"] == predict.WARMUP_KEYS


def test_an_unknown_family_is_rejected():
    import predict

    with pytest.raises(ValueError, match="모르는 모델 계열"):
        predict.load_predictors(warmup=False, family="sacas9")


def test_default_family_keeps_the_existing_nt_behaviour():
    """기본 시작 계열을 바꾸는 것은 #198의 범위다."""
    import predict

    assert predict.DEFAULT_FAMILY == "nt"
    assert predict.load_predictors(warmup=False) is predict.load_predictors(
        warmup=False, family="nt")


def test_unconnected_models_claim_no_checkpoint():
    """미연동 슬롯은 checkpoint도 단위도 주장하지 않는다."""
    import predict

    spec = predict.ModelSpec("B", "SaCas9", (), family="non_nt", name="SaCas9")
    predictor = predict.UnconnectedPredictor(spec)
    assert not predictor.connected and predictor.predict_batch([]) == []
    identity = predictor.model_identity()
    assert identity.version == "미연동" and identity.checkpoint_sha256 == ""
    assert identity.raw_unit == ""


def test_version_label_names_the_asset_and_the_checkpoint_digest():
    import predict

    spec = predict.NON_NT_REGISTRY[("A", "SpCas9")]
    assert predict.version_label(spec, "ab" * 32) == "spcas9-9hydra-v2 · sha256 abababababab"
    assert predict.version_label(spec, "") == "spcas9-9hydra-v2 · sha256 미확인"
    nt = predict.MODEL_REGISTRY[("A", "SpCas9")]
    assert predict.version_label(nt, "cd" * 32) == "sha256 cdcdcdcdcdcd"


def test_name_resolved_predictors_are_declared_to_the_packaging_spec():
    """이름으로만 적힌 예측기는 PyInstaller 분석에 보이지 않는다 (#196/#197).

    `FAMILY_PREDICTOR_CLASSES`의 문자열 항목은 런타임에 `importlib`이 푼다. 정적
    import가 없으므로 `packaging/desktop.spec`의 `hiddenimports`에 적지 않으면
    frozen 빌드에서만, 그 계열을 고른 순간에만 터진다. 두 곳이 어긋나면 여기서 드러난다.
    """
    from pathlib import Path

    import predict

    spec = (Path(__file__).resolve().parent.parent / "packaging/desktop.spec").read_text(
        encoding="utf-8")
    named = {entry.split(":")[0]
             for table in predict.FAMILY_PREDICTOR_CLASSES.values()
             for entry in table.values() if isinstance(entry, str)}
    assert named, "문자열로 적힌 예측기가 없다 — 이 검사의 전제가 바뀌었다"
    for module in sorted(named):
        # 이름이 실제 모듈이어야 하고,
        assert predict._predictor_class(
            next(entry for table in predict.FAMILY_PREDICTOR_CLASSES.values()
                 for entry in table.values()
                 if isinstance(entry, str) and entry.startswith(module + ":")))
        # 배포 spec이 그 이름을 알고 있어야 한다.
        assert f"'{module}'" in spec, f"{module}이 desktop.spec의 hiddenimports에 없다"
