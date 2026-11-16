"""Resource and offline policy shared by the packaging spike and predictors."""

from model_source import configure_offline, is_packaged, resource_root  # noqa: F401


def validate_bundle() -> None:
    """Reject incomplete copied folders before loading either large model."""
    required = (
        "models/Cas12a.safetensors",
        "models/SpCas9.safetensors",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.txt",
        "webview2/msedgewebview2.exe",
    )
    missing = [name for name in required if not (resource_root() / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete offline bundle: " + ", ".join(missing))
