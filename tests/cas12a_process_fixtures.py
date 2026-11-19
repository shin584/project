"""Real #197 renderer with guards at file/import system boundaries."""

import os
from pathlib import Path
import sys

from desktop.worker_renderer import NonNTComputationRenderer


class Cas12aOnlyRenderer(NonNTComputationRenderer):
    def __init__(self):
        super().__init__()
        log = Path(os.environ["ISSUE197_MODEL_ACCESS_LOG"])

        def audit(event, args):
            if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            path = os.fsdecode(args[0]).replace("\\", "/").lower()
            if (path.endswith(".safetensors")
                    or Path(path).name in {"tokenizer.json", "tokenizer_config.json", "vocab.txt"}
                    or "nucleotide-transformer" in path
                    or ("spcas9-9hydra-v2" in path and path.endswith(".keras"))):
                raise AssertionError(f"Unexpected NT or SpCas9 resource access: {path}")
            if path.endswith("cas12a_cnn_manifest.keras"):
                with log.open("a", encoding="utf-8") as out:
                    out.write(path + "\n")

        sys.addaudithook(audit)

    def warm_up(self, on_batch):
        result = super().warm_up(on_batch)
        assert not {"torch", "keras", "transformers", "safetensors"}.intersection(sys.modules)
        return result
