"""Online build-time step: export the tokenizer and stage a fixed WebView2."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--webview2",
        required=True,
        type=Path,
        help="Extracted Microsoft Fixed Version x64 runtime directory",
    )
    args = parser.parse_args()
    if not (args.webview2 / "msedgewebview2.exe").is_file():
        parser.error("--webview2 must directly contain msedgewebview2.exe")
    assets = ROOT / ".packaging" / "assets"
    if assets.exists():
        parser.error(f"Use a fresh staging directory; already exists: {assets}")
    from predict import NT_TOKENIZER_REPO
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NT_TOKENIZER_REPO)
    tokenizer.save_pretrained(assets / "tokenizer")
    # Reload from the export with network access disabled: never ship HF cache links.
    AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True)
    shutil.copytree(args.webview2, assets / "webview2")
    print(f"Staged offline resources at {assets}")


if __name__ == "__main__":
    main()
