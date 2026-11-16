"""Run real predictions, or the weight-free ``--preview`` desktop shell."""

import argparse

from desktop.app import main

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--preview", action="store_true", help="모델 가중치 없이 화면 미리보기"
)
raise SystemExit(main(preview=parser.parse_args().preview))
