"""Command line shared by ``python -m desktop`` and the copied executable."""

import argparse
from pathlib import Path


def main() -> int:
    import multiprocessing

    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="모델 가중치 없이 화면 미리보기")
    mode.add_argument("--validate", type=Path, metavar="OUTPUT", help="실제 창으로 게이트 2 산출물 저장 후 종료")
    args = parser.parse_args()
    if args.validate is not None:
        from desktop.validation import validate

        return validate(args.validate)
    from desktop.app import main as run

    return run(preview=args.preview)
