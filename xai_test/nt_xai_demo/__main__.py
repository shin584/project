"""NT XAI 데모 진입점. `python -m xai_test.nt_xai_demo <subcommand>`."""

from __future__ import annotations

import argparse
import sys

from xai_test.nt_xai_demo import prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nt_xai_demo", description="NT XAI 데모")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("prepare", help="데모 자료를 data/에 모으고 manifest를 만든다")
    commands.add_parser("verify", help="data/를 manifest와 대조한다")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        try:
            report = prepare.prepare()
        except prepare.PrepareRefused as error:
            print(error, file=sys.stderr)
            return 1
        print(report.render())
        return 0
    if args.command == "verify":
        problems = prepare.verify()
        if problems:
            print(
                "온전성 확인 실패:\n" + "\n".join(f"  - {p}" for p in problems),
                file=sys.stderr,
            )
            return 1
        print(f"온전성 확인 통과: {prepare.DEFAULT_DATA_DIR / prepare.MANIFEST}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
