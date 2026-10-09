"""Demo CLI for the XAI system (issues #21, #23).

Thin entry point: parses arguments, constructs an `XAISession` and prints.
All analysis and rendering lives in `xai_session.py`.

    python xai_demo.py --cached cases
    python xai_demo.py --cached explain P01
"""

import argparse
import sys

from xai_session import XAISession, XAISessionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SaCas9 XAI demo (text-only).")
    cached_help = "answer everything from the export cache without loading any model"
    parser.add_argument("--cached", action="store_true", help=cached_help)
    # Also accept `--cached` after the subcommand; SUPPRESS keeps the
    # subparser from resetting a flag given before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--cached", action="store_true", default=argparse.SUPPRESS, help=cached_help
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cases", parents=[common], help="list all Case Studies")
    explain = sub.add_parser("explain", parents=[common], help="explain one Case Study")
    explain.add_argument("query", help="Case Study ID, e.g. P01 or DISCORDANT_P01")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.cached:
        print(
            "live mode (model loading) is not implemented yet; rerun with --cached",
            file=sys.stderr,
        )
        return 2

    try:
        session = XAISession()
        if args.command == "cases":
            print(session.cases())
        elif args.command == "explain":
            print(session.explain(args.query))
    except (XAISessionError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # Korean interpretation lines must survive a cp949 Windows console.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
