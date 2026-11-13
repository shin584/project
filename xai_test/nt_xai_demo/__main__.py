"""NT XAI 데모 진입점.

인자 없이 실행하면 REPL, `<뉴클레아제> <뷰>`는 일회성 실행,
`prepare`/`verify`는 자료 준비와 온전성 확인이다.
"""

from __future__ import annotations

import argparse
import sys

from xai_test.nt_xai_demo import prepare, shell


def _speak_utf8() -> None:
    """Make the console take this demo's Korean output.

    Every view is Korean and uses an em dash, which the Windows cp949 console
    cannot encode: without this, a view raises `UnicodeEncodeError` instead of
    printing - in the middle of a live meeting. Reconfiguring the streams is
    cheap and belongs to the entry point; a stream that cannot be reconfigured
    is left as it is rather than failing the run.

    The error handler is left strict on purpose. Replacing unencodable text
    would turn a failure into mojibake on screen mid-meeting, and a figure that
    renders as garbage is worse than one that stops and says so.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    """`prepare`/`verify`, the REPL, or one view as a one-shot subcommand.

    The REPL is this demo's main entry point, so no arguments starts it. A
    one-shot call (`cas12a report`) runs the same view through the same
    `Shell.execute` the REPL uses, and is parsed here instead of being declared
    as argparse subcommands so the view list stays in one place
    (`shell.VIEWS`).
    """
    _speak_utf8()
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        return shell.repl()
    if tokens[0] not in ("prepare", "verify", "-h", "--help"):
        # `help` included: the shell's own help names the nucleases and views,
        # which argparse's does not.
        outcome = shell.run_once(tokens)
        if outcome.text:
            print(outcome.text, file=sys.stdout if outcome.ok else sys.stderr)
        return 0 if outcome.ok else 1

    parser = argparse.ArgumentParser(
        prog="nt_xai_demo",
        description="NT XAI 데모 (인자 없이 실행하면 REPL이 뜬다)",
        epilog="뷰: "
        + ", ".join(f"[cas12a|spcas9] {view.usage}" for view in shell.VIEWS),
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("prepare", help="데모 자료를 data/에 모으고 manifest를 만든다")
    commands.add_parser("verify", help="data/를 manifest와 대조한다")
    args = parser.parse_args(tokens)

    if args.command == "prepare":
        try:
            report = prepare.prepare()
        except prepare.PrepareRefused as error:
            print(error, file=sys.stderr)
            return 1
        print(report.render())
        return 0
    if args.command == "verify":
        data_dir = prepare.DEFAULT_DATA_DIR
        problems = prepare.verify(data_dir)
        if problems:
            print(
                "온전성 확인 실패:\n" + "\n".join(f"  - {p}" for p in problems),
                file=sys.stderr,
            )
            return 1
        print(prepare.render_verified(data_dir))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
