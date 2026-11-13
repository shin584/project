"""NT XAI 데모 진입점.

`<뉴클레아제>`는 그 뉴클레아제의 REPL, `<뉴클레아제> <번호|report>`는 같은
출력의 일회성 실행, `prepare`/`verify`는 자료 준비와 온전성 확인이다.
"""

from __future__ import annotations

import argparse
import sys

from xai_test.nt_xai_demo import launcher, prepare


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
    """`prepare`/`verify`, or one nuclease's demo (`cas12a`, `cas12a A1`).

    The nuclease is the first argument and fixes the session (#171): `cas12a`
    opens its REPL, `cas12a A1` prints the same screen once. The branching
    menu this replaced was deleted in #175, so this is the only entry point.
    """
    _speak_utf8()
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        print(launcher.usage(), file=sys.stderr)
        return 2
    if tokens[0] in ("-h", "--help", "help"):
        print(launcher.usage())
        return 0
    if tokens[0] not in ("prepare", "verify"):
        return launcher.run(
            tokens[0].lower(),
            tokens[1:],
            write=print,
            write_error=lambda text: print(text, file=sys.stderr),
        )

    parser = argparse.ArgumentParser(prog="nt_xai_demo", description=launcher.usage())
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


if __name__ == "__main__":
    sys.exit(main())
