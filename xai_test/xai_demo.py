"""Demo CLI for the XAI system (issues #21, #23, #25, #27, #28).

Thin entry point: parses arguments, constructs the predictors (one at a
time) and an `XAISession`, then prints or loops over stdin. All analysis
and rendering lives in `xai_session.py`.

    python xai_demo.py demo             # load models, then interactive prompt
    python xai_demo.py --cached demo    # interactive prompt, no model load
    python xai_demo.py cases
    python xai_demo.py report
    python xai_demo.py --cached explain P01
    python xai_demo.py --ig-steps 20 explain <36bp sequence>   # live, fewer IG steps

`cases` and `report` never need a model. `--cached` never constructs a
predictor, so a memory crash or model-load failure on the presenting laptop
still leaves a working demo.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

from xai_session import DEFAULT_IG_STEPS, Predictors, XAISession, XAISessionError

_XAI_TEST_ROOT = Path(__file__).resolve().parent
MODEL_A_WEIGHT_PATH = _XAI_TEST_ROOT / "best_model_fold1.pth"
NT_MODEL_DIR = _XAI_TEST_ROOT / "NT_sacas9_fintuned_model"
XGB_MODEL_PATH = _XAI_TEST_ROOT / "hybrid_xgb_model.json"


# The wrappers import torch/transformers at module level, so each import is
# deferred into its loader: `--cached` never pays for (or risks) them.
def _load_model_a():
    from model_a_wrapper import Model_A_Predictor

    return Model_A_Predictor(weight_path=str(MODEL_A_WEIGHT_PATH))


def _load_model_b():
    from model_b_wrapper import Model_B_Predictor

    return Model_B_Predictor(
        nt_model_dir=str(NT_MODEL_DIR), xgb_model_path=str(XGB_MODEL_PATH)
    )


def _load_model_b_xai():
    from model_b_xai_wrapper import Model_B_XAIPredictor

    return Model_B_XAIPredictor(nt_model_dir=str(NT_MODEL_DIR))


# (display name, `Predictors` field, loader), in load order.
PREDICTOR_LOADERS = (
    ("Model A (CNN+RNN)", "model_a", _load_model_a),
    ("Model B (NT 500M + 4 Phys + XGBoost)", "model_b", _load_model_b),
    (
        "Model B XAI predictor (NT regression checkpoint)",
        "model_b_xai",
        _load_model_b_xai,
    ),
)

HELP_TEXT = """명령어:
  <Case Study ID>   예: P01, R01, C01 또는 DISCORDANT_P01 - Case Study 설명
  <36bp sequence>   ACGT 36bp (대소문자 무관, PAM NNGRRN at 25-30) - live 설명
                    (--cached에서는 Testset sequence만 cache로 설명)
  cases             Case Study 목록
  report            Testset integrity check와 global findings
  help              이 도움말
  quit              종료 (exit, Ctrl+D/Ctrl+Z도 가능)"""

QUIT_COMMANDS = ("quit", "exit", "q")


def load_predictors(loaders=PREDICTOR_LOADERS) -> Predictors:
    """Construct each predictor strictly after the previous one has finished
    (7.8GB laptop: never two loads in flight), printing progress."""
    loaded = {}
    total = len(loaders)
    for i, (name, field, load) in enumerate(loaders, start=1):
        print(f"[{i}/{total}] {name} 로딩 중...", flush=True)
        start = time.perf_counter()
        loaded[field] = load()
        gc.collect()  # release load-time temporaries before the next model
        print(
            f"[{i}/{total}] {name} 완료 ({time.perf_counter() - start:.1f}s)",
            flush=True,
        )
    print("모든 모델 로딩 완료 - ready", flush=True)
    return Predictors(**loaded)


def _run_command(session, line: str) -> None:
    command = line.lower()
    if command == "help":
        print(HELP_TEXT)
    elif command == "cases":
        print(session.cases())
    elif command == "report":
        print(session.report())
    else:
        print(session.explain(line))


def run_session(session) -> None:
    """The interactive prompt. One failing command prints an error and
    returns to the prompt; only `quit` or end of input ends the session."""
    print("'help'로 명령어 목록을 볼 수 있습니다.")
    while True:
        try:
            line = input("xai> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.lower() in QUIT_COMMANDS:
            return
        try:
            _run_command(session, line)
        except KeyboardInterrupt:  # Ctrl+C on a slow command, not on the demo
            print("\n중단됨 - 프롬프트로 돌아갑니다")
        except XAISessionError as e:
            print(f"error: {e}")
        except Exception as e:  # noqa: BLE001 - one bad command must never end the demo
            print(f"error: {type(e).__name__}: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SaCas9 XAI demo (text-only).")
    cached_help = "answer everything from the export cache without loading any model"
    steps_help = (
        "Integrated Gradients step count for live typed-sequence explanations "
        f"(default {DEFAULT_IG_STEPS}; lower it if a response is too slow)"
    )
    parser.add_argument("--cached", action="store_true", help=cached_help)
    parser.add_argument(
        "--ig-steps", type=int, default=DEFAULT_IG_STEPS, metavar="N", help=steps_help
    )
    # Also accept the options after the subcommand; SUPPRESS keeps the
    # subparser from resetting a value given before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--cached", action="store_true", default=argparse.SUPPRESS, help=cached_help
    )
    common.add_argument(
        "--ig-steps", type=int, default=argparse.SUPPRESS, metavar="N", help=steps_help
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "demo",
        parents=[common],
        help="load the models, then answer commands at an interactive prompt",
    )
    sub.add_parser("cases", parents=[common], help="list all Case Studies")
    sub.add_parser(
        "report",
        parents=[common],
        help="Testset integrity check and global findings",
    )
    explain = sub.add_parser(
        "explain", parents=[common], help="explain a Case Study or a 36bp sequence"
    )
    explain.add_argument(
        "query", help="Case Study ID (e.g. P01, DISCORDANT_P01) or a 36bp sequence"
    )
    return parser


# Subcommands that recompute scores live unless `--cached` is given.
LIVE_COMMANDS = ("demo", "explain")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    predictors = None
    if args.command in LIVE_COMMANDS and not args.cached:
        try:
            predictors = load_predictors()
        except Exception as e:  # noqa: BLE001 - e.g. MemoryError on the laptop
            print(
                f"error: 모델 로딩 실패 ({type(e).__name__}: {e})"
                " - --cached로 다시 실행하면 모델 없이 cache로 답합니다",
                file=sys.stderr,
            )
            return 1
    try:
        session = XAISession(predictors=predictors, ig_steps=args.ig_steps)
        if args.command == "demo":
            run_session(session)
        else:
            _run_command(session, getattr(args, "query", args.command))
    except (XAISessionError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # Korean interpretation lines must survive a cp949 Windows console.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
