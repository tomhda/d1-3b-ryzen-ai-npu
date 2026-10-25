"""Command line interface: download, decide, screen-check, serve."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ID = "Tomoali/d1-3B-ryzen-ai-npu"

SCREEN_CHECK = Path(__file__).resolve().parent / "screen_check.json"


def _read_state(value: str) -> str:
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def _load_questions_file(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("questions file must hold an object")
    wrapped = data.get("questions")
    if isinstance(wrapped, dict) and "type" not in wrapped:
        return wrapped
    return data


def _screen_check_questions() -> tuple[str, dict[str, Any], list[str]]:
    preset = json.loads(SCREEN_CHECK.read_text(encoding="utf-8"))
    questions: dict[str, Any] = {}
    order: list[str] = []
    for entry in preset["questions"]:
        key = entry["key"]
        question = dict(entry["question"])
        threshold = entry.get("threshold")
        invert = entry.get("invert", False)
        if question.get("type") == "noul":
            if threshold is not None:
                question["threshold"] = threshold
            question["invert"] = bool(invert)
        questions[key] = question
        order.append(key)
    return preset.get("state", ""), questions, order


def cmd_download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed; install the download extra:\n"
            "  pip install 'd1npu[download]'",
            file=sys.stderr,
        )
        return 2
    kwargs: dict[str, Any] = {"local_dir": args.to}
    if args.revision:
        kwargs["revision"] = args.revision
    snapshot_download(REPO_ID, **kwargs)
    print(f"downloaded {REPO_ID} to {args.to}", flush=True)
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    from .engine import D1

    state = _read_state(args.state)
    questions = _load_questions_file(args.questions)
    engine = D1(args.model)
    try:
        answers = engine.decide(state, questions, args.image)
    finally:
        engine.close()
    print(json.dumps(answers, ensure_ascii=False, indent=2), flush=True)
    return 0


def cmd_screen_check(args: argparse.Namespace) -> int:
    from .engine import D1

    default_state, questions, order = _screen_check_questions()
    state = args.state if args.state is not None else default_state
    engine = D1(args.model)
    try:
        answers = engine.decide(state, questions, args.image)
    finally:
        engine.close()
    compact: dict[str, dict[str, Any]] = {}
    for key in order:
        answer = answers[key]
        if answer["type"] == "noul":
            compact[key] = {"p": answer["noul"], "decision": answer["decision"]}
        elif answer["type"] == "choice":
            pick = answer["choice"]
            compact[key] = {
                "p": answer["probabilities"][pick],
                "decision": pick,
            }
        else:
            compact[key] = {"p": answer["score"], "decision": answer["score"]}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    return serve(args.model, args.host, args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="d1npu")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download the model folder")
    p.add_argument("--to", required=True, help="local directory to write to")
    p.add_argument("--revision", default=None)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("decide", help="run questions and print answers JSON")
    p.add_argument("--model", required=True)
    p.add_argument("--state", required=True, help="text or @FILE")
    p.add_argument("--questions", required=True, help="questions FILE.json")
    p.add_argument("--image", default=None)
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("screen-check", help="run the screen_check preset")
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--state", default=None)
    p.set_defaults(func=cmd_screen_check)

    p = sub.add_parser("serve", help="serve the Decisions API over HTTP")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
