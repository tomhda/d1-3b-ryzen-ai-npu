"""Stdlib HTTP server for the Decisions API (ThreadingHTTPServer + one lock).

Endpoints ``POST /api/alpha/decisions`` and ``POST /api/v1/systemone`` take
``{"model": ..., "state": str | list, "questions": {...}}`` and answer with
``{"model": "d1-3b-ryzen-ai-npu", "answers": ..., "usage": ..., "timing": ...}``.
"""
from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .engine import MODEL_ID, D1
from .questions import validate_questions
from .request_parsing import parse_body, parse_state

PATHS = ("/api/alpha/decisions", "/api/v1/systemone")


def build_response(
    decide_fn: Any, body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    """Run one request through ``decide_fn``; pure enough for unit tests.

    ``decide_fn`` is ``(state, questions, image) -> (answers, timing)``
    where ``image`` is PIL image bytes (or None) and ``timing`` carries at
    least ``input_tokens``.
    """
    try:
        raw_state, questions = parse_body(body)
        text_state, image_bytes = parse_state(raw_state)
        valid = validate_questions(questions)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    image = None
    if image_bytes is not None:
        from PIL import Image

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except OSError as exc:  # includes PIL.UnidentifiedImageError
            return 400, {"error": f"image could not be decoded: {exc}"}
    try:
        answers, timing = decide_fn(text_state, valid, image)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    usage = {
        "input_tokens": int(timing.get("input_tokens", 0)),
        "output_tokens": 0,
    }
    return 200, {
        "model": MODEL_ID,
        "answers": answers,
        "usage": usage,
        "timing": timing,
    }


class Handler(BaseHTTPRequestHandler):
    engine: D1
    lock: threading.Lock

    def log_message(self, *args: Any) -> None:  # keep stdout for answers
        pass

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        if self.path not in PATHS:
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "invalid Content-Length"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"null")
        except ValueError as exc:
            self._send(400, {"error": f"invalid JSON: {exc}"})
            return

        def run(state: Any, questions: Any, image: Any) -> Any:
            with self.lock:
                answers = self.engine.decide(state, questions, image)
                return answers, dict(self.engine.last_timing or {})

        status, payload = build_response(run, body)
        self._send(status, payload)


def serve(model_dir: str | Path, host: str = "127.0.0.1", port: int = 8787) -> int:
    """Serve forever (until Ctrl-C); returns the exit code."""
    engine = D1(model_dir)

    class BoundHandler(Handler):
        pass

    BoundHandler.engine = engine
    BoundHandler.lock = threading.Lock()
    server = ThreadingHTTPServer((host, port), BoundHandler)
    print(f"serving {MODEL_ID} on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()
    return 0
