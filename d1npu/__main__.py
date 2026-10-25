"""Entry point: flush stdio and end with os._exit (native providers crash otherwise)."""
import os
import sys

# The NPU compiler and the native providers write their logs to file
# descriptor 1. Keep the real stdout for the answers and point descriptor 1
# at stderr, so that stdout holds only the JSON.
sys.stdout.flush()
_answers = os.fdopen(
    os.dup(1),
    "w",
    encoding=sys.stdout.encoding or "utf-8",
    errors=sys.stdout.errors or "strict",
)
os.dup2(2, 1)
sys.stdout = _answers

from .cli import main  # noqa: E402

code = 0
try:
    code = main()
except SystemExit as exc:
    code = exc.code or 0
sys.stdout.flush()
sys.stderr.flush()
os._exit(int(code) if isinstance(code, int) else 0)
