"""d1npu: d1-3B decisions on the Ryzen AI NPU.

``D1`` is imported lazily so ``import d1npu`` stays light on machines
without the Ryzen AI environment (unit tests, servers that only need the
request parsing helpers).
"""
from __future__ import annotations

from typing import Any

__all__ = ["D1", "MODEL_ID"]


def __getattr__(name: str) -> Any:
    if name == "D1":
        from .engine import D1 as _D1

        return _D1
    if name == "MODEL_ID":
        from .engine import MODEL_ID as _MODEL_ID

        return _MODEL_ID
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
