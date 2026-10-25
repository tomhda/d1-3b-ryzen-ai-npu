"""Fixed vision-grid checks (stdlib only).

The NPU vision encoder takes exactly one 640x400 image, which the processor
turns into 24x40 patches with all 960 patches valid. Anything else is a loud
ValueError, never silent padding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

VISION_WIDTH = 640
VISION_HEIGHT = 400
VISION_SIZE = (VISION_WIDTH, VISION_HEIGHT)
VISION_SPATIAL = [[24, 40]]
VISION_VALID_PATCHES = 960


def _to_list(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _mask_sum(mask: Any) -> int:
    if isinstance(mask, bool):
        raise ValueError("invalid pixel_attention_mask")
    if isinstance(mask, int):
        return mask
    total = getattr(mask, "sum", None)
    if callable(total):
        return int(total())
    flat = _to_list(mask)
    if isinstance(flat, (list, tuple)):
        n = 0
        stack = [flat]
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                stack.extend(item)
            else:
                n += int(item)
        return n
    return int(flat)


def validate_vision_grid(spatial_shapes: Any, pixel_attention_mask: Any) -> None:
    """Raise ValueError unless the processor output is 24x40 with 960 valid."""
    spatial = _to_list(spatial_shapes)
    if spatial != VISION_SPATIAL:
        raise ValueError(
            f"NPU vision input must be 24x40 patches, got {spatial!r} "
            "(resize the image to 640x400 first)"
        )
    n_valid = _mask_sum(pixel_attention_mask)
    if n_valid != VISION_VALID_PATCHES:
        raise ValueError(
            "NPU vision input must have 960 valid patches, "
            f"got {n_valid}"
        )


def vision_cache_key(vision_onnx_path: str | Path) -> str:
    """Cache key from the vision ONNX size + mtime so a new model re-compiles."""
    path = Path(vision_onnx_path)
    stat = path.stat()
    return f"modelcachekey_{path.stem}_{stat.st_size}_{int(stat.st_mtime)}"
