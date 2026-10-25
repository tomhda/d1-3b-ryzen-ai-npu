"""Assemble the Hugging Face model folder layout from conversion outputs.

Sources: converted ``language/`` dir, ``embed_tokens`` ONNX from
``onnx-community/d1-3B-ONNX``, the converted vision ONNX, and the text files
from ``LiquidAI/d1-3B`` (``preprocessor_config.json`` falls back to the ONNX dir,
which is the only source that has it; ``NOTICE.md`` is copied when present).
Only copies files; never modifies weights.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

LANGUAGE_FILES = (
    "genai_config.json",
    "model.onnx",
    "model.onnx.data",
    "model.pb.bin",
    "rai_config.json",
)

SRC_TEXT_FILES = (
    "prompt.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "config.json",
    "LICENSE",
)

OPTIONAL_TEXT_FILES = ("NOTICE.md",)


def assemble(
    language_dir: pathlib.Path,
    onnx_dir: pathlib.Path,
    vision_onnx: pathlib.Path,
    src_dir: pathlib.Path,
    output: pathlib.Path,
) -> None:
    language_dir = language_dir.resolve()
    onnx_dir = onnx_dir.resolve()
    vision_onnx = vision_onnx.resolve()
    src_dir = src_dir.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    (output / "language").mkdir(parents=True, exist_ok=True)
    (output / "embed").mkdir(parents=True, exist_ok=True)
    (output / "vision").mkdir(parents=True, exist_ok=True)

    for name in LANGUAGE_FILES:
        src = language_dir / name
        if not src.exists():
            raise FileNotFoundError(f"language dir missing: {src}")
        shutil.copyfile(src, output / "language" / name)
    cache_src = language_dir / "cache"
    if not (cache_src / "txn_bins.zip").exists():
        raise FileNotFoundError(f"language cache missing: {cache_src}")
    shutil.copytree(cache_src, output / "language" / "cache", dirs_exist_ok=True)

    for name in ("embed_tokens_q4.onnx", "embed_tokens_q4.onnx_data"):
        src = onnx_dir / "onnx" / name
        if not src.exists():
            raise FileNotFoundError(f"ONNX dir missing: {src}")
        shutil.copyfile(src, output / "embed" / name)

    # The graph names its weight file after itself, so the file is copied
    # under the same name rather than renamed.
    if vision_onnx.name != "vision_encoder_npu.onnx":
        raise ValueError(
            "build the vision encoder as vision_encoder_npu.onnx "
            f"(vision.py --output .../vision_encoder_npu.onnx), got {vision_onnx.name}"
        )
    vision_data = vision_onnx.parent / (vision_onnx.name + "_data")
    if not vision_onnx.exists():
        raise FileNotFoundError(f"vision onnx missing: {vision_onnx}")
    if not vision_data.exists():
        raise FileNotFoundError(f"vision data missing: {vision_data}")
    shutil.copyfile(vision_onnx, output / "vision" / "vision_encoder_npu.onnx")
    shutil.copyfile(
        vision_data, output / "vision" / "vision_encoder_npu.onnx_data")

    for name in SRC_TEXT_FILES:
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"src dir missing: {src}")
        shutil.copyfile(src, output / name)
    pre = src_dir / "preprocessor_config.json"
    if not pre.exists():
        pre = onnx_dir / "preprocessor_config.json"
    if not pre.exists():
        raise FileNotFoundError("preprocessor_config.json missing")
    shutil.copyfile(pre, output / "preprocessor_config.json")
    for name in OPTIONAL_TEXT_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copyfile(src, output / name)
    print(output, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-dir", required=True, help="language/ folder written by language.py")
    parser.add_argument("--onnx-dir", required=True, help="local copy of onnx-community/d1-3B-ONNX")
    parser.add_argument("--vision-onnx", required=True, help="vision_encoder_npu.onnx written by vision.py")
    parser.add_argument("--src-dir", required=True, help="local copy of LiquidAI/d1-3B")
    parser.add_argument("--output", required=True, help="model folder to create")
    args = parser.parse_args(argv)
    assemble(
        pathlib.Path(args.language_dir),
        pathlib.Path(args.onnx_dir),
        pathlib.Path(args.vision_onnx),
        pathlib.Path(args.src_dir),
        pathlib.Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
