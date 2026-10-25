"""Convert the 4-bit d1 decoder for the Ryzen AI NPU.

1. Prepare a ``model_generate`` input folder from ``onnx-community/d1-3B-ONNX``
   (the decoder graph as ``model.onnx`` with its weight files, the text
   ``config.json`` and the tokenizer files).
2. Run ``model_generate --npu --basic`` with ``max_seq_len=512``,
   ``genai-config=standalone``, ``no_prune_logits``, ``model_type=lfm2-2.6b``,
   calling ``model_generate.cli.main()`` and
   checking its return value, then verifying the output graph contains
   ``com.ryzenai`` ops.
3. Copy the result into the ``language/`` layout and set
   ``hybrid_opt_max_seq_length`` to ``"4096"`` (must be a power of two;
   3072 fails at session creation).

All paths are arguments; nothing is hard-coded. Runs on Windows in the
Ryzen AI 1.8.0 conda env (needs ``model_generate``, ``onnx``).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

LANGUAGE_FILES = (
    "genai_config.json",
    "model.onnx",
    "model.onnx.data",
    "model.pb.bin",
    "rai_config.json",
)


def prepare_input(onnx_dir: pathlib.Path, work_dir: pathlib.Path) -> None:
    """Build the standalone decoder folder without re-exporting weights."""
    import onnx

    onnx_dir = onnx_dir.resolve()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((onnx_dir / "config.json").read_text(encoding="utf-8"))[
        "text_config"
    ]
    cfg["rope_theta"] = cfg["rope_parameters"]["rope_theta"]
    cfg["block_ff_dim"] = cfg["intermediate_size"]
    (work_dir / "config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        shutil.copyfile(onnx_dir / name, work_dir / name)
    graph = onnx.load(
        onnx_dir / "onnx" / "decoder_model_merged_q4.onnx",
        load_external_data=False,
    )
    graph.graph.name = "d1_lfm2_2_6b_decoder"
    onnx.save(graph, work_dir / "model.onnx")
    for name in [
        "decoder_model_merged_q4.onnx_data",
        "decoder_model_merged_q4.onnx_data_1",
    ]:
        target = work_dir / name
        if (
            not target.exists()
            or target.stat().st_size != (onnx_dir / "onnx" / name).stat().st_size
        ):
            shutil.copyfile(onnx_dir / "onnx" / name, target)
    genai = {
        "model": {
            "type": "lfm2",
            "context_length": 4096,
            "vocab_size": 128000,
            "bos_token_id": 124894,
            "eos_token_id": 124900,
            "pad_token_id": 124893,
            "decoder": {
                "filename": "model.onnx",
                "hidden_size": 2048,
                "head_size": 64,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 30,
                "session_options": {"provider_options": []},
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
            },
        },
        "search": {"max_length": 4096, "past_present_share_buffer": False},
    }
    (work_dir / "genai_config.json").write_text(
        json.dumps(genai, indent=2), encoding="utf-8"
    )
    print(work_dir, flush=True)


def run_conversion(
    work_dir: pathlib.Path, output: pathlib.Path, timeout: int
) -> None:
    """Run model_generate and verify com.ryzenai ops in the output graph."""
    import onnx

    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            "Use a new output directory for each conversion attempt: "
            + str(output)
        )
    cmd = [
        sys.executable,
        "-B",
        "-c",
        "from model_generate.cli import main; import sys; sys.exit(main())",
        "--npu",
        "--basic",
        "--input",
        str(work_dir),
        "--output",
        str(output),
        "-v",
        "--extra_options",
        "model_type=lfm2-2.6b",
        "max_seq_len=512",
        "genai-config=standalone",
    ]
    cmd.append("no_prune_logits")
    print("COMMAND", json.dumps(cmd), flush=True)
    started = time.perf_counter()
    # model_generate starts the onnx_utils console script by name; put the
    # environment's script folders on PATH so it also works without
    # ``conda activate``.
    env = os.environ.copy()
    prefix = pathlib.Path(sys.executable).parent
    folders = [prefix / "Scripts", prefix / "Library" / "bin", prefix]
    env["PATH"] = os.pathsep.join([str(p) for p in folders] + [env.get("PATH", "")])
    proc = subprocess.Popen(cmd, env=env)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["taskkill.exe", "/PID", str(proc.pid), "/T", "/F"], check=False
        )
        proc.wait(timeout=20)
        raise TimeoutError(f"conversion timed out after {timeout}s")
    if rc != 0:
        raise RuntimeError(f"model_generate failed with return code {rc}")
    model = onnx.load(output / "model.onnx", load_external_data=False)

    def nodes(graph):
        for node in graph.node:
            yield node
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    yield from nodes(attr.g)

    custom = [
        (n.domain, n.op_type)
        for n in nodes(model.graph)
        if n.domain not in ("", "com.microsoft")
    ]
    ryzenai = [op for domain, op in custom if domain == "com.ryzenai"]
    print(
        "CONVERTED",
        json.dumps(
            {
                "elapsed_s": time.perf_counter() - started,
                "custom_node_count": len(custom),
                "custom_operators": sorted(set(custom)),
            }
        ),
        flush=True,
    )
    if not ryzenai:
        raise RuntimeError("converted graph has no com.ryzenai ops")


def assemble_language(
    converted: pathlib.Path, language_dir: pathlib.Path
) -> None:
    """Copy the conversion result into the language/ layout (4096 ctx)."""
    converted = converted.resolve()
    language_dir = language_dir.resolve()
    language_dir.mkdir(parents=True, exist_ok=True)
    for name in LANGUAGE_FILES:
        src = converted / name
        if name in ("genai_config.json", "model.onnx") and not src.exists():
            raise FileNotFoundError(f"conversion output missing: {src}")
        if src.exists():
            shutil.copyfile(src, language_dir / name)
    cache_src = converted / "cache"
    if cache_src.is_dir():
        shutil.copytree(cache_src, language_dir / "cache", dirs_exist_ok=True)
    config_path = language_dir / "genai_config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["model"]["decoder"]["session_options"][
        "hybrid_opt_max_seq_length"
    ] = "4096"
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(language_dir, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", required=True, help="local copy of onnx-community/d1-3B-ONNX")
    parser.add_argument("--work-dir", required=True, help="model_generate input folder to create")
    parser.add_argument("--output", required=True, help="model_generate output folder")
    parser.add_argument("--language-dir", required=True, help="language/ folder to write")
    parser.add_argument("--timeout", type=int, default=2400, help="seconds allowed for model_generate")
    args = parser.parse_args(argv)
    prepare_input(pathlib.Path(args.onnx_dir), pathlib.Path(args.work_dir))
    run_conversion(
        pathlib.Path(args.work_dir), pathlib.Path(args.output), args.timeout
    )
    assemble_language(pathlib.Path(args.output), pathlib.Path(args.language_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
