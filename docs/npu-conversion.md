# Putting d1-3B on the Ryzen AI NPU: what broke and how it was fixed

Ryzen AI Software 1.8.0, ONNX Runtime from its conda environment, Ryzen AI 7 PRO 350 (XDNA 2), NPU driver 32.0.203.329, Windows 11 build 26200. The full Japanese notes, with all measurements, are in [ja/research.md](ja/research.md).

## Language model

The 4-bit decoder from `onnx-community/d1-3B-ONNX` (`decoder_model_merged_q4`, `MatMulNBits`, block 32) is converted with `model_generate --npu --basic`. All 167 `MatMulNBits` nodes become `com.ryzenai` NPU operators (`MatMulNBitsBf`). Conv, GQA and the embedding stay on the CPU.

- **8-bit weights are not converted.** An 8-bit `MatMulNBits` decoder passes through `--basic` unchanged: the tool prints `No bins were filtered -- nothing to output` and still exits with code 0. The NPU kernel name (`a16fw4acc16f`) also indicates fixed 4-bit weights.
- **`model_generate` returns 0 on errors.** Call `model_generate.cli.main()` and check its return value, then check that the output graph contains `com.ryzenai` operators.
- **`genai-config` value.** The command-line entry upper-cases the value while the code that reads it expects lower case (`invalid GenaiMode value: 'STANDALONE'`). Passing it as `--extra_options genai-config=standalone` works.
- **Context length.** The `--basic` output sets `hybrid_opt_max_seq_length` to 512. Setting it to 4096 in `genai_config.json` lets the same model read prompts of about 2,300 tokens. The value must be a power of two; 3072 fails at session creation.
- **Token fusion** (`--token_fusion`) was not used: d1 reads the state once and branches each question from a copy of the KV cache, and the fused path broke that.

## Vision encoder (SigLIP2 + projector)

Running the fp32 vision encoder through `VitisAIExecutionProvider` (VAIML, bf16) needed three changes.

| Stage | Symptom | Cause | Fix |
|---|---|---|---|
| Session creation | Access violation (0xC0000005) after 1–2.5 s | An `Expand` in the position-embedding resize whose shape is computed at run time from the `spatial_shapes` input (found by bisecting sub-models) | Fix the input to 640x400 (24x40 = 960 patches), fold the shape logic into constants, and decompose the 27 `com.microsoft::MultiHeadAttention` nodes into standard operators |
| Compilation (default `optimize_level` 2) | `aiecompiler 77-5732: Multiple tiling parameters per interleaving point` | The fc1 → fc2 intermediate buffer of each MLP (960x4304, bf16, about 8 MB). Reproduced with an MLP-only model in 35 s; padding the width to 4352 gives the same error | Split every MLP into two halves along the hidden width and add the partial fc2 outputs before the fc2 bias. The whole graph then compiles as one NPU subgraph (988 nodes) in 614 s |
| `optimize_level` 1 | Compiles, but the output is wrong (cosine 0.02–0.03 to fp32) | `MatMul` returns wrong values under `optimize_level` 1: a single-MatMul model gives cosine 0.70–0.73 regardless of weights or row count, against 0.99996–0.99998 with the default level | Keep the default level |

With the split model the NPU output has cosine 0.9957 / 0.9969 to the fp32 reference on two screenshots, and one image takes 0.41 s. The compiled result is cached by the provider (`cache_dir`, `cache_key`).

The rebuilt GELUs of the split MLPs do not carry over the original `approximate="tanh"` attribute, so they compute the exact (erf) GELU. On the CPU the result has cosine 0.99998 to the original vision encoder. All measurements in the notes were made with this model.

The 4-bit vision encoder from the ONNX export was not used: its image features have cosine 0.946–0.961 to fp32 and changed some answers.

## Other findings

- **DirectML (ONNX Runtime 1.27) skips RoPE inside GQA.** `do_rotary=1` is not applied. Moving RoPE out of GQA into explicit operators makes the DirectML result match the CPU.
- **Language and vision providers coexist in one process.** `RyzenAILightExecutionProvider` (decoder) and `VitisAIExecutionProvider` (vision) were used from the same Python process without conflicts.
- **Shutdown.** The native providers can crash while the interpreter shuts down; the command-line tools end with `os._exit()` after flushing their output.
