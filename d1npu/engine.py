"""NPU runtime: tokenizer, CPU embeddings, NPU decoder, lazy NPU vision.

The state is read once into a KV cache (trunk) and each question branches
from a private copy of it. The decoder runs on RyzenAILightExecutionProvider
with IO binding and a fixed KV capacity; the vision encoder runs on
VitisAIExecutionProvider with a writable build cwd. Images are stretched to
640x400 (bicubic) and must give 24x40 = 960 patches.

Third-party imports (onnxruntime, transformers, PIL, numpy, ml_dtypes) stay
inside methods so this module imports on plain Linux; only ``D1`` itself
needs the Ryzen AI environment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .questions import check_seq_len, format_answer, validate_questions, validate_state
from .vision_check import VISION_SIZE, validate_vision_grid, vision_cache_key

MODEL_ID = "d1-3b-ryzen-ai-npu"

DEFAULT_RYZEN_AI_DIR = "C:/Program Files/RyzenAI/1.8.0/deployment"

IMAGE_TOKEN_ID_FALLBACK = 124907
FEATURE_DIM = 2048


def digest_cache(past: dict[str, Any]) -> str:
    """Stable sha256 over a KV-cache dict (numpy arrays or plain bytes)."""
    h = hashlib.sha256()
    for key in sorted(past):
        value = past[key]
        if isinstance(value, (bytes, bytearray)):
            h.update(bytes(value))
        elif isinstance(value, memoryview):
            h.update(value.tobytes())
        elif hasattr(value, "tobytes"):
            try:
                h.update(value.tobytes())
            except Exception:  # noqa: BLE001 - fall back to repr
                h.update(repr(value).encode("utf-8"))
        else:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def default_vision_cache_dir() -> str:
    """%LOCALAPPDATA%\\d1npu\\vision-cache, or ~/.cache on other systems."""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return str(Path(local) / "d1npu" / "vision-cache")
    return str(Path.home() / ".cache" / "d1npu" / "vision-cache")


class D1:
    """d1-3B decisions on the Ryzen AI NPU."""

    def __init__(
        self,
        model_dir: str | Path,
        ryzen_ai_dir: str | Path = DEFAULT_RYZEN_AI_DIR,
        vision_cache_dir: str | Path | None = None,
        threads: int = 4,
    ) -> None:
        # Absolute, because the vision session is created from a build cwd.
        self.model_dir = Path(model_dir).resolve()
        self.language_dir = self.model_dir / "language"
        self.embed_path = self.model_dir / "embed" / "embed_tokens_q4.onnx"
        self.vision_path = (
            self.model_dir / "vision" / "vision_encoder_npu.onnx"
        )
        for needed in (
            self.model_dir / "prompt.py",
            self.model_dir / "tokenizer.json",
            self.model_dir / "chat_template.jinja",
            self.language_dir / "genai_config.json",
            self.embed_path,
            self.vision_path,
        ):
            if not needed.exists():
                raise FileNotFoundError(f"model file missing: {needed}")
        self.threads = int(threads)
        self.prompt = self._load_prompt(self.model_dir / "prompt.py")
        self.tok = self._load_tokenizer(self.model_dir / "tokenizer.json")
        try:
            self.image_token_id = int(self.tok.convert_tokens_to_ids("<image>"))
        except Exception:  # noqa: BLE001 - keep the known id
            self.image_token_id = IMAGE_TOKEN_ID_FALLBACK
        self.vision_cache_dir = (
            str(vision_cache_dir)
            if vision_cache_dir is not None
            else default_vision_cache_dir()
        )
        self.vision_cache_key = vision_cache_key(self.vision_path)
        self.vis = None
        self.processor = None
        self.last_timing: dict[str, Any] | None = None
        self.emb = self._load_embed_session()
        self.dec, self.outputs, self.max_seq_len = self._load_decoder_session(
            Path(ryzen_ai_dir)
        )

    # ------------------------------------------------------------ setup

    @staticmethod
    def _load_prompt(path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(
            "d1npu_model_prompt", str(path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load prompt module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _load_tokenizer(tokenizer_json: Path) -> Any:
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json),
            bos_token="<|startoftext|>",
            eos_token="<|im_end|>",
            pad_token="<|pad|>",
            extra_special_tokens={
                "image_token": "<image>",
                "image_start_token": "<|image_start|>",
                "image_end_token": "<|image_end|>",
                "image_thumbnail": "<|img_thumbnail|>",
            },
        )

    def _load_embed_session(self) -> Any:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return ort.InferenceSession(
            str(self.embed_path), so, providers=["CPUExecutionProvider"]
        )

    def _load_decoder_session(
        self, deploy: Path
    ) -> tuple[Any, list[str], int]:
        import onnxruntime as ort

        dll = deploy / "onnxruntime_providers_ryzenai.dll"
        if not dll.exists():
            raise RuntimeError(
                "RyzenAILightExecutionProvider is unavailable: "
                f"missing {dll}"
            )
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            self._dll_handles = [add_dll_directory(str(deploy))]
        os.environ["PATH"] = str(deploy) + os.pathsep + os.environ["PATH"]
        ort.register_execution_provider_library(
            "RyzenAILightExecutionProvider", str(dll)
        )
        devices = [
            d
            for d in ort.get_ep_devices()
            if d.ep_name == "RyzenAILightExecutionProvider"
            # The provider can also list the integrated GPU.
            and d.device.type == ort.OrtHardwareDeviceType.NPU
        ]
        if not devices:
            raise RuntimeError(
                "No RyzenAILightExecutionProvider devices; "
                "do not silently fall back to CPU"
            )
        cfg = json.loads(
            (self.language_dir / "genai_config.json").read_text(encoding="utf-8")
        )
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.add_provider_for_devices(devices, {})
        options = cfg["model"]["decoder"].get("session_options", {})
        max_seq_len = int(
            options.get(
                "hybrid_opt_max_seq_length", cfg["model"]["context_length"]
            )
        )
        for key, value in options.items():
            if key in ["provider_options", "custom_ops_library"] or not isinstance(
                value, (str, int, float, bool)
            ):
                continue
            if key == "external_data_file":
                if value == "null":
                    continue
                value = str(self.language_dir / str(value))
            so.add_session_config_entry(key, str(value))
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("dd_cache", str(self.language_dir / "cache"))
        so.add_session_config_entry("hybrid_opt_token_backend", "npu")
        so.register_custom_ops_library(str(dll))
        path = self.language_dir / cfg["model"]["decoder"].get(
            "filename", "model.onnx"
        )
        dec = ort.InferenceSession(str(path), so)
        outputs = [o.name for o in dec.get_outputs()]
        return dec, outputs, max_seq_len

    def _ensure_vision(self) -> None:
        if self.vis is not None:
            return
        print(
            "Compiling the vision encoder for the NPU "
            "(first run takes about 10 minutes)...",
            file=sys.stderr,
            flush=True,
        )
        import onnxruntime as ort

        os.environ.setdefault("XLNX_ENABLE_CACHE", "1")
        if "VitisAIExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "VitisAIExecutionProvider is unavailable; "
                "do not silently fall back to CPU"
            )
        cache_dir = Path(self.vision_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        options = [
            {
                "cache_dir": str(cache_dir),
                "cache_key": self.vision_cache_key,
                "enable_cache_file_io_in_mem": 0,
            }
        ]
        build_cwd = cache_dir / "build-cwd"
        build_cwd.mkdir(parents=True, exist_ok=True)
        previous = os.getcwd()
        try:
            os.chdir(build_cwd)
            self.vis = ort.InferenceSession(
                str(self.vision_path),
                providers=["VitisAIExecutionProvider"],
                provider_options=options,
            )
        finally:
            os.chdir(previous)

    def _ensure_processor(self) -> None:
        if self.processor is not None:
            return
        from transformers import Lfm2VlImageProcessorFast, Lfm2VlProcessor

        image_processor = Lfm2VlImageProcessorFast.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        chat_template = (self.model_dir / "chat_template.jinja").read_text(
            encoding="utf-8"
        )
        self.processor = Lfm2VlProcessor(
            image_processor=image_processor,
            tokenizer=self.tok,
            chat_template=chat_template,
        )

    # ------------------------------------------------------------ public API

    def decide(
        self,
        state: str | list[str],
        questions: dict[str, dict[str, Any]],
        image: Any = None,
    ) -> dict[str, dict[str, Any]]:
        """Run the questions over the state (and at most one image)."""
        state = validate_state(state)
        valid = validate_questions(questions)
        qs = {k: self.prompt.as_question(v) for k, v in valid.items()}
        trunk, branches, features, vision_s = self._prepare(state, qs, image)
        longest = len(trunk) + max(len(b) for b in branches.values())
        check_seq_len(0, longest, self.max_seq_len)
        probs, timing = self._evaluate_tree(trunk, branches, qs, features, vision_s)
        answers: dict[str, dict[str, Any]] = {}
        for key, dist in probs.items():
            qtype = valid[key]["type"]
            if qtype == "noul":
                labels = ["yes", "no"]
            elif qtype == "choice":
                labels = list(qs[key].criteria)
            else:
                labels = [str(i) for i in range(len(qs[key].criteria))]
            answers[key] = format_answer(
                qtype,
                labels,
                list(dist),
                threshold=valid[key].get("threshold", 0.5),
                invert=valid[key].get("invert", False),
                legend=valid[key]["criteria"] if qtype == "score" else None,
            )
        self.last_timing = timing
        return answers

    def close(self) -> None:
        """Release session references.

        The native Ryzen AI providers can crash during interpreter shutdown
        (their global objects are torn down after Python starts finalizing
        modules), so processes using this class must flush stdio and
        terminate with ``os._exit(code)`` instead of a normal return once
        the result is written. The CLI (``python -m d1npu``) does this for
        every command; do the same in any long-lived host process.
        """
        self.vis = None
        self.dec = None
        self.emb = None
        self.processor = None

    # ------------------------------------------------------------ internals

    def _prepare(
        self, state: str | list[str], qs: dict[str, Any], image: Any
    ) -> tuple[list[int], dict[str, list[int]], Any, float]:
        import numpy as np

        prefix = self.prompt.prefix_text(
            self.tok, state, self.tok.bos_token, images="<image>" if image else ""
        )
        features = None
        vision_s = 0.0
        if image is not None:
            started = time.perf_counter()
            pil_image = self._coerce_image(image)
            self._ensure_processor()
            x = self.processor(
                text=[prefix],
                images=[[pil_image]],
                return_tensors="pt",
                add_special_tokens=False,
            )
            x = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in x.items()}
            validate_vision_grid(
                x["spatial_shapes"], x["pixel_attention_mask"]
            )
            n = int(x["pixel_attention_mask"].sum(1).max())
            feeds = {
                "pixel_values": x["pixel_values"][:, :n].astype(np.float32),
                "pixel_attention_mask": x["pixel_attention_mask"][:, :n].astype(
                    np.int64
                ),
                "spatial_shapes": x["spatial_shapes"].astype(np.int64),
            }
            self._ensure_vision()
            for desc in self.vis.get_inputs():
                if (
                    desc.name == "pixel_values"
                    and len(desc.shape) == 3
                    and isinstance(desc.shape[1], int)
                ):
                    want = int(desc.shape[1])
                    have = feeds["pixel_values"].shape[1]
                    if have < want:
                        pad = want - have
                        feeds["pixel_values"] = np.pad(
                            feeds["pixel_values"], ((0, 0), (0, pad), (0, 0))
                        )
                        feeds["pixel_attention_mask"] = np.pad(
                            feeds["pixel_attention_mask"], ((0, 0), (0, pad))
                        )
            present = {i.name for i in self.vis.get_inputs()}
            feeds = {
                k: np.ascontiguousarray(v)
                for k, v in feeds.items()
                if k in present
            }
            features = self.vis.run(None, feeds)[0]
            vision_s = time.perf_counter() - started
            trunk = x["input_ids"][0].tolist()
        else:
            trunk = self.tok.encode(prefix, add_special_tokens=False)
        branches = {
            k: self.tok.encode(self.prompt.suffix_text(self.tok, q), add_special_tokens=False)
            for k, q in qs.items()
        }
        return trunk, branches, features, vision_s

    @staticmethod
    def _coerce_image(image: Any) -> Any:
        from pathlib import Path as _Path

        from PIL import Image

        if isinstance(image, (str, _Path)):
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")
        return pil.resize(VISION_SIZE, Image.Resampling.BICUBIC)

    def _embeds(self, ids: list[int], features: Any = None) -> Any:
        import numpy as np

        arr = np.array([ids], np.int64)
        e = self.emb.run(None, {"input_ids": arr})[0]
        if features is not None:
            pos = np.where(arr[0] == self.image_token_id)[0]
            flat = features.reshape(-1, FEATURE_DIM)
            assert len(pos) == len(flat), (len(pos), flat.shape)
            e[0, pos] = flat
        return e

    def _forward(
        self,
        ids: list[int],
        past: dict[str, Any] | None = None,
        past_len: int = 0,
        features: Any = None,
        cache: bool = False,
    ) -> tuple[Any, dict[str, Any] | None]:
        import numpy as np

        check_seq_len(past_len, len(ids), self.max_seq_len)
        import ml_dtypes

        types = {
            "tensor(float)": (np.float32, 1),
            "tensor(float16)": (np.float16, 10),
            "tensor(bfloat16)": (ml_dtypes.bfloat16, 16),
            "tensor(int64)": (np.int64, 7),
            "tensor(int32)": (np.int32, 6),
        }
        inputs = self.dec.get_inputs()
        fixed = next(
            (
                i.shape[2]
                for i in inputs
                if i.name.startswith("past_key_values.")
                and isinstance(i.shape[2], int)
                and i.shape[2] > 0
            ),
            None,
        )
        length = past_len + len(ids)
        if fixed and length > fixed:
            raise ValueError(
                f"prompt {length} exceeds fixed NPU capacity {fixed}"
            )
        mask = np.zeros((1, fixed or length), np.int64)
        mask[0, :length] = 1
        feeds = {
            "inputs_embeds": self._embeds(ids, features),
            "attention_mask": mask,
            "position_ids": np.arange(past_len, length, dtype=np.int64)[None, :],
        }
        for inp in inputs:
            dtype, _ = types[inp.type]
            if inp.name.startswith("past_conv."):
                shape = tuple(
                    d if isinstance(d, int) and d > 0 else 1 for d in inp.shape
                )
                feeds[inp.name] = past[inp.name].copy() if past else np.zeros(
                    shape, dtype
                )
            elif inp.name.startswith("past_key_values."):
                feeds[inp.name] = (
                    past[inp.name].copy()
                    if past
                    else np.zeros((1, 8, fixed or 0, 64), dtype)
                )
        bound = {
            i.name: np.ascontiguousarray(feeds[i.name], dtype=types[i.type][0])
            for i in inputs
        }
        io = self.dec.io_binding()
        for inp in inputs:
            v = bound[inp.name]
            io.bind_input(
                inp.name, "cpu", 0, types[inp.type][1], v.shape, v.ctypes.data
            )
        outputs: dict[str, Any] = {}
        bound_output_names: list[str] = []
        for out in self.dec.get_outputs():
            if not cache and out.name != "logits":
                continue
            dtype, code = types[out.type]
            if out.name == "logits":
                io.bind_output(out.name, "cpu")
                bound_output_names.append(out.name)
                continue
            elif out.name.startswith("present_conv."):
                shape = tuple(
                    d if isinstance(d, int) and d > 0 else 1 for d in out.shape
                )
            elif out.name.startswith("present."):
                size = (
                    out.shape[2]
                    if isinstance(out.shape[2], int) and out.shape[2] > 0
                    else (fixed or length)
                )
                shape = (1, 8, size, 64)
            else:
                raise ValueError("unknown output " + out.name)
            v = np.empty(shape, dtype)
            outputs[out.name] = v
            io.bind_output(out.name, "cpu", 0, code, v.shape, v.ctypes.data)
            bound_output_names.append(out.name)
        self.dec.run_with_iobinding(io)
        logits = io.get_outputs()[bound_output_names.index("logits")].numpy()
        state: dict[str, Any] = {}
        if cache:
            for name, v in outputs.items():
                if name == "logits":
                    continue
                name = name.replace("present_conv.", "past_conv.").replace(
                    "present.", "past_key_values."
                )
                state[name] = v.copy()
        return logits[0, -1].astype(np.float64), state if cache else None

    def _evaluate_tree(
        self,
        trunk: list[int],
        branches: dict[str, list[int]],
        qs: dict[str, Any],
        features: Any,
        vision_s: float,
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        started = time.perf_counter()
        _, past = self._forward(trunk, features=features, cache=True)
        trunk_s = time.perf_counter() - started
        before = digest_cache(past)
        probs: dict[str, list[float]] = {}
        branch_s: dict[str, float] = {}
        for key, branch in branches.items():
            branch_started = time.perf_counter()
            logits, _ = self._forward(branch, past, len(trunk))
            probs[key] = self.prompt.readout(self.tok, qs[key], logits)
            branch_s[key] = time.perf_counter() - branch_started
        elapsed = time.perf_counter() - started
        after = digest_cache(past)
        assert before == after, "shared cache was mutated"
        branch_tokens = {k: len(v) for k, v in branches.items()}
        trunk_tokens = len(trunk)
        timing = {
            "vision_s": vision_s,
            "trunk_s": trunk_s,
            "branches_s": sum(branch_s.values()),
            "total_s": elapsed + vision_s,
            "trunk_tokens": trunk_tokens,
            "branch_s": branch_s,
            "branch_tokens": branch_tokens,
            "input_tokens": trunk_tokens + sum(branch_tokens.values()),
        }
        return probs, timing
