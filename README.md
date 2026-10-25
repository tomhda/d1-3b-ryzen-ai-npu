# d1-3B on the Ryzen AI NPU

Runtime and conversion scripts for [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) on the NPU of AMD Ryzen AI 300-series processors (XDNA 2). Both the language model and the vision encoder run on the NPU. The converted model is on Hugging Face: [Tomoali/d1-3B-ryzen-ai-npu](https://huggingface.co/Tomoali/d1-3B-ryzen-ai-npu).

d1 answers typed questions about a state (text and at most one image) in one forward pass and returns probabilities:

- `noul`: yes or no
- `choice`: one of several labels
- `score`: a level on a scale

The state is read once into a KV cache. Each question then branches from a copy of that cache, so adding questions does not re-read the state.

## Requirements

- Windows 11 on a Ryzen AI 300-series processor, with the NPU driver installed
- [Ryzen AI Software 1.8.0](https://ryzenai.docs.amd.com/), installed in `C:/Program Files/RyzenAI/1.8.0`. Its conda environment `ryzen-ai-1.8.0` provides ONNX Runtime with `RyzenAILightExecutionProvider` and `VitisAIExecutionProvider`.

## Install

Clone the Ryzen AI environment and install the package into the clone:

```bash
conda create --clone ryzen-ai-1.8.0 -n d1npu
conda activate d1npu
pip install -e ".[download]"
```

The package does not depend on the PyPI `onnxruntime`, which would shadow the ONNX Runtime from Ryzen AI Software.

## Download the model

```bash
python -m d1npu download --to C:/models/d1-3B-ryzen-ai-npu
```

The download is about 3.5 GB. The first time an image is read, the vision encoder is compiled for the NPU. This takes about 10 minutes, and the result is cached in `%LOCALAPPDATA%\d1npu\vision-cache`.

## Use

### Screen check

`screen-check` runs the preset in [`d1npu/screen_check.json`](d1npu/screen_check.json) on a browser screenshot. The preset has 10 questions, among them:

- whether a login form is shown
- the page type
- error messages and dialogs
- loading indicators
- CAPTCHAs, tables, date pickers and menus

```bash
python -m d1npu screen-check --model C:/models/d1-3B-ryzen-ai-npu --image screenshot.png
```

Each answer has a probability `p` and a `decision`. For yes/no questions, `decision` applies the threshold tuned for that question.

### Your own questions

The questions file uses the same JSON form as the official d1 package:

```json
{
  "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
  "tone": {"type": "choice", "instructions": "What is the tone of the message?",
           "criteria": {"calm": null, "annoyed": null, "angry": null}}
}
```

```bash
python -m d1npu decide --model C:/models/d1-3B-ryzen-ai-npu --state @message.txt --questions questions.json
```

Pass `--image` to add a screenshot. Every image is stretched to 640x400, the only input size of the NPU vision encoder.

A yes/no question can also carry two optional fields:

- `threshold`: the probability of yes at or above which `decision` is yes (default 0.5)
- `invert`: set when the question asks for absence. The answer then reports 1 − P(yes), the probability that the thing is present, and `decision` applies the threshold to that.

### HTTP server

```bash
python -m d1npu serve --model C:/models/d1-3B-ryzen-ai-npu --port 8787
```

`POST /api/alpha/decisions` and `POST /api/v1/systemone` take `{"state": ..., "questions": ...}`, in the shape of the OpenRouter Decisions API. The state is a string, or a list of strings plus at most one `{"type": "image_url", "image_url": {"url": ...}}` object holding a PNG or JPEG data URL. Requests are handled one at a time.

### Python

```python
from d1npu import D1

d1 = D1("C:/models/d1-3B-ryzen-ai-npu")
answers = d1.decide("state text", {"q": {"type": "noul", "instructions": "..."}}, image="screenshot.png")
```

The native providers can crash while Python shuts down. After the output is written, end the process with `os._exit(0)`.

## Measurements

Measured on a Ryzen AI 7 PRO 350 (32 GB, NPU driver 32.0.203.329) with 27 questions over text and two screenshots:

- Same choice as the official fp32 model on all 27 questions, with a largest probability difference of 0.130.
- 3.51 answers per second, re-reading the images each pass, at about 1.1 J per answer. CPU usage is unchanged from idle. On the same machine, llama.cpp on the GPU gives 2.45 answers per second at 3.6 J.

On 100 browser screenshots with the 10 screen questions, prompts and thresholds were tuned on 59 screenshots and evaluated on the held-out 41:

- Baseline prompts: 77.3% accuracy
- Preset prompts and thresholds: 95.8% accuracy

Choosing the next action from a long list of UI candidates gave about 40% acceptable choices. The 3B model gets similar results without the NPU conversion.

Notes:

- How the model was put on the NPU: [docs/npu-conversion.md](docs/npu-conversion.md)
- Full measurements, in Japanese: [docs/ja/research.md](docs/ja/research.md)

## Converting the model yourself

The scripts in [`conversion/`](conversion/) rebuild the model folder from `onnx-community/d1-3B-ONNX` and `LiquidAI/d1-3B`:

- `language.py`: converts the 4-bit decoder with `model_generate --npu --basic`
- `vision.py`: builds the fixed-shape vision encoder from the fp32 `onnx/vision_encoder.onnx`
- `assemble.py`: lays out the folder

Run them in the Ryzen AI environment, which provides `model_generate`:

```bash
python conversion/language.py --onnx-dir d1-3B-ONNX --work-dir work/input --output work/output --language-dir work/language
python conversion/vision.py --input d1-3B-ONNX/onnx/vision_encoder.onnx --output work/vision_encoder_npu.onnx --check
python conversion/assemble.py --language-dir work/language --onnx-dir d1-3B-ONNX --vision-onnx work/vision_encoder_npu.onnx --src-dir d1-3B --output d1-3B-ryzen-ai-npu
```

Each script lists its arguments with `--help`.

## License

The code in this repository is under the MIT License (`LICENSE`).

The model files are under the LFM Open License v1.0. That license does not cover commercial use by legal entities with USD 10 million or more in annual revenue. This project is not affiliated with Liquid AI or AMD.
