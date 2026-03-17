# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Simple browser demo for diffusion image generation.

Usage:
    ./bazelw run //max/examples/diffusion:web_demo -- \
        --model black-forest-labs/FLUX.2-klein-4b
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import cast
from urllib.parse import parse_qs

from jinja2 import Environment
from max.driver import DeviceSpec
from max.interfaces import (
    PipelineTask,
    PixelGenerationInputs,
    RequestID,
)
from max.interfaces.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import (
    OpenResponsesRequestBody,
    OutputImageContent,
)
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.interfaces.cache_mixin import DenoisingCacheConfig
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)


DEFAULT_PROMPT = "A cat holding a sign that says hello world"
_PAGE_TEMPLATE = Environment(autoescape=True).from_string("""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Diffusion Web Demo</title>
    <style>
      :root {
        --bg: #f4efe6;
        --bg-accent: #dce7e0;
        --panel: rgba(255, 252, 247, 0.88);
        --border: rgba(41, 52, 47, 0.12);
        --text: #1f2825;
        --muted: #64716c;
        --accent: #1f6f5f;
        --accent-strong: #155247;
        --error: #9a2f2f;
        --shadow: 0 22px 70px rgba(31, 40, 37, 0.14);
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        min-height: 100vh;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(255, 255, 255, 0.9), transparent 36%),
          radial-gradient(circle at bottom right, rgba(31, 111, 95, 0.10), transparent 28%),
          linear-gradient(135deg, var(--bg), var(--bg-accent));
        font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      }
      .page {
        max-width: 1240px;
        margin: 0 auto;
        padding: 40px 24px 56px;
      }
      .hero {
        margin-bottom: 28px;
      }
      .eyebrow {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(31, 111, 95, 0.10);
        color: var(--accent-strong);
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      h1 {
        margin: 16px 0 10px;
        font-size: clamp(2.2rem, 5vw, 4rem);
        line-height: 0.96;
      }
      .subtle {
        color: var(--muted);
      }
      .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 18px;
      }
      .chip {
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--border);
        font-size: 14px;
      }
      .layout {
        display: grid;
        grid-template-columns: minmax(300px, 400px) minmax(0, 1fr);
        gap: 24px;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 28px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(16px);
      }
      .form-panel {
        padding: 22px;
      }
      .result-panel {
        padding: 18px;
        min-height: 520px;
      }
      label {
        display: block;
        margin-bottom: 16px;
        font-size: 14px;
        font-weight: 600;
      }
      textarea,
      input {
        width: 100%;
        margin-top: 8px;
        border: 1px solid rgba(31, 40, 37, 0.12);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.92);
        color: var(--text);
        padding: 14px 16px;
        font: inherit;
      }
      textarea {
        min-height: 180px;
        resize: vertical;
      }
      .row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
      }
      button {
        width: 100%;
        border: 0;
        border-radius: 18px;
        padding: 15px 18px;
        margin-top: 8px;
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        color: white;
        font: inherit;
        font-weight: 700;
        cursor: pointer;
        transition: transform 120ms ease, box-shadow 120ms ease;
        box-shadow: 0 18px 36px rgba(21, 82, 71, 0.24);
      }
      button:hover {
        transform: translateY(-1px);
      }
      .message {
        display: grid;
        gap: 6px;
        padding: 14px 16px;
        border-radius: 18px;
        margin-bottom: 14px;
      }
      .error {
        background: rgba(154, 47, 47, 0.08);
        border: 1px solid rgba(154, 47, 47, 0.18);
        color: var(--error);
      }
      .result-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin-bottom: 14px;
        color: var(--muted);
      }
      .result-meta strong {
        color: var(--text);
        font-size: 1.1rem;
      }
      .result-image {
        width: 100%;
        display: block;
        border-radius: 22px;
        background: white;
        border: 1px solid var(--border);
      }
      .empty-state {
        min-height: 480px;
        border-radius: 22px;
        display: grid;
        place-items: center;
        padding: 24px;
        text-align: center;
        background:
          linear-gradient(135deg, rgba(255, 255, 255, 0.85), rgba(219, 231, 224, 0.7));
        border: 1px dashed rgba(31, 40, 37, 0.16);
      }
      @media (max-width: 920px) {
        .layout {
          grid-template-columns: 1fr;
        }
      }
      @media (max-width: 640px) {
        .page {
          padding: 24px 14px 40px;
        }
        .row {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <span class="eyebrow">MAX Diffusion Demo</span>
        <h1>Prompt to image in one page.</h1>
        <p class="subtle">
          Submit a prompt, set resolution and seed, and render directly in the browser.
        </p>
        <div class="hero-meta">
          <span class="chip">Model: {{ model_label }}</span>
          <span class="chip">{{ fixed_settings }}</span>
        </div>
      </section>

      <section class="layout">
        <form id="demo-form" class="panel form-panel" method="post" action="/generate">
          <label>
            Prompt
            <textarea name="prompt" required>{{ prompt_value }}</textarea>
          </label>
          <div class="row">
            <label>
              Height
              <input type="number" min="1" step="1" name="height" value="{{ height_value }}" />
            </label>
            <label>
              Width
              <input type="number" min="1" step="1" name="width" value="{{ width_value }}" />
            </label>
            <label>
              Seed
              <input type="number" step="1" name="seed" value="{{ seed_value }}" />
            </label>
          </div>
          <button id="generate-button" type="submit">Generate Image</button>
        </form>

        <section class="panel result-panel">
          {% if error %}
            <div class="message error">
              <strong>Generation failed.</strong>
              <span>{{ error }}</span>
            </div>
          {% endif %}

          {% if result %}
            <div class="result-meta">
              <span>Latency</span>
              <strong>{{ latency_ms }} ms</strong>
              <span>{{ result.width }} x {{ result.height }}</span>
              <span>seed {{ seed_label }}</span>
            </div>
            <img
              class="result-image"
              src="data:image/png;base64,{{ result.image_data }}"
              alt="{{ result.prompt }}"
            />
          {% else %}
            <div class="empty-state">
              <p>Submit a prompt to generate an image.</p>
              <p class="subtle">Latency and the generated image will appear here.</p>
            </div>
          {% endif %}
        </section>
      </section>
    </main>

    <script>
      const form = document.getElementById("demo-form");
      const button = document.getElementById("generate-button");
      form.addEventListener("submit", () => {
        button.disabled = true;
        button.textContent = "Generating...";
      });
    </script>
  </body>
</html>
""")


@dataclass(frozen=True)
class GenerationForm:
    prompt: str
    height: int | None
    width: int | None
    seed: int | None


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    height: int
    width: int
    seed: int | None
    latency_s: float
    image_data: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a simple web UI for diffusion image generation.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Identifier of the model to use for generation.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to.",
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        action="append",
        default=None,
        help="Optional model weight file path(s). Can be specified multiple times.",
    )
    parser.add_argument(
        "--quantization-encoding",
        type=str,
        default=None,
        choices=[
            "float32",
            "bfloat16",
            "q4_k",
            "q4_0",
            "q6_k",
            "float8_e4m3fn",
            "float4_e2m1fnx2",
            "gptq",
        ],
        help="Optional weight encoding type.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt applied to every request.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=4,
        help="Number of denoising steps to run for each request.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=1,
        help="Number of warmup requests to run on startup.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum primary tokenizer length.",
    )
    parser.add_argument(
        "--secondary-max-length",
        type=int,
        default=None,
        help="Maximum secondary tokenizer length.",
    )
    parser.add_argument(
        "--default-prompt",
        default=DEFAULT_PROMPT,
        help="Initial prompt shown in the web form.",
    )
    parser.add_argument(
        "--default-height",
        type=int,
        default=1024,
        help="Initial height shown in the web form.",
    )
    parser.add_argument(
        "--default-width",
        type=int,
        default=1024,
        help="Initial width shown in the web form.",
    )
    parser.add_argument(
        "--default-seed",
        type=int,
        default=42,
        help="Initial seed shown in the web form.",
    )
    parser.add_argument(
        "--first-block-caching",
        action="store_true",
        help="Enable first-block step cache optimization.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=None,
        help="Relative-difference threshold for step cache.",
    )
    parser.add_argument(
        "--taylorseer",
        action="store_true",
        help="Enable TaylorSeer cache optimization.",
    )
    parser.add_argument(
        "--taylorseer-cache-interval",
        type=int,
        default=None,
        help="Steps between full computations for TaylorSeer.",
    )
    parser.add_argument(
        "--taylorseer-warmup-steps",
        type=int,
        default=None,
        help="Warmup steps for TaylorSeer factor gathering.",
    )
    parser.add_argument(
        "--taylorseer-max-order",
        type=int,
        default=None,
        choices=[1, 2],
        help="Taylor expansion order: 1=linear, 2=quadratic.",
    )

    args = parser.parse_args(argv)
    assert args.port > 0, "port must be positive"
    assert args.num_inference_steps > 0, "num-inference-steps must be positive"
    assert args.guidance_scale > 0.0, "guidance-scale must be positive"
    if args.default_height is not None:
        assert args.default_height > 0, "default-height must be positive"
    if args.default_width is not None:
        assert args.default_width > 0, "default-width must be positive"
    return args


def _optional_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be positive")
    return parsed


def _first_image_data(contents: list[object] | None) -> str:
    if not contents:
        raise RuntimeError("no images were generated")

    for content in contents:
        if isinstance(content, OutputImageContent) and content.image_data:
            return content.image_data

    raise RuntimeError("generation completed without inline image data")


class DiffusionWebDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._lock = Lock()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._closed = False
        self._tokenizer, self._pipeline = self._build_runtime()
        if args.num_warmups > 0:
            self._warmup()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.close()

    def default_form(self) -> GenerationForm:
        return GenerationForm(
            prompt=self._args.default_prompt,
            height=self._args.default_height,
            width=self._args.default_width,
            seed=self._args.default_seed,
        )

    def model_label(self) -> str:
        return self._args.model

    def fixed_settings(self) -> str:
        return (
            f"{self._args.num_inference_steps} steps"
            f" | guidance {self._args.guidance_scale}"
            f" | warmups {self._args.num_warmups}"
        )

    def _build_runtime(
        self,
    ) -> tuple[PixelGenerationTokenizer, PixelGenerationPipeline[PixelContext]]:
        print(f"Loading model for web demo: {self._args.model}")
        config = PipelineConfig(
            model=MAXModelConfig(
                model_path=self._args.model,
                device_specs=[DeviceSpec.accelerator()],
                weight_path=(
                    [Path(p) for p in self._args.weight_path]
                    if self._args.weight_path
                    else []
                ),
                quantization_encoding=self._args.quantization_encoding,
            ),
            runtime=PipelineRuntimeConfig(
                prefer_module_v3=True,
            ),
        )
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            config.model.huggingface_model_repo,
            prefer_module_v3=config.runtime.prefer_module_v3,
            task=PipelineTask.PIXEL_GENERATION,
        )
        assert arch is not None, (
            "No matching diffusion architecture found for the provided model."
        )

        has_tokenizer_2 = False
        diffusers_config = config.model.diffusers_config
        max_length = self._args.max_length
        secondary_max_length = self._args.secondary_max_length

        if (
            max_length is None
            and diffusers_config is not None
            and (components_config := diffusers_config.get("components", None))
            and (components_config.get("tokenizer", None) is not None)
        ):
            max_length = components_config["tokenizer"]["config_dict"].get(
                "model_max_length", None
            )
            if arch.name in (
                "Flux2Pipeline_ModuleV3",
                "Flux2KleinPipeline_ModuleV3",
            ):
                max_length = 512
            print(f"Using max length: {max_length} for tokenizer")

        if (
            secondary_max_length is None
            and diffusers_config is not None
            and (components_config := diffusers_config.get("components", None))
            and (components_config.get("tokenizer_2", None) is not None)
        ):
            has_tokenizer_2 = True
            secondary_max_length = components_config["tokenizer_2"][
                "config_dict"
            ].get("model_max_length", None)
            print(
                "Using secondary max length:"
                f" {secondary_max_length} for tokenizer_2"
            )

        tokenizer = PixelGenerationTokenizer(
            model_path=self._args.model,
            pipeline_config=config,
            subfolder="tokenizer",
            max_length=max_length,
            subfolder_2="tokenizer_2" if has_tokenizer_2 else None,
            secondary_max_length=(
                secondary_max_length if has_tokenizer_2 else None
            ),
        )

        if not issubclass(arch.pipeline_model, DiffusionPipeline):
            raise TypeError(
                "Selected architecture does not implement DiffusionPipeline: "
                f"{arch.pipeline_model}"
            )
        pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
        cache_config = DenoisingCacheConfig(
            first_block_caching=self._args.first_block_caching,
            residual_threshold=self._args.residual_threshold,
            taylorseer=self._args.taylorseer,
            taylorseer_cache_interval=self._args.taylorseer_cache_interval,
            taylorseer_warmup_steps=self._args.taylorseer_warmup_steps,
            taylorseer_max_order=self._args.taylorseer_max_order,
        )
        pipeline = PixelGenerationPipeline[PixelContext](
            pipeline_config=config,
            pipeline_model=pipeline_model,
            cache_config=cache_config,
        )
        return tokenizer, pipeline

    def _warmup(self) -> None:
        form = self.default_form()
        warmup_prompt = form.prompt or DEFAULT_PROMPT
        print(f"Running {self._args.num_warmups} warmup request(s)")
        for i in range(self._args.num_warmups):
            print(f"Warmup {i + 1}/{self._args.num_warmups}")
            self._run_generation(
                GenerationForm(
                    prompt=warmup_prompt,
                    height=form.height,
                    width=form.width,
                    seed=form.seed,
                )
            )
        print("Warmup complete")

    def generate(self, form: GenerationForm) -> GenerationResult:
        with self._lock:
            return self._run_generation(form)

    def _run_generation(self, form: GenerationForm) -> GenerationResult:
        return self._loop.run_until_complete(self._generate_async(form))

    async def _generate_async(self, form: GenerationForm) -> GenerationResult:
        prompt = form.prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        request = OpenResponsesRequest(
            request_id=RequestID(),
            body=OpenResponsesRequestBody(
                model=self._args.model,
                input=prompt,
                seed=form.seed,
                provider_options=ProviderOptions(
                    image=ImageProviderOptions(
                        negative_prompt=self._args.negative_prompt,
                        height=form.height,
                        width=form.width,
                        steps=self._args.num_inference_steps,
                        guidance_scale=self._args.guidance_scale,
                    )
                ),
            ),
        )

        t0 = perf_counter()
        context = await self._tokenizer.new_context(request)
        inputs = PixelGenerationInputs[PixelContext](
            batch={context.request_id: context}
        )
        outputs = self._pipeline.execute(inputs)
        output = outputs[context.request_id]
        output = await self._tokenizer.postprocess(output)
        latency_s = perf_counter() - t0

        if not output.is_done:
            raise RuntimeError(
                f"generation finished with status {output.final_status}"
            )

        return GenerationResult(
            prompt=prompt,
            height=context.height,
            width=context.width,
            seed=form.seed,
            latency_s=latency_s,
            image_data=_first_image_data(output.output),
        )


def _render_page(
    app: DiffusionWebDemo,
    form: GenerationForm,
    *,
    result: GenerationResult | None = None,
    error: str | None = None,
) -> str:
    return _PAGE_TEMPLATE.render(
        error=error,
        fixed_settings=app.fixed_settings(),
        height_value="" if form.height is None else str(form.height),
        latency_ms=(
            f"{result.latency_s * 1000.0:.1f}" if result is not None else None
        ),
        model_label=app.model_label(),
        prompt_value=form.prompt,
        result=result,
        seed_label=(
            "random"
            if result is not None and result.seed is None
            else (str(result.seed) if result is not None else None)
        ),
        seed_value="" if form.seed is None else str(form.seed),
        width_value="" if form.width is None else str(form.width),
    )


class _ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def _make_handler(app: DiffusionWebDemo) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._respond_plain("ok\n")
                return
            if self.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._respond_html(_render_page(app, app.default_form()))

        def do_POST(self) -> None:
            if self.path != "/generate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_form = self.rfile.read(content_length).decode("utf-8")
                parsed = parse_qs(raw_form, keep_blank_values=True)
                form = GenerationForm(
                    prompt=parsed.get("prompt", [""])[0],
                    height=_optional_int(parsed.get("height", [""])[0]),
                    width=_optional_int(parsed.get("width", [""])[0]),
                    seed=_optional_seed(parsed.get("seed", [""])[0]),
                )
                result = app.generate(form)
                self._respond_html(_render_page(app, form, result=result))
            except Exception as exc:
                fallback_form = app.default_form()
                try:
                    fallback_form = GenerationForm(
                        prompt=parsed.get("prompt", [""])[0],
                        height=_optional_int(parsed.get("height", [""])[0]),
                        width=_optional_int(parsed.get("width", [""])[0]),
                        seed=_optional_seed(parsed.get("seed", [""])[0]),
                    )
                except Exception:
                    pass
                self._respond_html(
                    _render_page(app, fallback_form, error=str(exc)),
                    status=HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, fmt: str, *args: object) -> None:
            print(
                f"[web-demo] {self.address_string()} - {fmt % args}"
            )

        def _respond_plain(
            self, body: str, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _respond_html(
            self, body: str, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def _optional_seed(raw: str | None) -> int | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return int(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = DiffusionWebDemo(args)
    atexit.register(app.close)

    server = _ReusableHTTPServer((args.host, args.port), _make_handler(app))
    print(
        "Serving diffusion demo at "
        f"http://{args.host}:{args.port} (health: /healthz)"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web demo")
    finally:
        server.server_close()
        app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
