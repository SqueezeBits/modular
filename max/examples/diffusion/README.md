# Diffusion Image Generation Examples

This directory contains examples for using the MAX diffusion pipeline for image generation.

## Overview

The MAX diffusion pipeline supports:
- **Offline generation**: Direct Python API for generating images
- **OpenAI-compatible API**: Server with `/v1/images/generations` endpoint
- **Multiple models**: FLUX.1-dev, and other diffusion models

## Examples

### 1. Offline Generation (`offline_generation.py`)

Basic example using the `ImageGenerator` directly:

```bash
python offline_generation.py
```

```python
from max.entrypoints.diffusion import ImageGenerator
from max.pipelines import PipelineConfig

config = PipelineConfig(model_path="black-forest-labs/FLUX.1-dev")
generator = ImageGenerator(config)

# Generate returns a list of PIL Images
images = generator.generate(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=3.5,
)
images[0].save("output.png")
```

### 2. OpenAI API Example (`openai_api_example.py`)

Using the OpenAI-compatible `ImageGenerationRequest`:

```bash
python openai_api_example.py
```

```python
from max.entrypoints.diffusion import ImageGenerator
from max.interfaces import ImageGenerationRequest
from max.pipelines import PipelineConfig

config = PipelineConfig(model_path="black-forest-labs/FLUX.1-dev")
generator = ImageGenerator(config)

# Use OpenAI-compatible request format
request = ImageGenerationRequest(
    prompt="A futuristic city skyline at sunset",
    size="1024x1024",
    n=1,
    response_format="b64_json",
    num_inference_steps=30,
    guidance_scale=3.5,
)

# create() returns an OpenAI-compatible response
response = generator.create(request)
# response.data[0].b64_json contains the base64-encoded image
```

### 3. Client Example (`client_example.py`)

Connecting to the OpenAI-compatible server:

```bash
# Start the server
max images serve --model black-forest-labs/FLUX.1-dev --port 8000

# Run the client
python client_example.py
```

## CLI Commands

### Generate Images

```bash
# Basic generation
max images generate \
    --model black-forest-labs/FLUX.1-dev \
    --prompt "A beautiful sunset over mountains" \
    --size 1024x1024 \
    --output output.png

# With custom parameters
max images generate \
    --model black-forest-labs/FLUX.1-dev \
    --prompt "A cyberpunk city" \
    --size 1792x1024 \
    --num-inference-steps 50 \
    --guidance-scale 7.5 \
    --seed 42 \
    --output landscape.png
```

### Start Server

```bash
# Start OpenAI-compatible API server
max images serve \
    --model black-forest-labs/FLUX.1-dev \
    --port 8000
```

## API Reference

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | Required | Text description of the desired image |
| `model` | string | null | Model to use for generation |
| `n` | integer | 1 | Number of images to generate (1-10) |
| `size` | string | "1024x1024" | Image size (e.g., "1024x1024", "1792x1024") |
| `quality` | string | "standard" | Image quality |
| `response_format` | string | "b64_json" | Response format ("url" or "b64_json") |
| `output_format` | string | "png" | Output format ("png", "jpeg", "webp") |
| `num_inference_steps` | integer | 50 | Number of denoising steps |
| `guidance_scale` | float | 3.5 | Classifier-free guidance scale |
| `seed` | integer | null | Random seed for reproducibility |

### Response Format

```json
{
  "created": 1713833628,
  "data": [
    {
      "b64_json": "iVBORw0KGgo...",
      "revised_prompt": "A beautiful sunset over mountains"
    }
  ]
}
```

## curl Examples

### Generate Image

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "size": "1024x1024",
    "n": 1,
    "response_format": "b64_json",
    "num_inference_steps": 30,
    "guidance_scale": 3.5
  }'
```

### Generate and Save

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat", "size": "512x512"}' \
  | jq -r '.data[0].b64_json' | base64 -d > output.png
```

### List Models

```bash
curl http://localhost:8000/v1/models
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Using with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.images.generate(
    model="black-forest-labs/FLUX.1-dev",
    prompt="A majestic dragon flying over a castle",
    size="1024x1024",
    n=1,
    response_format="b64_json",
)

# Save the image
import base64
image_bytes = base64.b64decode(response.data[0].b64_json)
with open("dragon.png", "wb") as f:
    f.write(image_bytes)
```

## Supported Models

- `black-forest-labs/FLUX.1-dev` - Flux 1 Dev

## Environment Variables

| Variable | Description |
|----------|-------------|
| `USE_TORCH_RANDN` | Set to "1" to use torch-based random latents |
| `SEED` | Random seed for reproducibility |
