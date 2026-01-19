# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Example: Client code for connecting to the diffusion API server.

This example demonstrates how to connect to the OpenAI-compatible
image generation server using various client methods.

Prerequisites:
    1. Start the server:
       max images serve --model black-forest-labs/FLUX.1-dev --port 8000

    2. Run this client:
       python client_example.py

Dependencies:
    pip install requests openai httpx
"""

import base64
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

PORT = args.port


def example_with_requests() -> None:
    """Example using the requests library."""
    import requests

    base_url = f"http://localhost:{PORT}"

    # Check server health
    response = requests.get(f"{base_url}/health")
    print(f"Server health: {response.json()}")

    # List available models
    response = requests.get(f"{base_url}/v1/models")
    models = response.json()
    print(f"Available models: {[m['id'] for m in models['data']]}")

    # Generate an image
    request_data = {
        "prompt": "A beautiful sunset over the ocean with palm trees",
        "size": "1024x1024",
        "n": 1,
        "response_format": "b64_json",
        # Diffusion-specific parameters
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
    }

    print(f"\nGenerating image: {request_data['prompt']}")
    response = requests.post(
        f"{base_url}/v1/images/generations",
        json=request_data,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Created at: {result['created']}")

        # Save the image
        output_dir = Path("outputs/client")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img_data in enumerate(result["data"]):
            if "b64_json" in img_data:
                image_bytes = base64.b64decode(img_data["b64_json"])
                output_path = output_dir / f"requests_output_{i}.png"
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                print(f"Image saved to: {output_path}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def example_with_openai_client() -> None:
    """Example using the official OpenAI Python client.

    Note: The OpenAI client can connect to any OpenAI-compatible server.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("Please install openai: pip install openai")
        return

    # Point to local server
    client = OpenAI(
        base_url=f"http://localhost:{PORT}/v1",
        api_key="not-needed",  # API key not required for local server
    )

    # List models
    models = client.models.list()
    print(f"Available models: {[m.id for m in models.data]}")

    # Generate an image
    print("\nGenerating image with OpenAI client...")

    # Note: The OpenAI client's images.generate() may not support
    # all diffusion-specific parameters. Use raw HTTP for full control.
    response = client.images.generate(
        model="black-forest-labs/FLUX.1-dev",
        prompt="A majestic dragon flying over a medieval castle",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )

    print(f"Created at: {response.created}")

    # Save the image
    output_dir = Path("outputs/client")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_data in enumerate(response.data):
        if img_data.b64_json:
            image_bytes = base64.b64decode(img_data.b64_json)
            output_path = output_dir / f"openai_client_output_{i}.png"
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            print(f"Image saved to: {output_path}")


def example_with_httpx_async() -> None:
    """Example using httpx for async requests."""
    import asyncio

    try:
        import httpx
    except ImportError:
        print("Please install httpx: pip install httpx")
        return

    async def generate_image() -> None:
        async with httpx.AsyncClient(timeout=300.0) as client:
            base_url = f"http://localhost:{PORT}"

            # Generate image
            request_data = {
                "prompt": "A cyberpunk street scene at night with neon lights",
                "size": "1024x1024",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": 30,
                "guidance_scale": 3.5,
            }

            print(f"Generating: {request_data['prompt']}")
            response = await client.post(
                f"{base_url}/v1/images/generations",
                json=request_data,
            )

            if response.status_code == 200:
                result = response.json()

                output_dir = Path("outputs/client")
                output_dir.mkdir(parents=True, exist_ok=True)

                for i, img_data in enumerate(result["data"]):
                    if "b64_json" in img_data:
                        image_bytes = base64.b64decode(img_data["b64_json"])
                        output_path = output_dir / f"httpx_output_{i}.png"
                        with open(output_path, "wb") as f:
                            f.write(image_bytes)
                        print(f"Image saved to: {output_path}")
            else:
                print(f"Error: {response.status_code}")

    asyncio.run(generate_image())


def example_curl_commands() -> None:
    """Print curl commands for reference."""
    print("\n" + "=" * 60)
    print("CURL COMMAND EXAMPLES")
    print("=" * 60)

    print("\n1. Check server health:")
    print(f"   curl http://localhost:{PORT}/health")

    print("\n2. List available models:")
    print(f"   curl http://localhost:{PORT}/v1/models")

    print("\n3. Generate an image:")
    print(
        f"""   curl http://localhost:{PORT}/v1/images/generations \\
     -H "Content-Type: application/json" \\
     -d '{
            "prompt": "A beautiful sunset over mountains",
       "size": "1024x1024",
       "n": 1,
       "response_format": "b64_json",
       "num_inference_steps": 30,
       "guidance_scale": 3.5
     }'"""
    )

    print("\n4. Generate and save image (with jq):")
    print(
        f"""   curl -s http://localhost:{PORT}/v1/images/generations \\
     -H "Content-Type: application/json" \\
     -d '{"prompt": "A cat", "size": "512x512"}' \\
     | jq -r '.data[0].b64_json' | base64 -d > output.png"""
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Diffusion API Client Examples")
    print("=" * 60)
    print("\nMake sure the server is running:")
    print(
        f"  max images serve --model black-forest-labs/FLUX.1-dev --port {PORT}"
    )
    print()

    # Print curl examples first
    # example_curl_commands()

    # Uncomment to run the examples:
    print("\n" + "=" * 60)
    print("Running requests example...")
    print("=" * 60)
    example_with_requests()

    print("\n" + "=" * 60)
    print("Running OpenAI client example...")
    print("=" * 60)
    example_with_openai_client()

    print("\n" + "=" * 60)
    print("Running httpx async example...")
    print("=" * 60)
    example_with_httpx_async()
