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

"""Runtime helpers for bundled NVIDIA user-space libraries.

Some VAE paths still rely on cuDNN fallbacks today, so we preload the bundled
NVIDIA wheel libraries at process startup. Once those paths are fully ported to
Mojo/MAX-native implementations, this helper should be removed.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

_NVIDIA_RUNTIME_WHEELS = (
    (
        "MODULAR_DIFFUSION_NVIDIA_CUDA_RUNTIME_ROOT",
        ("site-packages", "nvidia", "cuda_runtime", "lib"),
        ("libcudart.so.12",),
    ),
    (
        "MODULAR_DIFFUSION_NVIDIA_CUBLAS_ROOT",
        ("site-packages", "nvidia", "cublas", "lib"),
        ("libcublasLt.so.12", "libcublas.so.12", "libnvblas.so.12"),
    ),
    (
        "MODULAR_DIFFUSION_NVIDIA_CUDNN_ROOT",
        ("site-packages", "nvidia", "cudnn", "lib"),
        (
            "libcudnn.so.9",
            "libcudnn_graph.so.9",
            "libcudnn_ops.so.9",
            "libcudnn_adv.so.9",
            "libcudnn_cnn.so.9",
            "libcudnn_heuristic.so.9",
            "libcudnn_engines_precompiled.so.9",
            "libcudnn_engines_runtime_compiled.so.9",
        ),
    ),
)


@lru_cache(maxsize=1)
def preload_bundled_nvidia_runtime_libraries() -> None:
    """Preload bundled NVIDIA libraries needed by current cuDNN fallback paths."""
    load_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for env_var, relative_dir, libraries in _NVIDIA_RUNTIME_WHEELS:
        if not (root := os.getenv(env_var)):
            continue

        lib_dir = Path(root, *relative_dir)
        if not lib_dir.is_dir():
            continue

        for library in libraries:
            lib_path = lib_dir / library
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=load_mode)
