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
"""UniPC multistep scheduler shim for diffusion models."""

import numpy as np
import numpy.typing as npt


class UniPCMultistepScheduler:
    """Minimal scheduler interface compatible with PixelGenerationTokenizer."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        order: int = 1,
        use_flow_sigmas: bool = False,
        flow_shift: float = 1.0,
        time_shift_type: str = "exponential",
        final_sigmas_type: str = "zero",
        **unused_kwargs,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.order = int(order)
        self.use_flow_sigmas = use_flow_sigmas
        self.flow_shift = float(flow_shift)
        self.time_shift_type = time_shift_type
        self.final_sigmas_type = final_sigmas_type

    def retrieve_timesteps_and_sigmas(
        self,
        image_seq_len: int,
        num_inference_steps: int,
        reverse: bool = False,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Build timestep/sigma schedule, with flow-matching support."""
        del image_seq_len

        if self.use_flow_sigmas:
            # Flow-matching sigma schedule (matches diffusers UniPC flow path)
            sigmas = np.linspace(
                1.0, 1.0 / num_inference_steps, num_inference_steps,
                dtype=np.float64,
            )

            # Apply flow_shift (exponential time shift)
            if self.flow_shift != 1.0 and self.time_shift_type == "exponential":
                sigmas = (
                    self.flow_shift * sigmas
                    / (1.0 + (self.flow_shift - 1.0) * sigmas)
                )

            sigmas = sigmas.astype(np.float32)

            if self.final_sigmas_type == "zero":
                sigmas = np.append(sigmas, np.float32(0.0))
            else:
                sigmas = np.append(sigmas, sigmas[-1])

            # Timesteps in [0, num_train_timesteps] range for transformer
            timesteps = (sigmas[:-1] * self.num_train_timesteps).astype(
                np.float32
            )
        else:
            # Discrete-time schedule (original behavior)
            timesteps = np.linspace(
                self.num_train_timesteps - 1,
                0,
                num_inference_steps,
                dtype=np.float32,
            )
            sigmas = timesteps / float(self.num_train_timesteps)
            if reverse:
                timesteps = (
                    (float(self.num_train_timesteps) - timesteps)
                    / float(self.num_train_timesteps)
                ).astype(np.float32)
            else:
                timesteps = (
                    timesteps / float(self.num_train_timesteps)
                ).astype(np.float32)
            sigmas = np.append(sigmas.astype(np.float32), np.float32(0.0))

        return timesteps, sigmas
