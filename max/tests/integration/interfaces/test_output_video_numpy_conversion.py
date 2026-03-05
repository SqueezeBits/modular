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

"""Tests for numpy/bytes to OutputVideoContent conversion."""

import base64

import numpy as np
from max.interfaces.generation import GenerationOutput
from max.interfaces.request import RequestID
from max.interfaces.request.open_responses import OutputVideoContent
from max.interfaces.status import GenerationStatus


def test_output_video_from_numpy_frames_gif() -> None:
    frames = np.random.rand(8, 64, 64, 3).astype(np.float32)

    output = OutputVideoContent.from_numpy_frames(
        frames,
        format="gif",
        frames_per_second=12,
    )

    assert output.type == "output_video"
    assert output.video_data is not None
    assert output.format == "gif"
    assert output.frames_per_second == 12
    assert output.num_frames == 8
    assert output.width == 64
    assert output.height == 64


def test_output_video_from_encoded_bytes() -> None:
    payload = b"fake-video-bytes"
    output = OutputVideoContent.from_encoded_bytes(
        payload,
        format="mp4",
        frames_per_second=25,
        num_frames=16,
        width=704,
        height=512,
    )

    assert output.type == "output_video"
    assert output.video_data is not None
    assert base64.b64decode(output.video_data) == payload
    assert output.format == "mp4"
    assert output.frames_per_second == 25
    assert output.num_frames == 16
    assert output.width == 704
    assert output.height == 512


def test_generation_output_with_video_content() -> None:
    frames = np.random.rand(4, 32, 32, 3).astype(np.float32)
    video = OutputVideoContent.from_numpy_frames(
        frames, format="gif", frames_per_second=8
    )

    output = GenerationOutput(
        request_id=RequestID("req-video"),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[video],
    )

    assert output.is_done
    assert output.output[0].type == "output_video"
