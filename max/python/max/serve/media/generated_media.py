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
"""Temporary media storage for file-backed generated outputs."""

from __future__ import annotations

import base64
import mimetypes
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
from max.interfaces.request.open_responses import OutputImageContent
from PIL import Image


class GeneratedMediaStorageError(RuntimeError):
    """Raised when generated media cannot be retained within local storage."""


@dataclass(frozen=True)
class StoredMediaAsset:
    """Metadata for a generated file persisted on disk."""

    asset_id: str
    path: Path
    media_type: str
    filename: str
    kind: str
    size_bytes: int


class GeneratedMediaStore:
    """Stores generated media files for later download via HTTP."""

    def __init__(
        self,
        root_dir: Path,
        max_storage_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self._root_dir = root_dir
        self._images_dir = root_dir / "images"
        self._videos_dir = root_dir / "videos"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._videos_dir.mkdir(parents=True, exist_ok=True)
        self._max_storage_bytes = max_storage_bytes
        self._images: dict[str, StoredMediaAsset] = {}
        self._videos: dict[str, StoredMediaAsset] = {}
        self._assets: dict[str, StoredMediaAsset] = {}
        self._total_storage_bytes = 0

    def get_image(self, image_id: str) -> StoredMediaAsset | None:
        return self._images.get(image_id)

    def get_video(self, video_id: str) -> StoredMediaAsset | None:
        return self._videos.get(video_id)

    def save_image_content(
        self, content: OutputImageContent
    ) -> StoredMediaAsset:
        return self.save_image_contents([content])[0]

    def save_image_contents(
        self, contents: list[OutputImageContent]
    ) -> list[StoredMediaAsset]:
        if not contents:
            return []

        payloads = [
            (
                _decode_output_image_bytes(content),
                (content.format or "png").lower(),
            )
            for content in contents
        ]
        self._ensure_capacity(
            sum(len(image_bytes) for image_bytes, _ in payloads)
        )

        saved_assets: list[StoredMediaAsset] = []
        try:
            for image_bytes, image_format in payloads:
                asset = self._write_asset(
                    directory=self._images_dir,
                    extension=image_format,
                    default_media_type=f"image/{image_format}",
                    payload=image_bytes,
                    kind="image",
                )
                self._register_asset(asset)
                saved_assets.append(asset)
        except Exception:
            for asset in saved_assets:
                self._delete_asset(asset)
            raise
        return saved_assets

    def save_video_frames(
        self,
        frame_contents: list[OutputImageContent],
        frames_per_second: int,
    ) -> StoredMediaAsset:
        if not frame_contents:
            raise ValueError("Cannot save a video without any frames.")

        output_path = self._videos_dir / f"{uuid4().hex}.mp4"
        try:
            _encode_mp4(
                [
                    _decode_output_image_frame(content)
                    for content in frame_contents
                ],
                output_path,
                frames_per_second,
            )
            size_bytes = output_path.stat().st_size
            self._ensure_capacity(size_bytes)
            asset = StoredMediaAsset(
                asset_id=output_path.stem,
                path=output_path,
                media_type="video/mp4",
                filename=output_path.name,
                kind="video",
                size_bytes=size_bytes,
            )
            self._register_asset(asset)
            return asset
        except Exception:
            output_path.unlink(missing_ok=True)
            raise

    def encode_video_frames(
        self,
        frame_contents: list[OutputImageContent],
        frames_per_second: int,
    ) -> bytes:
        if not frame_contents:
            raise ValueError("Cannot encode a video without any frames.")

        frames = [
            _decode_output_image_frame(content) for content in frame_contents
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            dir=self._videos_dir,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            _encode_mp4(frames, tmp_path, frames_per_second)
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    def _write_asset(
        self,
        directory: Path,
        extension: str,
        default_media_type: str,
        payload: bytes,
        kind: str,
    ) -> StoredMediaAsset:
        asset_id = uuid4().hex
        output_path = directory / f"{asset_id}.{extension}"
        output_path.write_bytes(payload)
        media_type = (
            mimetypes.guess_type(output_path.name)[0] or default_media_type
        )
        return StoredMediaAsset(
            asset_id=asset_id,
            path=output_path,
            media_type=media_type,
            filename=output_path.name,
            kind=kind,
            size_bytes=len(payload),
        )

    def _register_asset(self, asset: StoredMediaAsset) -> None:
        self._assets[asset.asset_id] = asset
        self._total_storage_bytes += asset.size_bytes
        if asset.kind == "image":
            self._images[asset.asset_id] = asset
        else:
            self._videos[asset.asset_id] = asset

    def _delete_asset(self, asset: StoredMediaAsset) -> None:
        if asset.asset_id not in self._assets:
            return

        self._assets.pop(asset.asset_id, None)
        if asset.kind == "image":
            self._images.pop(asset.asset_id, None)
        else:
            self._videos.pop(asset.asset_id, None)
        self._total_storage_bytes -= asset.size_bytes
        asset.path.unlink(missing_ok=True)

    def _ensure_capacity(self, additional_bytes: int) -> None:
        if additional_bytes > self._max_storage_bytes:
            raise GeneratedMediaStorageError(
                "Generated media exceeds local storage limit: "
                f"{additional_bytes} bytes requested, "
                f"{self._max_storage_bytes} bytes available."
            )

        while (
            self._total_storage_bytes + additional_bytes
            > self._max_storage_bytes
            and self._assets
        ):
            _, oldest_asset = next(iter(self._assets.items()))
            self._delete_asset(oldest_asset)


def _decode_output_image_bytes(content: OutputImageContent) -> bytes:
    if content.image_data is None:
        raise ValueError(
            "Only inline output_image payloads can be persisted to disk."
        )
    return base64.b64decode(content.image_data)


def encode_video_bytes_b64(video_bytes: bytes) -> str:
    return base64.b64encode(video_bytes).decode("utf-8")


def _decode_output_image_frame(content: OutputImageContent) -> np.ndarray:
    image_bytes = _decode_output_image_bytes(content)
    with Image.open(BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _encode_mp4(
    frames: list[np.ndarray], output_path: Path, frames_per_second: int
) -> None:
    import av
    import av.video

    height, width = frames[0].shape[:2]
    container = av.open(str(output_path), mode="w")
    stream: av.video.VideoStream = container.add_stream(  # type: ignore[assignment]
        "libx264",
        rate=frames_per_second,
    )
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.codec_context.options = {"crf": "18", "preset": "medium"}

    try:
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(
                frame_array.astype(np.uint8, copy=False),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
