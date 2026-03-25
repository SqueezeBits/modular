#!/usr/bin/env python3
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

"""DWPose-based preprocessing for Wan-Animate driving videos.

Extracts skeleton pose renders and cropped face regions from raw driving
video frames using YOLOX person detection and DWPose whole-body keypoint
estimation (both via ONNX Runtime).

This module is used by both ``wan_animate_move_diffusers.py`` (diffusers
reference runner) and ``simple_offline_video_generation.py`` (MAX pipeline
runner).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def save_preprocessed(
    pose_frames: list[Image.Image],
    face_frames: list[Image.Image],
    output_dir: str | Path,
    fps: int = 30,
) -> tuple[Path, Path]:
    """Save preprocessed pose and face frames as mp4 files.

    :param pose_frames: Skeleton renders from ``preprocess_driving_video``.
    :param face_frames: Face crops from ``preprocess_driving_video``.
    :param output_dir: Directory to write ``pose.mp4`` and ``face.mp4``.
    :param fps: Frame rate for the output videos.
    :returns: ``(pose_path, face_path)`` — paths to the written files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pose_path = out / "pose.mp4"
    face_path = out / "face.mp4"
    _save_pil_frames_as_mp4(pose_frames, str(pose_path), fps)
    _save_pil_frames_as_mp4(face_frames, str(face_path), fps)
    print(f"  Saved {len(pose_frames)} pose frames → {pose_path}")
    print(f"  Saved {len(face_frames)} face frames → {face_path}")
    return pose_path, face_path


def load_preprocessed(
    input_dir: str | Path,
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Load previously saved pose and face videos from a directory.

    :param input_dir: Directory containing ``pose.mp4`` and ``face.mp4``.
    :returns: ``(pose_frames, face_frames)`` — lists of PIL images.
    """
    d = Path(input_dir)
    pose_path = d / "pose.mp4"
    face_path = d / "face.mp4"
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose video not found: {pose_path}")
    if not face_path.exists():
        raise FileNotFoundError(f"Face video not found: {face_path}")
    pose_frames = _load_video_frames_ffmpeg(str(pose_path))
    face_frames = _load_video_frames_ffmpeg(str(face_path))
    print(
        f"  Loaded {len(pose_frames)} pose frames, "
        f"{len(face_frames)} face frames from {d}"
    )
    return pose_frames, face_frames


def _save_pil_frames_as_mp4(
    frames: list[Image.Image], output_path: str, fps: int
) -> None:
    """Encode a list of PIL images to an mp4 file via ffmpeg."""
    if not frames:
        return
    w, h = frames[0].size
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "18",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = b"".join(
        np.array(f, dtype=np.uint8).tobytes() for f in frames
    )
    _, stderr = proc.communicate(input=raw)
    if proc.returncode != 0:
        print(f"WARNING: ffmpeg returned {proc.returncode}: {stderr.decode()}")


def _load_video_frames_ffmpeg(video_path: str) -> list[Image.Image]:
    """Load video frames as a list of PIL images using ffmpeg."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path,
    ]
    info = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    w, h = (int(x) for x in info.split(","))

    cmd = [
        "ffmpeg", "-i", video_path, "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-v", "error", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    raw = proc.stdout
    frame_size = w * h * 3
    num_frames = len(raw) // frame_size
    frames: list[Image.Image] = []
    for i in range(num_frames):
        arr = np.frombuffer(
            raw, dtype=np.uint8, count=frame_size, offset=i * frame_size
        ).reshape(h, w, 3)
        frames.append(Image.fromarray(arr))
    return frames


def _letterbox_resize(image: np.ndarray, target_size: int = 512) -> Image.Image:
    """Crop and letterbox-resize a region to a square of ``target_size``."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return Image.fromarray(canvas)


# ---------------------------------------------------------------------------
# ONNX-based DWPose detector
# ---------------------------------------------------------------------------

# YOLOX preprocessing / postprocessing adapted from the DWPose reference code.


def _yolox_preprocess(
    img: np.ndarray, input_size: tuple[int, int] = (640, 640)
) -> tuple[np.ndarray, float]:
    """Letterbox-pad and normalise an image for YOLOX."""
    h, w = img.shape[:2]
    ratio = min(input_size[0] / h, input_size[1] / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    blob = padded.transpose(2, 0, 1).astype(np.float32)[np.newaxis]
    return blob, ratio


def _yolox_postprocess(
    output: np.ndarray,
    ratio: float,
    score_thr: float = 0.3,
    nms_thr: float = 0.45,
) -> np.ndarray:
    """Decode YOLOX output into bounding boxes (x1, y1, x2, y2, score)."""
    grids = []
    strides_list = []
    for stride in (8, 16, 32):
        hh = 640 // stride
        ww = 640 // stride
        xv, yv = np.meshgrid(np.arange(ww), np.arange(hh))
        grid = np.stack((xv.flatten(), yv.flatten()), axis=1)
        grids.append(grid)
        strides_list.append(np.full((grid.shape[0], 1), stride))
    grids = np.concatenate(grids, axis=0)
    strides_arr = np.concatenate(strides_list, axis=0)

    output = output[0]
    # Decode boxes.
    output[:, :2] = (output[:, :2] + grids) * strides_arr
    output[:, 2:4] = np.exp(output[:, 2:4]) * strides_arr

    # Score = objectness * class_score. COCO person class = 0.
    scores = output[:, 4] * output[:, 5]
    mask = scores > score_thr
    output = output[mask]
    scores = scores[mask]

    if len(scores) == 0:
        return np.empty((0, 5), dtype=np.float32)

    # Convert cx, cy, w, h → x1, y1, x2, y2.
    boxes = np.empty_like(output[:, :4])
    boxes[:, 0] = output[:, 0] - output[:, 2] / 2
    boxes[:, 1] = output[:, 1] - output[:, 3] / 2
    boxes[:, 2] = output[:, 0] + output[:, 2] / 2
    boxes[:, 3] = output[:, 1] + output[:, 3] / 2
    boxes /= ratio

    # NMS.
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
            boxes[order[1:], 3] - boxes[order[1:], 1]
        )
        iou = inter / (area_i + area_j - inter)
        inds = np.where(iou <= nms_thr)[0]
        order = order[inds + 1]

    boxes = boxes[keep]
    scores = scores[keep]
    return np.column_stack([boxes, scores])


def _dwpose_preprocess(
    img: np.ndarray, bbox: np.ndarray, input_size: tuple[int, int] = (288, 384)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop and affine-transform a person bbox for the DWPose network."""
    x1, y1, x2, y2 = bbox[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    # Expand bbox to match input aspect ratio.
    aspect = input_size[0] / input_size[1]  # w/h
    if bw / max(bh, 1e-6) > aspect:
        bh = bw / aspect
    else:
        bw = bh * aspect
    scale = np.array([bw * 1.25, bh * 1.25], dtype=np.float32)
    center = np.array([cx, cy], dtype=np.float32)

    # Affine transform.
    src = np.array(
        [
            [cx, cy - scale[1] * 0.5],
            [cx + scale[0] * 0.5, cy],
            [cx, cy],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [input_size[0] * 0.5, 0],
            [input_size[0], input_size[1] * 0.5],
            [input_size[0] * 0.5, input_size[1] * 0.5],
        ],
        dtype=np.float32,
    )
    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(img, M, input_size, flags=cv2.INTER_LINEAR)
    blob = warped.transpose(2, 0, 1).astype(np.float32) / 255.0
    # Normalize with ImageNet stats.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    blob = (blob - mean) / std
    return blob[np.newaxis], center, scale


def _decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    input_size: tuple[int, int] = (288, 384),
    simcc_split_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode SimCC outputs into keypoint coordinates and scores.

    The DWPose ONNX model outputs two 1-D distributions per keypoint
    (``simcc_x`` and ``simcc_y``) at ``simcc_split_ratio`` times the input
    resolution.
    """
    # simcc_x: (n_kps, W*ratio), simcc_y: (n_kps, H*ratio)
    x_idx = simcc_x.argmax(axis=1).astype(np.float32)
    y_idx = simcc_y.argmax(axis=1).astype(np.float32)
    x_score = np.take_along_axis(
        simcc_x, x_idx.astype(int)[:, None], axis=1
    ).squeeze(1)
    y_score = np.take_along_axis(
        simcc_y, y_idx.astype(int)[:, None], axis=1
    ).squeeze(1)
    scores = np.minimum(x_score, y_score)

    keypoints = np.stack(
        [x_idx / simcc_split_ratio, y_idx / simcc_split_ratio], axis=1
    )

    # Invert the affine to get original image coords.
    src = np.array(
        [
            [center[0], center[1] - scale[1] * 0.5],
            [center[0] + scale[0] * 0.5, center[1]],
            [center[0], center[1]],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [input_size[0] * 0.5, 0],
            [input_size[0], input_size[1] * 0.5],
            [input_size[0] * 0.5, input_size[1] * 0.5],
        ],
        dtype=np.float32,
    )
    M_inv = cv2.getAffineTransform(dst, src)
    ones = np.ones((len(keypoints), 1), dtype=np.float32)
    kps_hom = np.concatenate([keypoints, ones], axis=1)
    keypoints = (M_inv @ kps_hom.T).T
    return keypoints, scores


# Limb connections for COCO-WholeBody 17-keypoint body layout:
#   0=nose 1=L_eye 2=R_eye 3=L_ear 4=R_ear
#   5=L_shoulder 6=R_shoulder 7=L_elbow 8=R_elbow
#   9=L_wrist 10=R_wrist 11=L_hip 12=R_hip
#   13=L_knee 14=R_knee 15=L_ankle 16=R_ankle
_BODY_LIMBS = [
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 6),  # shoulders
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
    (11, 12),  # hips
    (5, 11),
    (6, 12),  # torso
    (0, 1),
    (0, 2),  # eyes
    (1, 3),
    (2, 4),  # ears
]
_LIMB_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
]


def _draw_skeleton(
    canvas: np.ndarray,
    body_kps: np.ndarray,
    body_scores: np.ndarray,
    hand_kps: np.ndarray | None = None,
    hand_scores: np.ndarray | None = None,
    score_thr: float = 0.3,
) -> np.ndarray:
    """Render coloured skeleton limbs onto *canvas*."""
    H, W = canvas.shape[:2]
    stickwidth = max(2, min(H, W) // 120)

    for idx, (i, j) in enumerate(_BODY_LIMBS):
        if (
            i >= len(body_kps)
            or j >= len(body_kps)
            or body_scores[i] < score_thr
            or body_scores[j] < score_thr
        ):
            continue
        x1, y1 = int(body_kps[i, 0]), int(body_kps[i, 1])
        x2, y2 = int(body_kps[j, 0]), int(body_kps[j, 1])
        color = _LIMB_COLORS[idx % len(_LIMB_COLORS)]
        cv2.line(canvas, (x1, y1), (x2, y2), color, stickwidth)

    # Draw hand keypoints as small circles.
    if hand_kps is not None and hand_scores is not None:
        for k in range(len(hand_kps)):
            if hand_scores[k] < score_thr:
                continue
            x, y = int(hand_kps[k, 0]), int(hand_kps[k, 1])
            cv2.circle(
                canvas, (x, y), max(1, stickwidth // 2), (255, 255, 255), -1
            )

    return canvas


def _load_onnx_sessions(
    device: str,
) -> tuple[Any, Any]:
    """Download (if needed) and load the YOLOX and DWPose ONNX models."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    det_path = hf_hub_download("yzd-v/DWPose", "yolox_l.onnx")
    pose_path = hf_hub_download("yzd-v/DWPose", "dw-ll_ucoco_384.onnx")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.startswith("cuda")
        else ["CPUExecutionProvider"]
    )
    det_session = ort.InferenceSession(det_path, providers=providers)
    pose_session = ort.InferenceSession(pose_path, providers=providers)
    return det_session, pose_session


def preprocess_driving_video(
    frames: list[Image.Image],
    device: str = "cpu",
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Extract pose skeleton renders and face crops from raw video frames.

    Uses DWPose via ONNX Runtime (YOLOX + DWPose wholebody model) for
    keypoint detection.  Pose frames are rendered as coloured stick-figure
    skeletons.  Face frames are cropped from the original image around the
    detected face keypoints and letterbox-resized to 512x512.

    :param frames: Raw driving video as a list of PIL RGB images.
    :param device: ``"cpu"`` or ``"cuda"`` -- controls ONNX Runtime provider.
    :returns: ``(pose_frames, face_frames)`` -- lists of PIL images the same
        length as *frames*.
    """
    det_session, pose_session = _load_onnx_sessions(device)

    pose_frames: list[Image.Image] = []
    face_frames: list[Image.Image] = []
    last_face_crop: Image.Image | None = None

    for idx, pil_frame in enumerate(frames):
        img_rgb = np.array(pil_frame, dtype=np.uint8)
        H_orig, W_orig = img_rgb.shape[:2]

        # --- Person detection via YOLOX ---
        blob, ratio = _yolox_preprocess(img_rgb)
        det_out = det_session.run(
            None, {det_session.get_inputs()[0].name: blob}
        )
        bboxes = _yolox_postprocess(det_out[0], ratio)

        canvas = np.zeros((H_orig, W_orig, 3), dtype=np.uint8)
        face_crop: Image.Image | None = None

        if len(bboxes) > 0:
            # Use the highest-scoring person.
            best = bboxes[bboxes[:, 4].argmax()]

            # --- DWPose keypoint estimation ---
            pose_blob, center, scale = _dwpose_preprocess(img_rgb, best)
            pose_out = pose_session.run(
                None, {pose_session.get_inputs()[0].name: pose_blob}
            )
            # DWPose outputs SimCC format: [simcc_x, simcc_y].
            simcc_x = pose_out[0][0]  # (n_kps, W*ratio)
            simcc_y = pose_out[1][0]  # (n_kps, H*ratio)
            keypoints, scores = _decode_simcc(simcc_x, simcc_y, center, scale)

            # COCO-WholeBody: body 0-16, face 23-90, hands 91-132.
            body_kps = keypoints[:17]
            body_scores = scores[:17]
            hand_kps = keypoints[91:133] if len(keypoints) > 91 else None
            hand_scores = scores[91:133] if len(scores) > 91 else None

            canvas = _draw_skeleton(
                canvas, body_kps, body_scores, hand_kps, hand_scores
            )

            # --- Face crop ---
            face_kps = keypoints[23:91]
            face_sc = scores[23:91]
            visible = face_sc > 0.3
            if visible.sum() >= 5:
                vis_kps = face_kps[visible]
                x_min, y_min = vis_kps.min(axis=0).astype(int)
                x_max, y_max = vis_kps.max(axis=0).astype(int)
                bw, bh = x_max - x_min, y_max - y_min
                x_min = max(0, x_min - int(bw * 0.3))
                x_max = min(W_orig, x_max + int(bw * 0.3))
                y_min = max(0, y_min - int(bh * 0.3))
                y_max = min(H_orig, y_max + int(bh * 0.6))
                if x_max > x_min and y_max > y_min:
                    crop = img_rgb[y_min:y_max, x_min:x_max]
                    face_crop = _letterbox_resize(crop, 512)

        pose_frames.append(Image.fromarray(canvas))

        if face_crop is not None:
            last_face_crop = face_crop
        elif last_face_crop is not None:
            face_crop = last_face_crop
        else:
            face_crop = Image.new("RGB", (512, 512))
        face_frames.append(face_crop)

        if (idx + 1) % 10 == 0 or idx == len(frames) - 1:
            print(f"  Preprocessed frame {idx + 1}/{len(frames)}", flush=True)

    return pose_frames, face_frames
