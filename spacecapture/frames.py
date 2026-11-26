"""Pick sharp, evenly spaced frames out of a video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _sharpness(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    s = 320.0 / max(h, w)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_32F).var())


def extract_frames(
    video: Path,
    out_dir: Path,
    target_fps: float = 5.0,
    min_frames: int = 60,
    max_frames: int = 800,
    max_side: int = 1600,
    log=print,
) -> dict:
    """Split the video into equal time windows and keep the sharpest frame of each.

    Frames far blurrier than the video's median are dropped entirely, so a shaky
    moment produces a gap instead of a smeared image that would hurt SfM.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없어요: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scores = []
    size = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = frame.shape[1], frame.shape[0]
        scores.append(_sharpness(frame))
    cap.release()
    n = len(scores)
    if n == 0:
        raise RuntimeError("영상에서 프레임을 읽지 못했어요.")

    duration = n / fps
    k = int(np.clip(round(duration * target_fps), min_frames, max_frames))
    k = min(k, n)
    scores = np.asarray(scores)
    # Blur is judged against the surrounding +-1.5 s, not the whole video: a
    # close-up of a plain (white) wall has little texture everywhere and is not
    # blurry, while a shake blurs one moment relative to its neighbours. A global
    # floor dropped 113/486 plain-wall frames of a house walk-through and the
    # resulting gaps broke the trajectory (2026-11-26).
    half = max(1, int(round(1.5 * fps)))
    edges = np.linspace(0, n, k + 1).astype(int)
    chosen = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            j = int(a) + int(np.argmax(scores[a:b]))
            local = float(np.median(scores[max(0, j - half):j + half + 1]))
            if scores[j] >= 0.35 * local:
                chosen.append(j)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(video))
    names, idx, ci = [], 0, 0
    while ci < len(chosen):
        ok, frame = cap.read()
        if not ok:
            break
        if idx == chosen[ci]:
            h, w = frame.shape[:2]
            if max(h, w) > max_side:
                s = max_side / max(h, w)
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            name = f"frame_{ci:05d}.jpg"
            cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            names.append(name)
            ci += 1
        idx += 1
    cap.release()
    log(f"  영상 {n}프레임({duration:.1f}초, {size[0]}x{size[1]}) → 선명한 프레임 {len(names)}장 선택")
    return {"video_frames": n, "fps": fps, "duration": duration, "video_size": list(size),
            "selected": len(names), "names": names, "frame_index": chosen[: len(names)]}
