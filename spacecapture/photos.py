"""Prepare an unordered collection of local listing photos for reconstruction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def taken_at(path: Path) -> str | None:
    """EXIF 촬영 시각(DateTimeOriginal + 1/100초)을 'YYYY-MM-DDTHH:MM:SS.ss' 문자열로. 없으면 None.
    촬영 경로 재생 순서에만 쓴다(위치 추정에는 쓰지 않음)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769)
            raw = sub.get(36867) or exif.get(306)
            frac = str(sub.get(37521) or "").strip()
    except Exception:  # noqa: BLE001 - 시각이 없거나 읽을 수 없는 사진
        return None
    if not isinstance(raw, str) or len(raw) < 19 or raw.startswith("0000"):
        return None
    stamp = raw[:10].replace(":", "-")+"T"+raw[11:19]
    return stamp+("."+frac[:2].ljust(2, "0") if frac.isdigit() else "")


def write_previews(scene_dir: Path, image_dir: Path, cameras: list[dict]) -> None:
    if not image_dir.is_dir():
        return
    for i,camera in enumerate(cameras):
        source=image_dir/camera["name"]
        if not source.is_file(): continue
        image=cv2.imdecode(np.fromfile(source,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None: continue
        h,w=image.shape[:2]; scale=min(1.,240/max(h,w))
        image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,80])
        if ok:
            name=f"thumb_{i:04d}.jpg"; encoded.tofile(scene_dir/name); camera["preview"]=name


def prepare_photos(folder: Path, out_dir: Path, max_photos: int = 150,
                   max_side: int = 1600, log=print) -> dict:
    folder, out_dir = Path(folder).resolve(), Path(out_dir).resolve()
    if not folder.is_dir():
        raise RuntimeError(f"사진 폴더를 찾을 수 없어요: {folder}")
    if out_dir == folder:
        raise RuntimeError("원본 사진 폴더와 처리 폴더는 달라야 해요.")
    if max_photos < 3 or max_side < 64:
        raise ValueError("사진 수 제한은 3 이상, 최대 영상 크기는 64px 이상이어야 해요.")
    files = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.name.casefold())
    if len(files) > max_photos:
        raise RuntimeError(f"사진이 {len(files)}장이에요. 같은 층이나 연결되는 공간의 사진을 {max_photos}장 이하로 나눠 주세요.")
    if len(files) < 3:
        raise RuntimeError("서로 겹치는 부분을 다른 위치에서 찍은 사진이 최소 3장 필요해요.")
    # Validate and deduplicate before replacing any previous prepared images.
    selected, seen, skipped = [], set(), []
    for path in files:
        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, cv2.error):
            image = None
        if image is None or min(image.shape[:2]) < 64:
            skipped.append({"file": path.name, "reason": "읽을 수 없거나 너무 작은 사진"})
            continue
        h, w = image.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        fingerprint = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if fingerprint in seen:
            skipped.append({"file": path.name, "reason": "중복 사진"})
            continue
        seen.add(fingerprint)
        selected.append((path.name, [w, h], image, taken_at(path)))
    if len(selected) < 3:
        raise RuntimeError("중복·손상 사진을 제외하면 3장 미만이에요. 서로 다른 위치에서 찍은 사진을 추가해 주세요.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    manifest = []
    for i, (original, original_size, image, when) in enumerate(selected):
        name = f"photo_{i:05d}.jpg"
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"사진 변환에 실패했어요: {original}")
        encoded.tofile(out_dir / name)  # Unicode paths work on Windows too.
        manifest.append({"name": name, "original": original, "original_size": original_size,
                         "size": [image.shape[1], image.shape[0]], **({"taken_at": when} if when else {})})
    info = {"selected": len(manifest), "names": [p["name"] for p in manifest],
            "photos": manifest, "skipped": skipped, "input_count": len(files)}
    (out_dir.parent / "photos.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  사진 {len(files)}장 → {len(manifest)}장 준비, 중복·손상 제외 {len(skipped)}장 (촬영 순서 무관)")
    return info
