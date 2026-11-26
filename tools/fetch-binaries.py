"""저장소에 넣지 않는 큰 외부 파일을 공식 배포처에서 받아 해시를 검사한다.

- Brush v0.3.0 (Apache-2.0, 가우시안 스플랫 학습, Windows x64): tools/brush-v0.3.0/brush_app.exe
- YOLOX-s COCO ONNX (Apache-2.0, 실내 사물 검출): models/yolox_s.onnx

이미 있고 해시가 맞으면 건너뛴다. 해시가 다르면 저장하지 않는다.
사용: python tools/fetch-binaries.py [--root 다른_폴더]
"""
import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRUSH_ZIP = "https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-x86_64-pc-windows-msvc.zip"
BRUSH_EXE_SHA256 = "37e46cbf808b9983dd15a5f9a25328dbe43e7e06d53c4f59fbeaeb10e3a5b34a"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "spacecapture-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_yolox(root: Path) -> str:
    meta = json.loads((ROOT/"models/yolox_s.json").read_text(encoding="utf-8"))
    target = root/"models/yolox_s.onnx"
    if target.exists() and sha256(target.read_bytes()) == meta["sha256"]:
        return "이미 있음"
    data = download(meta["source"])
    if sha256(data) != meta["sha256"]:
        raise RuntimeError("YOLOX 파일 해시가 기록과 달라 저장하지 않았어요.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return f"받음 ({len(data):,}바이트)"


def fetch_brush(root: Path) -> str:
    target = root/"tools/brush-v0.3.0/brush_app.exe"
    if target.exists() and sha256(target.read_bytes()) == BRUSH_EXE_SHA256:
        return "이미 있음"
    archive = download(BRUSH_ZIP)
    published = download(BRUSH_ZIP+".sha256").decode().split()[0].lower()
    if sha256(archive) != published:
        raise RuntimeError("Brush 압축 파일 해시가 배포처 기록과 달라요.")
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names = [n for n in z.namelist() if n.lower().endswith("brush_app.exe")]
        if not names:
            raise RuntimeError("압축 파일 안에 brush_app.exe가 없어요.")
        exe = z.read(names[0])
        others = {Path(n).name: z.read(n) for n in z.namelist() if Path(n).name in ("LICENSE", "README.md", "CHANGELOG.md")}
    if sha256(exe) != BRUSH_EXE_SHA256:
        raise RuntimeError("brush_app.exe 해시가 검증된 값과 달라 저장하지 않았어요.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(exe)
    for name, data in others.items():
        (target.parent/name).write_bytes(data)
    return f"받음 ({len(exe):,}바이트)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT), help="저장할 프로젝트 폴더(기본: 이 저장소)")
    parser.add_argument("--skip-brush", action="store_true", help="Windows 외 환경 등에서 Brush를 받지 않음")
    args = parser.parse_args()
    root = Path(args.root)
    print("YOLOX-s:", fetch_yolox(root), flush=True)
    if not args.skip_brush:
        print("Brush v0.3.0:", fetch_brush(root), flush=True)


if __name__ == "__main__":
    sys.exit(main())
