"""공식 ProcTHOR의 지정 분할만 내려받고 Git LFS SHA-256을 검사한다."""
from __future__ import annotations

import gzip
import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "samples" / "external" / "procthor-10k"
REPO = "allenai/procthor-10k"


def receive(url, limit):
    req = urllib.request.Request(url, headers={"User-Agent": "spaceCapture-dataset-validation/1"})
    with urllib.request.urlopen(req, timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("다운로드 크기가 허용 범위를 넘었어요.")
    return data


def main():
    parser=argparse.ArgumentParser(description="공식 집 구조의 분할별 다운로드·해시 검사")
    parser.add_argument("--split",choices=["train","val","test"],default="val")
    args=parser.parse_args()
    split=args.split
    started = time.perf_counter()
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / ("download.json" if split=="val" else f"download-{split}.json")
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = DEST / f"{split}.jsonl.gz"
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == old["sha256"]:
            print(f"이미 검증된 공식 자료를 사용합니다: {target}")
            return
    pinned=DEST/"download.json"
    commit = (json.loads(pinned.read_text(encoding="utf-8"))["commit"] if pinned.exists() else
              json.loads(receive(f"https://api.github.com/repos/{REPO}/commits/main", 200_000))["sha"])
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("공식 저장소 버전을 확인할 수 없어요.")
    base = f"https://raw.githubusercontent.com/{REPO}/{commit}"
    license_data = receive(f"{base}/LICENSE", 50_000)
    if b"Apache License" not in license_data or b"Version 2.0" not in license_data:
        raise ValueError("허용한 Apache-2.0 라이선스를 확인할 수 없어요.")
    pointer = receive(f"{base}/{split}.jsonl.gz", 4096).decode("ascii")
    expected = re.search(r"oid sha256:([0-9a-f]{64})", pointer)
    size = re.search(r"^size (\d+)$", pointer, re.M)
    if not expected or not size or not 0 < int(size[1]) < 100_000_000:
        raise ValueError("자료 분할의 Git LFS 크기·해시를 확인할 수 없어요.")
    url = f"https://media.githubusercontent.com/media/{REPO}/{commit}/{split}.jsonl.gz"
    data = receive(url, int(size[1]))
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != int(size[1]) or digest != expected[1]:
        raise ValueError("공식 Git LFS의 파일 크기 또는 SHA-256과 일치하지 않아요.")
    # Count JSON records with a separate decompression bound; no downloaded code runs.
    import io
    count=decoded_size=0
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        while line:=stream.readline(5_000_001):
            decoded_size+=len(line)
            if len(line)>5_000_000 or decoded_size>750_000_000:
                raise ValueError("압축 해제 크기가 허용 범위를 넘었어요.")
            if line.strip():
                json.loads(line)
                count+=1
    if count != (10000 if split=="train" else 1000):
        raise ValueError("공식 분할의 집 수가 예상과 달라요.")
    (DEST / f"{split}.jsonl.gz").write_bytes(data)
    (DEST / "LICENSE").write_bytes(license_data)
    (DEST / "README.upstream.md").write_bytes(receive(f"{base}/README.md", 50_000))
    manifest = {"dataset": "ProcTHOR-10K", "upstream_split": split, "commit": commit,
                "url": url, "license": "Apache-2.0", "sha256": digest, "bytes": len(data),
                "houses": count, "seconds": round(time.perf_counter() - started, 3),
                "downloaded_images": 0, "downloaded_assets": 0, "neural_training_performed": False,
                "note": "공식 집 JSON만 받았습니다. 실제 주택 사진 또는 시뮬레이터 에셋이 아닙니다."}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
