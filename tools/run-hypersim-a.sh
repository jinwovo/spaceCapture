#!/bin/sh
# Hypersim 장면 사진(cam_00 tonemap)을 기존 photos 파이프라인에 넣는다. 정답 포즈·상자는 넘기지 않는다.
for s in "$@"; do
  src=samples/external/hypersim/$s/images/scene_cam_00_final_preview
  dst=artifacts/hypersim-a/run/$s/photos; mkdir -p $dst
  for f in $src/*.tonemap.jpg; do cp "$f" "$dst/$(basename $f .tonemap.jpg).jpg"; done
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m spacecapture photos $dst --name hsa-$s --no-splat > artifacts/hypersim-a/run/$s/process.log 2>&1
  echo "== $s $(tail -1 artifacts/hypersim-a/run/$s/process.log)"
done
