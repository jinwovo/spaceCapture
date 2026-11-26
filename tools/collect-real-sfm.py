"""LH 평면도를 자체 렌더 사진으로 만들고 실제 photos 파이프라인으로 복원한다 (장시간·재개 가능·작업자 분할).

렌더 결과는 artifacts/space-learning/real/<id>/, 복원 장면은 scenes/space-train-<id>/ 에 둔다.
이미 끝난 집은 건너뛴다. 정답은 추론에 쓰지 않는다(평가·학습 표시 전용).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from spacecapture.plan_image import auto_meters_per_pixel, render_config
from spacecapture.synth_property import generate_property

LH = ROOT/"samples/external/lh-floorplans"
OUT = ROOT/"artifacts/space-learning/real"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--tag", default="", help="결과 이름 태그 (예: v2 → real-v2/, scenes/space-v2-*)")
    args = parser.parse_args()
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    manifest = json.loads((LH/"manifest.json").read_text(encoding="utf-8"))
    splits = set(args.splits.split(","))
    order = {"test": 0, "validation": 1, "train": 2}  # 독립 평가용 시험 집을 먼저 처리
    out = OUT.parent/(f"real-{args.tag}" if args.tag else "real"); prefix = f"space-{args.tag}-" if args.tag else "space-train-"
    # 문으로 모두 이어진 집(정답 기준)을 먼저, 그 안에서 시험→검증→학습 순서
    houses = sorted([h for h in data["houses"] if h["id"].startswith("lh-") and h["split"] in splits and h["spaces"] >= 2],
                    key=lambda h: (not h.get("connected", True), order[h["split"]], h["id"]))[args.worker::args.workers]
    log_path = out/f"collect-worker{args.worker}.jsonl"; out.mkdir(parents=True, exist_ok=True)
    for h in houses:
        folder = out/h["id"]; scene = ROOT/"scenes"/f"{prefix}{h['id']}"
        if (scene/"scene.json").exists() and (folder/"property.gt.json").exists():
            continue
        started = time.perf_counter(); index = int(h["id"][3:])
        try:
            if not (folder/"property.gt.json").exists():
                image = cv2.imdecode(np.fromfile(LH/manifest["plans"][index]["file"], np.uint8), cv2.IMREAD_COLOR)
                _, seg = auto_meters_per_pixel(image)
                config, _ = render_config(seg, seed=index, room_views=12)
                generate_property(folder, config=config, render_photos=True, log=lambda *_: None, seed=index,
                                  truth_metadata={"family": h["family"], "split": h["split"], "purpose": "실제 SfM 분포 학습·평가"})
            rendered = time.perf_counter()-started
            result = subprocess.run([sys.executable, "-m", "spacecapture", "photos", str(folder/"photos"), "--name", scene.name, "--no-splat", "--max-photos", "250"],
                                    cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            status = "ok" if (scene/"scene.json").exists() else "sfm_failed"
            record = {"id": h["id"], "split": h["split"], "status": status, "render_s": round(rendered, 1),
                      "total_s": round(time.perf_counter()-started, 1)}
            if status != "ok": record["error"] = (result.stdout+result.stderr)[-300:]
        except Exception as error:
            record = {"id": h["id"], "split": h["split"], "status": "error", "error": str(error)[:300]}
        with open(log_path, "a", encoding="utf-8") as f: f.write(json.dumps(record, ensure_ascii=False)+"\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
