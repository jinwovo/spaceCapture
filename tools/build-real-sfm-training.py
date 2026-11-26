"""학습 분할 LH 평면도를 실제로 렌더링·SfM 복원해 작은 방 후보 특징과 정답 표시를 만든다.

모의 관측과 실제 SfM의 분포 차이를 줄이기 위한 학습 자료다. 정답 방은 카메라 정렬(학습 표시 전용)로
복원 좌표에 옮겨 후보 표시에만 쓰며, 추론 입력에는 들어가지 않는다. 시험 분할 단지는 쓰지 않는다.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pycolmap
from spacecapture.align import arrays_from_reconstruction
from spacecapture.pipeline import umeyama
from spacecapture.plan_image import auto_meters_per_pixel, render_config
from spacecapture.property_graph import infer_property
from spacecapture.synth_property import generate_property

spec = importlib.util.spec_from_file_location("tsm", ROOT/"tools/train-space-model.py")
tsm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tsm)
OUT = ROOT/"artifacts/space-learning/real"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="층화 분할이 기록된 실행 폴더의 dataset.json")
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    manifest = json.loads((tsm.LH/"manifest.json").read_text(encoding="utf-8"))
    chosen = [h for h in sorted(data["houses"], key=lambda h: h["id"])
              if h["id"].startswith("lh-") and h["split"] in ("train", "validation") and h["spaces"] >= 4][:args.count]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {"X": [], "y": [], "house": [], "split": []}; log = []
    for h in chosen:
        started = time.perf_counter(); index = int(h["id"][3:])
        folder = OUT/h["id"]; scene_name = f"space-train-{h['id']}"
        if not (folder/"property.gt.json").exists():
            image = cv2.imdecode(np.fromfile(tsm.LH/manifest["plans"][index]["file"], np.uint8), cv2.IMREAD_COLOR)
            _, seg = auto_meters_per_pixel(image)
            config, _ = render_config(seg, seed=index, room_views=12)
            generate_property(folder, config=config, render_photos=True, log=lambda *_: None, seed=index,
                              truth_metadata={"family": h["family"], "split": h["split"], "purpose": "학습 자료(실제 SfM 분포)"})
        rendered = time.perf_counter()-started
        if not (ROOT/"scenes"/scene_name/"scene.json").exists():
            subprocess.run([sys.executable, "-m", "spacecapture", "photos", str(folder/"photos"), "--name", scene_name, "--no-splat"],
                           cwd=ROOT, check=False, capture_output=True)
        scene_file = ROOT/"scenes"/scene_name/"scene.json"
        if not scene_file.exists():
            log.append({"id": h["id"], "status": "sfm_failed"}); continue
        scene = json.loads(scene_file.read_text(encoding="utf-8"))
        truth = json.loads((folder/"property.gt.json").read_text(encoding="utf-8"))
        ref = {p["id"]: p for p in truth["photos"]}
        cams = [c for c in scene["cameras"] if c.get("source_photo") in ref]
        T = np.array([ref[c["source_photo"]]["position"] for c in cams]); M = np.array([c["position"] for c in cams])
        keep_cams = np.ones(len(T), bool)
        for _ in range(10):  # 잘못 놓인 일부 사진은 빼고 정렬(학습 표시 전용)
            s, R, t = umeyama(T[keep_cams], M[keep_cams])
            err = np.linalg.norm(M-(s*T@R.T+t), axis=1)/s
            new = err < max(.05, 3*np.median(err[keep_cams]))
            if (new == keep_cams).all(): break
            keep_cams = new
        inlier_fraction = float(keep_cams.mean())
        rmse = float(np.sqrt(np.mean(err[keep_cams]**2))) if inlier_fraction >= .8 else 9.
        moved = json.loads(json.dumps(truth))
        for space in moved["spaces"]:
            space["polygon_xz"] = [(s*R@np.array([x, 0, z])+t)[[0, 2]].tolist() for x, z in space["polygon_xz"]]
        aligned = arrays_from_reconstruction(pycolmap.Reconstruction(ROOT/"scenes"/scene_name/"work/colmap/aligned"))
        keep = aligned.reliable(min_angle=6.)
        debug = {}
        infer_property(aligned.xyz[keep], aligned.centers, aligned.names, [o for o, k in zip(aligned.observations, keep) if k],
                       planes=scene["layout"]["planes"], rotations=aligned.R_cw, intrinsics=aligned.intrinsics, debug=debug, small_room_model=False)
        labels = tsm.label_candidates(debug, moved) if rmse < .1 else []
        for c, y in zip(debug.get("small_candidates", []), labels):
            rows["X"].append(c["features"]); rows["y"].append(float(y)); rows["house"].append(h["id"]); rows["split"].append(h["split"])
        log.append({"id": h["id"], "split": h["split"], "registered": scene["quality"]["registered"], "camera_rmse_cm": round(rmse*100, 2), "inlier_cameras": round(inlier_fraction, 3),
                    "candidates": len(labels), "positives": int(sum(labels)), "seconds": round(time.perf_counter()-started, 1),
                    "render_seconds": round(rendered, 1), "used": rmse < .1})
        print(json.dumps(log[-1], ensure_ascii=False), flush=True)
    np.savez_compressed(OUT/"real-features.npz", X=np.asarray(rows["X"], np.float32).reshape(-1, 12), y=np.asarray(rows["y"], np.float32),
                        house=np.asarray(rows["house"]), split=np.asarray(rows["split"]))
    (OUT/"real-features.json").write_text(json.dumps({"houses": log, "note": "실제 렌더 사진→SfM 복원 후보. 정답은 학습 표시에만 사용."},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
