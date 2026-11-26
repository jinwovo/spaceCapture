"""공간 윤곽 후처리 비교: 실제 SfM 분할(문으로 연결된 집)에서 관측 윤곽과 벽 맞춤 윤곽의 IoU를 잰다. 평가 전용.
복원 입력·정렬된 정답은 regularize-cache/에 캐시한다(retune-*-threshold.py도 사용)."""
import argparse
import copy
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from spacecapture.property_evaluation import evaluate_graph
from spacecapture.property_graph import infer_property

spec = importlib.util.spec_from_file_location("rsf", ROOT/"tools/real-sfm-features.py")
rsf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rsf)
CACHE = ROOT/"artifacts/space-learning/regularize-cache"


def load_cases(split, connected_only=True):
    rsf.configure("v2")
    data = json.loads((ROOT/"artifacts/space-learning/lh-v2-dataset.json").read_text(encoding="utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    cases = []
    for h in data["houses"]:
        if h["split"] != split or (connected_only and not h.get("connected")): continue
        path = CACHE/f"{h['id']}.pkl"
        if path.exists():
            cases.append(pickle.loads(path.read_bytes())); continue
        scene_dir = ROOT/"scenes"/f"space-v2-{h['id']}"
        if not (scene_dir/"scene.json").exists(): continue
        import pycolmap
        from spacecapture.align import arrays_from_reconstruction
        scene = json.loads((scene_dir/"scene.json").read_text(encoding="utf-8"))
        truth = json.loads((rsf.REAL/h["id"]/"property.gt.json").read_text(encoding="utf-8"))
        ref = {p["id"]: p for p in truth["photos"]}
        cams = [c for c in scene["cameras"] if c.get("source_photo") in ref]
        T = np.array([ref[c["source_photo"]]["position"] for c in cams]); M = np.array([c["position"] for c in cams])
        s, R, t, keep, err = rsf.robust_alignment(T, M)
        if keep.mean() < .8 or float(np.sqrt(np.mean(err[keep]**2))) >= .1: continue
        moved = copy.deepcopy(truth)
        for sp in moved["spaces"]:
            sp["polygon_xz"] = [(s*R@np.array([x, 0, z])+t)[[0, 2]].tolist() for x, z in sp["polygon_xz"]]
        for p in moved["portals"]:
            p["center_xz"] = (s*R@np.array([p["center_xz"][0], 0, p["center_xz"][1]])+t)[[0, 2]].tolist(); p["width_m"] *= s
        a = arrays_from_reconstruction(pycolmap.Reconstruction(scene_dir/"work/colmap/aligned")); k = a.reliable(min_angle=6.)
        case = {"id": h["id"], "args": (a.xyz[k], a.centers, list(a.names), [o for o, kk in zip(a.observations, k) if kk]),
                "kw": dict(planes=scene["layout"]["planes"], rotations=a.R_cw, intrinsics=a.intrinsics), "truth": moved}
        path.write_bytes(pickle.dumps(case))
        cases.append(case)
    return cases


def run_variant(cases, variant):
    """observed: 관측 윤곽 그대로, base: 설치 설정(벽 맞춤 직각 윤곽, spacecapture/space_regularize.py)."""
    rows = []
    for c in cases:
        g = infer_property(*c["args"], **c["kw"], regularize=variant != "observed")
        r = evaluate_graph(g, c["truth"])
        rows.append({"id": c["id"], "iou": float(r["mean_space_iou"]), "spaces": r["predicted_space_candidates"],
                     "truth": r["truth_spaces"], "matched": r["matched_spaces"], "portal_recall": r["portal_recall"]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--variants", nargs="*", default=["observed", "base"])
    args = parser.parse_args()
    t = time.perf_counter(); cases = load_cases(args.split); print(f"{len(cases)}채 준비 {time.perf_counter()-t:.0f}초", flush=True)
    report = {}
    for v in args.variants:
        rows = run_variant(cases, v)
        exact = [r for r in rows if r["spaces"] == r["truth"] == r["matched"]]
        report[v] = {"mean_iou": round(float(np.mean([r["iou"] for r in rows])), 4),
                     "count_exact": round(len(exact)/len(rows), 4),
                     "mean_iou_when_count_exact": round(float(np.mean([r["iou"] for r in exact])), 4) if exact else None,
                     "rows": rows}
        print(v, {k: val for k, val in report[v].items() if k != "rows"}, flush=True)
    out = ROOT/"artifacts/space-learning"/f"regularize-{args.split}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")



if __name__ == "__main__":
    main()
