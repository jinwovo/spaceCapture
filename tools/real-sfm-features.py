"""저장된 실제 SfM 복원(scenes/space-train-*)에서 작은 방 후보 특징·정답 표시를 만들고,
기존 규칙과 모델의 공간·문 결과를 정답과 비교한다(강건 카메라 정렬, 평가 전용).
"""
import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pycolmap
from spacecapture.align import arrays_from_reconstruction
from spacecapture.pipeline import umeyama
from spacecapture.property_evaluation import evaluate_graph
from spacecapture.property_graph import infer_property

spec = importlib.util.spec_from_file_location("tsm", ROOT/"tools/train-space-model.py")
tsm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tsm)
REAL = ROOT/"artifacts/space-learning/real"
PREFIX = "space-train-"


def configure(tag=""):
    """결과 태그(v2 등)에 맞춰 렌더 폴더·장면 이름 접두어를 바꾼다."""
    global REAL, PREFIX
    REAL = ROOT/"artifacts/space-learning"/(f"real-{tag}" if tag else "real")
    PREFIX = f"space-{tag}-" if tag else "space-train-"


def robust_alignment(T, M):
    keep = np.ones(len(T), bool)
    for _ in range(10):
        s, R, t = umeyama(T[keep], M[keep])
        err = np.linalg.norm(M-(s*T@R.T+t), axis=1)/s
        new = err < max(.05, 3*np.median(err[keep]))
        if (new == keep).all(): break
        keep = new
    return s, R, t, keep, err


def truth_connected(truth):
    """정답 기준으로 사진 대상 공간이 모두 문으로 이어졌는지. 평면도 변환에서 문을 놓치면 방이 막힌 집이 된다."""
    ids = [s["id"] for s in truth["spaces"]]
    parent = {i: i for i in ids}
    def root(i):
        while parent[i] != i: i = parent[i]
        return i
    for p in truth["portals"]:
        a, b = p["spaces"]
        if a in parent and b in parent: parent[root(a)] = root(b)
    return len({root(i) for i in ids}) == 1


def house_rows(hid, split, models):
    folder = REAL/hid; scene_dir = ROOT/"scenes"/f"{PREFIX}{hid}"
    scene = json.loads((scene_dir/"scene.json").read_text(encoding="utf-8"))
    truth = json.loads((folder/"property.gt.json").read_text(encoding="utf-8"))
    ref = {p["id"]: p for p in truth["photos"]}
    cams = [c for c in scene["cameras"] if c.get("source_photo") in ref]
    T = np.array([ref[c["source_photo"]]["position"] for c in cams]); M = np.array([c["position"] for c in cams])
    s, R, t, keep, err = robust_alignment(T, M)
    usable = keep.mean() >= .8 and float(np.sqrt(np.mean(err[keep]**2))) < .1
    info = {"id": hid, "split": split, "registered": scene["quality"]["registered"], "photos": len(truth["photos"]),
            "inlier_cameras": round(float(keep.mean()), 3), "truth_connected": truth_connected(truth), "camera_rmse_cm": round(float(np.sqrt(np.mean(err[keep]**2)))*100, 2), "usable": bool(usable)}
    if not usable:
        return info, []
    moved = copy.deepcopy(truth)
    for space in moved["spaces"]:
        space["polygon_xz"] = [(s*R@np.array([x, 0, z])+t)[[0, 2]].tolist() for x, z in space["polygon_xz"]]
    for portal in moved["portals"]:
        portal["center_xz"] = (s*R@np.array([portal["center_xz"][0], 0, portal["center_xz"][1]])+t)[[0, 2]].tolist()
        portal["width_m"] *= s
    aligned = arrays_from_reconstruction(pycolmap.Reconstruction(scene_dir/"work/colmap/aligned"))
    k = aligned.reliable(min_angle=6.)
    args = (aligned.xyz[k], aligned.centers, aligned.names, [o for o, kk in zip(aligned.observations, k) if kk])
    kw = dict(planes=scene["layout"]["planes"], rotations=aligned.R_cw, intrinsics=aligned.intrinsics)
    rows = []
    for label, model in models.items():
        debug = {}
        small, merge = model if isinstance(model, tuple) else (model, False)
        graph = infer_property(*args, **kw, debug=debug, small_room_model=small, merge_model=merge)
        r = evaluate_graph(graph, moved)
        info[label] = {"iou": float(r["mean_space_iou"]), "spaces": r["predicted_space_candidates"], "truth": r["truth_spaces"],
                       "matched": r["matched_spaces"], "portal_recall": r["portal_recall"], "portal_precision": r["portal_precision"],
                       "center_cm": r["portal_center_mean_error_cm"]}
        if label == "rule":
            for cand, y in zip(debug.get("small_candidates", []), tsm.label_candidates(debug, moved)):
                rows.append((cand["features"], float(y)))
    return info, rows


def summarize(infos, label):
    rs = [i[label] for i in infos if i.get("usable") and label in i]
    if not rs: return None
    return {"houses": len(rs), "mean_iou": round(float(np.mean([r["iou"] for r in rs])), 4),
            "space_count_exact": round(float(np.mean([r["spaces"] == r["truth"] for r in rs])), 4),
            "extra_spaces": int(sum(max(0, r["spaces"]-r["truth"]) for r in rs)),
            "missing_spaces": int(sum(max(0, r["truth"]-r["matched"]) for r in rs)),
            "portal_recall": round(float(np.mean([r["portal_recall"] for r in rs])), 4),
            "portal_precision": round(float(np.mean([r["portal_precision"] for r in rs])), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", action="append", default=[], help="이름=경로 (여러 번)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    configure(args.tag)
    args.out = args.out or str(REAL/"real-features.npz")
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    models = {"rule": (False, False)}
    for item in args.model:
        name, path = item.split("=", 1)
        if path == "installed": models[name] = (None, None)      # 설치된 작은 방·합치기 모델
        elif path == "space-only": models[name] = (None, False)
        elif path == "merge-only": models[name] = (False, None)
        elif path.startswith("with-merge:"): models[name] = (None, Path(path.split(":", 1)[1]))  # 설치 작은 방 + 지정 합치기
        else: models[name] = (Path(path), False)
    started = time.perf_counter()
    infos, X, y, house, split = [], [], [], [], []
    for h in data["houses"]:
        if not h["id"].startswith("lh-"): continue
        if not (ROOT/"scenes"/f"{PREFIX}{h['id']}"/"scene.json").exists() or not (REAL/h["id"]/"property.gt.json").exists(): continue
        try:
            info, rows = house_rows(h["id"], h["split"], models)
        except Exception as error:
            info, rows = {"id": h["id"], "split": h["split"], "usable": False, "error": str(error)[:200]}, []
        infos.append(info)
        for f, label in rows:
            X.append(f); y.append(label); house.append(h["id"]); split.append(h["split"])
    np.savez_compressed(args.out, X=np.asarray(X, np.float32).reshape(-1, 12), y=np.asarray(y, np.float32),
                        house=np.asarray(house), split=np.asarray(split))
    report = {"houses": infos, "seconds": round(time.perf_counter()-started, 1),
              "summary": {sp: {m: summarize([i for i in infos if i["split"] == sp], m) for m in models} for sp in ("train", "validation", "test")},
              "summary_connected": {sp: {m: summarize([i for i in infos if i["split"] == sp and i.get("truth_connected")], m) for m in models}
                                    for sp in ("train", "validation", "test")},
              "candidates": {sp: {"count": int(sum(1 for s in split if s == sp)), "positives": int(sum(l for l, s in zip(y, split) if s == sp))}
                             for sp in ("train", "validation", "test")},
              "scope": "LH 실제 평면도 구조를 자체 렌더한 사진의 실제 SfM 복원. 정답은 강건 카메라 정렬로 옮겨 평가·표시에만 사용."}
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "houses"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
