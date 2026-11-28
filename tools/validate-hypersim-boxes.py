"""Hypersim 사진 복원으로 가구 크기·방향 상자(object_boxes)를 정답 3D 상자와 비교한다.

추론은 사진만 쓴다(정답 포즈·상자 미사용). 평가 때만 우리 카메라 위치를 정답 카메라에 닮음 변환(umeyama)으로
맞춰 우리 상자를 정답 좌표(미터, z 위)로 옮긴다. 배율 오차(촬영 높이 1.45m 가정)는 이 정렬로 지워지므로
크기 오차는 '배율 보정 후' 값이다. 대응: 같은 종류(우리 COCO → 정답 NYU40) 중 바닥 중심 1m 이내 가장 가까운 것.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools"))
from hypersim_gt import MATCH, load
from spacecapture.pipeline import umeyama


CORE = {"bed", "sofa", "chair", "table", "desk", "television", "refrigerator", "toilet"}


def gt_box(o):
    axes = o["R"]  # 열 = 상자 축(정답 좌표)
    k = int(np.argmax(np.abs(axes[2, :])))          # z(위)에 가장 가까운 축 = 높이
    hor = [i for i in range(3) if i != k]
    i_long = max(hor, key=lambda i: o["extent"][i]); i_short = min(hor, key=lambda i: o["extent"][i])
    a = axes[:2, i_long]; yaw = np.degrees(np.arctan2(a[1], a[0])) % 180
    top = o["center"][2]+o["extent"][k]/2
    return {"xy": o["center"][:2], "w": o["extent"][i_long], "d": o["extent"][i_short], "h": o["extent"][k], "top": top, "yaw": yaw}


def refit(scene_name, scene):
    """현재 object_boxes 코드·설정으로 저장된 사진별 검출에서 상자를 다시 맞춘다(장면 파일은 바꾸지 않음)."""
    from spacecapture.object_boxes import cameras_from_reconstruction, class_labels, fit_boxes
    from spacecapture.plyio import read_vertices
    d = ROOT/"scenes"/scene_name
    v = read_vertices(d/"points.ply")
    fit_boxes(scene["objects"]["instances"], scene["objects"]["detections"], cameras_from_reconstruction(d/"work/colmap/aligned"),
              class_labels(), points=np.stack([v["x"], v["y"], v["z"]], 1))
    for o in scene["objects"]["instances"]:
        if o.get("box3d"): o["box3d"]["display"] = o["box3d"]["reliable"] and o["class"] not in ("refrigerator", "oven", "microwave", "sink")


def evaluate(scene_name, gt_dir, do_refit=False, scene=None):
    """scene을 주면(실험용 재검출 결과 등) 파일 대신 그것을 평가한다."""
    if scene is None: scene = json.loads((ROOT/"scenes"/scene_name/"scene.json").read_text(encoding="utf-8"))
    if do_refit: refit(scene_name, scene)
    frames, objects = load(gt_dir)
    cams = [c for c in scene["cameras"] if c.get("source_photo", "").rsplit(".", 1)[0] in frames]
    T = np.array([frames[c["source_photo"].rsplit(".", 1)[0]][0] for c in cams]); M = np.array([c["position"] for c in cams])
    s, R, t = umeyama(T, M)
    cam_err = np.linalg.norm(M-(s*T@R.T+t), axis=1)/s
    to_gt = lambda X: ((np.asarray(X, float)-t)@R)/s
    floor_gt = to_gt([0, 0, 0])[2]
    ours = []
    for o in (scene.get("objects") or {}).get("instances", []):
        b = o.get("box3d")
        if not b or o["class"] not in MATCH: continue
        yaw = np.radians(b["yaw_deg"])
        x_axis = to_gt(np.array(b["center"])+[np.cos(yaw), 0, -np.sin(yaw)])-to_gt(b["center"])
        c = to_gt(b["center"])
        ours.append({"class": o["class"], "xy": c[:2], "w": b["size"][0]/s, "d": b["size"][2]/s, "h": b["size"][1]/s,
                     "top": to_gt([b["center"][0], b["bottom"]+b["size"][1], b["center"][2]])[2],
                     "yaw": np.degrees(np.arctan2(x_axis[1], x_axis[0])) % 180, "reliable": b["reliable"], "display": b.get("display", b["reliable"]),
                     "views": b["views"], "rejected": b.get("rejected"), "source": o.get("source", "sfm_points"),
                     "edge_error": b.get("edge_error"), "inside_points": b.get("inside_points")})
    gts = [dict(gt_box(o), cls=o["class"], instance=o["instance"], pixels=o["pixels"]) for o in objects]
    rows, used = [], set()
    # 표시한 상자를 먼저 대응시킨다(표시 안 한 상자가 정답을 가로채지 않게)
    for q in sorted(ours, key=lambda q: (not q["display"], -q["views"])):
        cand = [(np.linalg.norm(q["xy"]-g["xy"]), k) for k, g in enumerate(gts) if g["cls"] in MATCH[q["class"]] and k not in used]
        cand = [c for c in cand if c[0] <= 1.0]
        row = {"class": q["class"], "reliable": q["reliable"], "display": q["display"], "rejected": q["rejected"], "views": q["views"],
               "source": q["source"], "edge_error": q["edge_error"], "inside_points": q["inside_points"],
               "nearest_gt": min(((round(float(np.linalg.norm(q["xy"]-g["xy"])), 2), g["cls"]) for g in gts), default=None)}
        if cand:
            e, k = min(cand); used.add(k); g = gts[k]
            dyaw = abs((q["yaw"]-g["yaw"]+90) % 180-90)
            row.update(matched=g["cls"], center_cm=round(e*100, 1), width_cm=round((q["w"]-g["w"])*100, 1), depth_cm=round((q["d"]-g["d"])*100, 1),
                       height_cm=round((q["h"]-g["h"])*100, 1), top_cm=round((q["top"]-g["top"])*100, 1),
                       yaw_err_deg=round(dyaw, 1) if g["w"] > 1.15*g["d"] else None, gt_size=[round(float(g["w"]), 2), round(float(g["d"]), 2), round(float(g["h"]), 2)],
                       ours_size=[round(float(q["w"]), 2), round(float(q["d"]), 2), round(float(q["h"]), 2)])
        rows.append(row)
    return {"scene": scene_name, "registered": scene["quality"]["registered"], "camera_rmse_cm": round(float(np.sqrt(np.mean(cam_err**2)))*100, 2),
            "scale_vs_truth": round(1/float(s), 4), "floor_offset_cm": round(float(floor_gt-min(g["top"]-g["h"] for g in gts))*100, 1) if gts else None,
            "gt_furniture": sum(1 for g in gts if any(g["cls"] in v for v in MATCH.values())),
            # 재현율 분모: 대응 종류이면서 분할 영상에서 충분히 보인(2만 화소 이상, 라벨 프레임 합) 정답 가구
            "gt_visible": sum(1 for g in gts if any(g["cls"] in v for v in MATCH.values()) and g["pixels"] >= 20000),
            # 핵심 가구만(기타 가구·수납장 제외): 침대·소파·의자·탁자·책상·TV·냉장고·변기
            "gt_visible_core": sum(1 for g in gts if g["cls"] in CORE and g["pixels"] >= 20000),
            "matched_core": sum(1 for x in rows if x.get("display") and x.get("matched") in CORE), "rows": rows}


def apply_settings(pairs):
    import spacecapture.object_boxes as ob
    for kv in pairs:
        k, v = kv.split("=")
        ob.SETTINGS[k] = v if k == "floor_mode" else bool(float(v)) if k == "floor_ray" else float(v)


def _job(task):
    name, refit_, pairs = task
    apply_settings(pairs)
    return evaluate(f"hs-{name}", ROOT/"samples/external/hypersim"/name, refit_)


def standard_size(row):
    """정답 상자가 해당 종류의 일반 크기 범위(±15%) 안인지(라운지 의자·머리판 포함 침대 등 정의 차이 제외)."""
    from spacecapture.object_boxes import size_range
    lo, hi = size_range(row["class"])
    return all(lo[i]*.85 <= s <= hi[i]*1.15 for i, s in enumerate(row["gt_size"]))


def summarize(out):
    shown = [x for r in out for x in r["rows"] if x["display"] and "matched" in x]
    std = [x for x in shown if standard_size(x)]
    allfit = [x for r in out for x in r["rows"] if "matched" in x]
    med = lambda rows, k: round(float(np.median([abs(x[k]) for x in rows if x.get(k) is not None])), 1) if any(x.get(k) is not None for x in rows) else None
    displayed = sum(1 for r in out for x in r["rows"] if x["display"]); visible = sum(r["gt_visible"] for r in out)
    return {"scenes": len(out), "boxes": sum(len(r["rows"]) for r in out), "displayed": displayed,
            "precision": round(len(shown)/max(1, displayed), 3), "recall_visible": round(len(shown)/max(1, visible), 3), "gt_visible": visible,
            "recall_core": round(sum(r["matched_core"] for r in out)/max(1, sum(r["gt_visible_core"] for r in out)), 3),
            "gt_visible_core": sum(r["gt_visible_core"] for r in out),
            "displayed_matched": len(shown), "displayed_unmatched": sum(1 for r in out for x in r["rows"] if x["display"] and "matched" not in x),
            **{f"displayed_median_abs_{k}": med(shown, k) for k in ("center_cm", "width_cm", "depth_cm", "height_cm", "top_cm", "yaw_err_deg")},
            "standard_size_matched": len(std),
            **{f"standard_median_abs_{k}": med(std, k) for k in ("center_cm", "width_cm", "depth_cm", "height_cm", "top_cm", "yaw_err_deg")},
            **{f"all_median_abs_{k}": med(allfit, k) for k in ("width_cm", "depth_cm", "height_cm", "yaw_err_deg")}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("scenes", nargs="+"); ap.add_argument("--refit", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="object_boxes.SETTINGS 키=값")
    ap.add_argument("--max-camera-rmse-cm", type=float, default=10.); ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=str(ROOT/"artifacts/hypersim/boxes.json")); ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    apply_settings(args.set)
    tasks = [(n, args.refit, args.set) for n in args.scenes]
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool: results = pool.map(_job, tasks)
    else:
        results = [_job(x) for x in tasks]
    out = []
    for name, r in zip(args.scenes, results):
        if r["camera_rmse_cm"] > args.max_camera_rmse_cm:  # 복원 자체가 틀린 장면은 상자 평가에서 뺀다
            print(name, "제외: 카메라 정렬 오차", r["camera_rmse_cm"], "cm"); continue
        out.append(r)
        if args.quiet: continue
        print(name, {k: v for k, v in r.items() if k != "rows"})
        for row in r["rows"]:
            if "matched" in row: print("   ", row)
    summary = summarize(out)
    print(json.dumps(summary, ensure_ascii=False))
    (ROOT/"artifacts/hypersim").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "scenes": out}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
