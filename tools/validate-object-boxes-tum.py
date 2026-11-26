"""가구 크기·방향 상자(object_boxes)를 TUM RGB-D 실제 사무실 영상의 깊이로 검증한다.

추론은 RGB 사진만 쓴다(깊이·정답 궤적 미사용). 기준은 같은 사진의 깊이 영상을 정답 궤적으로 3D로 올리고,
카메라 정렬(umeyama, 평가 전용)로 장면 좌표에 옮긴 점이다. 사물마다 검출 상자 안의 깊이 점 중 맞춘 상자를
사방 0.3m 넓힌 영역(바닥 3cm 위)의 점으로 윗면 높이(95%)와 가로·깊이(5~95%)를 잰다. 가려진 면은 덜 잡히므로
기준 치수도 근사값이다.
"""
import importlib.util, json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("vot", ROOT/"tools/validate-objects-tum.py"); vot = importlib.util.module_from_spec(spec); spec.loader.exec_module(vot)
from spacecapture.pipeline import umeyama
from spacecapture.object_boxes import associate, cameras_from_reconstruction, class_labels

scene_dir = ROOT/"scenes/tum-long-office"
scene = json.loads((scene_dir/"scene.json").read_text(encoding="utf-8"))
frames, gt_t, gt, d_t, depth = vot.select_frames(150)
by_name = {f"{t:.6f}.png": (t, rel) for t, rel in frames}
cams_scene = [c for c in scene["cameras"] if c.get("source_photo") in by_name]
pose = {}
for t, rel in frames:
    g = vot.nearest(gt_t, gt, t); tx, ty, tz, qx, qy, qz, qw = map(float, g[1]); pose[f"{t:.6f}.png"] = (vot.quat_to_matrix(qx, qy, qz, qw), np.array([tx, ty, tz]))
T = np.array([pose[c["source_photo"]][1] for c in cams_scene]); M = np.array([c["position"] for c in cams_scene])
s, R, t = umeyama(T, M)
cams = cameras_from_reconstruction(scene_dir/"work/colmap/aligned")
photo_src = {c["name"]: c["source_photo"] for c in scene["cameras"]}
labels = class_labels()
rows = []
for o in scene["objects"]["instances"]:
    b = o.get("box3d")
    if not b: continue
    yaw = np.radians(b["yaw_deg"]); c_, s_ = np.cos(yaw), np.sin(yaw)
    w, h, d = b["size"]
    pts = []
    for box, cam, use in associate(o, scene["objects"]["detections"], cams, labels):
        name = next(n for n, cc in cams.items() if cc is cam)
        src = photo_src.get(name)
        if src not in by_name: continue
        tt, rel = by_name[src]; dd = vot.nearest(d_t, depth, tt)
        dimg = cv2.imread(str(vot.SEQ/dd[1][0]), cv2.IMREAD_UNCHANGED).astype(np.float32)/vot.DEPTH_SCALE
        sx, sy = 640/cam[2][0], 480/cam[2][1]   # 정규화 사진 크기 → 원본 640x480
        x0, y0, x1, y1 = box*[sx, sy, sx, sy]
        ys, xs = np.nonzero(dimg > 0); keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1); xs, ys = xs[keep], ys[keep]
        z = dimg[ys, xs]; Rg, Cg = pose[src]
        world = np.stack([(xs-vot.CX)/vot.FX*z, (ys-vot.CY)/vot.FY*z, z], 1)@Rg.T+Cg
        pts.append(s*world@R.T+t)
    if not pts: continue
    P = np.vstack(pts); Q = P-np.array(b["center"])
    lx, lz = Q[:, 0]*c_-Q[:, 2]*s_, Q[:, 0]*s_+Q[:, 2]*c_
    near = (np.abs(lx) < w/2+.3) & (np.abs(lz) < d/2+.3) & (P[:, 1] > .03) & (P[:, 1] < b["bottom"]+h+.5)
    if near.sum() < 200: continue
    ref_top = float(np.percentile(P[near, 1], 95)); ref_w = float(np.ptp(np.percentile(lx[near], [5, 95]))); ref_d = float(np.ptp(np.percentile(lz[near], [5, 95])))
    rows.append({"class": o["class"], "reliable": b["reliable"], "views": b["views"], "fit_top_m": round(b["bottom"]+h, 3), "ref_top_m": round(ref_top, 3),
                 "fit_width_m": w, "ref_width_m": round(ref_w, 3), "fit_depth_m": d, "ref_depth_m": round(ref_d, 3), "depth_points": int(near.sum())})
ok = [r for r in rows if r["reliable"]]
err = lambda k: [abs(r[f"fit_{k}_m"]-r[f"ref_{k}_m"])*100 for r in ok]
summary = {"objects_compared": len(rows), "reliable_compared": len(ok),
           "top_height_median_abs_cm": round(float(np.median(err("top"))), 1), "width_median_abs_cm": round(float(np.median(err("width"))), 1),
           "depth_median_abs_cm": round(float(np.median(err("depth"))), 1), "scale_vs_truth": round(float(s), 4)}
out = ROOT/"artifacts/object-validation/tum-boxes.json"
out.write_text(json.dumps({"summary": summary, "objects": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
for r in rows: print(" ", r)
