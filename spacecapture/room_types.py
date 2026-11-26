"""공간 후보의 용도(거실·침실·욕실·기타) 추정: 복원된 공간 그래프의 기하·연결 특징 + 학습 MLP.

입력은 property_graph의 공간 다각형·개구부뿐이다(사진·정답 도면을 쓰지 않음). 사진 속 사물 근거
(objects.usage_hints: 변기·침대·소파·주방 가전)가 있으면 그것을 우선한다. 결과는 확정이 아니라 추정이다.
모델은 지문(가중치 SHA·임계값·특징·구조) 검사를 통과할 때만 쓰고, 실패하면 용도를 비워 둔다.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["living", "bedroom", "bathroom", "other"]
# 윤곽 처리·분할 개수에 민감한 꼭짓점 수·사각 채움률·공간 수는 뺐다(교차검증 0.824→0.818로 비슷, 2026-11-11).
FEATURES = ["log_area", "area_fraction", "area_rank", "is_largest", "aspect", "inner_width_m",
            "portal_count", "max_portal_width", "min_portal_width", "neighbor_count", "touches_largest",
            "center_offset", "exterior_fraction", "neighbor_max_log_area", "neighbor_min_log_area", "only_small_neighbors"]
VERSION = 3
ARCHITECTURE = [len(FEATURES), 32, 16, len(CLASSES)]
MODEL = Path(__file__).resolve().parents[1]/"models/room-type-mlp-v1.npz"
CELL = .05


def _area(poly):
    p = np.asarray(poly, float)
    return abs(.5*np.sum(p[:, 0]*np.roll(p[:, 1], -1)-np.roll(p[:, 0], -1)*p[:, 1]))


def space_features(graph):
    """공간마다 특징 벡터(FEATURES 순서). 다각형이 3점 미만이면 None."""
    spaces = graph.get("spaces", [])
    polys = [np.asarray(s.get("polygon_xz", []), float) for s in spaces]
    valid = [len(p) >= 3 for p in polys]
    if not any(valid):
        return [None]*len(spaces)
    areas = np.array([_area(p) if ok else 0. for p, ok in zip(polys, valid)])
    total = max(areas.sum(), 1e-6)
    order = np.argsort(-areas); rank = np.empty(len(spaces)); rank[order] = np.arange(len(spaces))
    largest = spaces[int(order[0])]["id"]
    allp = np.vstack([p for p, ok in zip(polys, valid) if ok])
    hull = cv2.convexHull(allp.astype(np.float32))
    center = np.average([p.mean(0) if ok else allp.mean(0) for p, ok in zip(polys, valid)], axis=0, weights=np.maximum(areas, 1e-6))
    radius = max(float(np.sqrt(total/np.pi)), .5)
    portals = {s["id"]: [] for s in spaces}; neighbors = {s["id"]: set() for s in spaces}
    for p in graph.get("portals", []):
        a, b = p["spaces"]
        for x, y in ((a, b), (b, a)):
            if x in portals:
                portals[x].append(float(p.get("width_m", 0.))); neighbors[x].add(y)
    rows = []
    for i, (space, poly, ok) in enumerate(zip(spaces, polys, valid)):
        if not ok:
            rows.append(None); continue
        low = poly.min(0)-CELL*2; size = np.ceil((poly.max(0)-low)/CELL).astype(int)+3
        mask = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(mask, [np.round((poly-low)/CELL).astype(np.int32)], 1)
        inner = float(cv2.distanceTransform(mask, cv2.DIST_L2, 5).max()*CELL*2)
        (_, _), (w, h), _ = cv2.minAreaRect(poly.astype(np.float32))
        dist = np.array([abs(cv2.pointPolygonTest(hull, (float(x), float(z)), True)) for x, z in poly])
        widths = portals[space["id"]]
        index = {s["id"]: k for k, s in enumerate(spaces)}
        nb = [areas[index[n]] for n in neighbors[space["id"]] if n in index]
        rows.append([float(np.log(max(areas[i], .1))), float(areas[i]/total), float(rank[i]/max(1, len(spaces)-1)),
                     float(space["id"] == largest), float(min(w, h)/max(w, h, 1e-6)), inner,
                     float(len(widths)), float(max(widths, default=0.)), float(min(widths, default=0.)),
                     float(len(neighbors[space["id"]])), float(largest in neighbors[space["id"]]),
                     float(np.linalg.norm(poly.mean(0)-center)/radius), float(np.mean(dist < .5)),
                     float(np.log(max(max(nb, default=.1), .1))), float(np.log(max(min(nb, default=.1), .1))),
                     float(bool(nb) and space["id"] != largest and largest not in neighbors[space["id"]])])
    return rows


def predict(features, weights):
    x = np.clip((np.asarray(features, np.float32)-weights["mean"])/weights["std"], -8, 8)
    a = np.maximum(0, x@weights["w1"]+weights["b1"])
    b = np.maximum(0, a@weights["w2"]+weights["b2"])
    z = b@weights["w3"]+weights["b3"]
    z = z-z.max(1, keepdims=True); e = np.exp(z)
    return e/e.sum(1, keepdims=True)


def fingerprint(weights_sha256, threshold):
    payload = {"format": "room-type-v1", "weights_sha256": weights_sha256, "threshold": float(threshold),
               "feature_version": VERSION, "features": FEATURES, "classes": CLASSES, "architecture": ARCHITECTURE}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_model(path=MODEL):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("features") != FEATURES or meta.get("classes") != CLASSES or meta.get("feature_version") != VERSION:
        raise ValueError("공간 용도 모델의 특징·분류 형식이 현재 코드와 달라요.")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != meta.get("sha256") or meta.get("fingerprint") != fingerprint(sha, meta.get("threshold", -1)):
        raise ValueError("공간 용도 모델의 해시·임계값 지문이 기록과 달라요.")
    with np.load(path, allow_pickle=False) as data:
        weights = {k: data[k].copy() for k in data.files}
    n, h1, h2, c = ARCHITECTURE
    shapes = {"mean": (n,), "std": (n,), "w1": (n, h1), "b1": (h1,), "w2": (h1, h2), "b2": (h2,), "w3": (h2, c), "b3": (c,)}
    if set(weights) != set(shapes) or any(weights[k].shape != v for k, v in shapes.items()) or \
            any(not np.isfinite(v).all() for v in weights.values()):
        raise ValueError("공간 용도 모델의 가중치가 올바르지 않아요.")
    return weights, meta


def estimate_usage(graph, model=None):
    """graph의 공간마다 usage_estimate를 붙인다. 사물 근거(usage_candidates)가 하나면 그것을 우선한다.
    model=None이면 설치 모델, False면 사용하지 않음."""
    spaces = graph.get("spaces", [])
    if not spaces:
        return {"method": "none"}
    path = MODEL if model is None else model
    if path is False or not Path(path).exists():
        return {"method": "unavailable"}
    try:
        weights, meta = load_model(path)
    except (OSError, ValueError, KeyError) as error:
        return {"method": "unavailable", "model_error": str(error)}
    rows = space_features(graph)
    idx = [i for i, r in enumerate(rows) if r is not None]
    probs = predict([rows[i] for i in idx], weights) if idx else np.zeros((0, len(CLASSES)))
    threshold = float(meta["threshold"])
    for k, i in enumerate(idx):
        space = spaces[i]
        p = {c: round(float(v), 3) for c, v in zip(CLASSES, probs[k])}
        best = max(p, key=p.get)
        objects = [u["usage"] for u in space.get("usage_candidates", [])]
        if len(set(objects)) == 1:
            usage, source = objects[0], "objects"   # 사물 근거: bathroom/bedroom/living/kitchen
        elif len(idx) >= 2 and p[best] >= threshold and best != "other":  # 공간이 하나뿐이면 집 안 상대 관계가 없어 추정하지 않는다
            usage, source = best, "geometry_model"
        else:
            usage, source = None, "undecided"
        space["usage_estimate"] = {"usage": usage, "source": source, "probabilities": p}
    return {"method": "room_type_mlp", "model_id": meta.get("id"), "threshold": threshold,
            "scope": "공간 크기·모양·연결로 추정한 용도 후보. 사진 속 사물 근거가 있으면 그것을 우선. 확정 아님."}
