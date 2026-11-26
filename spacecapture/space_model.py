"""작은 방(욕실·다용도실 등) 후보 판별: 후보 생성·기하 특징·학습 모델(NumPy MLP, CPU).

공간 분할의 1.1m 중심 영역에 잡히지 않는 작은 방 후보를 여러 폭에서 만들고, 각 후보를
새 공간 중심으로 쓸지 판단한다. 모델이 없거나 검사에 실패하면 기존 규칙(0.75m 폭의 독립
영역, 안에 카메라 2대 이상)을 그대로 쓴다. 이 모듈에는 정답 도면을 전달하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

FEATURES = ["width_m", "log_area", "cameras_inside", "cameras_near", "standalone", "log_component_ratio",
            "distance_to_core_m", "clearance_m", "enclosure", "camera_fraction", "elongation", "rule_accept"]
VERSION = 1
ARCHITECTURE = [len(FEATURES), 24, 12, 1]
WIDTHS = (.75, .6, .9)  # 기존 규칙의 0.75m 후보를 먼저 만들어 대체 경로 결과를 보존
MODEL = Path(__file__).resolve().parents[1]/"models/space-candidate-mlp-v1.npz"


def _odd(value):
    return max(3, int(round(value)) | 1)


def small_room_candidates(free, cores, camera_cells, step, widths=WIDTHS):
    """기존 중심 영역 밖의 작은 방 후보와 특징. free/cores는 0/1 격자, camera_cells는 (x,y) 셀."""
    height, width = free.shape
    occupied = np.zeros_like(free)
    for core in cores:
        occupied |= core
    to_core = cv2.distanceTransform((1-occupied).astype(np.uint8), cv2.DIST_L2, 5)*step if occupied.any() else np.full(free.shape, 99., np.float32)
    clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5)*step
    cams = np.array([[np.clip(x, 0, width-1), np.clip(y, 0, height-1)] for x, y in camera_cells], int).reshape(-1, 2)
    taken = np.zeros_like(free)
    candidates = []
    for w in widths:
        d = _odd(w/step)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        eroded = cv2.erode(free, kernel)
        grown = cv2.dilate(occupied, kernel)
        count, parts, stats, _ = cv2.connectedComponentsWithStats(eroded)
        sliver = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(.3/step),)*2)
        for label in range(1, count):
            component = (parts == label).astype(np.uint8)
            standalone = not np.any(component & grown)
            if standalone:
                pieces = [component]
            else:
                rest = cv2.morphologyEx(component & (1-grown), cv2.MORPH_OPEN, sliver)
                n, sub = cv2.connectedComponents(rest)
                pieces = [(sub == k).astype(np.uint8) for k in range(1, n)]
            for piece in pieces:
                area = float(piece.sum()*step**2)
                if area < .25 or np.any(piece & taken):
                    continue
                proximity = cv2.distanceTransform(1-piece, cv2.DIST_L2, 5)*step
                near = proximity[cams[:, 1], cams[:, 0]] if len(cams) else np.array([])
                inside, close = int(np.sum(near < .4)), int(np.sum(near < 1.))
                region = cv2.dilate(piece, kernel)
                ring = (cv2.dilate(piece, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd((w+.6)/step),)*2)) > 0) & (region == 0)
                ys, xs = np.nonzero(piece)
                extent = np.array([np.ptp(xs)+1, np.ptp(ys)+1], float)
                rule = bool(standalone and w == .75 and inside >= 2)
                features = [w, np.log(area), inside, close, float(standalone),
                            float(np.log(component.sum()/max(1, piece.sum()))), float(to_core[piece > 0].min()),
                            float(clearance[region > 0].max()), float(np.mean(free[ring] == 0)) if ring.any() else 0.,
                            inside/max(1, len(cams)), float(extent.max()/extent.min()), float(rule)]
                candidates.append({"mask": piece, "features": features, "width": w, "standalone": standalone,
                                   "rule_accept": rule})
                taken |= cv2.dilate(piece, sliver)
    return candidates


def predict(features, weights):
    x = np.clip((np.asarray(features, np.float32)-weights["mean"])/weights["std"], -8, 8)
    a = np.maximum(0, x@weights["w1"]+weights["b1"])
    b = np.maximum(0, a@weights["w2"]+weights["b2"])
    return 1/(1+np.exp(-np.clip((b@weights["w3"]+weights["b3"]).reshape(-1), -35, 35)))


def fingerprint(weights_sha256, threshold):
    payload = {"format": "space-candidate-v1", "weights_sha256": weights_sha256, "threshold": float(threshold),
               "feature_version": VERSION, "features": FEATURES, "architecture": ARCHITECTURE, "widths": list(WIDTHS)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_model(path=MODEL):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("features") != FEATURES or meta.get("feature_version") != VERSION:
        raise ValueError("공간 후보 모델의 특징 형식이 현재 코드와 달라요.")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != meta.get("sha256") or meta.get("fingerprint") != fingerprint(sha, meta.get("threshold", -1)):
        raise ValueError("공간 후보 모델의 해시·임계값 지문이 기록과 달라요.")
    with np.load(path, allow_pickle=False) as data:
        weights = {k: data[k].copy() for k in data.files}
    shapes = {"mean": (len(FEATURES),), "std": (len(FEATURES),), "w1": (len(FEATURES), 24), "b1": (24,),
              "w2": (24, 12), "b2": (12,), "w3": (12, 1), "b3": (1,)}
    if set(weights) != set(shapes) or any(weights[k].shape != v for k, v in shapes.items()) or \
            any(not np.isfinite(v).all() for v in weights.values()):
        raise ValueError("공간 후보 모델의 가중치가 올바르지 않아요.")
    return weights, meta


def decide(candidates, model=None):
    """후보별 채택 여부. model=None이면 설치 모델을 시도하고 실패하면 기존 규칙을 쓴다."""
    if not candidates:
        return [], {"method": "none"}
    path = MODEL if model is None else model
    if path is not False and Path(path).exists():
        try:
            weights, meta = load_model(path)
            scores = predict([c["features"] for c in candidates], weights)
            return [bool(s >= meta["threshold"]) for s in scores], {"method": "space_mlp", "model_id": meta["id"],
                    "scores": [round(float(s), 4) for s in scores], "threshold": meta["threshold"]}
        except (OSError, ValueError, KeyError) as error:
            return [c["rule_accept"] for c in candidates], {"method": "rule", "model_error": str(error)}
    return [c["rule_accept"] for c in candidates], {"method": "rule"}


# ---- 문 후보 → 같은 공간 안의 트인 통로인지(합치기) 판별 ----
PORTAL_FEATURES = ["width_m", "clearance_width_m", "log_landmarks", "image_supported", "log_area_min", "log_area_max",
                   "width_to_room_min", "width_to_room_max", "openings_same_pair", "log_photos_min"]
PORTAL_ARCHITECTURE = [len(PORTAL_FEATURES), 16, 8, 1]
MERGE_MODEL = Path(__file__).resolve().parents[1]/"models/portal-merge-mlp-v1.npz"


def portal_features(portals, spaces, labels, distance, step):
    """문 후보마다 특징. distance는 자유 공간 거리변환(칸), labels는 공간 번호 격자."""
    half_width = {}
    for i, space in enumerate(spaces, 1):
        mask = labels == i
        half_width[space["id"]] = float(distance[mask].max()*step) if mask.any() else .1
    pair_count = {}
    for p in portals:
        key = tuple(sorted(p["spaces"])); pair_count[key] = pair_count.get(key, 0)+1
    area = {s["id"]: max(s["observed_area_m2"], .05) for s in spaces}
    photos = {s["id"]: len(s["photos"]) for s in spaces}
    rows = []
    for p in portals:
        a, b = p["spaces"]
        rooms = sorted((2*half_width[a], 2*half_width[b]))
        rows.append([p["width_m"], p.get("clearance_width_m", p["width_m"]), float(np.log1p(p["shared_landmarks"])),
                     float(p["status"] == "image_supported"), float(np.log(min(area[a], area[b]))), float(np.log(max(area[a], area[b]))),
                     p["width_m"]/max(rooms[0], .1), p["width_m"]/max(rooms[1], .1), float(pair_count[tuple(sorted(p["spaces"]))]),
                     float(np.log1p(min(photos[a], photos[b])))])
    return rows


def _load_generic(path, features, architecture, fmt):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"format": fmt, "weights_sha256": sha, "threshold": float(meta.get("threshold", -1)),
               "features": features, "architecture": architecture}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if meta.get("features") != features or sha != meta.get("sha256") or meta.get("fingerprint") != expected:
        raise ValueError("문 합치기 모델의 해시·임계값·특징 지문이 기록과 달라요.")
    with np.load(path, allow_pickle=False) as data:
        weights = {k: data[k].copy() for k in data.files}
    if any(not np.isfinite(v).all() for v in weights.values()) or weights["w1"].shape != (len(features), architecture[1]):
        raise ValueError("문 합치기 모델의 가중치가 올바르지 않아요.")
    return weights, meta


def merge_fingerprint(weights_sha256, threshold):
    payload = {"format": "portal-merge-v1", "weights_sha256": weights_sha256, "threshold": float(threshold),
               "features": PORTAL_FEATURES, "architecture": PORTAL_ARCHITECTURE}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def decide_merges(features, model=None):
    """문 후보별 합치기 여부. 모델이 없거나 검사 실패 시 합치지 않는다(기존 동작)."""
    if not features:
        return [], {"method": "none"}
    path = MERGE_MODEL if model is None else model
    if path is False or not Path(path).exists():
        return [False]*len(features), {"method": "keep"}
    try:
        weights, meta = _load_generic(path, PORTAL_FEATURES, PORTAL_ARCHITECTURE, "portal-merge-v1")
        scores = predict(features, weights)
        return [bool(s >= meta["threshold"]) for s in scores], {"method": "merge_mlp", "scores": [round(float(s), 4) for s in scores],
                                                               "threshold": meta["threshold"]}
    except (OSError, ValueError, KeyError) as error:
        return [False]*len(features), {"method": "keep", "model_error": str(error)}
