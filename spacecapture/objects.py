"""사진 속 실내 사물 검출(YOLOX, Apache-2.0, CPU/OpenCV DNN)과 SfM 점을 이용한 3D 위치 추정.

검출기는 COCO 80종으로 학습된 공개 가중치다. 이 모듈은 공간 용도를 확정하지 않고
'사물 근거가 있는 용도 후보'만 기록한다. 외부 호스팅 API를 쓰지 않는다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parents[1]/"models/yolox_s.onnx"
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]
KOREAN = {"chair": "의자", "couch": "소파", "potted plant": "화분", "bed": "침대", "dining table": "식탁·책상",
          "toilet": "변기", "tv": "TV·모니터", "laptop": "노트북", "microwave": "전자레인지", "oven": "오븐·레인지",
          "toaster": "토스터", "sink": "세면대·싱크대", "refrigerator": "냉장고", "book": "책", "clock": "시계",
          "vase": "꽃병", "bench": "벤치", "keyboard": "키보드", "bottle": "병", "cup": "컵", "bowl": "그릇",
          "teddy bear": "인형", "hair drier": "드라이어", "remote": "리모컨", "mouse": "마우스",
          "cell phone": "휴대폰", "suitcase": "여행가방", "person": "사람"}
INDOOR = set(KOREAN) - {"person"}
# 크기에 따라 같은 사물로 합칠 거리(m)
MERGE_RADIUS = {"bed": 1.2, "couch": 1.0, "dining table": .9, "refrigerator": .7, "tv": .7, "oven": .6}
STRONG_VIEWS, STRONG_SCORE = 3, .6
# TUM 사무실 영상에서 칸막이 수납장→냉장고, 접시→싱크대, 모니터→TV, 책상→식탁 오검출을 확인했다.
# 그래서 단독으로 쓰는 근거는 변기·침대·소파만 두고, 주방은 서로 다른 가전 2종(점수 0.7 이상)을 요구한다.
SINGLE_RULES = [("toilet", "bathroom", "욕실 후보"), ("bed", "bedroom", "침실 후보"), ("couch", "living", "거실 후보")]
KITCHEN = {"oven", "refrigerator", "microwave", "sink"}


def model_metadata(path=MODEL):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("sha256"):
        raise ValueError("사물 검출 모델의 SHA-256이 기록과 달라요.")
    return meta


class Detector:
    def __init__(self, path=MODEL, input_size=640):
        self.meta = model_metadata(path)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.size = input_size
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = input_size//stride
            yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack([xx, yy], -1).reshape(-1, 2))
            strides.append(np.full((n*n, 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def detect(self, image, score_threshold=.35, nms_threshold=.45, classes=INDOOR):
        """BGR 이미지 → [{class, label, score, box:[x0,y0,x1,y1]}] (원본 픽셀 좌표)."""
        h, w = image.shape[:2]
        ratio = min(self.size/h, self.size/w)
        padded = np.full((self.size, self.size, 3), 114, np.uint8)
        resized = cv2.resize(image, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_LINEAR)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)   # YOLOX 0.1.1: BGR 0~255, 정규화 없음
        self.net.setInput(blob)
        out = self.net.forward()[0]
        xy = (out[:, :2]+self.grid)*self.stride
        wh = np.exp(np.clip(out[:, 2:4], -10, 10))*self.stride
        scores = out[:, 4:5]*out[:, 5:]
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        keep = best >= score_threshold
        boxes = np.concatenate([xy-wh/2, wh], 1)[keep]/ratio
        cls, best = cls[keep], best[keep]
        results = []
        for c in np.unique(cls):
            name = COCO[c]
            if classes is not None and name not in classes:
                continue
            idx = np.flatnonzero(cls == c)
            chosen = cv2.dnn.NMSBoxes(boxes[idx].tolist(), best[idx].tolist(), score_threshold, nms_threshold)
            for k in np.asarray(chosen).reshape(-1):
                x, y, bw, bh = boxes[idx[k]]
                results.append({"class": name, "label": KOREAN.get(name, name), "score": round(float(best[idx[k]]), 4),
                                "box": [round(float(max(0, x)), 1), round(float(max(0, y)), 1),
                                        round(float(min(w, x+bw)), 1), round(float(min(h, y+bh)), 1)]})
        return sorted(results, key=lambda r: -r["score"])


def locate_detection(box, point_xyz, point_uv, depth, shrink=.2, min_points=6):
    """한 검출 상자 안의 관측 3D 점 중 앞쪽 무리를 사물 위치로 쓴다 (벽 등 뒤쪽 점 제외)."""
    x0, y0, x1, y1 = box
    dx, dy = (x1-x0)*shrink/2, (y1-y0)*shrink/2
    inside = (point_uv[:, 0] > x0+dx) & (point_uv[:, 0] < x1-dx) & (point_uv[:, 1] > y0+dy) & (point_uv[:, 1] < y1-dy) & (depth > .1)
    if inside.sum() < min_points:
        return None
    d = depth[inside]
    front = d <= np.percentile(d, 25)*1.15
    if front.sum() < 3:
        return None
    P = point_xyz[inside][front]
    return {"position": np.median(P, 0), "points": int(front.sum()), "spread": float(np.median(np.linalg.norm(P-np.median(P, 0), axis=1)))}


def fuse_instances(observations, min_views=2):
    """여러 사진의 같은 종류·가까운 위치 검출을 한 사물로 합친다."""
    instances = []
    for obs in sorted(observations, key=lambda o: -o["score"]):
        radius = MERGE_RADIUS.get(obs["class"], .5)
        for inst in instances:
            if inst["class"] == obs["class"] and np.linalg.norm(inst["_sum"]/inst["_n"]-obs["position"]) < radius:
                inst["_sum"] += obs["position"]; inst["_n"] += 1
                inst["views"].append(obs["photo"]); inst["scores"].append(obs["score"])
                break
        else:
            instances.append({"class": obs["class"], "label": KOREAN.get(obs["class"], obs["class"]),
                              "_sum": np.array(obs["position"], float), "_n": 1, "views": [obs["photo"]], "scores": [obs["score"]]})
    result = []
    for inst in instances:
        if len(set(inst["views"])) < min_views:
            continue
        result.append({"class": inst["class"], "label": inst["label"],
                       "position": (inst["_sum"]/inst["_n"]).round(3).tolist(),
                       "view_count": len(set(inst["views"])), "max_score": round(max(inst["scores"]), 3),
                       "photos": sorted(set(inst["views"]))[:8]})
    return result


def usage_hints(spaces, objects):
    """공간 다각형 안의 강한 사물 근거로 용도 후보를 만든다. 확정이 아니라 근거 목록이다."""
    hints = {}
    for space in spaces:
        polygon = np.array(space["polygon_xz"], np.float32)
        inside = [o for o in objects if cv2.pointPolygonTest(polygon, (float(o["position"][0]), float(o["position"][2])), True) > -.3]
        strong = [o for o in inside if o["view_count"] >= STRONG_VIEWS and o["max_score"] >= STRONG_SCORE]
        found = []
        for cls, key, label in SINGLE_RULES:
            evidence = [o for o in strong if o["class"] == cls]
            if evidence:
                found.append({"usage": key, "label": label, "evidence": [f'{o["label"]}({o["view_count"]}장)' for o in evidence]})
        kitchen = [o for o in strong if o["class"] in KITCHEN and o["max_score"] >= .7]
        if len({o["class"] for o in kitchen}) >= 2:
            found.append({"usage": "kitchen", "label": "주방 후보", "evidence": [f'{o["label"]}({o["view_count"]}장)' for o in kitchen]})
        hints[space["id"]] = {"objects": [o["label"] for o in inside], "usage_candidates": found}
    return hints


def analyze_objects(image_dir, names, centers, rotations, intrinsics, points, observations, detector=None, log=print):
    """사진별 검출 → 3D 위치 → 사물 합치기. 좌표는 입력 점·카메라와 같은 좌표계."""
    detector = detector or Detector()
    P = np.asarray(points, float)
    per_camera = [[] for _ in names]
    for k, observers in enumerate(observations):
        for j in observers:
            per_camera[j].append(k)
    located, per_photo = [], {}
    for j, name in enumerate(names):
        image = cv2.imdecode(np.fromfile(str(Path(image_dir)/name), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        detections = detector.detect(image)
        per_photo[name] = detections
        if not detections or not per_camera[j]:
            continue
        idx = np.array(per_camera[j])
        R = np.asarray(rotations[j], float)
        cam = (P[idx]-np.asarray(centers[j], float))@R.T
        iw, ih, f = intrinsics[j]
        depth = cam[:, 2]
        uv = cam[:, :2]/np.maximum(depth[:, None], 1e-6)*f+[iw/2, ih/2]
        for det in detections:
            spot = locate_detection(det["box"], P[idx], uv, depth)
            if spot:
                located.append({"class": det["class"], "score": det["score"], "photo": name, **spot})
    instances = fuse_instances(located)
    log(f"  사물 검출: 사진 {len(per_photo)}장, 3D 위치 {len(located)}건 → 여러 사진에서 확인된 사물 {len(instances)}개")
    return instances, per_photo
