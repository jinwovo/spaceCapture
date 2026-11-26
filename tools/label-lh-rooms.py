"""LH 평면도 이미지의 실 이름 글자(OCR)를 정답 공간에 붙여 공간 용도 정답을 만든다. 학습·평가 표시 전용.

OCR: RapidOCR(Apache-2.0) + PaddleOCR 한국어 인식 모델(Apache-2.0), onnxruntime(MIT) CPU. 도구 전용 의존성이며
추론 파이프라인은 OCR을 쓰지 않는다. 글자 위치(픽셀)를 평면도 축척(m/px)으로 바꿔 어느 정답 공간 안에 있는지 본다.
글자가 없는 평면도(색 렌더 양식 등)의 공간은 'unknown'으로 남겨 학습·평가에서 뺀다.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from spacecapture.plan_image import auto_meters_per_pixel

LH = ROOT/"samples/external/lh-floorplans"
OUT = ROOT/"artifacts/space-learning/room-labels"
# 순서가 우선순위다(한 공간에 여러 글자가 있으면 앞의 것). OCR 오인식(파우더름·펜트리·실외긴실 등)을 넓게 받는다.
RULES = [("living", r"거실|건실"), ("kitchen", r"주방|식당"), ("bathroom", r"욕실|욕|화장실|파우더|파우|더룸|더름"),
         ("bedroom", r"침실|침"), ("other", r"드레스|드레|스룸|팬트리|펜트리|펀트리|트리|다용도|세탁|실외|발코|현관|창고|알파")]
CLASSES = ["living", "kitchen", "bathroom", "bedroom", "other"]


def label_from(names):
    """한 영역 안 글자 분류 집합 → 용도. 거실+침실, 침실+욕실처럼 여러 실이 섞이면 'mixed'(학습·평가 제외)."""
    names = set(names)
    if "living" in names or "kitchen" in names:
        return "mixed" if "bedroom" in names else "living"
    if "bedroom" in names and "bathroom" in names:
        return "mixed"
    return next((c for c in ("bedroom", "bathroom", "other") if c in names), "unknown")


def category(text):
    for name, pattern in RULES:
        if re.search(pattern, text):
            return name
    return None


def ocr_engine():
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
    return RapidOCR(params={"Rec.lang_type": LangRec.KOREAN, "Rec.ocr_version": OCRVersion.PPOCRV5,
                            "Rec.model_type": ModelType.MOBILE, "Det.limit_side_len": 2500, "Global.log_level": "error"})


def label_house(index, truth, plan, engine):
    cache = OUT/"ocr"/f"{index:04d}.json"
    image = cv2.imdecode(np.fromfile(LH/plan["file"], np.uint8), cv2.IMREAD_COLOR)
    if cache.exists():
        texts = json.loads(cache.read_text(encoding="utf-8"))
    else:
        r = engine(image)
        boxes = r.boxes if r.boxes is not None else []
        texts = [{"text": t, "score": round(float(s), 3), "center_px": np.asarray(b, float).mean(0).round(1).tolist()}
                 for b, t, s in zip(boxes, r.txts or [], r.scores or [])]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    mpp, seg = auto_meters_per_pixel(image)
    # 정답 공간과 변환 결과가 같은지(같은 축척·좌표) 면적으로 확인
    seg_area = {s["id"]: s["area_m2"] for s in seg["spaces"]}
    consistent = all(abs(seg_area.get(s["id"], -9)-s["area_m2"]) < .05 for s in truth["spaces"])
    polygons = {s["id"]: np.array(s["polygon_xz"], np.float32) for s in truth["spaces"]}
    found = {sid: [] for sid in polygons}
    points = []
    for t in texts:
        name = category(t["text"])
        if not name or t["score"] < .5:
            continue
        xz = (float(t["center_px"][0]*mpp), float(t["center_px"][1]*mpp))
        points.append({"category": name, "text": t["text"], "xz": [round(xz[0], 3), round(xz[1], 3)]})
        best = max(polygons, key=lambda sid: cv2.pointPolygonTest(polygons[sid], xz, True), default=None)
        if best is not None and cv2.pointPolygonTest(polygons[best], xz, True) > -.3:
            found[best].append((name, t["text"]))
    labels = {}
    for sid, items in found.items():
        names = {n for n, _ in items}
        # 평면도 변환이 여러 실을 한 영역으로 합친 경우(거실+침실, 침실+욕실)는 정답이 모호해 'mixed'로 빼 둔다.
        label = label_from(names)
        labels[sid] = {"label": label, "has_kitchen": "kitchen" in names, "texts": [t for _, t in items]}
    return {"consistent": consistent, "mpp": mpp, "text_count": len(texts), "spaces": labels, "text_points": points}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT/"artifacts/space-learning/lh-v2-dataset.json"))
    parser.add_argument("--real", default=str(ROOT/"artifacts/space-learning/real-v2"))
    args = parser.parse_args()
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    plans = json.loads((LH/"manifest.json").read_text(encoding="utf-8"))["plans"]
    engine = ocr_engine(); started = time.perf_counter(); result = {}
    for h in data["houses"]:
        truth_path = Path(args.real)/h["id"]/"property.gt.json"
        if not truth_path.exists():
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        r = label_house(int(h["id"][3:]), truth, plans[int(h["id"][3:])], engine)
        r["split"] = h["split"]; result[h["id"]] = r
        print(h["id"], r["consistent"], {k: v["label"] for k, v in r["spaces"].items()}, flush=True)
    counts = {}
    for r in result.values():
        if not r["consistent"]: continue
        for v in r["spaces"].values(): counts[v["label"]] = counts.get(v["label"], 0)+1
    report = {"houses": result, "label_counts": counts, "seconds": round(time.perf_counter()-started, 1),
              "ocr": "RapidOCR 3.10 (Apache-2.0) + PP-OCRv5 korean mobile rec (Apache-2.0), onnxruntime CPU",
              "scope": "평면도 글자 기반 정답 표시. 학습·평가 전용이며 추론에 쓰지 않는다."}
    (OUT/"labels.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"label_counts": counts, "inconsistent": sum(not r["consistent"] for r in result.values()),
                      "seconds": report["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
