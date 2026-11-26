"""공간 용도(거실·침실·욕실·기타) MLP 학습. 새 실행 폴더에 저장하고 설치는 activate-room-type-model.py로.

자료: LH 평면도 구조를 자체 렌더한 사진의 실제 SfM 복원(scenes/space-v2-*)에서 설치 파이프라인으로 만든 공간 후보.
정답: 평면도 실 이름 글자(OCR, tools/label-lh-rooms.py)를 평가용 정렬로 대응된 정답 공간에서 가져온다.
집 단위 기존 분할(학습/검증/시험)을 그대로 쓰고, 시험 분할은 마지막 점수에만 쓴다.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from spacecapture.property_evaluation import evaluate_graph
from spacecapture.property_graph import infer_property
from spacecapture.room_types import ARCHITECTURE, CLASSES, FEATURES, VERSION, fingerprint, predict, space_features

LABELS = ROOT/"artifacts/space-learning/room-labels/labels.json"
SPLITS = ("train", "validation", "test")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT/"tools"/file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def _plan_to_scene(original, moved):
    """평가용 정답 정렬(평면도 좌표 → 복원 좌표)을 원래·이동 정답 다각형 꼭짓점에서 다시 구한다(2D 닮음 변환)."""
    from spacecapture.pipeline import umeyama
    A = np.array([[x, 0., z] for s in original["spaces"] for x, z in s["polygon_xz"]])
    B = np.array([[x, 0., z] for s in moved["spaces"] for x, z in s["polygon_xz"]])
    s, R, t = umeyama(A, B)
    return lambda xz: (s*R@np.array([xz[0], 0., xz[1]])+t)[[0, 2]]


def build_rows(log=print):
    """예측 공간마다, 그 다각형 안에 들어오는 평면도 실 이름 글자로 정답 용도를 붙인다."""
    import cv2
    reg = _load("reg", "experiment-regularize.py"); llr = _load("llr", "label-lh-rooms.py")
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["houses"]
    real = ROOT/"artifacts/space-learning/real-v2"
    rows = {s: {"X": [], "y": [], "house": [], "space": []} for s in SPLITS}
    skipped = {"mixed": 0, "unknown": 0}
    for split in SPLITS:
        for c in reg.load_cases(split, connected_only=False):
            info = labels.get(c["id"])
            if not info or not info["consistent"] or not info.get("text_points"):
                continue
            original = json.loads((real/c["id"]/"property.gt.json").read_text(encoding="utf-8"))
            to_scene = _plan_to_scene(original, c["truth"])
            texts = [(p["category"], to_scene(p["xz"])) for p in info["text_points"]]
            graph = infer_property(*c["args"], **c["kw"])
            for space, f in zip(graph["spaces"], space_features(graph)):
                if f is None: continue
                poly = np.asarray(space["polygon_xz"], np.float32)
                inside = [cat for cat, xz in texts if cv2.pointPolygonTest(poly, (float(xz[0]), float(xz[1])), True) > .05]
                lab = llr.label_from(inside)
                if lab not in CLASSES:
                    skipped[lab] = skipped.get(lab, 0)+1; continue
                rows[split]["X"].append(f); rows[split]["y"].append(CLASSES.index(lab))
                rows[split]["house"].append(c["id"]); rows[split]["space"].append(space["id"])
        log(f"{split}: {len(rows[split]['y'])}개 공간, 분포 {np.bincount(rows[split]['y'], minlength=len(CLASSES)).tolist()}, 제외 {skipped}")
    return {s: {k: np.asarray(v, np.float32 if k == "X" else None) for k, v in r.items()} for s, r in rows.items()}


def rule_baseline(X):
    """비교용 규칙: 가장 큰 공간=거실, 안쪽 폭 2.0m·면적 5㎡ 미만=욕실, 나머지=침실."""
    f = {n: i for i, n in enumerate(FEATURES)}
    out = np.full(len(X), CLASSES.index("bedroom"))
    out[(X[:, f["inner_width_m"]] < 2.0) & (np.exp(X[:, f["log_area"]]) < 5.)] = CLASSES.index("bathroom")
    out[X[:, f["is_largest"]] > .5] = CLASSES.index("living")
    return out


def train(X, y, Xv, yv, epochs=600, seed=0, log=print):
    rng = np.random.default_rng(seed)
    mean, std = X.mean(0), X.std(0)+1e-6
    n, h1, h2, c = ARCHITECTURE
    W = {"w1": rng.normal(0, np.sqrt(2/n), (n, h1)), "b1": np.zeros(h1), "w2": rng.normal(0, np.sqrt(2/h1), (h1, h2)),
         "b2": np.zeros(h2), "w3": rng.normal(0, np.sqrt(2/h2), (h2, c)), "b3": np.zeros(c)}
    counts = np.bincount(y, minlength=c); cw = (len(y)/(c*np.maximum(counts, 1)))[y]
    m = {k: np.zeros_like(v) for k, v in W.items()}; v2 = {k: np.zeros_like(v) for k, v in W.items()}
    Y = np.eye(c)[y]; best = (np.inf, None, 0)
    for epoch in range(1, epochs+1):
        x = (X-mean)/std
        a1 = np.maximum(0, x@W["w1"]+W["b1"]); a2 = np.maximum(0, a1@W["w2"]+W["b2"]); z = a2@W["w3"]+W["b3"]
        z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = (p-Y)*cw[:, None]/cw.sum()
        grads = {"w3": a2.T@g, "b3": g.sum(0)}
        d2 = (g@W["w3"].T)*(a2 > 0); grads.update(w2=a1.T@d2, b2=d2.sum(0))
        d1 = (d2@W["w2"].T)*(a1 > 0); grads.update(w1=x.T@d1, b1=d1.sum(0))
        for k in W:
            if k.startswith("w"): grads[k] = grads[k]+1e-3*W[k]
            m[k] = .9*m[k]+.1*grads[k]; v2[k] = .999*v2[k]+.001*grads[k]**2
            W[k] -= 3e-3*(m[k]/(1-.9**epoch))/(np.sqrt(v2[k]/(1-.999**epoch))+1e-8)
        weights = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), **{k: v.astype(np.float32) for k, v in W.items()}}
        pv = predict(Xv, weights); loss = -np.mean(np.log(pv[np.arange(len(yv)), yv]+1e-9))
        if loss < best[0]: best = (loss, {k: v.copy() for k, v in weights.items()}, epoch)
        if epoch-best[2] > 80 and Xv is not X: break
    log(f"검증 손실 최소 {best[0]:.4f} @ {best[2]}회")
    return best[1], best[2]


def metrics(y, pred, decided=None):
    decided = np.ones(len(y), bool) if decided is None else decided
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(y[decided], pred[decided]): conf[t, p] += 1
    per = {c: {"precision": round(conf[i, i]/max(1, conf[:, i].sum()), 3), "recall": round(conf[i, i]/max(1, conf[i].sum()), 3),
               "support": int(conf[i].sum())} for i, c in enumerate(CLASSES)}
    return {"accuracy": round(float(np.mean(y[decided] == pred[decided])) if decided.any() else 0., 4),
            "coverage": round(float(decided.mean()), 4), "per_class": per, "confusion": conf.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", default=None, help="특징을 재사용할 실행 폴더")
    args = parser.parse_args()
    run = ROOT/"artifacts/room-type-learning/runs"/dt.datetime.now().strftime("%Y%m%d-%H%M%S"); run.mkdir(parents=True)
    started = time.perf_counter()
    if args.reuse:
        src = Path(args.reuse); src = src if src.is_absolute() else ROOT/src
        rows = {s: dict(np.load(src/f"{s}.npz", allow_pickle=False)) for s in SPLITS}
    else:
        rows = build_rows(lambda s: print(s, flush=True))
    for s in SPLITS: np.savez_compressed(run/f"{s}.npz", **rows[s])
    # 검증 분할이 작아(11채) 학습+검증을 단지(family) 단위 5겹 교차검증으로 반복 수·임계값을 고른 뒤 전체로 다시 학습한다.
    data = json.loads((ROOT/"artifacts/space-learning/lh-v2-dataset.json").read_text(encoding="utf-8"))
    family = {h["id"]: h["family"] for h in data["houses"]}
    X = np.vstack([rows["train"]["X"], rows["validation"]["X"]]); y = np.concatenate([rows["train"]["y"], rows["validation"]["y"]]).astype(int)
    houses = np.concatenate([rows["train"]["house"], rows["validation"]["house"]])
    fams = sorted({family[h] for h in houses}); fold_of = {f: i % 5 for i, f in enumerate(np.random.default_rng(0).permutation(fams))}
    fold = np.array([fold_of[family[h]] for h in houses])
    oof = np.zeros((len(y), len(CLASSES))); chosen = []
    for k in range(5):
        tr_m, va_m = fold != k, fold == k
        w, e = train(X[tr_m], y[tr_m], X[va_m], y[va_m], log=lambda s: None)
        oof[va_m] = predict(X[va_m], w); chosen.append(e)
    epochs = int(np.median(chosen))
    print(f"교차검증 정확도 {np.mean(oof.argmax(1) == y):.3f}, 반복 {chosen} → {epochs}", flush=True)
    weights, _ = train(X, y, X, y, epochs=epochs, log=lambda s: None)
    best = None
    for th in np.linspace(.3, .95, 66):
        d = (oof.max(1) >= th) & (oof.argmax(1) != CLASSES.index("other"))
        score = int(np.sum(d & (oof.argmax(1) == y)))-3*int(np.sum(d & (oof.argmax(1) != y)))
        if best is None or score > best[0]: best = (score, float(th))
    threshold = round(best[1], 3)
    model_dir = run/"model"; model_dir.mkdir(); path = model_dir/"room-type-mlp-v1.npz"
    np.savez_compressed(path, **weights); sha = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {"id": "room-type-mlp-v1", "feature_version": VERSION, "features": FEATURES, "classes": CLASSES, "architecture": ARCHITECTURE,
            "sha256": sha, "threshold": threshold, "fingerprint": fingerprint(sha, threshold), "enabled_by_default": False,
            "training": "LH 평면도 구조 자체 렌더 사진의 실제 SfM 공간 후보 + 평면도 실 이름 OCR 정답",
            "real_photo_training": False, "license_note": "가중치는 이 프로젝트에서 처음부터 학습. LH 자료는 저작권 확보 표시 N(상업 학습 전 제공기관 확인 필요)."}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"neural_training_performed": True, "device": "CPU / NumPy", "epochs": epochs, "threshold": threshold,
              "cross_validation": {"folds": 5, "group": "단지(family)", "accuracy": round(float(np.mean(oof.argmax(1) == y)), 4), "epochs": chosen},
              "counts": {s: np.bincount(rows[s]["y"].astype(int), minlength=len(CLASSES)).tolist() for s in SPLITS},
              "houses": {s: len(set(rows[s]["house"].tolist())) for s in SPLITS}, "classes": CLASSES,
              "model_sha256": sha, "fingerprint": meta["fingerprint"]}
    for s in ("validation", "test"):
        X, y = rows[s]["X"], rows[s]["y"].astype(int); p = predict(X, weights)
        decided = (p.max(1) >= threshold) & (p.argmax(1) != CLASSES.index("other"))
        report[s] = {"model_all": metrics(y, p.argmax(1)), "model_decided": metrics(y, p.argmax(1), decided),
                     "rule": metrics(y, rule_baseline(X))}
    report["seconds"] = round(time.perf_counter()-started, 1)
    (run/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "houses", "threshold")}, ensure_ascii=False))
    for s in ("validation", "test"):
        print(s, "규칙", report[s]["rule"]["accuracy"], "| 모델 전체", report[s]["model_all"]["accuracy"],
              "| 모델 확정분", report[s]["model_decided"]["accuracy"], "적용률", report[s]["model_decided"]["coverage"])
        print("  ", {c: v for c, v in report[s]["model_all"]["per_class"].items()})
    print(f"결과: {run.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
