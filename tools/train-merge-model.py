"""문 후보 → 같은 공간 안의 트인 통로(합치기) 판별 MLP 학습·평가 (CPU/NumPy, 새 실행 폴더에만 저장).

자료: 작은 방 학습과 같은 LH·ProcTHOR 모의 관측(층화 분할) + 선택적으로 실제 렌더·SfM 복원 장면.
정답 표시: 문 후보 양쪽 예측 공간이 같은 정답 공간에 대부분 속하면 '합치기'. 추론에는 정답을 쓰지 않는다.
"""
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pycolmap
from spacecapture.align import arrays_from_reconstruction
from spacecapture.property_evaluation import evaluate_graph
from spacecapture.property_graph import infer_property
from spacecapture.space_model import PORTAL_ARCHITECTURE, PORTAL_FEATURES, merge_fingerprint, predict

spec = importlib.util.spec_from_file_location("tsm", ROOT/"tools/train-space-model.py")
tsm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tsm)
spec = importlib.util.spec_from_file_location("rsf", ROOT/"tools/real-sfm-features.py")
rsf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rsf)
SPLITS = ("train", "validation", "test")


def owners(labels, lower, step, truth, count):
    raster = np.zeros(labels.shape, np.int32)
    for k, sp in enumerate(truth["spaces"], 1):
        cv2.fillPoly(raster, [np.round((np.array(sp["polygon_xz"])-lower)/step).astype(np.int32)], k)
    out = {}
    for i in range(1, count+1):
        c = raster[labels == i]; total = len(c); c = c[c > 0]
        if len(c) and np.bincount(c).max() >= .5*total: out[f"space-{i}"] = int(np.bincount(c).argmax())
        else: out[f"space-{i}"] = 0
    return out


def portal_rows(graph, debug, truth):
    own = owners(debug["labels"], debug["lower"], debug["step"], truth, len(graph["spaces"]))
    rows = []
    for p, f in zip(graph["portals"], debug.get("portal_merge_features", [])):
        a, b = (own[x] for x in p["spaces"])
        if a == 0 or b == 0: continue  # 정답 공간에 대응하지 않는 영역은 표시하지 않는다
        rows.append((f, float(a == b)))
    return rows


def sim_cases(sims_root, data, variants):
    for h in data["houses"]:
        for v in range(variants):
            yield h, v, lambda h=h, v=v: tsm.load_sim(sims_root/"sims"/h["id"], v)


def collect(sims_root, data, variants, log):
    rows = {s: {"X": [], "y": [], "house": []} for s in SPLITS}; started = time.perf_counter()
    for n, (h, v, loader) in enumerate(sim_cases(sims_root, data, variants)):
        P, C, obs, truth = loader()
        try:
            names = [p["id"] for p in truth["photos"]]
            from spacecapture.layout import analyze_layout
            from spacecapture.room import detect_room
            layout = analyze_layout(P, C, detect_room(P, C)); debug = {}
            graph = infer_property(P, C, names, obs, planes=layout["planes"], debug=debug, merge_model=False)
        except Exception:
            continue
        for f, y in portal_rows(graph, debug, truth):
            r = rows[h["split"]]; r["X"].append(f); r["y"].append(y); r["house"].append(f"{h['id']}/{v}")
        if (n+1) % 100 == 0: log(f"  문 후보 특징 {n+1}건 ({time.perf_counter()-started:.0f}초)")
    return {s: {"X": np.asarray(r["X"], np.float32).reshape(-1, len(PORTAL_FEATURES)), "y": np.asarray(r["y"], np.float32),
                "house": np.asarray(r["house"])} for s, r in rows.items()}, time.perf_counter()-started


def real_cases(data):
    for h in data["houses"]:
        if (ROOT/"scenes"/f"{rsf.PREFIX}{h['id']}"/"scene.json").exists() and (rsf.REAL/h["id"]/"property.gt.json").exists():
            yield h


def real_inputs(hid):
    scene_dir = ROOT/"scenes"/f"{rsf.PREFIX}{hid}"
    scene = json.loads((scene_dir/"scene.json").read_text(encoding="utf-8"))
    truth = json.loads((rsf.REAL/hid/"property.gt.json").read_text(encoding="utf-8"))
    ref = {p["id"]: p for p in truth["photos"]}
    cams = [c for c in scene["cameras"] if c.get("source_photo") in ref]
    T = np.array([ref[c["source_photo"]]["position"] for c in cams]); M = np.array([c["position"] for c in cams])
    s, R, t, keep, err = rsf.robust_alignment(T, M)
    if keep.mean() < .8 or float(np.sqrt(np.mean(err[keep]**2))) >= .1: return None
    moved = copy.deepcopy(truth)
    for space in moved["spaces"]:
        space["polygon_xz"] = [(s*R@np.array([x, 0, z])+t)[[0, 2]].tolist() for x, z in space["polygon_xz"]]
    for portal in moved["portals"]:
        portal["center_xz"] = (s*R@np.array([portal["center_xz"][0], 0, portal["center_xz"][1]])+t)[[0, 2]].tolist(); portal["width_m"] *= s
    aligned = arrays_from_reconstruction(pycolmap.Reconstruction(scene_dir/"work/colmap/aligned"))
    k = aligned.reliable(min_angle=6.)
    args = (aligned.xyz[k], aligned.centers, aligned.names, [o for o, kk in zip(aligned.observations, k) if kk])
    kw = dict(planes=scene["layout"]["planes"], rotations=aligned.R_cw, intrinsics=aligned.intrinsics)
    return args, kw, moved


def train(X, y, Xv, yv, epochs=400, log=print):
    rng = np.random.default_rng(31)
    w = {"mean": X.mean(0), "std": np.maximum(X.std(0), 1e-3)}
    for i, (a, b) in enumerate(zip(PORTAL_ARCHITECTURE[:-1], PORTAL_ARCHITECTURE[1:]), 1):
        w[f"w{i}"] = (rng.standard_normal((a, b))*np.sqrt(2/a)).astype(np.float32); w[f"b{i}"] = np.zeros(b, np.float32)
    keys = ["w1", "b1", "w2", "b2", "w3", "b3"]; m = {k: np.zeros_like(w[k]) for k in keys}; v = {k: np.zeros_like(w[k]) for k in keys}
    norm = np.clip((X-w["mean"])/w["std"], -8, 8); ratio = max(1., float(np.sum(y == 0))/max(1, np.sum(y == 1)))
    best_loss, best, stale, step = np.inf, None, 0, 0
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 128):
            ids = order[start:start+128]; x, t = norm[ids], y[ids]
            a = np.maximum(0, x@w["w1"]+w["b1"]); b = np.maximum(0, a@w["w2"]+w["b2"])
            p = 1/(1+np.exp(-np.clip((b@w["w3"]+w["b3"]).reshape(-1), -35, 35)))
            sw = np.where(t > 0, ratio, 1).astype(np.float32); dz = ((p-t)*sw/sw.sum())[:, None]
            g = {"w3": b.T@dz, "b3": dz.sum(0)}; db = (dz@w["w3"].T)*(b > 0); g.update(w2=a.T@db, b2=db.sum(0))
            da = (db@w["w2"].T)*(a > 0); g.update(w1=x.T@da, b1=da.sum(0)); step += 1
            for k in keys:
                grad = g[k]+(1e-4*w[k] if k.startswith("w") else 0)
                m[k] = .9*m[k]+.1*grad; v[k] = .999*v[k]+.001*grad*grad
                w[k] -= .002*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        pv = np.clip(predict(Xv, w), 1e-6, 1-1e-6)
        loss = float(-.5*(np.log(pv[yv == 1]).mean()+np.log(1-pv[yv == 0]).mean())) if (yv == 1).any() else float(-np.log(1-pv).mean())
        if loss < best_loss-1e-5: best_loss, best, stale = loss, {k: val.copy() for k, val in w.items()}, 0
        else: stale += 1
        if epoch % 50 == 0: log(f"  학습 {epoch+1}/{epochs}, 검증 손실 {loss:.4f}")
        if stale >= 50: break
    return best, epoch+1


def metrics(y, accept):
    y = y > 0; tp, fp, fn = int(np.sum(y & accept)), int(np.sum(~y & accept)), int(np.sum(y & ~accept))
    return {"portals": int(len(y)), "should_merge": int(y.sum()), "true_merge": tp, "false_merge": fp, "missed": fn,
            "precision": round(tp/max(1, tp+fp), 4), "recall": round(tp/max(1, tp+fn), 4)}


def e2e_sim(sims_root, data, variants, model):
    from spacecapture.layout import analyze_layout
    from spacecapture.room import detect_room
    rows = []
    for h, v, loader in sim_cases(sims_root, {"houses": [h for h in data["houses"] if h["split"] == "test"]}, variants):
        P, C, obs, truth = loader()
        try:
            layout = analyze_layout(P, C, detect_room(P, C))
            g = infer_property(P, C, [p["id"] for p in truth["photos"]], obs, planes=layout["planes"], merge_model=model)
            r = evaluate_graph(g, truth); rows.append((h["id"].startswith("lh-"), r))
        except Exception:
            continue
    return summarize(rows)


def summarize(rows):
    def s(rs):
        if not rs: return None
        return {"cases": len(rs), "mean_iou": round(float(np.mean([r["mean_space_iou"] for r in rs])), 4),
                "space_count_exact": round(float(np.mean([r["predicted_space_candidates"] == r["truth_spaces"] for r in rs])), 4),
                "extra_spaces": int(sum(max(0, r["predicted_space_candidates"]-r["truth_spaces"]) for r in rs)),
                "missing_spaces": int(sum(max(0, r["truth_spaces"]-r["matched_spaces"]) for r in rs)),
                "portal_recall": round(float(np.mean([r["portal_recall"] for r in rs])), 4),
                "portal_precision": round(float(np.mean([r["portal_precision"] for r in rs])), 4)}
    return {"all": s([r for _, r in rows]), "lh": s([r for lh, r in rows if lh]), "procthor": s([r for lh, r in rows if not lh])}


def e2e_real(data, split, model, small=None):
    rows = []
    for h in real_cases({"houses": [h for h in data["houses"] if h["split"] == split]}):
        inp = real_inputs(h["id"])
        if inp is None: continue
        args, kw, moved = inp
        rows.append((bool(h.get("connected", True)), evaluate_graph(infer_property(*args, **kw, small_room_model=small, merge_model=model), moved)))
    out = summarize(rows)
    return {"all": out["all"], "connected": out["lh"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", required=True, help="모의 관측이 있는 실행 폴더")
    parser.add_argument("--dataset", required=True, help="층화 분할 dataset.json")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--reuse-features", default=None)
    parser.add_argument("--real", action="store_true", help="실제 SfM 복원도 학습·평가에 사용")
    parser.add_argument("--real-dataset", default=None, help="실제 SfM 자료의 분할 dataset.json (기본: --dataset)")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    rsf.configure(args.tag)
    sims_root = Path(args.sims); sims_root = (sims_root if sims_root.is_absolute() else ROOT/sims_root).resolve()
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    real_data = json.loads(Path(args.real_dataset).read_text(encoding="utf-8")) if args.real_dataset else data
    run = ROOT/"artifacts/merge-learning/runs"/dt.datetime.now().strftime("%Y%m%d-%H%M%S"); run.mkdir(parents=True)
    print(f"실행 폴더: {run.relative_to(ROOT)}", flush=True)
    started = time.perf_counter()
    if args.reuse_features:
        src = Path(args.reuse_features); src = src if src.is_absolute() else ROOT/src
        rows = {s: dict(np.load(src/f"{s}.npz", allow_pickle=False)) for s in SPLITS}; feature_seconds = None
    else:
        rows, feature_seconds = collect(sims_root, data, args.variants, lambda s: print(s, flush=True))
    real_counts = None
    if args.real:
        real_counts = {}
        for split in ("train", "validation"):
            X, y = [], []
            for h in real_cases({"houses": [h for h in real_data["houses"] if h["split"] == split]}):
                inp = real_inputs(h["id"])
                if inp is None: continue
                a, kw, moved = inp; debug = {}
                g = infer_property(*a, **kw, debug=debug, merge_model=False)
                for f, lab in portal_rows(g, debug, moved): X.append(f); y.append(lab)
            if X:
                reps = 3 if split == "train" else 1
                rows[split]["X"] = np.vstack([rows[split]["X"]]+[np.asarray(X, np.float32)]*reps)
                rows[split]["y"] = np.concatenate([rows[split]["y"]]+[np.asarray(y, np.float32)]*reps)
            real_counts[split] = {"portals": len(X), "should_merge": int(sum(y))}
    for s in SPLITS: np.savez_compressed(run/f"{s}.npz", **{k: v for k, v in rows[s].items() if k in ("X", "y", "house")})
    counts = {s: {"portals": int(len(rows[s]["y"])), "should_merge": int(rows[s]["y"].sum())} for s in SPLITS}
    print(json.dumps({"counts": counts, "real": real_counts}, ensure_ascii=False), flush=True)
    t0 = time.perf_counter()
    weights, epochs = train(rows["train"]["X"], rows["train"]["y"], rows["validation"]["X"], rows["validation"]["y"], log=lambda s: print(s, flush=True))
    training_seconds = time.perf_counter()-t0
    vs = predict(rows["validation"]["X"], weights); vy = rows["validation"]["y"] > 0
    best = max(((int(np.sum(vy & (vs >= th)))-3*int(np.sum(~vy & (vs >= th))), float(th)) for th in np.linspace(.3, .99, 70)))
    threshold = best[1]
    model_dir = run/"model"; model_dir.mkdir(); path = model_dir/"portal-merge-mlp-v1.npz"
    np.savez_compressed(path, **weights); sha = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {"id": "portal-merge-mlp-v1", "features": PORTAL_FEATURES, "architecture": PORTAL_ARCHITECTURE, "sha256": sha,
            "threshold": threshold, "fingerprint": merge_fingerprint(sha, threshold), "enabled_by_default": False,
            "training": "LH 실제 평면도·ProcTHOR 구조의 모의 관측" + (" + 실제 렌더·SfM 복원" if args.real else ""), "real_photo_training": False}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"counts": counts, "real_sfm_extra": real_counts, "threshold": threshold, "epochs": epochs,
              "feature_seconds": round(feature_seconds, 1) if feature_seconds else None, "training_seconds": round(training_seconds, 2),
              "candidate_test": {"keep": metrics(rows["test"]["y"], np.zeros(len(rows["test"]["y"]), bool)),
                                 "model": metrics(rows["test"]["y"], predict(rows["test"]["X"], weights) >= threshold)},
              "candidate_validation": metrics(rows["validation"]["y"], vs >= threshold), "model_sha256": sha, "fingerprint": meta["fingerprint"]}
    t0 = time.perf_counter()
    report["end_to_end_sim_test"] = {"keep": e2e_sim(sims_root, data, args.variants, False), "model": e2e_sim(sims_root, data, args.variants, path)}
    if args.real:
        report["end_to_end_real_test"] = {"keep": e2e_real(real_data, "test", False), "model": e2e_real(real_data, "test", path)}
    report["end_to_end_seconds"] = round(time.perf_counter()-t0, 1); report["total_seconds"] = round(time.perf_counter()-started, 1)
    (run/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"학습 결과: {run.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
