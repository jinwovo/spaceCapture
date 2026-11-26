"""작은 방 후보 판별 MLP를 LH 실제 평면도·ProcTHOR 구조의 모의 관측으로 학습·평가한다 (CPU/NumPy).

결과는 새 실행 폴더(artifacts/space-learning/runs/<시각>)에만 저장하고 설치 모델은 바꾸지 않는다.
LH는 단지, ProcTHOR는 공식 분할+구조 유형으로 나눈다. 9번·227번 평면도 단지는 시험 분할에 고정한다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from spacecapture.layout import analyze_layout
from spacecapture.plan_image import auto_meters_per_pixel, render_config
from spacecapture.procthor import convert_house
from spacecapture.property_evaluation import evaluate_graph
from spacecapture.property_graph import infer_property
from spacecapture.room import detect_room
from spacecapture.space_model import ARCHITECTURE, FEATURES, VERSION, fingerprint, predict
from spacecapture.synth_property import generate_property

LH = ROOT/"samples/external/lh-floorplans"
PROCTHOR = ROOT/"samples/external/procthor-10k"
FIXED_TEST_PLANS = (9, 227)
USED_PROCTHOR = {"val-00140", "val-00043", "test-00186"}
SPLITS = ("train", "validation", "test")


def split_of(family, salt="space-v1"):
    v = int(hashlib.sha256(f"{salt}/{family}".encode()).hexdigest()[:8], 16) % 100
    return "train" if v < 70 else "validation" if v < 85 else "test"


def stratified_split(houses, ratios=(.7, .15, .15), salt="space-v2"):
    """출처(LH/ProcTHOR)별로 단지·구조 유형 단위를 유지하며 집 수 기준 70/15/15로 나눈다."""
    fixed = {h["family"] for h in houses if h.get("fixed_test")}
    out = {}
    for source in ("lh/", "procthor/"):
        families = {}
        for h in houses:
            if h["family"].startswith(source): families.setdefault(h["family"], []).append(h["id"])
        total = sum(len(v) for v in families.values()); done = 0
        for fam in sorted(families, key=lambda f: hashlib.sha256(f"{salt}/{f}".encode()).hexdigest()):
            if fam in fixed: split = "test"
            else:
                frac = done/max(1, total)
                split = "train" if frac < ratios[0] else "validation" if frac < ratios[0]+ratios[1] else "test"
            done += len(families[fam])
            for hid in families[fam]: out[hid] = split
    return out


def prepare(run, lh_limit, procthor_per_split, log=print):
    sims = run/"sims"; sims.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((LH/"manifest.json").read_text(encoding="utf-8"))
    fixed = {manifest["plans"][i]["complex"] for i in FIXED_TEST_PLANS}
    houses, skipped = [], {}
    for index, plan in enumerate(manifest["plans"][:lh_limit]):
        family = "lh/"+plan["complex"]
        split = "test" if plan["complex"] in fixed else split_of(family)
        try:
            image = cv2.imdecode(np.fromfile(LH/plan["file"], np.uint8), cv2.IMREAD_COLOR)
            mpp, seg = auto_meters_per_pixel(image)
            total = sum(s["area_m2"] for s in seg["spaces"])
            if not (3 <= len(seg["spaces"]) <= 18 and len(seg["portals"]) >= 2 and 25 <= total <= 250):
                raise ValueError("평면도 양식·크기 조건 밖")
            config, excluded = render_config(seg, seed=index, room_views=12)
            if len(config["spaces"]) < 2:
                raise ValueError("사진 대상 공간 2개 미만")
        except Exception as error:  # 홍보물·투시도 등 양식이 다른 도면은 건너뛴다
            skipped[f"lh-{index:04d}"] = str(error)[:80]; continue
        hid = f"lh-{index:04d}"
        generate_property(sims/hid, config=config, render_photos=False, log=lambda *_: None, landmark_density=40, seed=index,
                          truth_metadata={"family": family, "split": split, "source": "LH 주택 평면도 현황", "scale": "door_width_auto"})
        houses.append({"id": hid, "family": family, "split": split, "spaces": len(config["spaces"]), "portals": len(config["portals"]),
                       "fixed_test": plan["complex"] in fixed})
        if len(houses) % 20 == 0: log(f"  LH 모의 관측 {len(houses)}채 (건너뜀 {len(skipped)})")
    # ProcTHOR roomSpecId repeats across the official splits, so the split is
    # assigned per structure type; the official files are only a source pool.
    counts = {k: 0 for k in SPLITS}
    for name in ("train", "val", "test"):
        with gzip.open(PROCTHOR/f"{name}.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream):
                if min(counts.values()) >= procthor_per_split: break
                hid = f"{name}-{line_no:05d}"
                if hid in USED_PROCTHOR: continue
                house = json.loads(line)
                family = "procthor/"+house["metadata"]["roomSpecId"]
                split = split_of(family)
                if counts[split] >= procthor_per_split: continue
                try:
                    config, _ = convert_house(house, hid, seed=line_no)
                except (ValueError, KeyError):
                    continue
                if len(config["spaces"]) < 2: continue
                generate_property(sims/f"pt-{hid}", config=config, render_photos=False, log=lambda *_: None, landmark_density=40,
                                  seed=line_no, truth_metadata={"family": family, "split": split})
                houses.append({"id": f"pt-{hid}", "family": family, "split": split,
                               "spaces": len(config["spaces"]), "portals": len(config["portals"])})
                counts[split] += 1
    log(f"  ProcTHOR: {counts}")
    families = {}
    for h in houses:
        if families.setdefault(h["family"], h["split"]) != h["split"]:
            raise ValueError("같은 단지·구조 유형이 여러 분할에 들어갔어요.")
    data = {"houses": houses, "skipped": skipped, "fixed_test_complexes": sorted(fixed),
            "split_unit": "LH 단지 / ProcTHOR 공식 분할+roomSpecId", "observations": "정답 구조의 이상적 관측 + 잡음·관측 누락 변형",
            "real_photo_training": False}
    (run/"dataset.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def load_sim(folder, variant):
    rng = np.random.default_rng(variant*7919+int(hashlib.sha256(folder.name.encode()).hexdigest()[:6], 16))
    with np.load(folder/"observations.npz") as z:
        P, C = z["points"].copy(), z["cameras"].copy(); offsets, observers = z["offsets"], z["observers"]
    obs = [observers[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    truth = json.loads((folder/"property.gt.json").read_text(encoding="utf-8"))
    if variant:
        P = P+rng.normal(0, .008*variant, P.shape)
        C = C+rng.normal(0, .005*variant, C.shape)
        kept = [[j for j in o if rng.random() > .25*variant] for o in obs]
        keep = np.array([len(o) >= 2 for o in kept])
        P, obs = P[keep], [o for o, k in zip(kept, keep) if k]
    return P, C, obs, truth


def run_inference(P, C, obs, truth, model):
    names = [p["id"] for p in truth["photos"]]
    layout = analyze_layout(P, C, detect_room(P, C))
    debug = {}
    graph = infer_property(P, C, names, obs, planes=layout["planes"], debug=debug, small_room_model=model)
    return graph, debug


def label_candidates(debug, truth):
    labels, lower, step = debug.get("labels"), debug.get("lower"), debug.get("step")
    if labels is None: return []
    h, w = labels.shape
    raster = np.zeros((h, w), np.int32)
    for k, space in enumerate(truth["spaces"], 1):
        poly = np.round((np.array(space["polygon_xz"])-lower)/step).astype(np.int32)
        cv2.fillPoly(raster, [poly], k)
    covered = set()
    core_count = debug["initial_core_count"]
    for i in range(1, core_count+1):
        cells = raster[labels == i]; cells = cells[cells > 0]
        if len(cells): covered.add(int(np.bincount(cells).argmax()))
    out = []
    order = sorted(range(len(debug["small_candidates"])), key=lambda k: -int(debug["small_candidates"][k]["mask"].sum()))
    claimed = set(); positive = {}
    for k in order:
        cells = raster[debug["small_candidates"][k]["mask"] > 0]
        best, frac = 0, 0.
        if len(cells):
            counts = np.bincount(cells, minlength=len(truth["spaces"])+1); counts[0] = 0
            best = int(counts.argmax()); frac = counts[best]/len(cells)
        positive[k] = bool(best and frac >= .6 and best not in covered and best not in claimed)
        if positive[k]: claimed.add(best)
    return [positive[k] for k in range(len(debug["small_candidates"]))]


def collect(run, data, variants, log=print):
    rows = {s: {"X": [], "y": [], "house": [], "rule": []} for s in SPLITS}
    started = time.perf_counter()
    for n, house in enumerate(data["houses"]):
        for variant in range(variants):
            P, C, obs, truth = load_sim(run/"sims"/house["id"], variant)
            try:
                _, debug = run_inference(P, C, obs, truth, False)
            except Exception as error:
                log(f"  추론 실패 {house['id']}: {error}"); continue
            labels = label_candidates(debug, truth)
            for cand, y in zip(debug.get("small_candidates", []), labels):
                r = rows[house["split"]]
                r["X"].append(cand["features"]); r["y"].append(float(y)); r["house"].append(f"{house['id']}/{variant}"); r["rule"].append(cand["rule_accept"])
        if (n+1) % 40 == 0: log(f"  후보 특징 {n+1}/{len(data['houses'])}채 ({time.perf_counter()-started:.0f}초)")
    out = {}
    for s, r in rows.items():
        out[s] = {"X": np.asarray(r["X"], np.float32).reshape(-1, len(FEATURES)), "y": np.asarray(r["y"], np.float32),
                  "house": np.asarray(r["house"]), "rule": np.asarray(r["rule"], bool)}
        np.savez_compressed(run/f"{s}.npz", **out[s])
    return out, time.perf_counter()-started


def train(data, validation, epochs=300, log=print):
    rng = np.random.default_rng(23)
    X, y = data["X"], data["y"]
    sizes = ARCHITECTURE
    weights = {"mean": X.mean(0), "std": np.maximum(X.std(0), 1e-3)}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
        weights[f"w{i}"] = (rng.standard_normal((a, b))*np.sqrt(2/a)).astype(np.float32)
        weights[f"b{i}"] = np.zeros(b, np.float32)
    keys = ["w1", "b1", "w2", "b2", "w3", "b3"]
    m = {k: np.zeros_like(weights[k]) for k in keys}; v = {k: np.zeros_like(weights[k]) for k in keys}
    norm = np.clip((X-weights["mean"])/weights["std"], -8, 8)
    ratio = max(1., float(np.sum(y == 0))/max(1, np.sum(y == 1)))
    best_loss, best, stale, step, history = np.inf, None, 0, 0, []
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 256):
            ids = order[start:start+256]
            x, t = norm[ids], y[ids]
            a = np.maximum(0, x@weights["w1"]+weights["b1"]); b = np.maximum(0, a@weights["w2"]+weights["b2"])
            p = 1/(1+np.exp(-np.clip((b@weights["w3"]+weights["b3"]).reshape(-1), -35, 35)))
            sw = np.where(t > 0, ratio, 1).astype(np.float32)
            dz = ((p-t)*sw/sw.sum())[:, None]
            g = {"w3": b.T@dz, "b3": dz.sum(0)}
            db = (dz@weights["w3"].T)*(b > 0); g.update(w2=a.T@db, b2=db.sum(0))
            da = (db@weights["w2"].T)*(a > 0); g.update(w1=x.T@da, b1=da.sum(0))
            step += 1
            for k in keys:
                grad = g[k]+(1e-4*weights[k] if k.startswith("w") else 0)
                m[k] = .9*m[k]+.1*grad; v[k] = .999*v[k]+.001*grad*grad
                weights[k] -= .002*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        pv = np.clip(predict(validation["X"], weights), 1e-6, 1-1e-6); vy = validation["y"]
        loss = float(-.5*(np.log(pv[vy == 1]).mean()+np.log(1-pv[vy == 0]).mean())) if (vy == 1).any() else float(-np.log(1-pv).mean())
        history.append(loss)
        if loss < best_loss-1e-5: best_loss, best, stale = loss, {k: val.copy() for k, val in weights.items()}, 0
        else: stale += 1
        if epoch % 25 == 0: log(f"  학습 {epoch+1}/{epochs}, 검증 손실 {loss:.4f}")
        if stale >= 40: break
    return best, history


def candidate_metrics(y, accept):
    y = y > 0
    tp, fp, fn = int(np.sum(y & accept)), int(np.sum(~y & accept)), int(np.sum(y & ~accept))
    return {"candidates": int(len(y)), "positives": int(y.sum()), "true_accept": tp, "false_accept": fp, "missed": fn,
            "precision": round(tp/max(1, tp+fp), 4), "recall": round(tp/max(1, tp+fn), 4)}


def end_to_end(run, data, split, model, variants):
    rows = []
    for house in [h for h in data["houses"] if h["split"] == split]:
        for variant in range(variants):
            P, C, obs, truth = load_sim(run/"sims"/house["id"], variant)
            try:
                graph, _ = run_inference(P, C, obs, truth, model)
                r = evaluate_graph(graph, truth)
            except Exception:
                continue
            small = [s for s in truth["spaces"] if s.get("area_m2", 99) < 6.]
            rows.append({"iou": r["mean_space_iou"], "extra": max(0, r["predicted_space_candidates"]-r["truth_spaces"]),
                         "missing": max(0, r["truth_spaces"]-r["matched_spaces"]), "portal_recall": r["portal_recall"],
                         "portal_precision": r["portal_precision"], "count_exact": r["predicted_space_candidates"] == r["truth_spaces"],
                         "lh": house["id"].startswith("lh-")})
    def summary(rs):
        if not rs: return None
        return {"cases": len(rs), "mean_iou": round(float(np.mean([r["iou"] for r in rs])), 4),
                "space_count_exact": round(float(np.mean([r["count_exact"] for r in rs])), 4),
                "extra_spaces": int(sum(r["extra"] for r in rs)), "missing_spaces": int(sum(r["missing"] for r in rs)),
                "portal_recall": round(float(np.mean([r["portal_recall"] for r in rs])), 4),
                "portal_precision": round(float(np.mean([r["portal_precision"] for r in rs])), 4)}
    return {"all": summary(rows), "lh": summary([r for r in rows if r["lh"]]), "procthor": summary([r for r in rows if not r["lh"]])}


def main():
    parser = argparse.ArgumentParser(description="작은 방 후보 판별 MLP 학습 (새 실행 폴더에 저장)")
    parser.add_argument("--lh-limit", type=int, default=281)
    parser.add_argument("--procthor-per-split", type=int, default=60)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--reuse", default=None, help="자료·특징을 재사용할 기존 실행 폴더")
    parser.add_argument("--extra", default=None, help="실제 SfM 복원 후보 특징(real-features.npz). 집별 분할을 따르고 3배 가중")
    args = parser.parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = ROOT/"artifacts/space-learning/runs"/stamp; run.mkdir(parents=True)
    started = time.perf_counter()
    sims_root = run
    if args.reuse:
        src = Path(args.reuse); src = (src if src.is_absolute() else ROOT/src).resolve(); sims_root = src
        data = json.loads((src/"dataset.json").read_text(encoding="utf-8"))
        merged = {}
        for split in SPLITS:
            with np.load(src/f"{split}.npz", allow_pickle=False) as z:
                for key in ("X", "y", "house", "rule"):
                    merged.setdefault(key, []).append(z[key])
        merged = {k: np.concatenate(v) for k, v in merged.items()}
        feature_seconds = None
        print(f"재사용: {src.relative_to(ROOT)} → 새 실행 폴더 {run.relative_to(ROOT)}", flush=True)
    else:
        print(f"실행 폴더: {run.relative_to(ROOT)} (설치 모델은 바꾸지 않아요)", flush=True)
        data = prepare(run, args.lh_limit, args.procthor_per_split, lambda s: print(s, flush=True))
        rows, feature_seconds = collect(run, data, args.variants, lambda s: print(s, flush=True))
        merged = {k: np.concatenate([rows[s][k] for s in SPLITS]) for k in ("X", "y", "house", "rule")}
    lh_plans = json.loads((LH/"manifest.json").read_text(encoding="utf-8"))["plans"]
    fixed_families = {"lh/"+lh_plans[i]["complex"] for i in FIXED_TEST_PLANS}
    for h in data["houses"]:
        h["fixed_test"] = h["family"] in fixed_families
    assignment = stratified_split(data["houses"])
    for h in data["houses"]: h["split"] = assignment[h["id"]]
    families = {}
    for h in data["houses"]:
        if families.setdefault(h["family"], h["split"]) != h["split"]: raise ValueError("같은 단지·구조 유형이 여러 분할에 들어갔어요.")
    data["split_unit"] = "출처별 단지/구조 유형 단위, 집 수 기준 70/15/15 층화"
    (run/"dataset.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    owner = np.array([assignment[h.split("/")[0]] for h in merged["house"]])
    X = np.column_stack([merged["X"][:, :len(FEATURES)-1] if merged["X"].shape[1] == len(FEATURES)-1 else merged["X"][:, :len(FEATURES)-1],
                         merged["rule"].astype(np.float32)])
    rows = {s: {"X": X[owner == s], "y": merged["y"][owner == s], "house": merged["house"][owner == s], "rule": merged["rule"][owner == s]} for s in SPLITS}
    real_counts = None
    if args.extra:
        with np.load(args.extra, allow_pickle=False) as z:
            extra = {k: z[k] for k in z.files}
        real_counts = {}
        for split in ("train", "validation"):
            m = extra["split"] == split
            reps = 3 if split == "train" else 1  # 실제 SfM 후보는 적어서 학습에서 3배 가중
            rows[split]["X"] = np.vstack([rows[split]["X"]]+[extra["X"][m]]*reps)
            rows[split]["y"] = np.concatenate([rows[split]["y"]]+[extra["y"][m]]*reps)
            rows[split]["house"] = np.concatenate([rows[split]["house"]]+[np.char.add("real/", extra["house"][m])]*reps)
            rows[split]["rule"] = np.concatenate([rows[split]["rule"]]+[extra["X"][m][:, -1] > .5]*reps)
            real_counts[split] = {"candidates": int(m.sum()), "positives": int(extra["y"][m].sum()), "houses": len(set(extra["house"][m]))}
    for s in SPLITS: np.savez_compressed(run/f"{s}.npz", **rows[s])
    counts = {s: {"houses": sum(h["split"] == s for h in data["houses"]),
                  "lh": sum(h["split"] == s and h["id"].startswith("lh-") for h in data["houses"]),
                  "candidates": int(len(rows[s]["y"])), "positives": int(rows[s]["y"].sum())} for s in SPLITS}
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    t0 = time.perf_counter()
    weights, history = train(rows["train"], rows["validation"], log=lambda s: print(s, flush=True))
    training_seconds = time.perf_counter()-t0
    vs = predict(rows["validation"]["X"], weights); vy = rows["validation"]["y"] > 0
    best = None
    for th in np.linspace(.2, .95, 76):
        a = vs >= th; tp = np.sum(vy & a); fp = np.sum(~vy & a)
        score = tp-2*fp
        if best is None or score > best[0]: best = (score, float(th))
    threshold = best[1]
    model_dir = run/"model"; model_dir.mkdir(exist_ok=True)
    path = model_dir/"space-candidate-mlp-v1.npz"
    np.savez_compressed(path, **weights)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {"id": "space-candidate-mlp-v1", "feature_version": VERSION, "features": FEATURES, "architecture": ARCHITECTURE,
            "sha256": sha, "threshold": threshold, "fingerprint": fingerprint(sha, threshold), "enabled_by_default": False,
            "training": "LH 실제 평면도(공공데이터포털, 이용허락범위 제한 없음)·ProcTHOR(Apache-2.0) 구조의 모의 관측",
            "real_photo_training": False, "license_note": "가중치는 이 프로젝트에서 처음부터 학습. LH 자료는 저작권 확보 표시 N(상업 학습 전 제공기관 확인 필요)."}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"neural_training_performed": True, "device": "CPU / NumPy", "architecture": ARCHITECTURE, "threshold": threshold,
              "feature_seconds": round(feature_seconds, 1) if feature_seconds else None, "training_seconds": round(training_seconds, 2),
              "epochs": len(history), "counts": counts, "real_sfm_extra": real_counts, "skipped_lh_plans": len(data["skipped"]),
              "candidate_test": {"rule": candidate_metrics(rows["test"]["y"], rows["test"]["rule"]),
                                 "model": candidate_metrics(rows["test"]["y"], predict(rows["test"]["X"], weights) >= threshold)},
              "candidate_validation": {"rule": candidate_metrics(rows["validation"]["y"], rows["validation"]["rule"]),
                                       "model": candidate_metrics(rows["validation"]["y"], vs >= threshold)},
              "model_sha256": sha, "fingerprint": meta["fingerprint"]}
    t0 = time.perf_counter()
    report["end_to_end_test"] = {"rule": end_to_end(sims_root, data, "test", False, args.variants),
                                 "model": end_to_end(sims_root, data, "test", path, args.variants)}
    report["end_to_end_seconds"] = round(time.perf_counter()-t0, 1)
    report["total_seconds"] = round(time.perf_counter()-started, 1)
    (run/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"학습 결과: {run.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
