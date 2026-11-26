"""작은 방 후보 모델의 임계값만 다시 고른다(가중치 불변). 새 실행 폴더에 저장하고 설치는 activate-space-model.py로.

학습 때 임계값은 후보 단위 점수(tp-2fp)로 골랐지만, 실제 SfM 복원의 공간 IoU는 더 낮은 임계값에서 좋았다
(2026-11-11). 실제 SfM 학습+검증 분할(문으로 연결된 집)의 전체 파이프라인 평균 IoU로 고르고,
시험 분할은 선택에 쓰지 않는다. 동점이면 더 높은(보수적인) 임계값.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from spacecapture.property_evaluation import evaluate_graph
from spacecapture.property_graph import infer_property
from spacecapture.space_model import fingerprint, load_model


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT/"tools"/file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def write_model(folder, weights_path, meta, threshold):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"space-candidate-mlp-v1.npz"
    shutil.copyfile(weights_path, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    new = dict(meta, threshold=float(threshold), sha256=sha, fingerprint=fingerprint(sha, threshold), enabled_by_default=False)
    new.pop("activation", None)
    path.with_suffix(".json").write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    load_model(path)
    return path, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="가중치를 가져올 기존 실행 폴더")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[.3, .4, .5, .6, .7, .8, .89])
    parser.add_argument("--sims", default=str(ROOT/"artifacts/space-learning/runs/20261110-123639"), help="모의 시험 자료 폴더")
    parser.add_argument("--variants", type=int, default=3)
    args = parser.parse_args()
    src = Path(args.run); src = (src if src.is_absolute() else ROOT/src).resolve()
    weights_path = src/"model/space-candidate-mlp-v1.npz"
    _, meta = load_model(weights_path)
    reg = _load("reg", "experiment-regularize.py")
    tsm = _load("tsm", "train-space-model.py")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = ROOT/"artifacts/space-learning/runs"/f"{stamp}-threshold"; run.mkdir(parents=True)
    started = time.perf_counter()
    cases = {s: reg.load_cases(s) for s in ("train", "validation")}
    print({s: len(c) for s, c in cases.items()}, flush=True)
    sweep = []
    for th in args.thresholds:
        path, _ = write_model(run/"sweep"/f"{th:.2f}", weights_path, meta, th)
        row = {"threshold": th}
        for s, cs in cases.items():
            r = [evaluate_graph(infer_property(*c["args"], **c["kw"], small_room_model=path), c["truth"]) for c in cs]
            row[s] = {"houses": len(r), "mean_iou": round(float(np.mean([x["mean_space_iou"] for x in r])), 4),
                      "count_exact": round(float(np.mean([x["predicted_space_candidates"] == x["truth_spaces"] == x["matched_spaces"] for x in r])), 4),
                      "portal_recall": round(float(np.mean([x["portal_recall"] for x in r])), 4)}
        n = row["train"]["houses"]+row["validation"]["houses"]
        row["selection_iou"] = round((row["train"]["mean_iou"]*row["train"]["houses"]+row["validation"]["mean_iou"]*row["validation"]["houses"])/n, 4)
        sweep.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    best = max(sweep, key=lambda r: (r["selection_iou"], r["threshold"]))
    shutil.rmtree(run/"sweep")
    path, new = write_model(run/"model", weights_path, meta, best["threshold"])
    data = json.loads((src/"dataset.json").read_text(encoding="utf-8"))
    shutil.copyfile(src/"dataset.json", run/"dataset.json")
    sims = Path(args.sims)
    report = {"neural_training_performed": False, "weights_from": str(src.relative_to(ROOT)), "threshold": best["threshold"],
              "previous_threshold": meta["threshold"], "selection": "실제 SfM 학습+검증 분할(문으로 연결된 집) 전체 파이프라인 평균 공간 IoU",
              "sweep": sweep, "model_sha256": new["sha256"], "fingerprint": new["fingerprint"],
              "end_to_end_test": {"rule": tsm.end_to_end(sims, data, "test", False, args.variants),
                                  "model": tsm.end_to_end(sims, data, "test", path, args.variants)},
              "seconds": round(time.perf_counter()-started, 1)}
    (run/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sweep"}, ensure_ascii=False, indent=1))
    print(f"결과: {run.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
