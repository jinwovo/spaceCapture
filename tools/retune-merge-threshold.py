"""문 합치기 모델의 임계값만 다시 고른다(가중치 불변). 새 실행 폴더에 저장하고 설치는 activate-merge-model.py로.

실제 SfM 학습+검증 분할(문으로 연결된 집)의 전체 파이프라인 평균 공간 IoU로 고른다. 시험 분할은 선택에 쓰지 않는다.
동점이면 더 높은(합치기를 덜 하는) 임계값.
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
from spacecapture.space_model import MERGE_MODEL, PORTAL_ARCHITECTURE, PORTAL_FEATURES, _load_generic, merge_fingerprint


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT/"tools"/file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def write_model(folder, weights_path, meta, threshold):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"portal-merge-mlp-v1.npz"
    shutil.copyfile(weights_path, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    new = dict(meta, threshold=float(threshold), sha256=sha, fingerprint=merge_fingerprint(sha, threshold), enabled_by_default=False)
    new.pop("activation", None)
    path.with_suffix(".json").write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    _load_generic(path, PORTAL_FEATURES, PORTAL_ARCHITECTURE, "portal-merge-v1")
    return path, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="가중치를 가져올 기존 합치기 실행 폴더")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[.9, .93, .95, .97, .99])
    parser.add_argument("--sims", default=str(ROOT/"artifacts/space-learning/runs/20261110-123639"))
    parser.add_argument("--dataset", default=str(ROOT/"artifacts/space-learning/runs/20261111-123226/dataset.json"))
    parser.add_argument("--real-dataset", default=str(ROOT/"artifacts/space-learning/lh-v2-dataset.json"))
    parser.add_argument("--variants", type=int, default=2)
    args = parser.parse_args()
    src = Path(args.run); src = (src if src.is_absolute() else ROOT/src).resolve()
    weights_path = src/"model/portal-merge-mlp-v1.npz"
    _, meta = _load_generic(weights_path, PORTAL_FEATURES, PORTAL_ARCHITECTURE, "portal-merge-v1")
    reg = _load("reg", "experiment-regularize.py")
    tmm = _load("tmm", "train-merge-model.py")
    tmm.rsf.configure("v2")
    run = ROOT/"artifacts/merge-learning/runs"/(dt.datetime.now().strftime("%Y%m%d-%H%M%S")+"-threshold"); run.mkdir(parents=True)
    started = time.perf_counter()
    cases = {s: reg.load_cases(s) for s in ("train", "validation")}
    sweep = []
    for th in args.thresholds:
        path, _ = write_model(run/"sweep"/f"{th:.3f}", weights_path, meta, th)
        row = {"threshold": th}
        for s, cs in cases.items():
            r = [evaluate_graph(infer_property(*c["args"], **c["kw"], merge_model=path), c["truth"]) for c in cs]
            row[s] = {"houses": len(r), "mean_iou": round(float(np.mean([x["mean_space_iou"] for x in r])), 4),
                      "portal_recall": round(float(np.mean([x["portal_recall"] for x in r])), 4)}
        n = row["train"]["houses"]+row["validation"]["houses"]
        row["selection_iou"] = round((row["train"]["mean_iou"]*row["train"]["houses"]+row["validation"]["mean_iou"]*row["validation"]["houses"])/n, 4)
        sweep.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    best = max(sweep, key=lambda r: (r["selection_iou"], r["threshold"]))
    shutil.rmtree(run/"sweep")
    path, new = write_model(run/"model", weights_path, meta, best["threshold"])
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    real_data = json.loads(Path(args.real_dataset).read_text(encoding="utf-8"))
    sims = Path(args.sims)
    report = {"neural_training_performed": False, "weights_from": str(src.relative_to(ROOT)), "threshold": best["threshold"],
              "previous_threshold": meta["threshold"], "selection": "실제 SfM 학습+검증 분할(문으로 연결된 집) 전체 파이프라인 평균 공간 IoU",
              "sweep": sweep, "model_sha256": new["sha256"], "fingerprint": new["fingerprint"],
              "end_to_end_sim_test": {"keep": tmm.e2e_sim(sims, data, args.variants, False), "model": tmm.e2e_sim(sims, data, args.variants, path)},
              "end_to_end_real_test": {"keep": tmm.e2e_real(real_data, "test", False), "model": tmm.e2e_real(real_data, "test", path)}}
    report["seconds"] = round(time.perf_counter()-started, 1)
    (run/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sweep"}, ensure_ascii=False, indent=1))
    print(f"결과: {run.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
