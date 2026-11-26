"""공간 용도 모델을 검사한 뒤에만 설치한다. 실패하면 설치하지 않는다(용도 추정 없음 = 기존 동작)."""
import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spacecapture.room_types import MODEL, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run = Path(args.run); run = (run if run.is_absolute() else ROOT/run).resolve(); path = run/"model/room-type-mlp-v1.npz"
    _, meta = load_model(path)
    report = json.loads((run/"result.json").read_text(encoding="utf-8"))
    problems = []
    if report["fingerprint"] != meta["fingerprint"]: problems.append("학습 보고서와 모델 지문이 달라요.")
    test = report["test"]
    if test["model_all"]["accuracy"] < test["rule"]["accuracy"]: problems.append("시험 정확도가 비교 규칙보다 낮아요.")
    if test["model_decided"]["accuracy"] < .8: problems.append("확정 표시 공간의 시험 정확도가 0.8 미만이에요.")
    if report["cross_validation"]["accuracy"] < .7: problems.append("교차검증 정확도가 0.7 미만이에요.")
    record = {"run": str(run.relative_to(ROOT)), "fingerprint": meta["fingerprint"], "threshold": meta["threshold"],
              "test": {k: {"accuracy": v["accuracy"], "coverage": v["coverage"]} for k, v in test.items()},
              "cross_validation": report["cross_validation"], "problems": problems,
              "checked_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    (run/"activation-check.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=1))
    if problems:
        print("활성화 거절"); sys.exit(1)
    if args.check_only:
        print("검사 통과(설치하지 않음)"); return
    if MODEL.exists():
        backup = ROOT/"artifacts/room-type-learning/previous-models"/dt.datetime.now().strftime("%Y%m%d-%H%M%S"); backup.mkdir(parents=True)
        for p in (MODEL, MODEL.with_suffix(".json")): shutil.copy2(p, backup/p.name)
    meta.update(enabled_by_default=True, activation=str((run/"activation-check.json").relative_to(ROOT)))
    tmp = MODEL.with_suffix(".tmp"); shutil.copyfile(path, tmp); tmp.replace(MODEL)
    MODEL.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    load_model(MODEL)
    print("검사를 통과한 공간 용도 모델을 설치했어요.")


if __name__ == "__main__":
    main()
