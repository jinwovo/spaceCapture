"""외부 합성 구조 기반 사진 복원을 정답과 별도로 비교하고 수치를 보존한다."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacecapture.property_evaluation import evaluate_scene_property


def main():
    parser=argparse.ArgumentParser(description="외부 합성 구조의 복원 결과 비교")
    parser.add_argument("--stage",choices=["baseline","current"],default="current")
    args=parser.parse_args()
    output=ROOT/"artifacts/procthor-validation"
    preparation=json.loads((output/"preparation.json").read_text(encoding="utf-8"))
    report={"stage":args.stage,"external_synthetic_cases":2,"real_photo_cases":0,
            "neural_training_performed":False,"cases":[],
            "limitation":"자체 재질·촬영 위치의 합성 사진입니다. 실제 매물 정확도 또는 미사용 테스트 집의 일반화 점수가 아닙니다."}
    for case in preparation["selected"]:
        index=case["index"]
        folder=output/f"val-{index:04d}"
        scene_name=f"procthor-val-{index:04d}"
        scene=json.loads((ROOT/"scenes"/scene_name/"scene.json").read_text(encoding="utf-8"))
        truth=json.loads((folder/"property.gt.json").read_text(encoding="utf-8"))
        started=time.perf_counter()
        result=evaluate_scene_property(scene,truth,output=folder/f"comparison-{args.stage}")
        report["cases"].append({"scene":scene_name,"input_photos":case["photos"],
            "prepared_photos":scene["quality"]["frames"],"registered":scene["quality"]["registered"],
            "pipeline_seconds":scene["timing_s"],"floor_found":scene["floor_found"],
            "ceiling":{"measured_m":scene["room"]["height"],"source":scene["room"]["height_source"],
                       "truth_m":truth["size"][1],"scale_assumed":True},
            "comparison_seconds":round(time.perf_counter()-started,4),**result})
        print(f'{scene_name}: {scene["quality"]["registered"]}장 등록, '
              f'공간 {result["predicted_space_candidates"]}/{result["truth_spaces"]}, '
              f'IoU {result["mean_space_iou"]}, 문 연결 재현율 {result["portal_recall"]}')
    (output/f"{args.stage}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
