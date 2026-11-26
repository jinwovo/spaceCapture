"""실행 폴더의 모델을 실제 사진 SfM 점군에 적용하고, 추론 뒤에만 높이 정답과 비교.

기록에는 추론 설정 지문과 입력 장면(points/splat) 해시·상자·카메라 높이,
평가용 정답 파일 해시를 함께 남긴다. 설치 모델은 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from spacecapture.ceiling_model import classify_candidates,candidate_features,load_model
from spacecapture.ceiling_records import LEARNING,file_sha256,rel,run_model_path,scene_inputs
from spacecapture.plyio import read_vertices,splat_centers
from spacecapture.room import find_ceiling


def main():
    parser=argparse.ArgumentParser(description="천장 모델 사진 복원 회귀 검사")
    parser.add_argument("--run",default=str(LEARNING))
    args=parser.parse_args()
    run=Path(args.run); model=run_model_path(run)
    cases=[("synth",2.5),("procthor-val-0140",4.2257958761529775),
           ("procthor-val-0043",4.203950440134644),("property-rooms-corridor",2.6)]
    truth_files={}
    new_case=LEARNING/"photo-test.json"
    if new_case.exists():
        selected=json.loads(new_case.read_text(encoding="utf-8"))
        gt=ROOT/selected["folder"]/"property.gt.json"
        truth=json.loads(gt.read_text(encoding="utf-8"))
        if (ROOT/"scenes"/selected["scene"]/"scene.json").exists():
            cases.append((selected["scene"],truth["size"][1]))
            truth_files[selected["scene"]]={"path":rel(gt),"sha256":file_sha256(gt),
                                            "selection":{"path":rel(new_case),"sha256":file_sha256(new_case)}}
    _,metadata=load_model(model)
    report={"model":rel(model),"model_sha256":metadata["sha256"],"threshold":metadata["threshold"],
            "inference_fingerprint":metadata["inference_fingerprint"],"real_photo_cases":0,
            "purpose":"기존 사진 복원 결과의 개발 회귀 검사. 미사용 시험 점수와 구분.",
            "inputs":{},"truth_files":truth_files,"cases":[]}
    for name,truth in cases:
        folder=ROOT/"scenes"/name
        scene=json.loads((folder/"scene.json").read_text(encoding="utf-8"))
        report["inputs"][name]=scene_inputs(name)
        v=read_vertices(folder/"points.ply"); sparse=np.stack([v[k] for k in ('x','y','z')],1)
        box=(scene["room"]["min"][0],scene["room"]["max"][0],scene["room"]["min"][2],scene["room"]["max"][2])
        top=max(c["position"][1] for c in scene["cameras"])
        for kind,P in [("sparse",sparse)]+([("sparse+splat",np.vstack([sparse,splat_centers(folder/scene["files"]["splat"])]))] if scene["files"].get("splat") else []):
            started=time.perf_counter()
            ceiling,analysis=classify_candidates(P,top,box,model)
            geometry=find_ceiling(P,top,box)
            candidates,_=candidate_features(P,top,box)
            uncapped=candidates[0] if candidates else None
            result={"scene":name,"input":kind,"points":len(P),"truth_height_m":truth,
                    "model_height_m":ceiling["pos"] if ceiling else None,
                    "legacy_height_m":geometry["pos"] if geometry else None,
                    "uncapped_height_m":uncapped["pos"] if uncapped else None,
                    "error_cm":round((ceiling["pos"]-truth)*100,3) if ceiling else None,
                    "seconds":round(time.perf_counter()-started,4),"analysis":analysis}
            report["cases"].append(result)
            print(f'{name} / {kind}: 기존 {result["legacy_height_m"]}, 신경망 {result["model_height_m"]}, 오차 {result["error_cm"]}cm')
    (run/"sfm-regression.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"추론 설정 지문 {metadata['inference_fingerprint'][:16]}… 로 {len(report['inputs'])}개 장면을 기록했어요.")


if __name__=="__main__": main()
