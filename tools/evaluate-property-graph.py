"""저장된 SfM 복원으로 공간·문 후보를 현재 코드로 다시 계산해 정답과 비교한다.

장면 파일은 바꾸지 않는다. 정답은 추론 뒤 평가에만 쓴다(카메라 정렬 비교).
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
import pycolmap
from spacecapture.align import arrays_from_reconstruction
from spacecapture.property_evaluation import evaluate_scene_property
from spacecapture.property_graph import infer_property

CASES={
    "property-two-rooms":"artifacts/property-validation/two_rooms_photos",
    "property-rooms-corridor":"artifacts/property-validation/rooms_corridor_photos",
    "property-rooms-corridor-posefix":"artifacts/property-validation/rooms_corridor_photos",
    "procthor-val-0140":"artifacts/procthor-validation/val-0140",
    "procthor-val-0140-posefix":"artifacts/procthor-validation/val-0140",
    "procthor-val-0043":"artifacts/procthor-validation/val-0043",
    "procthor-val-0043-final":"artifacts/procthor-validation/val-0043",
    "procthor-val-0043-rep1":"artifacts/procthor-validation/val-0043",
    "procthor-val-0043-rep2":"artifacts/procthor-validation/val-0043",
    "procthor-val-0043-rep3":"artifacts/procthor-validation/val-0043",
    "procthor-test-0186-final":"artifacts/ceiling-learning/photo-test-0186",
    "lh-plan-0009":"artifacts/lh-validation/plan-0009",
    "lh-plan-0227":"artifacts/lh-validation/plan-0227",
    "lh-plan-0009-g1":"artifacts/lh-validation/plan-0009",
    "lh-plan-0009-g2":"artifacts/lh-validation/plan-0009",
    "lh-plan-0009-g3":"artifacts/lh-validation/plan-0009",
    "lh-plan-0227-g1":"artifacts/lh-validation/plan-0227",
    "lh-plan-0227-final":"artifacts/lh-validation/plan-0227",
    "lh-plan-0227-day2":"artifacts/lh-validation/plan-0227",
    "procthor-val-0043-g1":"artifacts/procthor-validation/val-0043",
    "property-rooms-corridor-g1":"artifacts/property-validation/rooms_corridor_photos",
}


def recompute(name,model=None,merge=None):
    folder=ROOT/"scenes"/name
    scene=json.loads((folder/"scene.json").read_text(encoding="utf-8"))
    pycolmap.logging.minloglevel=3
    aligned=arrays_from_reconstruction(pycolmap.Reconstruction(folder/"work/colmap/aligned"))
    keep=aligned.reliable(min_angle=6.)
    started=time.perf_counter()
    graph=infer_property(aligned.xyz[keep],aligned.centers,aligned.names,
        [o for o,k in zip(aligned.observations,keep) if k],planes=scene["layout"]["planes"],
        rotations=aligned.R_cw,intrinsics=aligned.intrinsics,small_room_model=model,merge_model=merge)
    return scene,graph,time.perf_counter()-started


def main():
    parser=argparse.ArgumentParser(description="공간·문 후보 재계산 비교")
    parser.add_argument("--label",required=True)
    parser.add_argument("--scenes",nargs="*",default=list(CASES))
    parser.add_argument("--model",default="installed",help="installed | rule | 모델 .npz 경로")
    args=parser.parse_args()
    report={"label":args.label,"ground_truth_used_for_inference":False,"cases":{}}
    for name in args.scenes:
        if not (ROOT/"scenes"/name/"work/colmap/aligned").exists():
            print(f"{name}: 복원 결과 없음, 건너뜀"); continue
        truth=json.loads((ROOT/CASES[name]/"property.gt.json").read_text(encoding="utf-8"))
        model=None if args.model=="installed" else False if args.model=="rule" else Path(args.model)
        scene,graph,seconds=recompute(name,model)
        scene["property_graph"]=graph
        result=evaluate_scene_property(scene,truth)
        report["cases"][name]={k:result[k] for k in ("predicted_space_candidates","truth_spaces","mean_space_iou",
            "portal_precision","portal_recall","truth_portals","image_supported_portals","geometry_only_portals",
            "portal_center_mean_error_cm","portal_width_mean_error_cm")}|{"pose_rmse_cm":result["alignment"].get("pose_rmse_cm"),"infer_seconds":round(seconds,3)}
        c=report["cases"][name]
        print(f'{name}: 공간 {c["predicted_space_candidates"]}/{c["truth_spaces"]} IoU {c["mean_space_iou"]} '
              f'문 P{c["portal_precision"]}/R{c["portal_recall"]} 중심 {c["portal_center_mean_error_cm"]}cm 폭 {c["portal_width_mean_error_cm"]}cm ({seconds:.1f}초)')
    out=ROOT/"artifacts/portal-placement"/f"{args.label}.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=float),encoding="utf-8")


if __name__=="__main__": main()
