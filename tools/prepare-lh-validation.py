"""LH 공공주택 평면도 1장을 평가용 정답·자체 렌더 사진으로 만든다.

축척은 주석 파일의 치수 사슬 합계(사람/모델이 읽은 값)와 자동 검출한 끝 눈금으로 정한다.
변환 결과는 겹침 이미지로 확인해야 한다. 정답은 추론에 전달하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from spacecapture.plan_image import meters_per_pixel,render_config,segment_plan
from spacecapture.synth_property import generate_property

DATA=ROOT/"samples/external/lh-floorplans"


def main():
    parser=argparse.ArgumentParser(description="LH 평면도 → 평가 정답·자체 렌더 사진")
    parser.add_argument("index",type=int)
    parser.add_argument("--no-render",action="store_true")
    parser.add_argument("--room-views",type=int,default=20)
    parser.add_argument("--suffix",default="")
    args=parser.parse_args()
    manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
    note=json.loads((DATA/f"annotations/plan-{args.index:04d}.json").read_text(encoding="utf-8"))
    plan=manifest["plans"][args.index]
    image=cv2.imdecode(np.fromfile(DATA/plan["file"],np.uint8),cv2.IMREAD_COLOR)
    started=time.perf_counter()
    mpp,scale=meters_per_pixel(image,note["scale"]["total_mm"]/1000,note["scale"]["side"])
    segmentation=segment_plan(image,mpp)
    config,excluded=render_config(segmentation,height=note["ceiling_height_m"],seed=args.index,room_views=args.room_views)
    out=ROOT/f"artifacts/lh-validation/plan-{args.index:04d}{args.suffix}"
    out.mkdir(parents=True,exist_ok=True)
    overlay=image.copy()
    for s in segmentation["spaces"]:
        poly=(np.array(s["polygon_xz"])/mpp).astype(np.int32)
        color=(255,0,0) if s["id"] in {t["id"] for t in config["spaces"]} else (160,160,160)
        cv2.polylines(overlay,[poly],True,color,max(2,image.shape[1]//400))
        cv2.putText(overlay,s["id"],tuple(poly.mean(0).astype(int)),0,image.shape[1]/1600,color,max(2,image.shape[1]//600))
    for p in segmentation["portals"]:
        a,b=(np.array(p["endpoints_xz"])/mpp).astype(int); cv2.line(overlay,tuple(a),tuple(b),(0,140,255),max(3,image.shape[1]//250))
    cv2.imencode(".jpg",overlay)[1].tofile(out/"overlay.jpg")
    metadata={"id":f"lh-plan-{args.index:04d}","license":"공공데이터포털 '이용허락범위 제한 없음' (LH 주택 평면도 현황)",
              "commercial_training_allowed":False,"origin":"public_plan_image_local_render",
              "source_dataset":"LH 주택 평면도 현황 (data.go.kr 15037046)","source":{"plan":plan,"archive_sha256":manifest["archive_sha256"]},
              "annotation":note,"room_views":args.room_views,"scale_detection":scale,"meters_per_pixel":mpp,"excluded_small_spaces":excluded,
              "render_modifications":{"materials":"자체 생성 텍스처","objects_omitted":True,"windows_sealed":True,
                                      "doors":"열린 개구부(높이 2.1m)","wall_thickness":"도면 벽 두께 유지"},
              "evaluation_only":True,"real_property_validation":False,
              "note":"실제 국내 공공주택 평면 구조를 자체 렌더링한 합성 사진입니다. 실제 매물 사진의 정확도가 아닙니다."}
    if args.no_render:
        print(json.dumps({"spaces":len(config["spaces"]),"portals":len(config["portals"]),"poses":len(config["poses"])},ensure_ascii=False)); return
    generate_property(out,kind=f"lh-{args.index}",seed=args.index,config=config,truth_metadata=metadata,log=print)
    print(f"준비 완료 {time.perf_counter()-started:.1f}초: {out}")


if __name__=="__main__": main()
