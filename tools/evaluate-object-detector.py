"""COCO val2017 상업 이용 가능 실내 이미지로 사물 검출기의 클래스별 AP50을 잰다 (평가 전용)."""
import collections
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from spacecapture.objects import Detector,KOREAN

KEY=["bed","toilet","couch","tv","dining table","oven","refrigerator","sink","microwave","chair","potted plant","laptop"]


def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1]); x1,y1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0)
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def ap(records,n_truth):
    if n_truth==0: return None
    records=sorted(records,key=lambda r:-r[0]); tp=np.cumsum([r[1] for r in records]); fp=np.cumsum([1-r[1] for r in records])
    recall=tp/n_truth; precision=tp/np.maximum(tp+fp,1e-9)
    total=0.
    for t in np.linspace(0,1,101):
        p=precision[recall>=t]; total+=p.max() if len(p) else 0
    return total/101


def main():
    folder=ROOT/"samples/external/coco"
    d=json.loads((folder/"instances_val2017.json").read_text(encoding="utf-8"))
    manifest=json.loads((folder/"indoor-manifest.json").read_text(encoding="utf-8"))
    cats={c["id"]:c["name"] for c in d["categories"]}
    truth=collections.defaultdict(list)
    ids={r["id"] for r in manifest["images"]}
    for a in d["annotations"]:
        if a["image_id"] in ids and cats[a["category_id"]] in KEY:
            x,y,w,h=a["bbox"]; truth[(a["image_id"],cats[a["category_id"]])].append({"box":[x,y,x+w,y+h],"crowd":a.get("iscrowd",0),"used":False})
    detector=Detector()
    records=collections.defaultdict(list); started=time.perf_counter()
    for r in manifest["images"]:
        image=cv2.imread(str(folder/r["file"]))
        for det in detector.detect(image,score_threshold=.05,classes=set(KEY)):
            cands=truth.get((r["id"],det["class"]),[])
            best,bi=0,-1
            for k,t in enumerate(cands):
                v=iou(det["box"],t["box"])
                if v>best: best,bi=v,k
            if best>=.5 and not cands[bi]["used"] and not cands[bi]["crowd"]:
                cands[bi]["used"]=True; records[det["class"]].append((det["score"],1))
            elif best>=.5 and cands[bi]["crowd"]: continue
            else: records[det["class"]].append((det["score"],0))
    seconds=time.perf_counter()-started
    result={}
    for c in KEY:
        n=sum(1 for (i,k),v in truth.items() if k==c for t in v if not t["crowd"])
        a=ap(records[c],n)
        result[c]={"label":KOREAN.get(c,c),"truth_boxes":n,"ap50":round(a,3) if a is not None else None}
    valid=[v["ap50"] for v in result.values() if v["ap50"] is not None]
    report={"images":len(manifest["images"]),"seconds":round(seconds,1),"per_image_s":round(seconds/len(manifest["images"]),3),
            "map50_indoor":round(float(np.mean(valid)),3),"per_class":result,"model":detector.meta["id"],"model_sha256":detector.meta["sha256"],
            "scope":"COCO val2017 중 상업 이용 가능 라이선스 실내 이미지 138장. 공개 가중치의 학습 분할과 겹치지 않는 표준 검증 분할. 국내 매물 사진 성능이 아님."}
    out=ROOT/"artifacts/object-detection"; out.mkdir(parents=True,exist_ok=True)
    (out/"coco-indoor.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="per_class"},ensure_ascii=False))
    for c,v in result.items(): print(f'  {v["label"]:8s} 정답 {v["truth_boxes"]:3d}  AP50 {v["ap50"]}')


if __name__=="__main__": main()
