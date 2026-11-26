"""COCO val2017 중 상업 이용 가능 라이선스(CC BY / CC BY-SA / 저작권 제한 없음 / 미국 정부)이면서
실내 주요 사물이 2종 이상 있는 이미지만 받아 검출기 평가에 쓴다. 학습에는 쓰지 않는다."""
import collections
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"samples/external/coco"
KEY={"bed","toilet","couch","tv","dining table","oven","refrigerator","sink","microwave","chair"}
ALLOWED={4,5,7,8}


def main():
    d=json.loads((FOLDER/"instances_val2017.json").read_text(encoding="utf-8"))
    cats={c["id"]:c["name"] for c in d["categories"]}
    licenses={l["id"]:l for l in d["licenses"]}
    per=collections.defaultdict(set)
    for a in d["annotations"]:
        if not a.get("iscrowd"): per[a["image_id"]].add(cats[a["category_id"]])
    chosen=[im for im in d["images"] if im["license"] in ALLOWED and len(per[im["id"]]&KEY)>=2]
    out=FOLDER/"val-indoor"; out.mkdir(exist_ok=True)
    records=[]; started=time.perf_counter(); total=0
    for im in chosen:
        path=out/im["file_name"]
        if not path.exists():
            request=urllib.request.Request(im["coco_url"],headers={"User-Agent":"spaceCapture-evaluation"})
            with urllib.request.urlopen(request,timeout=60) as response: data=response.read()
            path.write_bytes(data)
        data=path.read_bytes(); total+=len(data)
        records.append({"id":im["id"],"file":f"val-indoor/{im['file_name']}","sha256":hashlib.sha256(data).hexdigest(),
                        "license":licenses[im["license"]]["name"],"license_url":licenses[im["license"]]["url"],
                        "flickr_url":im.get("flickr_url"),"width":im["width"],"height":im["height"]})
    manifest={"source":"COCO 2017 val (images.cocodataset.org), annotations CC BY 4.0","selection":"상업 이용 가능 라이선스 + 실내 주요 사물 2종 이상",
              "purpose":"사물 검출기 평가 전용. 학습·재배포에 쓰지 않음","images":records,"bytes":total,
              "seconds":round(time.perf_counter()-started,1),
              "annotations_zip_sha256":hashlib.sha256((FOLDER/"annotations_trainval2017.zip").read_bytes()).hexdigest()}
    (FOLDER/"indoor-manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=1),encoding="utf-8")
    print(f"{len(records)}장, {total:,}바이트, {manifest['seconds']}초")


if __name__=="__main__": main()
