"""정답 공간·문과 추론 공간·문을 정답 좌표에 겹쳐 그린다 (평가용, 카메라 정렬)."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from spacecapture.pipeline import umeyama


def main(scene_name,truth_folder,output):
    g=json.loads((ROOT/truth_folder/"property.gt.json").read_text(encoding="utf-8")); ref={p["id"]:p for p in g["photos"]}
    s=json.loads((ROOT/"scenes"/scene_name/"scene.json").read_text(encoding="utf-8"))
    pairs=[(ref[c["source_photo"]]["position"],c["position"]) for c in s["cameras"] if c.get("source_photo") in ref]
    T=np.array([p[0] for p in pairs]); M=np.array([p[1] for p in pairs]); sc,R,t=umeyama(T,M)
    inv=lambda xz:(((np.array([xz[0],0,xz[1]])-t)@R)/sc)[[0,2]]
    allp=np.vstack([np.array(sp["polygon_xz"]) for sp in g["spaces"]])
    low=allp.min(0)-.8; high=allp.max(0)+.8; k=900/max(high-low)
    img=np.full((int((high-low)[1]*k),int((high-low)[0]*k),3),255,np.uint8)
    P=lambda xz:tuple(((np.asarray(xz)-low)*k).astype(int))
    for sp in g["spaces"]:
        cv2.polylines(img,[np.array([P(p) for p in sp["polygon_xz"]],np.int32)],True,(0,160,0),4)
        cv2.putText(img,sp["id"],P(np.mean(sp["polygon_xz"],0)+[0,.3]),0,.6,(0,120,0),2)
    for d in g["portals"]:
        a,b=d["endpoints_xz"]; cv2.line(img,P(a),P(b),(0,140,255),8)
    for sp in s["property_graph"]["spaces"]:
        cv2.polylines(img,[np.array([P(inv(p)) for p in sp["polygon_xz"]],np.int32)],True,(255,0,0),2)
        cv2.putText(img,sp["id"],P(inv(sp["center_xz"])),0,.55,(255,0,0),2)
    for p in s["property_graph"]["portals"]:
        cv2.circle(img,P(inv(p["center_xz"])),8,(0,0,255) if p["status"]=="image_supported" else (150,150,150),-1)
    for c in T: cv2.circle(img,P(c[[0,2]]),2,(0,0,0),-1)
    cv2.putText(img,"green: truth  blue: inferred  orange: truth door  red: inferred door",(10,20),0,.5,(0,0,0),1)
    cv2.imencode(".png",img)[1].tofile(output)


if __name__=="__main__": main(*sys.argv[1:4])
