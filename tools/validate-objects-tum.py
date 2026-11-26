"""TUM RGB-D(CC BY 4.0) 실제 사무실 영상으로 사물 3D 위치 추정을 검증한다.

추론: 선택한 RGB 프레임만 기존 photos 파이프라인에 넣는다(깊이·정답 궤적 미사용).
기준: 같은 프레임의 깊이 영상과 정답 카메라 궤적으로 검출 상자의 3D 위치를 계산한다.
같은 검출기를 쓰므로 검출 정확도가 아니라 'SfM 점으로 계산한 3D 위치'를 평가한다.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from spacecapture.objects import Detector,fuse_instances,locate_detection
from spacecapture.pipeline import umeyama

SEQ=ROOT/"samples/external/tum-rgbd/rgbd_dataset_freiburg3_long_office_household"
OUT=ROOT/"artifacts/object-validation"
FX,FY,CX,CY,DEPTH_SCALE=535.4,539.2,320.1,247.6,5000.


def table(name):
    rows=[]
    for line in (SEQ/name).read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        parts=line.split(); rows.append((float(parts[0]),parts[1:]))
    return rows


def nearest(times,values,t,tolerance=.03):
    k=int(np.argmin(np.abs(times-t)))
    return values[k] if abs(times[k]-t)<=tolerance else None


def quat_to_matrix(qx,qy,qz,qw):
    return np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])


def select_frames(count):
    rgb=table("rgb.txt"); gt=table("groundtruth.txt"); depth=table("depth.txt")
    gt_t=np.array([t for t,_ in gt]); d_t=np.array([t for t,_ in depth])
    usable=[(t,v[0]) for t,v in rgb if nearest(gt_t,gt,t) is not None and nearest(d_t,depth,t) is not None]
    chosen=[]
    for window in np.array_split(np.arange(len(usable)),count):
        best=max(window,key=lambda i:cv2.Laplacian(cv2.imread(str(SEQ/usable[i][1]),cv2.IMREAD_GRAYSCALE),cv2.CV_64F).var())
        chosen.append(usable[best])
    return chosen,gt_t,gt,d_t,depth


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--frames",type=int,default=150)
    parser.add_argument("--scene",default="tum-long-office")
    parser.add_argument("--skip-pipeline",action="store_true")
    args=parser.parse_args()
    started=time.perf_counter()
    frames,gt_t,gt,d_t,depth=select_frames(args.frames)
    photos=OUT/"tum-photos"
    if not args.skip_pipeline:
        shutil.rmtree(photos,ignore_errors=True); photos.mkdir(parents=True)
        for t,rel in frames: shutil.copyfile(SEQ/rel,photos/f"{t:.6f}.png")
        import subprocess
        subprocess.run([sys.executable,"-m","spacecapture","photos",str(photos),"--name",args.scene,"--no-splat"],cwd=ROOT,check=True)
    scene=json.loads((ROOT/"scenes"/args.scene/"scene.json").read_text(encoding="utf-8"))
    detector=Detector()
    reference=[]; gt_pose={}
    for t,rel in frames:
        g=nearest(gt_t,gt,t); d=nearest(d_t,depth,t)
        tx,ty,tz,qx,qy,qz,qw=map(float,g[1]); R=quat_to_matrix(qx,qy,qz,qw); C=np.array([tx,ty,tz])
        gt_pose[f"{t:.6f}.png"]=C
        image=cv2.imread(str(SEQ/rel)); dimg=cv2.imread(str(SEQ/d[1][0]),cv2.IMREAD_UNCHANGED).astype(np.float32)/DEPTH_SCALE
        ys,xs=np.nonzero(dimg>0); z=dimg[ys,xs]
        cam=np.stack([(xs-CX)/FX*z,(ys-CY)/FY*z,z],1); world=cam@R.T+C
        uv=np.stack([xs,ys],1).astype(float)
        for det in detector.detect(image):
            spot=locate_detection(det["box"],world,uv,z,min_points=30)
            if spot: reference.append({"class":det["class"],"score":det["score"],"photo":f"{t:.6f}.png",**spot})
    ref=fuse_instances(reference)
    cams=[c for c in scene["cameras"] if c.get("source_photo") in gt_pose]
    T=np.array([gt_pose[c["source_photo"]] for c in cams]); M=np.array([c["position"] for c in cams])
    s,R,t=umeyama(T,M)
    camera_error=np.linalg.norm(M-(s*T@R.T+t),axis=1)/s
    to_gt=lambda X:((np.asarray(X)-t)@R)/s
    ours=scene.get("objects",{}).get("instances",[])
    pairs=[]
    used=set()
    for o in sorted(ours,key=lambda o:-o["view_count"]):
        p=to_gt(o["position"])
        cands=[(np.linalg.norm(p-np.array(r["position"])),k) for k,r in enumerate(ref) if r["class"]==o["class"] and k not in used]
        if cands:
            e,k=min(cands)
            if e<=1.0:
                used.add(k); pairs.append({"class":o["class"],"label":o["label"],"error_cm":round(float(e)*100,1),
                                           "views_ours":o["view_count"],"views_reference":ref[k]["view_count"]}); continue
        pairs.append({"class":o["class"],"label":o["label"],"error_cm":None,"views_ours":o["view_count"]})
    matched=[p for p in pairs if p["error_cm"] is not None]
    report={"sequence":"TUM RGB-D fr3/long_office_household (CC BY 4.0)","frames":len(frames),
            "registered":scene["quality"]["registered"],"camera_alignment_rmse_cm":round(float(np.sqrt(np.mean(camera_error**2)))*100,2),
            "scale_vs_truth":round(float(s),4),"reference_instances":len(ref),"our_instances":len(ours),
            "matched":len(matched),"unmatched_ours":len(ours)-len(matched),"missed_reference":len(ref)-len(matched),
            "median_error_cm":round(float(np.median([p["error_cm"] for p in matched])),1) if matched else None,
            "mean_error_cm":round(float(np.mean([p["error_cm"] for p in matched])),1) if matched else None,
            "pairs":pairs,"reference":[{k:r[k] for k in ("class","label","view_count","position")} for r in ref],
            "seconds":round(time.perf_counter()-started,1),"pipeline_timing_s":scene.get("timing_s"),
            "objects_seconds":scene.get("objects",{}).get("seconds"),
            "scope":"실제 사무실 영상. 위치 비교는 카메라 궤적으로 배율·좌표를 맞춘 뒤(평가 전용) 같은 종류 사물 1m 이내 대응. 깊이 기준도 검출 상자 앞쪽 점의 중앙값이라 사물 중심의 정답이 아님."}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"tum-long-office.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("pairs","reference")},ensure_ascii=False))
    for p in pairs: print("  ",p)


if __name__=="__main__": main()
