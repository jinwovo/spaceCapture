"""허용된 구조 자료로 천장 후보 MLP를 실제 학습·검증한다. CPU/NumPy."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

from spacecapture.ceiling_model import FEATURES,VERSION,candidate_features,inference_fingerprint,predict,sigmoid,audit_dataset
from spacecapture.ceiling_records import (DATA_FILES,LEARNING,data_hashes,ensure_not_installed_location,load_split,
                                          metrics,model_metadata,new_run_dir,rel,run_model_path)

SPLIT_LABEL={"train":"학습","validation":"검증","test":"시험"}


def fingerprint(house):
    polygons=[np.array([[p["x"],p["z"]] for p in r["floorPolygon"]]) for r in house["rooms"]]
    points=np.vstack(polygons)
    center=points.min(0)
    scale=max(float(np.ptp(points,axis=0).max()),1e-5)
    variants=[]
    for swap in (False,True):
        for sx,sz in ((1,1),(-1,1),(1,-1),(-1,-1)):
            shaped=[((p-center)/scale)[:,::-1] if swap else (p-center)/scale for p in polygons]
            shaped=[p*[sx,sz] for p in shaped]
            low=np.vstack(shaped).min(0)
            # Vertex ordering and collinear extra vertices do not define a new template.
            canonical=[]
            for p in shaped:
                p=p-low
                q=cv2.approxPolyDP(p.astype(np.float32).reshape(-1,1,2),.001,True).reshape(-1,2)
                canonical.append(sorted(tuple(v) for v in np.round(q,3).tolist()))
            variants.append(json.dumps(sorted(canonical),separators=(',',':')))
    return hashlib.sha256(min(variants).encode()).hexdigest()


def cloud(house,seed,variant):
    rng=np.random.default_rng(seed)
    H=max(p["y"] for w in house["walls"] for p in w["polygon"])
    factor=float(rng.uniform(.65,1.4)); target=float(np.clip(H*factor,1.85,6.0))
    polygons=[np.array([[p["x"],p["z"]] for p in r["floorPolygon"]],np.float32) for r in house["rooms"]]
    floor=np.vstack(polygons); low=floor.min(0); high=floor.max(0)
    density=float(rng.uniform(4,18))
    points,labels=[],[]
    seen=set()
    for wall in house["walls"]:
        if wall.get("empty"): continue
        a,b=wall["polygon"][:2]
        p0,p1=np.array([a["x"],a["z"]]),np.array([b["x"],b["z"]])
        key=tuple(sorted((tuple(np.round(p0,4)),tuple(np.round(p1,4)))))
        if key in seen: continue
        seen.add(key)
        count=max(15,int(np.linalg.norm(p1-p0)*target*density))
        t=rng.random(count)
        xz=p0+(p1-p0)*t[:,None]
        y=rng.uniform(0,target,count)
        # Posters/frames on several walls may form a misleading height peak.
        if variant>=2:
            stripe=rng.choice(count,max(3,count//6),replace=False)
            y[stripe]=min(target-.2,1.6+variant*.35)+rng.normal(0,.009,len(stripe))
        points.append(np.column_stack([xz[:,0],y,xz[:,1]])); labels.extend([0]*count)
    for polygon in polygons:
        plow,phigh=polygon.min(0),polygon.max(0)
        area=abs(cv2.contourArea(polygon))
        for ceiling in (False,True):
            if ceiling and variant==3: continue
            multiplier=float(rng.uniform(.15,.7)) if ceiling and variant==0 else float(rng.uniform(1.5,5))
            count=max(12,int(area*multiplier))
            xz=rng.uniform(plow,phigh,(count*3,2))
            xz=xz[[cv2.pointPolygonTest(polygon,tuple(p),False)>0 for p in xz]][:count]
            if ceiling and variant==2:
                xz=xz[xz[:,0]<plow[0]+.22*(phigh[0]-plow[0])]
            points.append(np.column_stack([xz[:,0],np.full(len(xz),target if ceiling else 0.),xz[:,1]]))
            labels.extend([int(ceiling)]*len(xz))
    # Furniture tops are horizontal but not necessarily ceilings.
    for _ in range(2):
        center=rng.uniform(low,high)
        half=rng.uniform(.3,1.3,2)
        xz=center+rng.uniform(-half,half,(80,2))
        y=float(rng.uniform(.8,min(3.1,target-.12)))
        points.append(np.column_stack([xz[:,0],np.full(len(xz),y),xz[:,1]])); labels.extend([0]*len(xz))
    P=np.vstack(points); label=np.array(labels,np.uint8)
    if len(P)>6500:
        keep=rng.choice(len(P),6500,replace=False); P,label=P[keep],label[keep]
    noise=float(rng.choice([.003,.008,.015,.025]))
    P+=rng.normal(0,noise,P.shape)
    count=max(5,len(P)//100)
    P=np.vstack([P,rng.uniform([low[0],0,low[1]],[high[0],target+.8,high[1]],(count,3))])
    label=np.r_[label,np.zeros(count,np.uint8)]
    camera_top=float(rng.uniform(1.1,min(1.65,target-.3)))
    box=(float(low[0]),float(high[0]),float(low[1]),float(high[1]))
    return P,label,camera_top,box,target


def select_houses(limit):
    directory=ROOT/"samples/external/procthor-10k"
    families=sorted({json.loads(line)["metadata"]["roomSpecId"] for line in gzip.open(directory/"val.jsonl.gz","rt")},
                    key=lambda s:hashlib.sha256(("ceiling-v1/"+s).encode()).hexdigest())
    groups={"train":families[:10],"validation":families[10:13],"test":families[13:]}
    result={}; used=set(); sources={}
    for split,upstream in (("train","train"),("validation","val"),("test","test")):
        manifest=directory/("download.json" if upstream=="val" else f"download-{upstream}.json")
        source=json.loads(manifest.read_text(encoding="utf-8"))
        if source["license"]!="Apache-2.0": raise ValueError("허용한 원본 라이선스가 아니에요.")
        if hashlib.sha256((directory/f"{upstream}.jsonl.gz").read_bytes()).hexdigest()!=source["sha256"]:
            raise ValueError("학습 원본 해시가 다운로드 기록과 달라요.")
        counts=Counter(); examples=[]
        with gzip.open(directory/f"{upstream}.jsonl.gz","rt",encoding="utf-8") as stream:
            for index,line in enumerate(stream):
                house=json.loads(line); family=house["metadata"]["roomSpecId"]
                if family not in groups[split] or counts[family]>=limit: continue
                signature=fingerprint(house)
                if signature in used: continue
                used.add(signature); counts[family]+=1
                examples.append((f"{upstream}-{index:05d}",family,signature,house))
        result[split]=examples; sources[split]=source
        print(f"{SPLIT_LABEL[split]}: 구조 유형 {len(counts)}개, 집 {len(examples)}개 선택",flush=True)
    return result,groups,sources


def prepare(limit,output):
    houses,groups,sources=select_houses(limit)
    datasets={}; manifest={"feature_version":VERSION,"groups":groups,"sources":sources,"houses":{},
        "split_unit":"upstream_split + roomSpecId + normalized_plan_fingerprint",
        "real_photo_training":False,"description":"구조에서 직접 샘플링한 노이즈 점군. 실제 SfM 학습 자료가 아닙니다."}
    for split,examples in houses.items():
        features,targets,clouds,heights,truths,descriptors=[],[],[],[],[],[]
        manifest["houses"][split]=[]
        for index,(identifier,family,signature,house) in enumerate(examples):
            manifest["houses"][split].append({"id":identifier,"family":family,"plan_fingerprint":signature})
            for variant in range(4):
                seed=int(hashlib.sha256(f"{identifier}/{variant}".encode()).hexdigest()[:8],16)
                P,labels,top,box,H=cloud(house,seed,variant)
                candidates,X=candidate_features(P,top,box)
                cid=len(descriptors)
                y=np.array([int(abs(c["pos"]-H)<.08 and labels[c["point_indices"]].mean()>=.5) for c in candidates],np.uint8)
                descriptors.append({"id":identifier,"variant":variant,"truth_height":H,"camera_top":top,
                                    "detectable":bool(y.any())})
                features.extend(X); targets.extend(y); clouds.extend([cid]*len(y))
                heights.extend(c["pos"] for c in candidates); truths.extend([H]*len(y))
            if index%8==0: print(f"  {SPLIT_LABEL[split]} 특징 생성: {index+1}/{len(examples)}집",flush=True)
        datasets[split]={"X":np.asarray(features,np.float32).reshape(-1,len(FEATURES)),"y":np.array(targets,np.float32),
                         "cloud":np.array(clouds,int),"height":np.array(heights),"truth":np.array(truths),
                         "descriptors":descriptors}
        data=datasets[split]
        np.savez_compressed(output/f"{split}.npz",**{k:v for k,v in data.items() if k!="descriptors"})
        (output/f"{split}-clouds.json").write_text(json.dumps(descriptors,ensure_ascii=False),encoding="utf-8")
        print(f"{SPLIT_LABEL[split]}: 후보 {len(data['y'])}개, 천장 정답 {int(data['y'].sum())}개",flush=True)
    (output/"dataset.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    audit_dataset(manifest)
    return datasets,manifest


def train(data,validation,epochs,log=print):
    rng=np.random.default_rng(17)
    X,y=data["X"],data["y"]
    weights={"mean":X.mean(0),"std":np.maximum(X.std(0),1e-3)}
    for key,shape in (("w1",(len(FEATURES),32)),("w2",(32,16)),("w3",(16,1))):
        weights[key]=(rng.standard_normal(shape)*np.sqrt(2/shape[0])).astype(np.float32)
        weights["b"+key[1:]]=np.zeros(shape[1],np.float32)
    keys=["w1","b1","w2","b2","w3","b3"]
    m={k:np.zeros_like(weights[k]) for k in keys}; v={k:np.zeros_like(weights[k]) for k in keys}
    normalized=np.clip((X-weights["mean"])/weights["std"],-8,8)
    ratio=max(1,float(np.sum(y==0))/max(1,np.sum(y==1)))
    step=0; best_loss=float('inf'); best=None; history=[]; stale=0
    for epoch in range(epochs):
        order=rng.permutation(len(y))
        for start in range(0,len(y),256):
            ids=order[start:start+256]; x=normalized[ids]; target=y[ids]
            a=np.maximum(0,x@weights["w1"]+weights["b1"])
            b=np.maximum(0,a@weights["w2"]+weights["b2"])
            score=sigmoid((b@weights["w3"]+weights["b3"]).reshape(-1))
            sample_weight=np.where(target>0,ratio,1).astype(np.float32)
            dz=((score-target)*sample_weight/sample_weight.sum())[:,None]
            grads={"w3":b.T@dz,"b3":dz.sum(0)}
            db=(dz@weights["w3"].T)*(b>0)
            grads.update(w2=a.T@db,b2=db.sum(0))
            da=(db@weights["w2"].T)*(a>0)
            grads.update(w1=x.T@da,b1=da.sum(0))
            step+=1
            for k in keys:
                grad=grads[k]+(1e-4*weights[k] if k.startswith('w') else 0)
                m[k]=.9*m[k]+.1*grad; v[k]=.999*v[k]+.001*grad*grad
                weights[k]-=.0015*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        probabilities=np.clip(predict(validation["X"],weights),1e-6,1-1e-6)
        vy=validation["y"]
        loss=float(-.5*(np.log(probabilities[vy==1]).mean()+np.log(1-probabilities[vy==0]).mean()))
        history.append({"epoch":epoch+1,"validation_balanced_bce":loss})
        if loss<best_loss-1e-5:
            best_loss=loss; best={k:value.copy() for k,value in weights.items()}; stale=0
        else: stale+=1
        if epoch%10==0: log(f"  학습 {epoch+1}/{epochs}, 검증 손실 {loss:.4f}")
        if stale>=35: break
    return best,history


def main():
    parser=argparse.ArgumentParser(description="천장 후보 신경망의 CPU 학습·별도 구조 평가 (새 실행 폴더에 저장)")
    parser.add_argument("--houses-per-family",type=int,default=8)
    parser.add_argument("--epochs",type=int,default=180)
    parser.add_argument("--reuse-data",nargs="?",const=str(LEARNING),default=None,
                        help="기존 실행 폴더의 특징 자료를 복사해 재사용 (기본: 현재 v1 기록 폴더)")
    parser.add_argument("--run-dir",default=None,help="결과를 저장할 새 실행 폴더 (기본: artifacts/ceiling-learning/runs/시각)")
    args=parser.parse_args()
    if not 1<=args.houses_per_family<=50 or not 1<=args.epochs<=1000:
        parser.error("유형별 집 수는 1~50, 학습 횟수는 1~1000이어야 해요.")
    if args.run_dir:
        output=Path(args.run_dir)
        if output.exists() and any(output.iterdir()): parser.error("실행 폴더가 비어 있지 않아요. 새 폴더를 지정해 주세요.")
        output.mkdir(parents=True,exist_ok=True)
    else: output=new_run_dir()
    model=run_model_path(output)
    ensure_not_installed_location(model)
    print(f"실행 폴더: {rel(output)} (현재 설치 모델은 바꾸지 않아요)",flush=True)
    started=time.perf_counter()
    if args.reuse_data:
        source=Path(args.reuse_data)
        data_hashes(source)
        for name in DATA_FILES: shutil.copyfile(source/name,output/name)
        manifest=json.loads((output/"dataset.json").read_text(encoding="utf-8"))
        datasets={split:load_split(output,split) for split in ("train","validation","test")}
    else: datasets,manifest=prepare(args.houses_per_family,output)
    split_audit=audit_dataset(manifest)
    prepared=time.perf_counter()
    weights,history=train(datasets["train"],datasets["validation"],args.epochs,lambda s:print(s,flush=True))
    validation_scores=predict(datasets["validation"]["X"],weights)
    choices=[]
    for threshold in np.linspace(.5,.9999,180):
        score=metrics(datasets["validation"],validation_scores,float(threshold))
        if score["wrong_detection_rate"]<=.03:
            choices.append((score["correct_detections"],-score["wrong_detections"],float(threshold),score))
    if not choices:
        threshold=1.; validation_metric=metrics(datasets["validation"],validation_scores,threshold)
    else: _,_,threshold,validation_metric=max(choices)
    test_scores=predict(datasets["test"]["X"],weights)
    report={"neural_training_performed":True,"architecture":[len(FEATURES),32,16,1],"optimizer":"Adam",
        "device":"CPU / NumPy","preparation_seconds":round(prepared-started,3),
        "training_seconds":round(time.perf_counter()-prepared,3),"epochs":len(history),"threshold":threshold,
        "feature_version":VERSION,"features":FEATURES,
        "validation":validation_metric,"test":metrics(datasets["test"],test_scores,threshold),
        "legacy_test":metrics(datasets["test"],np.ones(len(test_scores)),.5,legacy=True),
        "uncapped_geometry_test":metrics(datasets["test"],np.ones(len(test_scores)),.5),
        "real_photo_cases":0,"dataset_split":manifest["split_unit"],"split_audit":split_audit,
        "data_files":data_hashes(output),"data_files_recorded_by":"training","history":history}
    model.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(model,**weights)
    meta=model_metadata(model,threshold,manifest,output/"result.json")
    model.with_suffix(".json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    report["model_sha256"]=meta["sha256"]
    report["inference_fingerprint"]=inference_fingerprint(meta["id"],meta["sha256"],threshold)
    shutil.copyfile(ROOT/"samples/external/procthor-10k/LICENSE",model.parent/"LICENSE.procthor")
    (output/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="history"},ensure_ascii=False,indent=2))
    print(f"학습 결과를 {rel(output)}에 저장했어요. 설치 전 validate-ceiling-model.py --run, activate-ceiling-model.py --run으로 검사해 주세요.")


if __name__=="__main__": main()
