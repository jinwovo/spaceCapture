"""천장 후보의 3D 근거를 판별하는 작은 NumPy 신경망. CPU 전용 경로.

사진 의미 분류나 높이 회귀가 아니다. 후보 높이는 항상 실제 관측점에서
계산하며, 이 모듈에는 집의 도면·정답 높이를 전달하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry import histogram_peaks

FEATURES = ["log_support", "layer_fraction", "extent_x", "extent_z", "area_fraction",
            "height_std", "plane_residual", "normal_y_mean_12", "normal_y_p25_12",
            "horizontal_fraction_12", "planarity_12", "normal_y_mean_24", "normal_y_p25_24",
            "horizontal_fraction_24", "planarity_24", "neighbor_radius", "vertical_spread",
            "points_above_fraction", "peak_concentration", "height_above_camera", "interior_fraction"]
VERSION = 1
ARCHITECTURE = [len(FEATURES), 32, 16, 1]
FINGERPRINT_FORMAT = "ceiling-inference-v1"
DEFAULT_MODEL = Path(__file__).resolve().parents[1]/"models/ceiling-mlp-v1.npz"


def inference_fingerprint(model_id, weights_sha256, threshold):
    """가중치·임계값·특징 버전/순서·구조를 묶은 추론 설정 지문.

    가중치가 같아도 임계값이나 특징 순서가 바뀌면 판정 결과가 달라지므로
    검증 기록은 가중치 해시가 아니라 이 지문으로 연결한다.
    """
    if not isinstance(threshold,(float,int)) or isinstance(threshold,bool):
        raise ValueError("천장 모델의 판별 임계값이 올바르지 않아요.")
    payload={"format":FINGERPRINT_FORMAT,"model_id":str(model_id),"weights_sha256":str(weights_sha256),
             "threshold":float(threshold),"feature_version":VERSION,"features":FEATURES,
             "architecture":ARCHITECTURE,"input_clip":8}
    text=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def active_model():
    """설치된 모델 경로. 의도적으로 비활성화한 경우에만 None을 돌려준다.

    메타데이터가 없거나 손상됐는데 가중치가 남아 있으면 기존 높이층 방식으로
    조용히 돌아가지 않도록 경로를 돌려주고, 이후 검사 실패로 높이를 가정한다.
    """
    path=DEFAULT_MODEL.with_suffix(".json")
    if not path.exists():
        return DEFAULT_MODEL if DEFAULT_MODEL.exists() else None
    try:
        metadata=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata,dict): raise ValueError("잘못된 모델 기록")
    except (OSError,ValueError):
        return DEFAULT_MODEL
    return None if metadata.get("enabled_by_default") is False else DEFAULT_MODEL


def candidate_features(points, cam_top, box):
    """후보와 특징만 만든다. 정답 정보는 받지 않는다."""
    points=np.asarray(points,float).reshape(-1,3)
    points=points[np.isfinite(points).all(1)]
    x0,x1,z0,z1=box
    width,depth=x1-x0,z1-z0
    if len(points)<30 or min(width,depth)<=.1:
        return [],np.empty((0,len(FEATURES)),np.float32)
    above=points[points[:,1]>cam_top+.15]
    if len(above)<8:
        return [],np.empty((0,len(FEATURES)),np.float32)
    _,neighbors=cKDTree(points).query(points,k=min(24,len(points)),workers=1)
    normals=[]
    for count in (12,24):
        local=points[neighbors[:,:min(count,len(points))]]
        centered=local-local.mean(1,keepdims=True)
        eigen,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',centered,centered)/local.shape[1])
        ny=np.abs(vectors[:,1,0])
        planarity=(eigen[:,1]-eigen[:,0])/np.maximum(eigen[:,2],1e-10)
        normals.append((ny,planarity))
    local=points[neighbors]
    radius=np.linalg.norm(local[:,-1]-points,axis=1)
    vertical=np.ptp(local[:,:,1],axis=1)/np.maximum(radius,1e-5)
    centers,_=histogram_peaks(above[:,1],.02)
    result,features=[],[]
    for c in sorted(centers,reverse=True):
        if not cam_top+.15<c<=10.:
            continue
        mask=np.abs(points[:,1]-c)<.04
        distance=np.minimum.reduce([points[:,0]-x0,x1-points[:,0],points[:,2]-z0,z1-points[:,2]])
        inner=np.flatnonzero(mask & (distance>.25))
        if len(inner)<8:
            continue
        layer=points[inner]
        extent=np.percentile(layer[:,[0,2]],95,axis=0)-np.percentile(layer[:,[0,2]],5,axis=0)
        if np.any(extent<np.maximum(.8,.4*np.array([width,depth]))):
            continue
        design=np.column_stack([layer[:,0],layer[:,2],np.ones(len(layer))])
        fit=np.linalg.lstsq(design,layer[:,1],rcond=None)[0]
        residual=float(np.median(np.abs(layer[:,1]-design@fit)))
        f=[np.log1p(len(inner)),float(mask.sum()/len(above)),extent[0]/width,extent[1]/depth,
           float(np.prod(extent)/(width*depth)),float(np.std(layer[:,1])),residual]
        for ny,planarity in normals:
            f.extend([float(ny[inner].mean()),float(np.percentile(ny[inner],25)),
                      float(np.mean(ny[inner]>.85)),float(planarity[inner].mean())])
        surrounding=np.abs(points[:,1]-c)<.18
        f.extend([float(np.median(radius[inner])),float(np.median(vertical[inner])),
                  float(np.mean(above[:,1]>c+.12)),float(mask.sum()/max(1,surrounding.sum())),
                  float((c-cam_top)/max(cam_top,.5)),float(len(inner)/mask.sum())])
        result.append({"pos":float(np.median(layer[:,1])),"support":int(mask.sum()),"interior":len(inner),
                       "interior_extent_xz":extent.tolist(),"point_indices":inner})
        features.append(f)
    return result,np.asarray(features,np.float32).reshape(-1,len(FEATURES))


def sigmoid(value):
    return 1/(1+np.exp(-np.clip(value,-35,35)))


def predict(features,weights):
    x=np.clip((np.asarray(features,np.float32)-weights["mean"])/weights["std"],-8,8)
    a=np.maximum(0,x@weights["w1"]+weights["b1"])
    b=np.maximum(0,a@weights["w2"]+weights["b2"])
    return sigmoid((b@weights["w3"]+weights["b3"]).reshape(-1))


def load_model(path=DEFAULT_MODEL,require_active=False):
    """가중치와 기록을 검사해 읽는다. require_active는 설치·활성화된 모델만 허용한다."""
    path=Path(path)
    meta=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(meta,dict):
        raise ValueError("천장 모델 기록 형식이 올바르지 않아요.")
    if meta.get("feature_version")!=VERSION or meta.get("features")!=FEATURES:
        raise ValueError("천장 모델의 특징 형식이 현재 코드와 달라요.")
    threshold=meta.get("threshold")
    if not isinstance(threshold,(float,int)) or isinstance(threshold,bool) or not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError("천장 모델의 판별 임계값이 올바르지 않아요.")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta.get("sha256"):
        raise ValueError("천장 모델의 SHA-256이 기록과 달라요.")
    fingerprint=inference_fingerprint(meta.get("id"),meta["sha256"],threshold)
    if meta.get("inference_fingerprint")!=fingerprint:
        raise ValueError("천장 모델의 추론 설정 지문(가중치·임계값·특징)이 기록과 달라요.")
    if require_active:
        activation=meta.get("activation")
        if meta.get("enabled_by_default") is not True or not isinstance(activation,dict)                 or activation.get("inference_fingerprint")!=fingerprint:
            raise ValueError("현재 추론 설정이 활성화 검증 기록과 일치하지 않아요.")
    with np.load(path,allow_pickle=False) as data:
        weights={key:data[key].copy() for key in data.files}
    shapes={"mean":(len(FEATURES),),"std":(len(FEATURES),),"w1":(len(FEATURES),32),"b1":(32,),
            "w2":(32,16),"b2":(16,),"w3":(16,1),"b3":(1,)}
    if set(weights)!=set(shapes) or any(weights[k].shape!=v for k,v in shapes.items()):
        raise ValueError("천장 모델의 가중치 크기가 올바르지 않아요.")
    if any(not np.isfinite(v).all() for v in weights.values()) or np.any(weights["std"]<=0):
        raise ValueError("천장 모델의 가중치가 유효하지 않아요.")
    return weights,meta


def audit_dataset(manifest):
    """같은 집·구조 유형·정규화 도면이 여러 분할에 들어가는 것을 거절한다."""
    seen_ids,seen_families,seen_plans={},{},{}
    counts={}
    for split in ("train","validation","test"):
        houses=manifest["houses"][split]
        if not houses: raise ValueError("비어 있는 학습·검증·시험 분할이 있어요.")
        for house in houses:
            for field,seen in (("id",seen_ids),("family",seen_families),("plan_fingerprint",seen_plans)):
                value=house[field]
                if value in seen and seen[value]!=split:
                    raise ValueError(f"자료 분할 사이에 같은 {field}가 겹쳐요.")
                seen[value]=split
        counts[split]={"houses":len(houses),"families":len({h["family"] for h in houses})}
    return counts


def classify_candidates(points,cam_top,box,path=DEFAULT_MODEL,require_active=False):
    weights,meta=load_model(path,require_active)
    candidates,features=candidate_features(points,cam_top,box)
    scores=predict(features,weights)
    records=[]
    accepted=None
    for candidate,score in zip(candidates,scores):
        candidate={k:v for k,v in candidate.items() if k!="point_indices"}
        records.append(candidate|{"model_score":round(float(score),5)})
        if accepted is None and score>=meta["threshold"]:
            accepted=candidate|{"method":"geometry_mlp","model_id":meta["id"],"model_score":round(float(score),5)}
    return accepted,{"model_id":meta["id"],"model_sha256":meta["sha256"],"threshold":meta["threshold"],
                     "inference_fingerprint":meta["inference_fingerprint"],"candidates":records,
                     "real_photo_validated":False,"score_is_accuracy_probability":False,
                     "message":"합성 구조로 학습한 천장 후보 판별입니다. 점수는 실제 정확도 확률이 아닙니다."}
