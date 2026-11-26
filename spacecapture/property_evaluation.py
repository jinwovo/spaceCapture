"""Compare predicted observed spaces/openings against a separate paired plan."""
from __future__ import annotations

import copy

import cv2
import numpy as np


def evaluate_graph(prediction, truth, cell=.05):
    predicted, actual = prediction["spaces"], truth["spaces"]
    if len(actual)>15:
        raise ValueError("현재 정답 비교는 층별 공간 15개까지 지원해요. 층을 나눠 비교해 주세요.")
    vertices = [p for s in predicted + actual for p in s["polygon_xz"]]
    if not vertices:
        raise ValueError("비교할 공간 다각형이 없어요.")
    vertices = np.asarray(vertices)
    low, high = vertices.min(0)-.1, vertices.max(0)+.1
    step = max(cell, float(np.max(high-low))/1000)
    size = np.ceil((high-low)/step).astype(int)+1
    def raster(space):
        result = np.zeros((size[1],size[0]),np.uint8)
        polygon = np.round((np.array(space["polygon_xz"])-low)/step).astype(np.int32)
        cv2.fillPoly(result,[polygon],1)
        return result.astype(bool)
    pm, tm = [raster(s) for s in predicted], [raster(s) for s in actual]
    overlap = np.zeros((len(predicted),len(actual)))
    for i,p in enumerate(pm):
        for j,t in enumerate(tm):
            overlap[i,j] = np.sum(p&t)/max(1,np.sum(p|t))
    # Dynamic programming assignment avoids assuming room IDs/order are shared.
    states = {0:(0.,[])}
    for i in range(len(predicted)):
        updated = dict(states)
        for mask,(score,matches) in states.items():
            for j in range(min(len(actual),15)):
                if mask & (1<<j) or overlap[i,j]<.2: continue
                target=mask | (1<<j); candidate=(score+overlap[i,j],matches+[(i,j)])
                if target not in updated or candidate[0]>updated[target][0]: updated[target]=candidate
        states=updated
    _,matches=max(states.values(),key=lambda item:item[0])
    mapping={predicted[i]["id"]:actual[j]["id"] for i,j in matches}
    supported=[p for p in prediction["portals"] if p.get("status")=="image_supported"]
    # A room pair can have multiple doors. Match individual openings, with a
    # location gate, instead of collapsing them into one graph edge.
    tolerance=.5
    candidates=[]
    for opening in supported:
        ids=[mapping.get(s) for s in opening["spaces"]]
        options=[]
        if None not in ids:
            for j,door in enumerate(truth["portals"]):
                error=float(np.linalg.norm(np.array(opening["center_xz"])-door["center_xz"]))
                if sorted(ids)==sorted(door["spaces"]) and error<=tolerance:
                    options.append((error,j))
        candidates.append([j for _,j in sorted(options)])
    matches_by_door={}
    def assign(i,seen):
        for j in candidates[i]:
            if j in seen: continue
            seen.add(j)
            if j not in matches_by_door or assign(matches_by_door[j],seen):
                matches_by_door[j]=i
                return True
        return False
    for i in sorted(range(len(supported)),key=lambda i:len(candidates[i])):
        assign(i,set())
    errors, widths = [], []
    for j,i in matches_by_door.items():
        opening,door=supported[i],truth["portals"][j]
        errors.append(float(np.linalg.norm(np.array(opening["center_xz"])-door["center_xz"])))
        widths.append(abs(opening["width_m"]-door["width_m"]))
    precision=len(matches_by_door)/max(1,len(supported))
    recall=len(matches_by_door)/max(1,len(truth["portals"]))
    return {"predicted_space_candidates":len(predicted),"truth_spaces":len(actual),
            "truth_rooms":sum(s.get("kind") not in {"corridor","hall"} for s in actual),
            "confirmed_room_count":prediction.get("confirmed_room_count"),
            "mean_space_iou":round(sum(overlap[i,j] for i,j in matches)/max(len(actual),len(predicted),1),4),
            "matched_spaces":len(matches),"portal_precision":round(precision,4),"portal_recall":round(recall,4),
            "truth_portals":len(truth["portals"]),
            "portal_matching_tolerance_cm":tolerance*100,
            "image_supported_portals":sum(p["status"]=="image_supported" for p in prediction["portals"]),
            "geometry_only_portals":sum(p["status"]=="geometry_only" for p in prediction["portals"]),
            "portal_center_mean_error_cm":round(float(np.mean(errors))*100,2) if errors else None,
            "portal_width_mean_error_cm":round(float(np.mean(widths))*100,2) if widths else None,
            "room_mapping":mapping,"note":"방 용도와 통행 가능한 문을 확정한 정확도가 아닌, 별도 정답에 대한 관측 영역·연결 후보의 기하 비교입니다."}


def evaluate_scene_property(scene, truth, anchors=None, output=None):
    from .pipeline import umeyama
    from .plan_reference import align_to_plan, reference_status, validate_plan, write_comparison
    validate_plan(truth)
    reference={p["id"]:p for p in truth.get("photos",[])}
    pairs=[(reference[c["source_photo"]]["position"],c["position"]) for c in scene["cameras"]
           if c.get("source_photo") in reference]
    prediction=copy.deepcopy(scene["property_graph"])
    alignment={"method":"shared_scene_coordinates"}
    if anchors is not None:
        prediction,alignment=align_to_plan(prediction,anchors)
    elif len(pairs)>=3:
        true=np.array([p[0] for p in pairs]); measured=np.array([p[1] for p in pairs])
        scale,rotation,translation=umeyama(true,measured)
        def inverse(xz):
            data=np.array(xz,float).reshape(-1,2)
            xyz=np.column_stack([data[:,0],np.zeros(len(data)),data[:,1]])
            return ((xyz-translation)@rotation/scale)[:,[0,2]].tolist()
        for space in prediction["spaces"]:
            space["polygon_xz"]=inverse(space["polygon_xz"])
            space["center_xz"]=inverse([space["center_xz"]])[0]
        for portal in prediction["portals"]:
            portal["center_xz"]=inverse([portal["center_xz"]])[0]
            portal["width_m"]/=scale
        error=np.linalg.norm(measured-(scale*true@rotation.T+translation),axis=1)
        alignment={"method":"evaluation_only_camera_alignment","cameras":len(pairs),
                   "scale_vs_truth":round(scale,5),"pose_rmse_cm":round(float(np.sqrt(np.mean(error**2)))*100,3)}
    elif truth.get("coordinate_frame")!="scene":
        raise ValueError("도면 좌표 정렬 근거가 없어요. --anchors로 벽 모서리 대응점 3개 이상을 지정하거나 정답 촬영 위치·같은 장면 좌표의 도면을 제공해 주세요.")
    result=evaluate_graph(prediction,truth) | {"alignment":alignment,"reference":reference_status(truth)}
    result["metric_scope"]=("scale_normalized_shape" if alignment.get("fit_scale") or
                            alignment["method"]=="evaluation_only_camera_alignment" else "original_scale_geometry")
    result["neural_training_performed"]=False
    result["neural_training_performed_in_evaluation"]=False
    analysis=scene.get("room",{}).get("ceiling_analysis") or {}
    result["neural_ceiling_inference"]={"used":bool(analysis.get("model_id")),"model_id":analysis.get("model_id"),
        "scope":"천장 높이 후보 판별. 영역·문 연결을 신경망으로 학습한 결과가 아닙니다."}
    if output is not None:
        write_comparison(output,prediction,truth,result)
    return result
