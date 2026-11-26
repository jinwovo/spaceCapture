"""공식 ProcTHOR 집 JSON을 평가용 구조·자체 렌더 설정으로 변환한다.

가구/재질 에셋이나 시뮬레이터 코드를 로드하지 않는다. 현재 변환기는 열린
내부 문이 있는 직교 벽 구조만 지원하며 빈 벽(개방형 의미 영역)은 보류한다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .synth import look_rotation


def _xz(point):
    return np.array([point["x"], point["z"]], dtype=float)


def _wall(wall):
    polygon = wall["polygon"]
    a, b = _xz(polygon[0]), _xz(polygon[1])
    delta = b - a
    if not np.isfinite([*a, *b]).all() or np.linalg.norm(delta) < .05:
        raise ValueError("벽의 좌표·길이가 올바르지 않아요.")
    if min(abs(delta)) > 1e-5:
        raise ValueError("현재 변환기는 비스듬한 벽을 지원하지 않아요.")
    horizontal = int(abs(delta[1]) > abs(delta[0]))
    normal = 1 - horizontal
    bottom, top = min(p["y"] for p in polygon), max(p["y"] for p in polygon)
    if abs(bottom) > 1e-5 or top <= 1.6:
        raise ValueError("한 층의 바닥 y=0 구조만 지원해요.")
    key = (normal, round(float(a[normal]), 5), round(float(min(a[horizontal], b[horizontal])), 5),
           round(float(max(a[horizontal], b[horizontal])), 5))
    return a, b, horizontal, normal, key, top


def photo_poses(spaces, polygons, portals, room_views=20, portal_views=4, min_candidates=6):
    """공간마다 둘러보는 촬영 위치와 문 양쪽에서 문을 향한 촬영 위치. 정답 방 ID는 기록용이다."""
    poses = []
    for space in spaces:
        polygon = polygons[space["id"]]
        low, high = polygon.min(0), polygon.max(0)
        candidates = []
        for x in np.linspace(low[0]+.5, high[0]-.5, 13):
            for z in np.linspace(low[1]+.5, high[1]-.5, 13):
                clearance = cv2.pointPolygonTest(polygon, (float(x),float(z)), True)
                if clearance > .45:
                    candidates.append((x,z,clearance))
        if len(candidates) < min_candidates:
            raise ValueError("공간 안에 안전한 촬영 위치를 충분히 만들 수 없어요.")
        candidates = np.array(candidates)
        center = candidates[np.argmax(candidates[:,2]),:2]
        for j in range(room_views):
            theta = 2*math.pi*j/room_views
            direction = np.array([math.cos(theta),math.sin(theta)])
            delta = candidates[:,:2]-center
            score = delta@direction - 2*np.abs(delta@np.array([-direction[1],direction[0]]))
            x,z = candidates[np.argmax(score),:2]
            camera = np.array([x,1.45,z])
            target = np.array([center[0], 1.1 if j%4 else 1.9, center[1]])
            if np.linalg.norm(target[[0,2]]-camera[[0,2]]) < .15:
                other = candidates[np.argmax(np.linalg.norm(candidates[:,:2]-[x,z],axis=1)),:2]
                target[[0,2]] = other
            poses.append((camera, look_rotation(target-camera), space["id"]))
    for portal in portals:
        center = np.array(portal["center_xz"])
        tangent = np.diff(portal["endpoints_xz"],axis=0)[0]; tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1],tangent[0]])
        for side in (-1.,1.):
            for j in range(portal_views):
                xz = center + side*(.35+j*.22)*normal
                owners = [sid for sid in portal["spaces"] if cv2.pointPolygonTest(polygons[sid], tuple(xz), False)>0]
                if len(owners)!=1:
                    continue
                target = center-side*1.5*normal
                camera = np.array([xz[0],1.45,xz[1]])
                poses.append((camera,look_rotation(np.array([target[0],1.2,target[1]])-camera),owners[0]))
    return poses


def convert_house(house, house_id, seed=0):
    """정답 변환만 수행한다. 이 결과는 사진 추론기에 전달하면 안 된다."""
    if any(w.get("empty") for w in house["walls"]):
        raise ValueError("빈 벽으로 이어진 개방형 의미 영역은 현재 변환 대상에서 보류해요.")
    spaces = [{"id": r["id"], "kind": "room", "source_room_type": r.get("roomType"),
               "polygon_xz": [_xz(p).tolist() for p in r["floorPolygon"]]} for r in house["rooms"]]
    if not spaces:
        raise ValueError("공간이 없는 집이에요.")
    polygons = {s["id"]: np.array(s["polygon_xz"], np.float32) for s in spaces}
    ids = set(polygons)
    walls = {w["id"]: _wall(w) for w in house["walls"] if w.get("roomId") in ids}
    height = float(np.median([w[-1] for w in walls.values()]))
    if any(abs(w[-1]-height) > .01 for w in walls.values()):
        raise ValueError("천장 높이가 다른 벽은 현재 변환 대상에서 보류해요.")
    floor = np.vstack(list(polygons.values()))
    offset = floor.min(0)
    width, depth = floor.max(0)-offset
    for space in spaces:
        space["polygon_xz"] = (polygons[space["id"]]-offset).tolist()
    polygons = {s["id"]: np.array(s["polygon_xz"], np.float32) for s in spaces}
    holes, portals = {}, []
    exterior = 0
    for door in house.get("doors", []):
        pair = [door["room0"], door["room1"]]
        if len(set(pair)) != 2 or not set(pair) <= ids or "exterior" in door["wall1"]:
            exterior += 1
            continue
        if door.get("openness", 0) < .99:
            raise ValueError("내부 문의 완전 개방 상태가 확인되지 않아 변환을 보류해요.")
        a, b, horizontal, normal, key, _ = walls[door["wall0"]]
        length = np.linalg.norm(b-a)
        hole = door["holePolygon"]
        lo, hi = min(p["x"] for p in hole), max(p["x"] for p in hole)
        top = max(p["y"] for p in hole)
        if not (-.01 <= lo < hi <= length+.01) or not 1.6 < top <= height+.01:
            raise ValueError("문 구멍의 위치·높이가 벽 범위를 벗어났어요.")
        if abs(min(p["y"] for p in hole)) > .01:
            raise ValueError("바닥에서 떠 있는 개구부는 현재 지원하지 않아요.")
        ends = np.array([a+(b-a)*lo/length, a+(b-a)*hi/length])-offset
        g_lo, g_hi = sorted(ends[:,horizontal])
        holes.setdefault(key, []).append((g_lo, g_hi, top))
        # Both wall records refer to the same physical cut; world coordinates
        # avoid mirroring the hole on the wall whose vertex order is reversed.
        if door["wall1"] not in walls or walls[door["wall1"]][4] != key:
            raise ValueError("문 양쪽 벽의 실제 위치가 일치하지 않아요.")
        portals.append({"id": door["id"], "spaces": pair, "center_xz": ends.mean(0).tolist(),
                        "endpoints_xz": ends.tolist(), "width_m": float(hi-lo), "height_m": float(top),
                        "kind": "doorway", "material": "open"})
    surfaces = []
    unique = {w[4]: w for w in walls.values()}
    for key, (_, _, horizontal, normal, _, _) in unique.items():
        _, position, low, high = key
        low -= offset[horizontal]; high -= offset[horizontal]
        def add(lo, hi, bottom=0., top=height):
            if hi-lo > 1e-5 and top-bottom > 1e-5:
                surfaces.append({"axis": 0 if normal==0 else 2, "pos": float(position-offset[normal]),
                                 "lo": float(lo), "hi": float(hi), "bottom": float(bottom), "top": float(top)})
        cursor = low
        for lo, hi, top in sorted(holes.get(key, [])):
            if lo < cursor-.02:
                raise ValueError("문 구멍이 겹쳐 있어요.")
            add(cursor, lo); add(lo, hi, top)
            cursor = hi
        add(cursor, high)
    # Decompose orthogonal floor polygons into exact rectangular cells. This
    # keeps floor/ceiling landmarks out of the missing corner of an L-shaped house.
    xs = sorted(set(float(x) for p in polygons.values() for x in p[:,0]))
    zs = sorted(set(float(z) for p in polygons.values() for z in p[:,1]))
    horizontal_surfaces = []
    for x0,x1 in zip(xs[:-1],xs[1:]):
        for z0,z1 in zip(zs[:-1],zs[1:]):
            center = ((x0+x1)/2, (z0+z1)/2)
            if any(cv2.pointPolygonTest(p, center, False)>0 for p in polygons.values()):
                for y in (0.,height):
                    horizontal_surfaces.append({"axis":1, "pos":y, "lo":x0, "hi":x1, "bottom":z0, "top":z1})
    poses = photo_poses(spaces, polygons, portals)
    order = np.random.default_rng(seed).permutation(len(poses))
    config = {"size":[float(width),height,float(depth)], "surfaces":surfaces,
              "horizontal_surfaces":horizontal_surfaces, "spaces":spaces, "portals":portals,
              "poses":[poses[i] for i in order]}
    metadata = {"id": house_id, "license":"Apache-2.0", "commercial_training_allowed":True,
                "origin":"external_structure_local_render", "source_dataset":"ProcTHOR-10K",
                "source_room_spec_id":house.get("metadata",{}).get("roomSpecId"),
                "render_modifications":{"materials":"자체 생성 텍스처", "objects_omitted":True,
                                        "windows_sealed":True, "exterior_doors_sealed":exterior,
                                        "wall_thickness_m":0., "indoor_doors":"원본의 완전 개방 상태 유지"},
                "note":"외부 합성 구조 JSON을 자체 렌더링한 평가 자료입니다. 실사·공식 시뮬레이터 이미지가 아닙니다."}
    return config, metadata
