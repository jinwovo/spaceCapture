"""촬영 경로 순서: 뷰어의 '촬영 경로 재생'이 카메라를 어떤 순서로 이을지 정한다.

- 영상 프레임이 대부분이면 프레임 순서. 섞인 사진은 비용이 가장 적게 드는 자리에 끼워 넣는다.
- 사진에 촬영 시각(EXIF DateTimeOriginal)이 대부분 있으면 시각 순서. 시각이 없는 사진은 끼워 넣는다.
- 둘 다 없으면 '걸어서 찍은 것처럼' 짧은 경로: 거리 + 보는 방향 차이 + 다른 공간으로 넘어가는 벌점을
  비용으로, 여러 시작점의 가까운 이웃 경로를 2-opt로 다듬어 가장 싼 것을 고른다.
순서는 재생용 추정이며 실제 촬영 순서라는 뜻이 아니다(method로 구분).
"""
from __future__ import annotations

import numpy as np

ANGLE_WEIGHT = .4    # 보는 방향 1 rad 차이 = 0.4 m
SPACE_PENALTY = 2.   # 다른 공간 후보로 넘어가면 2 m 더함(한 방을 다 찍고 다음 방으로)
MAX_STARTS = 12


def _forward(quaternion):
    """three.js 카메라 사원수(xyzw) → 보는 방향(-z)."""
    x, y, z, w = quaternion
    return -np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])


def cost_matrix(cameras, photo_spaces=None):
    P = np.array([c["position"] for c in cameras], float)
    F = np.array([_forward(c["quaternion"]) for c in cameras])
    F[:, 1] = 0; F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    dist = np.linalg.norm(P[:, None, [0, 2]]-P[None, :, [0, 2]], axis=2)
    ang = np.arccos(np.clip(F@F.T, -1, 1))
    C = dist+ANGLE_WEIGHT*ang
    if photo_spaces:
        sp = [photo_spaces.get(c["name"]) for c in cameras]
        diff = np.array([[a is not None and b is not None and a != b for b in sp] for a in sp])
        C = C+SPACE_PENALTY*diff
    return C


def path_cost(order, C):
    return float(sum(C[a, b] for a, b in zip(order[:-1], order[1:])))


def _two_opt(order, C, max_rounds=200):
    """열린 경로 2-opt: 구간 [i+1..j]를 뒤집어 비용이 줄면 받아들인다(i=-1은 경로 앞쪽 뒤집기)."""
    order = np.array(order); n = len(order)
    for _ in range(max_rounds):
        improved = False
        for i in range(-1, n-2):
            j = np.arange(i+2 if i >= 0 else 1, n)
            c = order[j]; d = order[np.minimum(j+1, n-1)]; last = j == n-1
            cd = np.where(last, 0., C[c, d])
            if i < 0:  # 앞 이음이 없음: 뒤집으면 (c, d)가 (order[0], d)로 바뀐다
                delta = np.where(last, 0., C[order[0], d])-cd
            else:
                a, b = order[i], order[i+1]
                delta = C[a, c]+np.where(last, 0., C[b, d])-C[a, b]-cd
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                order[i+1:j[k]+1] = order[i+1:j[k]+1][::-1].copy(); improved = True
        if not improved: break
    return order.tolist()


def _nearest(start, C):
    n = len(C); left = set(range(n)); left.discard(start); order = [start]
    while left:
        p = order[-1]; q = min(left, key=lambda k: C[p, k]); order.append(q); left.discard(q)
    return order


def shortest_order(C):
    n = len(C)
    if n <= 2: return list(range(n))
    # 시작점 후보: 다른 카메라에서 가장 먼(가장자리) 카메라들과 0번
    far = np.argsort(-C.sum(1))[:MAX_STARTS-1].tolist()
    best = None
    for s in dict.fromkeys([0]+far):
        o = _two_opt(_nearest(s, C), C)
        v = path_cost(o, C)
        if best is None or v < best[1]: best = (o, v)
    return best[0]


def insert_cheapest(order, extra, C):
    order = list(order)
    for k in extra:
        if not order: order.append(k); continue
        cands = [(C[k, order[0]], 0), (C[order[-1], k], len(order))]
        cands += [(C[order[i], k]+C[k, order[i+1]]-C[order[i], order[i+1]], i+1) for i in range(len(order)-1)]
        order.insert(min(cands)[1], k)
    return order


def capture_order(cameras, photos=None, photo_spaces=None):
    """반환: {method, names, jumps_over_2m, length_m}. 카메라 2대 미만이면 None."""
    if len(cameras) < 2: return None
    C = cost_matrix(cameras, photo_spaces)
    framed = [k for k, c in enumerate(cameras) if c.get("video_frame") is not None]
    taken = {p["name"]: p.get("taken_at") for p in (photos or []) if p.get("taken_at")}
    timed = [k for k, c in enumerate(cameras) if c["name"] in taken]
    if len(framed) >= .8*len(cameras) or (framed and len(framed) >= len(timed)):
        base = sorted(framed, key=lambda k: cameras[k]["video_frame"]); method = "video_frame"
    elif len(timed) >= .8*len(cameras) and len({taken[cameras[k]["name"]] for k in timed}) > 1:
        base = sorted(timed, key=lambda k: (taken[cameras[k]["name"]], cameras[k]["name"])); method = "exif_time"
    else:
        base, method = None, "shortest_path"
    if base is None:
        order = shortest_order(C)
    else:
        order = insert_cheapest(base, [k for k in range(len(cameras)) if k not in set(base)], C)
    P = np.array([cameras[k]["position"] for k in order], float)
    steps = np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1)
    return {"method": method, "names": [cameras[k]["name"] for k in order],
            "length_m": round(float(steps.sum()), 2), "jumps_over_2m": int((steps > 2).sum())}
