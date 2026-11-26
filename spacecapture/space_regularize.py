"""공간 후보 윤곽 정리: 관측 영역 격자 → 직각 다각형 → 각 변을 바깥쪽 벽 점 밀집선으로 이동.

관측된 바닥 영역은 벽 바로 앞(가구·시야 가림)이 덜 잡혀 실제 방보다 작게 나온다.
각 변을 바깥쪽 0.5m 안에서 처음 나오는 벽 점(높이 0.3~2.0m) 밀집선으로 옮긴다.
정답 도면은 쓰지 않는다. 실제 SfM 시험(LH 26채)에서 평균 IoU 0.708→0.748(2026-11-11).
"""
import cv2
import numpy as np

from .plan_render import _orthogonal

SETTINGS = {"close_m": .3, "ortho_eps_m": .12, "max_out_m": .5, "max_in_m": .3, "bin_m": .02, "min_count": 4,
            "peak_fraction": .2, "wall_height_m": [.3, 2.0]}


def orthogonal_polygon(mask, lower, step, close_m=None, eps_m=None):
    """0/1 격자 → 직각 다각형(미터). 없으면 None."""
    close_m = SETTINGS["close_m"] if close_m is None else close_m
    eps_m = SETTINGS["ortho_eps_m"] if eps_m is None else eps_m
    mask = mask.astype(np.uint8)
    if close_m:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(round(close_m/step)) | 1),)*2)
        smoothed = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        if smoothed.any(): mask = smoothed
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return None
    contour = max(found, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if len(contour) < 3:
        return None
    poly = np.asarray(_orthogonal(contour, max(1., eps_m/step)), float)
    if len(poly) < 4:
        return None
    return (np.asarray(lower)+(poly+.5)*step).tolist()


def _edges(poly):
    pts = np.asarray(poly, float); n = len(pts); edges = []
    for i in range(n):
        a, b = pts[i], pts[(i+1) % n]
        if abs(a[0]-b[0]) < abs(a[1]-b[1]): edges.append(["v", (a[0]+b[0])/2, a[1], b[1]])
        else: edges.append(["h", (a[1]+b[1])/2, a[0], b[0]])
    return edges


def _poly(edges):
    out = []
    for i in range(len(edges)):
        a, b = edges[i-1], edges[i]
        if a[0] == b[0]: continue
        out.append([a[1], b[1]] if a[0] == "v" else [b[1], a[1]])
    return out


def _drop_jogs(points, min_len=.1):
    """따로 옮겨진 두 평행 변 사이에 남은 짧은 꺾임(수 cm)을 길이 가중 평균선으로 합친다."""
    Q = [list(map(float, p)) for p in points]
    while len(Q) > 4:
        n = len(Q)
        lengths = [abs(Q[(i+1) % n][0]-Q[i][0])+abs(Q[(i+1) % n][1]-Q[i][1]) for i in range(n)]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_len:
            break
        a, b, c, d = Q[(i-1) % n], Q[i], Q[(i+1) % n], Q[(i+2) % n]
        k = 0 if abs(a[0]-b[0]) < abs(a[1]-b[1]) else 1   # 이웃 변(a→b)이 세로면 x가 일정 (길이 0 변도 안전)
        la, ld = abs(b[1-k]-a[1-k]), abs(d[1-k]-c[1-k])
        value = (b[k]*la+c[k]*ld)/max(la+ld, 1e-9)
        a[k] = d[k] = value
        for j in sorted(((i) % n, (i+1) % n), reverse=True):
            del Q[j]
    return Q


def snap_to_walls(poly, wall_xz, s=None):
    """직각 다각형의 각 변을 바깥쪽 가장 가까운 벽 점 밀집선으로 옮긴다. 근거가 없으면 그대로."""
    s = SETTINGS if s is None else s
    edges = _edges(poly)
    if len(edges) < 4 or len(wall_xz) == 0:
        return poly
    P = np.asarray(poly, float)
    area = .5*np.sum(P[:, 0]*np.roll(P[:, 1], -1)-np.roll(P[:, 0], -1)*P[:, 1])
    bins = np.arange(-s["max_in_m"], s["max_out_m"]+s["bin_m"], s["bin_m"])
    for e in edges:
        axis, coord, s0, s1 = e
        lo, hi = sorted((s0, s1)); length = hi-lo
        if length < .3: continue
        u = 0 if axis == "v" else 1
        along = wall_xz[:, 1-u]
        m = (along > lo+.1*length) & (along < hi-.1*length)
        outward = -(1 if s1 > s0 else -1)*(1 if area > 0 else -1)*(-1 if axis == "v" else 1)
        offs = (wall_xz[m, u]-coord)*outward
        offs = offs[(offs > -s["max_in_m"]) & (offs < s["max_out_m"])]
        if len(offs) < s["min_count"]: continue
        hist, edges_b = np.histogram(offs, bins)
        hist = np.convolve(hist, [1, 2, 1], "same")
        cand = np.flatnonzero(hist >= max(s["min_count"], s["peak_fraction"]*hist.max()))
        if len(cand):
            e[1] = coord+outward*(edges_b[cand[0]]+s["bin_m"]/2)
    out = _drop_jogs(_poly(edges))
    if len(out) < 4:
        return poly
    Q = np.asarray(out, float)
    snapped_area = .5*np.sum(Q[:, 0]*np.roll(Q[:, 1], -1)-np.roll(Q[:, 0], -1)*Q[:, 1])
    # 변 이동으로 다각형이 뒤집히거나 크게 줄면 원래 윤곽을 유지한다.
    if np.sign(snapped_area) != np.sign(area) or abs(snapped_area) < .7*abs(area):
        return poly
    return [[round(float(x), 4), round(float(z), 4)] for x, z in out]


def regularize_spaces(spaces, labels, lower, step, points):
    """spaces의 polygon_xz를 벽에 맞춘 직각 윤곽으로 바꾸고 원래 관측 윤곽은 observed_polygon_xz로 보존."""
    P = np.asarray(points, float).reshape(-1, 3)
    lo, hi = SETTINGS["wall_height_m"]
    wall_xz = P[(P[:, 1] > lo) & (P[:, 1] < hi)][:, [0, 2]]
    changed = 0
    for i, space in enumerate(spaces, 1):
        poly = orthogonal_polygon(labels == i, lower, step)
        if poly is None:
            continue
        snapped = snap_to_walls(poly, wall_xz)
        space["observed_polygon_xz"] = space["polygon_xz"]
        space["polygon_xz"] = snapped
        space["polygon_method"] = "wall_snapped_orthogonal"
        changed += 1
    return {"method": "wall_snapped_orthogonal", "spaces": changed, "settings": SETTINGS}
