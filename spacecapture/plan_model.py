"""공간 후보 → 깔끔한 건축 도면 모델: 공유 벽 중심선, 일정한 벽 두께, 문 개구부, 순 면적.

공간 후보 다각형(property_graph)은 방마다 따로 직각화돼 있어서 이웃 방 사이에 간격이 생기고,
같은 벽이 방마다 조금씩 다른 위치에 놓인다. 그대로 그리면 벽 두께가 들쭉날쭉하고 모서리가 뭉개진다.

1. 모든 공간의 가로·세로 변 위치를 축별로 묶어(0.35m 이내) 공통 기준선을 만든다.
2. 공간을 5cm 격자에 그리고, 집 안쪽의 빈 틈은 가장 가까운 공간으로 채운다(이웃 방은 중간선에서 만남).
3. 기준선이 만드는 직사각형 칸마다 다수결로 공간을 정한다 → 모든 방이 같은 기준선을 공유하는 직각 다각형.
4. 칸 경계 중 양쪽 공간이 다른 곳이 벽이다(바깥 0.2m, 안쪽 0.12m). 문 후보(image_supported)는 그 두 공간
   사이 벽에 개구부로 낸다.

정답 도면은 쓰지 않는다. 결과는 scene["floorplan"]에 저장하고 2D 도면(plan_render)과 뷰어 구조 모형이 함께 쓴다.
"""
from __future__ import annotations

import cv2
import numpy as np

VERSION = 1
SETTINGS = {"cell_m": .05, "line_tol_m": .1, "min_gap_m": 0., "fill_gap_m": 0., "wall_max_m": .45, "touch_m": .12,
            "min_room_m2": .8, "smooth_m": .5, "support_window_m": .3, "support_min_per_m": 1., "hole_fill_m2": .3, "exterior_m": .2, "interior_m": .12, "door_min_m": .6,
            "door_max_leaf_m": 1.2, "door_margin_m": .05, "jamb_min_m": .25}


def _cluster(values, weights, tol, min_gap=0.):
    """1차원 값들을 폭 tol 이내로 묶어 가중 평균 위치를 돌려준다."""
    order = np.argsort(values)
    groups, cur = [], []
    for i in order:
        if cur and values[i]-values[cur[0]] > tol:
            groups.append(cur); cur = []
        cur.append(i)
    if cur: groups.append(cur)
    lines = [[float(np.average([values[i] for i in g], weights=[weights[i] for i in g])), float(sum(weights[i] for i in g))]
             for g in groups]
    while len(lines) > 2:  # 너무 가까운 이웃 기준선(좁은 띠 칸 → 짧은 꺾임·가시 벽)은 무거운 쪽으로 합친다
        gaps = [lines[k+1][0]-lines[k][0] for k in range(len(lines)-1)]
        k = int(np.argmin(gaps))
        if gaps[k] >= min_gap: break
        (v0, w0), (v1, w1) = lines[k], lines[k+1]
        lines[k:k+2] = [[(v0*w0+v1*w1)/(w0+w1), w0+w1]]
    return [v for v, _ in lines]


def _polygon_area(poly):
    p = np.asarray(poly, float)
    return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))-np.dot(p[:, 1], np.roll(p[:, 0], -1)))/2)


def _components(mask):
    n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    return [(comp == k) for k in range(1, n)]


def _trace(cells, xs, zs):
    """칸 묶음(bool, [z, x]) → 바깥 경계 다각형(미터, 꼭짓점은 기준선 교점)."""
    edges = {}
    nz, nx = cells.shape
    for j, i in zip(*np.nonzero(cells)):
        # 칸 (i, j)의 네 변을 한 방향으로 돌며 등록, 이웃 칸과 공유하는 변은 서로 지운다
        for a, b in (((i, j), (i+1, j)), ((i+1, j), (i+1, j+1)), ((i+1, j+1), (i, j+1)), ((i, j+1), (i, j))):
            if (b, a) in edges: del edges[(b, a)]
            else: edges[(a, b)] = True
    nxt = {}
    for a, b in edges: nxt.setdefault(a, []).append(b)
    loops = []
    while nxt:
        start = next(iter(nxt)); loop = [start]; cur = start
        while True:
            b = nxt[cur].pop()
            if not nxt[cur]: del nxt[cur]
            if b == start: break
            loop.append(b); cur = b
            if cur not in nxt: loop = None; break  # 닫히지 않은 경로(수치 오차)는 버린다
        if loop: loops.append(loop)
    best = None
    for loop in loops:
        pts = [(xs[i], zs[j]) for i, j in loop]
        simple = []
        for k in range(len(pts)):  # 일직선 위 중간 꼭짓점 제거
            a, b, c = pts[k-1], pts[k], pts[(k+1) % len(pts)]
            if (abs(a[0]-b[0]) < 1e-9 and abs(b[0]-c[0]) < 1e-9) or (abs(a[1]-b[1]) < 1e-9 and abs(b[1]-c[1]) < 1e-9):
                continue
            simple.append([float(b[0]), float(b[1])])
        if len(simple) >= 4 and (best is None or _polygon_area(simple) > _polygon_area(best)):
            best = simple
    return best


def _spaces(scene):
    graph = scene.get("property_graph") or {}
    spaces = [s for s in graph.get("spaces", []) if len(s.get("polygon_xz", [])) >= 3]
    box_ok = scene.get("layout", {}).get("room_box_supported", True) is not False and "room" in scene
    if not spaces or (len(spaces) == 1 and box_ok):  # 방 하나면 측정한 직사각형 방 상자를 쓴다
        if "room" not in scene:
            return [], []
        room = scene["room"]
        x0, z0, x1, z1 = room["min"][0], room["min"][2], room["max"][0], room["max"][2]
        base = spaces[0] if spaces else {}
        spaces = [{"id": base.get("id", "room"), "polygon_xz": [[x0, z0], [x1, z0], [x1, z1], [x0, z1]],
                   "usage_candidates": base.get("usage_candidates", []), "usage_estimate": base.get("usage_estimate")}]
        return spaces, []
    return spaces, [p for p in graph.get("portals", []) if p.get("status") == "image_supported"]


def _usage(space):
    estimate = space.get("usage_estimate") or {}
    if estimate.get("usage"):  # 사물 근거 우선, 없으면 공간 기하 모델 추정(room_types.estimate_usage)
        return estimate["usage"]
    usage = [u["usage"] for u in space.get("usage_candidates", [])]
    return usage[0] if len(usage) == 1 else None


class _WallSupport:
    """벽 높이(0.2m~천장-0.1m) 점의 x·z 누적 격자. 사각형 안 점 수를 빠르게 센다."""
    def __init__(self, points, height, step=.05):
        P = np.asarray(points, float)
        P = P[(P[:, 1] > .2) & (P[:, 1] < max(1., height-.1))]
        self.ok = len(P) > 0
        if not self.ok: return
        self.low, self.step = P[:, [0, 2]].min(0)-1, step
        idx = np.floor((P[:, [0, 2]]-self.low)/step).astype(int)
        shape = idx.max(0)+2
        grid = np.zeros((shape[0], shape[1]), np.int32); np.add.at(grid, (idx[:, 0], idx[:, 1]), 1)
        self.sat = np.pad(grid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    def count(self, x0, z0, x1, z1):
        if not self.ok: return 0
        i0, k0 = np.clip(np.floor((np.array([x0, z0])-self.low)/self.step).astype(int), 0, np.array(self.sat.shape)-1)
        i1, k1 = np.clip(np.floor((np.array([x1, z1])-self.low)/self.step).astype(int)+1, 0, np.array(self.sat.shape)-1)
        s = self.sat
        return int(s[i1, k1]-s[i0, k1]-s[i1, k0]+s[i0, k0])


def build(scene, settings=None, points=None):
    """scene → 도면 모델(dict, 미터·장면 좌표 x/z). 공간이 없으면 None."""
    S = {**SETTINGS, **(settings or {})}
    spaces, portals = _spaces(scene)
    if not spaces:
        return None
    cell = S["cell_m"]
    polys = [np.asarray(s["polygon_xz"], float) for s in spaces]
    allp = np.vstack(polys)
    low = allp.min(0)-1.
    shape = (int(np.ceil((allp.max(0)[1]-low[1]+1.)/cell))+1, int(np.ceil((allp.max(0)[0]-low[0]+1.)/cell))+1)
    G = lambda pts: np.round((np.asarray(pts, float)-low)/cell).astype(np.int32)

    # 1) 공통 기준선
    vx, wx, vz, wz = [], [], [], []
    for p in polys:
        for a, b in zip(p, np.roll(p, -1, 0)):
            if abs(a[0]-b[0]) <= abs(a[1]-b[1]): vx.append((a[0]+b[0])/2); wx.append(abs(a[1]-b[1])+1e-3)
            else: vz.append((a[1]+b[1])/2); wz.append(abs(a[0]-b[0])+1e-3)
    xs = _cluster(np.array(vx), wx, S["line_tol_m"], S["min_gap_m"]); zs = _cluster(np.array(vz), wz, S["line_tol_m"], S["min_gap_m"])
    if len(xs) < 2 or len(zs) < 2:
        return None
    xs, zs = [round(v, 4) for v in xs], [round(v, 4) for v in zs]  # 벽 사각형 좌표와 같은 자릿수(실내 다각형 칸 정렬)

    # 2) 세밀 격자(큰 공간부터). fill_gap_m>0이면 집 안쪽 틈을 가장 가까운 공간으로 채운다(기본 끔).
    labels = np.zeros(shape, np.int32)
    order = sorted(range(len(spaces)), key=lambda i: -_polygon_area(polys[i]))
    for i in order:
        m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [G(polys[i])], 1)
        labels[(m > 0) & (labels == 0)] = i+1
    if S["smooth_m"] > 0:  # 방마다 작은 들쭉날쭉(계단 모양 윤곽)을 닫힘·열림으로 다듬는다
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(round(S["smooth_m"]/cell)) | 1),)*2)
        smooth = np.zeros_like(labels)
        for i in order:
            m = (labels == i+1).astype(np.uint8)
            m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
            smooth[(m > 0) & (smooth == 0)] = i+1
        labels = smooth
    filled = labels
    if S["fill_gap_m"] > 0:
        gap = max(1, int(round(S["fill_gap_m"]/cell)))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2*gap+1,)*2)
        inside = cv2.morphologyEx((labels > 0).astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        free = (labels == 0).astype(np.uint8)
        dist, idx = cv2.distanceTransformWithLabels(free, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(idx.max()+1, np.int32); lut[idx[free == 0]] = labels[free == 0]
        filled = labels.copy(); grow = (free > 0) & inside & (dist <= gap); filled[grow] = lut[idx[grow]]

    # 3) 기준선 칸 다수결
    gx = np.clip(np.round((np.array(xs)-low[0])/cell).astype(int), 0, shape[1])
    gz = np.clip(np.round((np.array(zs)-low[1])/cell).astype(int), 0, shape[0])
    nx, nz = len(xs)-1, len(zs)-1
    dxs, dzs = np.diff(xs), np.diff(zs)
    cells = np.zeros((nz, nx), np.int32)
    for j in range(nz):
        for i in range(nx):
            block = filled[gz[j]:max(gz[j+1], gz[j]+1), gx[i]:max(gx[i+1], gx[i]+1)].ravel()
            if not block.size: continue
            counts = np.bincount(block, minlength=len(spaces)+1)
            best = int(np.argmax(counts[1:]))+1
            cells[j, i] = best if counts[best] > counts[0] else 0
    cell_area = np.outer(dzs, dxs)
    for lab in range(1, len(spaces)+1):  # 방마다 가장 큰 덩어리만 남긴다
        parts = _components(cells == lab)
        if len(parts) > 1:
            parts.sort(key=lambda m: -cell_area[m].sum())
            for m in parts[1:]: cells[m] = 0

    # 4) 벽 칸: 양옆(또는 위아래)이 막힌 얇은 빈 띠(≤ wall_max_m)는 실제 벽 두께로 본다.
    #    양쪽이 같은 방이면 관측이 덜 된 방 일부라 그 방으로 채운다.
    WALL = -1
    def at(j, i): return cells[j, i] if 0 <= j < nz and 0 <= i < nx else 0
    for _ in range(4):
        changed = False
        for j, i in zip(*np.nonzero(cells == 0)):
            l, r, u, d = at(j, i-1), at(j, i+1), at(j-1, i), at(j+1, i)
            if dxs[i] <= S["wall_max_m"] and l and r:
                cells[j, i] = l if (l == r and l > 0) else WALL; changed = True
            elif dzs[j] <= S["wall_max_m"] and u and d:
                cells[j, i] = u if (u == d and u > 0) else WALL; changed = True
        if not changed: break
    def border(m): return m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any()
    unobserved = []
    for m in _components(np.pad(cells, 1) == 0):  # 집 안쪽 빈칸: 아주 작으면 벽, 크면 미관측 영역
        if border(m): continue
        m = m[1:-1, 1:-1]
        if cell_area[m].sum() < S["hole_fill_m2"]: cells[m] = WALL
        else:
            poly = _trace(m, xs, zs)
            if poly: unobserved.append(poly)

    rooms = []
    for lab in range(1, len(spaces)+1):
        m = cells == lab
        if not m.any(): continue
        poly = _trace(m, xs, zs)
        if cell_area[m].sum() < S["min_room_m2"] or poly is None:
            cells[m] = WALL if cell_area[m].sum() < .3 else 0; continue
        rooms.append({"label": lab, "id": spaces[lab-1]["id"], "polygon_xz": poly, "usage": _usage(spaces[lab-1])})
    outside = np.zeros((nz+2, nx+2), bool)
    for m in _components(np.pad(cells, 1) == 0):
        if border(m): outside |= m

    # 5) 벽 사각형(미터 x0,z0,x1,z1): 벽 칸 + 칸 경계(방-방 맞닿음은 가운데, 바깥·미관측 쪽은 방 바깥에)
    rects = []  # [x0, z0, x1, z1, kind]
    for j, i in zip(*np.nonzero(cells == WALL)):
        rects.append([xs[i], zs[j], xs[i+1], zs[j+1], "interior"])
    P = np.pad(cells, 1); O = outside
    edges = []
    for j in range(nz):
        for i in range(nx+1):
            a, b = P[j+1, i], P[j+1, i+1]
            if a != b: edges.append(("x", xs[i], zs[j], zs[j+1], a, b, O[j+1, i], O[j+1, i+1]))
    for j in range(nz+1):
        for i in range(nx):
            a, b = P[j, i+1], P[j+1, i+1]
            if a != b: edges.append(("z", zs[j], xs[i], xs[i+1], a, b, O[j, i+1], O[j+1, i+1]))
    support = _WallSupport(points, float((scene.get("room") or {}).get("height") or 2.4)) if points is not None else None
    segs = []
    for axis, line, t0, t1, a, b, oa, ob in edges:
        if a > 0 and b > 0: kind, side, th = "touch", 0., S["touch_m"]
        elif a == 0 or b == 0:  # 바깥·미관측 쪽으로 벽을 세운다 (방 쪽 벽면 = 기준선)
            outer = oa if a == 0 else ob
            kind, side = ("exterior" if outer else "unobserved"), (-1. if a == 0 else 1.)
            th = S["exterior_m"] if outer else S["interior_m"]
        else: continue  # 방-벽칸 경계는 벽칸 자체가 벽
        is_open = False
        if support is not None and support.ok and kind != "touch":  # 벽 점이 거의 없는 바깥 경계 = 관측 끝(벽 근거 없음)
            w = S["support_window_m"]
            box = (line-w, t0-w, line+w, t1+w) if axis == "x" else (t0-w, line-w, t1+w, line+w)
            is_open = support.count(*box)/max(t1-t0, 1e-6) < S["support_min_per_m"]
        pair = (int(min(a, b)), int(max(a, b)))
        last = segs[-1] if segs else None
        if last and last["axis"] == axis and last["at"] == line and last["pair"] == pair and last["side"] == side                 and last["open"] == is_open and abs(last["t1"]-t0) < 1e-9:
            last["t1"] = t1
        else:
            segs.append({"axis": axis, "at": line, "t0": t0, "t1": t1, "kind": kind, "side": side, "th": th, "pair": pair, "open": is_open})
    open_edges = []
    for s in segs:
        if s["open"]:  # 벽을 세우지 않고 관측 경계선으로만 남긴다
            open_edges.append([s["at"], s["t0"], s["at"], s["t1"]] if s["axis"] == "x" else [s["t0"], s["at"], s["t1"], s["at"]])
            s["across"] = (s["at"], s["at"]); continue
        if s["kind"] == "touch": c0, c1 = s["at"]-s["th"]/2, s["at"]+s["th"]/2
        elif s["side"] > 0: c0, c1 = s["at"], s["at"]+s["th"]
        else: c0, c1 = s["at"]-s["th"], s["at"]
        s["across"] = (c0, c1)
        e = s["th"]/2 if s["kind"] == "touch" else 0.
        r = [c0, s["t0"]-e, c1, s["t1"]+e] if s["axis"] == "x" else [s["t0"]-e, c0, s["t1"]+e, c1]
        rects.append(r+["exterior" if s["kind"] == "exterior" else "interior"])
    # 볼록 모서리: 바깥쪽 두 벽 사각형 사이에 빈 정사각형을 채운다
    filled_v = np.pad(cells != 0, 1)
    for j in range(nz+1):
        for i in range(nx+1):
            q = filled_v[j:j+2, i:i+2]
            if q.sum() != 1: continue
            (qj, qi), = np.argwhere(q)
            oj, oi = 1-qj, 1-qi  # 대각선 반대 칸
            outer = O[j+oj, i+oi]
            th = S["exterior_m"] if outer else S["interior_m"]
            x0 = xs[i]-th if oi == 0 else xs[i]; z0 = zs[j]-th if oj == 0 else zs[j]
            corner = [x0, z0, x0+th, z0+th]
            touches = any(r[0] <= corner[2]+1e-6 and r[2] >= corner[0]-1e-6 and r[1] <= corner[3]+1e-6 and r[3] >= corner[1]-1e-6
                          for r in rects if r[4] != "corner")
            if touches: rects.append(corner+["corner"])  # 벽 근거 없는 두 경계 사이 모서리는 세우지 않음

    # 6) 문: 두 공간 사이(맞닿은 경계 또는 벽 칸 띠)에 개구부
    by_label = {r["label"]: r for r in rooms}
    by_id = {r["id"]: r["label"] for r in rooms}
    runs = [{"pair": s["pair"], "axis": s["axis"], "across": s["across"], "t0": s["t0"], "t1": s["t1"]}
            for s in segs if s["kind"] == "touch"]
    for axis in ("x", "z"):  # 벽 칸 띠: 좌우로 끼인 세로 벽(x) / 위아래로 끼인 가로 벽(z)
        for j, i in zip(*np.nonzero(cells == WALL)):
            a, b = (at(j, i-1), at(j, i+1)) if axis == "x" else (at(j-1, i), at(j+1, i))
            if a <= 0 or b <= 0 or a == b: continue
            across = (xs[i], xs[i+1]) if axis == "x" else (zs[j], zs[j+1])
            t0, t1 = (zs[j], zs[j+1]) if axis == "x" else (xs[i], xs[i+1])
            pair = (int(min(a, b)), int(max(a, b)))
            prev = next((r for r in runs if r["pair"] == pair and r["axis"] == axis and r["across"] == across
                         and (abs(r["t1"]-t0) < 1e-9 or abs(r["t0"]-t1) < 1e-9)), None)
            if prev: prev["t0"], prev["t1"] = min(prev["t0"], t0), max(prev["t1"], t1)
            else: runs.append({"pair": pair, "axis": axis, "across": across, "t0": t0, "t1": t1})
    doors, cuts = [], []
    area = {r["label"]: _polygon_area(r["polygon_xz"]) for r in rooms}
    for portal in portals:
        la, lb = (by_id.get(s) for s in portal["spaces"])
        if la is None or lb is None: continue
        pair = (min(la, lb), max(la, lb))
        cands = [r for r in runs if r["pair"] == pair]
        if not cands: continue
        c = np.asarray(portal["center_xz"], float)
        def dist(r):
            along, acr = (c[1], c[0]) if r["axis"] == "x" else (c[0], c[1])
            return max(0., r["across"][0]-acr, acr-r["across"][1])+max(0., r["t0"]-along, along-r["t1"])
        r = min(cands, key=dist)
        length = r["t1"]-r["t0"]; margin = S["door_margin_m"]
        width = float(np.clip(portal.get("width_m", .9), S["door_min_m"], max(S["door_min_m"], length-2*margin)))
        if width >= length: width = length; mid = (r["t0"]+r["t1"])/2
        else: mid = float(np.clip(c[1] if r["axis"] == "x" else c[0], r["t0"]+margin+width/2, r["t1"]-margin-width/2))
        lo, hi = mid-width/2, mid+width/2
        a0, a1 = r["across"]
        cuts.append([a0-.01, lo, a1+.01, hi] if r["axis"] == "x" else [lo, a0-.01, hi, a1+.01])
        into = la if area[la] <= area[lb] else lb  # 문짝은 작은 방 쪽으로 연다(국내 도면 관례)
        probe = [a1+.15, mid] if r["axis"] == "x" else [mid, a1+.15]
        inside_hi = cv2.pointPolygonTest(np.asarray(by_label[into]["polygon_xz"], np.float32), (float(probe[0]), float(probe[1])), False) >= 0
        doors.append({"spaces": [spaces[la-1]["id"], spaces[lb-1]["id"]], "portal_id": portal.get("id"), "axis": r["axis"],
                      "across": [round(a0, 4), round(a1, 4)], "lo": round(lo, 4), "hi": round(hi, 4), "width_m": round(width, 3),
                      "swing": 1. if inside_hi else -1., "into": spaces[into-1]["id"], "leaf": width <= S["door_max_leaf_m"]})
    walls, patches = [], []  # patches: 문턱·잘라 낸 문틀 조각 자리(바닥으로 칠함)
    for x0, z0, x1, z1, kind in rects:
        pieces = [[x0, z0, x1, z1]]
        for cx0, cz0, cx1, cz1 in cuts:
            ix0, iz0, ix1, iz1 = max(x0, cx0), max(z0, cz0), min(x1, cx1), min(z1, cz1)
            if ix1-ix0 > 1e-3 and iz1-iz0 > 1e-3: patches.append([round(float(v), 4) for v in (ix0, iz0, ix1, iz1)])
            nxt = []
            for px0, pz0, px1, pz1 in pieces:
                if cx0 >= px1 or cx1 <= px0 or cz0 >= pz1 or cz1 <= pz0: nxt.append([px0, pz0, px1, pz1]); continue
                if px0 < cx0: nxt.append([px0, pz0, cx0, pz1])
                if cx1 < px1: nxt.append([cx1, pz0, px1, pz1])
                ix0, ix1 = max(px0, cx0), min(px1, cx1)
                if pz0 < cz0: nxt.append([ix0, pz0, ix1, cz0])
                if cz1 < pz1: nxt.append([ix0, cz1, ix1, pz1])
            pieces = nxt
        split = len(pieces) != 1 or pieces[0] != [x0, z0, x1, z1]
        for q in pieces:
            w_, h_ = q[2]-q[0], q[3]-q[1]
            if split and max(w_, h_) < S["jamb_min_m"]:  # 문 옆에 남은 손톱만 한 벽 조각(문틀 잡티)
                patches.append([round(float(v), 4) for v in q]); continue
            if w_ > 1e-3 and h_ > 1e-3:
                walls.append({"rect_xz": [round(float(v), 4) for v in q], "exterior": kind in ("exterior", "corner")})

    # 7) 실내 바닥(벽면 안쪽): 방 칸 다각형에서 겹치는 벽을 뺀 영역. 면적·평가·도면 바닥에 쓴다.
    fine = .025
    for r in rooms:
        poly = np.asarray(r["polygon_xz"], float)
        lo2 = poly.min(0)-.3
        size = (int(np.ceil((poly.max(0)[1]-lo2[1]+.3)/fine)), int(np.ceil((poly.max(0)[0]-lo2[0]+.3)/fine)))
        F = lambda q: np.round((np.asarray(q, float)-lo2)/fine).astype(np.int32)
        mask = np.zeros(size, np.uint8); cv2.fillPoly(mask, [F(poly)], 1)
        for w in walls:
            q0, q1 = F(w["rect_xz"][:2]), F(w["rect_xz"][2:])
            mask[max(q0[1], 0):max(q1[1], 0), max(q0[0], 0):max(q1[0], 0)] = 0
        # 정확한 직각 다각형: 방 꼭짓점·벽 사각형 좌표로 만든 칸 중 방 안이면서 벽이 아닌 칸을 잇는다
        bx0, bz0 = poly.min(0); bx1, bz1 = poly.max(0)
        near = [w["rect_xz"] for w in walls if w["rect_xz"][0] < bx1 and w["rect_xz"][2] > bx0 and w["rect_xz"][1] < bz1 and w["rect_xz"][3] > bz0]
        ux = sorted({float(v) for v in poly[:, 0]} | {float(np.clip(q[k], bx0, bx1)) for q in near for k in (0, 2)})
        uz = sorted({float(v) for v in poly[:, 1]} | {float(np.clip(q[k], bz0, bz1)) for q in near for k in (1, 3)})
        cx_, cz_ = (np.array(ux[:-1])+ux[1:])/2, (np.array(uz[:-1])+uz[1:])/2
        keep = np.zeros((len(cz_), len(cx_)), bool)
        contour = poly.astype(np.float32)
        for jj, zc in enumerate(cz_):
            for ii, xc in enumerate(cx_):
                keep[jj, ii] = cv2.pointPolygonTest(contour, (float(xc), float(zc)), False) > 0 and                     not any(q[0] <= xc <= q[2] and q[1] <= zc <= q[3] for q in near)
        parts = _components(keep)
        inner, net = poly.tolist(), _polygon_area(poly)
        if parts:
            big = max(parts, key=lambda m: (np.outer(np.diff(uz), np.diff(ux))*m).sum())
            traced = _trace(big, ux, uz)
            if traced: inner, net = traced, float((np.outer(np.diff(uz), np.diff(ux))*big).sum())
        d = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cz, cx = np.unravel_index(np.argmax(d), d.shape)
        x0, z0 = np.min(inner, 0); x1, z1 = np.max(inner, 0)
        r.update({"interior_polygon_xz": inner, "area_m2": round(net, 3),
                  "anchor_xz": [round(float(cx*fine+lo2[0]), 3), round(float(cz*fine+lo2[1]), 3)], "inradius_m": round(float(d.max())*fine, 3),
                  "bbox_xz": [float(x0), float(z0), float(x1), float(z1)]})
    ext = [w["rect_xz"] for w in walls if w["exterior"]] or [w["rect_xz"] for w in walls]
    for r in rooms: del r["label"]
    return {"version": VERSION, "method": "shared_wall_lines", "rooms": rooms, "walls": walls, "doors": doors, "floor_patches": patches,
            "open_edges": [[round(float(v), 4) for v in e] for e in open_edges],
            "unobserved": unobserved, "lines": {"x": [round(v, 4) for v in xs], "z": [round(v, 4) for v in zs]},
            "bounds_xz": ([min(q[0] for q in ext), min(q[1] for q in ext), max(q[2] for q in ext), max(q[3] for q in ext)] if ext else None),
            "settings": S, "skipped_portals": len(portals)-len(doors)}
