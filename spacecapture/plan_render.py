"""국내 매물 평면도 양식의 2D 도면(SVG·PNG): 검은 벽, 나무·타일 바닥, 문 열림, 실 이름·면적, mm 치수, 가구.

입력은 scene.json의 공간 후보·문 후보·사물이다. 도면은 사진 기반 추정이며 실측 도면이 아니다.
용도 이름은 사물 근거가 있을 때만 '(추정)'으로 붙이고, 없으면 '공간 N'으로 쓴다.
"""
from __future__ import annotations

import math
from html import escape
from pathlib import Path

import cv2
import numpy as np

CELL = .025
WALL = .18
PYEONG = 3.3058
FONT = Path("C:/Windows/Fonts/malgun.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/malgunbd.ttf")
COLORS = {"bg": "#ffffff", "wall": "#222222", "wood": "#e9dcc6", "wood_line": "#d8c7aa", "tile": "#c9cdd1",
          "tile_line": "#aeb3b8", "text": "#1d1d1d", "sub": "#6b6b6b", "dim": "#555555", "furniture": "#8a8a8a",
          "door": "#444444"}
USAGE_NAME = {"bathroom": "욕실", "bedroom": "침실", "living": "거실", "kitchen": "주방"}
FURNITURE = {"bed": (1.5, 2.0), "couch": (2.0, .9), "dining table": (1.4, .8), "toilet": (.42, .68), "sink": (.55, .45),
             "refrigerator": (.9, .75), "oven": (.6, .6), "tv": (1.2, .12), "microwave": (.5, .4)}


def _grid(points, low, shape):
    return np.round((np.asarray(points, float)-low)/CELL).astype(np.int32)


def _orthogonal(contour, eps):
    """윤곽을 가로·세로 변만 남도록 정리한다 (실내 도면은 대부분 직각)."""
    pts = cv2.approxPolyDP(contour.reshape(-1, 1, 2).astype(np.float32), eps, True).reshape(-1, 2).astype(float)
    if len(pts) < 4:
        x0, y0 = contour.min(0); x1, y1 = contour.max(0)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    n = len(pts)
    edges = []
    for i in range(n):
        a, b = pts[i], pts[(i+1) % n]
        horizontal = abs(b[0]-a[0]) >= abs(b[1]-a[1])
        edges.append(("h", (a[1]+b[1])/2) if horizontal else ("v", (a[0]+b[0])/2))
    merged = []
    for e in edges:
        if merged and merged[-1][0] == e[0]:
            merged[-1] = (e[0], (merged[-1][1]+e[1])/2)
        else:
            merged.append(e)
    if len(merged) > 1 and merged[0][0] == merged[-1][0]:
        merged[0] = (merged[0][0], (merged[0][1]+merged[-1][1])/2); merged.pop()
    if len(merged) < 4 or len(merged) % 2:
        x0, y0 = contour.min(0); x1, y1 = contour.max(0)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    out = []
    for i in range(len(merged)):
        a, b = merged[i-1], merged[i]
        out.append([a[1], b[1]] if a[0] == "v" else [b[1], a[1]])
    return np.array(out, float)


def build_plan(scene):
    """scene → 도면 기하(미터). 공간 후보가 없으면 방 상자를 한 공간으로 쓴다."""
    graph = scene.get("property_graph") or {}
    spaces = [s for s in graph.get("spaces", []) if len(s.get("polygon_xz", [])) >= 3]
    box_ok = scene.get("layout", {}).get("room_box_supported", True) is not False and "room" in scene
    if not spaces or (len(spaces) == 1 and box_ok):  # 방 하나면 측정한 직사각형 방 상자를 쓴다
        room = scene["room"]
        usage = spaces[0].get("usage_candidates", []) if spaces else []
        estimate = spaces[0].get("usage_estimate") if spaces else None
        x0, z0, x1, z1 = room["min"][0], room["min"][2], room["max"][0], room["max"][2]
        spaces = [{"id": "room", "polygon_xz": [[x0, z0], [x1, z0], [x1, z1], [x0, z1]], "usage_candidates": usage,
                   "usage_estimate": estimate}]
    allp = np.vstack([np.array(s["polygon_xz"], float) for s in spaces])
    low = allp.min(0)-1.
    shape = (int(np.ceil((allp.max(0)[1]-low[1]+1.)/CELL))+1, int(np.ceil((allp.max(0)[0]-low[0]+1.)/CELL))+1)
    labels = np.zeros(shape, np.int32)
    order = sorted(range(len(spaces)), key=lambda i: -cv2.contourArea(np.array(spaces[i]["polygon_xz"], np.float32)))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(.4/CELL) | 1,)*2)
    for i in order:
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [_grid(spaces[i]["polygon_xz"], low, shape)], 1)
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        labels[(mask > 0) & (labels == 0)] = i+1
    rooms = []
    for i, space in enumerate(spaces):
        mask = (labels == i+1).astype(np.uint8)
        if mask.sum()*CELL**2 < .8:
            continue
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = _orthogonal(max(found, key=cv2.contourArea).reshape(-1, 2), .12/CELL)
        contour = np.round(contour).astype(np.float64)
        mask = np.zeros(shape, np.uint8); cv2.fillPoly(mask, [np.round(contour).astype(np.int32)], 1)
        mask &= (labels == 0) | (labels == i+1)
        labels[labels == i+1] = 0; labels[mask > 0] = i+1
        if mask.sum()*CELL**2 < .8:
            continue
        usage = [u["usage"] for u in space.get("usage_candidates", [])]
        estimate = space.get("usage_estimate") or {}
        if estimate.get("usage"):  # 사물 근거 우선, 없으면 공간 기하 모델 추정(room_types.estimate_usage)
            usage = [estimate["usage"]]
        ys, xs = np.nonzero(mask)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        grown, _ = cv2.findContours(cv2.dilate(mask, np.ones((3, 3), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rooms.append({"id": space["id"], "label_index": i, "mask": mask,
                      "outline": (max(grown, key=cv2.contourArea).reshape(-1, 2)*CELL+low).tolist(),
                      "area_m2": float(cv2.contourArea(contour.astype(np.float32))*CELL**2), "usage": usage[0] if len(usage) == 1 else None,
                      "anchor": [cx*CELL+low[0], cy*CELL+low[1]],
                      "bbox": [contour[:, 0].min()*CELL+low[0], contour[:, 1].min()*CELL+low[1],
                               contour[:, 0].max()*CELL+low[0], contour[:, 1].max()*CELL+low[1]]})
    # 붙어 있는 두 방 사이에도 벽 두께가 보이도록 경계에서 양쪽을 반씩 깎는다.
    half = max(1, int(round(WALL/2/CELL)))
    trimmed = labels.copy()
    for room in rooms:
        i = room["label_index"]+1
        others = ((labels > 0) & (labels != i)).astype(np.uint8)
        near = cv2.dilate(others, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*half+1,)*2)) > 0
        trimmed[(labels == i) & near] = 0
    labels = trimmed
    for room in rooms:
        mask = (labels == room["label_index"]+1).astype(np.uint8)
        grown, _ = cv2.findContours(cv2.dilate(mask, np.ones((3, 3), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        room["mask"] = mask  # 면적·치수는 깎기 전 관측 영역 기준을 유지한다
        raw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        room["edge"] = (max(raw, key=cv2.contourArea).reshape(-1, 2)*CELL+low+CELL/2).tolist() if raw else []
        room["outline"] = (max(grown, key=cv2.contourArea).reshape(-1, 2)*CELL+low).tolist()
    occupied = labels > 0
    wall_px = int(round(WALL/CELL))
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1,)*2)
    # 집 외곽: 방들을 1m로 닫고 구멍을 채운 범위. 그 안의 빈 칸 중 얇은 것은 벽, 넓은 것은 미관측.
    closed = cv2.morphologyEx(occupied.astype(np.uint8), cv2.MORPH_CLOSE, disk(int(.5/CELL)))
    outer, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    footprint = np.zeros(shape, np.uint8); cv2.drawContours(footprint, outer, -1, 1, -1)
    unknown = (footprint > 0) & ~occupied
    wide = cv2.morphologyEx(unknown.astype(np.uint8), cv2.MORPH_OPEN, disk(int(.25/CELL))) > 0
    band = cv2.dilate(footprint, disk(wall_px)) > 0
    wall = (band & ~occupied & ~wide) | (unknown & ~wide)
    wall |= (cv2.dilate(wide.astype(np.uint8), disk(2)) > 0) & ~wide
    wall &= ~occupied
    found_u, _ = cv2.findContours(wide.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    unobserved = [(c.reshape(-1, 2)*CELL+low).tolist() for c in found_u if cv2.contourArea(c)*CELL**2 >= .3]
    doors = []
    id_to_room = {r["id"]: r for r in rooms}
    for portal in graph.get("portals", []):
        if portal.get("status") != "image_supported":
            continue
        a, b = (id_to_room.get(s) for s in portal["spaces"])
        if a is None or b is None:
            continue
        c = np.array(portal["center_xz"], float)
        pc = _grid(c, low, shape)
        def nearest(room):
            ys, xs = np.nonzero(room["mask"]); d = (xs-pc[0])**2+(ys-pc[1])**2; j = np.argmin(d)
            return np.array([xs[j], ys[j]])*CELL+low
        pa, pb = nearest(a), nearest(b)
        normal = pb-pa
        axis = 0 if abs(normal[0]) >= abs(normal[1]) else 1  # 벽을 가로지르는 축
        width = float(np.clip(portal["width_m"], .7, 2.4))
        mid = (pa+pb)/2
        along = 1-axis
        lo, hi = mid[along]-width/2, mid[along]+width/2
        cut = np.zeros(shape, np.uint8)
        p0 = np.array([0., 0.]); p1 = np.array([0., 0.])
        p0[along], p1[along] = lo, hi
        p0[axis], p1[axis] = min(pa[axis], pb[axis])-.05, max(pa[axis], pb[axis])+.05
        g0, g1 = _grid(p0, low, shape), _grid(p1, low, shape)
        cv2.rectangle(cut, tuple(map(int, g0)), tuple(map(int, g1)), 1, -1)
        wall &= cut == 0
        small = a if a["area_m2"] <= b["area_m2"] else b
        inner = pa if small is a else pb
        doors.append({"cut": [p0.tolist(), p1.tolist()], "axis": axis, "lo": lo, "hi": hi, "wall_a": float(min(pa[axis], pb[axis])), "wall_b": float(max(pa[axis], pb[axis])),
                      "into": float(inner[axis]), "direction": 1. if inner[axis] > mid[axis] else -1., "width": width,
                      "leaf": width <= 1.2})
    found, hierarchy = cv2.findContours(wall.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    walls = []  # (바깥 윤곽, [구멍]) — 구멍까지 채우지 않도록 짝지어 둔다
    if hierarchy is not None:
        for i, c in enumerate(found):
            if hierarchy[0][i][3] != -1 or cv2.contourArea(c) < 4: continue
            holes, child = [], hierarchy[0][i][2]
            while child != -1:
                holes.append((found[child].reshape(-1, 2)*CELL+low).tolist()); child = hierarchy[0][child][0]
            walls.append(((c.reshape(-1, 2)*CELL+low).tolist(), holes))
    objects = []
    # 가구 아이콘도 용도 후보와 같은 강한 근거(사진 3장·점수 0.6)만 그린다. 주방 가전·세면대는 오검출이
    # 잦아(TUM 사무실: 수납장→냉장고, 접시→싱크대) 해당 공간에 주방·욕실 후보가 있을 때만 그린다.
    contexts = {}
    for room in rooms:
        for u in spaces[room["label_index"]].get("usage_candidates", []):
            contexts.setdefault(u["usage"], []).append(room)
    def in_rooms(x, z, usage):
        g = _grid([x, z], low, shape)
        return any(r["mask"][min(max(g[1], 0), shape[0]-1), min(max(g[0], 0), shape[1]-1)] for r in contexts.get(usage, []))
    for o in (scene.get("objects") or {}).get("instances", []):
        if o["class"] not in FURNITURE or o.get("view_count", 0) < 3 or o.get("max_score", 0) < .6:
            continue
        x, z = o["position"][0], o["position"][2]
        if o["class"] in ("refrigerator", "oven", "microwave") and not in_rooms(x, z, "kitchen"): continue
        if o["class"] == "sink" and not (in_rooms(x, z, "kitchen") or in_rooms(x, z, "bathroom")): continue
        objects.append({"class": o["class"], "label": o["label"], "x": x, "z": z})
    return {"rooms": rooms, "walls": walls, "unobserved": unobserved, "wall_mask": wall, "low": low.tolist(), "shape": shape, "doors": doors,
            "objects": objects, "extent": [allp.min(0).tolist(), allp.max(0).tolist()]}


def _chain(values, merge=.25, min_len=.45):
    values = sorted(values)
    merged = []
    for v in values:
        if merged and v-merged[-1][-1] < merge: merged[-1].append(v)
        else: merged.append([v])
    points = [float(np.mean(m)) for m in merged]
    out = [points[0]]
    for p in points[1:]:
        if p-out[-1] >= min_len: out.append(p)
    return out


def _names(rooms):
    count = {}
    for r in sorted(rooms, key=lambda r: (r["anchor"][1], r["anchor"][0])):
        if r["usage"]:
            base = USAGE_NAME[r["usage"]]
            count[base] = count.get(base, 0)+1
            r["name"] = base if base in ("거실", "주방") else f"{base}{count[base]}"
            r["estimated"] = True
        else:
            count["공간"] = count.get("공간", 0)+1
            r["name"] = f"공간 {count['공간']}"
            r["estimated"] = False


def render(scene, out_dir: Path, title=None):
    plan = build_plan(scene)
    if not plan["rooms"]:
        return None
    _names(plan["rooms"])
    (ex0, ez0), (ex1, ez1) = plan["extent"]
    W, H = 1400, 1100
    margin_l, margin_t, margin_r, margin_b = 150, 230, 190, 150
    s = min((W-margin_l-margin_r)/max(ex1-ex0, .5), (H-margin_t-margin_b)/max(ez1-ez0, .5))
    ox = margin_l+((W-margin_l-margin_r)-(ex1-ex0)*s)/2-ex0*s
    oz = margin_t+((H-margin_t-margin_b)-(ez1-ez0)*s)/2-ez0*s
    P = lambda x, z: (ox+x*s, oz+z*s)
    prims = []
    for r in plan["rooms"]:
        pts = [P(x, z) for x, z in r["outline"]]
        prims.append(("poly", pts, "tile" if r["usage"] == "bathroom" else "wood"))
    for outer, holes in plan["walls"]:
        prims.append(("wall", [P(x, z) for x, z in outer], [[P(x, z) for x, z in h] for h in holes]))
    for r in plan["rooms"]:  # 방 경계를 벽 두께로 덧그려 방 사이 벽이 얇게 보이지 않게 한다
        if r.get("edge"): prims.append(("stroke", [P(x, z) for x, z in r["edge"]], WALL*s))
    for d in plan["doors"]:  # 문턱도 바닥으로 이어서 그린다
        (x0, z0), (x1, z1) = d["cut"]
        prims.append(("poly", [P(x0, z0), P(x1, z0), P(x1, z1), P(x0, z1)], "wood"))
    for u in plan["unobserved"]:
        prims.append(("poly", [P(x, z) for x, z in u], "hatch"))
    for o in plan["objects"]:
        w, d = FURNITURE[o["class"]]
        prims.append(("furniture", o["class"], P(o["x"], o["z"]), (w*s, d*s)))

    for d in plan["doors"]:
        if not d["leaf"]:
            continue
        hinge = [0., 0.]; hinge[1-d["axis"]] = d["lo"]; hinge[d["axis"]] = d["wall_a"] if d["direction"] > 0 else d["wall_b"]
        leaf_end = list(hinge); leaf_end[d["axis"]] += d["direction"]*d["width"]
        other = list(hinge); other[1-d["axis"]] = d["hi"]
        prims.append(("door", P(*hinge), P(*leaf_end), P(*other), d["width"]*s))
    for r in plan["rooms"]:
        x, y = P(*r["anchor"])
        area = r["area_m2"]
        prims.append(("text", x, y-6, r["name"], 21, True, COLORS["text"], "middle"))
        sub = f"{area:.1f}㎡ ({area/PYEONG:.1f}평)" + (" · 추정" if r["estimated"] else "")
        prims.append(("text", x, y+20, sub, 14, False, COLORS["sub"], "middle"))
    xs = _chain([v for r in plan["rooms"] for v in (r["bbox"][0], r["bbox"][2])])
    zs = _chain([v for r in plan["rooms"] for v in (r["bbox"][1], r["bbox"][3])])
    top = oz+ez0*s-60
    right = ox+ex1*s+60
    for chain, horizontal in ((xs, True), (zs, False)):
        for k, v in enumerate(chain):
            if horizontal:
                prims.append(("line", (ox+v*s, top-12), (ox+v*s, top+12), COLORS["dim"], 1.4))
            else:
                prims.append(("line", (right-12, oz+v*s), (right+12, oz+v*s), COLORS["dim"], 1.4))
        if horizontal:
            prims.append(("line", (ox+chain[0]*s, top), (ox+chain[-1]*s, top), COLORS["dim"], 1))
            prims.append(("line", (ox+chain[0]*s, top-40), (ox+chain[-1]*s, top-40), COLORS["dim"], 1))
            prims.append(("text", (ox+(chain[0]+chain[-1])/2*s), top-48, f"{round((chain[-1]-chain[0])*100)*10:,}", 15, True, COLORS["dim"], "middle"))
        else:
            prims.append(("line", (right, oz+chain[0]*s), (right, oz+chain[-1]*s), COLORS["dim"], 1))
            prims.append(("line", (right+40, oz+chain[0]*s), (right+40, oz+chain[-1]*s), COLORS["dim"], 1))
            prims.append(("text", right+54, oz+(chain[0]+chain[-1])/2*s, f"{round((chain[-1]-chain[0])*100)*10:,}", 15, True, COLORS["dim"], "vstart"))
        for a, b in zip(chain[:-1], chain[1:]):
            label = f"{round((b-a)*100)*10:,}"
            if horizontal: prims.append(("text", ox+(a+b)/2*s, top-8, label, 13, False, COLORS["dim"], "middle"))
            else: prims.append(("text", right+8, oz+(a+b)/2*s, label, 13, False, COLORS["dim"], "vstart"))
    total = sum(r["area_m2"] for r in plan["rooms"])
    heading = title or scene.get("name", "평면도")
    scale_note = "마커 기준 크기" if scene.get("scale", {}).get("method") == "marker" else "촬영 높이 가정 크기 · 실측 보정 필요"
    prims.append(("text", 60, 62, f"{heading} 평면도", 28, True, COLORS["text"], "start"))
    prims.append(("text", 60, 98, f"관측 실내 면적 합계 {total:.1f}㎡ ({total/PYEONG:.1f}평) · 공간 {len(plan['rooms'])}개 · 문 {len(plan['doors'])}개", 16, False, COLORS["sub"], "start"))
    prims.append(("text", 60, H-58, f"사진으로 추정한 도면입니다. 치수(mm)·면적은 관측 영역 기준이며 실측과 다를 수 있습니다. {scale_note}.", 14, False, COLORS["sub"], "start"))
    prims.append(("text", 60, H-34, "실 이름의 '추정'은 사진 속 사물(변기·침대·소파·주방 가전) 또는 공간 크기·모양·연결 학습 모델 근거입니다. 확정이 아닙니다.", 14, False, COLORS["sub"], "start"))
    out_dir = Path(out_dir)
    svg = _svg(prims, W, H)
    (out_dir/"plan.svg").write_text(svg, encoding="utf-8")
    _png(prims, W, H, out_dir/"plan.png")
    return {"rooms": [{"id": r["id"], "name": r["name"], "area_m2": round(r["area_m2"], 2), "estimated": r["estimated"]} for r in plan["rooms"]],
            "doors": len(plan["doors"]), "objects": len(plan["objects"]), "total_area_m2": round(total, 2), "files": ["plan.svg", "plan.png"]}


def _furniture_shapes(kind, center, size):
    """가구 선화: (사각형 목록, 원 목록) 화면 좌표."""
    cx, cy = center; w, h = size
    x0, y0 = cx-w/2, cy-h/2
    rects, circles = [(x0, y0, w, h, 6)], []
    if kind == "bed":
        rects += [(x0+w*.08, y0+h*.05, w*.38, h*.14, 4), (x0+w*.54, y0+h*.05, w*.38, h*.14, 4), (x0, y0+h*.3, w, h*.7, 0)]
    elif kind == "couch":
        rects += [(x0, y0, w, h*.28, 4), (x0, y0, w*.12, h, 4), (x0+w*.88, y0, w*.12, h, 4)]
    elif kind == "toilet":
        rects = [(x0+w*.1, y0, w*.8, h*.28, 4)]; circles.append((cx, y0+h*.62, w*.42, h*.36))
    elif kind == "sink":
        circles.append((cx, cy, w*.32, h*.3))
    elif kind == "dining table":
        for dx in (-.3, .3):
            for dy in (-1, 1):
                rects.append((cx+dx*w-w*.11, cy+dy*(h/2+h*.18)-h*.14, w*.22, h*.28, 4))
    elif kind in ("refrigerator", "oven", "microwave"):
        rects.append((x0+w*.1, y0+h*.1, w*.8, h*.8, 2))
    return rects, circles


def _svg(prims, W, H):
    defs = f'''<defs>
<pattern id="wood" width="120" height="22" patternUnits="userSpaceOnUse"><rect width="120" height="22" fill="{COLORS['wood']}"/>
<line x1="0" y1="21.5" x2="120" y2="21.5" stroke="{COLORS['wood_line']}" stroke-width="1"/><line x1="70" y1="0" x2="70" y2="22" stroke="{COLORS['wood_line']}" stroke-width="1"/></pattern>
<pattern id="tile" width="24" height="24" patternUnits="userSpaceOnUse"><rect width="24" height="24" fill="{COLORS['tile']}"/>
<path d="M0 23.5H24M23.5 0V24" stroke="{COLORS['tile_line']}" stroke-width="1"/></pattern>
<pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="10" height="10" fill="#f1f1f1"/><line x1="0" y1="0" x2="0" y2="10" stroke="#c4c4c4" stroke-width="2"/></pattern></defs>'''
    body = []
    for p in prims:
        if p[0] == "poly":
            body.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in p[1])}" fill="url(#{p[2]})"/>')
        elif p[0] == "wall":
            d = "".join("M"+"L".join(f"{x:.1f} {y:.1f}" for x, y in ring)+"Z" for ring in [p[1]]+p[2])
            body.append(f'<path d="{d}" fill="{COLORS["wall"]}" fill-rule="evenodd"/>')
        elif p[0] == "door":
            (hx, hy), (lx, ly), (ox_, oy_), r = p[1], p[2], p[3], p[4]
            sweep = 1 if ((lx-hx)*(oy_-hy)-(ly-hy)*(ox_-hx)) < 0 else 0
            body.append(f'<line x1="{hx:.1f}" y1="{hy:.1f}" x2="{lx:.1f}" y2="{ly:.1f}" stroke="{COLORS["door"]}" stroke-width="2.4"/>')
            body.append(f'<path d="M{lx:.1f} {ly:.1f} A{r:.1f} {r:.1f} 0 0 {sweep} {ox_:.1f} {oy_:.1f}" fill="none" stroke="{COLORS["door"]}" stroke-width="1.2" stroke-dasharray="4 3"/>')
        elif p[0] == "furniture":
            rects, circles = _furniture_shapes(p[1], p[2], p[3])
            for x, y, w, h, rr in rects:
                body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rr}" fill="#fafafa" fill-opacity=".55" stroke="{COLORS["furniture"]}" stroke-width="1.4"/>')
            for x, y, rx, ry in circles:
                body.append(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="#fafafa" stroke="{COLORS["furniture"]}" stroke-width="1.4"/>')
        elif p[0] == "stroke":
            body.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in p[1])}" fill="none" stroke="{COLORS["wall"]}" stroke-width="{p[2]:.1f}" stroke-linejoin="miter"/>')
        elif p[0] == "line":
            body.append(f'<line x1="{p[1][0]:.1f}" y1="{p[1][1]:.1f}" x2="{p[2][0]:.1f}" y2="{p[2][1]:.1f}" stroke="{p[3]}" stroke-width="{p[4]}"/>')
        elif p[0] == "text":
            _, x, y, text, size, bold, color, anchor = p
            weight = ' font-weight="700"' if bold else ""
            if anchor == "vstart":
                body.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}"{weight} text-anchor="middle" transform="rotate(90 {x:.1f} {y:.1f})">{escape(text)}</text>')
            else:
                body.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}"{weight} text-anchor="{anchor}">{escape(text)}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="사진 기반 추정 평면도">'
            f'{defs}<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/><g font-family="Malgun Gothic, Apple SD Gothic Neo, sans-serif">{"".join(body)}</g></svg>')


def _png(prims, W, H, path):
    from PIL import Image, ImageDraw, ImageFont
    k = 2
    img = Image.new("RGB", (W*k, H*k), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    fonts = {}
    def font(size, bold):
        key = (size, bold)
        if key not in fonts:
            f = FONT_BOLD if bold and FONT_BOLD.exists() else FONT
            fonts[key] = ImageFont.truetype(str(f), size*k) if f.exists() else ImageFont.load_default()
        return fonts[key]
    S = lambda pts: [(x*k, y*k) for x, y in pts]
    for p in prims:
        if p[0] == "poly":
            mask = Image.new("L", img.size, 0); ImageDraw.Draw(mask).polygon(S(p[1]), fill=255)
            layer = Image.new("RGB", img.size, {"hatch": "#f1f1f1"}.get(p[2], COLORS.get(p[2], "#f1f1f1"))); ld = ImageDraw.Draw(layer)
            if p[2] == "hatch":
                for c in range(-img.size[1], img.size[0], 14*k): ld.line([(c, 0), (c+img.size[1], img.size[1])], fill="#c4c4c4", width=2*k)
            elif p[2] == "wood":
                for y in range(0, img.size[1], 22*k): ld.line([(0, y), (img.size[0], y)], fill=COLORS["wood_line"], width=k)
                for row, y in enumerate(range(0, img.size[1], 22*k)):
                    for x in range((row*45*k) % (120*k), img.size[0], 120*k): ld.line([(x, y), (x, y+22*k)], fill=COLORS["wood_line"], width=k)
            else:
                for y in range(0, img.size[1], 24*k): ld.line([(0, y), (img.size[0], y)], fill=COLORS["tile_line"], width=k)
                for x in range(0, img.size[0], 24*k): ld.line([(x, 0), (x, img.size[1])], fill=COLORS["tile_line"], width=k)
            img.paste(layer, (0, 0), mask)
        elif p[0] == "wall":
            mask = Image.new("L", img.size, 0); md = ImageDraw.Draw(mask)
            md.polygon(S(p[1]), fill=255)
            for hole in p[2]: md.polygon(S(hole), fill=0)
            img.paste(Image.new("RGB", img.size, COLORS["wall"]), (0, 0), mask)
        elif p[0] == "door":
            (hx, hy), (lx, ly), (ox_, oy_), r = p[1], p[2], p[3], p[4]
            draw.line([(hx*k, hy*k), (lx*k, ly*k)], fill=COLORS["door"], width=int(2.4*k))
            a0 = math.degrees(math.atan2(ly-hy, lx-hx)); a1 = math.degrees(math.atan2(oy_-hy, ox_-hx))
            if (a1-a0) % 360 > 180: a0, a1 = a1, a0
            draw.arc([(hx-r)*k, (hy-r)*k, (hx+r)*k, (hy+r)*k], a0, a1, fill=COLORS["door"], width=k)
        elif p[0] == "furniture":
            rects, circles = _furniture_shapes(p[1], p[2], p[3])
            for x, y, w, h, rr in rects:
                draw.rounded_rectangle([x*k, y*k, (x+w)*k, (y+h)*k], radius=rr*k, outline=COLORS["furniture"], width=int(1.4*k))
            for x, y, rx, ry in circles:
                draw.ellipse([(x-rx)*k, (y-ry)*k, (x+rx)*k, (y+ry)*k], outline=COLORS["furniture"], width=int(1.4*k), fill="#fafafa")
        elif p[0] == "stroke":
            pts = S(p[1]); w = max(1, int(p[2]*k))
            draw.line(pts+[pts[0], pts[1]], fill=COLORS["wall"], width=w, joint="curve")
        elif p[0] == "line":
            draw.line([(p[1][0]*k, p[1][1]*k), (p[2][0]*k, p[2][1]*k)], fill=p[3], width=max(1, int(p[4]*k)))
        elif p[0] == "text":
            _, x, y, text, size, bold, color, anchor = p
            f = font(size, bold)
            if anchor == "vstart":
                box = draw.textbbox((0, 0), text, font=f); tw, th = box[2]-box[0], box[3]-box[1]
                t = Image.new("RGBA", (tw+4, th+8), (255, 255, 255, 0)); ImageDraw.Draw(t).text((2, -box[1]+2), text, font=f, fill=color)
                t = t.rotate(-90, expand=True); img.paste(t, (int(x*k-t.size[0]/2), int(y*k-t.size[1]/2)), t)
            else:
                ax = {"middle": "ms", "start": "ls", "end": "rs"}[anchor]
                draw.text((x*k, y*k), text, font=f, fill=color, anchor=ax)
    img = img.resize((W, H), Image.LANCZOS)
    img.save(path, optimize=True)
