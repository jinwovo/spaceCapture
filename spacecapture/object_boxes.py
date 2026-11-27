"""가구 크기·방향 상자: 여러 사진의 2D 검출 상자와 맞도록 3D 상자를 맞춘다.

테슬라 화면처럼 인식한 사물을 '깔끔한 3D 모형'으로 다시 그리려면 위치뿐 아니라 크기·방향이 필요하다.
SfM 점은 가구 표면에 드물게 찍혀서 점만으로 상자를 만들기 어렵다. 대신 사진마다 검출 상자가 있고
카메라 위치·방향·초점거리를 알므로, 3D 상자의 8개 꼭짓점을 각 사진에 투영한 외곽 사각형이 검출 상자와
같아지도록 (중심 x·z, 가로·깊이·높이, 필요하면 바닥 높이)를 찾는다.

- 바닥에 놓이는 가구(침대·소파·의자·식탁·변기·냉장고 등)는 밑면을 바닥(y=0)에 둔다.
- 방향은 벽 축(장면 x/z축)에 맞춘다(가로·깊이를 따로 풀어 90° 회전도 포함). 의자는 45°도 시험한다.
- 화면 가장자리에 잘린 검출 변은 쓰지 않는다(물체가 화면 밖으로 이어짐).
- 종류별 일반 크기를 약한 사전값으로 쓰고, 맞춘 크기가 종류별 허용 범위를 벗어나거나 사진과 맞지 않으면
  `reliable=false`로 남긴다(모형을 그리지 않음).
정답은 쓰지 않는다. 좌표는 장면 좌표(미터, +Y 위, 바닥 y=0).
"""
from __future__ import annotations

import numpy as np

# 종류: (가로, 깊이, 높이) 일반값, (최소, 최대) 각 축, 바닥 접지 여부.  가로 ≥ 깊이로 정의(방향은 따로 풂).
PRIORS = {
    "bed": ((1.5, 2.05, .5), ((.8, 1.7, .25), (2.1, 2.4, 1.2)), True),
    "couch": ((1.9, .9, .85), ((1.0, .6, .5), (3.2, 1.8, 1.1)), True),
    "chair": ((.5, .52, .9), ((.35, .35, .6), (.8, .8, 1.3)), True),
    "dining table": ((1.2, .8, .74), ((.6, .5, .6), (2.4, 1.3, .95)), True),
    "bench": ((1.2, .4, .45), ((.6, .25, .3), (2.2, .8, .9)), True),
    "toilet": ((.4, .68, .75), ((.3, .45, .55), (.6, .9, 1.0)), True),
    "refrigerator": ((.8, .75, 1.8), ((.5, .5, 1.2), (1.2, .95, 2.2)), True),
    "oven": ((.6, .6, .9), ((.45, .45, .4), (1.0, .8, 1.0)), True),
    "sink": ((.6, .5, .85), ((.35, .3, .6), (1.6, .8, 1.0)), True),
    "potted plant": ((.4, .4, .7), ((.15, .15, .2), (1.0, 1.0, 2.0)), True),
    "tv": ((1.1, .12, .65), ((.5, .03, .3), (2.0, .5, 1.2)), False),
    "microwave": ((.5, .38, .3), ((.35, .25, .2), (.7, .55, .45)), False),
}
EDGE_PX = 4.        # 이 안쪽이면 화면에 잘린 변
MIN_VIEWS = 3       # 상자를 맞추는 최소 사진 수
MAX_ERROR = .12     # 변 오차(검출 상자 대각선 대비) 중앙값 허용치
# 속이 찬 가구는 아래 바닥을 가린다. 상자 바닥면 안쪽에 바닥 점(y<6cm)이 많이 보이면 바닥·벽을 잘못 본 것.
# (TUM 실제 냉장고·모니터 0~8%, 무늬 바닥을 침대로 잘못 본 경우 47~100%)
SOLID = {"bed", "couch", "refrigerator", "oven", "toilet"}
MAX_FLOOR_VISIBLE = .3
MIN_FOOT_POINTS = 20
# 바닥 접지 가구의 밑면: "fixed"는 y=0 고정, "soft"는 밑면도 풀되 바닥(y=0) 쪽으로 당긴다(σ=floor_sigma_m).
# 장면 바닥 추정이 틀리면(층이 나뉜 집·창밖 지면) 고정 접지가 상자 높이를 크게 늘린다.
SETTINGS = {"floor_mode": "fixed", "floor_sigma_m": .15, "floor_weight": .002,
            "reassociate": 1,          # 맞춘 상자 중심으로 사진별 검출을 다시 대응시켜 한 번 더 맞춤
            "min_reliable_views": 6,   # 믿을 만한 상자의 최소 사진 수(Hypersim 개발 장면에서 선택: 점·중심 중복 규칙과 함께 6)
            "dup_overlap": .5,         # 같은 계열 가구끼리 작은 쪽 바닥면의 이 비율 이상 겹치면 중복
            "outside_weight": 1.,      # <1이면 투영이 검출 밖으로 나가는 것을 덜 벌함(가림 대응)
            "floor_ray": False, "ray_min_score": .35, "ray_radius_frac": .5,
            "min_inside_points": 1,    # 상자 안 SfM 점이 이보다 적으면 믿지 않음(허상 TV 등 점 0개)
            "ray_min_inside_points": 10,
            "dup_center_frac": .5}     # 같은 계열 두 상자 중심 거리가 (큰 쪽 가로 × 이 값)보다 가까우면 중복  # 바닥 광선 후보(바닥 오인 시 허상)는 상자 안 점 10개 이상  # 바닥 광선 후보(SfM 점 없이 사물 찾기)
# 같은 가구가 다른 종류로도 잡힌다(소파 → 소파+의자). 계열 안에서 중복을 합친다.
FAMILY = {"couch": "seat", "chair": "seat", "bench": "seat", "toilet": "seat", "dining table": "table"}


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def corners(center_xz, size, yaw, bottom=0.):
    """상자 꼭짓점 8개(장면 좌표). size=(가로 x, 깊이 z, 높이)."""
    w, d, h = size
    local = np.array([[sx*w/2, y, sz*d/2] for sx in (-1, 1) for sz in (-1, 1) for y in (0., h)])
    return local@_rot(yaw).T+np.array([center_xz[0], bottom, center_xz[1]])


def project_box(P, cam):
    """꼭짓점 → 화면 외곽 사각형. 카메라 뒤로 넘어가면 None."""
    C, R, (iw, ih, f) = cam
    X = (P-C)@R.T
    if (X[:, 2] < .05).any():
        return None
    uv = X[:, :2]/X[:, 2:3]*f+[iw/2, ih/2]
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


OUTWARD = np.array([-1., -1., 1., 1.])


def _residuals(params, views, yaw, grounded):
    cx, cz, lw, ld, lh = params[:5]
    bottom = params[5] if len(params) > 5 else 0.
    P = corners((cx, cz), np.exp([lw, ld, lh]), yaw, bottom)
    out = []
    for box, cam, use in views:
        proj = project_box(P, cam)
        scale = np.hypot(box[2]-box[0], box[3]-box[1])
        if proj is None:
            out.extend([1.]*int(use.sum())); continue
        out.extend(((proj-box)*OUTWARD/scale)[use])   # 양수 = 투영이 검출보다 바깥
    return np.asarray(out)


def _loss(params, views, yaw, grounded, prior, center0):
    r = _residuals(params, views, yaw, grounded)
    huber = np.where(np.abs(r) < .1, .5*r**2, .1*(np.abs(r)-.05))
    # 가림 때문에 검출 상자는 실제보다 작게 잡히기 쉽다(하한). 투영이 검출보다 바깥인 쪽은 덜 벌한다.
    huber = huber*np.where(r > 0, SETTINGS["outside_weight"], 1.)
    size_pen = np.sum(((params[2:5]-np.log(prior))/.35)**2)*.002   # 일반 크기 쪽으로 약하게
    move_pen = np.sum((params[:2]-center0)**2)/(.8**2)*.002
    floor_pen = (params[5]/SETTINGS["floor_sigma_m"])**2*SETTINGS["floor_weight"] if grounded and len(params) > 5 else 0.
    return float(huber.mean())+size_pen+move_pen+floor_pen


def _nelder_mead(f, x0, step, iters=600, tol=1e-7):
    n = len(x0)
    pts = [np.asarray(x0, float)]+[np.asarray(x0, float)+np.eye(n)[i]*step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals); pts = [pts[i] for i in order]; vals = [vals[i] for i in order]
        if abs(vals[-1]-vals[0]) < tol: break
        c = np.mean(pts[:-1], 0)
        xr = c+(c-pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c+2*(c-pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c+.5*(pts[-1]-c); fc = f(xc)
            if fc < vals[-1]: pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]]+[pts[0]+.5*(p-pts[0]) for p in pts[1:]]; vals = [vals[0]]+[f(p) for p in pts[1:]]
    k = int(np.argmin(vals))
    return pts[k], vals[k]


def associate(instance, detections_by_photo, cams, labels):
    """사물 위치를 각 사진에 투영해 같은 종류 검출 상자 중 그 점을 품은(가장 가까운) 상자를 고른다."""
    views = []
    p = np.asarray(instance["position"], float)
    for name, dets in detections_by_photo.items():
        if name not in cams: continue
        C, R, (iw, ih, f) = cams[name]
        X = (p-C)@R.T
        if X[2] < .1: continue
        u, v = X[:2]/X[2]*f+[iw/2, ih/2]
        best = None
        for d in dets:
            if labels.get(d.get("class") or d.get("label")) != instance["class"]: continue
            x0, y0, x1, y1 = d["box"]
            pad = .1*max(x1-x0, y1-y0)
            if not (x0-pad <= u <= x1+pad and y0-pad <= v <= y1+pad): continue
            dist = np.hypot(u-(x0+x1)/2, v-(y0+y1)/2)
            if best is None or dist < best[0]: best = (dist, d)
        if best:
            box = np.asarray(best[1]["box"], float)
            use = np.array([box[0] > EDGE_PX, box[1] > EDGE_PX, box[2] < iw-EDGE_PX, box[3] < ih-EDGE_PX])
            if use.sum() >= 2: views.append((box, cams[name], use))
    return views


def fit_box(instance, views):
    """한 사물의 3D 상자. 사진이 부족하거나 종류 사전값이 없으면 None."""
    cls = instance["class"]
    if cls not in PRIORS or len(views) < MIN_VIEWS:
        return None
    prior, (lo, hi), grounded = PRIORS[cls]
    center0 = np.asarray(instance["position"], float)[[0, 2]]
    yaws = [0., np.pi/4] if cls == "chair" else [0.]
    best = None
    for yaw in yaws:
        for swap in (False, True):  # 가로·깊이를 바꾼 시작점(90° 회전)
            pw, pd, ph = prior
            if swap: pw, pd = pd, pw
            x0 = [center0[0], center0[1], np.log(pw), np.log(pd), np.log(ph)]
            step = [.3, .3, .3, .3, .25]
            if grounded and SETTINGS["floor_mode"] == "soft":
                x0.append(0.); step.append(.1)
            elif not grounded:
                x0.append(max(.0, float(instance["position"][1])-ph/2)); step.append(.2)
            f = lambda q: _loss(q, views, yaw, grounded, np.array([pw, pd, ph]), center0)
            x, v = _nelder_mead(f, x0, step)
            x, v = _nelder_mead(f, x, [s*.3 for s in step])
            if best is None or v < best[1]: best = (x, v, yaw)
    x, v, yaw = best
    w, d, h = np.exp(x[2:5])
    if w < d:  # 가로 ≥ 깊이로 정리하고 방향에 90°를 더한다
        w, d, yaw = d, w, yaw+np.pi/2
    bottom = float(x[5]) if len(x) > 5 else 0.
    r = np.abs(_residuals(x, views, best[2], grounded))
    error = float(np.median(r))
    in_range = all(lo[i]*.85 <= s <= hi[i]*1.15 for i, s in enumerate((w, d, h)))
    return {"center": [round(float(x[0]), 3), round(bottom+h/2, 3), round(float(x[1]), 3)],
            "size": [round(float(w), 3), round(float(h), 3), round(float(d), 3)],   # 가로(x), 높이(y), 깊이(z) — 회전 전
            "yaw_deg": round(float(np.degrees(yaw)) % 180, 1), "bottom": round(bottom, 3),
            "views": len(views), "edge_error": round(error, 4), "in_class_range": bool(in_range),
            "reliable": bool(in_range and error <= MAX_ERROR)}


def facing(box, walls):
    """등받이·머리 쪽(벽에 붙는 쪽) 방향. 깊이 축 양 끝 중 가까운 벽 쪽을 뒤로 본다. 반환: 앞쪽 단위벡터(x, z)."""
    yaw = np.radians(box["yaw_deg"])
    w, h, d = box["size"]
    c = np.array([box["center"][0], box["center"][2]])
    axis = np.array([np.sin(yaw), np.cos(yaw)])  # 깊이 축(로컬 z)의 장면 방향
    if not walls:
        return (-axis).round(4).tolist()
    def gap(point):
        best = 9.
        for x0, z0, x1, z1 in walls:
            dx = max(x0-point[0], 0, point[0]-x1); dz = max(z0-point[1], 0, point[1]-z1)
            best = min(best, np.hypot(dx, dz))
        return best
    back_plus = gap(c+axis*d/2) <= gap(c-axis*d/2)
    return (-axis if back_plus else axis).round(4).tolist()


def floor_visible(box, points):
    """상자 바닥면(안쪽 80%) 위·상자 높이 아래의 점 중 바닥 높이(y<6cm) 점 비율과 점 수."""
    P = np.asarray(points, float)
    if not len(P): return None, 0
    yaw = np.radians(box["yaw_deg"]); c, s = np.cos(yaw), np.sin(yaw)
    Q = P-np.array(box["center"])
    lx, lz = Q[:, 0]*c-Q[:, 2]*s, Q[:, 0]*s+Q[:, 2]*c
    w, h, d = box["size"]
    foot = (np.abs(lx) < w/2*.8) & (np.abs(lz) < d/2*.8) & (P[:, 1] < box["bottom"]+1.1*h)
    n = int(foot.sum())
    return (float((foot & (P[:, 1] < box["bottom"]+.06)).sum())/n if n else None), n


def floor_ray_candidates(detections_by_photo, cams, labels, min_views=3, max_range=12.):
    """바닥 가구 후보를 SfM 점 없이 만든다: 검출 상자 아래 변 가운데를 지나는 광선과 바닥(y=0)의 교점을
    여러 사진에서 모아 같은 종류끼리 묶는다. 가구 표면에 SfM 점이 적어 기존 위치 추정이 실패해도 후보가 생긴다.
    아래 변이 화면에 잘린 검출은 쓰지 않는다. 반환 형식은 objects.fuse_instances와 같다."""
    hits = {}
    for name, dets in detections_by_photo.items():
        if name not in cams: continue
        C, R, (iw, ih, f) = cams[name]
        for d in dets:
            cls = labels.get(d.get("class") or d.get("label"))
            if cls not in PRIORS or not PRIORS[cls][2]: continue
            x0, y0, x1, y1 = d["box"]
            if y1 >= ih-EDGE_PX or d.get("score", 1) < SETTINGS["ray_min_score"]: continue
            ray = R.T@np.array([((x0+x1)/2-iw/2)/f, (y1-ih/2)/f, 1.])
            if ray[1] > -1e-3: continue                       # 바닥을 향하지 않음
            s = -C[1]/ray[1]
            if s <= 0 or s*np.linalg.norm(ray) > max_range: continue
            P = C+s*ray
            hdir = np.array([ray[0], ray[2]]); hdir /= max(np.linalg.norm(hdir), 1e-9)
            xz = P[[0, 2]]+hdir*PRIORS[cls][0][1]/2         # 아래 변은 앞면 → 깊이 절반만큼 안쪽이 중심
            hits.setdefault(cls, []).append((xz, name, float(d.get("score", 1))))
    out = []
    for cls, items in hits.items():
        radius = max(.3, SETTINGS["ray_radius_frac"]*PRIORS[cls][0][0])
        clusters = []
        for xz, name, score in sorted(items, key=lambda q: -q[2]):
            best = None
            for c in clusters:
                dist = np.linalg.norm(np.median(c["xz"], 0)-xz)
                if dist < radius and name not in c["photos"] and (best is None or dist < best[0]): best = (dist, c)
            if best: best[1]["xz"].append(xz); best[1]["photos"].add(name); best[1]["scores"].append(score)
            else: clusters.append({"xz": [xz], "photos": {name}, "scores": [score]})
        for c in clusters:
            if len(c["photos"]) < min_views: continue
            xz = np.median(c["xz"], 0)
            out.append({"class": cls, "label": cls, "position": [round(float(xz[0]), 3), round(PRIORS[cls][0][2]/2, 3), round(float(xz[1]), 3)],
                        "view_count": len(c["photos"]), "max_score": round(max(c["scores"]), 3), "photos": sorted(c["photos"])[:8],
                        "source": "floor_ray"})
    return out


def inside_points(box, points, shrink=.9):
    """상자 안(가로·깊이 90%, 높이 10~100%)에 있는 SfM 점 수. 표면에 점이 하나도 없는 상자는 근거가 약하다."""
    P = np.asarray(points, float)
    if not len(P): return 0
    yaw = np.radians(box["yaw_deg"]); c, s = np.cos(yaw), np.sin(yaw)
    Q = P-np.array(box["center"])
    lx, lz = Q[:, 0]*c-Q[:, 2]*s, Q[:, 0]*s+Q[:, 2]*c
    w, h, d = box["size"]
    y = P[:, 1]-box["bottom"]
    return int(((np.abs(lx) < w/2*shrink) & (np.abs(lz) < d/2*shrink) & (y > .1*h) & (y < h*1.05)).sum())


def fit_boxes(instances, detections_by_photo, cams, labels, points=None, walls=None):
    """사물 목록에 `box3d`를 붙인다(제자리 수정). 반환: 맞춘 개수, 믿을 만한 개수.
    SETTINGS["floor_ray"]이면 바닥 광선 후보(floor_ray_candidates) 중 기존 사물과 떨어진 것을 목록에 더한다."""
    fitted = reliable = 0
    if SETTINGS["floor_ray"]:
        from .objects import KOREAN
        for cand in floor_ray_candidates(detections_by_photo, cams, labels):
            near = [i for i in instances if i["class"] == cand["class"]
                    and np.hypot(i["position"][0]-cand["position"][0], i["position"][2]-cand["position"][2]) < max(.4, .5*PRIORS[cand["class"]][0][0])]
            if not near:
                cand["label"] = KOREAN.get(cand["class"], cand["class"]); instances.append(cand)
    for inst in instances:
        views = associate(inst, detections_by_photo, cams, labels)
        box = fit_box(inst, views)
        for _ in range(int(SETTINGS["reassociate"])):
            if box is None: break
            moved = dict(inst, position=box["center"])
            again = fit_box(inst, associate(moved, detections_by_photo, cams, labels))
            if again is None or again["views"] < box["views"]*.8: break
            box = again
        if box is None:
            inst.pop("box3d", None); continue
        if box["views"] < SETTINGS["min_reliable_views"]:
            box["reliable"] = False; box["rejected"] = "few_views"
        if points is not None:
            box["inside_points"] = inside_points(box, points)
            need = SETTINGS["ray_min_inside_points"] if inst.get("source") == "floor_ray" else SETTINGS["min_inside_points"]
            if box["inside_points"] < need:  # 바닥 광선 후보는 SfM 점 근거가 없으므로 상자 안 점을 따로 요구
                box["reliable"] = False; box["rejected"] = "no_points_inside"
        if points is not None and inst["class"] in SOLID and PRIORS[inst["class"]][2]:
            frac, n = floor_visible(box, points)
            box["floor_visible"] = None if frac is None else round(frac, 3)
            if frac is not None and n >= MIN_FOOT_POINTS and frac > MAX_FLOOR_VISIBLE:
                box["reliable"] = False; box["rejected"] = "floor_visible"
        box["front_xz"] = facing(box, walls or [])
        inst["box3d"] = box
        fitted += 1
    # 같은 가구가 두 사물로 잡힌 경우(바닥면이 많이 겹침): 사진이 더 많은 쪽만 남긴다
    boxed = sorted([i for i in instances if i.get("box3d", {}).get("reliable")], key=lambda i: -i["box3d"]["views"])
    for k, a in enumerate(boxed):
        for b in boxed[:k]:
            same = b["class"] == a["class"] or (FAMILY.get(a["class"]) and FAMILY.get(a["class"]) == FAMILY.get(b["class"]))
            ca, cb = np.array(a["box3d"]["center"])[[0, 2]], np.array(b["box3d"]["center"])[[0, 2]]
            close = np.linalg.norm(ca-cb) < SETTINGS["dup_center_frac"]*max(a["box3d"]["size"][0], b["box3d"]["size"][0])
            if b["box3d"]["reliable"] and same and (close or footprint_overlap(a["box3d"], b["box3d"]) > SETTINGS["dup_overlap"]):
                a["box3d"]["reliable"] = False; a["box3d"]["rejected"] = "duplicate"; break
    reliable = sum(1 for i in instances if i.get("box3d", {}).get("reliable"))
    return fitted, reliable


def footprint_overlap(a, b):
    """작은 쪽 바닥면 대비 겹친 비율."""
    return footprint_iou(a, b, ratio="min")


def footprint_iou(a, b, step=.02, ratio="union"):
    """두 상자 바닥면(회전 포함)의 IoU(또는 작은 쪽 대비 겹침), 격자 근사."""
    import cv2
    polys = []
    for box in (a, b):
        P = corners((box["center"][0], box["center"][2]), (box["size"][0], box["size"][2], 1.), np.radians(box["yaw_deg"]))
        polys.append(P[P[:, 1] == 0][:, [0, 2]][[0, 1, 3, 2]])
    allp = np.vstack(polys); low = allp.min(0); size = np.ceil((allp.max(0)-low)/step).astype(int)+2
    masks = []
    for poly in polys:
        m = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(m, [np.round((poly-low)/step).astype(np.int32)], 1); masks.append(m > 0)
    inter = (masks[0] & masks[1]).sum()
    if ratio == "min": return float(inter/max(1, min(masks[0].sum(), masks[1].sum())))
    return float(inter/max(1, (masks[0] | masks[1]).sum()))


def orient(instances, walls):
    """도면 벽이 정해진 뒤 앞쪽 방향을 다시 정한다."""
    for inst in instances or []:
        if inst.get("box3d"): inst["box3d"]["front_xz"] = facing(inst["box3d"], walls)


def class_labels():
    from .objects import KOREAN
    labels = {v: k for k, v in KOREAN.items()}
    labels.update({k: k for k in KOREAN})
    return labels


def cameras_from_arrays(names, centers, rotations, intrinsics):
    return {n: (np.asarray(C, float), np.asarray(R, float), tuple(k)) for n, C, R, k in zip(names, centers, rotations, intrinsics)}


def cameras_from_reconstruction(path):
    """COLMAP 복원 폴더 → {사진 이름: (중심, R_cw, (폭, 높이, 초점))}."""
    import pycolmap
    from .sfm import cam_from_world
    rec = pycolmap.Reconstruction(str(path))
    cams = {}
    for iid in rec.reg_image_ids():
        img = rec.images[iid]
        M = cam_from_world(img).matrix(); R, t = M[:, :3], M[:, 3]
        cam = rec.cameras[img.camera_id]
        cams[img.name] = (-R.T@t, R, (cam.width, cam.height, float(cam.params[0])))
    return cams


def mark_display(instances, spaces):
    """도면·3D 모형에 그릴 가구(`box3d.display`). 상자가 믿을 만하고, 오검출이 잦은 주방 가전·세면대는
    해당 공간에 주방(세면대는 주방·욕실) 후보가 있을 때만. (TUM 사무실: 수납장→냉장고, 접시→싱크대)"""
    import cv2
    def usage_at(x, z):
        found = set()
        for s in spaces or []:
            if cv2.pointPolygonTest(np.asarray(s["polygon_xz"], np.float32), (float(x), float(z)), True) > -.3:
                found |= {u["usage"] for u in s.get("usage_candidates", [])}
        return found
    for inst in instances or []:
        box = inst.get("box3d")
        if not box: continue
        show = bool(box.get("reliable"))
        if show and inst["class"] in ("refrigerator", "oven", "microwave"):
            show = "kitchen" in usage_at(box["center"][0], box["center"][2])
        elif show and inst["class"] == "sink":
            show = bool({"kitchen", "bathroom"} & usage_at(box["center"][0], box["center"][2]))
        box["display"] = show
