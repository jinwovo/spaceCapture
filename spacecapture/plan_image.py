"""평면도 이미지(검은 벽·밝은 바닥 양식)에서 평가용 실·문 구조를 만든다.

LH 공공주택 평면도처럼 벽이 검게 채워진 도면을 대상으로 한다. 축척은 사람이 읽은
치수 사슬 합계와 이미지의 끝 눈금 위치로 정한다(문자 인식 없음). 결과는 평가용
정답 후보이며 사진 추론기에 전달하지 않는다. 자동 변환이므로 사람이 겹쳐 보고
확인한 뒤에만 정답으로 쓴다.
"""
from __future__ import annotations

import cv2
import numpy as np

from .procthor import photo_poses
from .synth import look_rotation

WALL_GRAY = 40
CLOSURE_MODE = "component"  # 2026-11-10 LH 98채 비교: component 75채 연결 vs directional 44채
GRID_M = .05


def dimension_ticks(gray, building_box, side="bottom", min_run_fraction=.25):
    """건물 바깥 치수선 띠에서 세로(또는 가로) 눈금선 위치를 찾는다."""
    x0, x1, y0, y1 = building_box
    if side in ("bottom", "top"):
        band = gray[y1+10:, :] if side == "bottom" else gray[:max(1, y0-10), :]
        dark = (band < 140).astype(np.uint8)
        runs = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(band.shape[0]*min_run_fraction*.5)))))
        profile = runs.sum(0)
    else:
        band = gray[:, x1+10:] if side == "right" else gray[:, :max(1, x0-10)]
        dark = (band < 140).astype(np.uint8)
        runs = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(band.shape[1]*min_run_fraction*.5)), 1)))
        profile = runs.sum(1)
    hits = np.flatnonzero(profile > 0)
    if len(hits) < 2:
        raise ValueError("치수선 눈금을 찾지 못했어요.")
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > 3)+1)
    return [float(np.mean(g)) for g in groups]


def wall_mask(image, threshold=WALL_GRAY):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = (gray < threshold).astype(np.uint8)
    return gray, mask


def meters_per_pixel(image, total_m, side="bottom"):
    gray, mask = wall_mask(image)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    ticks = dimension_ticks(gray, box, side)
    span = ticks[-1]-ticks[0]
    building = (box[1]-box[0]) if side in ("bottom", "top") else (box[3]-box[2])
    if not .7*building < span < 1.1*building:
        raise ValueError("치수선 끝 눈금 간격이 건물 크기와 맞지 않아요. 다른 치수 사슬을 지정해 주세요.")
    return total_m/span, {"side": side, "ticks_px": [ticks[0], ticks[-1]], "span_px": span, "building_px": building}


def _rectangles(mask):
    """격자 마스크를 직사각형 묶음으로 나눈다 (행 구간을 세로로 병합)."""
    rects, open_runs = [], {}
    for row in range(mask.shape[0]+1):
        runs = set()
        if row < mask.shape[0]:
            cols = np.flatnonzero(mask[row])
            if len(cols):
                for part in np.split(cols, np.flatnonzero(np.diff(cols) > 1)+1):
                    runs.add((int(part[0]), int(part[-1])+1))
        for run in list(open_runs):
            if run not in runs:
                rects.append((run[0], open_runs.pop(run), run[1], row))
        for run in runs:
            open_runs.setdefault(run, row)
    return rects  # (c0, r0, c1, r1) half-open


def segment_plan(image, mpp, door_max_m=1.05, min_room_m2=.8, exterior_band_m=.28, closure_mode=None):
    """실 영역과 문 틈 후보를 미터 좌표로 돌려준다."""
    gray, raw = wall_mask(image)
    px = lambda meters: max(1, int(round(meters/mpp)))
    wall = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((px(.06), px(.06)), np.uint8))
    # Outline from walls at least ~10cm thick: dimension lines, ticks and text are
    # thinner and otherwise pull the outline (and fake rooms) outside the building.
    thick = cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((px(.1), px(.1)), np.uint8))
    sealed = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (px(3.2), 1)))
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, px(3.2))))
    contours, _ = cv2.findContours(sealed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.zeros_like(wall)
    cv2.drawContours(outline, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    # Windows sit inside the exterior wall band; keep only the inside of that band.
    # The band is the measured exterior wall thickness (fixed 0.28m shrank rooms
    # next to thinner walls); exterior_band_m is only the fallback.
    edge = cv2.morphologyEx(outline, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    near_edge = cv2.dilate(edge.astype(np.uint8), np.ones((px(.6),)*2, np.uint8)) > 0
    thickness = cv2.distanceTransform(wall, cv2.DIST_L2, 5)
    samples = thickness[near_edge & (wall > 0)]
    band_m = float(np.clip(2*np.percentile(samples, 90)*mpp+.02, .12, .45)) if len(samples) else exterior_band_m
    inside = cv2.erode(outline, np.ones((px(band_m)*2+1,)*2, np.uint8))
    # Door gaps: bridge wall ends along each axis, then keep only the part that is
    # as thin as a wall across the bridging direction. A corridor about as wide as
    # a door (1.0-1.1m) is also bridged by the closing, but that bridge is long in
    # the other direction and must stay floor, or the corridor and every door off
    # it disappear (LH 180, 2026-11-10). Checked per direction, not per component,
    # because a door bridge can touch a corridor bridge (LH 45).
    thin = px(.45)
    h = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (px(door_max_m), 1))) & (1-wall)
    h &= 1-cv2.morphologyEx(h, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, thin)))
    v = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, px(door_max_m)))) & (1-wall)
    v &= 1-cv2.morphologyEx(v, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (thin, 1)))
    # A closure thinner than ~7cm is not a door leaf position: it bridges two
    # wall ends across an open passage (e.g. a 1.0m hall opening). Drop it so it
    # neither separates rooms nor becomes a portal (plan 227, 2026-11-10).
    mode = closure_mode or CLOSURE_MODE
    if mode in ("component", "union"):
        both = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (px(door_max_m), 1)))
        both = cv2.morphologyEx(both, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, px(door_max_m))))
        closure = ((both > 0) & (wall == 0)).astype(np.uint8)
        closure = cv2.morphologyEx(closure, cv2.MORPH_OPEN, np.ones((px(.07), px(.07)), np.uint8))
        count, comp, stats, _ = cv2.connectedComponentsWithStats(closure)
        keep = np.zeros(count, bool)
        for c in range(1, count):
            keep[c] = min(stats[c, cv2.CC_STAT_WIDTH], stats[c, cv2.CC_STAT_HEIGHT])*mpp <= .45
        closure = keep[comp].astype(np.uint8)
        if mode == "union":  # 덩어리 판정에서 복도와 함께 지워진 문 틈을 방향별 판정으로 되살린다
            directional = cv2.morphologyEx(((h | v) > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((px(.07), px(.07)), np.uint8))
            closure |= directional
    else:
        closure = ((h | v) > 0).astype(np.uint8)
        closure = cv2.morphologyEx(closure, cv2.MORPH_OPEN, np.ones((px(.07), px(.07)), np.uint8))
    door_closed = ((wall > 0) | (closure > 0)).astype(np.uint8)
    free = ((inside > 0) & (door_closed == 0)).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((px(.1), px(.1)), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free)
    rooms = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA]*mpp*mpp >= min_room_m2]
    h, w = wall.shape
    gw, gh = int(np.ceil(w*mpp/GRID_M)), int(np.ceil(h*mpp/GRID_M))
    spaces = []
    for k, label in enumerate(rooms, 1):
        grid = cv2.resize((labels == label).astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > .5
        grid = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        found, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            continue
        contour = max(found, key=cv2.contourArea).reshape(-1, 2)
        mask = np.zeros_like(grid); cv2.drawContours(mask, [contour.reshape(-1, 1, 2)], -1, 1, -1)
        polygon = (contour*GRID_M).astype(float)
        spaces.append({"id": f"lh-{k}", "kind": "room", "label": label, "grid": mask,
                       "polygon_xz": polygon.round(4).tolist(), "area_m2": round(float(mask.sum()*GRID_M**2), 3)})
    # Door gaps: cells closed only by the door-width closing, touching two rooms.
    # Anti-aliased wall borders form 1-2px "gaps" along every wall; the opening
    # above already removed them so door gaps are not chained together.
    gap = ((closure > 0) & (inside > 0)).astype(np.uint8)
    gap_count, gap_labels, gap_stats, _ = cv2.connectedComponentsWithStats(gap)
    room_of = {s["label"]: s["id"] for s in spaces}
    portals = []
    for g in range(1, gap_count):
        x, y, bw, bh, area = gap_stats[g]
        component = gap_labels == g
        ring = cv2.dilate(component.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        touching = [room_of[l] for l in np.unique(labels[ring & (free > 0)]) if l in room_of]
        if len(touching) != 2:
            continue
        long_px, short_px = max(bw, bh), min(bw, bh)
        width = long_px*mpp
        if not .55 <= width <= door_max_m+.05 or short_px*mpp > .45:
            continue
        x0, x1, z0, z1 = x*mpp, (x+bw)*mpp, y*mpp, (y+bh)*mpp
        if bw >= bh:
            ends = [[x0, (z0+z1)/2], [x1, (z0+z1)/2]]
        else:
            ends = [[(x0+x1)/2, z0], [(x0+x1)/2, z1]]
        portals.append({"id": f"lh-door-{len(portals)+1}", "spaces": touching,
                        "center_xz": np.mean(ends, axis=0).round(4).tolist(), "endpoints_xz": np.round(ends, 4).tolist(),
                        "width_m": round(width, 3), "thickness_m": round(short_px*mpp, 3),
                        "box_xz": [x0, z0, x1, z1], "kind": "doorway", "material": "open"})
    return {"spaces": spaces, "portals": portals, "mpp": mpp, "exterior_band_m": round(band_m, 3),
            "image_size": [w, h], "wall_pixels": int(wall.sum())}


def render_config(segmentation, height=2.3, seed=0, min_photo_area_m2=2.5, room_views=12,
                  walk_offsets=(-1.4, -.9, -.5, 0., .5, .9, 1.4)):
    """영역·문을 자체 렌더러 설정으로 바꾼다. 벽 두께를 유지하고 문 틈에 측면·인방을 만든다."""
    spaces = segmentation["spaces"]
    photographed = {s["id"] for s in spaces if s["area_m2"] >= min_photo_area_m2}
    surfaces, horizontal = [], []
    portals = segmentation["portals"]
    def add(axis, pos, lo, hi, bottom, top):
        if hi-lo > 1e-4 and top-bottom > 1e-4:
            surfaces.append({"axis": axis, "pos": float(pos), "lo": float(lo), "hi": float(hi),
                             "bottom": float(bottom), "top": float(top)})
    door_top = 2.1
    for space in spaces:
        polygon = np.array(space["polygon_xz"])
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            vertical = abs(a[0]-b[0]) < 1e-6  # constant x
            axis = 0 if vertical else 2
            pos = a[0] if vertical else a[1]
            lo, hi = sorted((a[1], b[1]) if vertical else (a[0], b[0]))
            cuts = []
            for portal in portals:
                if space["id"] not in portal["spaces"]:
                    continue
                bx0, bz0, bx1, bz1 = portal["box_xz"]
                along = (bz0, bz1) if vertical else (bx0, bx1)
                across = (bx0, bx1) if vertical else (bz0, bz1)
                if across[0]-.12 <= pos <= across[1]+.12 and along[1] > lo and along[0] < hi:
                    cuts.append((max(lo, along[0]), min(hi, along[1])))
            cursor = lo
            for c0, c1 in sorted(cuts):
                add(axis, pos, cursor, c0, 0., height)
                add(axis, pos, c0, c1, door_top, height)
                cursor = max(cursor, c1)
            add(axis, pos, cursor, hi, 0., height)
        for c0, r0, c1, r1 in _rectangles(space["grid"] > 0):
            for y in (0., height):
                horizontal.append({"axis": 1, "pos": y, "lo": c0*GRID_M, "hi": c1*GRID_M,
                                   "bottom": r0*GRID_M, "top": r1*GRID_M})
    for portal in portals:
        x0, z0, x1, z1 = portal["box_xz"]
        if x1-x0 >= z1-z0:  # door along x, wall thickness along z
            for x in (x0, x1):
                add(0, x, z0, z1, 0., height)
        else:
            for z in (z0, z1):
                add(2, z, x0, x1, 0., height)
        horizontal.append({"axis": 1, "pos": 0., "lo": x0, "hi": x1, "bottom": z0, "top": z1})
        horizontal.append({"axis": 1, "pos": door_top, "lo": x0, "hi": x1, "bottom": z0, "top": z1})
    truth_spaces = [{"id": s["id"], "kind": "room", "polygon_xz": s["polygon_xz"], "area_m2": s["area_m2"]}
                    for s in spaces if s["id"] in photographed]
    truth_portals = [{k: p[k] for k in ("id", "spaces", "center_xz", "endpoints_xz", "width_m", "kind", "material")}
                     | {"height_m": door_top} for p in portals if set(p["spaces"]) <= photographed]
    polygons = {s["id"]: np.array(s["polygon_xz"], np.float32) for s in truth_spaces}
    poses = photo_poses(truth_spaces, polygons, [], room_views=room_views)
    # Walk-through views: a photographer crossing each doorway with overlapping
    # shots on both sides and inside the jamb (thick walls leave few landmarks
    # visible from both rooms otherwise).
    for portal in truth_portals:
        center = np.array(portal["center_xz"])
        tangent = np.diff(portal["endpoints_xz"], axis=0)[0]; tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        for offset in walk_offsets:
            xz = center + offset*normal
            owners = [sid for sid in portal["spaces"] if cv2.pointPolygonTest(polygons[sid], tuple(map(float, xz)), False) > 0]
            owner = owners[0] if owners else min(portal["spaces"], key=lambda sid: abs(cv2.pointPolygonTest(polygons[sid], tuple(map(float, xz)), True)))
            for direction in ((-1., 1.) if abs(offset) < .3 else (-np.sign(offset),)):
                target = center + direction*1.8*normal + (.25 if offset > 0 else -.25)*tangent
                camera = np.array([xz[0], 1.45, xz[1]])
                poses.append((camera, look_rotation(np.array([target[0], 1.2, target[1]])-camera), owner))
    order = np.random.default_rng(seed).permutation(len(poses))
    allp = np.vstack([np.array(s["polygon_xz"]) for s in spaces])
    size = [float(allp[:, 0].max()), float(height), float(allp[:, 1].max())]
    config = {"size": size, "surfaces": surfaces, "horizontal_surfaces": horizontal,
              "spaces": truth_spaces, "portals": truth_portals, "poses": [poses[i] for i in order],
              "texture_style": "unique_noise"}
    excluded = [{"id": s["id"], "area_m2": s["area_m2"]} for s in spaces if s["id"] not in photographed]
    return config, excluded


def auto_meters_per_pixel(image, door_m=.85, iterations=4):
    """치수 판독 없이 문 폭 중앙값(국내 아파트 실내문 약 0.85m)으로 축척을 추정한다.

    학습 자료 생성용 근사값이다. 평가 정답에는 사람이 읽은 치수 사슬 축척을 쓴다.
    """
    _, mask = wall_mask(image)
    ys, xs = np.nonzero(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)))
    if not len(xs):
        raise ValueError("벽을 찾지 못했어요.")
    last_error = None
    for building_m in (11., 9., 13., 15., 8.):  # 첫 추정이 틀리면 문 두께 조건이 어긋나므로 여러 값으로 시작
        mpp = building_m/max(1, xs.max()-xs.min())
        try:
            for _ in range(iterations):
                seg = segment_plan(image, mpp)
                widths = [p["width_m"]/mpp for p in seg["portals"]]
                if len(widths) < 2:
                    raise ValueError("문이 2개 미만이라 축척을 추정할 수 없어요.")
                updated = door_m/float(np.median(widths))
                done = abs(updated/mpp-1) < .01
                mpp = updated
                if done: break
            return mpp, segment_plan(image, mpp)
        except ValueError as error:
            last_error = error
    raise last_error

