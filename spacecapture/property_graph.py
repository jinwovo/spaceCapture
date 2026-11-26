"""Observed-space and opening candidates from actual SfM observation rays.

Image connectivity is not room identity. Geometry here describes observed
regions, not semantic bedrooms or confirmed traversable doors. In particular,
optical visibility through glass must never be called a confirmed doorway.
"""
from __future__ import annotations

import heapq
from collections import Counter

import cv2
import numpy as np


def infer_property(points, cameras, names, observations, cell_size=.1, neck_width=1.1, planes=None,
                   rotations=None, intrinsics=None, debug=None, small_room_model=None, merge_model=None,
                   regularize=True):
    P, C = np.asarray(points, float).reshape(-1, 3), np.asarray(cameras, float).reshape(-1, 3)
    empty = {"version": 1, "status": "insufficient_observations", "spaces": [], "portals": [],
             "space_candidate_count": 0, "confirmed_room_count": None, "confirmed_door_count": None,
             "message": "여러 공간을 나누거나 문 연결을 판단할 관측 근거가 부족해요."}
    if len(P) < 30 or len(C) < 3 or len(observations) != len(P):
        return empty
    valid = np.isfinite(P).all(1) & (P[:, 1] > -.08) & (P[:, 1] < np.median(C[:, 1]) + .6)
    P, observations = P[valid], [o for o, keep in zip(observations, valid) if keep]
    if len(P) < 30:
        return empty
    lower = np.minimum(np.percentile(P[:, [0, 2]], 1, axis=0), C[:, [0, 2]].min(0)) - .3
    upper = np.maximum(np.percentile(P[:, [0, 2]], 99, axis=0), C[:, [0, 2]].max(0)) + .3
    step = max(float(cell_size), float(np.max(upper - lower)) / 480)
    dims = np.ceil((upper - lower) / step).astype(int) + 1
    if min(dims) < 8:
        return empty
    width, height = map(int, dims)
    free = np.zeros((height, width), np.uint8)
    to_pixel = lambda a: np.floor((np.asarray(a)[..., [0, 2]] - lower) / step).astype(int)
    pp, cc = to_pixel(P), to_pixel(C)
    for endpoint, observers in zip(pp, observations):
        if not (0 <= endpoint[0] < width and 0 <= endpoint[1] < height):
            continue
        for observer in observers[:12]:
            if 0 <= observer < len(C):
                cv2.line(free, tuple(cc[observer]), tuple(endpoint), 255, 1)
    for camera in cc:
        cv2.circle(free, tuple(camera), max(1, round(.15 / step)), 255, -1)
    # A floor point is not a vertical obstacle. Use height diversity at each
    # endpoint, below common lintels, so an observed door gap stays open.
    count = np.zeros((height, width), np.int32)
    low = np.full((height, width), np.inf)
    high = np.full((height, width), -np.inf)
    wall = (P[:, 1] > .18) & (P[:, 1] < min(1.95, np.median(C[:, 1]) + .35))
    wall &= (pp[:, 0] >= 0) & (pp[:, 0] < width) & (pp[:, 1] >= 0) & (pp[:, 1] < height)
    wx, wy, py = pp[wall, 0], pp[wall, 1], P[wall, 1]
    np.add.at(count, (wy, wx), 1)
    np.minimum.at(low, (wy, wx), py)
    np.maximum.at(high, (wy, wx), py)
    obstacles = ((count >= 3) & (high - low > .65)).astype(np.uint8)
    # Sparse features leave holes in a real wall. Project verified vertical
    # plane inliers onto short supported line runs, preserving larger gaps.
    # This fills sampling holes, not an entire fitted plane across a doorway.
    if planes:
        for plane in planes:
            start,end=np.asarray(plane["start"]),np.asarray(plane["end"])
            tangent=end-start
            length=float(np.linalg.norm(tangent))
            if length<.7: continue
            tangent/=length
            normal=np.array([-tangent[1],tangent[0]])
            near=wall & (np.abs((P[:,[0,2]]-start)@normal)<.05)
            points_on_plane=P[near]
            if len(points_on_plane)<25 or np.ptp(np.percentile(points_on_plane[:,1],[5,95]))<.7: continue
            along=(points_on_plane[:,[0,2]]-start)@tangent
            bins=max(2,int(np.ceil(length/step)))
            counts,_=np.histogram(along,bins=bins,range=(0,bins*step))
            supported=(counts>=2).astype(np.uint8)[None,:]
            supported=cv2.morphologyEx(supported,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8))[0]
            indices=np.flatnonzero(supported)
            for run in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
                if len(run)<2: continue
                a=start+(float(run[0])+.1)*step*tangent
                b=start+(float(run[-1])+.9)*step*tangent
                pa=tuple(np.floor((a-lower)/step).astype(int)); pb=tuple(np.floor((b-lower)/step).astype(int))
                cv2.line(obstacles,pa,pb,1,1)
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    free[obstacles > 0] = 0
    diameter = max(3, int(round(neck_width / step)) | 1)
    # Label the eroded cores before growing them. Opening (= erode+dilate)
    # would reconnect two rooms across a thin doorway during dilation.
    opened = cv2.erode(free, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
    n, components, stats, _ = cv2.connectedComponentsWithStats(opened)
    seeds = []
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] * step**2 < .6:
            continue
        ys, xs = np.nonzero(components == label)
        near_cameras = []
        for i, camera in enumerate(cc):
            if np.min((xs - camera[0])**2 + (ys - camera[1])**2) * step**2 < 1.0:
                near_cameras.append(i)
        if near_cameras:
            seeds.append((label, near_cameras))
    if not seeds:
        return empty | {"observed_ray_count": sum(len(o[:12]) for o in observations)}
    # Wider openings remain connected after a single 1.1m erosion. Split cores
    # at several widths while preserving smaller existing regions (corridors).
    # A split requires two sizeable child cores with camera support. Ground-truth
    # room counts and doorway positions never participate in this operation.
    cores=[(components==label).astype(np.uint8) for label,_ in seeds]
    widths=[float(neck_width)]
    split_events=0
    small_cores=0
    for core_width in (1.5,1.9,2.3):
        if core_width<=neck_width: continue
        widths.append(core_width)
        diameter=max(3,int(round(core_width/step))|1)
        eroded=cv2.erode(free,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter,diameter)))
        count,parts,part_stats,_=cv2.connectedComponentsWithStats(eroded)
        children=[(parts==label).astype(np.uint8) for label in range(1,count)
                  if part_stats[label,cv2.CC_STAT_AREA]*step**2>=.8]
        refined=[]
        for core in cores:
            inside=[]
            for child in children:
                if np.sum(child & core)<.95*np.sum(child): continue
                proximity=cv2.distanceTransform(1-child,cv2.DIST_L2,5)*step
                supporting=sum(proximity[np.clip(y,0,height-1),np.clip(x,0,width-1)]<.8 for x,y in cc)
                if supporting>=2: inside.append(child)
            if len(inside)>=2:
                refined.extend(inside); split_events+=1
            else:
                refined.append(core)
        cores=refined
    # Small rooms (bathrooms, utility rooms ~1.5-2m wide) vanish under the
    # 1.1m neck erosion. Add a narrower 0.75m core when it does not overlap an
    # existing core and has cameras inside it (real LH plans, 2026-11-10).
    # Small rooms (bathrooms, utility rooms ~1.5-2m wide) vanish under the
    # 1.1m neck erosion. Candidates at several narrower widths are judged by
    # the learned space-candidate model, or by the original rule (0.75m
    # standalone core with two cameras inside) when no model is installed.
    from .space_model import decide, small_room_candidates
    candidates=small_room_candidates(free,cores,cc,step)
    accepted,decision=decide(candidates,small_room_model)
    for candidate,keep in zip(candidates,accepted):
        if keep:
            cores.append(candidate["mask"]); small_cores+=1
    if debug is not None:
        debug.update(initial_core_count=len(cores)-small_cores,small_candidates=candidates,small_decision=decision)
    # Grow observed seeds without crossing occupied or unseen cells, in order of
    # decreasing clearance (distance-transform watershed). Regions then meet at
    # the narrowest neck, i.e. the door jambs. Plain BFS met wherever the two
    # wavefronts collided, which pushed openings up to ~40cm into one room and
    # measured a wider cross-section (procthor-val-0043 door|4|7, 2026-11-10).
    distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    labels = np.zeros_like(components, dtype=np.int32)
    queue = []
    order = 0
    def push_neighbors(y, x):
        nonlocal order
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and free[ny, nx] and not labels[ny, nx]:
                order += 1
                heapq.heappush(queue, (-float(distance[ny, nx]), order, ny, nx, int(labels[y, x])))
    for i, core in enumerate(cores, 1):
        ys, xs = np.nonzero(core)
        labels[ys, xs] = i
    for y, x in zip(*np.nonzero(labels)):
        push_neighbors(int(y), int(x))
    while queue:
        _, _, y, x, label = heapq.heappop(queue)
        if labels[y, x]:
            continue
        labels[y, x] = label
        push_neighbors(y, x)
    def build(labels, count):
        camera_labels = []
        for x, y in cc:
            camera_labels.append(int(labels[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]))
        spaces = []
        for i in range(1, count + 1):
            mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            polygon = cv2.approxPolyDP(contour, max(1., .08 / step), True).reshape(-1, 2)
            polygon = lower + (polygon + .5) * step
            ys, xs = np.nonzero(mask)
            photos = [names[j] for j, label in enumerate(camera_labels) if label == i]
            spaces.append({"id": f"space-{i}", "label": f"공간 후보 {i}", "kind": "unknown",
                           "polygon_xz": polygon.round(4).tolist(),
                           "center_xz": (lower + np.array([xs.mean(), ys.mean()]) * step).round(4).tolist(),
                           "observed_area_m2": round(float(mask.sum() * step**2), 3),
                           "photos": photos, "confirmed": False})
        # Borders between grown regions locate narrow optical openings. Room IDs
        # come from geometry, not filename order, photo count, or image graph clusters.
        pairs = set()
        for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1], labels[1:])):
            m = (a > 0) & (b > 0) & (a != b)
            pairs.update(tuple(sorted(p)) for p in zip(a[m].tolist(), b[m].tolist()))
        if debug is not None:
            debug.update(labels=labels.copy(),free=free.copy(),obstacles=obstacles.copy(),lower=lower,step=step,cameras=cc)
        portals = []
        for a, b in sorted(pairs):
            border = np.zeros_like(free)
            for dy, dx in ((0, 1), (1, 0)):
                yend, xend = height - dy, width - dx
                first, second = labels[:yend, :xend], labels[dy:, dx:]
                m = ((first == a) & (second == b)) | ((first == b) & (second == a))
                border[:yend, :xend][m] = 255
            n_regions, border_components = cv2.connectedComponents(border)
            for component in range(1, n_regions):
                yy, xx = np.nonzero(border_components == component)
                if len(xx) < 2:
                    continue
                center = lower + (np.array([xx.mean(), yy.mean()]) + .5) * step
                aperture = float(2 * np.percentile(distance[yy, xx], 90) * step)
                # Reported width is the jamb-to-jamb length of the border along its
                # main axis. The clearance (distance transform) underestimates wide
                # openings because their centre is also near other walls/corners; it
                # is kept for candidate filtering and ray support as before.
                cells=np.column_stack([xx,yy]).astype(float)
                axis=np.linalg.svd(cells-cells.mean(0),full_matrices=False)[2][0]
                opening_width=float((np.ptp((cells-cells.mean(0))@axis)+1)*step)
                if not .35 <= aperture <= 3.0:
                    continue
                cross_points, photo_counts = 0, Counter()
                eligible=set(range(len(C)))
                if rotations is not None and intrinsics is not None:
                    candidate=np.array([center[0],min(1.1,float(np.median(C[:,1]))),center[1]])
                    eligible=set()
                    for j,(rotation,calibration) in enumerate(zip(rotations,intrinsics)):
                        q=np.asarray(rotation)@(candidate-C[j])
                        iw,ih,f=calibration
                        if q[2]>.08 and abs(q[0]/q[2])<iw/(2*f)*.95 and abs(q[1]/q[2])<ih/(2*f)*.95:
                            eligible.add(j)
                for point, observers in zip(P, observations):
                    left = [j for j in observers if camera_labels[j] == a]
                    right = [j for j in observers if camera_labels[j] == b]
                    if not left or not right:
                        continue
                    # Both regions must see the same triangulated landmark, and a
                    # camera that has this opening in view must see it through a ray
                    # passing near the opening (not merely share a style). Only the
                    # looking-through camera needs the opening in view: a walk-through
                    # video crosses a door facing forward and never looks back, so
                    # requiring both sides to face the door gave 0 support (2026-11-26).
                    supporting = []
                    for j in left + right:
                        if j not in eligible:
                            continue
                        v = point[[0, 2]] - C[j, [0, 2]]
                        t = np.clip(float((center - C[j, [0, 2]]) @ v) / max(float(v @ v), 1e-8), 0, 1)
                        if np.linalg.norm(C[j, [0, 2]] + t * v - center) <= max(.25, aperture * .6):
                            supporting.append(j)
                    if supporting:
                        cross_points += 1
                        photo_counts.update(names[j] for j in set(left + right))
                by_region=[]
                photo_region={name:label for name,label in zip(names,camera_labels)}
                for region in (a,b):
                    by_region.extend([name for name,_ in photo_counts.most_common() if photo_region[name]==region][:6])
                portals.append({"id": f"opening-{len(portals)+1}", "spaces": [f"space-{a}", f"space-{b}"],
                                "center_xz": center.round(4).tolist(), "width_m": round(opening_width, 3),"clearance_width_m": round(aperture, 3),
                                "width_method": "border_extent",
                                "kind": "door_or_opening_candidate", "confirmed": False,
                                "shared_landmarks": cross_points,
                                "evidence_photos": by_region,
                                "status": "image_supported" if cross_points >= 8 and len({photo_region[n] for n in by_region})==2 else "geometry_only",
                                "view_projection_checked":rotations is not None and intrinsics is not None,
                                "glass_excluded": False})
        return spaces, portals, camera_labels

    spaces, portals, camera_labels = build(labels, len(cores))
    # Learned merge of openings that are really open passages inside one space
    # (e.g. a corridor opening into the living room). Without a model, nothing merges.
    from .space_model import decide_merges, portal_features
    merge_features = portal_features(portals, spaces, labels, distance, step)
    merges, merge_info = decide_merges(merge_features, merge_model)
    if any(merges):
        parent = list(range(len(spaces)+1))
        def root(i):
            while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for portal, merge in zip(portals, merges):
            if merge:
                a, b = (int(x.split("-")[1]) for x in portal["spaces"])
                parent[root(b)] = root(a)
        roots = sorted({root(i) for i in range(1, len(spaces)+1)})
        renumber = {r: k for k, r in enumerate(roots, 1)}
        lut = np.zeros(len(spaces)+1, np.int32)
        for i in range(1, len(spaces)+1): lut[i] = renumber[root(i)]
        labels = lut[labels]
        spaces, portals, camera_labels = build(labels, len(roots))
    if debug is not None:
        debug.update(portal_merge_features=merge_features, portal_merge=merges, portal_merge_info=merge_info)
    # Observed floor stops short of walls (occlusion, furniture). Square each
    # region and move its edges to the nearest outward wall-point line; the raw
    # observed outline stays in observed_polygon_xz. Openings use the grid labels.
    outline = {"method": "observed"}
    if regularize:
        from .space_regularize import regularize_spaces
        outline = regularize_spaces(spaces, labels, lower, step, points)
    return {"version": 1, "status": "candidate", "spaces": spaces, "portals": portals,
            "space_candidate_count": len(spaces), "confirmed_room_count": None, "confirmed_door_count": None,
            "image_supported_portal_count": sum(p["status"]=="image_supported" for p in portals),
            "geometry_only_portal_count": sum(p["status"]=="geometry_only" for p in portals),
            "segmentation":{"core_widths_m":widths,"split_events":split_events,"small_room_cores":small_cores,
                            "small_room_method":decision.get("method"),"merged_openings":int(sum(merges)),
                            "merge_method":merge_info.get("method"),"outline_method":outline["method"],
                            "scope":"여러 폭의 관측 영역 분할 후보이며 실제 방 개수는 미확정"},
            "grid": {"origin_xz": lower.round(4).tolist(), "cell_m": round(step, 4), "size": [width, height]},
            "photo_spaces": {name: f"space-{label}" if label else None for name, label in zip(names, camera_labels)},
            "message": "사진의 관측 영역과 공유 3D 특징으로 만든 공간·개구부 후보예요. 방 용도·통행 가능 여부·유리벽은 미확정이에요."}
