"""Extract observed vertical plane segments and check the rectangular assumption.

This is geometric evidence, not semantic glass detection or a complete floorplan.
No closed polygon is invented when the observed planes contradict the room box.
"""
from __future__ import annotations

import numpy as np


def analyze_layout(points: np.ndarray, cameras: np.ndarray, room: dict) -> dict:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    cameras = np.asarray(cameras, dtype=float).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    top = float(cameras[:, 1].max()) if len(cameras) else 1.6
    band = points[finite & (points[:, 1] > .25) & (points[:, 1] < min(room["height"] - .1, top + .7))]
    rng = np.random.default_rng(0)
    remaining = band.copy()
    planes = []
    minimum = max(30, int(len(band) * .025))
    for _ in range(12):
        if len(remaining) < minimum:
            break
        sample = remaining if len(remaining) <= 5000 else remaining[rng.choice(len(remaining), 5000, replace=False)]
        xz = sample[:, [0, 2]]
        best = None
        for _ in range(250):
            a, b = xz[rng.choice(len(xz), 2, replace=False)]
            direction = b - a
            length = np.linalg.norm(direction)
            if length < .4:
                continue
            normal = np.array([-direction[1], direction[0]]) / length
            offset = -float(normal @ a)
            mask = np.abs(xz @ normal + offset) < .04
            count = int(mask.sum())
            if best is None or count > best[0]:
                best = (count, normal, offset)
        if best is None or best[0] < minimum * len(sample) / max(len(remaining), 1):
            break
        _, normal, offset = best
        mask = np.abs(remaining[:, [0, 2]] @ normal + offset) < .04
        inliers = remaining[mask]
        remaining = remaining[~mask]
        # Horizontal furniture tops and tiny noisy patches are not wall planes.
        height_span = float(np.ptp(np.percentile(inliers[:, 1], [5, 95])))
        if height_span < .75:
            continue
        center = inliers[:, [0, 2]].mean(axis=0)
        _, _, vt = np.linalg.svd(inliers[:, [0, 2]] - center, full_matrices=False)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        projection = (inliers[:, [0, 2]] - center) @ direction
        lo, hi = np.percentile(projection, [2, 98])
        length = float(hi - lo)
        if length < .7:
            continue
        axis = int(np.argmax(np.abs(normal)))
        axis_aligned = float(np.min(np.abs(normal))) < np.sin(np.radians(12))
        planes.append({"start": (center + lo * direction).round(4).tolist(),
                       "end": (center + hi * direction).round(4).tolist(),
                       "normal_xz": normal.round(5).tolist(), "axis": "x" if axis == 0 else "z",
                       "position": round(float(center[axis]), 4), "axis_aligned": bool(axis_aligned),
                       "length": round(length, 4), "vertical_extent": round(height_span, 4),
                       "support": len(inliers), "fraction": round(len(inliers) / max(len(band), 1), 4)})
    # Check each proposed outer wall directly as well: a weakly textured wall
    # must not disappear merely because RANSAC consumed a few of its points.
    for key, wall in room["walls"].items():
        axis = 0 if key.startswith("x") else 2
        along = 2 if axis == 0 else 0
        near = band[np.abs(band[:, axis] - wall["pos"]) < .04]
        if len(near) < 30:
            continue
        vertical = float(np.ptp(np.percentile(near[:, 1], [5, 95])))
        lo, hi = np.percentile(near[:, along], [2, 98])
        if vertical < .75 or hi - lo < .7:
            continue
        if any(p["axis_aligned"] and p["axis"] == key[0] and abs(p["position"] - wall["pos"]) < .12 for p in planes):
            continue
        start = [wall["pos"], lo] if axis == 0 else [lo, wall["pos"]]
        end = [wall["pos"], hi] if axis == 0 else [hi, wall["pos"]]
        planes.append({"start": np.round(start, 4).tolist(), "end": np.round(end, 4).tolist(),
                       "normal_xz": [1.0, 0.0] if axis == 0 else [0.0, 1.0], "axis": key[0],
                       "position": wall["pos"], "axis_aligned": True, "length": round(float(hi - lo), 4),
                       "vertical_extent": round(vertical, 4), "support": len(near),
                       "fraction": round(len(near) / max(len(band), 1), 4), "method": "box_boundary_candidate"})
    boundaries = {}
    for key, wall in room["walls"].items():
        span = room["depth"] if key.startswith("x") else room["width"]
        candidates = [p for p in planes if p["axis_aligned"] and p["axis"] == key[0]
                      and abs(p["position"] - wall["pos"]) < .12]
        coverage = max((min(1.0, p["length"] / span) for p in candidates), default=0.0)
        boundaries[key] = {"coverage": round(coverage, 4), "supported": coverage >= .4 and not wall.get("fallback", False)}
    complex_planes = []
    for i, plane in enumerate(planes):
        span = room["depth"] if plane["axis"] == "x" else room["width"]
        if plane["length"] < max(1.6, .4 * span) or plane["fraction"] < .06:
            continue
        near_boundary = any(plane["axis"] == k[0] and abs(plane["position"] - w["pos"]) < .2
                            for k, w in room["walls"].items())
        if not plane["axis_aligned"] or not near_boundary:
            complex_planes.append(i)
    unsupported = [key for key, value in boundaries.items() if not value["supported"]]
    supported = not unsupported and not complex_planes
    status = "rectangular_candidate" if supported else "complex_geometry" if complex_planes else "incomplete_boundaries"
    messages = {
        "rectangular_candidate": "관측된 수직면이 직사각형 경계 후보와 맞아요. 유리벽·숨은 칸막이와 공간 용도는 별도 확인이 필요해요.",
        "complex_geometry": "내부 또는 비스듬한 수직면 후보가 있어 직사각형 도면을 보류했어요. 관측한 벽 후보만 표시해요.",
        "incomplete_boundaries": "닫힌 방을 만들 만큼 벽 경계의 근거가 부족해요. 관측한 벽 후보만 표시해요.",
    }
    return {"status": status, "room_box_supported": supported, "planes": planes,
            "boundary_evidence": boundaries, "complex_plane_indices": complex_planes,
            "unsupported_boundaries": unsupported, "band_points": len(band),
            "message": messages[status], "glass_detection": "not_checked",
            "scope": "observed_vertical_plane_candidates"}
