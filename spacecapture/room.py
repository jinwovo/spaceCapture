"""Room box (walls, ceiling) from points in the aligned metric frame.

Assumes a rectangular room whose walls were made axis-aligned by `align.py`.
A wall is the innermost strong histogram peak beyond every camera position that
has (almost) nothing behind it - furniture faces have the real wall behind them.
The ceiling is the highest horizontal layer that spreads over a real area, so a
wardrobe top is not mistaken for it.
"""

from __future__ import annotations

import numpy as np

from .geometry import histogram_peaks

DEFAULT_CEILING = 2.4  # m, typical Korean apartment


def boundary(v: np.ndarray, cam_extreme: float, side: str, bin_size: float = 0.02,
             margin: float = 0.15, max_beyond: float = 0.05) -> dict | None:
    """Wall position on one side of the cameras along one axis."""
    sign = 1.0 if side == "max" else -1.0
    region = v[sign * (v - cam_extreme) > margin]
    if len(region) < 15:
        return None
    centers, heights = histogram_peaks(region, bin_size)
    if len(centers) == 0:  # no peak at all: let the caller use its explicit fallback
        return None
    strong = heights >= max(3.0, 0.2 * heights.max())
    centers = centers[strong]
    if len(centers) == 0:  # only weak peaks: no supported wall on this side
        return None
    order = np.argsort(sign * centers)  # from the cameras outwards
    chosen = None
    for c in centers[order]:
        if np.mean(sign * (region - c) > 0.06) <= max_beyond:
            chosen = c
            break
    if chosen is None:
        chosen = centers[order][-1]
    near = region[np.abs(region - chosen) < 0.04]
    pos = float(np.median(near)) if len(near) else float(chosen)
    return {"pos": pos, "support": int(len(near)), "beyond": float(np.mean(sign * (region - pos) > 0.06))}


def find_ceiling(points: np.ndarray, cam_top: float, box: tuple[float, float, float, float],
                 min_extent: float = 0.8) -> dict | None:
    """Highest horizontal layer whose points spread over the room's interior.

    Not a ceiling: feature points on several walls at one height (poster tops, a
    shelf line) - they hug the walls; the top of a wardrobe or a door frame - it
    covers only a small patch next to a wall. When nothing qualifies the caller
    falls back to an assumed height, which the viewer labels as such.

    This older heuristic is retained for comparison and installations without
    an active model. ceiling_prediction applies the trained candidate filter.
    """
    above = points[points[:, 1] > cam_top + 0.15]
    if len(above) < 8:
        return None
    x0, x1, z0, z1 = box
    need = (max(min_extent, 0.4 * (x1 - x0)), max(min_extent, 0.4 * (z1 - z0)))
    centers, _ = histogram_peaks(above[:, 1], 0.02)
    for c in np.sort(centers)[::-1]:  # highest layer first
        layer = above[np.abs(above[:, 1] - c) < 0.04]
        if len(layer) < 8 or c > cam_top + 2.5:
            continue
        to_wall = np.minimum.reduce([layer[:, 0] - x0, x1 - layer[:, 0], layer[:, 2] - z0, z1 - layer[:, 2]])
        inner = layer[to_wall > 0.25]
        if len(inner) < 8:
            continue
        ext = [float(np.percentile(inner[:, k], 95) - np.percentile(inner[:, k], 5)) for k in (0, 2)]
        if ext[0] >= need[0] and ext[1] >= need[1]:
            return {"pos": float(np.median(layer[:, 1])), "support": int(len(layer)), "interior": int(len(inner)),
                    "interior_extent_xz": ext}
    return None


def _walls(band: np.ndarray, C: np.ndarray) -> dict:
    walls = {}
    for axis, name in ((0, "x"), (2, "z")):
        v = band[:, axis]
        for side in ("min", "max"):
            cam = C[:, axis].min() if side == "min" else C[:, axis].max()
            b = boundary(v, cam, side)
            if b is None:  # nothing seen on that side: fall back to the farthest points
                q = np.percentile(v, 1 if side == "min" else 99) if len(v) else cam
                b = {"pos": float(q), "support": 0, "beyond": 0.0, "fallback": True}
            walls[f"{name}_{side}"] = b
    return walls


def ceiling_prediction(points,top,box,ceiling_model=None):
    from .ceiling_model import active_model,classify_candidates
    installed=ceiling_model is None
    if installed:
        ceiling_model=active_model()
    if ceiling_model:
        try:
            # 설치 모델은 활성화 기록의 추론 설정 지문과 일치해야만 사용한다.
            return classify_candidates(points,top,box,ceiling_model,require_active=installed)
        except (OSError,ValueError,KeyError,TypeError) as error:
            return None,{"status":"model_unavailable","message":"천장 모델을 확인할 수 없어 높이를 가정했어요.",
                         "error":str(error),"real_photo_validated":False}
    return find_ceiling(points,top,box),None


def detect_room(points: np.ndarray, cameras: np.ndarray, ceiling_model=None) -> dict:
    C = cameras
    top = float(C[:, 1].max())
    # walls first (from the band up to a bit above the cameras), then the ceiling inside them
    walls = _walls(points[(points[:, 1] > 0.05) & (points[:, 1] < top + 0.5)], C)
    box = (walls["x_min"]["pos"], walls["x_max"]["pos"], walls["z_min"]["pos"], walls["z_max"]["pos"])
    ceiling,analysis=ceiling_prediction(points,top,box,ceiling_model)
    if ceiling is not None:  # more wall points are usable now that the ceiling height is known
        walls = _walls(points[(points[:, 1] > 0.05) & (points[:, 1] < ceiling["pos"] - 0.1)], C)
    height = ceiling["pos"] if ceiling else DEFAULT_CEILING

    x0, x1 = walls["x_min"]["pos"], walls["x_max"]["pos"]
    z0, z1 = walls["z_min"]["pos"], walls["z_max"]["pos"]
    return {
        "min": [x0, 0.0, z0],
        "max": [x1, height, z1],
        "width": x1 - x0,
        "depth": z1 - z0,
        "height": height,
        "height_source": "detected" if ceiling else "assumed",
        "height_method": ceiling.get("method","horizontal_layer") if ceiling else "assumed",
        "ceiling_analysis":analysis,
        "area_m2": (x1 - x0) * (z1 - z0),
        "walls": walls,
        "ceiling": ceiling,
    }


def shift_room(room: dict, dx: float, dz: float) -> dict:
    out = dict(room)
    out["min"] = [room["min"][0] + dx, room["min"][1], room["min"][2] + dz]
    out["max"] = [room["max"][0] + dx, room["max"][1], room["max"][2] + dz]
    out["walls"] = {k: {**v, "pos": v["pos"] + (dx if k.startswith("x") else dz)} for k, v in room["walls"].items()}
    return out
