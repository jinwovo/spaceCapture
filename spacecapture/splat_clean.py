"""가우시안 스플랫 잡티 정리: 집 밖·바닥 아래·천장 위·거의 투명·지나치게 큰 스플랫을 지운다.

Brush 학습 결과에는 사진 가장자리의 하늘·허공에 뜬 흐린 덩어리(floater)가 남아 3D가 지저분해 보인다.
복원된 공간 윤곽(property_graph) 또는 방 상자에서 0.5m 넘게 밖에 있는 스플랫을 지운다.
원본은 work/brush/ 에 그대로 남는다. 치수 계산(refine_splat_room)은 정리된 파일을 쓴다.
"""
from pathlib import Path

import cv2
import numpy as np

from .plyio import read_vertices

SETTINGS = {"min_opacity": .02, "max_size_m": 1.5, "margin_m": .5, "below_floor_m": .15, "above_ceiling_m": .5, "cell_m": .05}


def _header(path: Path) -> tuple[bytes, int]:
    with open(path, "rb") as f:
        head = b""
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unterminated PLY header")
            head += line
            if line.strip() == b"end_header":
                return head, len(head)


def footprint(scene: dict, margin_m: float, cell: float):
    """집 바닥 영역(공간 윤곽 합집합 + 여유) 격자와 원점. 공간이 없으면 방 상자."""
    polys = []
    for space in (scene.get("property_graph") or {}).get("spaces", []):
        for key in ("polygon_xz", "observed_polygon_xz"):
            if len(space.get(key) or []) >= 3:
                polys.append(np.asarray(space[key], float))
    if not polys and "room" in scene:
        r = scene["room"]
        polys = [np.array([[r["min"][0], r["min"][2]], [r["max"][0], r["min"][2]], [r["max"][0], r["max"][2]], [r["min"][0], r["max"][2]]])]
    if not polys:
        return None
    allp = np.vstack(polys)
    low = allp.min(0) - margin_m - cell
    shape = (int(np.ceil((allp.max(0)[1] - low[1] + margin_m) / cell)) + 2, int(np.ceil((allp.max(0)[0] - low[0] + margin_m) / cell)) + 2)
    mask = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.round((p - low) / cell).astype(np.int32)], 1)
    k = int(round(margin_m / cell)) * 2 + 1
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask, low


def clean_splat(src: Path, dst: Path, scene: dict, settings: dict | None = None) -> dict:
    s = SETTINGS | (settings or {})
    v = read_vertices(src)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    keep = np.isfinite(xyz).all(1)
    reasons = {}
    if "opacity" in v.dtype.names:
        transparent = 1 / (1 + np.exp(-v["opacity"].astype(np.float64))) < s["min_opacity"]
        reasons["transparent"] = int((keep & transparent).sum()); keep &= ~transparent
    if "scale_0" in v.dtype.names:
        size = np.exp(np.max(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1), 1))
        huge = size > s["max_size_m"]
        reasons["oversized"] = int((keep & huge).sum()); keep &= ~huge
    height = float(scene.get("room", {}).get("height", 2.4))
    top = max(height, 2.4) + s["above_ceiling_m"]
    vertical = (xyz[:, 1] < -s["below_floor_m"]) | (xyz[:, 1] > top)
    reasons["below_floor_or_above_ceiling"] = int((keep & vertical).sum()); keep &= ~vertical
    fp = footprint(scene, s["margin_m"], s["cell_m"])
    if fp is not None:
        mask, low = fp
        cell = np.floor((xyz[:, [0, 2]] - low) / s["cell_m"]).astype(int)
        inside = (cell[:, 0] >= 0) & (cell[:, 1] >= 0) & (cell[:, 0] < mask.shape[1]) & (cell[:, 1] < mask.shape[0])
        inside[inside] = mask[cell[inside, 1], cell[inside, 0]] > 0
        reasons["outside_home"] = int((keep & ~inside).sum()); keep &= inside
    head, _ = _header(src)
    count = int(keep.sum())
    lines = head.split(b"\n")
    lines = [f"element vertex {count}".encode() if l.startswith(b"element vertex") else l for l in lines]
    tmp = Path(dst).with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(b"\n".join(lines))
        v[keep].tofile(f)
    tmp.replace(dst)
    return {"method": "footprint_height_opacity_size", "input": int(len(v)), "kept": count,
            "removed": reasons, "settings": s}
