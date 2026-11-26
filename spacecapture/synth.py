"""Synthetic room video with known ground truth, for testing the pipeline.

Renders a textured box room (furniture + an ArUco marker of known size) along a
hand-held-like camera path with a small vectorized numpy ray caster. Because
every dimension is known, the reconstruction output can be checked numerically.

World frame: meters, +Y up, room spans x in [0, W], y in [0, H], z in [0, D].
Camera frame: OpenCV convention (x right, y down, z forward).
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOM = {"width": 4.2, "depth": 3.6, "height": 2.5}  # x, z, y extents
FURNITURE = [
    {"name": "bed", "min": [0.15, 0.0, 1.55], "max": [1.75, 0.55, 3.55]},
    {"name": "desk", "min": [3.05, 0.0, 0.05], "max": [4.15, 0.74, 0.70]},
    {"name": "wardrobe", "min": [3.55, 0.0, 2.30], "max": [4.15, 2.00, 3.55]},
]
# ArUco marker on the wall z=0, centre given as (x, y) on that wall.
MARKER = {"dictionary": "DICT_4X4_50", "id": 0, "size": 0.15, "wall": "z0", "center": [1.6, 1.25]}
PPM = 500  # texture pixels per meter
IMAGE_SIZE = (1280, 720)
HFOV_DEG = 70.0
FPS = 30
N_FRAMES = 330

_W, _D, _H = ROOM["width"], ROOM["depth"], ROOM["height"]


# --------------------------------------------------------------------------- textures


def _value_noise(rng: np.random.Generator, h: int, w: int, cell: float, octaves: int = 3) -> np.ndarray:
    """Smooth multi-octave noise in [0, 1], shape (h, w)."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        c = max(cell / (2**o), 1.0)
        gh, gw = max(int(h / c) + 2, 2), max(int(w / c) + 2, 2)
        grid = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return np.clip(out, 0, 1)


def _shapes(rng: np.random.Generator, img: np.ndarray, n: int, min_px: int, max_px: int) -> None:
    """Paint random posters / frames / circles / text so SIFT has something to hold on to."""
    h, w = img.shape[:2]
    for _ in range(n):
        kind = rng.integers(0, 4)
        color = tuple(int(c) for c in rng.integers(20, 235, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = int(rng.integers(min_px, max_px))
        if kind == 0:  # poster with inner pattern
            x2, y2 = min(x + s, w - 1), min(y + int(s * rng.uniform(0.6, 1.4)), h - 1)
            cv2.rectangle(img, (x, y), (x2, y2), color, -1)
            for _ in range(int(rng.integers(3, 9))):
                c2 = tuple(int(c) for c in rng.integers(0, 255, 3))
                px, py = int(rng.integers(x, max(x + 1, x2))), int(rng.integers(y, max(y + 1, y2)))
                cv2.circle(img, (px, py), int(rng.integers(3, max(4, s // 5))), c2, -1)
            cv2.rectangle(img, (x, y), (x2, y2), (30, 30, 30), max(2, s // 40))
        elif kind == 1:
            cv2.circle(img, (x, y), s // 2, color, -1)
            cv2.circle(img, (x, y), s // 4, tuple(255 - c for c in color), -1)
        elif kind == 2:
            cv2.line(img, (x, y), (x + s, y + int(rng.integers(-s, s))), color, max(2, s // 25))
        else:
            txt = "".join(chr(int(c)) for c in rng.integers(65, 91, int(rng.integers(3, 8))))
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s / 60.0, color, max(1, s // 30))


def _wall_texture(rng, w_m: float, base_bgr, n_shapes: int) -> np.ndarray:
    h, w = int(_H * PPM), int(w_m * PPM)
    noise = _value_noise(rng, h, w, 60)
    fine = rng.normal(0, 7, (h, w)).astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    for c in range(3):
        img[..., c] = base_bgr[c] * (0.82 + 0.3 * noise) + fine
    img = np.clip(img, 0, 255).astype(np.uint8)
    _shapes(rng, img, n_shapes, 30, 260)
    # baseboard along the floor (rows are measured from the ceiling down)
    cv2.rectangle(img, (0, h - int(0.08 * PPM)), (w - 1, h - 1), (225, 228, 232), -1)
    return img


def _floor_texture(rng) -> np.ndarray:
    h, w = int(_D * PPM), int(_W * PPM)  # rows = z, cols = x
    img = np.zeros((h, w, 3), np.float32)
    plank_w = int(0.18 * PPM)
    grain = _value_noise(rng, h // 20 + 2, w // 2 + 2, 2.0)
    grain = cv2.resize(grain, (w, h), interpolation=cv2.INTER_CUBIC)
    for x0 in range(0, w, plank_w):
        z = -int(rng.integers(0, int(1.2 * PPM)))
        while z < h:
            length = int(rng.uniform(0.8, 1.6) * PPM)
            tone = rng.uniform(0.75, 1.15)
            base = np.array([60, 105, 150], np.float32) * tone  # BGR wood
            za, zb = max(z, 0), min(z + length, h)
            if zb > za:
                g = grain[za:zb, x0 : x0 + plank_w, None]
                img[za:zb, x0 : x0 + plank_w] = base * (0.75 + 0.5 * g)
                img[za : za + 3, x0 : x0 + plank_w] *= 0.55  # end seam
            z += length
        img[:, x0 : x0 + 2] *= 0.5  # side seam
    img += rng.normal(0, 5, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _ceiling_texture(rng) -> np.ndarray:
    h, w = int(_D * PPM), int(_W * PPM)
    noise = _value_noise(rng, h, w, 80)
    img = np.clip(np.stack([235 + 15 * noise - 7] * 3, -1) + rng.normal(0, 3, (h, w, 1)), 0, 255).astype(np.uint8)
    cv2.circle(img, (w // 2, h // 2), int(0.25 * PPM), (250, 252, 255), -1)
    cv2.circle(img, (w // 2, h // 2), int(0.25 * PPM), (170, 170, 170), 4)
    cv2.circle(img, (int(w * 0.8), int(h * 0.25)), int(0.06 * PPM), (200, 200, 200), -1)
    _shapes(rng, img, 6, 20, 80)
    return img


def _tile_texture(rng, base_bgr) -> np.ndarray:
    s = 512
    noise = _value_noise(rng, s, s, 40)
    img = np.clip(np.stack([base_bgr[c] * (0.7 + 0.55 * noise) for c in range(3)], -1)
                  + rng.normal(0, 6, (s, s, 1)), 0, 255).astype(np.uint8)
    _shapes(rng, img, 10, 20, 120)
    return img


def _paint_marker(tex: np.ndarray) -> None:
    """Paint the ArUco marker (exact metric size) into the z=0 wall texture."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER["dictionary"]))
    side = int(round(MARKER["size"] * PPM))
    marker = cv2.aruco.generateImageMarker(dictionary, MARKER["id"], side, borderBits=1)
    cx, cy = MARKER["center"]
    col0 = int(round(cx * PPM - side / 2))
    row0 = int(round((_H - cy) * PPM - side / 2))  # rows run from the ceiling down
    quiet = int(0.025 * PPM)
    tex[row0 - quiet : row0 + side + quiet, col0 - quiet : col0 + side + quiet] = 255
    tex[row0 : row0 + side, col0 : col0 + side] = marker[..., None]


def build_textures(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    tex = {
        "floor": _floor_texture(rng),
        "ceiling": _ceiling_texture(rng),
        "x0": _wall_texture(rng, _D, (200, 215, 225), 16),
        "xW": _wall_texture(rng, _D, (190, 200, 180), 14),
        "z0": _wall_texture(rng, _W, (215, 205, 200), 16),
        "zD": _wall_texture(rng, _W, (180, 195, 215), 16),
        "furniture": [_tile_texture(rng, c) for c in ((150, 120, 200), (70, 110, 150), (190, 190, 185))],
    }
    _paint_marker(tex["z0"])
    return {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else [t.astype(np.float32) for t in v])
            for k, v in tex.items()}


# --------------------------------------------------------------------------- rendering


def _bilinear(tex: np.ndarray, u: np.ndarray, v: np.ndarray, wrap: bool = False) -> np.ndarray:
    h, w = tex.shape[:2]
    u = u - 0.5
    v = v - 0.5
    if wrap:
        u = np.mod(u, w)
        v = np.mod(v, h)
    else:
        u = np.clip(u, 0, w - 1.001)
        v = np.clip(v, 0, h - 1.001)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    fu = (u - u0)[:, None].astype(np.float32)
    fv = (v - v0)[:, None].astype(np.float32)
    u1 = (u0 + 1) % w if wrap else np.minimum(u0 + 1, w - 1)
    v1 = (v0 + 1) % h if wrap else np.minimum(v0 + 1, h - 1)
    return ((tex[v0, u0] * (1 - fu) + tex[v0, u1] * fu) * (1 - fv)
            + (tex[v1, u0] * (1 - fu) + tex[v1, u1] * fu) * fv)


def intrinsics() -> tuple[float, float, float, float]:
    w, h = IMAGE_SIZE
    f = (w / 2) / math.tan(math.radians(HFOV_DEG) / 2)
    return f, f, w / 2, h / 2


def look_rotation(forward: np.ndarray) -> np.ndarray:
    """World-from-camera rotation (columns = camera x/right, y/down, z/forward in world)."""
    f = forward / np.linalg.norm(forward)
    r = np.cross(np.array([0.0, -1.0, 0.0]), f)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f], axis=1)


def camera_path(n: int = N_FRAMES) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hand-held walk: ~1.1 loops around the room, looking across it (floor-wall line in view)."""
    cx, cz, rx, rz, height = 2.2, 1.75, 1.0, 0.75, 1.38
    poses = []
    for i in range(n):
        s = i / (n - 1) * 1.1 * 2 * math.pi
        pos = np.array([cx + rx * math.cos(s), height + 0.03 * math.sin(2.3 * s), cz + rz * math.sin(s)])
        a = math.atan2(rz * math.sin(s), rx * math.cos(s)) + math.pi + 0.6 * math.sin(1.5 * s)
        # mostly tilted down with two brief upward glances per loop (ceiling)
        p = -0.32 + 0.75 * max(0.0, math.sin(2 * s)) ** 6
        fwd = np.array([math.cos(a) * math.cos(p), math.sin(p), math.sin(a) * math.cos(p)])
        poses.append((pos, look_rotation(fwd)))
    return poses


_TEX = None
_DIRS = None


def _init_worker(seed: int) -> None:
    global _TEX, _DIRS
    _TEX = build_textures(seed)
    fx, fy, cx, cy = intrinsics()
    w, h = IMAGE_SIZE
    jj, ii = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    _DIRS = np.stack([(jj + 0.5 - cx) / fx, (ii + 0.5 - cy) / fy, np.ones_like(jj)], -1).reshape(-1, 3)


def render_frame(args: tuple[int, np.ndarray, np.ndarray]) -> tuple[int, np.ndarray]:
    idx, C, R_wc = args
    tex = _TEX
    d = (_DIRS @ R_wc.T).astype(np.float32)
    o = C.astype(np.float32)
    n = d.shape[0]
    ar = np.arange(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        # room interior: exit distance along each axis
        hi = np.array([_W, _H, _D], np.float32)
        t_axis = np.where(d > 0, (hi - o) * inv, (0 - o) * inv)
        t_axis[d == 0] = np.inf
        axis = np.argmin(t_axis, 1)
        t = t_axis[ar, axis]
        pos_dir = d[ar, axis] > 0
        # face ids: 0 floor, 1 ceiling, 2 x0, 3 xW, 4 z0, 5 zD
        face = np.select([axis == 1, axis == 0, axis == 2],
                         [np.where(pos_dir, 1, 0), np.where(pos_dir, 3, 2), np.where(pos_dir, 5, 4)])
        normal = np.zeros((n, 3), np.float32)
        normal[ar, axis] = np.where(pos_dir, -1.0, 1.0)
        # furniture boxes (exterior hits)
        for k, box in enumerate(FURNITURE):
            bmin, bmax = np.array(box["min"], np.float32), np.array(box["max"], np.float32)
            t1, t2 = (bmin - o) * inv, (bmax - o) * inv
            tmin_ax, tmax_ax = np.minimum(t1, t2), np.maximum(t1, t2)
            tnear, tfar = tmin_ax.max(1), tmax_ax.min(1)
            hit = (tnear <= tfar) & (tnear > 1e-5) & (tnear < t)
            if hit.any():
                ent = tmin_ax.argmax(1)
                t = np.where(hit, tnear, t)
                face = np.where(hit, 6 + k * 3 + ent, face)
                nrm = np.zeros((n, 3), np.float32)
                nrm[ar, ent] = -np.sign(d[ar, ent])
                normal = np.where(hit[:, None], nrm, normal)
    P = o + t[:, None] * d
    color = np.zeros((n, 3), np.float32)
    room_faces = {
        0: ("floor", lambda p: (p[:, 0] * PPM, p[:, 2] * PPM)),
        1: ("ceiling", lambda p: (p[:, 0] * PPM, p[:, 2] * PPM)),
        2: ("x0", lambda p: (p[:, 2] * PPM, (_H - p[:, 1]) * PPM)),
        3: ("xW", lambda p: ((_D - p[:, 2]) * PPM, (_H - p[:, 1]) * PPM)),
        4: ("z0", lambda p: (p[:, 0] * PPM, (_H - p[:, 1]) * PPM)),
        5: ("zD", lambda p: ((_W - p[:, 0]) * PPM, (_H - p[:, 1]) * PPM)),
    }
    for f, (name, uv) in room_faces.items():
        idx_f = np.nonzero(face == f)[0]
        if idx_f.size:
            u, v = uv(P[idx_f])
            color[idx_f] = _bilinear(tex[name], u, v)
    for k in range(len(FURNITURE)):
        for ax in range(3):
            idx_f = np.nonzero(face == 6 + k * 3 + ax)[0]
            if idx_f.size:
                p = P[idx_f]
                a, b = [(2, 1), (0, 2), (0, 1)][ax]
                color[idx_f] = _bilinear(tex["furniture"][k], p[:, a] * PPM, p[:, b] * PPM, wrap=True)
    # simple lighting: ceiling lamp + ambient
    light = np.array([_W / 2, _H - 0.15, _D / 2], np.float32)
    L = light - P
    dist = np.linalg.norm(L, axis=1) + 1e-6
    ndotl = np.clip((normal * L).sum(1) / dist, 0, 1)
    shade = 0.45 + 0.75 * ndotl / (1.0 + 0.12 * dist**2)
    img = color * shade[:, None]
    w, h = IMAGE_SIZE
    img = img.reshape(h, w, 3)
    rng = np.random.default_rng(idx)
    img = cv2.GaussianBlur(img, (0, 0), 0.6) + rng.normal(0, 2.0, img.shape).astype(np.float32)
    return idx, np.clip(img, 0, 255).astype(np.uint8)


def generate(out_path: Path, seed: int = 7, workers: int | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    poses = camera_path()
    w, h = IMAGE_SIZE
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {out_path}")
    frames: dict[int, np.ndarray] = {}
    next_idx = 0
    jobs = [(i, C, R) for i, (C, R) in enumerate(poses)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(seed,)) as ex:
        for i, img in ex.map(render_frame, jobs, chunksize=4):
            frames[i] = img
            while next_idx in frames:
                writer.write(frames.pop(next_idx))
                next_idx += 1
            if i % 30 == 0:
                print(f"  rendered {i + 1}/{len(jobs)}", flush=True)
    writer.release()

    fx, fy, cx, cy = intrinsics()
    gt = {
        "room": ROOM,
        "furniture": FURNITURE,
        "marker": MARKER,
        "image_size": list(IMAGE_SIZE),
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "fps": FPS,
        "camera_centers": [C.round(4).tolist() for C, _ in poses],
        "note": "World: meters, +Y up, room x in [0,W], y in [0,H], z in [0,D].",
    }
    out_path.with_suffix(".gt.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    return out_path
