"""Turn the arbitrary SfM frame into a metric, gravity-aligned room frame.

Target frame: meters, +Y up, floor at y = 0, walls parallel to the X / Z axes.

* up      - average "image up" of the hand-held camera, refined by fitting the floor plane
* yaw     - rotate about +Y until wall points stack into the sharpest X / Z histograms
* scale   - ArUco marker of known size if one is visible; otherwise assume the phone
            was held ~1.45 m above the floor (rough, flagged as an estimate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .geometry import (
    histogram_peaks,
    normalize,
    planar_extent,
    ransac_plane,
    reprojection_errors,
    rot_y,
    rotation_between,
    triangulate,
)
from .sfm import cam_from_world

ASSUMED_CAMERA_HEIGHT = 1.45  # m, phone at chest / eye level while walking


@dataclass
class SfmArrays:
    names: list[str]
    centers: np.ndarray  # (M, 3) camera centers
    R_cw: np.ndarray  # (M, 3, 3) cam_from_world rotations
    xyz: np.ndarray  # (N, 3)
    rgb: np.ndarray  # (N, 3) uint8
    track_len: np.ndarray  # (N,)
    error: np.ndarray  # (N,) reprojection error, px
    tri_angle: np.ndarray  # (N,) widest angle between the rays that see the point, degrees
    observations: list[list[int]]  # camera indices actually observing each point
    intrinsics: list[list[float]]  # image width, height, focal in pixels

    def reliable(self, min_angle: float = 6.0, min_track: int = 2, max_error: float = 4.0) -> np.ndarray:
        """Points safe to measure with: small-baseline points (e.g. a quick glance at a bare
        ceiling) can sit tens of centimeters off along the viewing ray."""
        return (self.track_len >= min_track) & (self.error < max_error) & (self.tri_angle >= min_angle)


def _max_ray_angle(point: np.ndarray, cams: np.ndarray) -> float:
    rays = point - cams
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1.0, 1.0))))


def arrays_from_reconstruction(rec: pycolmap.Reconstruction) -> SfmArrays:
    names, centers, rots, index, intrinsics = [], [], [], {}, []
    for iid in sorted(rec.reg_image_ids()):
        img = rec.images[iid]
        M = cam_from_world(img).matrix()
        R, t = M[:, :3], M[:, 3]
        index[iid] = len(names)
        names.append(img.name)
        rots.append(R)
        centers.append(-R.T @ t)
        camera=rec.cameras[img.camera_id]
        intrinsics.append([camera.width,camera.height,float(camera.params[0])])
    C = np.asarray(centers)
    pts = list(rec.points3D.values())
    angles = [_max_ray_angle(np.asarray(p.xyz), C[[index[e.image_id] for e in p.track.elements]]) for p in pts]
    return SfmArrays(
        names=names,
        centers=C,
        R_cw=np.asarray(rots),
        xyz=np.asarray([p.xyz for p in pts]).reshape(-1, 3),
        rgb=np.asarray([p.color for p in pts], np.uint8).reshape(-1, 3),
        track_len=np.asarray([p.track.length() for p in pts]),
        error=np.asarray([p.error for p in pts]),
        tri_angle=np.asarray(angles),
        observations=[[index[e.image_id] for e in p.track.elements] for p in pts],
        intrinsics=intrinsics,
    )


@dataclass
class Alignment:
    """x_new = scale * R @ x_old + t"""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    info: dict = field(default_factory=dict)

    def apply(self, X: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(X) @ self.rotation.T) + self.translation

    def shifted(self, dx: float, dz: float) -> "Alignment":
        return Alignment(self.scale, self.rotation, self.translation + np.array([dx, 0.0, dz]), self.info)

    def sim3d(self) -> pycolmap.Sim3d:
        return pycolmap.Sim3d(self.scale, pycolmap.Rotation3d(self.rotation), self.translation.reshape(3, 1))


def _camera_up(R_cw: np.ndarray) -> np.ndarray:
    # image "up" is -y in the COLMAP / OpenCV camera; bring it to world and average
    ups = np.einsum("mji,j->mi", R_cw, np.array([0.0, -1.0, 0.0]))
    return normalize(ups.mean(axis=0))


def floor_candidates(P: np.ndarray, C: np.ndarray, up0: np.ndarray, unit: float, rng) -> list[dict]:
    """Horizontal surfaces below the cameras, lowest first.

    Heights along the camera "up" form layers (floor, bed top, desk top, ...).
    Each layer must hold a real plane of some extent; one RANSAC over all low
    points would happily return a tilted plane through two different surfaces.
    """
    h = P @ up0
    cam_h = float(np.median(C @ up0))
    low = h < cam_h - 0.2 * unit
    hl, Pl = h[low], P[low]
    if len(hl) < 15:
        return []
    centers, _ = histogram_peaks(hl, 0.01 * unit)
    found: list[dict] = []
    for c in np.sort(centers):
        if any(abs(c - f["level"]) < 0.06 * unit for f in found):
            continue
        layer = Pl[np.abs(hl - c) < 0.03 * unit]
        if len(layer) < 15:
            continue
        res = ransac_plane(layer, thresh=0.01 * unit, iters=300, rng=rng, normal_hint=up0, max_angle_deg=12)
        if res is None or res[2].sum() < 12:
            continue
        n, d, inl = res
        if min(planar_extent(layer[inl], n)) < 0.15 * unit:  # a cluster, not a surface
            continue
        if np.median(C @ n + d) < 0:  # normal points at the cameras
            n, d = -n, -d
        if np.mean(C @ n + d > 0) < 0.9:
            continue
        found.append({"n": n, "d": float(d), "count": int(inl.sum()), "level": float(c),
                      "cam_height": float(np.median(C @ n + d))})
    if not found:
        return []
    strongest = max(f["count"] for f in found)
    found = [f for f in found if f["count"] >= 0.1 * strongest]  # drop stray layers of noise
    return sorted(found, key=lambda f: -f["cam_height"])  # lowest surface first


def estimate_yaw(P: np.ndarray, unit: float) -> float:
    """Rotation about +Y that makes walls axis aligned (sharpest X/Z histograms)."""
    xz = P[:, [0, 2]]
    if len(xz) > 40000:
        xz = xz[np.random.default_rng(0).choice(len(xz), 40000, replace=False)]
    bin_size = 0.02 * unit

    def score(theta: float) -> float:
        c, s = np.cos(theta), np.sin(theta)
        x = c * xz[:, 0] + s * xz[:, 1]
        z = -s * xz[:, 0] + c * xz[:, 1]
        total = 0.0
        for v in (x, z):
            h, _ = np.histogram(v, bins=max(int((v.max() - v.min()) / bin_size), 1))
            total += float((h.astype(np.float64) ** 2).sum())
        return total

    coarse = np.radians(np.arange(0.0, 90.0, 1.0))
    best = coarse[int(np.argmax([score(t) for t in coarse]))]
    fine = best + np.radians(np.arange(-1.0, 1.0001, 0.1))
    return float(fine[int(np.argmax([score(t) for t in fine]))])


def marker_scale(rec: pycolmap.Reconstruction, image_dir: Path, marker_size: float,
                 dictionary: str = "DICT_4X4_50") -> dict | None:
    """Triangulate ArUco corners with the SfM poses; compare to the printed size."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)
    obs: dict[tuple[int, int], list] = {}
    focal = 1.0
    for iid in rec.reg_image_ids():
        img = rec.images[iid]
        gray = cv2.imread(str(image_dir / img.name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue
        cam = rec.cameras[img.camera_id]
        focal = float(cam.params[0])
        P = cam_from_world(img).matrix()
        for mid, c in zip(np.asarray(ids).reshape(-1), corners):
            xn = np.asarray(cam.cam_from_img(np.asarray(c, np.float64).reshape(4, 2)), float)
            if xn.shape[1] == 3:
                xn = xn[:, :2] / xn[:, 2:3]
            for k in range(4):
                obs.setdefault((int(mid), k), []).append((P, xn[k]))

    found = {}
    for mid in sorted({m for m, _ in obs}):
        X = []
        for k in range(4):
            views = obs.get((mid, k), [])
            if len(views) < 3:
                break
            Ps, xs = [v[0] for v in views], [v[1] for v in views]
            Xk = triangulate(Ps, xs)
            if np.median(reprojection_errors(Ps, xs, Xk)) * focal > 3.0:
                break
            X.append(Xk)
        if len(X) < 4:
            continue
        X = np.asarray(X)
        sides = np.linalg.norm(X - np.roll(X, -1, axis=0), axis=1)
        diags = np.linalg.norm(X[[0, 1]] - X[[2, 3]], axis=1) / np.sqrt(2)
        lengths = np.concatenate([sides, diags])
        side, spread = float(lengths.mean()), float(lengths.std() / lengths.mean())
        if spread <= 0.08:
            found[mid] = {"side_units": side, "spread": round(spread, 4),
                          "views": min(len(obs[(mid, k)]) for k in range(4)), "corners_units": X}
    if not found:
        return None
    side = float(np.median([m["side_units"] for m in found.values()]))
    return {"scale": marker_size / side, "markers": found}


def compute_alignment(
    arrays: SfmArrays,
    rec: pycolmap.Reconstruction,
    image_dir: Path,
    marker_size: float | None = 0.15,
    marker_dict: str = "DICT_4X4_50",
    camera_height: float = ASSUMED_CAMERA_HEIGHT,
    log=print,
) -> Alignment:
    rng = np.random.default_rng(0)
    good = arrays.reliable(min_angle=5.0, min_track=3, max_error=2.0)
    P = arrays.xyz[good] if good.sum() > 100 else arrays.xyz
    C = arrays.centers
    unit = float(np.median(np.linalg.norm(P - C.mean(axis=0), axis=1)))
    up0 = _camera_up(arrays.R_cw)

    # metric scale first (independent of the floor), so floor candidates can be sanity-checked
    marker = marker_scale(rec, image_dir, marker_size, marker_dict) if marker_size else None
    candidates = floor_candidates(P, C, up0, unit, rng)
    floor = None
    for cand in candidates:
        if marker is None or 0.8 <= cand["cam_height"] * marker["scale"] <= 2.2:
            floor = cand
            break
        log(f"  {cand['cam_height'] * marker['scale']:.2f}m 아래 평면은 바닥으로 보기 어려워 건너뜀")
    if floor is not None:
        up, d = floor["n"], floor["d"]
        tilt = float(np.degrees(np.arccos(np.clip(abs(up @ up0), -1, 1))))
        log(f"  바닥 평면 검출: 점 {floor['count']}개 (카메라 기울기 추정과 {tilt:.1f}° 차이)")
    else:
        up = up0
        if marker is not None:  # floor unseen but scale known: assume the usual phone height
            d = -(float(np.median(C @ up0)) - camera_height / marker["scale"])
        else:
            d = -float(np.percentile(P @ up0, 1))
        log("  바닥 평면을 찾지 못해 카메라 방향으로 바닥 높이를 추정했어요 (높이 오차 가능)")

    R1 = rotation_between(up, np.array([0.0, 1.0, 0.0]))
    floor_y = -d  # rotated frame: points on the floor plane have y = up.x = -d
    P1 = P @ R1.T
    band = P1[(P1[:, 1] > floor_y + 0.1 * unit)]
    theta = estimate_yaw(band if len(band) > 50 else P1, unit)
    R = rot_y(theta) @ R1

    heights = C @ up + d
    scale_info: dict = {}
    if marker is not None:
        s = marker["scale"]
        implied = float(np.median(heights)) * s
        scale_info = {"method": "marker", "marker_size_m": marker_size,
                      "markers": {str(k): {kk: vv for kk, vv in v.items() if kk != "corners_units"}
                                  for k, v in marker["markers"].items()},
                      "implied_camera_height_m": round(implied, 3)}
        log(f"  마커로 실제 크기 계산: 마커 {len(marker['markers'])}개, 카메라 높이 {implied:.2f}m로 환산")
    else:
        s = camera_height / float(np.median(heights))
        scale_info = {"method": "camera_height", "assumed_camera_height_m": camera_height}
        log(f"  마커가 보이지 않아 카메라 높이 {camera_height}m 가정으로 크기를 추정했어요. 실제 길이로 보정이 필요해요.")

    info = {"unit": unit, "floor_found": floor is not None, "yaw_deg": float(np.degrees(theta)),
            "scale": scale_info, "median_camera_height_m": float(np.median(heights)) * s}
    return Alignment(scale=s, rotation=R, translation=np.array([0.0, -s * floor_y, 0.0]), info=info)
