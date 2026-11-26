"""Small, dependency-free geometry helpers (numpy only)."""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R with R @ a == b (both treated as directions)."""
    a, b = normalize(np.asarray(a, float)), normalize(np.asarray(b, float))
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:  # opposite vectors: rotate 180 deg about any orthogonal axis
        axis = np.cross(a, [1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.cross(a, [0.0, 1.0, 0.0])
        axis = normalize(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane n.x + d = 0 with |n| = 1."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[-1]
    return n, -float(n @ c)


def ransac_plane(
    points: np.ndarray,
    thresh: float,
    iters: int = 400,
    rng: np.random.Generator | None = None,
    normal_hint: np.ndarray | None = None,
    max_angle_deg: float | None = None,
    max_eval: int = 60000,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Plane RANSAC, optionally constrained to normals near `normal_hint`.

    Returns (n, d, inlier_mask over all `points`) or None.
    """
    rng = rng or np.random.default_rng(0)
    n_pts = len(points)
    if n_pts < 3:
        return None
    sample = points if n_pts <= max_eval else points[rng.choice(n_pts, max_eval, replace=False)]
    cos_max = np.cos(np.radians(max_angle_deg)) if max_angle_deg is not None else None
    best_n, best_d, best_count = None, 0.0, 0
    for _ in range(iters):
        p0, p1, p2 = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if cos_max is not None and abs(float(n @ normal_hint)) < cos_max:
            continue
        d = -float(n @ p0)
        count = int((np.abs(sample @ n + d) < thresh).sum())
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    if best_n is None:
        return None
    inliers = np.abs(points @ best_n + best_d) < thresh
    if inliers.sum() >= 3:  # refine on the inliers, keep the constraint
        n, d = fit_plane(points[inliers])
        if cos_max is None or abs(float(n @ normal_hint)) >= cos_max:
            best_n, best_d = n, d
            inliers = np.abs(points @ best_n + best_d) < thresh
    return best_n, best_d, inliers


def histogram_peaks(v: np.ndarray, bin_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima (centers, smoothed counts) of a 1D histogram."""
    lo = float(v.min())
    nb = max(int(np.ceil((float(v.max()) - lo) / bin_size)), 3)
    counts, edges = np.histogram(v, bins=nb, range=(lo, lo + nb * bin_size))
    sm = np.convolve(counts.astype(float), [0.25, 0.5, 0.25], mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    left = np.concatenate([[-1.0], sm[:-1]])
    right = np.concatenate([sm[1:], [-1.0]])
    is_peak = (sm >= left) & (sm >= right) & (sm > 0)
    return centers[is_peak], sm[is_peak]


def planar_extent(points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """5-95 percentile spans of points along two axes inside the plane."""
    a = normalize(np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]))
    b = np.cross(normal, a)
    pa, pb = points @ a, points @ b
    return (float(np.percentile(pa, 95) - np.percentile(pa, 5)),
            float(np.percentile(pb, 95) - np.percentile(pb, 5)))


def triangulate(projections: list[np.ndarray], points2d: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation from normalized image coordinates and 3x4 [R|t] matrices."""
    rows = []
    for P, (x, y) in zip(projections, points2d):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    X = vt[-1]
    return X[:3] / X[3]


def reprojection_errors(projections: list[np.ndarray], points2d: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Errors in normalized image units (multiply by focal length for pixels)."""
    Xh = np.append(X, 1.0)
    errs = []
    for P, xy in zip(projections, points2d):
        p = P @ Xh
        errs.append(np.inf if p[2] <= 0 else float(np.linalg.norm(p[:2] / p[2] - xy)))
    return np.asarray(errs)
