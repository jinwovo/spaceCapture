"""Minimal binary PLY read/write (points and 3D Gaussian splats)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
    "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
    "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8",
}


def write_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    arr = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def read_vertices(path: Path) -> np.ndarray:
    """Vertex element of a binary little-endian PLY as a numpy structured array."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        fields, count, in_vertex, fmt = [], 0, False, None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unterminated PLY header")
            parts = line.decode("ascii", "replace").split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
                elif count:
                    break  # only the vertex element (first) is needed
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError("list properties in vertex element are not supported")
                fields.append((parts[2], _TYPES[parts[1]]))
            elif parts[0] == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format: {fmt}")
        # skip any remaining header lines if we stopped early
        if line.strip() != b"end_header":
            while f.readline().strip() != b"end_header":
                pass
        return np.fromfile(f, dtype=np.dtype(fields), count=count)


def splat_centers(path: Path, min_opacity: float = 0.5, max_size: float = 0.15) -> np.ndarray:
    """Centers of reasonably opaque, reasonably small Gaussians (for geometry estimation)."""
    v = read_vertices(path)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    keep = np.ones(len(v), bool)
    if "opacity" in v.dtype.names:
        keep &= 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64))) >= min_opacity
    if "scale_0" in v.dtype.names:
        size = np.exp(np.max(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1), axis=1))
        keep &= size <= max_size
    return xyz[keep]
