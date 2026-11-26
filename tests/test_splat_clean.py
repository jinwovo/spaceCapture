import tempfile
import unittest
from pathlib import Path

import numpy as np

from spacecapture.plyio import read_vertices
from spacecapture.splat_clean import clean_splat

FIELDS = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4"), ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4")]


def write_ply(path, rows):
    data = np.array(rows, dtype=np.dtype(FIELDS))
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(data)
    header += "".join(f"property float {n}\n" for n, _ in FIELDS) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode()); data.tofile(f)


class SplatCleanTests(unittest.TestCase):
    def test_removes_outside_transparent_huge_and_vertical_outliers(self):
        small, opaque = np.log(.05), 3.
        rows = [(1, 1, 1, opaque, small, small, small),        # 방 안: 남김
                (1, .01, 1, opaque, small, small, small),      # 바닥: 남김
                (9, 1, 1, opaque, small, small, small),        # 집 밖 9m: 제거
                (1, -1, 1, opaque, small, small, small),       # 바닥 아래: 제거
                (1, 4, 1, opaque, small, small, small),        # 천장 위: 제거
                (1, 1, 1, -8., small, small, small),           # 거의 투명: 제거
                (1, 1, 1, opaque, np.log(3.), small, small)]   # 3m 덩어리: 제거
        scene = {"room": {"min": [0, 0, 0], "max": [3, 2.4, 3], "height": 2.4},
                 "property_graph": {"spaces": [{"polygon_xz": [[0, 0], [3, 0], [3, 3], [0, 3]]}]}}
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d)/"in.ply", Path(d)/"out.ply"
            write_ply(src, rows)
            info = clean_splat(src, dst, scene)
            out = read_vertices(dst)
        self.assertEqual(info["kept"], 2)
        self.assertEqual(len(out), 2)
        self.assertEqual(sum(info["removed"].values()), 5)
        self.assertTrue(np.allclose(sorted(out["y"]), [.01, 1], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
