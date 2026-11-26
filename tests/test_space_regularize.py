import unittest

import numpy as np

from spacecapture.space_regularize import orthogonal_polygon, regularize_spaces, snap_to_walls


def _wall_points(x0, z0, x1, z1, n=200):
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(n):
        y = rng.uniform(.4, 1.8)
        t = rng.uniform(0, 1)
        pts += [[x0+t*(x1-x0), y, z0], [x0+t*(x1-x0), y, z1], [x0, y, z0+t*(z1-z0)], [x1, y, z0+t*(z1-z0)]]
    return np.array(pts)


class SpaceRegularizeTests(unittest.TestCase):
    def test_edges_move_out_to_walls(self):
        step, lower = .1, np.array([-1., -1.])
        labels = np.zeros((60, 60), np.int32)
        labels[13:37, 13:47] = 1           # 관측 영역: 벽(0~4m × 0~3m)보다 약 0.3m 안쪽
        spaces = [{"id": "space-1", "polygon_xz": [[0, 0]]}]
        info = regularize_spaces(spaces, labels, lower, step, _wall_points(0, 0, 4, 3))
        self.assertEqual(info["spaces"], 1)
        poly = np.array(spaces[0]["polygon_xz"])
        self.assertAlmostEqual(poly[:, 0].min(), 0, delta=.05)
        self.assertAlmostEqual(poly[:, 0].max(), 4, delta=.05)
        self.assertAlmostEqual(poly[:, 1].min(), 0, delta=.05)
        self.assertAlmostEqual(poly[:, 1].max(), 3, delta=.05)
        self.assertEqual(spaces[0]["observed_polygon_xz"], [[0, 0]])

    def test_no_wall_evidence_keeps_outline(self):
        mask = np.zeros((40, 40), np.uint8); mask[10:30, 10:30] = 1
        poly = orthogonal_polygon(mask, [0, 0], .1)
        self.assertEqual(snap_to_walls(poly, np.zeros((0, 2))), poly)

    def test_far_walls_are_ignored(self):
        mask = np.zeros((40, 40), np.uint8); mask[10:30, 10:30] = 1
        poly = orthogonal_polygon(mask, [0, 0], .1)
        far = _wall_points(-2, -2, 6, 6)[:, [0, 2]]   # 0.5m보다 먼 벽은 쓰지 않는다
        self.assertEqual(np.round(snap_to_walls(poly, far), 3).tolist(), np.round(poly, 3).tolist())


if __name__ == "__main__":
    unittest.main()
