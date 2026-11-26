import copy
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pycolmap

from spacecapture.connectivity import match_connectivity
from spacecapture.layout import analyze_layout
from spacecapture.floorplan import write_floorplan


def room():
    return {"min": [0, 0, 0], "max": [6, 2.8, 5], "width": 6., "depth": 5., "height": 2.8,
            "height_source": "assumed", "area_m2": 30.,
            "walls": {"x_min": {"pos": 0}, "x_max": {"pos": 6}, "z_min": {"pos": 0}, "z_max": {"pos": 5}}}


def surfaces(vertices, seed=0):
    rng = np.random.default_rng(seed)
    result = []
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        a, b = np.array(a), np.array(b)
        count = max(200, int(np.linalg.norm(a - b) * 120))
        xz = a + rng.random((count, 1)) * (b - a) + rng.normal(0, .003, (count, 2))
        result.append(np.column_stack([xz[:, 0], rng.uniform(.3, 2.6, count), xz[:, 1]]))
    return np.vstack(result)


class LayoutChecks(unittest.TestCase):
    def test_rectangular_office_is_retained(self):
        points = surfaces([[0, 0], [6, 0], [6, 5], [0, 5]])
        out = analyze_layout(points, [[2, 1.5, 2]], room())
        self.assertTrue(out["room_box_supported"])
        self.assertEqual(out["glass_detection"], "not_checked")

    def test_l_shaped_office_does_not_become_filled_rectangle(self):
        points = surfaces([[0, 0], [6, 0], [6, 2], [3, 2], [3, 5], [0, 5]])
        out = analyze_layout(points, [[2, 1.5, 2]], room())
        self.assertFalse(out["room_box_supported"])
        self.assertEqual(out["status"], "complex_geometry")
        with tempfile.TemporaryDirectory() as temp:
            scene = {"name": "ㄱ자 사무실", "room": room(), "scale": {"method": "marker"}, "layout": out}
            write_floorplan(Path(temp), scene)
            svg = (Path(temp) / "floorplan.svg").read_text(encoding="utf-8")
            self.assertIn("바닥 면적 확정 보류", svg)
            self.assertNotIn("30.00 ㎡", svg)

    def test_oblique_wall_blocks_rectangular_plan(self):
        points = surfaces([[0, 0], [6, 0], [6, 2], [3, 5], [0, 5]])
        out = analyze_layout(points, [[2, 1.5, 2]], room())
        self.assertFalse(out["room_box_supported"])
        self.assertTrue(any(not p["axis_aligned"] for p in out["planes"]))

    def test_missing_wall_is_unknown_instead_of_inferred_glass(self):
        points = surfaces([[0, 0], [6, 0], [6, 5], [0, 5]])
        points = points[points[:, 0] > .05]
        out = analyze_layout(points, [[2, 1.5, 2]], room())
        self.assertFalse(out["room_box_supported"])
        self.assertIn("x_min", out["unsupported_boundaries"])
        self.assertEqual(out["glass_detection"], "not_checked")

    def test_open_ground_does_not_form_a_room(self):
        rng = np.random.default_rng(1)
        points = np.column_stack([rng.uniform(0, 6, 1000), rng.uniform(-.01, .02, 1000), rng.uniform(0, 5, 1000)])
        out = analyze_layout(points, [[2, 1.5, 2]], room())
        self.assertFalse(out["room_box_supported"])
        self.assertEqual(len(out["planes"]), 0)


class MatchGraphChecks(unittest.TestCase):
    def test_unrelated_groups_and_degenerate_matches_stay_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "matches.db"
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE images(image_id INTEGER, name TEXT)")
                db.execute("CREATE TABLE two_view_geometries(pair_id INTEGER, rows INTEGER, config INTEGER)")
                db.executemany("INSERT INTO images VALUES(?,?)", [(i, f"{i}.jpg") for i in range(1, 8)])
                pairs = [(1, 2, 50, 3), (2, 3, 70, 3), (4, 5, 60, 3), (5, 6, 100, 3),
                         (3, 4, 500, 1), (6, 7, 10, 3)]
                db.executemany("INSERT INTO two_view_geometries VALUES(?,?,?)",
                               [(pycolmap.image_pair_to_pair_id(a, b), n, c) for a, b, n, c in pairs])
                db.commit()
            result = match_connectivity(path)
            self.assertEqual(result["status"], "disconnected")
            self.assertEqual([g["count"] for g in result["groups"]], [3, 3, 1])
            self.assertEqual(result["reconstruction_groups"], 2)


class WallBoundaryTests(unittest.TestCase):
    def test_uniform_points_without_wall_peak_use_fallback_instead_of_crashing(self):
        import numpy as np
        from spacecapture.room import detect_room
        rng=np.random.default_rng(5)
        points=np.vstack([rng.uniform([0,0,0],[6,4.3,5],(800,3)),np.column_stack([rng.uniform(0,6,400),rng.normal(4.3,.003,400),rng.uniform(0,5,400)])])
        room=detect_room(points,np.array([[2,1.45,2],[3,1.45,2],[2,1.45,3]]))
        self.assertTrue(np.isfinite([room["width"],room["depth"]]).all())


if __name__ == "__main__":
    unittest.main()
