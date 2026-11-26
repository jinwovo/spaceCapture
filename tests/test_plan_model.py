"""공유 벽선 도면 모델 시험."""
import unittest

from spacecapture.plan_model import build


def scene(gap=.2, door=True):
    a = {"id": "a", "polygon_xz": [[0, 0], [4, 0], [4, 3], [0, 3]]}
    b = {"id": "b", "polygon_xz": [[4+gap, 0], [7, 0], [7, 3], [4+gap, 3]]}
    portals = [{"id": "o1", "spaces": ["a", "b"], "center_xz": [4+gap/2, 1.5], "width_m": .9, "status": "image_supported"}] if door else []
    return {"property_graph": {"spaces": [a, b], "portals": portals}}


def area(poly):
    s = 0
    for (x0, z0), (x1, z1) in zip(poly, poly[1:]+poly[:1]): s += x0*z1-x1*z0
    return abs(s)/2


class PlanModelTests(unittest.TestCase):
    def test_gap_between_rooms_becomes_wall_and_rooms_keep_faces(self):
        m = build(scene(.2))
        self.assertEqual(len(m["rooms"]), 2)
        rooms = {r["id"]: r for r in m["rooms"]}
        self.assertAlmostEqual(rooms["a"]["area_m2"], 12, delta=.05)   # 관측 벽면 유지(벽 칸은 방 밖)
        self.assertAlmostEqual(rooms["b"]["area_m2"], 2.8*3, delta=.05)
        inner = [w for w in m["walls"] if not w["exterior"] and abs(w["rect_xz"][0]-4) < 1e-6 and abs(w["rect_xz"][2]-4.2) < 1e-6]
        self.assertTrue(inner)                                          # 4.0~4.2m 사이가 벽
        for w in m["walls"]:                                            # 모든 축 정렬, 양의 크기
            x0, z0, x1, z1 = w["rect_xz"]; self.assertTrue(x1 > x0 and z1 > z0)

    def test_door_cuts_wall_with_portal_width(self):
        m = build(scene(.2))
        self.assertEqual(len(m["doors"]), 1)
        d = m["doors"][0]
        self.assertEqual(d["axis"], "x"); self.assertAlmostEqual(d["width_m"], .9, places=3)
        self.assertAlmostEqual((d["lo"]+d["hi"])/2, 1.5, places=3)
        for w in m["walls"]:                                            # 문 자리에 벽이 없다
            x0, z0, x1, z1 = w["rect_xz"]
            self.assertFalse(x0 < 4.1 < x1 and z0 < 1.5 < z1)
        self.assertTrue(m["floor_patches"])

    def test_touching_rooms_share_thin_wall_and_exterior_outside(self):
        m = build(scene(0., door=False))
        rooms = {r["id"]: r for r in m["rooms"]}
        self.assertAlmostEqual(rooms["a"]["area_m2"], 12-.06*3, delta=.05)
        ext = [w for w in m["walls"] if w["exterior"]]
        self.assertTrue(all(w["rect_xz"][2] <= 0 or w["rect_xz"][0] >= 0 for w in ext if w["rect_xz"][1] < 1.5 < w["rect_xz"][3] and w["rect_xz"][0] < 1))
        self.assertAlmostEqual(min(w["rect_xz"][0] for w in ext), -.2, places=3)  # 바깥 벽은 방 바깥 0.2m

    def test_interior_polygons_are_axis_aligned(self):
        m = build(scene(.3))
        for r in m["rooms"]:
            p = r["interior_polygon_xz"]
            for (x0, z0), (x1, z1) in zip(p, p[1:]+p[:1]):
                self.assertTrue(abs(x0-x1) < 1e-9 or abs(z0-z1) < 1e-9)
            self.assertAlmostEqual(area(p), r["area_m2"], delta=.01)


if __name__ == "__main__":
    unittest.main()
