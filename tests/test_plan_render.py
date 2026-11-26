"""국내 매물 평면도 양식 렌더러 시험."""
import tempfile
import unittest
from pathlib import Path

from spacecapture.plan_render import build_plan, render


def scene(usage=None):
    a = {"id": "space-1", "polygon_xz": [[0, 0], [4, 0], [4, 3], [0, 3]], "usage_candidates": []}
    b = {"id": "space-2", "polygon_xz": [[4.2, 0], [7, 0], [7, 3], [4.2, 3]],
         "usage_candidates": [{"usage": usage, "label": "", "evidence": []}] if usage else []}
    return {"name": "시험 집", "scale": {"method": "camera_height"},
            "property_graph": {"spaces": [a, b], "portals": [
                {"id": "o1", "spaces": ["space-1", "space-2"], "center_xz": [4.1, 1.5], "width_m": .9, "status": "image_supported"}]},
            "objects": {"instances": [{"class": "bed", "label": "침대", "position": [5.6, .5, 1.5], "view_count": 5, "max_score": .9},
                                      {"class": "sink", "label": "세면대", "position": [2, .8, 1], "view_count": 6, "max_score": .9}]}}


class PlanRenderTests(unittest.TestCase):
    def test_two_rooms_door_walls_and_area(self):
        plan = build_plan(scene())
        self.assertEqual(len(plan["rooms"]), 2)
        self.assertEqual(len(plan["doors"]), 1)
        areas = sorted(r["area_m2"] for r in plan["rooms"])
        self.assertAlmostEqual(areas[0], 2.8*3, delta=1.2)
        self.assertTrue(all(len(h) >= 1 for _, h in plan["walls"]))      # 방 자리는 벽의 구멍
        # 주방·욕실 후보가 없는 공간의 세면대 아이콘은 그리지 않는다
        self.assertEqual([o["class"] for o in plan["objects"]], ["bed"])

    def test_svg_png_and_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = render(scene("bedroom"), Path(temp))
            svg = (Path(temp)/"plan.svg").read_text(encoding="utf-8")
            self.assertTrue((Path(temp)/"plan.png").stat().st_size > 10_000)
            self.assertIn('fill-rule="evenodd"', svg)
            self.assertIn("침실1", svg); self.assertIn("공간 1", svg); self.assertIn("평)", svg)
            names = {r["name"]: r["estimated"] for r in summary["rooms"]}
            self.assertEqual(names, {"공간 1": False, "침실1": True})

    def test_single_room_scene_without_graph(self):
        s = {"name": "방", "scale": {"method": "marker"}, "room": {"min": [-2, 0, -1.5], "max": [2, 2.4, 1.5]}}
        with tempfile.TemporaryDirectory() as temp:
            summary = render(s, Path(temp))
            self.assertEqual(len(summary["rooms"]), 1)
            self.assertAlmostEqual(summary["rooms"][0]["area_m2"], 12, delta=1.5)


if __name__ == "__main__":
    unittest.main()
