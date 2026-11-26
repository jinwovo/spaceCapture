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
            "objects": {"instances": [
                {"class": "bed", "label": "침대", "position": [5.6, .5, 1.5], "view_count": 5, "max_score": .9,
                 "box3d": {"reliable": True, "display": True, "center": [5.6, .25, 1.5], "size": [1.5, .5, 2.0], "front_xz": [-1, 0]}},
                {"class": "couch", "label": "소파", "position": [2, .4, 1], "view_count": 6, "max_score": .9,
                 "box3d": {"reliable": False, "display": False, "center": [2, .4, 1], "size": [1.9, .8, .9], "front_xz": [0, 1]}},
                {"class": "chair", "label": "의자", "position": [1, .4, 2], "view_count": 6, "max_score": .9}]}}


class PlanRenderTests(unittest.TestCase):
    def test_two_rooms_door_walls_and_area(self):
        plan = build_plan(scene())
        self.assertEqual(len(plan["rooms"]), 2)
        self.assertEqual(len(plan["doors"]), 1)
        areas = sorted(r["area_m2"] for r in plan["rooms"])
        self.assertAlmostEqual(areas[0], 2.8*3, delta=1.2)
        self.assertTrue(plan["walls"] and all(len(w) == 4 for w in plan["walls"]))  # 벽은 사각형 목록
        # 크기·방향을 맞추고 표시 판정을 통과한 가구만 그린다(상자 없는 의자, 불합격 소파 제외)
        self.assertEqual([o["class"] for o in plan["objects"]], ["bed"])
        self.assertAlmostEqual(plan["objects"][0]["angle"], -90., places=3)

    def test_svg_png_and_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = render(scene("bedroom"), Path(temp))
            svg = (Path(temp)/"plan.svg").read_text(encoding="utf-8")
            self.assertTrue((Path(temp)/"plan.png").stat().st_size > 10_000)
            self.assertIn('fill-rule="nonzero"', svg)
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
