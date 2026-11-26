"""사물 검출 결과의 3D 위치·합치기·용도 후보 규칙 시험."""
import json
import unittest
from pathlib import Path

import numpy as np

from spacecapture.objects import MODEL, fuse_instances, locate_detection, usage_hints

ROOT = Path(__file__).resolve().parents[1]


class ObjectTests(unittest.TestCase):
    def test_box_uses_front_points_not_the_wall_behind(self):
        rng = np.random.default_rng(0)
        chair = np.column_stack([rng.normal(.2, .1, 60), rng.normal(0, .1, 60), rng.normal(2, .05, 60)])
        wall = np.column_stack([rng.uniform(0, 2, 200), rng.uniform(0, 2, 200), np.full(200, 4.)])
        P = np.vstack([chair, wall]); depth = P[:, 2]
        uv = np.column_stack([P[:, 0]/depth*500+320, P[:, 1]/depth*500+240])
        spot = locate_detection([200, 100, 600, 400], P, uv, depth)
        self.assertLess(abs(spot["position"][2]-2.0), .1)

    def test_fusion_needs_two_photos_and_keeps_distant_objects_apart(self):
        obs = [{"class": "chair", "score": .9, "photo": "a", "position": np.array([0, 0, 0.])},
               {"class": "chair", "score": .8, "photo": "b", "position": np.array([.2, 0, 0.])},
               {"class": "chair", "score": .8, "photo": "c", "position": np.array([3, 0, 0.])},
               {"class": "bed", "score": .9, "photo": "a", "position": np.array([5, 0, 0.])}]
        result = fuse_instances(obs)
        self.assertEqual([(r["class"], r["view_count"]) for r in result], [("chair", 2)])

    def test_usage_needs_strong_evidence_and_ignores_monitor_and_desk(self):
        square = {"id": "s", "polygon_xz": [[0, 0], [4, 0], [4, 4], [0, 4]]}
        def obj(cls, views, score):
            return {"class": cls, "label": cls, "position": [2, .5, 2], "view_count": views, "max_score": score}
        h = usage_hints([square], [obj("tv", 20, .95), obj("dining table", 12, .8), obj("refrigerator", 7, .81)])
        self.assertEqual(h["s"]["usage_candidates"], [])
        h = usage_hints([square], [obj("toilet", 4, .8), obj("bed", 2, .9)])
        self.assertEqual([u["usage"] for u in h["s"]["usage_candidates"]], ["bathroom"])
        h = usage_hints([square], [obj("refrigerator", 5, .8), obj("oven", 3, .75)])
        self.assertEqual([u["usage"] for u in h["s"]["usage_candidates"]], ["kitchen"])

    @unittest.skipUnless(MODEL.exists(), "사물 검출 모델 필요")
    def test_detector_model_hash_and_real_image(self):
        from spacecapture.objects import Detector
        import cv2
        detector = Detector()
        manifest = ROOT/"samples/external/coco/indoor-manifest.json"
        if not manifest.exists():
            self.skipTest("COCO 평가 이미지 없음")
        first = json.loads(manifest.read_text(encoding="utf-8"))["images"][3]
        found = detector.detect(cv2.imread(str(ROOT/"samples/external/coco"/first["file"])))
        self.assertIn("bed", {d["class"] for d in found})


if __name__ == "__main__":
    unittest.main()
