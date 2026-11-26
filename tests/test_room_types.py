import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spacecapture.room_types import CLASSES, FEATURES, MODEL, estimate_usage, load_model, space_features


def _rect(x0, z0, x1, z1):
    return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]


def _graph():
    spaces = [{"id": "space-1", "polygon_xz": _rect(0, 0, 8, 4.5)},       # 넓은 거실
              {"id": "space-2", "polygon_xz": _rect(0, 4.7, 3.5, 8)},     # 방
              {"id": "space-3", "polygon_xz": _rect(3.7, 4.7, 5.4, 7)},   # 작은 방(욕실 크기)
              {"id": "space-4", "polygon_xz": _rect(5.6, 4.7, 9, 8)}]     # 방
    portals = [{"spaces": ["space-1", s], "width_m": w} for s, w in (("space-2", .9), ("space-3", .7), ("space-4", .9))]
    return {"spaces": spaces, "portals": portals}


class RoomTypeTests(unittest.TestCase):
    def test_features_shape_and_largest(self):
        rows = space_features(_graph())
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(r) == len(FEATURES) for r in rows))
        f = {n: i for i, n in enumerate(FEATURES)}
        self.assertEqual(rows[0][f["is_largest"]], 1.)
        self.assertEqual(rows[2][f["portal_count"]], 1.)
        self.assertAlmostEqual(rows[2][f["max_portal_width"]], .7)
        self.assertLess(rows[2][f["inner_width_m"]], 2.)

    def test_missing_model_leaves_usage_empty(self):
        g = _graph()
        self.assertEqual(estimate_usage(g, model=False)["method"], "unavailable")
        self.assertTrue(all("usage_estimate" not in s for s in g["spaces"]))

    @unittest.skipUnless(MODEL.exists(), "설치 모델 없음")
    def test_installed_model_and_object_priority(self):
        g = _graph()
        g["spaces"][3]["usage_candidates"] = [{"usage": "bathroom"}]   # 사물 근거(변기)가 있으면 우선
        info = estimate_usage(g)
        self.assertEqual(info["method"], "room_type_mlp")
        est = {s["id"]: s["usage_estimate"] for s in g["spaces"]}
        self.assertEqual(est["space-1"]["usage"], "living")
        self.assertEqual(est["space-4"], est["space-4"] | {"usage": "bathroom", "source": "objects"})
        self.assertAlmostEqual(sum(est["space-2"]["probabilities"].values()), 1., places=2)

    @unittest.skipUnless(MODEL.exists(), "설치 모델 없음")
    def test_single_space_is_not_estimated(self):
        g = {"spaces": [{"id": "space-1", "polygon_xz": _rect(0, 0, 4, 3)}], "portals": []}
        estimate_usage(g)
        self.assertIsNone(g["spaces"][0]["usage_estimate"]["usage"])

    @unittest.skipUnless(MODEL.exists(), "설치 모델 없음")
    def test_changed_threshold_is_rejected(self):
        folder = Path(tempfile.mkdtemp())
        try:
            shutil.copy(MODEL, folder/MODEL.name)
            meta = json.loads(MODEL.with_suffix(".json").read_text(encoding="utf-8")); meta["threshold"] = 0.
            (folder/MODEL.with_suffix(".json").name).write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(folder/MODEL.name)
            g = _graph()
            self.assertEqual(estimate_usage(g, folder/MODEL.name)["method"], "unavailable")
        finally:
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
