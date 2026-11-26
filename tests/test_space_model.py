"""작은 방 후보 생성·규칙 대체 경로·모델 지문 검사."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spacecapture.space_model import FEATURES, MODEL, decide, fingerprint, load_model, small_room_candidates

STEP = .1


def grid_with_bathroom(door_m):
    """6x4m 큰 방 + 문(door_m)으로 이어진 1.6x1.8m 욕실. 0/1 격자, 0.1m 셀."""
    free = np.zeros((60, 100), np.uint8)
    free[10:50, 10:70] = 1                       # 큰 방
    free[12:30, 72:88] = 1                       # 욕실 (벽 0.2m)
    y0 = 20-int(door_m/STEP/2)
    free[y0:y0+int(door_m/STEP), 70:72] = 1      # 문
    core = cv2.erode(free, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, lab = cv2.connectedComponents(core)
    cores = [(lab == k).astype(np.uint8) for k in range(1, n) if (lab == k).sum() * STEP**2 >= .6]
    cameras = [(40, 30), (20, 20), (80, 18), (80, 25)]
    return free, cores, cameras


class SpaceCandidateTests(unittest.TestCase):
    def test_bathroom_behind_09m_door_is_a_candidate_but_rule_misses_it(self):
        free, cores, cams = grid_with_bathroom(.9)
        self.assertEqual(len(cores), 1)
        cands = small_room_candidates(free, cores, cams, STEP)
        inside = [c for c in cands if c["mask"][12:30, 72:88].sum() > .6*c["mask"].sum()]
        self.assertTrue(inside)
        self.assertEqual(len(inside[0]["features"]), len(FEATURES))
        accepted, info = decide(cands, model=False)
        self.assertEqual(info["method"], "rule")
        self.assertFalse(any(a for c, a in zip(cands, accepted) if c in inside))

    def test_narrow_door_bathroom_is_accepted_by_rule(self):
        free, cores, cams = grid_with_bathroom(.6)
        cands = small_room_candidates(free, cores, cams, STEP)
        accepted, _ = decide(cands, model=False)
        self.assertTrue(any(a and c["mask"][12:30, 72:88].sum() > 0 for c, a in zip(cands, accepted)))

    @unittest.skipUnless(MODEL.exists(), "설치된 공간 후보 모델 필요")
    def test_threshold_tampering_falls_back_to_rule(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/MODEL.name
            shutil.copy2(MODEL, path)
            meta = json.loads(MODEL.with_suffix(".json").read_text(encoding="utf-8"))
            meta["threshold"] = 0.0
            path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "지문"):
                load_model(path)
            free, cores, cams = grid_with_bathroom(.9)
            _, info = decide(small_room_candidates(free, cores, cams, STEP), model=path)
            self.assertEqual(info["method"], "rule")
            meta["fingerprint"] = fingerprint(meta["sha256"], 0.0)
            self.assertNotEqual(meta["fingerprint"], json.loads(MODEL.with_suffix(".json").read_text(encoding="utf-8"))["fingerprint"])


if __name__ == "__main__":
    unittest.main()
