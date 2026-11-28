"""가구 크기·방향 상자 맞추기 시험(가상 카메라 투영으로 정답을 알고 있는 경우)."""
import unittest

import numpy as np

from spacecapture.object_boxes import corners, floor_ray_candidates, facing, fit_box, fit_boxes, floor_visible, footprint_iou, mark_display, project_box


def look_at(C, target):
    z = np.asarray(target, float)-C; z /= np.linalg.norm(z)
    x = np.cross([0, -1, 0], z); x /= np.linalg.norm(x)   # COLMAP 카메라: x 오른쪽, y 아래, z 앞
    y = np.cross(z, x)
    return np.stack([x, y, z])


def views_for(center, size, yaw, positions, noise=0., seed=0):
    rng = np.random.default_rng(seed)
    P = corners(center, size, yaw)
    views = []
    for C in positions:
        C = np.asarray(C, float)
        cam = (C, look_at(C, [center[0], .4, center[1]]), (1280, 720, 1000.))
        box = project_box(P, cam)+rng.normal(0, noise, 4)
        views.append((box, cam, np.ones(4, bool)))
    return views


RING = [[2.5*np.cos(a), 1.5, 2.5*np.sin(a)] for a in np.linspace(.3, 2.6, 9)]


class ObjectBoxTests(unittest.TestCase):
    def test_recovers_bed_size_and_rotation(self):
        views = views_for((0., 0.), (1.6, 2.0, .55), np.pi/2, RING, noise=2.)   # 90° 돌아간 침대
        box = fit_box({"class": "bed", "position": [.15, .3, -.1]}, views)
        self.assertTrue(box["reliable"])
        w, h, d = box["size"]
        self.assertAlmostEqual(w, 2.0, delta=.12); self.assertAlmostEqual(d, 1.6, delta=.12)
        self.assertAlmostEqual(h, .55, delta=.08)
        self.assertAlmostEqual(box["yaw_deg"] % 180, 0., delta=1.)        # 가로 2.0m가 x축 → 0°
        self.assertLess(np.hypot(box["center"][0], box["center"][2]), .08)

    def test_floating_tv_bottom(self):
        views = views_for((1., .5), (1.2, .1, .7), 0., RING, noise=1.)
        P = corners((1., .5), (1.2, .1, .7), 0., bottom=.9)
        views = [(project_box(P, cam), cam, use) for _, cam, use in views]
        box = fit_box({"class": "tv", "position": [1., 1.2, .5]}, views)
        self.assertAlmostEqual(box["bottom"], .9, delta=.1)
        self.assertAlmostEqual(box["size"][0], 1.2, delta=.12)

    def test_too_few_views_or_unknown_class(self):
        views = views_for((0., 0.), (.5, .5, .9), 0., RING[:2])
        self.assertIsNone(fit_box({"class": "chair", "position": [0, .4, 0]}, views))
        self.assertIsNone(fit_box({"class": "cup", "position": [0, .4, 0]}, views_for((0., 0.), (.1, .1, .1), 0., RING)))

    def test_facing_points_away_from_wall(self):
        box = {"yaw_deg": 0., "size": [1.6, .5, 2.0], "center": [0., .25, 0.]}
        self.assertEqual(facing(box, [[-2, 1.1, 2, 1.3]]), [-0.0, -1.0])     # 벽이 +z 쪽 → 앞은 -z

    def test_floor_visible_rejects_floor_misread_as_bed(self):
        box = {"yaw_deg": 0., "size": [1.6, .5, 2.0], "center": [0., .25, 0.], "bottom": 0.}
        rng = np.random.default_rng(1)
        floor = np.c_[rng.uniform(-.6, .6, 200), rng.uniform(0, .03, 200), rng.uniform(-.8, .8, 200)]
        top = np.c_[rng.uniform(-.6, .6, 200), rng.uniform(.45, .5, 200), rng.uniform(-.8, .8, 200)]
        self.assertGreater(floor_visible(box, floor)[0], .9)
        self.assertLess(floor_visible(box, top)[0], .05)

    def test_duplicate_boxes_keep_the_better_one(self):
        a = {"yaw_deg": 0., "size": [.5, .9, .5], "center": [0., .45, 0.]}
        self.assertGreater(footprint_iou(a, dict(a, center=[.05, .45, .05])), .6)
        self.assertEqual(footprint_iou(a, dict(a, center=[2., .45, 0.])), 0.)

    def test_display_needs_kitchen_evidence_for_appliances(self):
        inst = [{"class": "refrigerator", "box3d": {"reliable": True, "center": [0, .9, 0]}},
                {"class": "chair", "box3d": {"reliable": True, "center": [0, .4, 0]}}]
        spaces = [{"polygon_xz": [[-2, -2], [2, -2], [2, 2], [-2, 2]], "usage_candidates": []}]
        mark_display(inst, spaces)
        self.assertEqual([i["box3d"]["display"] for i in inst], [False, True])
        spaces[0]["usage_candidates"] = [{"usage": "kitchen"}]
        mark_display(inst, spaces)
        self.assertTrue(inst[0]["box3d"]["display"])

    def test_floor_ray_candidates_find_object_without_sfm_points(self):
        P = corners((0., 0.), (1.9, .9, .85), 0.)          # 소파
        dets, cams = {}, {}
        for k, C in enumerate(RING):
            C = np.asarray(C, float)
            cam = (C, look_at(C, [0., .4, 0.]), (1280, 720, 1000.))
            cams[f"p{k}"] = cam
            dets[f"p{k}"] = [{"class": "couch", "score": .9, "box": project_box(P, cam).tolist()}]
        cands = floor_ray_candidates(dets, cams, {"couch": "couch"})
        self.assertEqual(len(cands), 1)
        self.assertGreaterEqual(cands[0]["view_count"], 3)   # 아래 변이 화면 밖인 사진은 제외
        self.assertLess(np.hypot(cands[0]["position"][0], cands[0]["position"][2]), .45)




class SizeRangeTests(unittest.TestCase):
    def test_ranges_follow_width_ge_depth(self):
        from spacecapture.object_boxes import PRIORS, size_range
        for cls in PRIORS:
            lo, hi = size_range(cls)
            self.assertGreaterEqual(lo[0], lo[1]); self.assertGreaterEqual(hi[0], hi[1])
        lo, hi = size_range("bed")   # 2.6 × 1.6 m 침대(가로 ≥ 깊이로 정리된 결과)는 범위 안
        self.assertTrue(lo[0]*.85 <= 2.6 <= hi[0]*1.15 and lo[1]*.85 <= 1.6 <= hi[1]*1.15)


if __name__ == "__main__":
    unittest.main()
