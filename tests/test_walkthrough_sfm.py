import unittest

import numpy as np

from spacecapture.sfm import hybrid_pairs, trajectory_deviation


class WalkthroughSfmTests(unittest.TestCase):
    def test_uneven_spacing_is_not_an_outlier(self):
        # 프레임이 건너뛰어진 직선 경로: 간격이 고르지 않아도 경로 위에 있으면 0
        C = np.array([[0, 0, 0], [.1, 0, 0], [.6, 0, 0], [.7, 0, 0], [1.5, 0, 0]], float)
        self.assertTrue(np.allclose(trajectory_deviation(C), 0))

    def test_off_path_frame_is_flagged(self):
        C = np.array([[0, 0, 0], [.1, 0, 0], [.2, .9, 0], [.3, 0, 0], [.4, 0, 0]], float)
        dev = trajectory_deviation(C)
        self.assertEqual(int(np.argmax(dev)), 2)
        self.assertGreater(dev[2], .8)

    def test_start_of_orbit_is_not_flagged(self):
        a = np.linspace(0, 2*np.pi, 40)
        C = np.stack([.6*np.cos(a), np.zeros_like(a), .6*np.sin(a)], 1)
        self.assertLess(trajectory_deviation(C).max(), .05)

    def test_hybrid_pairs(self):
        frames = [f"frame_{i:05d}.jpg" for i in range(400)]
        photos = [f"photo_{i:05d}.jpg" for i in range(5)]
        pairs = set(hybrid_pairs(frames + photos))
        self.assertIn(("frame_00000.jpg", "frame_00001.jpg"), pairs)            # 연속
        self.assertIn(("frame_00000.jpg", "frame_00396.jpg"), pairs)            # 핵심 프레임 루프 폐합
        self.assertIn(("photo_00000.jpg", "photo_00004.jpg"), pairs)            # 사진끼리
        self.assertIn(("frame_00000.jpg", "photo_00002.jpg"), pairs)            # 사진-핵심 프레임
        self.assertTrue(all(a < b for a, b in pairs))
        self.assertLess(len(pairs), 15000)  # 연속 약 8,400 + 핵심 프레임 100장 전체 쌍 4,950


if __name__ == "__main__":
    unittest.main()


class InteriorTextureTests(unittest.TestCase):
    def test_floor_wall_ceiling_look_like_a_home(self):
        from spacecapture.synth_property import _texture_interior
        floor = _texture_interior({"axis": 1, "pos": 0., "lo": 0, "hi": 2, "bottom": 0, "top": 2}, 1, 3)
        wall = _texture_interior({"axis": 0, "pos": 0., "lo": 0, "hi": 3, "bottom": 0, "top": 2.3}, 2, 3)
        ceiling = _texture_interior({"axis": 1, "pos": 2.3, "lo": 0, "hi": 2, "bottom": 0, "top": 2}, 3, 3)
        for image in (floor, wall, ceiling):
            self.assertEqual(image.dtype, np.uint8)
        b, g, r = floor.reshape(-1, 3).mean(0)
        self.assertGreater(r, b)                       # 나무 바닥: 붉은 갈색(BGR)
        self.assertGreater(wall.mean(), 180)           # 밝은 벽지
        self.assertGreater(ceiling.mean(), 225)        # 흰 천장
        self.assertFalse(np.array_equal(_texture_interior({"axis": 0, "pos": 0., "lo": 0, "hi": 3, "bottom": 0, "top": 2.3}, 9, 3), wall))  # 면마다 다른 무늬


class CaptureWarningTests(unittest.TestCase):
    def test_few_points_per_view_warns(self):
        from spacecapture.pipeline import _capture_warnings
        class Rec:
            def __init__(self, points, views): self.p, self.v = points, views
            def num_points3D(self): return self.p
            def num_reg_images(self): return self.v
        self.assertEqual(_capture_warnings(Rec(60000, 470), photos=False), [])
        self.assertIn("천천히", _capture_warnings(Rec(8000, 310), photos=False)[0])
        self.assertIn("겹치는", _capture_warnings(Rec(1000, 50), photos=True)[0])
