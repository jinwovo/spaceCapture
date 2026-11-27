"""3D 점 공유 연결 덩어리 시험(가짜 복원 객체)."""
import unittest
from types import SimpleNamespace as NS

from spacecapture.group_check import shared_point_components


def rec(images, tracks):
    ims = {i: NS(image_id=i, name=f"p{i}", has_pose=True) for i in images}
    pts = {k: NS(track=NS(elements=[NS(image_id=i) for i in t])) for k, t in enumerate(tracks)}
    return NS(images=ims, points3D=pts)


class SharedPointComponentTests(unittest.TestCase):
    def test_group_without_shared_points_is_separate(self):
        tracks = [(1, 2, 3)]*20 + [(4, 5)]*20      # 1-2-3과 4-5는 점을 하나도 함께 만들지 않음
        groups = shared_point_components(rec([1, 2, 3, 4, 5], tracks), 5)
        self.assertEqual([sorted(g) for g in groups], [["p1", "p2", "p3"], ["p4", "p5"]])

    def test_weak_chain_through_bridge_camera_stays_connected(self):
        tracks = [(1, 2)]*30 + [(2, 3)]*6 + [(3, 4)]*30   # 2-3 다리 6점(≥5)으로 이어짐
        self.assertEqual(len(shared_point_components(rec([1, 2, 3, 4], tracks), 5)), 1)
        self.assertEqual(len(shared_point_components(rec([1, 2, 3, 4], tracks), 10)), 2)


if __name__ == "__main__":
    unittest.main()
