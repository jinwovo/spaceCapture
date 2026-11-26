"""평면도 이미지 변환(plan_image) 시험: 직접 그린 도면으로 축척·실·문·열린 통로를 검사한다."""
import unittest

import cv2
import numpy as np

from spacecapture.plan_image import meters_per_pixel, render_config, segment_plan

PX = 100  # 1m = 100px


def draw_plan():
    """10m x 6m: 왼쪽 방 A(4m) | 문 0.8m | 오른쪽 B(6m)를 1.0m 열린 통로가 있는 벽으로 나눔."""
    image = np.full((1000, 1400, 3), 255, np.uint8)
    ox, oy, t = 200, 200, 20
    W, D = 10*PX, 6*PX
    cv2.rectangle(image, (ox, oy), (ox+W, oy+D), (215, 225, 230), -1)       # 바닥
    def wall(x0, y0, x1, y1):
        cv2.rectangle(image, (x0, y0), (x1, y1), (20, 20, 20), -1)
    wall(ox-t//2, oy-t//2, ox+W+t//2, oy+t//2); wall(ox-t//2, oy+D-t//2, ox+W+t//2, oy+D+t//2)
    wall(ox-t//2, oy-t//2, ox+t//2, oy+D+t//2); wall(ox+W-t//2, oy-t//2, ox+W+t//2, oy+D+t//2)
    x = ox+4*PX                                         # A|B 벽, y 2.0~2.8m 문
    wall(x-t//2, oy, x+t//2, oy+2*PX); wall(x-t//2, oy+int(2.8*PX), x+t//2, oy+D)
    y = oy+3*PX                                         # B 안의 가로벽, x 6.3~7.6m 열린 통로(1.3m)
    wall(x, y-t//2, ox+int(6.3*PX), y+t//2); wall(ox+int(7.6*PX), y-t//2, ox+W, y+t//2)
    for gx in (ox, ox+W):                               # 아래 치수선 끝 눈금 (벽 중심 기준 10m)
        cv2.line(image, (gx, oy+D+80), (gx, oy+D+160), (60, 60, 60), 2)
    cv2.line(image, (ox, oy+D+120), (ox+W, oy+D+120), (60, 60, 60), 1)
    return image


class PlanImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = draw_plan()
        cls.mpp, cls.scale = meters_per_pixel(cls.image, 10.0, "bottom")
        cls.seg = segment_plan(cls.image, cls.mpp)

    def test_scale_from_dimension_ticks(self):
        self.assertAlmostEqual(self.mpp, .01, delta=.0003)

    def test_door_is_portal_and_wide_passage_merges_spaces(self):
        big = [s for s in self.seg["spaces"] if s["area_m2"] > 5]
        self.assertEqual(len(big), 2)                    # 문 폭보다 넓은 1.3m 통로는 공간을 나누지 않는다
        self.assertEqual(len(self.seg["portals"]), 1)
        door = self.seg["portals"][0]
        self.assertAlmostEqual(door["width_m"], .8, delta=.06)
        self.assertAlmostEqual(door["center_xz"][0], 6.0, delta=.05)    # 이미지 원점 기준 (200px+4m)
        self.assertAlmostEqual(door["center_xz"][1], 4.4, delta=.05)
        areas = sorted(s["area_m2"] for s in big)
        self.assertAlmostEqual(areas[0], 3.8*5.8, delta=1.2)
        self.assertAlmostEqual(areas[1], 5.8*5.8-4.6*.2, delta=1.0)  # 칸막이벽 면적 제외

    def test_render_config_has_closed_walls_poses_and_truth(self):
        config, excluded = render_config(self.seg, height=2.3, seed=1, room_views=6)
        self.assertEqual(excluded, [])
        self.assertEqual(len(config["portals"]), 1)
        self.assertEqual(config["texture_style"], "unique_noise")
        self.assertGreater(len(config["poses"]), 12)
        door_tops = [s for s in config["surfaces"] if s["bottom"] == 2.1]
        self.assertEqual(len(door_tops), 2)              # 양쪽 방 벽의 문 위 부분


if __name__ == "__main__":
    unittest.main()


def draw_corridor_plan():
    """3m 방 두 개가 1.0m 폭 복도에 각각 0.8m 문으로 붙은 도면 + 건물 밖 굵은 치수 글자 대신 굵은 치수선."""
    image = np.full((1000, 1400, 3), 255, np.uint8)
    ox, oy, t = 200, 200, 20
    def wall(x0, y0, x1, y1):
        cv2.rectangle(image, (x0, y0), (x1, y1), (20, 20, 20), -1)
    W, D = 6*PX, 4*PX
    cv2.rectangle(image, (ox, oy), (ox+W, oy+D), (215, 225, 230), -1)
    wall(ox-t//2, oy-t//2, ox+W+t//2, oy+t//2); wall(ox-t//2, oy+D-t//2, ox+W+t//2, oy+D+t//2)
    wall(ox-t//2, oy-t//2, ox+t//2, oy+D+t//2); wall(ox+W-t//2, oy-t//2, ox+W+t//2, oy+D+t//2)
    y = oy+3*PX                                   # 복도(위 3m는 방, 아래 1m는 복도)
    for x0, x1 in ((ox, ox+3*PX), (ox+3*PX, ox+W)):
        wall(x0, y-t//2, x0+int(1.0*PX), y+t//2); wall(x0+int(1.8*PX), y-t//2, x1, y+t//2)   # 0.8m 문
    wall(ox+3*PX-t//2, oy, ox+3*PX+t//2, y)      # 두 방 사이 벽
    for gx in (ox, ox+W):
        cv2.line(image, (gx, oy+D+80), (gx, oy+D+160), (60, 60, 60), 2)
    cv2.line(image, (ox, oy+D+120), (ox+W, oy+D+120), (10, 10, 10), 6)   # 굵고 어두운 치수선
    return image


class CorridorPlanTests(unittest.TestCase):
    def test_door_wide_corridor_stays_floor_and_both_doors_connect(self):
        image = draw_corridor_plan()
        mpp, _ = meters_per_pixel(image, 6.0, "bottom")
        # 방향별 판정은 문 틈과 복도 닫힘을 구분한다. 기본 덩어리 판정은 문 틈이 복도와 붙으면
        # 함께 지워 방·복도가 합쳐지는 알려진 한계가 있다(아래 확인).
        merged = segment_plan(image, mpp, closure_mode="component")
        self.assertEqual(len(merged["spaces"]), 1)
        seg = segment_plan(image, mpp, closure_mode="directional")
        big = sorted(s["area_m2"] for s in seg["spaces"])
        self.assertEqual(len(seg["spaces"]), 3)                     # 방 2 + 복도 1, 건물 밖 가짜 공간 없음
        self.assertTrue(all(a < 10 for a in big))
        corridor = min(seg["spaces"], key=lambda s: s["area_m2"])["id"]
        self.assertEqual(sum(corridor in p["spaces"] for p in seg["portals"]), 2)
