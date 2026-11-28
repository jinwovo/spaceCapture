import random
import tempfile
import unittest
from pathlib import Path

from spacecapture.capture_path import capture_order
from spacecapture.photos import taken_at

FORWARD = [0., 0., 0., 1.]  # 단위 사원수: -z를 봄


def cam(name, x, z, frame=None):
    return {"name": name, "position": [x, 1.45, z], "quaternion": FORWARD, "video_frame": frame}


class CaptureOrderTests(unittest.TestCase):
    def test_shuffled_line_becomes_monotonic(self):
        cams = [cam(f"p{i}", i*.5, 0.) for i in range(30)]
        random.Random(0).shuffle(cams)
        path = capture_order(cams)
        xs = [int(n[1:]) for n in path["names"]]
        self.assertEqual(path["method"], "shortest_path")
        self.assertIn(xs, (sorted(xs), sorted(xs, reverse=True)))
        self.assertEqual(path["jumps_over_2m"], 0)

    def test_finishes_one_space_before_the_next(self):
        # 두 방이 1m 간격으로 붙어 있다: 공간 벌점이 있으면 방을 한 번만 넘어간다
        cams = [cam(f"a{i}", (i % 3)*.6, (i//3)*.6) for i in range(9)]+[cam(f"b{i}", 2.2+(i % 3)*.6, (i//3)*.6) for i in range(9)]
        spaces = {c["name"]: c["name"][0] for c in cams}
        names = capture_order(cams, photo_spaces=spaces)["names"]
        switches = sum(1 for a, b in zip(names[:-1], names[1:]) if a[0] != b[0])
        self.assertEqual(switches, 1)

    def test_video_order_with_inserted_photos(self):
        cams = [cam(f"f{i}", i*1., 0., frame=i) for i in range(10)]+[cam("photo", 4.5, .2)]
        names = capture_order(cams)["names"]
        self.assertEqual(capture_order(cams)["method"], "video_frame")
        self.assertEqual([n for n in names if n != "photo"], [f"f{i}" for i in range(10)])
        self.assertIn(names.index("photo"), (5,))  # f4와 f5 사이

    def test_exif_time_order(self):
        cams = [cam(f"p{i}", random.Random(i).uniform(0, 5), 0.) for i in range(6)]
        photos = [{"name": f"p{i}", "taken_at": f"2026-11-28T10:00:{50-i*5:02d}"} for i in range(6)]
        path = capture_order(cams, photos)
        self.assertEqual(path["method"], "exif_time")
        self.assertEqual(path["names"], [f"p{i}" for i in reversed(range(6))])


class TakenAtTests(unittest.TestCase):
    def test_reads_exif_and_handles_missing(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            im = Image.new("RGB", (80, 60), (120, 100, 90))
            exif = Image.Exif()
            exif.get_ifd(0x8769)[36867] = "2026:11:28 09:15:30"
            exif.get_ifd(0x8769)[37521] = "4"
            im.save(Path(d)/"a.jpg", exif=exif)
            im.save(Path(d)/"b.jpg")
            self.assertEqual(taken_at(Path(d)/"a.jpg"), "2026-11-28T09:15:30.40")
            self.assertIsNone(taken_at(Path(d)/"b.jpg"))
            self.assertIsNone(taken_at(Path(d)/"missing.jpg"))


if __name__ == "__main__":
    unittest.main()
