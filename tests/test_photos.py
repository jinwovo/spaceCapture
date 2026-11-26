import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np

from spacecapture.floorplan import write_floorplan
from spacecapture.photos import prepare_photos
from spacecapture.assessment import assess_scene


class PhotoInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "원본 사진"
        self.source.mkdir()
        self.output = self.root / "work/frames"

    def photo(self, name, size, seed):
        image = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        encoded.tofile(self.source / name)

    def test_unicode_mixed_sizes_duplicates_and_corrupt_images(self):
        self.photo("거실1.png", (1200, 800), 1)
        self.photo("거실2.png", (800, 1200), 2)
        self.photo("거실3.png", (640, 480), 3)
        original = (self.source / "거실1.png").read_bytes()
        (self.source / "중복.png").write_bytes(original)
        (self.source / "손상.jpg").write_bytes(b"invalid")
        info = prepare_photos(self.source, self.output, max_side=800, log=lambda _: None)
        self.assertEqual(info["selected"], 3)
        self.assertEqual(len(info["skipped"]), 2)
        self.assertEqual((self.source / "거실1.png").read_bytes(), original)
        self.assertEqual({tuple(p["size"]) for p in info["photos"]}, {(800, 533), (533, 800), (640, 480)})
        self.assertTrue(all((self.output / n).exists() for n in info["names"]))

    def test_insufficient_input_preserves_previous_prepared_images(self):
        self.output.mkdir(parents=True)
        old = self.output / "photo_00000.jpg"
        old.write_bytes(b"previous")
        self.photo("1.png", (100, 100), 1)
        self.photo("2.png", (100, 100), 2)
        with self.assertRaisesRegex(RuntimeError, "최소 3장"):
            prepare_photos(self.source, self.output)
        self.assertEqual(old.read_bytes(), b"previous")

    def test_same_source_and_destination_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "달라야"):
            prepare_photos(self.source, self.source)

    def test_photo_limit_does_not_silently_drop_rooms(self):
        for i in range(4):
            self.photo(f"{i}.png", (100, 100), i)
        with self.assertRaisesRegex(RuntimeError, "4장"):
            prepare_photos(self.source, self.output, max_photos=3)

    def test_plan_labels_assumptions_and_escapes_listing_title(self):
        scene = {"name": '<script>매물&사진</script>', "scale": {"method": "camera_height"},
                 "room": {"width": 4.2, "depth": 3.6, "area_m2": 15.12, "height": 2.4,
                          "height_source": "assumed", "walls": {k: {"fallback": k == "z_max"}
                           for k in ("x_min", "x_max", "z_min", "z_max")}}}
        write_floorplan(self.root, scene)
        text = (self.root / "floorplan.svg").read_text(encoding="utf-8")
        xml = ElementTree.fromstring(text)
        self.assertFalse(any(e.tag.endswith("script") for e in xml.iter()))
        self.assertIn("실측 보정 필요", text)
        self.assertIn("가정", text)
        self.assertIn('stroke-dasharray="12 8"', text)

    def test_good_geometry_does_not_approve_unknown_outdoor_or_glass(self):
        scene = {"quality": {"frames": 20, "registered": 20, "points": 1000},
                 "floor_found": True, "scale": {"method": "marker"},
                 "room": {"walls": {k: {"support": 100} for k in ("x_min", "x_max", "z_min", "z_max")}}}
        result = assess_scene(scene)
        self.assertEqual(result["status"], "candidate_geometry")
        self.assertFalse(result["automatic_floorplan_approved"])
        self.assertEqual(result["checks"]["glass_and_mirrors"]["status"], "not_checked")
        self.assertEqual(result["checks"]["indoor_outdoor_and_use"]["status"], "not_checked")
        self.assertEqual(result["checks"]["layout_topology"]["status"], "not_checked")
        scene["room"]["walls"]["x_min"]["fallback"] = True
        self.assertEqual(assess_scene(scene)["status"], "limited_geometry")


if __name__ == "__main__":
    unittest.main()
