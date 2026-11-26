"""Create explicitly synthetic inspection scenes for browser regression checks."""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_geometry_checks import room, surfaces
from spacecapture.layout import analyze_layout
from spacecapture.plyio import write_points
from spacecapture.pipeline import _write_scene
from spacecapture.floorplan import write_floorplan

points = surfaces([[0, 0], [6, 0], [6, 2], [3, 2], [3, 5], [0, 5]])
cameras = [[2, 1.5, 2], [2.2, 1.5, 2.3], [2.5, 1.5, 1.8]]
folder = ROOT / "scenes/validation-l-office"
folder.mkdir(parents=True, exist_ok=True)
scene = {"format": "spacecapture-scene", "version": 1, "name": "validation-l-office",
         "created": datetime.now().isoformat(), "units": "m", "up": "+y",
         "files": {"points": "points.ply", "splat": None, "floorplan": "floorplan.svg"},
         "room": room(), "room_source": "synthetic_bounding_box", "floor_found": True,
         "scale": {"method": "synthetic_truth"},
         "cameras": [{"name": f"synthetic_{i}", "video_frame": None, "position": p, "quaternion": [0, 0, 0, 1]}
                     for i, p in enumerate(cameras)],
         "quality": {"frames": 3, "registered": 3, "points": len(points), "reprojection_error_px": 0,
                     "warnings": ["검증용 합성 ㄱ자 사무실입니다. 사진에서 복원한 실제 공간이 아닙니다."]},
         "source": {"kind": "synthetic_geometry_test", "ground_truth_area_m2": 21.0}}
scene["layout"] = analyze_layout(points, cameras, scene["room"])
assert scene["layout"]["status"] == "complex_geometry"
write_points(folder / "points.ply", points, np.tile([80, 130, 180], (len(points), 1)).astype(np.uint8))
write_floorplan(folder, scene)
_write_scene(folder, scene)
assert scene["room"]["area_m2"] is None
print("합성 ㄱ자 사무실: 관측 벽 후보 생성, 직사각형 면적 미확정")
