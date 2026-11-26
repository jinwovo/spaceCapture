"""Exercise independent 2D plan registration on existing OWN synthetic scenes."""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spacecapture.property_evaluation import evaluate_scene_property


def main():
    output = ROOT/"artifacts/reference-plan-validation"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"external_property_cases": 0, "neural_training_performed": False, "cases": []}
    for kind, scene_name in (("two_rooms", "property-two-rooms"), ("rooms_corridor", "property-rooms-corridor")):
        scene = json.loads((ROOT/"scenes"/scene_name/"scene.json").read_text(encoding="utf-8"))
        truth = json.loads((ROOT/f"artifacts/property-validation/{kind}_photos/property.gt.json").read_text(encoding="utf-8"))
        actual = {photo["id"]: photo for photo in truth["photos"]}
        pairs = [(camera, actual[camera["source_photo"]]) for camera in scene["cameras"]
                 if camera.get("source_photo") in actual]
        goals = np.array([photo["position"] for _, photo in pairs])[:, [0, 2]]
        # Select a broad triangle, without fitting on the held-out camera positions.
        indices = max(itertools.combinations(range(len(pairs)), 3),
                      key=lambda ids: abs(np.linalg.det(goals[list(ids)][1:]-goals[ids[0]])))
        anchors = {"units": "metres", "fit_scale": False,
                   "note": "자체 합성 정답 촬영 위치의 바닥 투영 대응점입니다. 사람이 표기한 실제 주택 도면이나 벽 모서리 검증이 아닙니다.",
                   "points": []}
        for i, (camera, photo) in enumerate(pairs):
            anchors["points"].append({"id": camera["source_photo"],
                                      "scene_xz": [camera["position"][j] for j in (0, 2)],
                                      "plan_xz": [photo["position"][j] for j in (0, 2)],
                                      "role": "alignment" if i in indices else "holdout"})
        folder = output/kind
        folder.mkdir(parents=True, exist_ok=True)
        (folder/"anchors.json").write_text(json.dumps(anchors, ensure_ascii=False, indent=2), encoding="utf-8")
        started = time.perf_counter()
        result = evaluate_scene_property(scene, truth, anchors, folder)
        result["elapsed_seconds"] = round(time.perf_counter()-started, 4)
        (folder/"comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["cases"].append({"scene": scene_name, **result})
        print(f'{scene_name}: {result["elapsed_seconds"]:.3f}초, IoU {result["mean_space_iou"]:.4f}, '
              f'별도 대응점 오차 {result["alignment"]["holdout_mean_error_cm"]:.3f}cm')
    (output/"result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
