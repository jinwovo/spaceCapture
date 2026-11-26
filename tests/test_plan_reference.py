import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spacecapture.plan_reference import align_to_plan, reference_status, validate_plan
from spacecapture.property_evaluation import evaluate_graph, evaluate_scene_property


class PlanReferenceTests(unittest.TestCase):
    def setUp(self):
        self.graph = {"spaces": [{"id": "observed<1>", "polygon_xz": [[0, 0], [4, 0], [4, 3], [0, 3]],
                                  "center_xz": [2, 1.5]}], "portals": [], "confirmed_room_count": None}
        self.truth = {"spaces": [{"id": "room-1", "kind": "room", "polygon_xz": [[0, 0], [4, 0], [4, 3], [0, 3]]}],
                      "portals": [], "reference": {"match_level": "same_type"}}

    def anchors(self, scale=1, holdout_shift=0):
        # The reconstructed scene is rotated 90 degrees, translated, and optionally scaled.
        target = np.array([[0, 0], [4, 0], [0, 3], [4, 3]], float)
        rotation = np.array([[0, -1], [1, 0]])
        measured = scale*target@rotation+[7, -2]
        points = [{"id": str(i), "scene_xz": a.tolist(), "plan_xz": b.tolist(),
                   "role": "alignment" if i < 3 else "holdout"}
                  for i, (a, b) in enumerate(zip(measured, target))]
        points[-1]["plan_xz"][0] += holdout_shift
        graph = copy.deepcopy(self.graph)
        graph["spaces"][0]["polygon_xz"] = measured.tolist()
        graph["spaces"][0]["center_xz"] = (scale*np.array([2, 1.5])@rotation+[7, -2]).tolist()
        return graph, {"units": "metres", "points": points}

    def test_rigid_registration_and_holdout_without_camera_ground_truth(self):
        graph, anchors = self.anchors(holdout_shift=.3)
        original = copy.deepcopy(graph)
        result, details = align_to_plan(graph, anchors)
        self.assertAlmostEqual(details["anchor_rmse_cm"], 0.)
        self.assertAlmostEqual(details["holdout_mean_error_cm"], 30.)
        np.testing.assert_allclose(result["spaces"][0]["polygon_xz"], [[0,0], [4,0], [0,3], [4,3]], atol=1e-10)
        self.assertEqual(graph, original)

    def test_scale_fit_is_opt_in_and_reported_separately(self):
        graph, anchors = self.anchors(scale=2)
        _, fixed = align_to_plan(graph, anchors)
        self.assertEqual(fixed["scale_scene_to_plan"], 1.)
        self.assertGreater(fixed["anchor_rmse_cm"], 100.)
        anchors["fit_scale"] = True
        _, fitted = align_to_plan(graph, anchors)
        self.assertEqual(fitted["scale_scene_to_plan"], .5)
        self.assertEqual(fitted["anchor_rmse_cm"], 0.)

    def test_collinear_and_nonfinite_anchors_are_rejected(self):
        graph, anchors = self.anchors()
        for i, point in enumerate(anchors["points"][:3]):
            point["scene_xz"] = [i, 0]
        with self.assertRaisesRegex(ValueError, "직선"):
            align_to_plan(graph, anchors)
        anchors["points"][0]["scene_xz"] = [float("nan"), 0]
        with self.assertRaisesRegex(ValueError, "유한"):
            align_to_plan(graph, anchors)

    def test_reflection_is_not_silently_fitted(self):
        graph, anchors = self.anchors()
        for point in anchors["points"]:
            point["plan_xz"][0] *= -1
        anchors["fit_scale"] = True
        _, details = align_to_plan(graph, anchors)
        self.assertGreater(details["anchor_rmse_cm"], 50.)
        self.assertAlmostEqual(np.linalg.det(details["rotation_xz"]), 1.)

    def test_type_plan_never_becomes_verified_actual_unit_ground_truth(self):
        checks = {key: True for key in ("building", "unit_type", "mirror", "balcony", "renovation")}
        self.truth["reference"].update(source_url="https://example.org/plan", photo_source_url="https://example.org/photos",
                                       pairing_checks=checks)
        self.assertFalse(reference_status(self.truth)["ground_truth_eligible"])
        self.truth["reference"]["match_level"] = "same_unit"
        self.assertTrue(reference_status(self.truth)["ground_truth_eligible"])
        checks["renovation"] = None
        self.assertFalse(reference_status(self.truth)["ground_truth_eligible"])

    def test_comparison_writes_report_and_escaped_svg_without_changing_scene(self):
        graph, anchors = self.anchors()
        # Restore polygon winding after constructing anchor positions above.
        graph["spaces"][0]["polygon_xz"] = [graph["spaces"][0]["polygon_xz"][i] for i in (0,1,3,2)]
        scene = {"property_graph": graph, "cameras": []}
        original = copy.deepcopy(scene)
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate_scene_property(scene, self.truth, anchors, folder)
            self.assertEqual(result["mean_space_iou"], 1.)
            self.assertEqual(result["metric_scope"], "original_scale_geometry")
            self.assertFalse(result["reference"]["ground_truth_eligible"])
            self.assertFalse(result["neural_training_performed"])
            self.assertIn("observed&lt;1&gt;", (Path(folder)/"comparison.svg").read_text(encoding="utf-8"))
            self.assertTrue((Path(folder)/"comparison.json").is_file())
        self.assertEqual(scene, original)

    def test_bad_plan_space_references_are_rejected(self):
        self.truth["portals"] = [{"id": "door", "spaces": ["room-1", "missing"], "center_xz": [1, 1], "width_m": .9}]
        with self.assertRaisesRegex(ValueError, "양쪽"):
            validate_plan(self.truth)

    def test_multiple_doors_between_same_spaces_are_individually_matched(self):
        graph = copy.deepcopy(self.graph)
        graph["spaces"].append({"id": "observed-2", "polygon_xz": [[4,0],[7,0],[7,3],[4,3]]})
        truth = copy.deepcopy(self.truth)
        truth["spaces"].append({"id": "room-2", "polygon_xz": [[4,0],[7,0],[7,3],[4,3]]})
        for i,z in enumerate((.5,2.5)):
            graph["portals"].append({"id": str(i), "status": "image_supported", "spaces": ["observed<1>","observed-2"],
                                      "center_xz": [4,z], "width_m": .9})
            truth["portals"].append({"id": str(i), "spaces": ["room-1","room-2"], "center_xz": [4,z], "width_m": .9})
        self.assertEqual(evaluate_graph(graph, truth)["portal_recall"], 1.)
        graph["portals"][1]["center_xz"] = [4,1.3]
        result = evaluate_graph(graph, truth)
        self.assertEqual(result["portal_precision"], .5)
        self.assertEqual(result["portal_recall"], .5)


if __name__ == "__main__":
    unittest.main()
