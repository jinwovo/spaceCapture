import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spacecapture.ceiling_model import DEFAULT_MODEL,FEATURES,audit_dataset,candidate_features,load_model,predict
from spacecapture.room import detect_room


class CeilingLearningTests(unittest.TestCase):
    def test_split_rejects_shared_house_family_or_normalized_plan(self):
        manifest={"houses":{s:[{"id":s,"family":s,"plan_fingerprint":s}] for s in ("train","validation","test")}}
        self.assertEqual(audit_dataset(manifest)["test"]["houses"],1)
        for field in ("id","family","plan_fingerprint"):
            bad=copy.deepcopy(manifest)
            bad["houses"]["test"][0][field]="train"
            with self.assertRaisesRegex(ValueError,"겹쳐"):
                audit_dataset(bad)

    def test_neural_training_learns_nonlinear_xor(self):
        spec=importlib.util.spec_from_file_location("ceiling_training",Path(__file__).resolve().parents[1]/"tools/train-ceiling-model.py")
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        X=np.zeros((128,len(FEATURES)),np.float32)
        X[:,:2]=np.tile([[0,0],[0,1],[1,0],[1,1]],(32,1))
        y=np.tile([0,1,1,0],32).astype(np.float32)
        weights,history=module.train({"X":X,"y":y},{"X":X,"y":y},400,log=lambda _:None)
        self.assertGreater(np.mean((predict(X,weights)>.5)==y),.99)
        self.assertLess(history[-1]["validation_balanced_bce"],history[0]["validation_balanced_bce"]*.1)

    def test_candidate_extractor_allows_high_ceiling_and_has_no_truth_argument(self):
        rng=np.random.default_rng(2)
        points=np.column_stack([rng.uniform(0,6,400),rng.normal(4.3,.003,400),rng.uniform(0,5,400)])
        candidates,X=candidate_features(points,1.5,(0,6,0,5))
        self.assertTrue(any(abs(c["pos"]-4.3)<.02 for c in candidates))
        self.assertEqual(X.shape[1],len(FEATURES))
        self.assertTrue(np.isfinite(X).all())

    def test_model_hash_failure_does_not_fall_back_to_fake_detected_height(self):
        rng=np.random.default_rng(3)
        points=rng.uniform([0,0,0],[6,4.3,5],(1000,3))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"broken.npz"
            path.write_bytes(b"invalid")
            meta=json.loads(DEFAULT_MODEL.with_suffix(".json").read_text(encoding="utf-8"))
            path.with_suffix(".json").write_text(json.dumps(meta),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"SHA-256"):
                load_model(path)
            result=detect_room(points,np.array([[2,1.45,2],[3,1.45,2],[2,1.45,3]]),ceiling_model=path)
            self.assertEqual(result["height_source"],"assumed")
            self.assertEqual(result["ceiling_analysis"]["status"],"model_unavailable")

    def test_empty_or_invalid_points_do_not_produce_neural_candidates(self):
        for points in (np.empty((0,3)),np.full((100,3),np.nan)):
            candidates,X=candidate_features(points,1.5,(0,6,0,5))
            self.assertFalse(candidates)
            self.assertEqual(X.shape,(0,len(FEATURES)))


if __name__=="__main__": unittest.main()
