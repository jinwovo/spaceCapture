import json
import tempfile
import unittest
from pathlib import Path

from spacecapture.training_pairs import export_training_pairs


class TrainingPairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def property(self,name,license="MIT"):
        folder=self.root/name
        (folder/"photos").mkdir(parents=True)
        photos=[{"id":"a.jpg","space_id":"a","visible_portals":["ab","bc"]},
                {"id":"b.jpg","space_id":"b","visible_portals":["ab","bc"]},
                {"id":"c.jpg","space_id":"c","visible_portals":["bc"]}]
        for photo in photos: (folder/"photos"/photo["id"]).write_bytes(b"test-image-bytes")
        truth={"id":name,"license":license,"commercial_training_allowed":True,
               "spaces":[{"id":i} for i in ("a","b","c")],
               "portals":[{"id":"ab","spaces":["a","b"]},{"id":"bc","spaces":["b","c"]}],"photos":photos}
        (folder/"property.gt.json").write_text(json.dumps(truth),encoding="utf-8")
        return folder

    def test_splits_are_property_disjoint_and_far_visible_door_is_not_direct_link(self):
        folders=[self.property("first-house"),self.property("second-house")]
        output=self.root/"pairs"
        result=export_training_pairs(folders,output)
        self.assertFalse(set(result["splits"]["train"]["properties"]) & set(result["splits"]["test"]["properties"]))
        self.assertEqual(result["splits"]["validation"]["pairs"],0)
        samples=[json.loads(line) for split in ("train","test") for line in (output/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        far=[s for s in samples if s["target"]["space_ids"]==["a","c"]]
        self.assertTrue(all(s["target"]["shared_opening_ids"]==["bc"] for s in far))
        self.assertTrue(all(not s["target"]["cross_space_direct_opening"] for s in far))
        self.assertFalse(result["neural_training_performed"])

    def test_research_only_corpus_cannot_enter_commercial_training_export(self):
        with self.assertRaisesRegex(ValueError,"상업 학습"):
            export_training_pairs([self.property("not-allowed","CC-BY-NC-4.0"),self.property("owned")],self.root/"output")

    def test_duplicate_house_ids_cannot_be_split_into_train_and_test(self):
        folder=self.property("one-house")
        with self.assertRaisesRegex(ValueError,"같은 집 ID"):
            export_training_pairs([folder,folder],self.root/"output")

    def test_evaluation_only_upstream_split_cannot_enter_training_export(self):
        folder=self.property("upstream-validation")
        path=folder/"property.gt.json"
        truth=json.loads(path.read_text(encoding="utf-8")); truth["evaluation_only"]=True
        path.write_text(json.dumps(truth),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"평가 전용"):
            export_training_pairs([folder,self.property("owned")],self.root/"output")


if __name__=="__main__": unittest.main()
