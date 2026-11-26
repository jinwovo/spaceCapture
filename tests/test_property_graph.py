import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spacecapture.synth_property import configuration, generate_property
from spacecapture.property_graph import infer_property
from spacecapture.property_evaluation import evaluate_graph
from spacecapture.layout import analyze_layout
from spacecapture.room import detect_room
from spacecapture.synth import look_rotation


class PropertyGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp=tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.cases={}
        for kind in ("two_rooms","rooms_corridor"):
            folder=Path(temp.name)/kind
            generate_property(folder,kind,seed=9,render_photos=False,log=lambda _:None,landmark_density=60)
            truth=json.loads((folder/"property.gt.json").read_text(encoding="utf-8"))
            with np.load(folder/"observations.npz") as data:
                points,cameras=data["points"].copy(),data["cameras"].copy()
                offsets,indices=data["offsets"],data["observers"]
                observations=[indices[a:b].tolist() for a,b in zip(offsets[:-1],offsets[1:])]
            names=[photo["id"] for photo in truth["photos"]]
            layout=analyze_layout(points,cameras,detect_room(points,cameras))
            result=infer_property(points,cameras,names,observations,planes=layout["planes"])
            cls.cases[kind]=(points,cameras,names,observations,layout,result,truth)

    def test_same_room_views_are_not_counted_as_individual_rooms(self):
        *_,result,truth=self.cases["two_rooms"]
        self.assertEqual(result["space_candidate_count"],2)
        self.assertEqual(len(truth["photos"]),44)
        self.assertIsNone(result["confirmed_room_count"])
        metric=evaluate_graph(result,truth)
        self.assertGreater(metric["mean_space_iou"],.8)
        self.assertEqual(metric["portal_recall"],1.)

    def test_corridor_is_a_separate_unknown_space_and_not_a_confirmed_bedroom(self):
        *_,result,truth=self.cases["rooms_corridor"]
        self.assertEqual(len(result["spaces"]),3)
        metric=evaluate_graph(result,truth)
        self.assertEqual(metric["truth_rooms"],2)
        self.assertIsNone(metric["confirmed_room_count"])
        self.assertEqual(metric["portal_precision"],1.)
        self.assertEqual(metric["portal_recall"],1.)

    def test_opening_has_photos_from_both_sides_but_glass_is_not_excluded(self):
        *_,result,_truth=self.cases["two_rooms"]
        supported=[p for p in result["portals"] if p["status"]=="image_supported"]
        self.assertEqual(len(supported),1)
        portal=supported[0]
        sides={result["photo_spaces"][photo] for photo in portal["evidence_photos"]}
        self.assertEqual(sides,set(portal["spaces"]))
        self.assertFalse(portal["confirmed"])
        self.assertFalse(portal["glass_excluded"])

    def test_openings_sit_at_the_door_jambs_with_measured_width(self):
        # 거리변환 워터셰드: 경계가 가장 좁은 문틀에 놓여야 하고, 폭은 문틀 사이 길이다.
        for kind in ("two_rooms","rooms_corridor"):
            *_,result,truth=self.cases[kind]
            metric=evaluate_graph(result,truth)
            self.assertEqual(metric["portal_recall"],1.)
            self.assertLess(metric["portal_center_mean_error_cm"],15)
            self.assertLess(metric["portal_width_mean_error_cm"],15)
            for portal in result["portals"]:
                self.assertEqual(portal["width_method"],"border_extent")
                self.assertGreater(portal["clearance_width_m"],0)

    def test_no_shared_landmarks_cannot_produce_image_supported_door(self):
        P,C,names,observations,layout,_,_=self.cases["two_rooms"]
        filtered=[]
        for point,views in zip(P,observations):
            filtered.append([j for j in views if (C[j,0]<4)==(point[0]<4)])
        result=infer_property(P,C,names,filtered,planes=layout["planes"])
        self.assertFalse(any(p["status"]=="image_supported" for p in result["portals"]))

    def test_wrong_connectivity_is_penalized_by_separate_ground_truth(self):
        *_,result,truth=self.cases["two_rooms"]
        bad=copy.deepcopy(result)
        for portal in bad["portals"]: portal["spaces"]=["space-1","space-1"]
        metric=evaluate_graph(bad,truth)
        self.assertEqual(metric["portal_precision"],0.)
        self.assertEqual(metric["portal_recall"],0.)

    def test_camera_facing_away_cannot_be_evidence_for_a_visible_opening(self):
        P,C,names,observations,layout,_,_=self.cases["two_rooms"]
        center=np.array([4.,1.1,2.])
        rotations=[look_rotation(camera-center).T for camera in C]
        result=infer_property(P,C,names,observations,planes=layout["planes"],rotations=rotations,
                              intrinsics=[[960,640,593.]]*len(C))
        self.assertFalse(any(portal["status"]=="image_supported" for portal in result["portals"]))
        self.assertTrue(all(portal["view_projection_checked"] for portal in result["portals"]))

    def test_missing_observations_are_explicitly_insufficient(self):
        result=infer_property(np.zeros((20,3)),np.ones((3,3)),['a','b','c'],[])
        self.assertEqual(result["status"],"insufficient_observations")
        self.assertIsNone(result["confirmed_door_count"])

    def test_wide_opening_splits_without_losing_existing_narrow_corridor_cores(self):
        config=configuration("two_rooms",seed=19)
        config["surfaces"][4]["hi"]=1.3
        config["surfaces"][5]["lo"]=2.7
        config["surfaces"][6].update(lo=1.3,hi=2.7)
        config["portals"][0]["width_m"]=1.4
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            generate_property(path,config=config,seed=19,render_photos=False,landmark_density=60,log=lambda _:None)
            truth=json.loads((path/"property.gt.json").read_text(encoding="utf-8"))
            with np.load(path/"observations.npz") as data:
                points,cameras=data["points"],data["cameras"]
                offsets,indices=data["offsets"],data["observers"]
                observations=[indices[a:b].tolist() for a,b in zip(offsets[:-1],offsets[1:])]
            layout=analyze_layout(points,cameras,detect_room(points,cameras))
            result=infer_property(points,cameras,[p["id"] for p in truth["photos"]],observations,planes=layout["planes"])
            self.assertEqual(len(result["spaces"]),2)
            self.assertGreater(result["segmentation"]["split_events"],0)
            self.assertEqual(evaluate_graph(result,truth)["portal_recall"],1.)


if __name__=="__main__":
    unittest.main()
