import copy
import unittest

import numpy as np

from spacecapture.procthor import convert_house
from spacecapture.synth_property import raycast
from spacecapture.plan_reference import reference_status


def wall(identifier, room, a, b):
    return {"id":identifier, "roomId":room, "polygon":[
        {"x":x,"y":y,"z":z} for y in (0.,2.7) for x,z in (a,b)]}


def house():
    rooms=[{"id":f"r{i}","floorPolygon":[{"x":x,"y":0,"z":z} for x,z in
           [(i*4,0),(i*4+4,0),(i*4+4,4),(i*4,4)]]} for i in range(2)]
    walls=[]
    for i,room in enumerate(rooms):
        points=[(p["x"],p["z"]) for p in room["floorPolygon"]]
        for j,(a,b) in enumerate(zip(points,points[1:]+points[:1])):
            walls.append(wall(f"w{i}-{j}",room["id"],a,b))
    door={"id":"d1","room0":"r0","room1":"r1","wall0":"w0-1","wall1":"w1-3",
          "openness":1,"holePolygon":[{"x":.6,"y":0},{"x":1.7,"y":2.1}]}
    return {"rooms":rooms,"walls":walls,"doors":[door]}


class ProcThorTests(unittest.TestCase):
    def test_reversed_other_wall_has_the_same_physical_hole(self):
        config,meta=convert_house(house(),"sample")
        door=config["portals"][0]
        self.assertTrue(np.allclose(door["center_xz"],[4,1.15]))
        self.assertAlmostEqual(door["width_m"],1.1)
        # Rays through both sides of an asymmetric opening must reach an outer wall.
        t,_,_=raycast([3,1,1.15],[[1,0,0]],config)
        self.assertAlmostEqual(t[0],5)
        t,_,_=raycast([5,1,1.15],[[-1,0,0]],config)
        self.assertAlmostEqual(t[0],5)
        t,_,_=raycast([3,1,2.5],[[1,0,0]],config)
        self.assertAlmostEqual(t[0],1)
        self.assertEqual(meta["license"],"Apache-2.0")
        self.assertFalse(reference_status(meta)["real_property_validation"])

    def test_empty_walls_and_closed_doors_are_explicitly_held_back(self):
        example=house(); example["walls"][0]["empty"]=True
        with self.assertRaisesRegex(ValueError,"개방형"):
            convert_house(example,"sample")
        example=house(); example["doors"][0]["openness"]=0
        with self.assertRaisesRegex(ValueError,"개방 상태"):
            convert_house(example,"sample")

    def test_l_shaped_floor_does_not_create_landmarks_in_missing_corner(self):
        example=house()
        example["rooms"][0]["floorPolygon"]=[{"x":x,"y":0,"z":z} for x,z in
             [(0,0),(4,0),(4,4),(2,4),(2,2),(0,2)]]
        config,_=convert_house(example,"sample")
        floors=[p for p in config["horizontal_surfaces"] if p["pos"]==0]
        self.assertFalse(any(p["lo"]<1<p["hi"] and p["bottom"]<3<p["top"] for p in floors))

    def test_invalid_door_and_nonorthogonal_wall_are_rejected(self):
        example=house(); example["doors"][0]["holePolygon"][1]["x"]=10
        with self.assertRaisesRegex(ValueError,"범위"):
            convert_house(example,"sample")
        example=house(); example["walls"][0]["polygon"][1]["z"]+=.3
        with self.assertRaisesRegex(ValueError,"비스듬"):
            convert_house(example,"sample")

    def test_pose_rotations_are_finite_and_input_is_preserved(self):
        example=house(); original=copy.deepcopy(example)
        config,_=convert_house(example,"sample")
        self.assertEqual(example,original)
        for position,rotation,_ in config["poses"]:
            self.assertTrue(np.isfinite(position).all() and np.isfinite(rotation).all())
            np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-6)


if __name__=="__main__":
    unittest.main()
