import unittest

import numpy as np

from spacecapture.sfm import camera_height_outlier_indices,isolated_camera_group_indices,local_height_ratio_outlier_indices,focal_outlier_pose_indices,unsupported_pose_indices


class PoseSupportTests(unittest.TestCase):
    def setUp(self):
        self.points=np.random.default_rng(4).uniform(-4,4,(300,3))

    def test_remote_unsupported_pose_is_rejected_and_well_observed_one_is_kept(self):
        centers=[[0,1,0],[9000,1,0],[9000,1,0],[1,1,0]]
        self.assertEqual(unsupported_pose_indices(centers,[100,2,80,0],self.points),[1,3])

    def test_sparse_but_nearby_pose_is_not_rejected_by_photo_order(self):
        centers=np.array([[0,1,0],[4,1,4],[-4,1,-4]])
        self.assertEqual(unsupported_pose_indices(centers,[14,100,20],self.points),[])
        self.assertEqual(unsupported_pose_indices(centers[::-1],[20,100,14],self.points),[])

    def test_nearby_pose_below_minimum_support_is_rejected(self):
        # procthor-val-0043 재실행 사례: 관측점 4·5개 포즈가 방 근처에서 8~19m 틀린 위치로 등록.
        centers=np.array([[0,1,0],[1,1,1],[-1,1,-1]])
        self.assertEqual(unsupported_pose_indices(centers,[100,4,5],self.points),[1,2])
        self.assertEqual(unsupported_pose_indices(centers,[100,11,12],self.points),[1])

    def test_rejection_does_not_depend_on_scene_units(self):
        centers=np.array([[0,1,0],[9000,1,0]])
        self.assertEqual(unsupported_pose_indices(centers,[100,2],self.points),
                         unsupported_pose_indices(centers*.001,[100,2],self.points*.001))

    def test_weak_pose_with_absorbed_focal_is_rejected(self):
        # procthor-test-0186 사례 모사: 관측점 6개·초점비 2.35, 50개·0.458 / 나머지 약 600개·0.617
        focal=[.617]*20+[2.353,.458]; support=[600]*20+[6,50]
        self.assertEqual(focal_outlier_pose_indices(focal,support),[20,21])
        self.assertEqual(focal_outlier_pose_indices(focal[::-1],support[::-1]),[0,1])

    def test_only_one_symptom_keeps_the_pose(self):
        # 광각 사진(초점비가 다름)이라도 관측이 충분하면 유지, 관측이 적어도 초점이 정상이면 유지.
        self.assertEqual(focal_outlier_pose_indices([.617]*20+[.45],[600]*20+[400]),[])
        self.assertEqual(focal_outlier_pose_indices([.617]*20+[.62],[600]*20+[8]),[])
        self.assertEqual(focal_outlier_pose_indices([.6,2.],[5,600]),[])

    def test_misplaced_photo_group_is_found_by_camera_height(self):
        # LH 9번 재실행 모사: 욕실 묶음이 2.1~3.5m 높이로 잘못 놓임, 나머지는 1.45m 근처.
        heights=[1.45,1.44,1.46,1.45,1.47,1.43,2.23,2.58,3.54]+[1.45]*20
        self.assertEqual(camera_height_outlier_indices(heights),[6,7,8])
        # 무릎 높이(0.95m)·약간 높은 촬영(1.9m)은 유지한다.
        self.assertEqual(camera_height_outlier_indices([1.45]*10+[.95,1.9]),[])
        self.assertEqual(camera_height_outlier_indices([1.0,3.0,1.2]),[])

    def test_isolated_small_camera_group_is_candidate_but_main_house_is_not(self):
        rng=np.random.default_rng(1)
        house=np.column_stack([rng.uniform(0,10,40),np.full(40,1.45),rng.uniform(0,8,40)])
        far=np.column_stack([rng.uniform(30,31,6),np.full(6,1.45),rng.uniform(0,1,6)])
        self.assertEqual(isolated_camera_group_indices(np.vstack([house,far])),list(range(40,46)))
        self.assertEqual(isolated_camera_group_indices(house),[])

    def test_group_at_different_scale_is_found_by_local_height_ratio(self):
        rng=np.random.default_rng(2)
        heights=[1.45]*30+[1.45]*4
        floor=[rng.normal(0,.01,50) for _ in range(30)]
        floor+=[rng.normal(-1.8,.01,50) for _ in range(4)]   # 다른 배율 묶음: 본 바닥이 훨씬 아래
        self.assertEqual(local_height_ratio_outlier_indices(heights,floor),[30,31,32,33])
        # 바닥을 보지 않고 천장 쪽만 본 사진(비율이 작음)·관측이 적은 사진은 판정하지 않는다.
        floor[0]=rng.normal(2.2,.01,50); floor[1]=[0.]*5
        self.assertEqual(local_height_ratio_outlier_indices(heights,floor),[30,31,32,33])
        self.assertEqual(local_height_ratio_outlier_indices([1.45]*30,[rng.normal(0,.01,50) for _ in range(30)]),[])


if __name__=="__main__":
    unittest.main()
