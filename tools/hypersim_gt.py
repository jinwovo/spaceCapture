"""Hypersim 평가 정답 읽기: 카메라(미터), 사물 종류(NYU40 다수결), 사물 3D 상자(미터)."""
import csv
from pathlib import Path
import h5py, numpy as np

NYU = {4: "bed", 5: "chair", 6: "sofa", 7: "table", 14: "desk", 24: "refrigerator", 25: "television", 33: "toilet", 34: "sink",
       3: "cabinet", 17: "dresser", 32: "night stand", 36: "bathtub", 10: "bookshelf", 39: "otherfurniture"}
# 우리 검출 종류 → 대응 정답 종류
MATCH = {"bed": {"bed"}, "couch": {"sofa"}, "chair": {"chair"}, "dining table": {"table", "desk"}, "tv": {"television"},
         "refrigerator": {"refrigerator"}, "toilet": {"toilet"}, "sink": {"sink"}, "bench": {"sofa", "chair", "otherfurniture"}}


def h5(p):
    with h5py.File(p, "r") as f: return f["dataset"][()]


def load(scene_dir):
    d = Path(scene_dir)
    meta = {r["parameter_name"]: r["parameter_value"] for r in csv.DictReader(open(d/"_detail/metadata_scene.csv"))}
    m = float(meta["meters_per_asset_unit"])
    pos = h5(d/"_detail/cam_00/camera_keyframe_positions.hdf5")*m
    rot = h5(d/"_detail/cam_00/camera_keyframe_orientations.hdf5")   # 카메라→월드, 카메라는 -z를 봄(OpenGL)
    frames = {f"frame.{i:04d}": (pos[i], rot[i]) for i in range(len(pos))}
    bpos = h5(d/"_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5")*m
    bori = h5(d/"_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5")
    bext = h5(d/"_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5")*m
    votes = {}
    for p in sorted((d/"images/scene_cam_00_geometry_hdf5").glob("*.semantic_instance.hdf5")):
        inst = h5(p); sem = h5(str(p).replace(".semantic_instance.", ".semantic."))
        ok = (inst >= 0) & (sem > 0)
        pairs, counts = np.unique(np.stack([inst[ok], sem[ok]], 1), axis=0, return_counts=True)
        for (i, s), c in zip(pairs, counts): votes.setdefault(int(i), {}).setdefault(int(s), 0); votes[int(i)][int(s)] += int(c)
    objects = []
    for i, v in votes.items():
        s = max(v, key=v.get)
        if s not in NYU or not np.isfinite(bpos[i]).all(): continue
        objects.append({"instance": i, "class": NYU[s], "center": bpos[i], "R": bori[i], "extent": bext[i], "pixels": sum(v.values())})
    return frames, objects
