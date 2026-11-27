"""잘못 놓인 사진 묶음 찾기(정답 미사용).

반복 무늬(같은 모양 벽면·창)나 약한 연결 때문에 사진 묶음 하나가 통째로 엉뚱한 곳에 붙는 경우가 있다
(Hypersim 분석, 2026-11-27). 쌍별 에피폴라 일관성이나 '빈 공간 위반'(카메라와 자기 관측면 사이의 다른 점)은
잘못 놓인 카메라를 가려내지 못했다. 효과가 있었던 신호는 두 가지다.
1. 3D 점 공유 연결: 전역 SfM은 3D 점을 하나도 함께 만들지 못한 묶음도 시점 그래프만으로 배치한다.
2. 그 묶음과 나머지 사이 기하 검증 매칭이 현재 포즈와 맞는지(Sampson 거리).
"""
from __future__ import annotations

import numpy as np


def shared_point_components(rec, min_shared=3):
    """카메라 연결 덩어리(두 카메라가 3D 점을 min_shared개 이상 함께 볼 때 연결). 큰 것부터 이름 집합 목록.

    전역 SfM(회전·위치 평균)은 3D 점을 하나도 함께 만들지 못한 사진 묶음도 시점 그래프만으로 배치한다.
    이런 묶음은 반복 무늬 오매칭에 끌려 통째로 엉뚱한 곳에 놓일 수 있다(Hypersim ai_015: 18장, 1.3m).
    """
    names = {im.image_id: im.name for im in rec.images.values() if im.has_pose}
    parent = {i: i for i in names}
    def root(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    counts = {}
    for p in rec.points3D.values():
        ids = sorted({e.image_id for e in p.track.elements if e.image_id in names})
        for a in range(len(ids)):
            for b in range(a+1, len(ids)):
                k = (ids[a], ids[b]); counts[k] = counts.get(k, 0)+1
    for (a, b), c in counts.items():
        if c >= min_shared: parent[root(a)] = root(b)
    groups = {}
    for i, n in names.items(): groups.setdefault(root(i), set()).add(n)
    return sorted(groups.values(), key=len, reverse=True)


def cross_link_consistency(rec, db_path, group, rest, min_inliers=15, max_px=4.):
    """묶음과 나머지 사이 기하 검증 매칭이 현재 포즈와 맞는 비율.

    반환 (쌍 수, 매칭 수, 맞는 비율). 매칭마다 현재 두 카메라 포즈로 만든 기본 행렬의 Sampson 거리(px)가
    max_px 이하이면 맞는 것으로 센다. 제자리에 놓인 묶음은 연결 매칭이 포즈와 맞고, 반복 무늬에 끌려
    엉뚱한 곳에 놓인 묶음은 연결 매칭이 포즈와 어긋난다.
    """
    import pycolmap
    from .sfm import cam_from_world
    img = {im.name: im for im in rec.images.values() if im.has_pose}
    db = pycolmap.Database.open(str(db_path))
    try:
        ids = {im.image_id: im.name for im in db.read_all_images()}
        pairs, geoms = db.read_two_view_geometries()
        keypoints = {}
        def K(im):
            cam = rec.cameras[im.camera_id]; p = cam.params
            cx, cy = (p[1], p[2]) if len(p) >= 3 else (cam.width/2, cam.height/2)
            return np.array([[p[0], 0, cx], [0, p[0], cy], [0, 0, 1.]])
        n_pairs = n = ok = 0
        for pid, g in zip(pairs, geoms):
            a, b = (ids[i] for i in pycolmap.pair_id_to_image_pair(pid))
            if not ((a in group and b in rest) or (a in rest and b in group)) or a not in img or b not in img: continue
            m = g.inlier_matches
            if len(m) < min_inliers: continue
            for name in (a, b):
                if name not in keypoints: keypoints[name] = db.read_keypoints(img[name].image_id)[:, :2]
            Ma, Mb = cam_from_world(img[a]).matrix(), cam_from_world(img[b]).matrix()
            R = Mb[:, :3]@Ma[:, :3].T; t = Mb[:, 3]-R@Ma[:, 3]
            E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])@R
            Ka, Kb = K(img[a]), K(img[b])
            F = np.linalg.inv(Kb).T@E@np.linalg.inv(Ka)
            x1 = np.c_[keypoints[a][m[:, 0]], np.ones(len(m))]; x2 = np.c_[keypoints[b][m[:, 1]], np.ones(len(m))]
            Fx1 = x1@F.T; Ftx2 = x2@F
            d = np.abs(np.sum(x2*Fx1, 1))/np.sqrt(np.maximum(Fx1[:, 0]**2+Fx1[:, 1]**2+Ftx2[:, 0]**2+Ftx2[:, 1]**2, 1e-12))
            n_pairs += 1; n += len(m); ok += int((d < max_px).sum())
        return n_pairs, n, (ok/n if n else None)
    finally:
        db.close()
