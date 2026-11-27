"""Camera poses + sparse points with COLMAP (pycolmap), CPU only.

Frames pulled out of a video carry no EXIF, so COLMAP has no idea of the focal
length, and the global (GLOMAP) mapper depends on a reasonable focal prior.
We start from a typical phone-video field of view, let bundle adjustment refine
it, feed the refined value back as the new prior until it settles, and fall
back to other guesses (then to the incremental mapper) if that fails.
"""

from __future__ import annotations

import shutil
import json
import time
from pathlib import Path

import numpy as np
import pycolmap
from .connectivity import match_connectivity

# focal / max(width, height); 0.8 ~ 64 deg horizontal FOV, typical for phone main cameras in video mode
FOCAL_FACTORS = (0.8, 0.65, 1.0, 1.25)


def cam_from_world(image) -> pycolmap.Rigid3d:
    """pycolmap >= 3.12 exposes this as a method (rig support); older versions as a property."""
    T = image.cam_from_world
    return T() if callable(T) else T


def _set_focal_prior(db_path: Path, focal: float, relative: bool = False) -> None:
    db = pycolmap.Database.open(db_path)
    for cam in db.read_all_cameras():
        params = list(cam.params)
        params[0] = focal * max(cam.width, cam.height) if relative else focal
        cam.params = params
        cam.has_prior_focal_length = True
        db.update_camera(cam)
    db.close()


def _image_size(db_path: Path) -> tuple[int, int]:
    db = pycolmap.Database.open(db_path)
    cam = db.read_all_cameras()[0]
    db.close()
    return cam.width, cam.height


def _best(recs: dict) -> pycolmap.Reconstruction | None:
    return max(recs.values(), key=lambda r: (r.num_reg_images(), r.num_points3D()), default=None)


def _focal(rec: pycolmap.Reconstruction, relative: bool = False) -> float:
    cameras=[rec.cameras[rec.images[i].camera_id] for i in rec.reg_image_ids()]
    if relative:
        return float(np.median([c.params[0] / max(c.width, c.height) for c in cameras]))
    return float(np.median([c.params[0] for c in cameras]))


def _segment_distance(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0., 1.))
    return float(np.linalg.norm(p - (a + t * ab)))


def trajectory_deviation(C):
    """Distance of each camera from the path between its neighbours.

    Measured to the segment (not the midpoint) so uneven spacing from skipped
    frames is not mistaken for a jump; the ends use the line through the next
    two cameras. The old midpoint/half-jump rule peeled 46 good frames off the
    start of a house walk-through one by one (2026-11-26)."""
    C = np.asarray(C, float)
    dev = np.zeros(len(C))
    for i in range(1, len(C) - 1):
        dev[i] = _segment_distance(C[i], C[i - 1], C[i + 1])
    if len(C) >= 3:
        for i, (a, b) in ((0, (C[1], C[2])), (len(C) - 1, (C[-2], C[-3]))):
            d = b - a; n = np.linalg.norm(d)
            off = C[i] - a
            dev[i] = float(np.linalg.norm(off - np.dot(off, d) / n**2 * d)) if n > 1e-9 else float(np.linalg.norm(off))
    return dev


def drop_trajectory_outliers(rec: pycolmap.Reconstruction, log=print) -> list[str]:
    """A hand-held video moves smoothly, so a camera far from the midpoint of its
    neighbours was mis-registered (typically a glance at a bare ceiling or wall).
    Remove the worst one at a time - an outlier also drags its neighbours' midpoints."""
    dropped: list[str] = []
    tainted: set[int] = set()  # points triangulated with a wrong camera: wrong too
    while True:
        imgs = sorted((rec.images[i] for i in rec.reg_image_ids() if rec.images[i].name.startswith(FRAME_PREFIX)),
                      key=lambda im: im.name)
        if len(imgs) < 5:
            break
        C = np.asarray([np.asarray(im.projection_center()) for im in imgs])
        step = float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))
        dev = trajectory_deviation(C)
        threshold = max(4 * float(np.median(dev[1:-1])), 2 * step)
        j = int(np.argmax(dev))
        if dev[j] <= threshold:
            break
        tainted.update(p.point3D_id for p in imgs[j].points2D if p.has_point3D())
        rec.deregister_frame(imgs[j].frame_id)
        dropped.append(imgs[j].name)
    if dropped:
        for pid in tainted:
            if pid in rec.points3D:
                rec.delete_point3D(pid)
        ba = pycolmap.BundleAdjustmentOptions()
        ba.print_summary = False
        pycolmap.bundle_adjustment(rec, ba)  # re-settle the rest without the bad cameras
        log(f"  궤적에서 크게 벗어난 프레임 {len(dropped)}장과 그 점 {len(tainted)}개 제외 후 재조정: {', '.join(dropped)}")
    return dropped


MIN_PHOTO_SUPPORT = 12


def unsupported_pose_indices(centers, supports, points):
    """Order-independent, conservative rejection of unsupported photo poses.

    Poses with fewer than MIN_PHOTO_SUPPORT observed points are rejected anywhere.
    A distant but well-observed camera remains valid. Distances are relative to
    the robust landmark extent; neither room size nor ground truth is assumed.
    """
    points=np.asarray(points,float).reshape(-1,3)
    centers=np.asarray(centers,float).reshape(-1,3)
    support=np.asarray(supports)
    if len(points)<30:
        return np.flatnonzero(support==0).tolist()
    origin=np.median(points,axis=0)
    radius=max(float(np.percentile(np.linalg.norm(points-origin,axis=1),80)),1e-8)
    remote=np.linalg.norm(centers-origin,axis=1)>8*radius
    # 사진별 카메라는 포즈 6 + 초점 1 = 7개 미지수다. 관측점 12개 미만은 최소해(P4Pf)
    # 근처라 검증 여유가 없다. 2026-11-10 정답이 있는 사진 장면 8개에서 12개 미만
    # 포즈 3개가 모두 3.7~19m 틀렸고, 올바른 포즈 중 12개 미만은 없었다.
    weak=support<MIN_PHOTO_SUPPORT
    return np.flatnonzero((support==0) | weak | (remote & weak)).tolist()


def focal_outlier_pose_indices(focal_ratios, supports, focal_tolerance=1.25, support_fraction=.15, min_support=30):
    """사진별 카메라에서 초점거리가 포즈 오류를 흡수한 경우를 찾는다.

    관측점이 매우 적고(중앙값의 15% 미만) 동시에 정제된 초점거리가 다른 사진의
    중앙값과 25% 넘게 다른 사진만 제외한다. 둘 중 하나만 해당하면 유지한다
    (다른 폰·광각 렌즈 사진이나 일부만 겹치는 사진은 정상일 수 있다).
    """
    f=np.asarray(focal_ratios,float).reshape(-1)
    support=np.asarray(supports,float).reshape(-1)
    if len(f)<5 or not np.isfinite(f).all() or np.any(f<=0):
        return []
    weak=support<max(min_support,support_fraction*float(np.median(support)))
    off=np.abs(np.log(f/float(np.median(f))))>np.log(focal_tolerance)
    return np.flatnonzero(weak & off).tolist()


def camera_height_outlier_indices(heights, tolerance=.6):
    """정렬 후 바닥 기준 촬영 높이가 다른 사진들의 중앙값과 크게 다른 사진.

    손에 들고 찍은 실내 사진은 대체로 비슷한 높이에서 찍힌다. 문 너머로만 약하게
    연결된 작은 방 사진 묶음이 통째로 잘못된 위치·배율로 놓이면 높이가 크게 달라진다
    (LH 평면도 렌더 9번 재실행: 욕실 9장이 24~36m 밖, 높이 2.1~3.5m). 합성 사진은
    모두 1.45m에서 찍었으므로 실제 촬영 높이 분포에서는 기준을 다시 검증해야 한다.
    """
    y=np.asarray(heights,float).reshape(-1)
    if len(y)<5:
        return []
    return np.flatnonzero(np.abs(y-float(np.median(y)))>tolerance).tolist()


def deregister_photos(rec: pycolmap.Reconstruction, names, log=print):
    """이름으로 지정한 사진 포즈를 제외하고 남은 포즈로 다시 조정한다."""
    wanted=set(names)
    targets={rec.images[i].frame_id for i in rec.reg_image_ids() if rec.images[i].name in wanted}
    if rec.num_reg_images()-len(targets)<3:
        raise RuntimeError("관측 근거가 있는 사진 포즈가 3개 미만이에요. 더 겹치게 촬영해 주세요.")
    for frame_id in targets:
        rec.deregister_frame(frame_id)
    ba=pycolmap.BundleAdjustmentOptions(); ba.print_summary=False
    config=pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids(): config.add_image(image_id)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    pycolmap.logging.minloglevel=3
    summary=pycolmap.create_default_bundle_adjuster(ba,config,rec).solve()
    if not summary.is_solution_usable():
        raise RuntimeError("사진 포즈 제외 후 재조정 계산이 수렴 가능한 해를 만들지 못했어요. 이 결과를 저장하지 않습니다.")


def drop_unsupported_photo_poses(rec: pycolmap.Reconstruction, log=print):
    images=[rec.images[i] for i in sorted(rec.reg_image_ids())]
    centers=[im.projection_center() for im in images]
    supports=[sum(p.has_point3D() for p in im.points2D) for im in images]
    remote=unsupported_pose_indices(centers,supports,[p.xyz for p in rec.points3D.values()])
    focal=[]
    for im in images:
        camera=rec.cameras[im.camera_id]
        focal.append(float(camera.params[0])/max(camera.width,camera.height))
    degenerate=[i for i in focal_outlier_pose_indices(focal,supports) if i not in remote]
    bad=sorted(remote+degenerate)
    records=[{"name":images[i].name,"observed_points":supports[i],
              "reason":"사진 순서와 무관한 관측 근거 부족(관측점 12개 미만)·외곽 포즈"} for i in remote]
    records+=[{"name":images[i].name,"observed_points":supports[i],"focal_ratio":round(focal[i],4),
               "median_focal_ratio":round(float(np.median(focal)),4),
               "reason":"관측점이 매우 적고 초점거리가 다른 사진과 크게 달라 포즈 오류로 판단"} for i in degenerate]
    if len(images)-len(bad)<3:
        raise RuntimeError("관측 근거가 있는 사진 포즈가 3개 미만이에요. 더 겹치게 촬영해 주세요.")
    for i in bad:
        rec.deregister_frame(images[i].frame_id)
    if bad:
        ba=pycolmap.BundleAdjustmentOptions(); ba.print_summary=False
        config=pycolmap.BundleAdjustmentConfig()
        for image_id in rec.reg_image_ids(): config.add_image(image_id)
        config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        # global_mapping may reset native logging; keep app output in Korean.
        pycolmap.logging.minloglevel=3
        summary=pycolmap.create_default_bundle_adjuster(ba,config,rec).solve()
        if not summary.is_solution_usable():
            raise RuntimeError("사진 포즈 제외 후 재조정 계산이 수렴 가능한 해를 만들지 못했어요. 이 결과를 저장하지 않습니다.")
        log(f"  관측 근거가 부족한 사진 포즈 {len(bad)}장 제외 후 재조정: {', '.join(r['name'] for r in records)}")
    return records


# 3D 점을 함께 만든 카메라끼리만 '구조로 이어진' 것으로 본다(쌍별 공유 점 ≥ 이 값).
# Hypersim 23곳: 1~10 사이에서 결과 동일, 5를 쓴다(2026-11-27).
LINK_MIN_SHARED_POINTS = 5


def drop_unlinked_groups(rec: pycolmap.Reconstruction, log=print, min_shared=None, db_path=None):
    """가장 큰 묶음과 3D 점을 함께 만들지 못한 카메라 묶음을 제외한다(배치 근거 없음).

    전역 SfM은 3D 점을 하나도 함께 만들지 못한 묶음도 시점 그래프만으로 배치한다. 반복 무늬 오매칭에 끌리면
    묶음이 통째로 엉뚱한 곳에 놓인다. Hypersim 23곳·반복 실행(2026-11-27)에서 이런 묶음 8개(5~39장)는
    모두 정답 대비 1m 이상 어긋났고(ai_015 전부, ai_008 한 실행의 39장 → 전체 RMSE 158.7cm),
    제자리에 놓인 연결 없는 묶음은 한 번도 없었다. 정상 실행에서는 묶음이 생기지 않아 제외도 없다.
    연결 매칭과 현재 배치의 일치 비율은 기록만 한다(잘못 놓인 묶음도 0.45가 나온 적이 있어 판정에 안 씀).
    """
    from .group_check import cross_link_consistency, shared_point_components
    groups=shared_point_components(rec, LINK_MIN_SHARED_POINTS if min_shared is None else min_shared)
    if len(groups)<2:
        return []
    main=groups[0]; drop=[]; records=[]
    for g in groups[1:]:
        try:
            pairs,n,frac=cross_link_consistency(rec,db_path,g,main) if db_path else (0,0,None)
        except Exception:  # noqa: BLE001 - 기록용 지표
            pairs,n,frac=0,0,None
        log(f"  3D 점으로 이어지지 않은 사진 묶음 {len(g)}장 제외(배치 근거 없음, 연결 매칭 {n}개·배치와 맞는 비율 "
            f"{'-' if frac is None else f'{frac:.2f}'})")
        for name in sorted(g):
            records.append({"name":name,"group_size":len(g),"cross_pairs":pairs,"cross_matches":n,
                            "consistent_fraction":None if frac is None else round(frac,3),"action":"excluded",
                            "reason":"가장 큰 묶음과 3D 점을 함께 만들지 못한 묶음(시점 그래프만으로 배치, 위치 신뢰 불가)"})
        drop+=sorted(g)
    deregister_photos(rec,drop,log)
    log(f"  가장 큰 묶음 {len(main)}장으로 계속")
    return records


FRAME_PREFIX = "frame_"
# SIFT 추가 설정(실험·조정용). 비어 있으면 COLMAP 기본값.
SIFT_OPTIONS: dict = {}
# 영상 프레임 전용: 무늬 적은 흰 벽에서도 특징점을 잡도록 저대비 검출(기본 0.0067 → 0.003).
# 실내 재질 걷기 영상 477장: 등록 378→473장, 공간 IoU 0.687→0.713, 처리 6.7→10.1분.
# 사진 입력(LH 시험 4채)은 IoU 0.874→0.875로 차이가 없고 시간만 약 2배라 적용하지 않는다(2026-11-26).
VIDEO_SIFT_OPTIONS: dict = {"peak_threshold": 0.003}


def hybrid_pairs(names, frame_prefix=FRAME_PREFIX, overlap=15, keyframes=100):
    """영상 프레임 + 사진의 매칭 쌍.

    영상 프레임: 이웃 15장 + 2^k 간격(연속 매칭)과, 약 100장의 핵심 프레임끼리 전체 쌍(되돌아온 방을
    다시 잇는 루프 폐합). 사진: 사진끼리 전체 쌍 + 핵심 프레임 전부. 정답 정보는 쓰지 않는다."""
    frames = sorted(n for n in names if n.startswith(frame_prefix))
    photos = sorted(n for n in names if not n.startswith(frame_prefix))
    pairs = set()
    jumps = list(range(1, overlap + 1)) + [2 ** j for j in range(4, 12)]
    for i, a in enumerate(frames):
        for k in jumps:
            if i + k < len(frames):
                pairs.add((a, frames[i + k]))
    step = max(1, int(np.ceil(len(frames) / keyframes)))
    keys = frames[::step]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs.add((a, b))
    for i, a in enumerate(photos):
        for b in photos[i + 1:] + keys:
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def _global(db: Path, image_dir: Path, out: Path, seed: int = 0) -> pycolmap.Reconstruction | None:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    options = pycolmap.GlobalPipelineOptions()
    options.random_seed = seed  # same video in, same model out
    options.mapper.random_seed = seed
    return _best(pycolmap.global_mapping(db, image_dir, out, options))


def run_sfm(
    image_dir: Path,
    work_dir: Path,
    max_image_size: int = 1600,
    max_features: int = 8192,
    exhaustive_max: int = 150,
    focal_px: float | None = None,
    log=print,
    unordered: bool = False,
    shared_camera: bool = False,
    mapper: str = "global",
) -> pycolmap.Reconstruction:
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    if db.exists():
        db.unlink()
    n_images = len(list(image_dir.glob("*.jpg")))
    cpu = pycolmap.Device.cpu

    t = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"  # one phone camera: focal + radial distortion refined in BA
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = max_image_size
    extraction.sift.max_num_features = max_features
    for key, value in SIFT_OPTIONS.items():
        setattr(extraction.sift, key, value)
    video_extraction = pycolmap.FeatureExtractionOptions(extraction.todict())
    for key, value in VIDEO_SIFT_OPTIONS.items():
        if key not in SIFT_OPTIONS:
            setattr(video_extraction.sift, key, value)
    all_names = sorted(p.name for p in image_dir.glob("*.jpg"))
    frame_names = [n for n in all_names if n.startswith(FRAME_PREFIX)]
    mixed = not unordered and len(frame_names) < len(all_names)  # 영상 프레임 + 사진
    if mixed:
        # 영상 프레임은 한 카메라(같은 초점·왜곡), 사진은 사진마다 카메라(확대·다른 폰 가능)
        pycolmap.extract_features(db, image_dir, image_names=frame_names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=video_extraction, device=cpu)
        pycolmap.extract_features(db, image_dir, image_names=[n for n in all_names if n not in set(frame_names)],
                                  camera_mode=pycolmap.CameraMode.PER_IMAGE,
                                  reader_options=reader, extraction_options=extraction, device=cpu)
    else:
        camera_mode = pycolmap.CameraMode.PER_IMAGE if unordered and not shared_camera else pycolmap.CameraMode.SINGLE
        pycolmap.extract_features(db, image_dir, camera_mode=camera_mode,
                                  reader_options=reader, extraction_options=extraction if unordered else video_extraction, device=cpu)
    log(f"  특징점 추출 완료 ({time.time() - t:.0f}초)")

    t = time.time()
    matching = pycolmap.FeatureMatchingOptions()
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.random_seed = 0  # same video in, same model out
    if unordered or n_images <= exhaustive_max:
        # all pairs: slower, but also links the end of the walk back to its start
        pycolmap.match_exhaustive(db, matching_options=matching, verification_options=verification, device=cpu)
        mode = "전체 쌍"
    else:
        # 긴 영상·영상+사진: 연속 프레임 + 핵심 프레임 루프 폐합 + 사진 쌍 (2026-11-26)
        pair_list = work_dir / "pairs.txt"
        pairs = hybrid_pairs(all_names)
        pair_list.write_text(chr(10).join(f"{a} {b}" for a, b in pairs), encoding="utf-8")
        pairing = pycolmap.ImportedPairingOptions()
        pairing.match_list_path = str(pair_list)
        pycolmap.match_image_pairs(db, matching_options=matching, pairing_options=pairing,
                                   verification_options=verification, device=cpu)
        mode = f"연속 프레임+루프 폐합{'+사진' if mixed else ''}, {len(pairs)}쌍"
    log(f"  특징점 매칭 완료 ({mode}, {time.time() - t:.0f}초)")
    graph = match_connectivity(db)
    (work_dir / "match_graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  사진 연결 검사: {len(graph['groups'])}개 묶음, 기하 검증 쌍 {graph['verified_pairs']}개")

    t = time.time()
    width, height = _image_size(db)
    relative = (unordered and not shared_camera) or mixed  # 사진마다 크기가 달라 긴 변 대비 비율로 사전값을 준다
    if relative:
        guesses = [focal_px / max(width, height)] if focal_px else list(FOCAL_FACTORS)
    else:
        guesses = [focal_px] if focal_px else [f * max(width, height) for f in FOCAL_FACTORS]
    best: pycolmap.Reconstruction | None = None
    for gi, guess in enumerate(guesses if mapper == "global" else []):
        prior = guess
        for it in range(4):  # feed the refined focal back as the prior until it settles
            out = work_dir / f"sparse_try{gi}_{it}"
            _set_focal_prior(db, prior, relative=relative)
            rec = _global(db, image_dir, out)
            if rec is None:
                break
            refined = _focal(rec, relative=relative)
            if best is None or (rec.num_reg_images(), rec.num_points3D()) > (best.num_reg_images(), best.num_points3D()):
                best = rec
            if abs(refined - prior) / prior < 0.01:
                break
            prior = refined
        if best is not None and best.num_reg_images() >= 0.9 * n_images:
            break
    used = "global"
    if best is None or best.num_reg_images() < 0.6 * n_images:
        got = 0 if best is None else best.num_reg_images()
        log(f"  전역 SfM이 {got}/{n_images}장만 등록 → 증분 SfM으로 재시도")
        out = work_dir / "sparse_incremental"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        rec = _best(pycolmap.incremental_mapping(db, image_dir, out))
        if rec is not None and (best is None or rec.num_reg_images() > best.num_reg_images()):
            best, used = rec, "incremental"
    if best is None or best.num_reg_images() < 3:
        raise RuntimeError("카메라 위치를 복원하지 못했어요. 더 천천히, 겹치게 다시 촬영해 주세요.")
    if not unordered:
        drop_trajectory_outliers(best, log)  # 영상 프레임만 대상(이름 순서 = 시간 순서)
    if True:  # 영상도: 무늬 없는 벽만 본 프레임이 관측점 0~3개로 등록되는 경우가 있다(실내 재질 걷기 영상, 2026-11-26)
        # Removing a pose deletes points only it supported, which can leave other
        # poses below the support/focal criteria; repeat until stable (LH plan
        # 227, 2026-11-10: three wrong poses survived a single pass).
        excluded=[]
        for _ in range(4):
            removed=drop_unsupported_photo_poses(best,log)
            excluded+=removed
            if not removed: break
        linked=drop_unlinked_groups(best,log,db_path=db)
        excluded+=[r for r in linked if r["action"]=="excluded"]
        (work_dir/"pose_filter.json").write_text(json.dumps({"method":"unordered_observation_support+focal_outlier+unlinked_group_check",
            "excluded":excluded,
            "ground_truth_used":False},ensure_ascii=False,indent=2),encoding="utf-8")

    final = work_dir / "sparse"
    shutil.rmtree(final, ignore_errors=True)
    final.mkdir(parents=True)
    best.write(final)
    for tmp in work_dir.glob("sparse_try*"):
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"  카메라 위치 복원: {best.num_reg_images()}/{n_images}장, 3D 점 {best.num_points3D()}개, "
        f"초점거리 {_focal(best):.0f}px ({used}, {time.time() - t:.0f}초)")
    return best


def isolated_camera_group_indices(centers, gap_m=3.0, max_fraction=.25):
    """다른 모든 카메라와 gap_m 넘게 떨어진 작은 카메라 묶음.

    올바른 사진 복원에서는 문을 지나는 촬영이 방들을 잇는다. 문 너머로만 약하게
    이어진 작은 방 묶음이 통째로 잘못된 위치에 놓이면 집 범위 밖으로 떨어진다
    (LH 평면도 렌더 9번: 욕실 9장이 19~36m 밖). 그런 묶음의 상대 위치는 확정하지
    않고 제외한다. 미터 단위(정렬 후) 좌표를 받는다.
    """
    C=np.asarray(centers,float).reshape(-1,3)
    n=len(C)
    if n<5:
        return []
    d=np.linalg.norm(C[:,None,[0,2]]-C[None,:,[0,2]],axis=2)
    label=-np.ones(n,int); groups=0
    for start in range(n):
        if label[start]>=0: continue
        stack=[start]; label[start]=groups
        while stack:
            u=stack.pop()
            for v in np.flatnonzero((d[u]<=gap_m)&(label<0)):
                label[v]=groups; stack.append(v)
        groups+=1
    if groups==1:
        return []
    sizes=np.bincount(label)
    main=int(np.argmax(sizes))
    return [i for i in range(n) if label[i]!=main and sizes[label[i]]<=max_fraction*n]


def weakly_attached_group_names(rec: pycolmap.Reconstruction, names, centers, min_shared=15, min_shared_fraction=.1):
    """공간적으로 떨어진 작은 카메라 묶음 중 나머지와 공유하는 3D 점이 적은 묶음의 사진 이름.

    방 끝에서 찍어 떨어져 보이지만 올바른 사진은 공유 점이 많다(procthor-test-0186:
    공유 260 / 자체 180). 잘못 놓인 욕실 묶음은 공유 37~39 / 자체 1,405였다.
    """
    candidates=[names[i] for i in isolated_camera_group_indices(centers)]
    if not candidates:
        return []
    by_name={rec.images[i].name:i for i in rec.reg_image_ids()}
    group={by_name[n] for n in candidates if n in by_name}
    shared=own=0
    for point in rec.points3D.values():
        images={el.image_id for el in point.track.elements}
        if images&group:
            if images-group: shared+=1
            else: own+=1
    if shared<min_shared or shared<min_shared_fraction*(shared+own):
        return candidates
    return []


def local_height_ratio_outlier_indices(camera_heights, observed_heights, upper=1.6, min_points=20):
    """카메라마다 '자신이 본 점 중 가장 낮은 층'까지의 높이를 다른 카메라들과 비교한다.

    배율과 무관한 검사다. 문 너머로만 약하게 이어진 작은 방 묶음이 다른 배율로 놓이면
    이 비율이 크게 벌어진다(LH 9번 재실행: 67배·2.24배, 올바른 카메라는 최대 1.09배).
    바닥을 보지 않은 사진은 비율이 작아지므로 위쪽만 판정한다.
    """
    values=[]
    for height,ys in zip(camera_heights,observed_heights):
        ys=np.asarray(ys,float)
        values.append(float(height-np.percentile(ys,3)) if len(ys)>=min_points else np.nan)
    values=np.array(values)
    valid=np.isfinite(values)&(values>0)
    if valid.sum()<5:
        return []
    median=float(np.median(values[valid]))
    return np.flatnonzero(valid&(values>upper*median)).tolist()
