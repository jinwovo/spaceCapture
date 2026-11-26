"""End-to-end: room video -> metric, gravity-aligned 3D scene for the viewer.

scenes/<name>/
    scene.json   room box, scale, cameras, quality notes (read by the viewer)
    points.ply   sparse colored points (always)
    splat.ply    photo-real Gaussian splats (after the Brush step)
    work/        frames, COLMAP database / models, Brush dataset
"""

from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import splat as splatmod
from .align import ASSUMED_CAMERA_HEIGHT, arrays_from_reconstruction, compute_alignment
from .frames import extract_frames
from .photos import prepare_photos, write_previews
from .floorplan import write_floorplan
from .assessment import assess_scene
from .layout import analyze_layout
from .connectivity import match_connectivity
from .property_graph import infer_property
from .room_types import estimate_usage
from .property_svg import write_property_svg
from .plyio import read_vertices, splat_centers, write_points
from .room import detect_room, shift_room
from .sfm import run_sfm

REPO = Path(__file__).resolve().parent.parent
SCENES = REPO / "scenes"


class Log:
    """Print and append to scenes/<name>/process.log."""

    def __init__(self, path: Path | None = None):
        self.path = path

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


def slugify(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_-]+", "-", text).strip("-") or "scene"


def _quat_xyzw(m: np.ndarray) -> list[float]:
    """Rotation matrix -> quaternion [x, y, z, w]."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    return (q / np.linalg.norm(q)).round(6).tolist()


def _r(x, nd: int = 4):
    """Round floats (recursively) for compact JSON."""
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items() if k != "corners_units"}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _warnings(n_frames: int, n_reg: int, align_info: dict, room: dict, photos: bool = False) -> list[str]:
    w = []
    if n_reg < 0.9 * n_frames:
        w.append(f"사진 {n_frames - n_reg}장은 위치를 찾지 못해 빠졌어요. 같은 공간의 겹치는 사진을 더 넣어 주세요."
                 if photos else f"프레임 {n_frames - n_reg}장은 위치를 찾지 못해 빠졌어요. 천천히 움직이고 제자리 회전을 줄여 주세요.")
    if align_info["scale"]["method"] != "marker":
        w.append("사진만으로 실제 크기를 확정할 수 없어 촬영 높이를 가정했어요. 알려진 실측 길이로 뷰어에서 보정해 주세요."
                 if photos else "마커가 보이지 않아 촬영 높이로 실제 크기를 추정했어요. 뷰어에서 알려진 실제 길이로 보정하거나 마커와 함께 다시 찍어 주세요.")
    if not align_info["floor_found"]:
        w.append("바닥을 확실히 찾지 못했어요. 바닥과 벽이 만나는 선이 화면에 보이게 찍어 주세요.")
    if room["height_source"] != "detected":
        w.append(f"천장이 충분히 보이지 않아 천장 높이는 {room['height']:.1f}m로 가정했어요.")
    if room.get("ceiling_analysis"):
        w.append(room["ceiling_analysis"]["message"])
    missing = [k for k, v in room["walls"].items() if v.get("fallback")]
    if missing:
        w.append(f"일부 벽({', '.join(missing)})을 확실히 찾지 못했어요. 그쪽 벽도 화면에 담아 주세요.")
    return w


def _capture_warnings(rec, photos: bool) -> list[str]:
    """프레임당 3D 점이 적으면 흔들림·흐림·무늬 없는 화면이 많았다는 뜻이다(촬영 안내).
    합성 걷기 영상: 깨끗함 약 127점/장, 움직임 번짐 26, 폰 효과 전부 27(2026-11-27)."""
    per_view = rec.num_points3D() / max(1, rec.num_reg_images())
    if per_view >= 50:
        return []
    return ["사진마다 잡힌 3D 특징이 적어요. 흔들려 흐리거나 벽만 가까이 찍힌 장면이 많은 것 같아요. "
            "천천히 걷고, 벽에서 1m 이상 떨어져 바닥과 벽이 만나는 선이 함께 보이게 찍어 주세요."
            if not photos else
            "사진마다 잡힌 3D 특징이 적어요. 흔들림 없이, 서로 겹치는 부분이 넓게 보이도록 찍어 주세요."]


def _room_public(room: dict) -> dict:
    keys = ("min", "max", "width", "depth", "height", "height_source", "area_m2", "walls", "ceiling")
    return _r({k: room[k] for k in keys}|{k:room[k] for k in ("height_method","ceiling_analysis") if k in room})


def process(
    video: Path,
    name: str | None = None,
    scenes_root: Path = SCENES,
    marker_mm: float | None = 150.0,
    marker_dict: str = "DICT_4X4_50",
    target_fps: float = 5.0,
    max_frames: int = 800,
    camera_height: float = ASSUMED_CAMERA_HEIGHT,
    focal_px: float | None = None,
    splat: bool = True,
    steps: int = 15000,
    max_resolution: int = 1280,
    max_splats: int = 500000,
    photos: bool = False,
    extra_photos: Path | None = None,
) -> Path:
    """video: 영상(또는 photos=True면 사진 폴더). extra_photos: 영상과 함께 쓸 사진 폴더(영상+사진 혼합)."""
    video = Path(video)
    name = slugify(name or (video.name if photos else video.stem))
    scene_dir = scenes_root / name
    work = scene_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    (scene_dir / "process.log").unlink(missing_ok=True)
    log = Log(scene_dir / "process.log")
    timing: dict[str, float] = {}
    t_all = time.time()
    log(f"장면 '{name}' 처리 시작: {video}")

    t = time.time()
    if photos:
        log("[1/5] 사진 확인·중복 제거·크기 정리")
        finfo = prepare_photos(video, work / "frames", max_photos=max_frames, log=log)
    else:
        log("[1/5] 영상에서 선명한 프레임 고르기")
        finfo = extract_frames(video, work / "frames", target_fps=target_fps, max_frames=max_frames, log=log)
        if extra_photos:
            # 영상 + 사진: 영상으로 집을 훑고, 덜 찍힌 곳을 사진으로 보충한다. 한 좌표계로 함께 복원한다.
            log("  함께 올린 사진 확인·중복 제거·크기 정리")
            staging = work / "photos_in"
            pinfo = prepare_photos(Path(extra_photos), staging, max_photos=150, log=log)
            for n in pinfo["names"]:
                shutil.copyfile(staging / n, work / "frames" / n)
            finfo["photos"] = pinfo["photos"]; finfo["photo_input_count"] = pinfo["input_count"]
            finfo["photo_skipped"] = pinfo["skipped"]; finfo["selected"] += pinfo["selected"]
            log(f"  영상 프레임 {len(finfo['names'])}장 + 사진 {pinfo['selected']}장을 함께 복원해요.")
    timing["frames"] = time.time() - t

    t = time.time()
    log("[2/5] 카메라 위치 복원 (SfM, CPU)")
    rec = run_sfm(work / "frames", work / "colmap", focal_px=focal_px, log=log, unordered=photos)
    timing["sfm"] = time.time() - t

    t = time.time()
    log("[3/5] 바닥·방향·실제 크기 맞추기")
    arrays = arrays_from_reconstruction(rec)
    al = compute_alignment(arrays, rec, work / "frames", marker_size=(marker_mm / 1000.0) if marker_mm else None,
                           marker_dict=marker_dict, camera_height=camera_height, log=log)
    if True:  # 영상도 적용: 벽만 보이는 구간이 통째로 다른 높이·위치에 붙은 사례(2~5.4m, 2026-11-26)
        # 제외 후 다시 정렬하면 바닥 평면이 바뀌어 다른 잘못 놓인 묶음이 드러날 수 있다(LH 학습 자료 lh-0035).
        # 변화가 없을 때까지 최대 3회 반복한다.
        for _round in range(3):
            from .sfm import (camera_height_outlier_indices, deregister_photos, local_height_ratio_outlier_indices,
                              weakly_attached_group_names)
            aligned_centers = al.apply(arrays.centers)
            heights = aligned_centers[:, 1]
            point_heights = al.apply(arrays.xyz)[:, 1]
            seen_by = [[] for _ in arrays.names]
            for y, observers in zip(point_heights, arrays.observations):
                for j in observers:
                    seen_by[j].append(y)
            median = float(np.median(heights))
            dropped = [{"name": arrays.names[i], "camera_height_m": round(float(heights[i]), 3),
                        "median_camera_height_m": round(median, 3),
                        "reason": "정렬 후 촬영 높이가 다른 사진과 0.6m 넘게 달라 잘못 놓인 사진으로 판단"}
                       for i in camera_height_outlier_indices(heights)]
            seen = {d["name"] for d in dropped}
            dropped += [{"name": arrays.names[i], "reason": "본 점 기준 촬영 높이가 다른 사진의 1.6배를 넘어 다른 배율로 놓인 사진으로 판단"}
                        for i in local_height_ratio_outlier_indices(heights, seen_by) if arrays.names[i] not in seen]
            seen = {d["name"] for d in dropped}
            dropped += [{"name": n, "reason": "다른 사진과 3m 넘게 떨어진 작은 묶음이며 공유 3D 점이 적어 상대 위치를 확정할 수 없음"}
                        for n in weakly_attached_group_names(rec, arrays.names, aligned_centers) if n not in seen]
            if not dropped:
                break
            if dropped:
                deregister_photos(rec, [d["name"] for d in dropped], log)
                log(f"  위치를 확정할 수 없는 사진 {len(dropped)}장(촬영 높이·배율·고립 묶음) 제외 후 재조정·재정렬")
                pose_filter = work / "colmap/pose_filter.json"
                record = json.loads(pose_filter.read_text(encoding="utf-8")) if pose_filter.exists() else {"excluded": []}
                record["excluded"] += dropped
                record["method"] = record.get("method", "") + "+camera_height+local_height_ratio+isolated_group"
                pose_filter.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                sparse = work / "colmap/sparse"
                shutil.rmtree(sparse, ignore_errors=True); sparse.mkdir(parents=True); rec.write(sparse)
                arrays = arrays_from_reconstruction(rec)
                al = compute_alignment(arrays, rec, work / "frames", marker_size=(marker_mm / 1000.0) if marker_mm else None,
                                       marker_dict=marker_dict, camera_height=camera_height, log=log)

    log("[4/5] 방 치수 계산")
    keep = arrays.reliable(min_angle=6.0)
    if photos and keep.sum() < 30:
        raise RuntimeError("치수 계산에 쓸 신뢰 점이 부족해요. 같은 벽·가구가 겹치게 보이는 사진을 더 넣어 주세요.")
    P, C = al.apply(arrays.xyz[keep]), al.apply(arrays.centers)
    room = detect_room(P, C)
    cx = (room["min"][0] + room["max"][0]) / 2
    cz = (room["min"][2] + room["max"][2]) / 2
    al = al.shifted(-cx, -cz)  # origin at the middle of the floor
    room = shift_room(room, -cx, -cz)

    rec.transform(al.sim3d())
    aligned_dir = work / "colmap" / "aligned"
    shutil.rmtree(aligned_dir, ignore_errors=True)
    aligned_dir.mkdir(parents=True)
    rec.write(aligned_dir)
    aligned = arrays_from_reconstruction(rec)
    drift = float(np.abs(aligned.centers - al.apply(arrays.centers)).max())
    if drift > 1e-3:
        raise RuntimeError(f"좌표 변환 검증 실패 (오차 {drift:.4f}m)")
    write_points(scene_dir / "points.ply", aligned.xyz[keep], aligned.rgb[keep])
    timing["align"] = time.time() - t

    registered_cameras=[rec.cameras[rec.images[i].camera_id] for i in rec.reg_image_ids()]
    focal_ratio=float(np.median([c.params[0]/max(c.width,c.height) for c in registered_cameras]))
    cam=min(registered_cameras,key=lambda c:abs(c.params[0]/max(c.width,c.height)-focal_ratio))
    video_frame = {} if photos else dict(zip(finfo["names"], finfo["frame_index"]))
    originals = {p["name"]: p["original"] for p in finfo.get("photos", [])}
    scene = {
        "format": "spacecapture-scene",
        "version": 1,
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "units": "m",
        "up": "+y",
        "files": {"points": "points.ply", "splat": None, "floorplan": "floorplan.svg"},
        "room": _room_public(room),
        "room_source": "sparse",
        "scale": _r(al.info["scale"]),
        "floor_found": al.info["floor_found"],
        "cameras": [{"name": n, "video_frame": video_frame.get(n), "source_photo": originals.get(n), "position": _r(c.tolist()),
                     "quaternion": _quat_xyzw(R.T @ np.diag([1.0, -1.0, -1.0]))}
                    for n, c, R in zip(aligned.names, aligned.centers, aligned.R_cw)],
        "intrinsics": {"width": cam.width, "height": cam.height, "focal": round(float(cam.params[0]), 2)},
        "quality": {
            "frames": finfo["selected"],
            "registered": rec.num_reg_images(),
            "points": int(keep.sum()),
            "reprojection_error_px": round(float(np.mean(arrays.error)), 3),
            "warnings": _warnings(finfo["selected"], rec.num_reg_images(), al.info, room, photos=photos) + _capture_warnings(rec, photos),
            "points_per_view": round(rec.num_points3D() / max(1, rec.num_reg_images()), 1),
        },
        "source": ({"kind": "photos", "folder": video.name, "input_count": finfo["input_count"],
                    "skipped": finfo["skipped"], "photos": finfo["photos"]} if photos else
                   {"kind": "video+photos", "video": video.name, "duration_s": round(finfo["duration"], 2),
                    "video_size": finfo["video_size"], "folder": Path(extra_photos).name,
                    "input_count": finfo["photo_input_count"], "skipped": finfo["photo_skipped"], "photos": finfo["photos"]}
                   if extra_photos else
                   {"kind": "video", "video": video.name, "duration_s": round(finfo["duration"], 2),
                    "video_size": finfo["video_size"]}),
        "timing_s": _r(timing, 1),
    }
    scene["layout"] = analyze_layout(aligned.xyz[keep], aligned.centers, scene["room"])
    scene["property_graph"] = infer_property(aligned.xyz[keep], aligned.centers, aligned.names,
        [o for o,k in zip(aligned.observations,keep) if k], planes=scene["layout"]["planes"],
        rotations=aligned.R_cw,intrinsics=aligned.intrinsics)
    scene["objects"] = _detect_objects(work / "frames", aligned, keep, scene, log)
    scene["property_graph"]["usage"] = estimate_usage(scene["property_graph"])
    scene["connectivity"] = json.loads((work / "colmap" / "match_graph.json").read_text(encoding="utf-8"))
    for group in scene["connectivity"]["groups"]:
        group["source_photos"] = [originals.get(n, n) for n in group["images"]]
    if photos or extra_photos:
        scene["quality"]["unregistered_photos"] = [p["original"] for p in finfo["photos"] if p["name"] not in aligned.names]
        pose_filter=work/"colmap/pose_filter.json"
        if pose_filter.exists():
            excluded=json.loads(pose_filter.read_text(encoding="utf-8"))["excluded"]
            scene["quality"]["pose_rejections"]=[entry | {"source_photo":originals.get(entry["name"])} for entry in excluded]
        scene["quality"]["scope"] = "observed_property_candidates" if len(scene["property_graph"]["spaces"])>1 else "single_observed_space"
        scene["quality"]["warnings"].append("관측된 공간과 연결의 후보예요. 사진 연결 묶음이 실제 방 개수나 방 용도를 의미하지는 않아요.")
        scene["source"]["camera_parameters"] = "per_image" if photos else "video_single+photo_per_image"
    write_previews(scene_dir,work/"frames",scene["cameras"])
    write_floorplan(scene_dir, scene)
    _write_scene(scene_dir, scene)
    label="직사각형 구조 후보" if scene["layout"]["room_box_supported"] else "관측 경계 범위 (방 치수 미확정)"
    area_note=f"  직사각형 가정 면적 {room['area_m2']:.1f}㎡" if scene["layout"]["room_box_supported"] else "  바닥 면적 확정 보류"
    log(f"  {label}: 가로 {room['width']:.2f}m × 세로 {room['depth']:.2f}m × 높이 {room['height']:.2f}m"
        f"{'' if room['height_source'] == 'detected' else '(가정)'}"
        +area_note)
    for w in scene["quality"]["warnings"]:
        log(f"  ⚠ {w}")

    if splat:
        log("[5/5] 실사 3D 학습 (가우시안 스플래팅, Brush)")
        try:
            run_splat(scene_dir, steps=steps, max_resolution=max_resolution, max_splats=max_splats, log=log)
        except RuntimeError as e:
            log(f"  실사 3D 단계를 건너뜀: {e}")
            log(f"  나중에 다시 실행: python -m spacecapture splat {name}")
    else:
        log("[5/5] 실사 3D 학습 건너뜀 (--no-splat). 나중에: python -m spacecapture splat " + name)
    log(f"완료 ({(time.time() - t_all) / 60:.1f}분). 뷰어: cd viewer && npm run dev → 장면 '{name}' 선택")
    return scene_dir


def _write_scene(scene_dir: Path, scene: dict) -> None:
    if "layout" in scene:
        room = scene["room"]
        supported = scene["layout"]["room_box_supported"]
        room["dimensions_kind"] = "rectangular_candidate" if supported else "bounding_extent"
        if not supported:
            room["bounding_area_m2"] = round(room["width"] * room["depth"], 4)
            room["area_m2"] = None
        elif room.get("area_m2") is None:
            room["area_m2"] = round(room["width"] * room["depth"], 4)
    scene["assessment"] = assess_scene(scene)
    scene["files"]["assessment"] = "assessment.json"
    (scene_dir / "assessment.json").write_text(json.dumps(scene["assessment"], ensure_ascii=False, indent=1), encoding="utf-8")
    if "layout" in scene:
        scene["files"]["layout"] = "layout.json"
        (scene_dir / "layout.json").write_text(json.dumps(scene["layout"], ensure_ascii=False, indent=1), encoding="utf-8")
    if "connectivity" in scene:
        scene["files"]["connectivity"] = "connectivity.json"
        (scene_dir / "connectivity.json").write_text(json.dumps(scene["connectivity"], ensure_ascii=False, indent=1), encoding="utf-8")
    if "property_graph" in scene:
        scene["files"]["property_graph"] = "property_graph.json"
        scene["files"]["property_plan"] = "property.svg"
        (scene_dir/"property_graph.json").write_text(json.dumps(scene["property_graph"],ensure_ascii=False,indent=1),encoding="utf-8")
        write_property_svg(scene_dir,scene["property_graph"],scene.get("source",{}).get("title",scene["name"])+" · 공간 연결 후보")
    try:  # 공유 벽선 도면 모델: 2D 도면과 뷰어 구조 모형이 함께 쓴다
        from .plan_model import build as build_floorplan
        scene["floorplan"] = build_floorplan(scene)
        if scene["floorplan"] is None: scene.pop("floorplan")
    except Exception as error:
        scene.pop("floorplan", None)
        scene.setdefault("quality", {}).setdefault("notes", []).append(f"도면 모델 실패: {str(error)[:120]}")
    try:  # 국내 매물 평면도 양식의 도면 (실패해도 분석 결과 저장은 계속)
        from .plan_render import render
        summary = render(scene, scene_dir, title=scene.get("source", {}).get("title") or scene["name"])
        if summary:
            scene["files"]["plan_svg"], scene["files"]["plan_png"] = "plan.svg", "plan.png"
            scene["plan"] = summary
    except Exception as error:
        scene["plan"] = {"status": "failed", "message": "평면도 그림을 만들지 못했어요.", "error": str(error)[:200]}
    (scene_dir / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1), encoding="utf-8")


def load_scene(scene_dir: Path) -> dict:
    return json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))


def inspect_scene(scene_dir: Path) -> dict:
    """Recompute evidence for an existing scene without changing metric geometry."""
    scene = load_scene(scene_dir)
    vertices = read_vertices(scene_dir / "points.ply")
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], 1)
    from .ceiling_model import active_model
    if active_model():
        from .room import ceiling_prediction,DEFAULT_CEILING
        ceiling_points=points
        if scene.get("files",{}).get("splat"):
            ceiling_points=np.vstack([points,splat_centers(scene_dir/scene["files"]["splat"])])
        room=scene["room"]
        scene.setdefault("ceiling_before_model",{k:room.get(k) for k in ("height","height_source","ceiling")})
        box=(room["min"][0],room["max"][0],room["min"][2],room["max"][2])
        ceiling,analysis=ceiling_prediction(ceiling_points,max(c["position"][1] for c in scene["cameras"]),box)
        room.update(height=ceiling["pos"] if ceiling else DEFAULT_CEILING,
                    height_source="detected" if ceiling else "assumed",
                    height_method="geometry_mlp" if ceiling else "assumed",ceiling=ceiling,ceiling_analysis=analysis)
        room["max"][1]=room["height"]
        q=scene["quality"]
        q["warnings"]=_warnings(q["frames"],q["registered"],scene,room,photos=scene.get("source",{}).get("kind")=="photos")
        if q.get("points_per_view") is not None and q["points_per_view"] < 50:  # 처리 때 남긴 촬영 안내 유지
            class _Rec:  # _capture_warnings에 필요한 값만
                num_points3D=lambda self: q["points_per_view"]; num_reg_images=lambda self: 1
            q["warnings"]+=_capture_warnings(_Rec(),scene.get("source",{}).get("kind")=="photos")
        if scene.get("source",{}).get("kind")=="photos":
            q["warnings"].append("관측된 공간과 연결의 후보예요. 실제 방 개수·용도는 미확정이에요.")
    scene["layout"] = analyze_layout(points, [c["position"] for c in scene["cameras"]], scene["room"])
    aligned_dir=scene_dir/"work/colmap/aligned"
    if (aligned_dir/"cameras.bin").exists():
        import pycolmap
        aligned=arrays_from_reconstruction(pycolmap.Reconstruction(aligned_dir))
        keep=aligned.reliable(min_angle=6.)
        scene["property_graph"]=infer_property(aligned.xyz[keep],aligned.centers,aligned.names,
            [o for o,k in zip(aligned.observations,keep) if k], planes=scene["layout"]["planes"],
            rotations=aligned.R_cw,intrinsics=aligned.intrinsics)
        instances=(scene.get("objects") or {}).get("instances") or []
        if instances:  # 재계산한 공간에 기존 사물 검출 결과를 다시 붙인다(사진 재검출 없음)
            from .objects import usage_hints
            hints=usage_hints(scene["property_graph"]["spaces"],instances)
            for space in scene["property_graph"]["spaces"]:
                space["usage_candidates"]=hints.get(space["id"],{}).get("usage_candidates",[])
                space["objects"]=hints.get(space["id"],{}).get("objects",[])
        scene["property_graph"]["usage"]=estimate_usage(scene["property_graph"])
        focal_ratio=float(np.median([c[2]/max(c[:2]) for c in aligned.intrinsics]))
        iw,ih,f=min(aligned.intrinsics,key=lambda c:abs(c[2]/max(c[:2])-focal_ratio))
        scene["intrinsics"]={"width":int(iw),"height":int(ih),"focal":round(f,2)}
    if scene.get("source",{}).get("kind") in ("photos","video+photos"):
        scene["quality"]["scope"]="observed_property_candidates" if len(scene.get("property_graph",{}).get("spaces",[]))>1 else "single_observed_space"
        pose_filter=scene_dir/"work/colmap/pose_filter.json"
        if pose_filter.exists():
            originals={p["name"]:p["original"] for p in scene.get("source",{}).get("photos",[])}
            excluded=json.loads(pose_filter.read_text(encoding="utf-8"))["excluded"]
            scene["quality"]["pose_rejections"]=[entry | {"source_photo":originals.get(entry["name"])} for entry in excluded]
    database = scene_dir / "work/colmap/database.db"
    if database.exists():
        scene["connectivity"] = match_connectivity(database)
        originals = {p["name"]: p["original"] for p in scene.get("source", {}).get("photos", [])}
        for group in scene["connectivity"]["groups"]:
            group["source_photos"] = [originals.get(n, n) for n in group["images"]]
    scene["files"]["floorplan"] = "floorplan.svg"
    write_previews(scene_dir,scene_dir/"work/frames",scene["cameras"])
    write_floorplan(scene_dir, scene)
    _write_scene(scene_dir, scene)
    return scene


def run_splat(scene_dir: Path, steps: int = 15000, log=print,
              max_resolution: int = 1280, max_splats: int = 500000) -> Path:
    work = scene_dir / "work"
    t = time.time()
    dataset = splatmod.prepare_dataset(work / "colmap" / "aligned", work / "frames", work / "dataset")
    ply = splatmod.train(dataset, work / "brush", steps=steps, log=log,
                         max_resolution=max_resolution, max_splats=max_splats)
    out = scene_dir / "splat.ply"
    from .splat_clean import clean_splat
    cleaning = clean_splat(ply, out, load_scene(scene_dir))  # 집 밖·허공 잡티 정리(원본은 work/brush)
    removed = cleaning["input"] - cleaning["kept"]
    log(f"  3D 잡티 정리: {cleaning['input']:,}개 중 {removed:,}개 제거 (집 밖·바닥 아래·천장 위·거의 투명·과대)")
    scene = refine_splat_room(scene_dir, log=log)
    scene["splat_cleaning"] = cleaning
    scene["splat_training"] = json.loads((work / "brush" / "training.json").read_text(encoding="utf-8"))
    scene.setdefault("timing_s", {})["splat"] = round(time.time() - t, 1)
    _write_scene(scene_dir, scene)
    return out


def refine_splat_room(scene_dir: Path, log=print) -> dict:
    """Recheck learned geometry without requiring another GPU training run."""
    # denser geometry from the splats -> refine the room box (the frame itself stays fixed)
    scene = load_scene(scene_dir)
    out = scene_dir / "splat.ply"
    v = read_vertices(scene_dir / "points.ply")
    sparse = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    dense = splat_centers(out)
    cams = np.asarray([c["position"] for c in scene["cameras"]])
    refined = detect_room(np.vstack([sparse, dense]), cams)
    # Always compare with the SfM result: repeating splat must not accumulate drift.
    old = scene.setdefault("room_sparse", scene["room"])
    change = max(abs(refined["width"] / old["width"] - 1), abs(refined["depth"] / old["depth"] - 1))
    if change < 0.15:
        scene["room"] = _room_public(refined)
        scene["room_source"] = "sparse+splat"
        log(f"  스플랫으로 방 치수 보정: {refined['width']:.2f} × {refined['depth']:.2f} × {refined['height']:.2f}m")
    else:
        scene["room"] = old
        scene["room_source"] = "sparse"
        log(f"  스플랫 기반 치수가 {change:.0%} 달라 기존 값을 유지했어요")
    scene["files"]["splat"] = out.name
    scene["room_refinement"] = _r({
        "sparse_points": len(sparse), "reliable_splat_centers": len(dense),
        "candidate": _room_public(refined), "wall_change_fraction": change,
        "accepted": change < 0.15,
    })
    if scene["room"]["height_source"] != "detected":
        log(f"  천장 검출 근거 부족: 유효 스플랫 {len(dense):,}개에서도 넓게 퍼진 천장층을 찾지 못해 "
            f"{scene['room']['height']:.2f}m 가정값을 유지했어요")
    q = scene["quality"]
    q["warnings"] = _warnings(q["frames"], q["registered"], scene, scene["room"],
                              photos=scene.get("source", {}).get("kind") == "photos")
    if scene.get("source", {}).get("kind") == "photos":
        q["warnings"].append("관측된 공간과 연결의 후보예요. 사진 연결 묶음이 실제 방 개수나 방 용도를 의미하지는 않아요.")
    scene["files"]["floorplan"] = "floorplan.svg"
    # Geometry checks use triangulated points, not low-opacity visual splats.
    scene["layout"] = analyze_layout(sparse, cams, scene["room"])
    write_floorplan(scene_dir, scene)
    _write_scene(scene_dir, scene)
    return scene


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimizing |dst - (s R src + t)|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (a**2).sum(1).mean())
    return s, R, mu_d - s * R @ mu_s


def evaluate(scene_dir: Path, gt_path: Path) -> dict:
    """Compare a processed synthetic scene with its ground truth."""
    scene = load_scene(scene_dir)
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    r, g = scene["room"], gt["room"]
    got = sorted([r["width"], r["depth"]], reverse=True) + [r["height"]]
    want = sorted([g["width"], g["depth"]], reverse=True) + [g["height"]]
    rows = []
    for label, a, b in zip(("긴 변", "짧은 변", "높이"), got, want):
        rows.append({"what": label, "measured_m": round(a, 3), "truth_m": b, "error_cm": round((a - b) * 100, 1)})
    out = {"scale_method": scene["scale"]["method"], "rows": rows,
           "max_abs_error_cm": max(abs(x["error_cm"]) for x in rows)}

    # camera trajectory: rigid+scale fit of true camera centers onto the reconstruction
    pairs = [(gt["camera_centers"][c["video_frame"]], c["position"]) for c in scene["cameras"]
             if c.get("video_frame") is not None and c["video_frame"] < len(gt.get("camera_centers", []))]
    if len(pairs) >= 3:
        src = np.asarray([p[0] for p in pairs])
        dst = np.asarray([p[1] for p in pairs])
        s, R, t = umeyama(src, dst)
        err = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        out["trajectory"] = {"cameras": len(pairs), "rmse_cm": round(float(np.sqrt((err**2).mean())) * 100, 2),
                             "scale_vs_truth": round(s, 4)}
    return out

def _detect_objects(frames_dir, aligned, keep, scene, log=print):
    """사진별 실내 사물 검출 → SfM 점으로 3D 위치 → 공간 용도 후보(확정 아님)."""
    from .objects import MODEL, analyze_objects, usage_hints
    if not MODEL.exists():
        return {"status": "model_unavailable", "message": "사물 검출 모델이 없어 사물 인식을 건너뛰었어요."}
    try:
        started = time.time()
        instances, per_photo = analyze_objects(frames_dir, aligned.names, aligned.centers, aligned.R_cw, aligned.intrinsics,
                                               aligned.xyz[keep], [o for o, k in zip(aligned.observations, keep) if k], log=log)
    except (OSError, ValueError, cv2_error()) as error:
        return {"status": "model_unavailable", "message": "사물 검출 모델을 확인할 수 없어 사물 인식을 건너뛰었어요.", "error": str(error)}
    graph = scene.get("property_graph", {})
    hints = usage_hints(graph.get("spaces", []), instances) if graph.get("spaces") else {}
    for space in graph.get("spaces", []):
        space["usage_candidates"] = hints.get(space["id"], {}).get("usage_candidates", [])
        space["objects"] = hints.get(space["id"], {}).get("objects", [])
    return {"status": "detected", "model": "yolox-s-coco (Apache-2.0)", "instances": instances,
            "photos_with_detections": sum(1 for v in per_photo.values() if v),
            "detections": {k: [{"label": d["label"], "score": d["score"], "box": d["box"]} for d in v] for k, v in per_photo.items() if v},
            "seconds": round(time.time() - started, 1), "min_views": 2,
            "message": "여러 사진에서 같은 위치로 확인된 사물만 기록했어요. 공간 용도는 사물 근거의 후보이며 확정이 아니에요."}


def cv2_error():
    import cv2
    return cv2.error
