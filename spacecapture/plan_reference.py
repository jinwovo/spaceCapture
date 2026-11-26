"""Register a separately annotated plan for evaluation, without changing inference."""
from __future__ import annotations

import copy
import html
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


def reference_status(truth):
    if truth.get("origin") == "external_structure_local_render":
        return {"kind": "external_synthetic_structure", "ground_truth_eligible": True,
                "source_dataset": truth.get("source_dataset"), "source": truth.get("source"),
                "real_property_validation": False,
                "message": "외부 합성 구조를 자체 렌더링한 정답 비교입니다. 실제 매물 사진의 정확도로 집계하지 않습니다."}
    if truth.get("origin") == "procedurally_generated_here":
        return {"kind": "synthetic_ground_truth", "ground_truth_eligible": True,
                "message": "직접 생성한 사진과 도면의 합성 정답 비교입니다. 실제 매물 검증은 아닙니다."}
    reference = truth.get("reference", {})
    level = reference.get("match_level", "unverified")
    if level not in {"same_unit", "same_type", "unverified"}:
        raise ValueError("도면 대응 수준은 same_unit, same_type, unverified 중 하나여야 해요.")
    checks = reference.get("pairing_checks", {})
    required = ("building", "unit_type", "mirror", "balcony", "renovation")
    checked = all(checks.get(key) is True for key in required)
    sources = all(urlparse(reference.get(key, "")).scheme in {"http", "https"}
                  for key in ("source_url", "photo_source_url"))
    eligible = level == "same_unit" and checked and sources
    return {"kind": level, "ground_truth_eligible": eligible,
            "source_url": reference.get("source_url"),
            "photo_source_url": reference.get("photo_source_url"),
            "pairing_checks": checks,
            "message": "같은 실제 세대와 구조 조건을 확인한 자료입니다. 확인 내용은 제공된 기록을 기준으로 합니다."
            if eligible else "주택형 참고 도면 또는 대응 미확인 자료입니다. 실제 세대의 정답 정확도로 집계하지 않습니다."}


def _points(values, name, minimum=3):
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum or not np.isfinite(points).all():
        raise ValueError(f"{name}: 유한한 X/Z 좌표가 {minimum}개 이상 필요해요.")
    return points


def validate_plan(truth):
    spaces = truth.get("spaces", [])
    ids = [space["id"] for space in spaces]
    if not spaces or len(set(ids)) != len(ids):
        raise ValueError("도면의 공간이 비어 있거나 공간 ID가 중복돼요.")
    for space in spaces:
        polygon = _points(space["polygon_xz"], "공간 다각형")
        area = abs(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                          - polygon[:, 1]*np.roll(polygon[:, 0], -1)))/2
        if area < .01:
            raise ValueError("도면 공간 다각형의 면적이 너무 작아요. 좌표를 미터로 지정해 주세요.")
    portals = truth.get("portals", [])
    if len({p["id"] for p in portals}) != len(portals):
        raise ValueError("도면의 문 ID가 중복돼요.")
    for portal in portals:
        _points([portal["center_xz"]], "문 중심", minimum=1)
        if len(set(portal["spaces"])) != 2 or not set(portal["spaces"]) <= set(ids):
            raise ValueError("도면의 문이 연결하는 양쪽 공간 ID를 확인해 주세요.")
        if not np.isfinite(portal["width_m"]) or portal["width_m"] <= 0:
            raise ValueError("도면의 문 폭은 양의 미터 값이어야 해요.")


def align_to_plan(prediction, anchors):
    """Fit a horizontal proper similarity; holdout points never enter the fit."""
    if anchors.get("units") != "metres":
        raise ValueError("도면 대응점은 units=metres로 미터 단위를 명시해 주세요.")
    records = anchors.get("points", [])
    if len({point["id"] for point in records}) != len(records):
        raise ValueError("도면 대응점 ID가 중복돼요.")
    if any(point.get("role", "alignment") not in {"alignment", "holdout"} for point in records):
        raise ValueError("대응점 역할은 alignment 또는 holdout이어야 해요.")
    fitting = [point for point in records if point.get("role", "alignment") == "alignment"]
    measured = _points([point["scene_xz"] for point in fitting], "복원 대응점")
    target = _points([point["plan_xz"] for point in fitting], "도면 대응점")
    centered, goal = measured-measured.mean(0), target-target.mean(0)
    for data in (centered, goal):
        singular = np.linalg.svd(data, compute_uv=False)
        if singular[1] < max(1e-4, singular[0]*.01):
            raise ValueError("대응점이 한 직선에 몰려 있어요. 떨어진 벽 모서리 3개 이상을 지정해 주세요.")
    u, _, vt = np.linalg.svd(centered.T@goal)
    signs = np.diag([1., np.linalg.det(u@vt)])
    rotation = u@signs@vt  # Row-vector form; reflections are never silently fitted.
    fit_scale = anchors.get("fit_scale", False)
    if not isinstance(fit_scale, bool):
        raise ValueError("fit_scale은 true 또는 false로 지정해 주세요.")
    scale = float(np.sum((centered@rotation)*goal)/np.sum(centered**2)) if fit_scale else 1.
    if scale <= 1e-6 or not np.isfinite(scale):
        raise ValueError("도면 배율을 정렬할 수 없어요. 좌우 대칭과 대응점을 확인해 주세요.")
    translation = target.mean(0)-scale*measured.mean(0)@rotation
    def transform(values):
        return (scale*np.asarray(values, float)@rotation+translation).tolist()
    errors = np.linalg.norm(np.array(transform(measured))-target, axis=1)
    held = [point for point in records if point.get("role") == "holdout"]
    holdout_errors = []
    if held:
        a = _points([point["scene_xz"] for point in held], "별도 검증 복원점", minimum=1)
        b = _points([point["plan_xz"] for point in held], "별도 검증 도면점", minimum=1)
        holdout_errors = np.linalg.norm(np.array(transform(a))-b, axis=1).tolist()
    result = copy.deepcopy(prediction)
    for space in result["spaces"]:
        space["polygon_xz"] = transform(space["polygon_xz"])
        if "center_xz" in space:
            space["center_xz"] = transform([space["center_xz"]])[0]
    for portal in result["portals"]:
        portal["center_xz"] = transform([portal["center_xz"]])[0]
        portal["width_m"] *= scale
    details = {"method": "evaluation_only_plan_anchors", "fit_scale": fit_scale,
               "scale_scene_to_plan": round(scale, 6), "alignment_points": len(fitting),
               "anchor_rmse_cm": round(float(np.sqrt(np.mean(errors**2)))*100, 3),
               "holdout_points": len(held),
               "holdout_mean_error_cm": round(float(np.mean(holdout_errors))*100, 3) if held else None,
               "rotation_xz": rotation.tolist(), "translation_xz": translation.tolist()}
    details["message"] = ("도면 배율까지 맞춘 형상 비교입니다. 원래 실측 크기 정확도로 해석하지 않습니다."
                          if fit_scale else "원래 복원 배율을 유지하고 수평 회전·이동만 정렬했습니다.")
    return result, details


def comparison_svg(prediction, truth, metrics):
    vertices = np.array([p for graph in (prediction, truth) for space in graph["spaces"] for p in space["polygon_xz"]])
    low, high = vertices.min(0), vertices.max(0)
    factor = 650/max(float(np.max(high-low)), .1)
    def xy(point):
        return ((np.asarray(point)-low)*factor+np.array([40., 130.])).tolist()
    shapes = []
    for graph, color, dash, name in ((truth, "#16a34a", "", "도면"),
                                    (prediction, "#2563eb", 'stroke-dasharray="7 4"', "사진 복원")):
        for space in graph["spaces"]:
            coords = " ".join(f"{x:.2f},{y:.2f}" for x,y in map(xy, space["polygon_xz"]))
            shapes.append(f'<polygon points="{coords}" fill="{color}" fill-opacity=".09" stroke="{color}" stroke-width="2" {dash}><title>{name}: {html.escape(str(space["id"]))}</title></polygon>')
        for portal in graph.get("portals", []):
            if graph is prediction and portal.get("status") != "image_supported":
                continue
            x,y = xy(portal["center_xz"])
            shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>')
    height = float((high[1]-low[1])*factor+170)
    scope = metrics["reference"]["message"]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 {height:.1f}">'
            '<rect width="100%" height="100%" fill="white"/>'
            '<g font-family="sans-serif" font-size="12">'
            '<text x="30" y="27" font-size="18">도면과 사진 복원 비교 · 초록: 도면 / 파랑: 복원</text>'
            f'<text x="30" y="50">{html.escape(scope[:55])}</text>'
            f'<text x="30" y="68">{html.escape(scope[55:])}</text>'
            f'<text x="30" y="88">영역 IoU {metrics["mean_space_iou"]:.4f} · 연결 정밀도 {metrics["portal_precision"]:.3f} / 재현율 {metrics["portal_recall"]:.3f}</text>'
            f'<text x="30" y="108">{"배율 보정 후 형상 비교 · 원래 실측 정확도 아님" if metrics["metric_scope"]=="scale_normalized_shape" else "원래 복원 배율 유지 · 회전/이동 정렬"}</text>'
            +"".join(shapes)+"</g></svg>")


def write_comparison(output, prediction, truth, metrics):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"comparison.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output/"comparison.svg").write_text(comparison_svg(prediction, truth, metrics), encoding="utf-8")
