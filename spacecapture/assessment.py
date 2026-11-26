"""Keep geometric evidence separate from unimplemented semantic checks."""
from __future__ import annotations


def assess_scene(scene: dict) -> dict:
    quality, room = scene["quality"], scene["room"]
    frames, registered = quality["frames"], quality["registered"]
    ratio = registered / frames if frames else 0.0
    fallback = [key for key, wall in room["walls"].items() if wall.get("fallback")]
    limited = ratio < .9 or quality["points"] < 30 or bool(fallback) or not scene.get("floor_found")
    geometric_limited = limited
    layout, graph = scene.get("layout"), scene.get("connectivity")
    limited = limited or bool(layout and not layout["room_box_supported"]) or bool(graph and graph["status"] == "disconnected")
    return {
        "status": "inspection_required" if layout and not layout["room_box_supported"] else "limited_geometry" if limited else "candidate_geometry",
        "automatic_floorplan_approved": False,
        "scope": "observed_geometry" if layout and not layout["room_box_supported"] else "single_rectangular_space_assumption",
        "checks": {
            "geometric_evidence": {
                "status": "limited" if geometric_limited else "available",
                "registered_ratio": round(ratio, 4), "reliable_points": quality["points"],
                "fallback_walls": fallback, "floor_found": scene.get("floor_found", False),
                "message": "여러 사진의 기하 대응 근거예요. 공간 용도나 구조의 정확성을 보장하지는 않아요.",
            },
            "indoor_outdoor_and_use": {"status": "not_checked", "message": "실내·실외와 주택·사무실·회의실 용도는 자동 판별하지 않았어요."},
            "layout_topology": ({"status": "partially_checked", "result": layout["status"], "message": layout["message"]}
                                if layout else {"status": "not_checked", "message": "관측 벽의 구조 검사를 실행하지 않았어요."}),
            "photo_connectivity": ({"status": graph["status"], "groups": len(graph["groups"]),
                                    "verified_pairs": graph["verified_pairs"], "message": graph["message"]}
                                   if graph else {"status": "not_checked", "message": "사진 간 연결 검사를 실행하지 않았어요."}),
            "glass_and_mirrors": {"status": "not_checked", "message": "유리벽·거울·반사면을 자동 판별하거나 검증하지 않았어요."},
            "metric_scale": {"status": "marker_based" if scene["scale"]["method"] == "marker" else "assumed",
                             "message": "마커나 알려진 실측 길이가 없으면 크기는 추정치예요."},
            "space_opening_connections": ({"status":"partially_checked",
                "message":f"관측 영역 {len(scene['property_graph']['spaces'])}개와 문/개구부 후보를 공유 3D 관측으로 연결했어요. 실제 방 용도·통행 여부는 미확정이에요."}
                if "property_graph" in scene else {"status":"not_checked","message":"공간과 문 연결 검사를 실행하지 않았어요."}),
        },
        "message": ((layout["message"] + " ") if layout else "검토용 구조 후보입니다. ") +
                   "실내·실외 분류와 유리벽·거울 인식은 미검사 상태예요.",
        **({"layout_summary": layout["status"]} if layout else {}),
    }
