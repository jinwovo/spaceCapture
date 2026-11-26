"""Export the measured rectangular room as a standalone, explicitly scoped SVG."""
from __future__ import annotations

from html import escape
from pathlib import Path


def _write_observed_planes(scene_dir: Path, scene: dict) -> None:
    layout, room = scene["layout"], scene["room"]
    vertices = [p[k] for p in layout["planes"] for k in ("start", "end")]
    xs = [v[0] for v in vertices] + [room["min"][0], room["max"][0]]
    zs = [v[1] for v in vertices] + [room["min"][2], room["max"][2]]
    x0, z0, w, d = min(xs), min(zs), max(xs) - min(xs), max(zs) - min(zs)
    scale = min(700 / max(w, .1), 550 / max(d, .1))
    ox, oy = (1000 - w * scale) / 2, 200 + (550 - d * scale) / 2
    lines = []
    for i, plane in enumerate(layout["planes"]):
        a, b = plane["start"], plane["end"]
        color = "#d17b21" if i in layout["complex_plane_indices"] else "#245a86"
        lines.append(f'<line x1="{ox + (a[0]-x0)*scale:.2f}" y1="{oy + (a[1]-z0)*scale:.2f}" '
                     f'x2="{ox + (b[0]-x0)*scale:.2f}" y2="{oy + (b[1]-z0)*scale:.2f}" stroke="{color}"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="940" viewBox="0 0 1000 940">
<rect width="1000" height="940" fill="white"/>
<g font-family="Malgun Gothic,sans-serif" text-anchor="middle" fill="#18334d">
<text x="500" y="48" font-size="26">{escape(scene['name'])} · 관측 벽 후보</text>
<text x="500" y="90" font-size="19">직사각형 도면·바닥 면적 확정 보류</text>
<text x="500" y="130" font-size="16">{escape(layout['message'])}</text>
<g stroke-width="5" stroke-linecap="round">{''.join(lines)}</g>
<text x="500" y="815" font-size="17">선은 복원 점에서 찾은 수직면 후보이며 실제 벽을 확정하지 않습니다.</text>
<text x="500" y="854" font-size="16">선 사이를 임의로 연결하지 않았어요. 유리·거울·가구를 구분하는 검사가 필요해요.</text>
<text x="500" y="895" font-size="16">실제 크기: {'마커 기준' if scene['scale']['method']=='marker' else '촬영 높이 가정 · 실측 보정 필요'}</text>
</g></svg>'''
    (scene_dir / "floorplan.svg").write_text(svg, encoding="utf-8")


def write_floorplan(scene_dir: Path, scene: dict) -> None:
    if scene.get("layout", {}).get("room_box_supported") is False:
        _write_observed_planes(scene_dir, scene)
        return
    room = scene["room"]
    w, d = room["width"], room["depth"]
    if min(w, d) <= 0:
        raise RuntimeError("방 치수가 올바르지 않아 평면도를 만들 수 없어요.")
    scale = min(680 / w, 580 / d)
    pw, pd = w * scale, d * scale
    x, y = (1000 - pw) / 2, 180 + (580 - pd) / 2
    unit_note = "마커 기준 크기" if scene["scale"]["method"] == "marker" else "촬영 높이를 가정한 추정 크기 · 실측 보정 필요"
    height_note = "검출" if room["height_source"] == "detected" else "가정"
    edges = [
        ("x_min", x, y, x, y + pd), ("x_max", x + pw, y, x + pw, y + pd),
        ("z_min", x, y, x + pw, y), ("z_max", x, y + pd, x + pw, y + pd),
    ]
    lines = []
    for key, a, b, c, e in edges:
        dash = ' stroke-dasharray="12 8"' if room["walls"][key].get("fallback") else ""
        lines.append(f'<line x1="{a:.2f}" y1="{b:.2f}" x2="{c:.2f}" y2="{e:.2f}"{dash}/>')
    text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="940" viewBox="0 0 1000 940" role="img" aria-label="단일 방의 추정 평면도">
<rect width="1000" height="940" fill="white"/>
<g font-family="Malgun Gothic, sans-serif" fill="#18334d" text-anchor="middle">
<text x="500" y="48" font-size="26">{escape(scene['name'])} · 추정 평면도</text>
<text x="500" y="84" font-size="17">{escape(unit_note)}</text>
<text x="500" y="114" font-size="16">검토용 · 직사각형 공간 가정 · 실내/실외·유리벽·특이 구조 미검사</text>
<rect x="{x:.2f}" y="{y:.2f}" width="{pw:.2f}" height="{pd:.2f}" fill="#edf5fb"/>
<g stroke="#245a86" stroke-width="5">{''.join(lines)}</g>
<text x="500" y="{y - 22:.2f}" font-size="23">{w:.3f} m</text>
<text x="{x + pw + 62:.2f}" y="{y + pd / 2:.2f}" font-size="23" transform="rotate(90 {x + pw + 62:.2f} {y + pd / 2:.2f})">{d:.3f} m</text>
<text x="500" y="{y + pd / 2:.2f}" font-size="25">{room['area_m2']:.2f} ㎡</text>
<text x="500" y="825" font-size="18">천장 {room['height']:.2f} m ({height_note}) · 점선 벽은 위치 추정</text>
<text x="500" y="864" font-size="16">사진에 없는 공간과 연결 관계는 이 도면으로 확정할 수 없습니다.</text>
<text x="500" y="896" font-size="16">문·창·가구·내부 칸막이 미포함 · 건물 전체 도면이 아닙니다.</text>
</g></svg>'''
    (scene_dir / "floorplan.svg").write_text(text, encoding="utf-8")
