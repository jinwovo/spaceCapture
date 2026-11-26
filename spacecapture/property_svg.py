"""Export observed space/portal candidates without inventing closed house walls."""
from html import escape
from pathlib import Path


def write_property_svg(folder: Path, graph: dict, title="공간 연결 후보"):
    vertices=[v for s in graph["spaces"] for v in s["polygon_xz"]]
    if not vertices:
        (folder/"property.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="160"><text x="30" y="70" font-family="sans-serif" font-size="20">공간 연결을 판단할 관측 근거가 부족합니다.</text></svg>',encoding="utf-8")
        return
    x0,z0=min(p[0] for p in vertices),min(p[1] for p in vertices)
    w,d=max(p[0] for p in vertices)-x0,max(p[1] for p in vertices)-z0
    scale=min(760/max(w,.1),570/max(d,.1))
    parts=[]
    colors=["#dceafb","#f5e8cf","#dceedd","#eadff3","#d6ecee"]
    for i,space in enumerate(graph["spaces"]):
        coords=" ".join(f"{100+(x-x0)*scale:.1f},{150+(z-z0)*scale:.1f}" for x,z in space["polygon_xz"])
        parts.append(f'<polygon points="{coords}" fill="{colors[i%len(colors)]}" stroke="#31536d" stroke-width="2"/>')
        x,z=space["center_xz"]
        parts.append(f'<text x="{100+(x-x0)*scale}" y="{150+(z-z0)*scale}" text-anchor="middle">{escape(space["label"])} · 사진 {len(space["photos"])}장</text>')
    for opening in graph["portals"]:
        x,z=opening["center_xz"]; px,pz=100+(x-x0)*scale,150+(z-z0)*scale
        color="#dd8124" if opening["status"]=="image_supported" else "#929292"
        suffix="" if opening["status"]=="image_supported" else " · 근거 부족"
        parts.append(f'<circle cx="{px}" cy="{pz}" r="10" fill="{color}"/><text x="{px+14}" y="{pz-12}">{escape(opening["id"])} ({opening["shared_landmarks"]}개 공유점){suffix}</text>')
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="880" viewBox="0 0 1000 880">
<rect width="1000" height="880" fill="white"/><g font-family="Malgun Gothic,sans-serif" font-size="17" fill="#18334d">
<text x="100" y="50" font-size="25">{escape(title)}</text>
<text x="100" y="90">관측 영역 {len(graph['spaces'])}개 · 사진 근거 연결 {sum(p['status']=='image_supported' for p in graph['portals'])}개 · 실제 방 개수 미확정</text>
{''.join(parts)}
<text x="100" y="790">영역은 사진에 관측된 범위이며 실제 바닥 면적·방 용도를 확정하지 않습니다.</text>
<text x="100" y="825">문 후보는 광학 연결 근거입니다. 유리·거울·통행 가능 여부는 미검사입니다.</text>
</g></svg>'''
    (folder/"property.svg").write_text(svg,encoding="utf-8")
