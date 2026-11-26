"""Report connected image groups using geometrically verified COLMAP matches."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pycolmap


def match_connectivity(database: Path, min_inliers: int = 30) -> dict:
    with closing(sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True)) as db:
        names = dict(db.execute("SELECT image_id, name FROM images"))
        adjacency = {i: set() for i in names}
        edges = 0
        for pair_id, count, config in db.execute("SELECT pair_id, rows, config FROM two_view_geometries"):
            # Config 0=undefined, 1=degenerate. Planar matches still indicate a
            # possible connection, but are not proof of translational parallax.
            if count < min_inliers or config in (0, 1):
                continue
            a, b = pycolmap.pair_id_to_image_pair(pair_id)
            if a in adjacency and b in adjacency:
                adjacency[a].add(b)
                adjacency[b].add(a)
                edges += 1
    unseen, groups = set(names), []
    while unseen:
        seed = min(unseen)
        stack, component = [seed], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        unseen -= component
        groups.append(sorted(names[i] for i in component))
    groups.sort(key=lambda group: (-len(group), group))
    sizable = sum(len(group) >= 3 for group in groups)
    return {"status": "connected" if len(groups) == 1 and len(names) >= 3 else "disconnected",
            "min_verified_inliers": min_inliers, "images": len(names), "verified_pairs": edges,
            "groups": [{"id": i + 1, "images": group, "count": len(group),
                        "reconstruction_candidate": len(group) >= 3} for i, group in enumerate(groups)],
            "reconstruction_groups": sizable,
            "message": ("사진 간 기하 연결 근거를 찾았어요. 같은 방이나 실내임을 확정하는 검사는 아니에요."
                        if len(groups) == 1 else "사진이 서로 연결되지 않는 묶음으로 나뉘어요. 묶음 사이의 위치·방 연결은 확정하지 않았어요.")}
