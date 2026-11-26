"""실제 Brush 결과의 치수·재보정·학습 기록 검증 (프로젝트 루트에서 실행)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spacecapture.pipeline import Log, evaluate, load_scene, refine_splat_room, _write_scene
from spacecapture.plyio import read_vertices, splat_centers


def main():
    scene_dir = ROOT / "scenes/synth"
    output = ROOT / "artifacts/brush-validation"
    output.mkdir(parents=True, exist_ok=True)
    training = json.loads((scene_dir / "work/brush/training.json").read_text(encoding="utf-8"))
    assert training["returncode"] == 0, "Brush가 정상 종료되지 않았어요."
    checkpoints = training["checkpoints_s"]
    assert max(int(Path(n).stem.split("_")[-1]) for n in checkpoints) == training["steps"], "최종 스텝 파일이 없어요."
    scene = refine_splat_room(scene_dir, log=Log(scene_dir / "process.log"))
    first = scene["room"]
    again = refine_splat_room(scene_dir, log=lambda _: None)
    assert first == again["room"], "재보정을 반복하면 방 치수가 변해요."
    scene["splat_training"] = training
    _write_scene(scene_dir, scene)
    result = evaluate(scene_dir, ROOT / "samples/synth_room.gt.json")
    assert all(abs(row["error_cm"]) <= 3.5 for row in result["rows"][:2]), "가로·세로 오차가 3.5cm를 넘어요."
    assert bool(scene["room"]["ceiling"]) == (scene["room"]["height_source"] == "detected")
    warnings = scene["quality"]["warnings"]
    if scene["room"]["height_source"] == "assumed":
        assert any("천장" in text and "가정" in text for text in warnings), "가정 높이 안내가 없어요."
    else:
        assert not any("천장" in text and "가정" in text for text in warnings), "천장 검출 후에도 옛 경고가 남아 있어요."
    ply = scene_dir / "splat.ply"
    vertices = read_vertices(ply)
    centers = splat_centers(ply)
    x0, _, z0 = scene["room"]["min"]
    x1, _, z1 = scene["room"]["max"]
    interior = centers[(centers[:, 0] > x0 + .25) & (centers[:, 0] < x1 - .25)
                       & (centers[:, 2] > z0 + .25) & (centers[:, 2] < z1 - .25)
                       & (centers[:, 1] > 2.25)]
    report = {"training": training, "evaluation": result, "splats": len(vertices),
              "ply_bytes": ply.stat().st_size, "height_source": scene["room"]["height_source"],
              "reliable_centers": len(centers), "reliable_upper_interior_centers": len(interior),
              "repeat_refinement_stable": True, "warnings": warnings}
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
