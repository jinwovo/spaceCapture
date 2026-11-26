"""다운로드한 검증 분할을 조사하고 사전 규칙으로 고른 구조를 렌더링한다."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spacecapture.procthor import convert_house
from spacecapture.synth_property import generate_property
from spacecapture.plan_reference import validate_plan


def main():
    parser = argparse.ArgumentParser(description="공식 합성 집 구조의 평가용 자료 생성")
    parser.add_argument("--no-photos", action="store_true", help="기하 관측 정답만 생성")
    args = parser.parse_args()
    source = ROOT / "samples/external/procthor-10k"
    download = json.loads((source/"download.json").read_text(encoding="utf-8"))
    with gzip.open(source/"val.jsonl.gz", "rt", encoding="utf-8") as stream:
        houses = [json.loads(line) for line in stream if line.strip()]
    output = ROOT/"artifacts/procthor-validation"
    output.mkdir(parents=True, exist_ok=True)
    reasons, rooms, selected = Counter(), Counter(), {}
    started = time.perf_counter()
    # Select the first 3-space and 4-space compact house without empty walls.
    # Selection uses structure only, before any inference accuracy is measured.
    for index, house in enumerate(houses):
        count = len(house["rooms"])
        rooms[count] += 1
        try:
            config, metadata = convert_house(house, f"procthor-val-{index:04d}", seed=index)
            validate_plan(config)
        except (ValueError, KeyError) as error:
            reasons[str(error)] += 1
            continue
        reasons["변환 가능"] += 1
        if count in {3,4} and count not in selected and config["size"][0]*config["size"][2] < 100:
            selected[count] = (index,config,metadata)
    report = {"houses": len(houses), "source": download, "room_counts":dict(rooms),
              "conversion_counts":dict(reasons), "selected":[], "real_photo_evaluations":0,
              "neural_training_performed":False, "selection":"추론 전: 3/4공간, 경계상자 100제곱미터 미만, 변환 가능한 첫 사례",
              "upstream_split_policy":"원본 val 분할을 평가에만 사용. train/test와 섞거나 학습에 사용하지 않음."}
    for count,(index,config,metadata) in sorted(selected.items()):
        folder = output/f"val-{index:04d}"
        metadata["source"] = {"commit":download["commit"], "split":"val", "line_index":index,
                              "sha256":download["sha256"], "url":download["url"]}
        metadata["evaluation_only"] = True
        case_start = time.perf_counter()
        generate_property(folder,seed=index,config=config,truth_metadata=metadata,
                          render_photos=not args.no_photos,landmark_density=60)
        shutil.copyfile(source/"LICENSE", folder/"LICENSE.procthor")
        (folder/"NOTICE.txt").write_text("ProcTHOR-10K / Allen Institute for AI / Apache-2.0\n"
            "구조 JSON을 spaceCapture의 합성 사진·도면 형식으로 변환했습니다.\n"
            "텍스처와 촬영 위치는 자체 생성이며 원본 가구·재질·시뮬레이터 에셋은 포함하지 않습니다.\n"
            "원본 논문: Deitke et al., ProcTHOR: Large-Scale Embodied AI Using Procedural Generation, NeurIPS 2022.\n"
            +json.dumps(metadata["source"],ensure_ascii=False,indent=2),encoding="utf-8")
        report["selected"].append({"index":index,"folder":str(folder.relative_to(ROOT)),
            "spaces":count,"portals":len(config["portals"]),"photos":len(config["poses"]),
            "size":config["size"],"generation_seconds":round(time.perf_counter()-case_start,3)})
    report["seconds"] = round(time.perf_counter()-started,3)
    (output/"preparation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
