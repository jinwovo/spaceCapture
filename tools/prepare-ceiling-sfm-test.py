"""시험 구조 집합의 첫 변환 가능한 다중 공간 집을 사진으로 만든다."""
import gzip
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacecapture.procthor import convert_house
from spacecapture.synth_property import generate_property

manifest=json.loads((ROOT/"artifacts/ceiling-learning/dataset.json").read_text(encoding="utf-8"))
indices={int(h["id"].split('-')[1]) for h in manifest["houses"]["test"]}
source=ROOT/"samples/external/procthor-10k"
download=json.loads((source/"download-test.json").read_text(encoding="utf-8"))
for index,line in enumerate(gzip.open(source/"test.jsonl.gz","rt",encoding="utf-8")):
    if index not in indices: continue
    house=json.loads(line)
    if not 2<=len(house["rooms"])<=4: continue
    try: config,metadata=convert_house(house,f"procthor-test-{index:04d}",seed=index)
    except ValueError: continue
    metadata.update(evaluation_only=True,source={"commit":download["commit"],"split":"test","line_index":index,
                    "sha256":download["sha256"],"url":download["url"]})
    folder=ROOT/f"artifacts/ceiling-learning/photo-test-{index:04d}"
    generate_property(folder,config=config,truth_metadata=metadata,seed=index,landmark_density=60)
    shutil.copyfile(source/"LICENSE",folder/"LICENSE.procthor")
    (folder/"NOTICE.txt").write_text("ProcTHOR-10K / Allen Institute for AI / Apache-2.0\n"
        "공식 시험 집 JSON을 자체 사진·정답 도면 형식으로 변환했습니다.\n"
        "텍스처·촬영 위치는 자체 생성했으며 가구 제외·창문/현관 막음·벽 두께 0으로 수정했습니다.\n"
        "외부 에셋이나 사전학습 가중치는 포함하지 않았습니다. 원본 버전·해시는 property.gt.json에 있습니다.\n",encoding="utf-8")
    (ROOT/"artifacts/ceiling-learning/photo-test.json").write_text(json.dumps({"index":index,"folder":str(folder.relative_to(ROOT)),
        "scene":metadata["id"],"selection":"시험 집 중 현재 변환 가능한 첫 2~4공간 집. 추론 점수로 선택하지 않음."},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"시험 사진 준비 완료: {folder}",flush=True)
    break
else: raise RuntimeError("변환 가능한 다중 공간 시험 집이 없어요.")
