"""Export owned/approved photo-plan pairs, with property-disjoint splits.

This prepares supervised examples; it does not train a neural model or grant
rights to a third-party listing. Rights metadata must come from the data owner.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

ALLOWED = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}


def export_training_pairs(properties: list[Path], output: Path) -> dict:
    records=[]
    ids=set()
    for source in properties:
        source=Path(source).resolve()
        truth_path=source/"property.gt.json"
        truth=json.loads(truth_path.read_text(encoding="utf-8"))
        if truth.get("evaluation_only") is True:
            raise ValueError("평가 전용 원본 분할은 학습 예제에 넣을 수 없어요.")
        if truth.get("license") not in ALLOWED or truth.get("commercial_training_allowed") is not True:
            raise ValueError(f"상업 학습 권한 메타데이터를 확인할 수 없는 자료예요: {source.name}")
        property_id=truth["id"]
        if property_id in ids:
            raise ValueError("같은 집 ID가 중복돼요. 사진을 여러 폴더로 나눠도 같은 집은 한 분할에만 넣어야 해요.")
        ids.add(property_id)
        space_ids={space["id"] for space in truth["spaces"]}
        portal_ids={portal["id"] for portal in truth["portals"]}
        if len(space_ids)!=len(truth["spaces"]) or len(portal_ids)!=len(truth["portals"]):
            raise ValueError("정답 공간·문 ID가 중복돼요.")
        for portal in truth["portals"]:
            if len(set(portal["spaces"]))!=2 or not set(portal["spaces"])<=space_ids:
                raise ValueError("정답 문의 양쪽 공간을 확인해 주세요.")
        if len({p["id"] for p in truth["photos"]})!=len(truth["photos"]):
            raise ValueError("정답 사진 ID가 중복돼요.")
        photo_digest=hashlib.sha256()
        for photo in truth["photos"]:
            image=(source/"photos"/photo["id"]).resolve()
            if not image.is_relative_to(source/"photos") or not image.is_file():
                raise ValueError("정답에 연결된 원본 사진을 찾을 수 없어요.")
            if photo["space_id"] not in space_ids or not set(photo.get("visible_portals",[]))<=portal_ids:
                raise ValueError("사진의 공간·문 정답 참조를 확인해 주세요.")
            photo_digest.update(photo["id"].encode()); photo_digest.update(hashlib.sha256(image.read_bytes()).digest())
        records.append((property_id,source,truth,hashlib.sha256(truth_path.read_bytes()).hexdigest(),photo_digest.hexdigest()))
    if len(records)<2:
        raise ValueError("학습·평가를 집 단위로 나누려면 서로 다른 집이 최소 2개 필요해요.")
    records.sort(key=lambda entry:hashlib.sha256(entry[0].encode()).hexdigest())
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    manifest={"version":1,"task":"same_space_and_shared_opening","neural_training_performed":False,
              "split_unit":"property_id","properties":[],"splits":{s:{"properties":[],"pairs":0,"direct_connection_pairs":0} for s in ("train","validation","test")},
              "rights_note":"메타데이터 검사이며 제3자 사진의 실제 권한을 대신 확인하거나 부여하지 않습니다.",
              "limitation":"작은 합성 자료의 형식 검증용입니다. 실제 매물 사진의 학습 성능을 입증하지 않습니다."}
    handles={name:(output/f"{name}.jsonl").open("w",encoding="utf-8") for name in ("train","validation","test")}
    try:
        for i,(property_id,source,truth,digest,photos_digest) in enumerate(records):
            split="test" if i==len(records)-1 else "validation" if len(records)>=3 and i==len(records)-2 else "train"
            stats=manifest["splits"][split]
            stats["properties"].append(property_id)
            manifest["properties"].append({"id":property_id,"split":split,"source":str(source),"license":truth["license"],"truth_sha256":digest,"photos_sha256":photos_digest})
            portal_spaces={portal["id"]:set(portal["spaces"]) for portal in truth["portals"]}
            for a,b in itertools.combinations(truth["photos"],2):
                common=sorted(set(a.get("visible_portals",[]))&set(b.get("visible_portals",[])))
                direct=[portal for portal in common if portal_spaces[portal]=={a["space_id"],b["space_id"]}]
                linked=bool(direct) and a["space_id"]!=b["space_id"]
                sample={"property_id":property_id,"input_photos":[str(source/"photos"/a["id"]),str(source/"photos"/b["id"])],
                        "target":{"same_space":a["space_id"]==b["space_id"],"space_ids":[a["space_id"],b["space_id"]],
                                  "shared_opening_ids":common,"direct_opening_ids":direct,"cross_space_direct_opening":linked},
                        "target_floorplan":str(source/"property.gt.json")}
                handles[split].write(json.dumps(sample,ensure_ascii=False)+"\n")
                stats["pairs"]+=1; stats["direct_connection_pairs"]+=int(linked)
    finally:
        for file in handles.values(): file.close()
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest
