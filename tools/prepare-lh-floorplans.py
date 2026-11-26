"""LH '주택 평면도 현황'(공공데이터포털 15037046) ZIP을 검사하고 평면도 이미지를 꺼낸다.

원본은 그대로 보존한다. ZIP 경로 조작을 거절하고, JSON의 base64 이미지만 디코드해
PNG/JPG로 저장한다. 이 자료는 평면도뿐이며 같은 세대의 실내 사진은 없다.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"samples/external/lh-floorplans"
SOURCE={"dataset":"한국토지주택공사_주택 평면도 현황","portal_id":"15037046",
        "url":"https://www.data.go.kr/data/15037046/fileData.do","provider":"한국토지주택공사",
        "license_on_portal":"이용허락범위 제한 없음","copyright_ensured_flag":"N",
        "hold_basis":"한국토지주택공사법 제8조","downloaded":"2026-11-10",
        "attribution":"출처: 한국토지주택공사, 공공데이터포털(data.go.kr) 주택 평면도 현황"}


def entry_name(info):
    name=info.filename
    if not info.flag_bits&0x800:
        try: name=name.encode("cp437").decode("cp949")
        except UnicodeError: pass
    return name


def main():
    archive=FOLDER/"download.bin"
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    out=FOLDER/"images"; out.mkdir(exist_ok=True)
    plans=[]; table=None
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name=entry_name(info)
            if info.is_dir(): continue
            parts=Path(name).parts
            if name.startswith(("/",chr(92))) or ".." in parts or ":" in name or len(parts)>3:
                raise ValueError(f"허용하지 않는 ZIP 경로: {name}")
            data=z.read(info)
            if name.lower().endswith(".csv"):
                for encoding in ("utf-8-sig","cp949"):
                    try: text=data.decode(encoding); break
                    except UnicodeDecodeError: continue
                table=list(csv.DictReader(io.StringIO(text)))
                continue
            text=data.decode("utf-8-sig",errors="strict")
            # 확장자는 .json이지만 실제로는 <img src='data:image/...;base64,...'/> 조각인 파일이 있다.
            found=re.findall(r"data:(image/[a-zA-Z+]+);base64,([A-Za-z0-9+/=]+)",text)
            if found:
                items=[{"mime":m,"data":d} for m,d in found]
            else:
                record=json.loads(text)
                items=record if isinstance(record,list) else [record]
            items=[item["image"] if isinstance(item.get("image"),dict) else item for item in items]
            for k,item in enumerate(items):
                raw=item.get("data") or ""
                if "base64," in raw: raw=raw.split("base64,",1)[1]
                blob=base64.b64decode(raw,validate=True)
                image=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
                if image is None: raise ValueError(f"이미지 디코드 실패: {name}")
                ext=".png" if blob[:4]==b"\x89PNG" else ".jpg"
                key=hashlib.sha256(name.encode()).hexdigest()[:12]+(f"-{k}" if len(items)>1 else "")
                (out/(key+ext)).write_bytes(blob)
                plans.append({"id":key,"zip_entry":name,"complex":parts[0],"unit_type":Path(parts[-1]).stem,
                              "file":f"images/{key}{ext}","mime":item.get("mime"),"width":int(image.shape[1]),
                              "height":int(image.shape[0]),"sha256":hashlib.sha256(blob).hexdigest()})
    manifest={"source":SOURCE,"archive_sha256":digest,"archive_bytes":archive.stat().st_size,
              "plans":plans,"table_rows":len(table or []),"table_columns":list(table[0].keys()) if table else [],
              "interior_photos":False,"scale_annotation":"이미지에 표기된 치수 문자·축척은 자동 해석하지 않음",
              "note":"평면도만 있는 공공 자료. 같은 실제 세대의 사진/도면 쌍이 아니며 실사진 정확도 평가에 쓰지 않는다."}
    (FOLDER/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    if table: (FOLDER/"table.json").write_text(json.dumps(table,ensure_ascii=False,indent=1),encoding="utf-8")
    print(f"평면도 {len(plans)}장, 단지 {len({p['complex'] for p in plans})}곳, 표 {len(table or [])}행")


if __name__=="__main__": main()
