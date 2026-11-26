# 공공 평면도와 매물 사진을 비교하기

사진으로 복원한 집과 별도 평면도를 비교하는 방향이 맞다. 사진은 추론 입력으로, 도면은 별도 평가 자료로 보관한다. 도면으로 결과를 수정한 뒤 같은 도면으로 평가하면 사진만으로 복원한 성능을 측정할 수 없다.

## 실제로 확인한 공개 경로

확인일: 2026-11-09. [출처 목록과 확인 상태](public-plan-sources.json).

이후 국내외 연구 데이터·실측 영상·합성 자료까지 넓힌 **34개 항목 조사**는 [dataset-survey.md](dataset-survey.md)에 있다. 국내 AI Hub의 건축 도면(71465), 실내공간 3D 종합(550), 가상 실내 공간 3D 합성(71765)도 포함했다. 이들 자료의 도면/실내 사진/구조 정답 보유 여부와 이용 정책을 각각 확인했다.

| 공식 경로 | 사용할 수 있는 자료의 종류 | 현재 확보 상태 |
|---|---|---|
| [LH 하나로 내집](https://hanaro.lh.or.kr/), [LH 공식 개설 안내](https://www.lh.or.kr/gallery.es?act=view&bid=0003&list_no=11969&mid=a10502000000) | 견본주택·단지별 분양 사이트 연결. 사진·VR과 주택형 도면을 짝지을 후보 | [서울공릉 59/74㎡ 단지 페이지](https://hanaro.lh.or.kr/siteDetail/view?menuKey=60O0Um5kwF&siteDetailId=73) 확인. 개별 사이트 응답 지연으로 원본 사진/도면 쌍은 확보하지 못함 |
| [제주개발공사 공공데이터 목록](https://www.jpdc.co.kr/open/open/open.htm) → 주거복지 7번 → [공공데이터포털 15089013](https://www.data.go.kr/data/15089013/fileData.do) | 건설형 공공임대주택별 평면도 및 조감도 | 제공기관의 실제 항목·연결 확인. 조감도는 실내 촬영 사진과 다르며, 일치하는 실내 사진은 미확보 |
| [마이홈 공식 원주문막1 첨부 문서](https://www.myhome.go.kr/hws/com/fms/cvplFileDownload.do?atchFileId=17031a9c0c822&fileSn=1) | 46㎡ A/B형·51㎡형의 실 구획과 치수 표기 | 도면이 있는 문서를 확인. 같은 세대의 실내 사진은 미확보 |

위 자료는 출처 조사 결과다. 프로젝트에 원본을 내려받거나 학습 데이터로 포함하지 않았다. **외부 실제 주택의 사진/도면 정답 검증은 현재 0건**이다. 공공기관의 사이트에 공개됐다는 사실만으로 개별 사진·도면에 MIT/BSD/Apache 조건을 부여하지 않는다. 학습 권한은 사진과 도면 각각 확인한다. 제공기관에 연락하거나 권한 요청을 보내지는 않았다.

## 어떤 사진과 도면을 짝지어야 하나

단지명·주소 → 동/주택형 → 좌우 대칭 → 확장 여부 → 개조 상태를 확인한다. 같은 전용면적만으로 같은 구조라고 판단하지 않는다. 견본주택 사진과 같은 주택형 도면은 유용한 비교 자료지만 실제 매물 호실의 변경 상태까지 증명하지 않는다.

평가 JSON의 `reference.match_level`을 다음처럼 구분한다.

- `same_unit`: 같은 실제 세대의 사진/도면. `pairing_checks`의 `building`, `unit_type`, `mirror`, `balcony`, `renovation`이 모두 `true`이고 양쪽 출처 URL이 있어야 정답 집계 가능한 자료로 표시한다.
- `same_type`: 같은 주택형의 참고 비교. 영역 IoU·문 오차를 계산해도 실제 매물 정답 정확도로 집계하지 않는다.
- `unverified`: 대응을 확인하지 않은 참고 자료. 기본값이다.

이 분류는 제공한 확인 기록을 검사한다. 프로그램이 주소·호실·개조 상태나 저작권을 실제로 확인했다는 뜻은 아니다. 직접 생성한 기존 합성 정답은 `origin=procedurally_generated_here`로 별도 표시한다.

## 도면을 평가에 넣는 방법

현재 자동 PDF 해석·도면 OCR은 없다. 도면을 보고 공간 내부 윤곽을 `polygon_xz`에 미터 좌표로 표기하고 문을 연결된 두 공간 ID·중심·폭으로 표기한다. 벽 중심선 치수와 실제 벽 안쪽 치수를 섞으면 벽 두께가 오차에 포함되므로 내부 치수 기준을 맞춰야 한다. 외부 도면은 다음 형식을 쓴다. 아래 좌표는 형식 설명용이며 실제 공개 주택 도면에서 추출한 값이 아니다.

```json
{
  "id": "example-not-real-property",
  "reference": {
    "match_level": "unverified",
    "source_url": "",
    "photo_source_url": "",
    "pairing_checks": {
      "building": null, "unit_type": null, "mirror": null,
      "balcony": null, "renovation": null
    }
  },
  "spaces": [
    {"id": "room-a", "kind": "room", "polygon_xz": [[0,0],[4,0],[4,3],[0,3]]},
    {"id": "room-b", "kind": "room", "polygon_xz": [[4,0],[7,0],[7,3],[4,3]]}
  ],
  "portals": [
    {"id": "door-a-b", "spaces": ["room-a","room-b"], "center_xz": [4,1.5], "width_m": 0.9}
  ]
}
```

정답 촬영 위치를 모르는 일반 매물은 복원 장면과 도면의 같은 벽 모서리 등 **서로 떨어진 대응점 3개 이상**을 별도 JSON으로 지정한다. 대응점들이 한 직선에 몰리거나 유한하지 않은 좌표이면 거절한다. 임의로 좌우 반전하여 점수를 높이지 않는다. `holdout` 점은 정렬에 넣지 않고 별도 오차를 계산한다.

```json
{
  "units": "metres",
  "fit_scale": false,
  "points": [
    {"id":"a", "scene_xz":[-2,-1.5], "plan_xz":[0,0]},
    {"id":"b", "scene_xz":[2,-1.5], "plan_xz":[4,0]},
    {"id":"c", "scene_xz":[-2,1.5], "plan_xz":[0,3]},
    {"id":"d", "scene_xz":[2,1.5], "plan_xz":[4,3], "role":"holdout"}
  ]
}
```

```powershell
.\.venv\Scripts\python.exe -X utf8 -m spacecapture evaluate-property listing-house --gt plan.json --anchors anchors.json --out artifacts/listing-comparison
```

`comparison.json`에 영역 IoU, 연결 정밀도/재현율, 문 중심·폭 오차, 정렬에 사용한 점과 별도 검증점의 오차를 저장한다. `comparison.svg`는 초록색 도면과 파란색 사진 복원을 겹쳐 그린다. 같은 두 공간 사이에 문이 여러 개 있으면 개별 대응하고, 문 중심이 50cm 이내인 후보만 연결 정답으로 센다.

`fit_scale=false`는 원래 복원 크기를 유지하고 수평 회전/이동만 정렬한다. 크기를 모르는 사진의 **형상만 비교**하려면 `true`로 명시한다. 이때 결과는 `scale_normalized_shape`이며 실측 크기 정확도로 해석하지 않는다. 정답 촬영 위치를 사용하는 기존 평가도 배율을 정렬하므로 같은 표시를 붙인다. 추론 결과 파일은 변경하지 않는다.

## 실행한 검증

`tools/validate-reference-plan.py`로 이미 사진에서 복원한 자체 합성 집 두 개에 대해 새 평가 경로를 실행했다. 대응점은 합성 정답 촬영 위치의 바닥 투영이며, 실제 매물의 벽 모서리를 사람이 찾은 시험은 아니다. 정렬점 3개와 나머지 별도 검증점을 분리했다. 실행 결과와 겹침 그림은 `artifacts/reference-plan-validation/`에 저장한다.

향후 학습 자료를 늘릴 때 같은 집뿐 아니라 같은 단지의 반복 주택형 도면도 학습/평가 분할에 함께 들어가지 않게 묶어야 한다. 현재 `training-pairs`는 집 ID별 분할이며 주택형 그룹 분할은 아직 없다. 이번 구현은 평가 경로이고 신경망 재학습은 수행하지 않았다.
