# spaceCapture 상세 사용법 (한국어)

주택·사무실·회의실의 사진·영상 → 검토용 3D 공간과 2D 구조 후보 → 둘러보기 · 치수 확인 · 가구 배치 테스트.

최종 목표는 중개업자가 사진·영상과 공간 데이터를 입력하면 **3D 공간과 2D 도면**을 보여주는 앱입니다. 로컬 웹 업로드, 분석 대기열, 3D 보기, 추정 SVG 도면, 관측 공간·문 연결 후보와 근거 사진 이동을 구현했습니다. 같은 사진 연결 묶음을 방 개수로 세지 않습니다. 실제 방 용도·방 개수·통행 가능한 문은 미확정이며, 사이트 URL 가져오기도 미구현입니다.

## 준비 (한 번만)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd viewer; npm install; cd ..
```

실사 3D(가우시안 스플래팅)는 [Brush](https://github.com/ArthurBrussee/brush/releases)(Apache-2.0)를 씁니다.
`tools/brush-v0.3.0/brush_app.exe`에 이미 받아 두었어요. AMD/Intel GPU에서도 동작해요.

## 사용

웹 앱은 프로젝트 루트에서 다음과 같이 실행하세요.

```powershell
npm --prefix viewer run build
.\.venv\Scripts\python.exe -m spacecapture serve
```

**http://127.0.0.1:8765**에서 사진 여러 장 또는 영상 1개를 선택하고 분석을 시작합니다. 주택·사무실·회의실을 선택할 수 있으며 선택한 용도는 사용자 제공 정보로 기록합니다. 사진·영상은 이 PC에서 처리하고 외부 AI API로 보내지 않습니다. 여러 작업은 순서대로 처리합니다.

사진은 3~150장(장당 30MiB), 영상은 1개(2GiB), 전체 업로드는 작업당 2GiB 이하입니다. 원본과 작업 기록은 `artifacts/jobs/`에 남습니다. 사용자 지정 실외 자료는 실내 도면 생성에서 제외합니다. 자동 실내/실외 판별은 아직 지원하지 않습니다.

CLI도 사용할 수 있습니다.

```powershell
# 1) 크기 기준 마커 인쇄 (반드시 '실제 크기/100%'로 인쇄, 자로 100mm 확인)
.\.venv\Scripts\python.exe -m spacecapture marker

# 2) 영상 처리 → scenes/<이름>/
.\.venv\Scripts\python.exe -m spacecapture process 내방.mp4 --name myroom

# 3) 뷰어
cd viewer; npm run dev   # http://localhost:5173
```

촬영 팁: 방을 한 바퀴 천천히 걸으며 **방 건너편**을 비추고, 바닥과 벽이 만나는 선이 보이게, 천장도 한두 번, 벽에 붙인 마커를 2~3초 여러 각도에서.

780M에서 Brush를 빠르게 시험하려면 다음 설정을 쓰세요. 학습 진행 상황은 30초마다 표시하고 시간·설정·프로세스 메모리는 `scenes/<이름>/work/brush/training.json`에 저장합니다.

```powershell
.\.venv\Scripts\python.exe -m spacecapture splat synth --steps 3000 --max-resolution 640 --max-splats 150000
# 더 오래 학습해서 비교: --steps 15000 (나머지 옵션 동일)
```

일반 CLI 기본값은 15,000스텝, 최대 1,280px, 최대 500,000개 스플랫입니다. GPU 공유 메모리 사용량은 위 프로세스 메모리 수치에 포함되지 않습니다.

## 매물 사진으로 만들기

**같은 집·같은 층의 연결되는 공간 사진**을 한 폴더에 넣으세요. 사진 사이에 같은 벽·문·가구 등이 겹쳐 보여야 합니다. 방에서 문을 본 사진과 문밖 복도에서 본 사진을 함께 넣으면 연결 근거로 사용할 수 있습니다. 촬영 순서는 몰라도 됩니다. 입력은 최소 3장이지만, 장수만으로 복원 성공을 보장하지는 않습니다.

```powershell
# 먼저 사진의 3D 점과 추정 2D 도면 생성
.\.venv\Scripts\python.exe -m spacecapture photos .\매물사진 --name listing-room --no-splat
# 복원 성공 후 Brush 실사 표현 학습
.\.venv\Scripts\python.exe -m spacecapture splat listing-room --steps 3000 --max-resolution 640 --max-splats 150000
```

`scenes/listing-room/scene.json`, `points.ply`, `floorplan.svg`를 생성하고, Brush 실행 후 `splat.ply`도 저장합니다. 뷰어에서 `listing-room`을 선택해 3D와 위에서 본 평면도를 볼 수 있고 SVG는 독립된 2D 파일로 열 수 있습니다.

JPG/PNG/WebP/BMP/TIFF를 지원하며 기본 최대 150장입니다. 중복·손상 사진을 제외하고 원본은 보존합니다. 사진별 해상도와 카메라 설정을 따로 추정하고, 복원하지 못한 사진은 장면에 기록합니다.

마커가 없는 매물 사진의 치수는 **촬영 높이를 가정한 추정치**입니다. 알려진 실제 길이로 뷰어에서 보정해야 합니다. 관측 영역·문/개구부 후보는 지원하지만, 방 용도·실제 방 개수·문 인식·여러 층·집 전체 완성 도면·사이트 URL 자동 가져오기는 아직 확정하거나 지원하지 못합니다.

사진 간 기하 검증 매칭으로 **연결되는 사진 묶음·고립 사진**을 찾습니다. 이 묶음은 방 분류 결과가 아니며, 서로 연결되지 않는 묶음을 임의로 합쳐 집 구조로 만들지 않습니다. 현재는 복원된 부분과 미복원 사진을 함께 기록합니다.

관측 점에서 수직면 후보를 찾고 직사각형 경계를 부분 검사합니다. 긴 내부 면·비스듬한 면·빠진 경계 때문에 직사각형을 지지하지 못하면, 2D 파일에는 **관측한 벽 후보만** 그리고 면적 확정을 보류합니다. 뷰어의 걷기·가구 배치도 보류하며 기존 배치 저장값은 보존합니다. 실제 주택·사무실에서의 정확도 보장은 아닙니다.

공간 용도와 실내/실외, 유리벽·거울은 자동 판별하지 않습니다. `assessment.json`에 기하 복원 근거와 **미검사 항목**을 구분합니다. 곡선·가림·투명면·숨은 칸막이를 완전히 복원하는 기능도 없습니다. 검토용 구조 후보를 완성 도면으로 취급하면 안 됩니다.

기존 장면도 재학습 없이 검사할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m spacecapture inspect listing-room
```

## 여러 공간과 문 연결 확인

뷰어의 **집·공간 연결 후보**에서 영역별 사진과 문 양쪽의 근거 사진을 볼 수 있습니다. 사진을 누르면 그 촬영 위치로 이동합니다. 사진 근거가 있는 연결과 약한 기하 후보를 구분하고, 촬영 방향·화각도 확인합니다. 유리나 거울을 통한 시야는 실제로 걸어갈 수 있는 문을 뜻하지 않습니다.

`property.svg`는 여러 관측 영역과 연결의 도면이고, `floorplan.svg`는 기존 직사각형 후보 또는 관측 벽 후보 도면입니다. 둘 다 완성된 집 도면으로 확정하지 않습니다.

직접 만든 여러 공간의 사진·정답 도면으로 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m spacecapture synth-property --out samples/house --kind rooms_corridor
.\.venv\Scripts\python.exe -m spacecapture photos samples/house/photos --name house --no-splat
.\.venv\Scripts\python.exe -m spacecapture splat house --steps 3000 --max-resolution 640 --max-splats 150000
.\.venv\Scripts\python.exe -m spacecapture evaluate-property house --gt samples/house/property.gt.json
```

`synth-property`는 방 두 개 또는 방 두 개·복도를 직접 생성합니다. 추론에는 사진만 넣고, 방 윤곽·문·촬영 위치 정답은 평가용으로 별도 보관합니다. 현재 합성 사진 시험은 44장→영역 2개·사진 근거 연결 1개, 62장→영역 3개·사진 근거 연결 2개와 약한 후보 1개였습니다. 실제 매물 사진 성능은 별도 검증이 필요합니다.

## 사진·평면도 학습 데이터

[자료 조사](photo-plan-data.md)에 Rightmove·Zillow Showcase·ZInD 등의 사진/도면 제공 내용과 이용 조건을 정리했습니다. 공개 사이트의 사진을 학습 허용 자료로 간주하지 않고, 연구 전용 데이터·코드·가중치를 포함하지 않았습니다.

국내에는 LH 하나로 내집, 제주개발공사 공공임대주택 평면도 목록, 마이홈 공고 첨부 도면이 있습니다. 확인한 공식 경로와 확보 상태는 [공공 평면도 비교](reference-plan.md)에 기록했습니다. 실제 외부 매물과 같은 호실의 사진·도면을 확보해 검증한 건수는 아직 0건입니다.

조사를 국내외 **34개 항목**으로 넓힌 [데이터 비교표](dataset-survey.md)와 [출처 목록](dataset-survey.json)을 추가했습니다. 실제 사진·도면 쌍, 도면만 있는 자료, RGB-D 실측 자료, 합성 구조 데이터를 구분합니다. 기존 조건에 맞는 후보는 PanoInfinigen **indoor**(BSD-3-Clause), ProcTHOR-10K 집 JSON(Apache-2.0), Infinigen 생성기(BSD-3-Clause)입니다. 사진·도면 쌍을 모두 즉시 제공한다는 뜻은 아닙니다.

다운로드를 막던 Codex 오프라인 정책 문제는 실행 환경 변경 후 실제 HTTPS·파일 다운로드로 복구를 확인했습니다. [진단 결과와 재검증 방법](network-diagnostics.md)을 참고하세요.

[외부 합성 구조 검증](procthor-validation.md)에서는 ProcTHOR 구조 **1,000개**를 확보하고 두 집을 자체 렌더링해 사진만으로 복원했습니다. 4공간 사례의 영역 IoU **0.8723**, 문 연결 **3/3**, 문 중심 평균 오차 **15.04cm**입니다. 다른 3공간 사례는 공간 **2/3**, 문 **1/2**만 찾았습니다. 당시에는 실제 매물 사진 평가·신경망 학습을 수행하지 않았고 높은 천장 오검출도 남아 있었습니다.

이후 [천장 후보 신경망 학습](ceiling-learning.md)을 실제 수행했습니다. 학습 80집·검증 24집·시험 20집을 구조 유형과 도면 지문으로 분리하고, CPU에서 작은 기하 특징 MLP를 학습해 파이프라인에 연결했습니다. 약 48cm 낮게 잡던 합성 천장을 **4.226m**로 복원하고 근거가 부족한 다른 천장은 보류합니다. 사진 의미 분류·집 전체 도면 모델은 아직 학습하지 않았으며, 원래 synth의 천장도 **2.4m 가정**을 유지합니다.

일반 도면의 공간 윤곽·문을 미터 단위 JSON으로 표기하면 정답 촬영 위치 없이도 비교할 수 있습니다. 복원 장면과 도면의 같은 지점 3개 이상을 `anchors.json`에 지정하면 겹침 SVG와 영역·문 오차 JSON을 저장합니다. 현재 도면 OCR·대응점 자동 검출은 없습니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m spacecapture evaluate-property listing-house --gt plan.json --anchors anchors.json --out artifacts/listing-comparison
```

원래 복원 크기를 유지하는 비교와 도면 배율에 맞춘 형상 비교를 구분합니다. 같은 주택형의 참고 도면을 같은 실제 호실의 정답으로 집계하지 않습니다. 입력 형식은 위 문서에 있습니다. 새 비교 경로의 자체 합성 시험 결과는 `artifacts/reference-plan-validation/`에 저장했습니다.

직접 만든 자료 또는 허용된 권한 메타데이터가 있는 자료는 다음 명령으로 사진 관계 학습 예제로 변환합니다. 모델을 재학습하는 명령은 아닙니다.

```powershell
.\.venv\Scripts\python.exe -m spacecapture training-pairs artifacts/property-validation/two_rooms_photos artifacts/property-validation/rooms_corridor_photos --out artifacts/property-validation/training-pairs
```

같은 공간 여부, 같은 문이 보이는지, 그 문이 양쪽 공간을 직접 연결하는지를 정답으로 기록합니다. 집 ID 단위로 학습·평가를 분리하며 사진·정답의 해시와 권한 메타데이터를 남깁니다. 현재 자료는 합성 집 두 개뿐이며 일반 매물용 신경망 학습 성능을 입증하지 않습니다. 신경망 재학습은 수행하지 않았습니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m spacecapture synth                    # 정답 치수를 아는 합성 방 영상
.\.venv\Scripts\python.exe -m spacecapture process samples\synth_room.mp4 --name synth
.\.venv\Scripts\python.exe -m spacecapture evaluate synth           # 정답과 비교
```

합성 영상의 마커 기준 가로·세로 오차는 수 cm입니다. 천장 복원은 별도 검증 항목이며, 충분한 근거가 없으면 2.4m 가정값(정답 2.5m 대비 −10cm)을 표시합니다. 실제 Brush 검증 수치는 `artifacts/brush-validation/result.json`과 `AGENTS.md`를 참고하세요.
합성 프레임 59장을 순서를 섞은 사진으로 입력한 시험은 가로·세로 오차 +2.9/+2.8cm였습니다. 실제 매물 사진의 성능을 보장하는 결과는 아닙니다. 입력·평가 검사는 `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`로 실행합니다.
웹에서 같은 합성 사진 59장을 업로드해 실제 학습까지 검증했으며 가로·세로 오차는 +2.9/+2.7cm였습니다. 합성 ㄱ자 사무실에서는 직사각형 면적·도면 보류를 확인했습니다. 자세한 실행 기록과 화면은 `artifacts/workflow-validation/report.md`에 있습니다. Python 검사 18개와 뷰어 빌드가 통과했습니다.
여러 공간·문 연결 시험은 `artifacts/property-validation/report.md`에 기록했습니다. 현재 Python 검사 28개 및 뷰어 빌드가 통과했고, 문 양쪽 썸네일 로딩·촬영 위치 이동 브라우저 검사도 통과했습니다.
일반 도면 대응점 비교 추가 후 Python 검사 **36개**와 compileall을 통과했습니다. 합성 집 두 개의 원래 배율 유지 비교에서 영역 IoU **0.9239 / 0.9092**, 문 중심 평균 오차 **7.65 / 11.56cm**였습니다. 실제 공개 매물 정답 검증이나 신경망 재학습 결과는 아닙니다. 이번 변경은 Python 평가 경로와 문서이며 뷰어 소스는 바꾸지 않았습니다.
`scenes/synth-splattest`는 뷰어 시험용으로 정답 형상에서 만든 가짜 스플랫이에요. 지워도 됩니다.

## 구성 (모두 상업적 사용 가능한 라이선스)

| 단계 | 사용 |
|---|---|
| 프레임 선택 | OpenCV (흐린 프레임 제외) |
| 카메라 위치 | COLMAP 4 / GLOMAP (BSD), CPU, 초점거리 자동 수렴 |
| 바닥·방향·크기 | 직접 구현: 바닥 평면, 벽 방향 정렬, ArUco 마커 삼각측량 |
| 방 치수 | 직접 구현: 벽·천장 검출 |
| 실사 3D | Brush (Apache-2.0) |
| 뷰어 | three.js + Spark (MIT) |
