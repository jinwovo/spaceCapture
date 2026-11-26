# 사진·평면도 데이터 조사

확인일: 2026-11-10. 검토 항목 **38개**.

사진/도면 쌍, 도면 구조, 합성 실내, 실측 RGB-D, 국내 공공자료의 주요 계열 조사. 전 세계 자료를 빠짐없이 확인했다는 의미는 아닙니다.

자료의 존재, 이용 조건, 이 프로젝트의 입력 적합성, 실제 확보·평가는 별개다. 아래 표는 검색·원문 확인 결과이며 다운로드/학습 완료 목록이 아니다.

## 조사 결과

| 구분 | 항목 수 |
|---|---:|
| 기존 라이선스 조건에 맞는 후보 | 3 |
| 범위·조건 추가 확인 | 22 |
| 비상업 등 기존 조건으로 제외 | 12 |
| 공식 원문 확인 보류 | 1 |

기존 조건에 맞는 세 항목은 **PanoInfinigen indoor(BSD-3-Clause)**, **ProcTHOR-10K 집 JSON(Apache-2.0)**, **Infinigen 생성기(BSD-3-Clause)**다. 셋 모두 실제 매물 사진과 완성 평면도 정답의 쌍을 제공하는 자료로 확정한 것은 아니다.

CC BY 자료는 비상업 자료와 구분해 보존했지만, 현재 프로젝트의 BSD/MIT/Apache 자동 허용 목록을 확대하거나 외부 원본·코드·가중치를 포함하지 않았다. 조건부 상업 약관과 ShareAlike/ND 자료도 별도 검토 상태다.

## 사진·도면 쌍

| 자료·공식 출처 | 규모·자료 | 이용 조건·판단 | 이 앱에서의 용도와 한계 |
|---|---|---|---|
| [C3 / C3Po 데이터](https://c3po-correspondence.github.io/) | 약 90K 사진–평면도 쌍, 597개 장면<br>사진·도면 대응 있음 | 데이터: CC-BY-4.0 표기; 원본 Wikimedia/Flickr 자료 조건 별도 확인<br>코드: CC-BY-NC-SA-4.0<br>**범위·조건 추가 확인** · [조건 근거](https://huggingface.co/datasets/kwhuang/C3) | 사진과 도면의 대응점·카메라 위치 평가<br>교회·기념물 등 장면이 포함되어 주택 데이터와 구분해야 합니다. 실내/실외도 섞입니다. 데이터 카드 표기를 원본 매체 전체의 권리 확인으로 대신하지 않으며, C3Po 코드는 사용하지 않습니다. |
| [Rent3D](https://www.cs.toronto.edu/~fidler/projects/rent3D.html) | 임대주택 215채, 사진 1,570장(실외 포함)<br>사진·도면 대응 있음 | 데이터: 공식 페이지에서 상업 이용 허용 조건 확인 못함<br>코드: 별도 확인<br>**범위·조건 추가 확인** · [조건 근거](https://www.cs.toronto.edu/~fidler/projects/rent3D.html) | 방·문·창·도면 위치가 대응된 실제 부동산 사진 평가<br>용도에 매우 가깝지만 상업 사용 조건이 미확인입니다. 자료 문의처는 있으나 연락·신청을 보내지 않았습니다. |
| [Rent3D++ / Plan2Scene](https://3dlg-hcvc.github.io/plan2scene/) | Rent3D의 방·출입구·사진 소속 라벨 보완<br>사진·도면 대응 있음 | 데이터: 별도 데이터 신청; 상업 이용 허용 조건 확인 못함<br>코드: MIT<br>**범위·조건 추가 확인** · [조건 근거](https://github.com/3dlg-hcvc/plan2scene) | 희소한 매물 사진과 방/문 도면 라벨<br>도면과 사진을 입력받아 3D를 만드는 과제입니다. 도면 없이 사진만으로 복원한 결과와 구분해야 합니다. 코드 MIT를 원본 임대 사진 라이선스로 해석하지 않습니다. |
| [Zillow Indoor Dataset](https://github.com/zillow/zind) | 집 전체 파노라마·방 윤곽·문/창/개구부<br>사진·도면 대응 있음 | 데이터: 학술·비상업용; 상업 사용 별도 협의<br>코드: 별도 확인<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://github.com/zillow/zind#faq) | 여러 방 연결과 비정형 방의 참고 연구<br>공식 FAQ가 제품/서비스 알고리즘 개선을 허용하지 않습니다. 현재 프로젝트에서 내려받거나 평가·학습에 쓰지 않습니다. |
| [LIFULL HOME'S / NII](https://www.nii.ac.jp/dsc/idr/en/lifull/homes.html) | 2015 스냅샷 매물 533만 건, 사진·도면 포함<br>사진·도면 대응 있음 | 데이터: 대학/공공연구기관 연구자·학술 목적 한정 신청<br>코드: 별도 확인<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://www.nii.ac.jp/dsc/idr/en/lifull/documents/lifull_homes-policy.html) | 실제 매물 사진·도면·면적·주택 속성<br>공식 안내는 민간 기업 소속 신청을 받지 않습니다. 기본 이미지가 120×120이며 고해상도 자료는 별도 신청입니다. |

## 도면 구조

| 자료·공식 출처 | 규모·자료 | 이용 조건·판단 | 이 앱에서의 용도와 한계 |
|---|---|---|---|
| [ResPlan](https://github.com/m-agour/ResPlan) | 주택 도면 17,000개<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-4.0: 저자 소유 기하·분류·연결·정제 기여 범위<br>코드: MIT<br>**범위·조건 추가 확인** · [조건 근거](https://github.com/m-agour/ResPlan/blob/main/LICENSE) | 여러 방 윤곽·연결 그래프·구조 사전 분포<br>원본 도면 이미지·실내 사진·주소는 배포하지 않습니다. 실사진 복원 정답 쌍으로 사용할 수 없습니다. 중복 도면에 의한 분할 누출도 확인해야 합니다. |
| [Swiss Dwellings](https://archilyse.standfest.science/swiss-dwellings) | 건물 약 3,100개, 아파트 45,000개 이상<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-4.0<br>코드: 파이프라인 AGPL; 별도 상업 라이선스 선택지<br>**범위·조건 추가 확인** · [조건 근거](https://zenodo.org/records/7788422) | 미터 단위 방·문·개구부 기하와 복합 건물 구조<br>실내 촬영 사진 쌍이 아닙니다. 데이터와 파이프라인 라이선스가 다릅니다. AGPL 구현은 프로젝트에 포함하지 않았습니다. |
| [WAFFLE](https://github.com/TAU-VAILab/WAFFLE) | Wikimedia 기반 도면 18,556개<br>실내 사진·평면도 정답 쌍 아님 | 데이터: Wikimedia 원본 파일별 조건 확인 필요<br>코드: 공식 README가 Wikimedia Commons 조건을 연결<br>**범위·조건 추가 확인** · [조건 근거](https://github.com/TAU-VAILab/WAFFLE#license) | 도면의 범례·텍스트·건물 유형 해석<br>같은 주택 실내 사진과 방/문을 짝지은 정답이 아닙니다. 공용 라이선스 하나로 모든 원본을 허용했다고 취급하지 않습니다. |
| [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) | 도면 5,000개, 객체 범주 80종 이상<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-NC-4.0<br>코드: 공식 저장소 LICENSE 확인<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE) | 도면 이미지의 방·벽·문 분할 연구<br>비상업 조건이며 일반 실내 사진 입력의 정답 자료와도 다릅니다. |
| [FloorPlanCAD](https://floorplancad.github.io/) | 주거·상업 CAD 도면 15,663개 공개 버전<br>실내 사진·평면도 정답 쌍 아님 | 데이터: 공식 사이트/어노테이션 CC-BY-NC-4.0<br>코드: 별도 확인<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://floorplancad.github.io/) | CAD 선·문·창 기호 검출<br>비상업 조건입니다. CAD 도면과 같은 세대의 촬영 사진 쌍이 아닙니다. |
| [RPLAN / DeepLayout](https://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) | 주택 래스터 도면 계열<br>실내 사진·평면도 정답 쌍 아님 | 데이터: 이번 조사에서 공식 원문 조회 실패<br>코드: 별도 확인<br>**공식 원문 확인 보류** · [조건 근거](https://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) | 방 배치·평면도 생성 자료 후보<br>제3자 목록의 라이선스 표기를 채택하지 않았습니다. 상업 사용을 허용한 것으로 간주하지 않습니다. |

## 합성 실내

| 자료·공식 출처 | 규모·자료 | 이용 조건·판단 | 이 앱에서의 용도와 한계 |
|---|---|---|---|
| [Infinigen Indoors 생성기](https://github.com/princeton-vl/infinigen) | 방·여러 방·여러 층의 절차적 생성<br>실내 사진·평면도 정답 쌍 아님 | 데이터: 자체 절차 생성 결과; 외부 에셋 사용 시 별도 조건<br>코드: BSD-3-Clause<br>**기존 라이선스 조건에 맞는 후보** · [조건 근거](https://github.com/princeton-vl/infinigen/blob/main/LICENSE) | 정확한 구조를 가진 다양한 합성 사진/도면 생성<br>기존 실매물 사진 자료가 아닙니다. Blender와 렌더링 환경이 추가로 필요하며 이 PC에서 실행하지 않았습니다. 기본 코드의 외부 의존성까지 설치 승인된 것으로 해석하지 않습니다. |
| [PanoInfinigen / indoor](https://huggingface.co/datasets/prs-eth/PanoInfinigen) | 카드 표시 약 58.9K 행; RGB 4096×2048·깊이·법선<br>실내 사진·평면도 정답 쌍 아님 | 데이터: BSD-3-Clause (indoor/nature만)<br>코드: 별도 확인<br>**기존 라이선스 조건에 맞는 후보** · [조건 근거](https://huggingface.co/datasets/prs-eth/PanoInfinigen#license) | 방 구조·바닥/벽/천장 기하 검증과 파노라마 깊이 학습<br>미터 깊이 정답은 있으나 방/문으로 완성된 벡터 평면도 정답 쌍은 아닙니다. 같은 저장소 urban은 비상업 조건입니다. 전체 저장소 약 1.89TB라 소량 샘플부터 필요합니다. |
| [PanoInfinigen / urban](https://huggingface.co/datasets/prs-eth/PanoInfinigen) | 도시 파노라마 하위 구성<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-NC-4.0<br>코드: 별도 확인<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://huggingface.co/datasets/prs-eth/PanoInfinigen#license) | 실외 데이터 계열 식별<br>indoor와 라이선스가 다릅니다. 저장소 전체를 BSD로 취급하지 않으며 내려받지 않습니다. |
| [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/) | 아파트 변형·90개 이상 물체 에셋·문 등 관절 구조<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-4.0<br>코드: 시뮬레이터 별도<br>**범위·조건 추가 확인** · [조건 근거](https://aihabitat.org/datasets/replica_cad/) | 문과 물체·가림이 포함된 구조 정답 생성<br>실매물 촬영 사진이 아니며 같은 아파트의 가구 배치 변형이 많습니다. 이름이 비슷한 Replica 원본과 라이선스가 다릅니다. |
| [Hypersim](https://github.com/apple-aiml-research/ml-hypersim) | 461개 장면; 공개 이미지 74,619장<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-SA-3.0 (이미지 데이터); 원본 메시 구매 별도<br>코드: 별도 LICENSE 확인 필요<br>**범위·조건 추가 확인** · [조건 근거](https://github.com/apple-aiml-research/ml-hypersim#the-hypersim-dataset) | 실사풍 RGB·깊이·카메라·의미 라벨 검증<br>원본 3D 메시가 무료 배포된다고 해석하지 않습니다. 전체 약 1.9TB. CC0라는 제3자 설명은 공식 설명과 맞지 않습니다. |
| [Structured3D](https://structured3d-dataset.org/) | 주택 설계 3,500개, RGB·구조 주석<br>사진·도면 대응 있음 | 데이터: Structured3D 별도 이용 약관·신청<br>코드: MIT<br>**범위·조건 추가 확인** · [조건 근거](https://structured3d-dataset.org/#license) | 정확한 구조와 합성 실내 이미지 비교<br>공식 사이트가 데이터 약관과 코드 MIT를 명확히 구분합니다. 약관 원문 조회 실패로 상업 허용 여부를 확정하지 않았습니다. |
| [InteriorNet](https://interiornet.org/) | 대규모 합성 RGB-D·관성·카메라 궤적<br>실내 사진·평면도 정답 쌍 아님 | 데이터: 이용 약관 동의 후 제공; 에셋·레이아웃 권리 별도<br>코드: 별도 확인<br>**범위·조건 추가 확인** · [조건 근거](https://interiornet.org/) | 조명·센서·장면 변화가 있는 실내 복원 검증<br>공식 안내는 Kujiale의 에셋/레이아웃/장면 권리를 별도 유보합니다. 공개 페이지를 무제한 상업 라이선스로 읽지 않습니다. |
| [ProcTHOR-10K](https://github.com/allenai/procthor-10k) | train 10,000 / val 1,000 / test 1,000개 집 JSON<br>실내 사진·평면도 정답 쌍 아님 | 데이터: Apache-2.0 (공식 집 JSON 배포 저장소)<br>코드: Apache-2.0<br>**기존 라이선스 조건에 맞는 후보** · [조건 근거](https://github.com/allenai/procthor-10k/blob/main/LICENSE) | 여러 방·문·창 구조 정답과 사진 렌더링 입력<br>완성된 실매물 사진/평면도 쌍이 아닙니다. 사진 생성에 쓰는 시뮬레이터·물체/재질 에셋의 조건과 실행 환경은 별도 확인해야 합니다. |
| [Habitat Synthetic Scenes Dataset](https://3dlg-hcvc.github.io/hssd/) | 211개 장면<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-NC-4.0 (hssd-hab)<br>코드: 데이터와 별도<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://huggingface.co/datasets/hssd/hssd-hab) | 다양한 실내 구조의 연구 사례<br>공식 데이터 카드가 비상업 조건입니다. 라이브러리 코드가 공개되었다고 데이터까지 허용하지 않습니다. |
| [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) | 설계 기반 집·방·가구 3D 구조<br>실내 사진·평면도 정답 쌍 아님 | 데이터: 별도 연구 목적 이용 약관 (조회한 2020-06-18판)<br>코드: 도구별로 별도<br>**범위·조건 추가 확인** · [조건 근거](https://kaldir.vc.in.tum.de/panoptic_reconstruction/front3d-tos.pdf) | 설계와 렌더링을 통한 사진/도면 생성 후보<br>조회한 약관은 연구 목적을 한정합니다. 현행 제공처와 추가 에셋 조건을 확인하지 못해 상업용 자료로 채택하지 않았습니다. |
| [Aria Synthetic Environments](https://www.projectaria.com/datasets/ase/license/) | 미확인<br>실내 사진·평면도 정답 쌍 아님 | 데이터: Project Aria ASE 라이선스: 비상업 연구 전용<br>코드: 별도 확인<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://www.projectaria.com/datasets/ase/license/) | 참고만<br>상업 제품 개발 금지 조항 |

## 실측 영상·스캔

| 자료·공식 출처 | 규모·자료 | 이용 조건·판단 | 이 앱에서의 용도와 한계 |
|---|---|---|---|
| [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) | 실내 RGB·깊이 영상·정답 카메라 궤적<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-4.0 (별도 표시 없는 자료)<br>코드: BSD-2-Clause<br>**범위·조건 추가 확인** · [조건 근거](https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license) | 실제 사무실 영상의 카메라/점군 복원 검사 — 2026-11-10 fr3/long_office_household(1,483,556,251바이트)로 사물 3D 위치 검증 수행(docs/object-recognition.md).<br>집 전체 방/문 평면도 정답이 아닙니다. 실측 깊이·궤적 평가를 도면 정확도와 구분해야 합니다. |
| [OpenLORIS-Scene](https://github.com/lifelong-robotic-vision/OpenLORIS-Scene) | 사무실·복도 등 시간 변화 RGB-D 장면<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-ND-4.0<br>코드: 수집·처리 코드 별도 권리<br>**범위·조건 추가 확인** · [조건 근거](https://github.com/lifelong-robotic-vision/OpenLORIS-Scene/blob/master/download.md) | 사무실·반복 방문·가구 변화의 복원 검사<br>공식 안내는 상업 목적 사용을 허용하지만 파생 데이터 배포를 제한합니다. 현 BSD/MIT/Apache 조건으로 자동 채택하지 않았습니다. |
| [ARKitScenes](https://github.com/apple-aiml-research/ARKitScenes) | LiDAR 기반 실제 실내 RGB-D·포즈·표면<br>실내 사진·평면도 정답 쌍 아님 | 데이터: Apple 별도 약관; 조건부 상업 조항 있음<br>코드: Apple 별도 약관<br>**범위·조건 추가 확인** · [조건 근거](https://github.com/apple-aiml-research/ARKitScenes/blob/main/LICENSE) | 실제 폰/태블릿 실내 복원 및 치수 검사<br>현재 LICENSE는 2024년 8월 이전 월간 이용자 수 조건의 상업 조항을 포함합니다. 예전 NC 표기만으로 전면 금지라 단정하지 않으며, 일반 MIT/Apache와 같다고도 하지 않습니다. |
| [MultiScan](https://github.com/smartscenes/multiscan) | 실내 RGB-D·재구성·객체/가동부 라벨<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-NC-4.0<br>코드: MIT<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://huggingface.co/datasets/3dlg-hcvc/MultiScan) | 문·가동 물체 인식의 연구 사례<br>MIT는 코드이며 데이터는 비상업 조건입니다. |
| [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/) | 실측 실내 스캔·고해상도 사진<br>실내 사진·평면도 정답 쌍 아님 | 데이터: 비상업 연구·교육 목적 이용 약관<br>코드: 데이터와 별도<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://scannetpp.mlsg.cit.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf) | 정밀 실내 복원의 연구 기준 사례<br>조회한 공식 이용 약관은 비상업 연구/교육을 한정합니다. |
| [ETH3D](https://eth3d.ethz.ch/) | 실내·실외 다중 시점 이미지/스테레오<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC-BY-NC-SA-4.0 (사이트의 기본 데이터 조건)<br>코드: 데이터와 별도<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://eth3d.ethz.ch/) | 다중 시점 재구성 연구 기준 사례<br>ETH3D 원본과 별도로 안내하는 TUM RGB-D의 CC-BY 조건을 혼동하지 않습니다. |
| [Replica 원본](https://github.com/facebookresearch/Replica-Dataset) | 실제 실내 스캔 기반 장면<br>실내 사진·평면도 정답 쌍 아님 | 데이터: Replica Dataset Research Terms; 비상업 연구 조건<br>코드: 데이터와 별도<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://github.com/facebookresearch/Replica-Dataset/blob/main/LICENSE) | 고품질 실내 스캔 연구 사례<br>ReplicaCAD의 CC-BY 조건을 Replica 원본에 소급하지 않습니다. |
| [SpatialLM Dataset/Testset](https://huggingface.co/datasets/manycore-research/SpatialLM-Testset) | 미확인<br>실내 사진·평면도 정답 쌍 아님 | 데이터: CC BY-NC 4.0<br>코드: 별도 확인<br>**비상업 등 기존 조건으로 제외** · [조건 근거](https://huggingface.co/datasets/manycore-research/SpatialLM-Testset) | 참고만<br>비상업 |
| [COCO 2017 val (상업 이용 가능 라이선스 실내 이미지 138장)](https://cocodataset.org/) | 이미지 138장 / 21,358,820바이트, 주석 CC BY 4.0<br>실내 사진·평면도 정답 쌍 아님 | 데이터: 주석 CC BY 4.0, 이미지는 CC BY·CC BY-SA·저작권 제한 없음만 선택<br>코드: 별도 확인<br>**범위·조건 추가 확인** · [조건 근거](https://cocodataset.org/#termsofuse) | 사물 검출기 평가 전용(학습·재배포 안 함)<br>해외 사진. 국내 매물 사진 성능이 아닙니다. |

## 국내 공공 경로

| 자료·공식 출처 | 규모·자료 | 이용 조건·판단 | 이 앱에서의 용도와 한계 |
|---|---|---|---|
| [AI Hub 실내공간 3D 종합](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=550&topMenu=100) | 시설 28개·방 703개, 총 실내 공간 731개<br>실내 사진·평면도 정답 쌍 아님 | 데이터: AI Hub 이용정책·신청 절차<br>코드: 모델/도구별 확인 필요<br>**범위·조건 추가 확인** · [조건 근거](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) | 국내 실제 실내 점군·사진·객체 라벨<br>주요 라벨은 3D 객체이며 방/문 벡터 평면도 정답으로 즉시 사용할 수 없습니다. 일반 개방 데이터의 영리 연구개발 허용과 원본 데이터 배포 조건을 구분해야 합니다. |
| [AI Hub 가상 실내 공간 3D 합성](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71765&topMenu=100) | 공간 200개, 사진 4,000장; 주거·상업·사무 포함<br>실내 사진·평면도 정답 쌍 아님 | 데이터: AI Hub 이용정책·신청 절차<br>코드: 모델/도구별 확인 필요<br>**범위·조건 추가 확인** · [조건 근거](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) | 국내 주택·사무실의 3D 구조/이미지와 의미 라벨<br>원본 구조는 FBX와 이미지/JSON 형식이며 평가용 벡터 방·문 정답 변환이 필요합니다. CUDA 기반 제공 모델은 설치하지 않습니다. |
| [AI Hub 건축 도면](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71465&topMenu=100) | 48,033장: 평면·단면·입면·구조도 포함<br>실내 사진·평면도 정답 쌍 아님 | 데이터: AI Hub 이용정책·신청 절차<br>코드: 모델/도구별 확인 필요<br>**범위·조건 추가 확인** · [조건 근거](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) | 한국 도면의 방·문·벽·텍스트 해석<br>48,033장을 모두 평면도로 세면 안 됩니다. 같은 집 실내 사진 쌍은 확인하지 못했습니다. 제3자 포털의 CC0 표기를 채택하지 않았습니다. |
| [LH 하나로 내집](https://hanaro.lh.or.kr/) | 공식 견본주택·단지별 분양 사이트<br>사진·도면 쌍 미확보 | 데이터: 사진·도면 각각 이용 조건 미확인<br>코드: 별도 확인<br>**범위·조건 추가 확인** · [조건 근거](https://hanaro.lh.or.kr/siteDetail/view?menuKey=60O0Um5kwF&siteDetailId=73) | 동일 견본주택/주택형의 사진·VR와 도면을 찾는 경로<br>일부 단지 페이지까지 확인했지만 실제 파일 쌍은 확보하지 못했습니다. 같은 주택형 도면을 실제 매물 호실 정답으로 간주하지 않습니다. |
| [제주개발공사 공공임대 평면도·조감도](https://www.jpdc.co.kr/open/open/open.htm) | 공공데이터포털 15089013 연결<br>실내 사진·평면도 정답 쌍 아님 | 데이터: 공공데이터포털 표기 '이용허락범위 제한 없음'(2026-11-10 확인); 공공누리 유형 표기 없음<br>코드: 별도 확인<br>**범위·조건 추가 확인** · [조건 근거](https://www.data.go.kr/data/15089013/fileData.do) | 공공 주택형 도면 정답 후보<br>조감도는 실내 촬영 사진과 다릅니다. 같은 호실 실내 사진은 미확보입니다. |
| [마이홈 공고 첨부 주택형 도면](https://www.myhome.go.kr/hws/com/fms/cvplFileDownload.do?atchFileId=17031a9c0c822&fileSn=1) | 원주문막1 46㎡ A/B·51㎡형 예시 확인<br>실내 사진·평면도 정답 쌍 아님 | 데이터: 첨부 도면의 상업 사용 조건 미확인<br>코드: 별도 확인<br>**범위·조건 추가 확인** · [조건 근거](https://www.myhome.go.kr/hws/com/fms/cvplFileDownload.do?atchFileId=17031a9c0c822&fileSn=1) | 치수가 표기된 공식 주택형 도면 후보<br>실제 호실 사진이 짝지어 확보된 자료가 아닙니다. |
| [한국토지주택공사 주택 평면도 현황](https://www.data.go.kr/data/15037046/fileData.do) | 평면도 281장 / 65개 단지 / 주택형·전용면적·공급면적 표 281행 (ZIP 356,725,466바이트)<br>실내 사진·평면도 정답 쌍 아님 | 데이터: 공공데이터포털 표기 '이용허락범위 제한 없음', 저작권 확보 표시 N, 근거 한국토지주택공사법 제8조<br>코드: 별도 확인<br>**범위·조건 추가 확인** · [조건 근거](https://www.data.go.kr/data/15037046/fileData.do) | 실제 국내 공공주택 평면도(mm 치수·실 이름 표기). 구조 정답 작성·합성 사진 렌더링 기반 후보 — 2026-11-10 평면도 2장으로 실·문 정답 변환과 자체 렌더 사진 검증 수행(docs/lh-plan-validation.md).<br>실내 사진 없음. 도면 이미지라 벡터화·치수 해석이 필요하며 자동 변환 결과는 사람 확인 전 정답으로 쓰지 않습니다. 저작권 확보 표시가 N이라 상업 학습 전 제공기관 확인 필요. |

## 실제 사용과 테스트 상태

외부 실제 주택의 사진/도면 정답 평가 **0건**, 이번 조사에서 원본 자료 다운로드 **0건**, 신경망 재학습 **미수행**. 기존 합성 사진에서의 검증 수치를 외부 자료 결과로 재사용하지 않았다.

Python urllib로 PanoInfinigen 공식 README에 소량 HTTPS 연결을 시도했지만 **WinError 10013**으로 연결이 거절됐다. 웹 조사 도구는 열람 가능하나 로컬 분석 프로세스는 파일을 받을 수 없는 상태다. 접근 제한을 우회하거나 별도 계정으로 신청하지 않았다.

AI Hub는 영리·비영리 연구개발 목적을 안내하지만 원본 제공·국외 반출·판매 등 조건과 신청 절차가 있다. 일반 데이터와 KETI 등 개별 정책을 섞거나, 제3자 카탈로그의 CC0 표기를 그대로 채택하지 않았다.

## 비교 자료를 구성할 때 지킬 구분

실내 사진과 별도 도면이 모두 있어야 같은 집의 사진 복원 성능을 측정할 수 있다. 도면만 있는 자료는 방 연결·구조 사전 학습용이고, RGB-D만 있으면 깊이/포즈/표면 기하 평가용이다. 합성 렌더링 결과는 실제 매물 사진의 일반화 평가와 분리한다.

C3 같은 실내·실외 혼합 장면은 외벽 사진을 실내 도면 근거로 세지 않아야 한다. Rent3D처럼 실외·공용부도 포함하는 자료는 입력 분류 검증에도 유용하지만 이용 권한 확인이 선행된다.

집 ID뿐 아니라 단지·주택형·반복 도면·합성 원본 장면으로 분할해야 한다. 같은 아파트의 가구 배치만 바꾼 변형을 서로 다른 집의 독립 평가로 세지 않는다. 추론에는 사진만, 정답 도면/포즈는 평가 때만 사용한 결과와 도면을 입력한 보조 복원을 구분한다.

원문 목록: [dataset-survey.json](dataset-survey.json). 재생성: `.venv/Scripts/python.exe -X utf8 tools/report-datasets.py`.
