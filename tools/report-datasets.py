"""Validate the source inventory and render a Korean research report (no download)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = {"paired": "사진·도면 쌍", "floorplan": "도면 구조", "synthetic": "합성 실내",
              "rgbd": "실측 영상·스캔", "korean": "국내 공공 경로"}
DECISIONS = {"eligible": "기존 라이선스 조건에 맞는 후보", "review": "범위·조건 추가 확인",
             "excluded": "비상업 등 기존 조건으로 제외", "unverified": "공식 원문 확인 보류"}


def main():
    catalog = json.loads((ROOT/"docs/dataset-survey.json").read_text(encoding="utf-8"))
    records = catalog["entries"]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("조사 항목 ID가 중복됩니다.")
    for record in records:
        if record["category"] not in CATEGORIES or record["decision"] not in DECISIONS:
            raise ValueError("조사 항목의 분류를 확인해 주세요.")
        if record["strict_policy_fit"] and record["decision"] != "eligible":
            raise ValueError("기존 조건 부합 표시와 채택 상태가 다릅니다.")
        for key in ("source", "license_source"):
            if urlparse(record[key]).scheme != "https":
                raise ValueError("공식 출처는 HTTPS 주소로 기록해 주세요.")
    counts = Counter(r["decision"] for r in records)
    text = ["# 사진·평면도 데이터 조사", "", f'확인일: {catalog["checked_on"]}. 검토 항목 **{len(records)}개**.', "",
            catalog["scope"], "", "자료의 존재, 이용 조건, 이 프로젝트의 입력 적합성, 실제 확보·평가는 별개다. "
            "아래 표는 검색·원문 확인 결과이며 다운로드/학습 완료 목록이 아니다.", "",
            "## 조사 결과", "", "| 구분 | 항목 수 |", "|---|---:|"]
    text.extend(f"| {label} | {counts[key]} |" for key, label in DECISIONS.items())
    text.extend(["", "기존 조건에 맞는 세 항목은 **PanoInfinigen indoor(BSD-3-Clause)**, "
                 "**ProcTHOR-10K 집 JSON(Apache-2.0)**, **Infinigen 생성기(BSD-3-Clause)**다. "
                 "셋 모두 실제 매물 사진과 완성 평면도 정답의 쌍을 제공하는 자료로 확정한 것은 아니다.", "",
                 "CC BY 자료는 비상업 자료와 구분해 보존했지만, 현재 프로젝트의 BSD/MIT/Apache 자동 허용 목록을 확대하거나 "
                 "외부 원본·코드·가중치를 포함하지 않았다. 조건부 상업 약관과 ShareAlike/ND 자료도 별도 검토 상태다.", ""])
    for category, title in CATEGORIES.items():
        text.extend([f"## {title}", "", "| 자료·공식 출처 | 규모·자료 | 이용 조건·판단 | 이 앱에서의 용도와 한계 |",
                     "|---|---|---|---|"])
        for r in records:
            if r["category"] != category:
                continue
            pair = "사진·도면 대응 있음" if r["paired_plan_photos"] is True else "사진·도면 쌍 미확보" if r["paired_plan_photos"] is None else "실내 사진·평면도 정답 쌍 아님"
            text.append(f'| [{r["name"]}]({r["source"]}) | {r["scale"]}<br>{pair} | '
                        f'데이터: {r["data_license"]}<br>코드: {r["code_license"] or "별도 확인"}<br>'
                        f'**{DECISIONS[r["decision"]]}** · [조건 근거]({r["license_source"]}) | '
                        f'{r["use"]}<br>{r["limitation"]} |')
        text.append("")
    text.extend(["## 실제 사용과 테스트 상태", "",
                 "외부 실제 주택의 사진/도면 정답 평가 **0건**, 이번 조사에서 원본 자료 다운로드 **0건**, "
                 "신경망 재학습 **미수행**. 기존 합성 사진에서의 검증 수치를 외부 자료 결과로 재사용하지 않았다.", "",
                 "Python urllib로 PanoInfinigen 공식 README에 소량 HTTPS 연결을 시도했지만 **WinError 10013**으로 "
                 "연결이 거절됐다. 웹 조사 도구는 열람 가능하나 로컬 분석 프로세스는 파일을 받을 수 없는 상태다. "
                 "접근 제한을 우회하거나 별도 계정으로 신청하지 않았다.", "",
                 "AI Hub는 영리·비영리 연구개발 목적을 안내하지만 원본 제공·국외 반출·판매 등 조건과 신청 절차가 있다. "
                 "일반 데이터와 KETI 등 개별 정책을 섞거나, 제3자 카탈로그의 CC0 표기를 그대로 채택하지 않았다.", "",
                 "## 비교 자료를 구성할 때 지킬 구분", "",
                 "실내 사진과 별도 도면이 모두 있어야 같은 집의 사진 복원 성능을 측정할 수 있다. "
                 "도면만 있는 자료는 방 연결·구조 사전 학습용이고, RGB-D만 있으면 깊이/포즈/표면 기하 평가용이다. "
                 "합성 렌더링 결과는 실제 매물 사진의 일반화 평가와 분리한다.", "",
                 "C3 같은 실내·실외 혼합 장면은 외벽 사진을 실내 도면 근거로 세지 않아야 한다. "
                 "Rent3D처럼 실외·공용부도 포함하는 자료는 입력 분류 검증에도 유용하지만 이용 권한 확인이 선행된다.", "",
                 "집 ID뿐 아니라 단지·주택형·반복 도면·합성 원본 장면으로 분할해야 한다. "
                 "같은 아파트의 가구 배치만 바꾼 변형을 서로 다른 집의 독립 평가로 세지 않는다. "
                 "추론에는 사진만, 정답 도면/포즈는 평가 때만 사용한 결과와 도면을 입력한 보조 복원을 구분한다.", "",
                 "원문 목록: [dataset-survey.json](dataset-survey.json). 재생성: "
                 "`.venv/Scripts/python.exe -X utf8 tools/report-datasets.py`.", ""])
    (ROOT/"docs/dataset-survey.md").write_text("\n".join(text), encoding="utf-8")
    output = ROOT/"artifacts/dataset-survey"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"checked_on": catalog["checked_on"], "reviewed_entries": len(records),
               "counts": dict(counts), "categories": dict(Counter(r["category"] for r in records)),
               "strict_candidates": [{"id": r["id"], "kind": r["kind"], "paired_plan_photos": r["paired_plan_photos"]}
                                     for r in records if r["strict_policy_fit"]],
               "external_cases_evaluated": 0, "files_downloaded": 0,
               "neural_training_performed": False, "download_probe": catalog["download_probe"]}
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"조사 항목 {len(records)}개 · 출처/분류/중복 검사 통과")
    print(" / ".join(f"{label} {counts[key]}개" for key, label in DECISIONS.items()))
    print("외부 정답 평가 0건 · 원본 다운로드 0건 · 조사 보고서 저장 완료")


if __name__ == "__main__":
    main()
