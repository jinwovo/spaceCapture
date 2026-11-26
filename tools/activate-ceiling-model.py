"""학습 실행 폴더의 기록을 모두 검사한 뒤에만 천장 보조 모델을 설치·활성화한다.

검사: 가중치 SHA·임계값·특징 버전/순서를 묶은 추론 설정 지문이 모델 기록,
학습 보고서, 사진 회귀 보고서에서 같은지, 학습·검증·시험 자료 해시와 사진
회귀 입력 장면·정답 파일 해시가 그대로인지, 현재 설정으로 다시 계산한 점수가
보고서와 같은지. 하나라도 다르면 현재 설치 모델을 바꾸지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacecapture.ceiling_records import LEARNING,activate,check_activation

parser=argparse.ArgumentParser(description="천장 모델 활성화 검사·설치")
parser.add_argument("--run",default=str(LEARNING),help="학습 실행 폴더 (기본: 현재 v1 기록 폴더)")
parser.add_argument("--check-only",action="store_true",help="검사만 하고 설치하지 않음")
args=parser.parse_args()
try:
    record=check_activation(args.run) if args.check_only else activate(args.run)
except RuntimeError as error:
    print(f"활성화 거절: {error}")
    sys.exit(1)
print(json.dumps({k:record[k] for k in ("inference_fingerprint","model_sha256","threshold","test","actual_sfm_cases")},ensure_ascii=False,indent=2))
print("검사를 통과했어요." if args.check_only else
      "합성 구조 검증을 통과한 천장 후보 보조 모델을 활성화했어요. 실사 정확도는 아직 검증하지 않았어요.")
