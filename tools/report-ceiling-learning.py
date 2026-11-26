"""고정한 모델의 분할·가중치와 비교 점수를 재검사해 학습 보고서를 보강한다.

학습 당시 기록한 모델 해시·임계값은 덮어쓰지 않는다. 실행 폴더의 가중치나
임계값이 보고서와 다르면 실패한다.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacecapture.ceiling_records import LEARNING,refresh_training_report

parser=argparse.ArgumentParser(description="천장 모델 학습 보고서 재검사")
parser.add_argument("--run",default=str(LEARNING))
args=parser.parse_args()
try:
    report=refresh_training_report(args.run)
except RuntimeError as error:
    print(f"보고서 재검사 실패: {error}")
    sys.exit(1)
print(json.dumps({k:v for k,v in report.items() if k!="history"},ensure_ascii=False,indent=2))
