"""작은 방 후보 모델을 검사한 뒤에만 설치한다. 실패하면 설치 모델(또는 기존 규칙)을 그대로 둔다.

검사: ① 가중치 SHA·임계값 지문 ② 모의 시험 분할에서 기존 규칙보다 IoU가 낮지 않고 거짓 공간이 크게 늘지 않음
③ 저장된 실제 사진 SfM 복원 전 장면에서 장면별 IoU 0.02 넘게 하락·문 재현율 하락 없음.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacecapture.property_evaluation import evaluate_scene_property
from spacecapture.space_model import MODEL,load_model

spec=importlib.util.spec_from_file_location("rsf",ROOT/"tools/real-sfm-features.py")
rsf=importlib.util.module_from_spec(spec); spec.loader.exec_module(rsf)
spec=importlib.util.spec_from_file_location("epg",ROOT/"tools/evaluate-property-graph.py")
epg=importlib.util.module_from_spec(spec); spec.loader.exec_module(epg)


def real_scene_check(model_path):
    rows={}
    for name,truth_dir in epg.CASES.items():
        if not (ROOT/"scenes"/name/"work/colmap/aligned").exists(): continue
        truth=json.loads((ROOT/truth_dir/"property.gt.json").read_text(encoding="utf-8"))
        out={}
        for label,model in (("rule",False),("model",model_path)):
            scene,graph,_=epg.recompute(name,model); scene["property_graph"]=graph
            r=evaluate_scene_property(scene,truth)
            out[label]={"iou":float(r["mean_space_iou"]),"spaces":r["predicted_space_candidates"],"truth":r["truth_spaces"],
                        "portal_recall":r["portal_recall"],"portal_precision":r["portal_precision"]}
        rows[name]=out
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--real-dataset",default=str(ROOT/"artifacts/space-learning/lh-v2-dataset.json"))
    parser.add_argument("--real-tag",default="v2")
    args=parser.parse_args()
    run=Path(args.run); run=(run if run.is_absolute() else ROOT/run).resolve(); path=run/"model/space-candidate-mlp-v1.npz"
    _,meta=load_model(path)
    report=json.loads((run/"result.json").read_text(encoding="utf-8"))
    if report["fingerprint"]!=meta["fingerprint"]: sys.exit("활성화 거절: 학습 보고서와 모델 지문이 달라요.")
    e2e=report["end_to_end_test"]; rule,model=e2e["rule"]["all"],e2e["model"]["all"]
    problems=[]
    if model["mean_iou"]<rule["mean_iou"]-.005: problems.append("모의 시험 IoU가 기존 규칙보다 낮아요.")
    if model["extra_spaces"]>rule["extra_spaces"]+max(2,.05*model["cases"]): problems.append("모의 시험에서 거짓 공간이 크게 늘었어요.")
    real=real_scene_check(path)
    real_test=None
    if Path(args.real_dataset).exists():  # 실제 렌더·SfM 시험 분할(문으로 연결된 집)
        rsf.configure(args.real_tag)
        data=json.loads(Path(args.real_dataset).read_text(encoding="utf-8")); infos=[]
        for h in data["houses"]:
            if h["split"]!="test" or not (ROOT/"scenes"/f"{rsf.PREFIX}{h['id']}"/"scene.json").exists(): continue
            info,_=rsf.house_rows(h["id"],"test",{"rule":(False,None),"model":(path,None)})
            if info.get("truth_connected") and info.get("usable"): infos.append(info)
        real_test={k:rsf.summarize(infos,k) for k in ("rule","model")}
        if real_test["model"] and real_test["rule"]:
            if real_test["model"]["mean_iou"]<real_test["rule"]["mean_iou"]-.005: problems.append("실제 SfM 시험 IoU 하락")
            if real_test["model"]["portal_recall"]<real_test["rule"]["portal_recall"]-.01: problems.append("실제 SfM 시험 문 재현율 하락")
        print("실제 SfM 시험:",json.dumps(real_test,ensure_ascii=False))
    for name,r in real.items():
        if r["model"]["iou"]<r["rule"]["iou"]-.02: problems.append(f"{name}: 실제 복원 IoU {r['rule']['iou']:.3f}→{r['model']['iou']:.3f}")
        if r["model"]["portal_recall"]<r["rule"]["portal_recall"]: problems.append(f"{name}: 문 재현율 하락")
    record={"run":str(run.relative_to(ROOT)),"model_sha256":meta["sha256"],"fingerprint":meta["fingerprint"],"threshold":meta["threshold"],
            "simulated_test":e2e,"real_reconstructions":real,"real_sfm_test":real_test,"problems":problems,"checked_at":dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    (run/"activation-check.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    for name,r in real.items(): print(f'{name:30s} 규칙 IoU {r["rule"]["iou"]:.3f} 공간 {r["rule"]["spaces"]}/{r["rule"]["truth"]} | 모델 IoU {r["model"]["iou"]:.3f} 공간 {r["model"]["spaces"]}/{r["model"]["truth"]}')
    if problems:
        print("활성화 거절:"); [print("  "+p) for p in problems]; sys.exit(1)
    if args.check_only:
        print("검사 통과(설치하지 않음)"); return
    if MODEL.exists():
        backup=ROOT/"artifacts/space-learning/previous-models"/dt.datetime.now().strftime("%Y%m%d-%H%M%S"); backup.mkdir(parents=True)
        for p in (MODEL,MODEL.with_suffix(".json")): shutil.copy2(p,backup/p.name)
    meta.update(enabled_by_default=True,activation=str((run/"activation-check.json").relative_to(ROOT)))
    tmp=MODEL.with_suffix(".tmp"); shutil.copyfile(path,tmp); tmp.replace(MODEL)
    MODEL.with_suffix(".json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    load_model(MODEL)
    print("검사를 통과한 작은 방 후보 모델을 설치했어요.")


if __name__=="__main__": main()
