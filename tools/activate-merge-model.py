"""문 합치기 모델을 검사한 뒤에만 설치한다. 실패하면 설치하지 않는다(합치지 않는 기존 동작 유지)."""
import argparse
import datetime as dt
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacecapture.property_evaluation import evaluate_scene_property
from spacecapture.space_model import MERGE_MODEL,PORTAL_ARCHITECTURE,PORTAL_FEATURES,_load_generic

spec=importlib.util.spec_from_file_location("epg",ROOT/"tools/evaluate-property-graph.py")
epg=importlib.util.module_from_spec(spec); spec.loader.exec_module(epg)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    run=Path(args.run); run=(run if run.is_absolute() else ROOT/run).resolve(); path=run/"model/portal-merge-mlp-v1.npz"
    _,meta=_load_generic(path,PORTAL_FEATURES,PORTAL_ARCHITECTURE,"portal-merge-v1")
    report=json.loads((run/"result.json").read_text(encoding="utf-8"))
    problems=[]
    sim=report["end_to_end_sim_test"]; keep,model=sim["keep"]["all"],sim["model"]["all"]
    if model["mean_iou"]<keep["mean_iou"]-.005: problems.append("모의 시험 IoU 하락")
    if model["missing_spaces"]>keep["missing_spaces"]+max(2,.05*model["cases"]): problems.append("모의 시험에서 누락 공간이 크게 늘었어요(진짜 방을 합침)")
    real=report.get("end_to_end_real_test")
    if real:
        rk,rm=real["keep"].get("connected") or real["keep"].get("all"),real["model"].get("connected") or real["model"].get("all")
        if rk and rm:
            if rm["mean_iou"]<rk["mean_iou"]-.005: problems.append("실제 SfM 시험(연결된 집) IoU 하락")
            if rm["portal_recall"]<rk["portal_recall"]-.01: problems.append("실제 SfM 시험(연결된 집) 문 재현율 하락")
    stored={}
    for name,truth_dir in epg.CASES.items():
        if not (ROOT/"scenes"/name/"work/colmap/aligned").exists(): continue
        truth=json.loads((ROOT/truth_dir/"property.gt.json").read_text(encoding="utf-8")); out={}
        for label,m in (("keep",False),("model",path)):
            scene,graph,_=epg.recompute(name,None,m); scene["property_graph"]=graph; r=evaluate_scene_property(scene,truth)
            out[label]={"iou":float(r["mean_space_iou"]),"spaces":r["predicted_space_candidates"],"truth":r["truth_spaces"],"portal_recall":r["portal_recall"]}
        stored[name]=out
        if out["model"]["iou"]<out["keep"]["iou"]-.02: problems.append(f"{name}: IoU {out['keep']['iou']:.3f}→{out['model']['iou']:.3f}")
        if out["model"]["portal_recall"]<out["keep"]["portal_recall"]: problems.append(f"{name}: 문 재현율 하락")
        print(f'{name:30s} 유지 IoU {out["keep"]["iou"]:.3f} 공간 {out["keep"]["spaces"]}/{out["keep"]["truth"]} | 모델 IoU {out["model"]["iou"]:.3f} 공간 {out["model"]["spaces"]}/{out["model"]["truth"]}')
    record={"run":str(run.relative_to(ROOT)),"fingerprint":meta["fingerprint"],"threshold":meta["threshold"],"simulated":sim,"real_test":real,
            "stored_reconstructions":stored,"problems":problems,"checked_at":dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    (run/"activation-check.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    if problems:
        print("활성화 거절:"); [print("  "+p) for p in problems]; sys.exit(1)
    if args.check_only: print("검사 통과(설치하지 않음)"); return
    if MERGE_MODEL.exists():
        backup=ROOT/"artifacts/merge-learning/previous-models"/dt.datetime.now().strftime("%Y%m%d-%H%M%S"); backup.mkdir(parents=True)
        for p in (MERGE_MODEL,MERGE_MODEL.with_suffix(".json")): shutil.copy2(p,backup/p.name)
    meta.update(enabled_by_default=True,activation=str((run/"activation-check.json").relative_to(ROOT)))
    tmp=MERGE_MODEL.with_suffix(".tmp"); shutil.copyfile(path,tmp); tmp.replace(MERGE_MODEL)
    MERGE_MODEL.with_suffix(".json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    _load_generic(MERGE_MODEL,PORTAL_FEATURES,PORTAL_ARCHITECTURE,"portal-merge-v1")
    print("검사를 통과한 문 합치기 모델을 설치했어요.")


if __name__=="__main__": main()
