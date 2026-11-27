"""같은 사진으로 SfM을 여러 번 돌려(전역 SfM 실행 간 변동) 3D 점 연결 없는 묶음과 정답 오배치를 비교(평가 전용)."""
import sys, importlib.util
from pathlib import Path
import pycolmap
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools"))
import spacecapture.sfm as sfm
from spacecapture.group_check import shared_point_components, cross_link_consistency
spec = importlib.util.spec_from_file_location("e", ROOT/"tools/exp-sfm-a.py"); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
sfm.drop_unlinked_groups = lambda *a, **k: []   # 제외하지 않고 배치만 본다
name, runs = sys.argv[1], int(sys.argv[2])
sd = ROOT/"scenes"/f"hsa-{name}"
if not sd.exists(): sd = ROOT/"scenes"/f"hs-{name}"
gt = ROOT/"samples/external/hypersim"/name
for k in range(runs):
    out = ROOT/"artifacts/hypersim-a"/name/f"repeat{k}"
    rec = sfm.run_sfm(sd/"work/frames", out, log=lambda *_: None, unordered=True)
    ce = e.camera_error(rec, sd, gt); bad = set(ce["bad"]) if ce["all_rmse_cm"] > 5 else set()
    groups = shared_point_components(rec, 5)
    gs = []
    for g in groups[1:]:
        p, n, f = cross_link_consistency(rec, out/"database.db", g, groups[0])
        gs.append((len(g), len(g & bad), n, None if f is None else round(f, 2)))
    print(name, k, "reg", ce["registered"], "rmse", ce["all_rmse_cm"], "bad", len(bad), "groups(size,bad,matches,consistent)", gs, flush=True)
