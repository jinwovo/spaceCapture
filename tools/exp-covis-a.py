"""전역 복원에서 3D 점 공유(쌍별 공유 점 ≥ T) 연결 덩어리 중 가장 큰 것만 남길 때의 정답 오차(평가 전용)."""
import sys, importlib.util
from pathlib import Path
import numpy as np, pycolmap
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools"))
pycolmap.logging.minloglevel = 3
from spacecapture.group_check import shared_point_components
spec = importlib.util.spec_from_file_location("e", ROOT/"tools/exp-sfm-a.py"); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
Ts = [int(x) for x in sys.argv[1].split(",")]
for name in sys.argv[2:]:
    sd = ROOT/"scenes"/f"hs-{name}"; gt = ROOT/"samples/external/hypersim"/name
    a = ROOT/"artifacts/hypersim-a"/name/"base/sparse"
    rec = pycolmap.Reconstruction(str(a if a.exists() else sd/"work/colmap/sparse"))
    ce = e.camera_error(rec, sd, gt); bad = set(ce["bad"]) if ce["all_rmse_cm"] > 5 else set()
    out = [f"{name} base reg {ce['registered']} rmse {ce['all_rmse_cm']}"]
    for T in Ts:
        comps = shared_point_components(rec, T)
        drop = set().union(*comps[1:]) if len(comps) > 1 else set()
        for n in drop: rec2 = None
        r2 = pycolmap.Reconstruction(rec)
        for im in list(r2.images.values()):
            if im.name in drop and im.has_pose: r2.deregister_frame(im.frame_id)
        c2 = e.camera_error(r2, sd, gt)
        out.append(f"T{T}: drop {len(drop)} (bad {len(drop & bad)}) -> reg {c2['registered']} rmse {c2['all_rmse_cm']}")
    print(" | ".join(out), flush=True)
