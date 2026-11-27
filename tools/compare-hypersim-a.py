"""hs-(기존)·hsa-(묶음 검사 적용) 장면의 정답 카메라 정렬 오차 비교(평가 전용)."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools"))
spec = importlib.util.spec_from_file_location("rsf", ROOT/"tools/real-sfm-features.py"); rsf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rsf)
from hypersim_gt import load
for name in sys.argv[1:]:
    fr, _ = load(ROOT/"samples/external/hypersim"/name); row = [name]
    for pre in ("hs", "hsa"):
        f = ROOT/"scenes"/f"{pre}-{name}"/"scene.json"
        if not f.exists(): row.append(f"{pre}: -"); continue
        s = json.loads(f.read_text(encoding="utf-8"))
        cams = [c for c in s["cameras"] if c["source_photo"].rsplit(".", 1)[0] in fr]
        T = np.array([fr[c["source_photo"].rsplit(".", 1)[0]][0] for c in cams]); M = np.array([c["position"] for c in cams])
        _, _, _, keep, err = rsf.robust_alignment(T, M)
        row.append(f"{pre}: reg {s['quality']['registered']} rmse {np.sqrt(np.mean(err**2))*100:.1f}cm inliers {keep.mean():.2f}")
    print(" | ".join(row))
