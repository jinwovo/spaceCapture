"""전역 복원 카메라를 증분 복원 모델 분할(가장 큰 모델/다른 모델/미등록)로 나누고 정답 기준 오배치와 비교(평가 전용)."""
import sys, importlib.util
from pathlib import Path
import numpy as np, pycolmap
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools"))
pycolmap.logging.minloglevel = 3
spec = importlib.util.spec_from_file_location("e", ROOT/"tools/exp-sfm-a.py"); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
for name in sys.argv[1:]:
    sd = ROOT/"scenes"/f"hs-{name}"; gt = ROOT/"samples/external/hypersim"/name; d = ROOT/"artifacts/hypersim-a"/name
    base = pycolmap.Reconstruction(str(d/"base/sparse")); ce = e.camera_error(base, sd, gt)
    bad = set(ce["bad"]) if ce["all_rmse_cm"] > 5 else set()
    gnames = {i.name for i in base.images.values() if i.has_pose}
    models = sorted([pycolmap.Reconstruction(str(s)) for s in (d/"incremental/sparse_incremental").iterdir()], key=lambda r: -r.num_reg_images())
    parts = [{i.name for i in r.images.values() if i.has_pose} for r in models]
    main = parts[0] if parts else set(); others = set().union(*parts[1:]) if len(parts) > 1 else set()
    cat = {"main": gnames & main, "other": gnames & others, "none": gnames-main-others}
    print(name, "global", len(gnames), "bad", len(bad), {k: (len(v), len(v & bad)) for k, v in cat.items()}, "models", [len(p) for p in parts])
