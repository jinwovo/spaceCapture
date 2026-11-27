"""Hypersim 장면에서 SfM 변형(공유 카메라·증분 복원 등)을 정답 카메라와 비교하는 실험 도구(평가 전용).

추론에는 정답을 쓰지 않는다. 정답 카메라는 복원 뒤 닮음 정렬로 오차를 잴 때만 쓴다.
사용: tools/exp-sfm-a.py <variant> <scene...>   variant: base | shared | incremental | shared_incremental
"""
import json, sys, time, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools"))
from hypersim_gt import load
from spacecapture.sfm import run_sfm, cam_from_world
spec = importlib.util.spec_from_file_location("rsf", ROOT/"tools/real-sfm-features.py"); rsf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rsf)


def camera_error(rec, scene_dir, gt_dir):
    frames, _ = load(gt_dir)
    info = json.loads((scene_dir/"work/photos.json").read_text(encoding="utf-8"))
    orig = {p["name"]: p["original"].rsplit(".", 1)[0] for p in info["photos"]}
    names, C, T = [], [], []
    for im in rec.images.values():
        if not im.has_pose or orig.get(im.name) not in frames: continue
        M = cam_from_world(im).matrix(); C.append(-M[:, :3].T@M[:, 3]); T.append(frames[orig[im.name]][0]); names.append(im.name)
    C, T = np.array(C), np.array(T)
    s, R, t, keep, err = rsf.robust_alignment(T, C)
    return {"registered": len(names), "all_rmse_cm": round(float(np.sqrt(np.mean(err**2)))*100, 1),
            "inlier_rmse_cm": round(float(np.sqrt(np.mean(err[keep]**2)))*100, 1), "inliers": round(float(keep.mean()), 3),
            "bad": sorted(n for n, k in zip(names, keep) if not k)}


if __name__ == "__main__":
    variant, scenes = sys.argv[1], sys.argv[2:]
    kw = {"base": {}, "shared": {"shared_camera": True}, "incremental": {"mapper": "incremental"},
          "shared_incremental": {"shared_camera": True, "mapper": "incremental"}}[variant]
    out = ROOT/"artifacts/hypersim-a"; out.mkdir(parents=True, exist_ok=True)
    for name in scenes:
        sd = ROOT/"scenes"/f"hs-{name}"; t0 = time.time()
        rec = run_sfm(sd/"work/frames", out/name/variant, log=lambda *_: None, unordered=True, **kw)
        r = camera_error(rec, sd, ROOT/"samples/external/hypersim"/name); r["seconds"] = round(time.time()-t0, 1)
        print(name, variant, {k: v for k, v in r.items() if k != "bad"}, flush=True)
        (out/name/f"{variant}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
