"""Hypersim(Apple, CC BY-SA 3.0) 장면에서 평가에 필요한 파일만 HTTP Range로 받는다.

받는 것: cam_00의 색 사진(tonemap jpg), 카메라 위치·방향, 장면 단위(미터 배율), 사물별 3D 상자(위치·방향·크기),
그리고 사물 종류 판정용 의미·사물 분할(hdf5, 일부 프레임). 평가 전용이며 저장소에 넣지 않는다(samples/external).
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(Path(__file__).parent))
from remote_zip import open_remote

URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip"
OUT = ROOT/"samples/external/hypersim"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scenes", nargs="+"); ap.add_argument("--label-every", type=int, default=3)
    args = ap.parse_args()
    for scene in args.scenes:
        started = time.time(); dst = OUT/scene
        if (dst/"fetch.json").exists(): print(scene, "이미 있음"); continue
        try:
            z = open_remote(URL.format(scene)); names = z.namelist()
        except Exception as error:  # 없는 장면·접근 거부는 건너뜀
            print(scene, "건너뜀:", str(error)[:80]); continue
        want = [n for n in names if n.startswith(f"{scene}/_detail/") and (n.endswith(".csv") or "/cam_00/" in n or "/mesh/metadata_semantic_instance_bounding_box" in n)]
        want += [n for n in names if "/scene_cam_00_final_preview/" in n and n.endswith(".tonemap.jpg")]
        frames = sorted({n.split("/")[-1].split(".")[1] for n in names if "/scene_cam_00_final_preview/" in n and n.endswith(".tonemap.jpg")})
        labelled = frames[::args.label_every]
        want += [n for n in names if "/scene_cam_00_geometry_hdf5/" in n and n.split("/")[-1].split(".")[1] in labelled
                 and (n.endswith(".semantic.hdf5") or n.endswith(".semantic_instance.hdf5"))]
        total = 0
        for n in want:
            p = dst/n.split("/", 1)[1]; p.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(n); p.write_bytes(data); total += len(data)
        info = {"scene": scene, "files": len(want), "bytes": total, "frames": len(frames), "labelled_frames": len(labelled),
                "seconds": round(time.time()-started, 1), "source": URL.format(scene), "license": "CC BY-SA 3.0 (Hypersim dataset)"}
        (dst/"fetch.json").write_text(json.dumps(info, indent=1), encoding="utf-8"); print(info)


if __name__ == "__main__": main()
