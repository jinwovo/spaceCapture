"""Command line: python -m spacecapture <command> ..."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# numpy work here is tiny; a single BLAS thread also silences an OpenBLAS warning on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parent.parent


def _quiet_colmap(verbose: bool) -> None:
    try:
        import pycolmap

        pycolmap.logging.minloglevel = 0 if verbose else 3
    except Exception:  # older/newer pycolmap without this switch: just keep its logs
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m spacecapture", description="방 영상 → 실측 3D 공간")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = argparse.ArgumentParser(add_help=False)
    pr.add_argument("--name", help="장면 이름 (기본: 파일 이름)")
    pr.add_argument("--marker-mm", type=float, default=150.0,
                    help="인쇄한 마커의 검은 사각형 한 변 길이 mm (기본 150, 0이면 마커 사용 안 함)")
    pr.add_argument("--marker-dict", default="DICT_4X4_50")
    pr.add_argument("--camera-height", type=float, default=1.45,
                    help="마커가 없을 때 가정할 촬영 높이 m (기본 1.45)")
    pr.add_argument("--focal-px", type=float, help="초점거리를 알면 px 단위로 지정")
    pr.add_argument("--no-splat", action="store_true", help="실사 3D(Brush) 학습 건너뛰기")
    pr.add_argument("--steps", type=int, default=15000, help="Brush 학습 스텝 (기본 15000)")
    pr.add_argument("--max-resolution", type=int, default=1280, help="Brush 영상의 최대 변 길이 px (기본 1280)")
    pr.add_argument("--max-splats", type=int, default=500000, help="최대 스플랫 수 (기본 500000)")
    pr.add_argument("--verbose", action="store_true", help="COLMAP 로그까지 모두 출력")

    vp = sub.add_parser("process", parents=[pr], help="영상을 3D 장면으로 만들기")
    vp.add_argument("video", type=Path)
    vp.add_argument("--fps", type=float, default=5.0, help="초당 사용할 프레임 수 (기본 5: 흔들린 폰 영상에서도 빈 벽 구간이 이어지게)")
    vp.add_argument("--max-frames", type=int, default=800)
    vp.add_argument("--photos", type=Path, dest="extra_photos",
                    help="영상과 함께 쓸 사진 폴더(덜 찍힌 방·구석 보충). 영상과 사진을 한 좌표계로 함께 복원")
    pp = sub.add_parser("photos", parents=[pr], help="연결되는 공간의 사진 폴더 → 추정 3D와 2D 도면")
    pp.add_argument("folder", type=Path, help="서로 겹치는 사진이 들어 있는 폴더")
    pp.add_argument("--max-photos", type=int, default=150, help="사진 수 제한 (기본 150, 전체 쌍 매칭)")

    sp = sub.add_parser("splat", help="처리된 장면에 실사 3D 학습(Brush)만 다시 실행")
    sp.add_argument("name")
    sp.add_argument("--steps", type=int, default=15000)
    sp.add_argument("--max-resolution", type=int, default=1280, help="Brush 영상의 최대 변 길이 px (780M 시험: 640)")
    sp.add_argument("--max-splats", type=int, default=500000, help="최대 스플랫 수")
    sp.add_argument("--verbose", action="store_true")

    sy = sub.add_parser("synth", help="정답 치수를 아는 테스트용 합성 방 영상 만들기")
    sy.add_argument("--out", type=Path, default=REPO / "samples" / "synth_room.mp4")
    sy.add_argument("--seed", type=int, default=7)

    mk = sub.add_parser("marker", help="크기 기준 마커 PDF(A4) 만들기")
    mk.add_argument("--out", type=Path, default=REPO / "marker_150mm.pdf")
    mk.add_argument("--size-mm", type=float, default=150.0)
    mk.add_argument("--id", type=int, default=0)

    ev = sub.add_parser("evaluate", help="합성 영상 결과를 정답과 비교")
    ev.add_argument("name")
    ev.add_argument("--gt", type=Path, default=REPO / "samples" / "synth_room.gt.json")

    sub.add_parser("list", help="처리된 장면 목록")
    check = sub.add_parser("inspect", help="기존 장면의 사진 연결·벽 구조 검사 (재학습 없음)")
    check.add_argument("name")
    app = sub.add_parser("serve", help="사진·영상 업로드와 분석 앱 실행 (로컬)")
    app.add_argument("--port", type=int, default=8765)
    house = sub.add_parser("synth-property", help="직접 생성한 여러 공간 사진·문·정답 도면 쌍 만들기")
    house.add_argument("--out", type=Path, default=REPO/"samples/owned-property")
    house.add_argument("--kind", choices=["two_rooms","rooms_corridor"], default="two_rooms")
    house.add_argument("--seed", type=int, default=0)
    house.add_argument("--no-photos", action="store_true", help="빠른 기하 검증용으로 사진 렌더링 생략")
    property_eval=sub.add_parser("evaluate-property",help="집의 공간·문 연결 후보를 별도 정답 도면과 비교")
    property_eval.add_argument("name")
    property_eval.add_argument("--gt",type=Path,required=True)
    property_eval.add_argument("--anchors",type=Path,help="별도 도면과 복원 벽 모서리의 대응점 JSON (미터)")
    property_eval.add_argument("--out",type=Path,help="비교 JSON·SVG를 저장할 폴더")
    pairs=sub.add_parser("training-pairs",help="허용된 사진·도면 쌍을 집 단위로 나눈 학습 예제로 변환 (모델 학습 아님)")
    pairs.add_argument("properties",type=Path,nargs="+")
    pairs.add_argument("--out",type=Path,required=True)

    args = p.parse_args(argv)

    if args.cmd == "training-pairs":
        import json
        from .training_pairs import export_training_pairs
        result=export_training_pairs(args.properties,args.out)
        print(json.dumps(result["splits"],ensure_ascii=False,indent=2))
    elif args.cmd == "evaluate-property":
        import json
        from .pipeline import SCENES,load_scene
        from .property_evaluation import evaluate_scene_property
        anchors=json.loads(args.anchors.read_text(encoding="utf-8")) if args.anchors else None
        result=evaluate_scene_property(load_scene(SCENES/args.name),json.loads(args.gt.read_text(encoding="utf-8")),
                                       anchors=anchors,output=args.out)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.cmd == "synth-property":
        from .synth_property import generate_property

        generate_property(args.out,args.kind,args.seed,render_photos=not args.no_photos)
    elif args.cmd == "serve":
        from .server import serve

        serve(args.port)
    elif args.cmd in ("process", "photos"):
        _quiet_colmap(args.verbose)
        from .pipeline import process

        is_photos = args.cmd == "photos"
        process(args.folder if is_photos else args.video, name=args.name,
                marker_mm=args.marker_mm or None, marker_dict=args.marker_dict,
                target_fps=getattr(args, "fps", 3.0),
                max_frames=args.max_photos if is_photos else args.max_frames, camera_height=args.camera_height,
                focal_px=args.focal_px, splat=not args.no_splat, steps=args.steps,
                max_resolution=args.max_resolution, max_splats=args.max_splats, photos=is_photos,
                extra_photos=None if is_photos else args.extra_photos)
    elif args.cmd == "inspect":
        from .pipeline import SCENES, inspect_scene

        scene = inspect_scene(SCENES / args.name)
        print(scene["assessment"]["message"])
        print(scene.get("connectivity", {}).get("message", "사진 연결 근거 없음"))
    elif args.cmd == "splat":
        _quiet_colmap(args.verbose)
        from .pipeline import SCENES, Log, run_splat

        scene_dir = SCENES / args.name
        if not (scene_dir / "scene.json").exists():
            print(f"장면을 찾을 수 없어요: {scene_dir}")
            return 1
        run_splat(scene_dir, steps=args.steps, max_resolution=args.max_resolution,
                  max_splats=args.max_splats, log=Log(scene_dir / "process.log"))
    elif args.cmd == "synth":
        from .synth import generate

        out = generate(args.out, seed=args.seed)
        print(f"합성 영상: {out}\n정답 치수: {out.with_suffix('.gt.json')}")
    elif args.cmd == "marker":
        from .marker import make_marker_pdf

        out = make_marker_pdf(args.out, size_mm=args.size_mm, marker_id=args.id)
        print(f"마커 PDF: {out}  (반드시 '실제 크기/100%'로 인쇄하고, 자로 100mm 눈금을 확인하세요)")
    elif args.cmd == "evaluate":
        from .pipeline import SCENES, evaluate

        res = evaluate(SCENES / args.name, args.gt)
        print(f"스케일 방식: {res['scale_method']}")
        for row in res["rows"]:
            print(f"  {row['what']}: 측정 {row['measured_m']:.3f}m / 정답 {row['truth_m']:.3f}m → 오차 {row['error_cm']:+.1f}cm")
        print(f"최대 오차 {res['max_abs_error_cm']:.1f}cm")
        if "trajectory" in res:
            t = res["trajectory"]
            print(f"카메라 궤적 오차(RMSE) {t['rmse_cm']:.2f}cm ({t['cameras']}대), 크기 배율 {t['scale_vs_truth']:.4f} (1이면 정확)")
    elif args.cmd == "list":
        from .pipeline import SCENES, load_scene

        for d in sorted(SCENES.glob("*/scene.json")):
            s = load_scene(d.parent)
            r = s["room"]
            print(f"{s['name']:20s} {r['width']:.2f} × {r['depth']:.2f} × {r['height']:.2f}m  "
                  f"스케일:{s['scale']['method']}  실사3D:{'있음' if s['files'].get('splat') else '없음'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"처리 실패: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
