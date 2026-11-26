"""LH 평면도 구조로 '집 안을 걸으며 찍은 영상'을 합성한다(집 전체 영상 → 3D 경로 검증용).

사람이 촬영하듯 큰 공간에서 시작해 문을 지나 방마다 들어가 벽을 둘러보고(방 안 작은 원을 돌며 바깥을 봄),
되돌아 나와 다음 방으로 간다. 정답(방·문·카메라)은 property.gt.json에 따로 두고 추론에는 쓰지 않는다.
출력: <out>/walkthrough.mp4 (fps 3, 프레임 = 렌더 1장), <out>/property.gt.json, <out>/photos/
"""
import argparse
import heapq
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from spacecapture.plan_image import auto_meters_per_pixel, render_config
from spacecapture.synth import look_rotation
from spacecapture.synth_property import generate_property

LH = ROOT/"samples/external/lh-floorplans"
CELL = .05
EYE = 1.45


def walkable(config, seg):
    """방 다각형 + 문 상자 → 걸을 수 있는 격자(벽에서 0.25m 이상)."""
    allp = np.vstack([np.array(s["polygon_xz"]) for s in config["spaces"]])
    low = allp.min(0)-.5
    shape = (int(np.ceil((allp.max(0)[1]-low[1]+.5)/CELL))+1, int(np.ceil((allp.max(0)[0]-low[0]+.5)/CELL))+1)
    free = np.zeros(shape, np.uint8); rooms = {}
    for s in config["spaces"]:
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round((np.array(s["polygon_xz"])-low)/CELL).astype(np.int32)], 1)
        rooms[s["id"]] = m; free |= m
    ids = {s["id"] for s in config["spaces"]}
    for p in seg["portals"]:
        if not set(p["spaces"]) <= ids: continue
        x0, z0, x1, z1 = p["box_xz"]
        a = np.round((np.array([x0, z0])-low)/CELL).astype(int); b = np.round((np.array([x1, z1])-low)/CELL).astype(int)
        cv2.rectangle(free, tuple(a-1), tuple(b+1), 1, -1)
    clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5)*CELL
    return low, free, clearance, rooms


def shortest(clearance, start, goal, min_clear=.25):
    """8방향 A*. 벽에서 가까울수록 비용을 더해 가운데로 걷게 한다."""
    ok = clearance >= min_clear
    h, w = ok.shape
    start, goal = tuple(start), tuple(goal)
    if not ok[start[1], start[0]] or not ok[goal[1], goal[0]]:
        raise ValueError("경로 시작·끝이 걸을 수 없는 곳이에요.")
    dist = {start: 0.}; prev = {}; q = [(0., start)]
    while q:
        _, cur = heapq.heappop(q)
        if cur == goal: break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = cur[0]+dx, cur[1]+dy
            if not (0 <= nx < w and 0 <= ny < h and ok[ny, nx]): continue
            nd = dist[cur]+np.hypot(dx, dy)*(1+.6/max(clearance[ny, nx], .05))
            if nd < dist.get((nx, ny), 1e18):
                dist[(nx, ny)] = nd; prev[(nx, ny)] = cur
                heapq.heappush(q, (nd+np.hypot(goal[0]-nx, goal[1]-ny), (nx, ny)))
    if goal not in prev and goal != start:
        raise ValueError("경로를 찾지 못했어요.")
    path = [goal]
    while path[-1] != start: path.append(prev[path[-1]])
    return np.array(path[::-1], float)


def resample(points, step):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6: return points[:1]
    t = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1]/step))+1))
    return np.stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])], 1)


def pose(xz, yaw, pitch=-.17):
    forward = np.array([np.cos(yaw), np.sin(pitch), np.sin(yaw)])
    return np.array([xz[0], EYE, xz[1]]), look_rotation(forward)


def tour(config, seg, step=.22, turn=np.radians(12), orbit_frames=30):
    low, free, clearance, rooms = walkable(config, seg)
    to_cell = lambda xz: np.round((np.asarray(xz)-low)/CELL).astype(int)
    to_xz = lambda c: low+np.asarray(c, float)*CELL
    centers, radius = {}, {}
    for sid, m in rooms.items():
        d = cv2.distanceTransform(m, cv2.DIST_L2, 5)*CELL*(clearance >= .25)
        cy, cx = np.unravel_index(np.argmax(d), d.shape); centers[sid] = (cx, cy)
        radius[sid] = float(np.clip(.4*d.max(), .2, .7))
    adj = {s: set() for s in rooms}
    for p in config["portals"]:
        a, b = p["spaces"]
        if a in adj and b in adj: adj[a].add(b); adj[b].add(a)
    start = max(rooms, key=lambda s: rooms[s].sum())
    order, seen = [], set()
    def dfs(s):  # 방문 순서: 들어갔다가 되돌아 나오는 경로
        seen.add(s); order.append(s)
        for n in sorted(adj[s], key=lambda n: rooms[n].sum()):
            if n not in seen: dfs(n); order.append(s)
    dfs(start)
    frames, labels = [], []
    yaw = 0.
    def add(xz, y, sid):
        frames.append(pose(xz, y)); labels.append(sid)
    def turn_to(xz, target, sid):
        nonlocal yaw
        diff = (target-yaw+np.pi) % (2*np.pi)-np.pi
        for k in range(1, int(abs(diff)//turn)+1):
            add(xz, yaw+np.sign(diff)*turn*k, sid)
        yaw = target
    def orbit(sid):
        nonlocal yaw
        c = to_xz(centers[sid]); r = radius[sid]
        for k in range(orbit_frames+1):  # 원을 돌며 바깥(벽)을 본다
            a = yaw+2*np.pi*k/orbit_frames
            add(c+r*np.array([np.cos(a), np.sin(a)]), a, sid)
    orbited = {start}
    here = to_xz(centers[start]); orbit(start); here = to_xz(centers[start])+radius[start]*np.array([np.cos(yaw), np.sin(yaw)])
    for prev_sid, sid in zip(order, order[1:]):
        path = shortest(clearance, to_cell(here), centers[sid])
        pts = resample(np.array([to_xz(p) for p in path]), step)
        for a, b in zip(pts, pts[1:]):
            heading = float(np.arctan2(b[1]-a[1], b[0]-a[0]))
            turn_to(a, heading, prev_sid)
            owner = next((s for s, m in rooms.items() if m[tuple(to_cell(b)[::-1])]), prev_sid)
            add(b, heading, owner)
        here = pts[-1]
        if sid not in orbited:  # 처음 들어간 방만 둘러본다(되돌아 지나갈 때는 걷기만)
            orbited.add(sid); orbit(sid)
            here = to_xz(centers[sid])+radius[sid]*np.array([np.cos(yaw), np.sin(yaw)])
    return [(C, R, sid) for (C, R), sid in zip(frames, labels)], order


def smooth_noise(rng, n, scale, corr=.9):
    """천천히 변하는 잡음(AR(1)): 손떨림·노출 변화처럼 프레임 사이에 이어지는 흔들림."""
    out = np.zeros(n); value = 0.
    for i in range(n):
        value = corr*value + np.sqrt(1-corr**2)*rng.normal(0, scale); out[i] = value
    return out


def interpolate(poses, sub):
    """초당 3포즈 경로를 sub배로 촘촘하게(위치 선형, 회전 Rodrigues 보간)."""
    if sub <= 1:
        return poses
    out = []
    for (Ca, Ra, sa), (Cb, Rb, sb) in zip(poses, poses[1:]):
        rvec, _ = cv2.Rodrigues(Ra.T @ Rb)
        for k in range(sub):
            t = k/sub
            Rt, _ = cv2.Rodrigues(rvec*t)
            out.append((Ca+(Cb-Ca)*t, Ra @ Rt, sa if t < .5 else sb))
    out.append(poses[-1])
    return out


def handheld(poses, seed, fps=3.):
    """손으로 들고 걸을 때의 흔들림(정답 포즈도 이 값). 느린 흔들림은 fps와 무관하게 같은 시간 규모,
    fps가 높으면 손 떨림(약 8Hz, ±0.3°)을 더한다 → 프레임마다 번짐 정도가 달라진다."""
    rng = np.random.default_rng(seed+777); n = len(poses); s = 3./fps
    dx, dz = smooth_noise(rng, n, .025, .9**s), smooth_noise(rng, n, .025, .9**s)
    dy = smooth_noise(rng, n, .06, .97**s)                   # 촬영 높이 1.45 ± 약 0.1m
    roll, pitch, yaw = (smooth_noise(rng, n, np.radians(a), .9**s) for a in (2.5, 3., 2.))
    if fps > 3:
        for arr in (roll, pitch, yaw):
            arr += smooth_noise(rng, n, np.radians(.3), .3)
    out = []
    for i, (C, R, sid) in enumerate(poses):
        C = C + np.array([dx[i], dy[i], dz[i]])
        cr, sr = np.cos(roll[i]), np.sin(roll[i]); cp, sp = np.cos(pitch[i]), np.sin(pitch[i]); cy, sy = np.cos(yaw[i]), np.sin(yaw[i])
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]); Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        out.append((C, R @ Ry @ Rx @ Rz, sid))  # 카메라 좌표계에서 작은 회전
    return out


def phone_effects(names, poses, seed, focal, effects=("blur", "exposure", "noise", "distortion"), fps=3., shutter=1/50):
    """폰 영상 화질 효과를 한 장씩 만들어 돌려준다(메모리 절약): 자동 노출·화이트밸런스, 움직임 번짐,
    센서 노이즈, 비네팅, 렌즈 왜곡. 번짐 길이 = 셔터(1/50초) 동안의 회전·이동 속도 × 초점거리."""
    rng = np.random.default_rng(seed+999); n = len(names); s = 3./fps
    gain = np.exp(smooth_noise(rng, n, .1, .95**s))          # 노출 약 ±10%
    wb = np.exp(np.stack([smooth_noise(rng, n, .025, .97**s) for _ in range(3)], 1))
    first = cv2.imread(str(names[0])); h, w = first.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx-w/2)**2+(yy-h/2)**2)/((w/2)**2+(h/2)**2)
    vignette = (1-.28*r2)[..., None]
    k1 = -.06                                                # 약한 술통형 왜곡
    xn, yn = (xx-w/2)/focal, (yy-h/2)/focal
    factor = 1+k1*(xn**2+yn**2)
    map_x, map_y = (xn*factor*focal+w/2).astype(np.float32), (yn*factor*focal+h/2).astype(np.float32)
    for i, name in enumerate(names):
        a, b = poses[max(0, i-1)], poses[min(n-1, i+1)]
        dt = (min(n-1, i+1)-max(0, i-1))/fps or 1/fps
        rot = np.arccos(np.clip((np.trace(a[1].T@b[1])-1)/2, -1, 1))/dt   # rad/s
        speed = np.linalg.norm(b[0]-a[0])/dt                              # m/s
        length = int(np.clip((rot*focal + speed*focal/2.5)*shutter, 1, 25))  # 셔터 동안 움직인 픽셀(벽까지 약 2.5m)
        f = cv2.imread(str(name)).astype(np.float32)
        if length > 1 and "blur" in effects:
            kernel = np.zeros((length, length), np.float32)
            angle = rng.uniform(0, np.pi)
            cx = cy = (length-1)/2
            for t in np.linspace(-.5, .5, length*2):
                kernel[int(round(cy+t*(length-1)*np.sin(angle))), int(round(cx+t*(length-1)*np.cos(angle)))] = 1
            f = cv2.filter2D(f, -1, kernel/kernel.sum())
        if "exposure" in effects:
            f = f*gain[i]*wb[i]*vignette
        if "distortion" in effects:
            f = cv2.remap(f, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        noise = rng.normal(0, 3.5, f.shape)
        if "noise" in effects:
            f += noise
        yield np.clip(f, 0, 255).astype(np.uint8), length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=int, default=227, help="LH 평면도 번호(manifest 순서)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=float, default=3.)
    parser.add_argument("--style", default="interior", choices=["interior", "unique_noise"],
                        help="interior: 나무 마루·벽지·흰 천장(실제 집 느낌), unique_noise: 기존 알록달록 노이즈")
    parser.add_argument("--phone", action="store_true",
                        help="실제 폰 촬영 효과: 손떨림·높이·기울기 흔들림, 자동 노출·화이트밸런스, 움직임 번짐, 노이즈, 렌즈 왜곡")
    args = parser.parse_args()
    out = Path(args.out or ROOT/f"artifacts/walkthrough/lh-{args.plan:04d}"); out.mkdir(parents=True, exist_ok=True)
    plans = json.loads((LH/"manifest.json").read_text(encoding="utf-8"))["plans"]
    image = cv2.imdecode(np.fromfile(LH/plans[args.plan]["file"], np.uint8), cv2.IMREAD_COLOR)
    started = time.perf_counter()
    _, seg = auto_meters_per_pixel(image)
    config, _ = render_config(seg, seed=args.plan, room_views=12)
    poses, order = tour(config, seg)
    poses = interpolate(poses, max(1, int(round(args.fps/3))))   # 30fps면 3fps 경로를 10배로
    if args.phone:
        poses = handheld(poses, args.plan, args.fps)
    config["poses"] = poses
    config["texture_style"] = args.style
    print(f"경로: 방 순서 {order}, 프레임 {len(poses)}개 ({args.fps:g}fps)", flush=True)
    generate_property(out, config=config, render_photos=True, seed=args.plan, log=lambda *_: None,
                      truth_metadata={"purpose": "집 전체 걷기 영상 검증", "walk_order": order, "fps": args.fps})
    names = sorted((out/"photos").glob("listing_*.jpg"))
    first = cv2.imread(str(names[0])); h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out/"walkthrough.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    blur = []
    if args.phone:  # 정답 사진은 그대로 두고 영상 프레임에만 화질 효과
        gt = json.loads((out/"property.gt.json").read_text(encoding="utf-8"))
        for f, length in phone_effects(names, poses, args.plan, gt["intrinsics"]["focal"], fps=args.fps):
            writer.write(f); blur.append(length)
    else:
        for n in names: writer.write(cv2.imread(str(n)))
    writer.release()
    print(json.dumps({"frames": len(names), "seconds": round(time.perf_counter()-started, 1), "video": str(out/"walkthrough.mp4"),
                      "duration_s": round(len(names)/args.fps, 1),
                      "blur_px": {"median": float(np.median(blur)), "p90": float(np.percentile(blur, 90))} if blur else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
