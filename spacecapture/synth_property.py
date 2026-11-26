"""Owned, paired multi-space photos, geometry, doors and ground-truth plans.

The default geometry and all textures are generated here. A caller can also
provide licensed geometry, retaining its provenance through truth_metadata.
No third-party property photos or pretrained model weights are included.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from html import escape

import cv2
import numpy as np

from .synth import _bilinear, look_rotation


def configuration(kind="two_rooms", seed=0):
    rng = np.random.default_rng(seed)
    W, D, H = (8., 4., 2.6) if kind == "two_rooms" else (8., 6., 2.6)
    surfaces = []
    def wall(axis, pos, lo, hi, bottom=0., top=H):
        surfaces.append({"axis": axis, "pos": pos, "lo": lo, "hi": hi, "bottom": bottom, "top": top})
    wall(0, 0., 0., D); wall(0, W, 0., D)
    wall(2, 0., 0., W); wall(2, D, 0., W)
    if kind == "two_rooms":
        divisions, gap = [4.], (1.5, 2.5)
        rooms = [(0., 4., "room"), (4., 8., "room")]
    else:
        divisions, gap = [3., 5.], (2.5, 3.5)
        rooms = [(0., 3., "room"), (3., 5., "corridor"), (5., 8., "room")]
    portals = []
    for i, x in enumerate(divisions):
        wall(0, x, 0., gap[0]); wall(0, x, gap[1], D)
        wall(0, x, gap[0], gap[1], 2.1, H)
        portals.append({"id": f"door-{i+1}", "spaces": [f"gt-{i+1}", f"gt-{i+2}"],
                        "center_xz": [x, sum(gap)/2], "width_m": gap[1]-gap[0], "height_m": 2.1,
                        "material": "open", "kind": "doorway"})
    spaces = [{"id": f"gt-{i+1}", "kind": role, "polygon_xz": [[lo, 0.], [hi, 0.], [hi, D], [lo, D]]}
              for i, (lo, hi, role) in enumerate(rooms)]
    poses = []
    for i, (lo, hi, role) in enumerate(rooms):
        n = 18 if role == "room" else 10
        for j in range(n):
            angle = 2 * math.pi * j / n
            C = np.array([(lo+hi)/2 + max(.2, (hi-lo)/2-.7)*math.cos(angle),
                          1.45 + .015*math.sin(angle*2), D/2 + (D/2-.7)*math.sin(angle)])
            target = np.array([(lo+hi)/2, 1.1, D/2])
            if j % 3 == 0:
                target = np.array([divisions[min(i, len(divisions)-1)], 1.2, sum(gap)/2])
            poses.append((C, look_rotation(target-C), f"gt-{i+1}"))
    # The crossing views deliberately see common landmarks on both sides of an
    # opening; ordinary disconnected room photos must not be forced together.
    for i, x in enumerate(divisions):
        for side in (-1., 1.):
            for j in range(4):
                C = np.array([x+side*(.35+j*.2), 1.45, sum(gap)/2 + .1*math.sin(j)])
                target = np.array([x-side*1.5, 1.2, sum(gap)/2])
                poses.append((C, look_rotation(target-C), f"gt-{i+1 if side < 0 else i+2}"))
    order = rng.permutation(len(poses))
    return {"size": [W, H, D], "surfaces": surfaces, "spaces": spaces, "portals": portals,
            "poses": [poses[i] for i in order]}


def raycast(origin, directions, config):
    directions = np.asarray(directions, np.float32)
    origin = np.asarray(origin, np.float32)
    W, H, D = config["size"]
    n = len(directions)
    distance, face = np.full(n, np.inf, np.float32), np.full(n, -1, np.int32)
    planes = config["surfaces"] + config.get("horizontal_surfaces", [
        {"axis": 1, "pos": 0., "lo": 0., "hi": W, "bottom": 0., "top": D},
        {"axis": 1, "pos": H, "lo": 0., "hi": W, "bottom": 0., "top": D}])
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, plane in enumerate(planes):
            axis = plane["axis"]
            u, v = (2, 1) if axis == 0 else (0, 1) if axis == 2 else (0, 2)
            t = (plane["pos"]-origin[axis]) / directions[:, axis]
            pu, pv = origin[u]+t*directions[:, u], origin[v]+t*directions[:, v]
            hit = (t > 1e-5) & (t < distance) & (pu >= plane["lo"]-1e-5) & (pu <= plane["hi"]+1e-5)
            hit &= (pv >= plane["bottom"]-1e-5) & (pv <= plane["top"]+1e-5)
            distance[hit], face[hit] = t[hit], i
    return distance, face, planes


def _texture(width, height, seed):
    rng = np.random.default_rng(seed)
    ppm = 180
    w, h = max(64, round(width*ppm)), max(64, round(height*ppm))
    base = rng.integers(160, 225, 3)
    noise = cv2.resize(rng.normal(0, 13, (max(3,h//25), max(3,w//25))).astype(np.float32), (w,h))
    image = np.clip(base + noise[..., None] + rng.normal(0, 2, (h,w,1)), 0, 255).astype(np.uint8)
    for _ in range(max(15, round(width*height*4))):
        x, y = int(rng.integers(0,w)), int(rng.integers(0,h))
        size = int(rng.integers(12,90))
        color = tuple(int(c) for c in rng.integers(20,230,3))
        cv2.rectangle(image, (x,y), (x+size,y+size//2), color, -1)
        cv2.putText(image, str(rng.integers(10,999)), (x,y+size//2), cv2.FONT_HERSHEY_SIMPLEX, .4, (30,30,30), 1)
    return image


def _noise(rng, h, w, scales):
    out = np.zeros((h, w), np.float32)
    for cell, weight in scales:
        layer = rng.normal(0, 1, (max(2, h//cell+2), max(2, w//cell+2))).astype(np.float32)
        out += weight*cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def _texture_interior(plane, seed, house_seed=0, ppm=180):
    """실제 집처럼 보이는 재질: 나무 마루 바닥, 미색 벽지 벽(걸레받이·액자·콘센트), 흰 천장.

    무늬는 면마다 다른 난수로 만들어 반복이 없다(반복 무늬는 사진 사이 오매칭을 만든다).
    색은 BGR. 벽은 v=0이 바닥 쪽이다. (2026-11-26, 무지개색 노이즈 대신 쓰는 시험 장면 재질)
    """
    rng = np.random.default_rng(seed)
    house = np.random.default_rng(house_seed)
    width, height = plane["hi"]-plane["lo"], plane["top"]-plane["bottom"]
    w, h = max(32, round(width*ppm)), max(32, round(height*ppm))
    if plane["axis"] == 1 and plane["pos"] < .05:  # 바닥: 엇갈린 나무 마루
        tone = house.choice([[70, 120, 170], [95, 145, 190], [55, 95, 140], [120, 165, 200]]).astype(np.float32)
        image = np.zeros((h, w, 3), np.float32)
        plank_w = int(.14*ppm); y = -int(rng.integers(0, plank_w))
        while y < h:
            x = -int(rng.integers(0, int(1.4*ppm)))
            while x < w:
                length = int(rng.uniform(.9, 1.8)*ppm)
                y0, y1, x0, x1 = max(0, y), min(h, y+plank_w), max(0, x), min(w, x+length)
                if y1 > y0 and x1 > x0:
                    ph, pw = y1-y0, x1-x0
                    grain = cv2.resize(rng.normal(0, 1, (max(2, ph//2), max(2, pw//40))).astype(np.float32), (pw, ph))
                    grain += .5*cv2.resize(rng.normal(0, 1, (max(2, ph//6), max(2, pw//12))).astype(np.float32), (pw, ph))
                    image[y0:y1, x0:x1] = tone*rng.uniform(.85, 1.12) + 9*grain[..., None]
                    image[y0:y1, x0:x0+2] *= .6
                x += length
            image[max(0, y):max(0, y)+2] *= .6
            y += plank_w
    elif plane["axis"] == 1:  # 천장·문 위 인방 아랫면: 흰색
        image = np.full((h, w, 3), 238., np.float32) + 2.5*_noise(rng, h, w, ((40, 1.), (6, .6)))[..., None]
    else:  # 벽: 미색 실크 벽지 + 걸레받이 + 액자·콘센트
        paper = house.choice([[220, 226, 232], [216, 220, 222], [200, 214, 226], [208, 220, 214]]).astype(np.float32)
        image = paper*rng.uniform(.97, 1.03) + _noise(rng, h, w, ((60, 2.5), (12, 3.), (3, 3.)))[..., None]
        base_h = int(.08*ppm)
        image[:base_h] = np.array([235, 238, 240], np.float32) + rng.normal(0, 1.5, (min(base_h, h), w, 1))
        image[base_h:base_h+2] *= .82
        for _ in range(int(rng.poisson(max(.3, width*.35)))):  # 액자
            fw, fh = int(rng.uniform(.3, .8)*ppm), int(rng.uniform(.3, .6)*ppm)
            cx, cy = int(rng.uniform(0, max(1, w-fw))), int(rng.uniform(1.2, 1.7)*ppm)
            if cy+fh >= h or cx+fw >= w: continue
            palette = np.array([[70, 90, 110], [150, 170, 190], [110, 130, 120], [180, 190, 200], [60, 70, 80], [140, 120, 100]], np.float32)
            art = cv2.resize(palette[rng.integers(0, len(palette), (3, 4))]*rng.uniform(.8, 1.15, (3, 4, 1)).astype(np.float32),
                             (fw-16, fh-16), interpolation=cv2.INTER_CUBIC)  # 차분한 색 그림
            art += 12*_noise(rng, fh-16, fw-16, ((8, 1.), (3, .5)))[..., None]
            image[cy:cy+fh, cx:cx+fw] = np.array([40, 45, 50], np.float32) if rng.random() < .6 else np.array([90, 140, 180], np.float32)
            image[cy+8:cy+fh-8, cx+8:cx+fw-8] = art
        for _ in range(int(rng.poisson(max(.2, width*.25)))):  # 콘센트·스위치
            sw, sh = int(.08*ppm), int(.12*ppm)
            cx, cy = int(rng.uniform(0, max(1, w-sw))), int((rng.choice([.3, 1.15]))*ppm)
            if cy+sh >= h or cx+sw >= w: continue
            image[cy:cy+sh, cx:cx+sw] = 245.
            image[cy:cy+sh, cx:cx+1] = image[cy:cy+sh, cx+sw-1:cx+sw] = 170.
            image[cy:cy+1, cx:cx+sw] = image[cy+sh-1:cy+sh, cx:cx+sw] = 170.
    return np.clip(image, 0, 255).astype(np.uint8)


def _texture_unique(width, height, seed):
    """면마다 고유한 다중 스케일 노이즈·불규칙 얼룩 텍스처.

    기존 _texture는 모든 면에 축 정렬 사각형 모서리와 같은 글꼴 숫자를 반복해,
    서로 다른 방의 사진 사이에 기하 검증을 통과하는 오매칭이 많이 생겼다
    (LH 평면도 렌더 검증, 2026-11-10). 반복 무늬가 없는 실내 표면을 흉내 낸다.
    """
    rng = np.random.default_rng(seed)
    ppm = 180
    w, h = max(64, round(width*ppm)), max(64, round(height*ppm))
    image = np.zeros((h, w, 3), np.float32)
    for cell, weight in ((90, 34.), (30, 22.), (9, 14.), (3, 7.)):
        layer = rng.normal(0, 1, (max(2, h//cell+2), max(2, w//cell+2), 3)).astype(np.float32)
        image += weight*cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC)
    image += rng.integers(110, 200, 3).astype(np.float32)
    blobs = cv2.resize(rng.normal(0, 1, (max(2, h//40), max(2, w//40))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(3):
        level = float(rng.normal(1.1, .2))
        mask = blobs > level
        image[mask] = image[mask]*.35 + rng.integers(20, 235, 3).astype(np.float32)*.65
        blobs = np.roll(blobs, (int(rng.integers(0, h)), int(rng.integers(0, w))), axis=(0, 1))
    return np.clip(image, 0, 255).astype(np.uint8)


def _write_gt_svg(path, gt):
    W, _, D = gt["size"]
    scale = min(800/W, 600/D)
    body = []
    for i, space in enumerate(gt["spaces"]):
        coords = " ".join(f"{100+x*scale:.2f},{120+z*scale:.2f}" for x,z in space["polygon_xz"])
        center = np.mean(space["polygon_xz"], axis=0)
        body.append(f'<polygon points="{coords}" fill="{["#e4effc","#f8efd7","#e7f4e8"][i%3]}" stroke="#25445d" stroke-width="4"/>')
        label = {"room":"방", "corridor":"복도", "hall":"홀"}.get(space["kind"], "공간")
        body.append(f'<text x="{100+center[0]*scale}" y="{120+center[1]*scale}" text-anchor="middle">{escape(space["id"])} ({label})</text>')
    for door in gt["portals"]:
        x,z = door["center_xz"]
        endpoints = door.get("endpoints_xz", [[x,z-door["width_m"]/2],[x,z+door["width_m"]/2]])
        a,b = np.asarray(endpoints)
        body.append(f'<line x1="{100+a[0]*scale}" y1="{120+a[1]*scale}" x2="{100+b[0]*scale}" y2="{120+b[1]*scale}" stroke="#d87920" stroke-width="9"/>')
    title = "외부 합성 구조 기반 정답 도면" if gt.get("origin")=="external_structure_local_render" else "자체 생성 다중 공간 정답 도면"
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="850"><rect width="1000" height="850" fill="white"/><g font-family="Malgun Gothic,sans-serif" font-size="18"><text x="100" y="55">{title}</text>{"".join(body)}<text x="100" y="810">주황색: 출입구 정답 · 추론 결과와 별도 보관</text></g></svg>'
    path.write_text(svg, encoding="utf-8")


def generate_property(output: Path, kind="two_rooms", seed=0, render_photos=True, log=print, landmark_density=120,
                      config=None, truth_metadata=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = configuration(kind, seed) if config is None else config
    W,H,D = config["size"]
    # Sample surface landmarks independently from the inference grid/room seeds.
    rng = np.random.default_rng(seed+123)
    _, _, planes = raycast([1,1,1], [[1,0,0]], config)
    all_points = []
    for plane in planes:
        count = max(100, int((plane["hi"]-plane["lo"])*(plane["top"]-plane["bottom"])*landmark_density))
        uv = rng.uniform([plane["lo"],plane["bottom"]], [plane["hi"],plane["top"]], (count,2))
        axis = plane["axis"]
        u,v = (2,1) if axis == 0 else (0,1) if axis == 2 else (0,2)
        points = np.zeros((count,3)); points[:,axis] = plane["pos"]; points[:,u],points[:,v] = uv.T
        all_points.append(points)
    P = np.vstack(all_points)
    observations = [[] for _ in P]
    width,height = 960,640
    focal = width/(2*math.tan(math.radians(78)/2))
    photos = []
    style = config.get("texture_style")
    if style == "interior":
        textures = [_texture_interior(p,seed*100+i,house_seed=seed) for i,p in enumerate(planes)] if render_photos else []
    else:
        make_texture = _texture_unique if style == "unique_noise" else _texture
        textures = [make_texture(p["hi"]-p["lo"],p["top"]-p["bottom"],seed*100+i) for i,p in enumerate(planes)] if render_photos else []
    if render_photos:
        (output/"photos").mkdir(exist_ok=True)
    if render_photos:
        jj,ii = np.meshgrid(np.arange(width,dtype=np.float32),np.arange(height,dtype=np.float32))
        dirs = np.stack([(jj+.5-width/2)/focal,(ii+.5-height/2)/focal,np.ones_like(jj)],-1).reshape(-1,3)
    for index,(C,R,space_id) in enumerate(config["poses"]):
        name = f"listing_{index:04d}.jpg"
        camera_points = (P-C)@R
        projected = camera_points[:,:2]/np.maximum(camera_points[:,2:3],1e-6)*focal + [width/2,height/2]
        in_view = (camera_points[:,2]>.1)&(projected[:,0]>0)&(projected[:,0]<width)&(projected[:,1]>0)&(projected[:,1]<height)
        indices = np.flatnonzero(in_view)
        if len(indices):
            vectors = P[indices]-C
            lengths = np.linalg.norm(vectors,axis=1)
            t,_,_ = raycast(C,vectors/lengths[:,None],config)
            visible = indices[np.abs(t-lengths)<.025]
            for point_index in visible: observations[point_index].append(index)
        visible_doors=[]
        for door in config["portals"]:
            x,z=door["center_xz"]
            center=np.array([x,1.,z]); cp=(center-C)@R
            if cp[2] <= .1: continue
            uv=cp[:2]/cp[2]*focal+[width/2,height/2]
            t,_,_=raycast(C,[(center-C)/np.linalg.norm(center-C)],config)
            if 0<uv[0]<width and 0<uv[1]<height and t[0]>=np.linalg.norm(center-C)-.03:
                visible_doors.append(door["id"])
        photos.append({"id":name,"space_id":space_id,"position":C.round(5).tolist(),
                       "world_from_camera":R.round(7).tolist(),"visible_portals":visible_doors})
        if render_photos:
            world_dirs=dirs@R.T
            t,face,_=raycast(C,world_dirs,config)
            pixels=C+t[:,None]*world_dirs
            color=np.zeros((len(dirs),3),np.float32)
            for i,plane in enumerate(planes):
                mask=face==i
                axis=plane["axis"]; u,v=(2,1) if axis==0 else (0,1) if axis==2 else (0,2)
                color[mask]=_bilinear(textures[i],(pixels[mask,u]-plane["lo"])*180,(pixels[mask,v]-plane["bottom"])*180)
            if style == "interior":
                color[face<0]=(225,228,230)  # 면 사이로 빠진 광선(모서리 틈)은 검게 두지 않고 벽색으로
            image=cv2.GaussianBlur(color.reshape(height,width,3),(0,0),.55)
            ok,encoded=cv2.imencode('.jpg',np.clip(image,0,255).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok: raise RuntimeError("합성 사진 저장에 실패했어요.")
            encoded.tofile(output/"photos"/name)
        if index%12==0: log(f"  다중 공간 자료 생성 {index+1}/{len(config['poses'])}")
    keep=np.array([len(o)>=2 for o in observations])
    P=P[keep]; observations=[o for o,k in zip(observations,keep) if k]
    offsets=np.cumsum([0]+[len(o) for o in observations])
    np.savez_compressed(output/"observations.npz",points=P,cameras=np.array([p[0] for p in config["poses"]]),
                        offsets=offsets,observers=np.array([i for o in observations for i in o],np.int32))
    gt={"id":f"owned-{kind}-{seed}","license":"MIT","origin":"procedurally_generated_here",
        "commercial_training_allowed":True,"size":config["size"],"spaces":config["spaces"],"portals":config["portals"],
        "photos":photos,"intrinsics":{"width":width,"height":height,"focal":focal},
        "note":"직접 생성한 정답 데이터입니다. 추론기에는 방·문 정답을 전달하지 않습니다."}
    if truth_metadata:
        gt.update(truth_metadata)
    (output/"property.gt.json").write_text(json.dumps(gt,ensure_ascii=False,indent=2),encoding="utf-8")
    _write_gt_svg(output/"floorplan.gt.svg",gt)
    log(f"  합성 사진·도면 쌍: 사진 {len(photos)}장, 공간 {len(gt['spaces'])}개, 출입구 {len(gt['portals'])}개")
    return output
