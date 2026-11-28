// AI 재구성 보기: 테슬라 주행 화면처럼 '인식해서 다시 그린' 집을 보여 준다.
// - 도면 모델(벽·바닥·문·관측 경계) + 크기·방향을 맞춘 가구 모형을 어두운 배경·부드러운 그림자로 그린다.
// - 실사 3D(가우시안 스플랫)를 투명도로 겹쳐 AI 모형과 실제 촬영을 함께 비교한다.
// - 촬영 경로 재생: 촬영한 카메라 궤적을 따라 시점을 움직인다(주행 기록 재생처럼).
import { makeLabel } from "./labels.js";
import { spaceNames, structureModel } from "./property.js";

const BG_TOP = "#1a212b", BG_BOTTOM = "#0d1117";

function gradientTexture(THREE) {
  const c = document.createElement("canvas"); c.width = 2; c.height = 256;
  const g = c.getContext("2d"), grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, BG_TOP); grad.addColorStop(1, BG_BOTTOM);
  g.fillStyle = grad; g.fillRect(0, 0, 2, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

const smooth = (t) => t * t * (3 - 2 * t);

export class AIView {
  constructor({ THREE, scene, renderer, world, cam, orbit, toast, baseLights = [] }) {
    Object.assign(this, { THREE, scene, renderer, world, cam, orbit, toast, baseLights });
    this.on = false;
    this.group = null;
    this.rig = this.trail = this.marker = this.model = null;
    this.blend = 0;
    this.bg = gradientTexture(THREE);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;  // r18x: PCFSoft 제거됨(PCF + radius로 부드럽게)
    // 그림자를 만드는 조명은 AI 보기에서만 켠다(다른 보기의 모형은 그림자를 받지 않음).
    this.sun = new THREE.DirectionalLight(0xffffff, 2.6);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0006;
    this.sun.shadow.normalBias = 0.02;
    this.sun.shadow.radius = 4;
    this.sun.visible = false;
    this.fill = new THREE.HemisphereLight(0xcfe3ff, 0x1b222b, 0.7);
    this.fill.visible = false;
    scene.add(this.sun, this.sun.target, this.fill);
    this.replay = { path: null, t: 0, playing: false, speed: 1, savedFov: null, view: "chase", chase: null };
  }

  dispose() {
    if (!this.group) return;
    this.group.traverse((o) => { o.geometry?.dispose?.(); if (o.material) [].concat(o.material).forEach((m) => m.dispose()); if (o.element) o.element.remove(); });
    this.group.removeFromParent();
    this.group = null;
    this.rig = this.trail = this.marker = this.model = null;
  }

  build(data) {
    this.dispose();
    this.stop();
    this.data = data;
    const { THREE } = this;
    const graph = data.property_graph;
    this.group = new THREE.Group();
    this.group.visible = false;
    const height = data.room?.height ?? 2.4;
    let model = null;
    if (data.floorplan?.walls?.length || graph?.spaces?.length) {
      model = structureModel(THREE, graph ?? { spaces: [], portals: [] }, height, data.floorplan, data.objects?.instances, "ai");
      this.group.add(model);
    }
    this.model = model;
    // 방 이름 · 면적 라벨(벽 단면 위에 떠 있게)
    const names = graph?.spaces?.length ? spaceNames(graph) : {};
    const labelY = Math.min(1.1, height) + 0.35;
    if (data.floorplan?.rooms?.length) {
      for (const r of data.floorplan.rooms) {
        const n = names[r.id];
        const text = `${n?.name ?? "공간"} · ${r.area_m2.toFixed(1)}㎡`;
        const label = makeLabel(text, "ai-room");
        label.position.set(r.anchor_xz[0], labelY, r.anchor_xz[1]);
        this.group.add(label);
      }
    } else if (graph?.spaces?.length) {
      for (const s of graph.spaces) {
        const n = names[s.id];
        const label = makeLabel(`${n.name} · ${n.area.toFixed(1)}㎡`, "ai-room");
        label.position.set(s.center_xz[0], labelY, s.center_xz[1]);
        this.group.add(label);
      }
    }
    // 촬영 경로(바닥 위 하늘색 선)
    this.capturePath = data.capture_path;
    this.replay.path = this.buildPath(data.cameras ?? []);
    if (this.replay.path) {
      const pts = this.replay.path.keys.map((k) => new THREE.Vector3(k.pos.x, 0.03, k.pos.z));
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      this.trail = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x46c2ff, transparent: true, opacity: 0.55 }));
      this.trail.visible = false;  // 재생을 시작하면 보인다(사진 순서 경로는 복잡할 수 있음)
      this.group.add(this.trail);
      this.marker = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.16, 32).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: 0x46c2ff, side: THREE.DoubleSide }));
      this.marker.position.copy(pts[0]); this.marker.visible = false; this.group.add(this.marker);
    }
    this.world.add(this.group);
    this.fitShadow();
    return !!model;
  }

  // 촬영 순서: 영상은 프레임 순서, 사진은 가까운 사진끼리 잇는 순서(걸어서 찍은 것처럼).
  buildPath(cameras) {
    const { THREE } = this;
    if (cameras.length < 2) return null;
    let order;
    const planned = this.capturePath?.names;
    const byName = new Map(cameras.map((c) => [c.name, c]));
    if (planned?.length >= 2 && planned.every((n) => byName.has(n))) {
      // 분석이 정한 순서(영상 프레임 / 사진 촬영 시각 / 방별 짧은 경로)
      order = planned.map((n) => byName.get(n));
    } else if (cameras.filter((c) => c.video_frame != null).length >= cameras.length * 0.8) {
      order = [...cameras].sort((a, b) => (a.video_frame ?? 1e9) - (b.video_frame ?? 1e9));
    } else {
      const left = [...cameras]; order = [left.shift()];
      while (left.length) {
        const p = order[order.length - 1].position;
        let best = 0, bd = Infinity;
        left.forEach((c, i) => { const d = Math.hypot(c.position[0] - p[0], c.position[1] - p[1], c.position[2] - p[2]); if (d < bd) { bd = d; best = i; } });
        order.push(left.splice(best, 1)[0]);
      }
    }
    // 거의 같은 자리·방향의 프레임은 건너뛰어 부드럽게
    const keys = [];
    for (const c of order) {
      const pos = new THREE.Vector3(...c.position), quat = new THREE.Quaternion(...c.quaternion);
      const last = keys[keys.length - 1];
      if (last && last.pos.distanceTo(pos) < 0.12 && last.quat.angleTo(quat) < 0.15) continue;
      keys.push({ pos, quat, name: c.name, photo: c.source_photo ?? c.name });
    }
    if (keys.length < 2) return null;
    // 구간 시간: 걷는 속도 0.7m/s, 회전 60°/s 중 느린 쪽. 너무 먼 점프(4m 초과)는 1초로 빠르게.
    const times = [0];
    for (let i = 1; i < keys.length; i++) {
      const dist = keys[i].pos.distanceTo(keys[i - 1].pos), ang = keys[i].quat.angleTo(keys[i - 1].quat);
      const dt = dist > 4 ? 1 : Math.max(dist / 0.7, ang / (Math.PI / 3), 0.12);
      times.push(times[i - 1] + dt);
    }
    const curve = new THREE.CatmullRomCurve3(keys.map((k) => k.pos), false, "centripetal");
    return { keys, times, curve, duration: times[times.length - 1] };
  }

  fitShadow() {
    if (!this.model) return;
    const { THREE } = this;
    const box = new THREE.Box3().setFromObject(this.model);
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
    const r = Math.max(size.x, size.z) * 0.75 + 1;
    this.sun.position.set(c.x - r * 0.55, c.y + r * 0.85, c.z - r * 0.5);  // 카메라(앞쪽) 반대편에서 비춰 그림자가 보이게
    this.sun.target.position.copy(c);
    const sc = this.sun.shadow.camera;
    sc.left = -r; sc.right = r; sc.top = r; sc.bottom = -r; sc.near = 0.1; sc.far = r * 4;
    sc.updateProjectionMatrix();
  }

  setScale(k) { if (this.group) this.fitShadow(); }

  enable(on) {
    this.on = on && !!this.group;
    if (this.group) this.group.visible = this.on;
    this.sun.visible = this.fill.visible = this.on;
    // 기본 조명을 낮춰 그림자·입체감이 보이게 한다(끄면 원래 세기로)
    for (const l of this.baseLights) { if (l.userData.base == null) l.userData.base = l.intensity; l.intensity = this.on ? l.userData.base * 0.15 : l.userData.base; }
    if (this.on) { this.savedBg = this.scene.background; this.scene.background = this.bg; this.fitShadow(); }
    else if (this.savedBg !== undefined) { this.scene.background = this.savedBg; this.savedBg = undefined; }
    return this.on;
  }

  overview() {
    if (!this.model) return;
    const { THREE } = this;
    const box = new THREE.Box3().setFromObject(this.model), c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
    const r = Math.max(size.x, size.z);
    this.cam.position.set(c.x + 0.25 * r, 0.95 * r + 1, c.z + 0.75 * r);
    this.orbit.target.set(c.x, 0, c.z);
    this.orbit.update();
  }

  // 실사 3D 겹치기(0 = AI 모형만, 1 = 실사 그대로)
  applyBlend(splat, value) {
    this.blend = value;
    if (!splat) return;
    splat.opacity = Math.max(0.001, value);
  }

  // ---------------------------------------------------------------- 촬영 경로 재생
  // view: "chase" = 촬영자를 뒤·위에서 따라가는 시점(주행 화면처럼), "eye" = 촬영 카메라 그대로(1인칭)
  get hasPath() { return !!this.replay.path; }

  setView(view) {
    const R = this.replay;
    R.view = view === "eye" ? "eye" : "chase";
    this.applyFov();
    R.chase = null;
    if (R.path) this.applyPose(R.t, 0);
  }

  applyFov() {
    const R = this.replay;
    if (R.savedFov == null) return;
    const it = R.intrinsics;
    const capture = it?.focal && it?.height ? Math.min(90, Math.max(30, 2 * Math.atan(it.height / 2 / it.focal) * 180 / Math.PI)) : R.savedFov;
    this.cam.fov = R.view === "eye" ? capture : 55;  // 1인칭은 촬영 카메라와 같은 화각(실사 겹치기가 맞도록)
    this.cam.updateProjectionMatrix();
  }

  play(intrinsics) {
    const R = this.replay;
    if (!R.path) return false;
    if (R.t >= R.path.duration) R.t = 0;
    R.intrinsics = intrinsics;
    if (R.savedFov == null) { R.savedFov = this.cam.fov; R.chase = null; }
    this.applyFov();
    R.playing = true;
    this.orbit.enabled = false;
    this.showGuides(true);
    return true;
  }

  pause() {
    const R = this.replay;
    if (!R.playing) return;
    R.playing = false;
    this.release();
  }

  stop() {
    const R = this.replay;
    R.playing = false;
    if (R.savedFov != null) { this.cam.fov = R.savedFov; this.cam.updateProjectionMatrix(); R.savedFov = null; }
    R.chase = null;
    this.showGuides(false);
    this.release();
  }

  showGuides(on) {
    if (this.trail) this.trail.visible = on;
    if (this.marker) this.marker.visible = on;
    if (this.rig) this.rig.visible = on && this.replay.view !== "eye";
  }

  // 재생을 멈추면 지금 보는 방향 앞쪽을 중심으로 다시 둘러볼 수 있게 한다.
  release() {
    const { THREE } = this;
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(this.cam.quaternion);
    this.orbit.target.copy(this.cam.position).addScaledVector(fwd, this.replay.view === "eye" ? 1.2 : 3);
    this.orbit.enabled = true;
    this.orbit.update();
  }

  seek(fraction) {
    const R = this.replay;
    if (!R.path) return;
    if (R.savedFov == null) { R.savedFov = this.cam.fov; this.applyFov(); }
    R.t = Math.max(0, Math.min(1, fraction)) * R.path.duration;
    R.chase = null;
    this.showGuides(true);
    this.applyPose(R.t, 0);
  }

  // 촬영 카메라 표시(테슬라 화면의 '내 차'): 작은 몸체 + 바라보는 방향의 시야 사각뿔
  ensureRig() {
    if (this.rig) return this.rig;
    const { THREE } = this;
    const rig = new THREE.Group();
    const body = new THREE.Mesh(new THREE.SphereGeometry(0.09, 20, 14), new THREE.MeshStandardMaterial({ color: 0x46c2ff, emissive: 0x0b4a66, roughness: 0.4 }));
    rig.add(body);
    const d = 0.55, hw = 0.32, hh = 0.22;
    const v = [0, 0, 0, -hw, hh, -d, 0, 0, 0, hw, hh, -d, 0, 0, 0, -hw, -hh, -d, 0, 0, 0, hw, -hh, -d,
      -hw, hh, -d, hw, hh, -d, hw, hh, -d, hw, -hh, -d, hw, -hh, -d, -hw, -hh, -d, -hw, -hh, -d, -hw, hh, -d];
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
    rig.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x46c2ff })));
    rig.visible = false;
    this.group.add(rig);
    this.rig = rig;
    return rig;
  }

  // 시간 t의 촬영 자세: 위치는 곡선, 방향은 이웃 키 사이 부드러운 보간
  poseAt(t) {
    const { keys, times, curve } = this.replay.path;
    let i = 1;
    while (i < times.length - 1 && times[i] < t) i++;
    const t0 = times[i - 1], t1 = times[i], u = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 1;
    const s = smooth(u);
    const pos = curve.getPoint(((i - 1) + s) / (keys.length - 1));
    const quat = keys[i - 1].quat.clone().slerp(keys[i].quat, s);
    this.current = keys[u < 0.5 ? i - 1 : i];
    return { pos, quat };
  }

  applyPose(t, dt = 0) {
    const { THREE } = this;
    const R = this.replay;
    const { pos, quat } = this.poseAt(t);
    const k = this.world.scale.x;
    if (this.marker) this.marker.position.set(pos.x, 0.03, pos.z);
    const rig = this.ensureRig();
    rig.position.copy(pos); rig.quaternion.copy(quat);
    rig.visible = R.view !== "eye";
    if (R.view === "eye") {
      this.cam.position.copy(pos).multiplyScalar(k);
      this.cam.quaternion.copy(quat);
      return;
    }
    // 따라가기: 촬영자가 바라보는 수평 방향 뒤 3.2m·위 3.2m에서 앞쪽 1.4m 지점을 본다. 흔들림은 부드럽게 따라간다.
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(quat); fwd.y = 0;
    if (fwd.lengthSq() < 1e-4) fwd.set(0, 0, -1); fwd.normalize();
    const eye = pos.clone().addScaledVector(fwd, -3.2); eye.y = pos.y + 3.2;
    const look = pos.clone().addScaledVector(fwd, 1.4); look.y = Math.max(0.3, pos.y - 0.7);
    eye.multiplyScalar(k); look.multiplyScalar(k);
    if (!R.chase || dt === 0) R.chase = { eye: eye.clone(), look: look.clone() };
    else {
      const a = 1 - Math.exp(-dt * 3.5);
      R.chase.eye.lerp(eye, a); R.chase.look.lerp(look, a);
    }
    this.cam.position.copy(R.chase.eye);
    this.cam.lookAt(R.chase.look);
  }

  update(dt) {
    const R = this.replay;
    if (!R.playing || !R.path) return null;
    R.t += dt * R.speed;
    if (R.t >= R.path.duration) { R.t = R.path.duration; this.applyPose(R.t, dt); this.pause(); return { done: true, fraction: 1 }; }
    this.applyPose(R.t, dt);
    return { done: false, fraction: R.t / R.path.duration, photo: this.current?.photo };
  }
}
