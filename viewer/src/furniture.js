import * as THREE from "three";
import { fmtCm, fmtLen, makeLabel } from "./labels.js";

// Common Korean furniture sizes (cm): width x depth x height
export const PRESETS = [
  { name: "싱글 침대", w: 100, d: 200, h: 45, color: "#9fb6d8" },
  { name: "슈퍼싱글 침대", w: 110, d: 200, h: 45, color: "#9fb6d8" },
  { name: "퀸 침대", w: 150, d: 200, h: 45, color: "#8ca8d0" },
  { name: "킹 침대", w: 165, d: 205, h: 45, color: "#7e9cc9" },
  { name: "2인 소파", w: 160, d: 85, h: 85, color: "#cfa883" },
  { name: "3인 소파", w: 210, d: 90, h: 85, color: "#c39a72" },
  { name: "책상", w: 120, d: 60, h: 74, color: "#a9c8a0" },
  { name: "4인 식탁", w: 140, d: 80, h: 75, color: "#bba78f" },
  { name: "의자", w: 45, d: 50, h: 85, color: "#d4b68e" },
  { name: "옷장", w: 120, d: 60, h: 200, color: "#c9bba4" },
  { name: "책장", w: 80, d: 30, h: 180, color: "#bba98e" },
  { name: "서랍장", w: 80, d: 45, h: 90, color: "#c6b198" },
  { name: "TV장", w: 180, d: 40, h: 45, color: "#a2abb7" },
  { name: "양문형 냉장고", w: 91, d: 92, h: 180, color: "#d7dde4" },
  { name: "드럼 세탁기", w: 69, d: 75, h: 98, color: "#e1e5ea" },
  { name: "화장대", w: 80, d: 40, h: 75, color: "#e2c7c7" },
];

const EPS = 0.005; // touching is fine
const SNAP = 0.04; // pull flush against a wall closer than this
const BAD = new THREE.Color("#e25555");
const SELECT = new THREE.Color("#2f6fed");
let nextId = 1;

function corners(it) {
  // same rotation as THREE's rotation.y: (x, z) -> (x c + z s, -x s + z c)
  const c = Math.cos(it.rot);
  const s = Math.sin(it.rot);
  const hw = it.w / 2;
  const hd = it.d / 2;
  return [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]].map(([lx, lz]) => [it.x + lx * c + lz * s, it.z - lx * s + lz * c]);
}

function extent(it) {
  const cs = corners(it);
  const xs = cs.map((p) => p[0]);
  const zs = cs.map((p) => p[1]);
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minZ: Math.min(...zs), maxZ: Math.max(...zs) };
}

function overlaps(a, b) {
  // separating axis test for two rectangles on the floor
  const ca = corners(a);
  const cb = corners(b);
  const axes = [a, b].flatMap((it) => {
    const c = Math.cos(it.rot);
    const s = Math.sin(it.rot);
    return [[c, -s], [s, c]];
  });
  for (const [ax, az] of axes) {
    const pa = ca.map(([x, z]) => x * ax + z * az);
    const pb = cb.map(([x, z]) => x * ax + z * az);
    if (Math.max(...pa) <= Math.min(...pb) + EPS || Math.max(...pb) <= Math.min(...pa) + EPS) return false;
  }
  return true;
}

class Item {
  constructor(spec) {
    this.id = spec.id ?? nextId;
    nextId = Math.max(nextId, this.id + 1);
    this.name = spec.name ?? "가구";
    this.w = spec.w;
    this.d = spec.d;
    this.h = spec.h;
    this.x = spec.x ?? 0;
    this.z = spec.z ?? 0;
    this.rot = spec.rot ?? 0;
    this.color = spec.color ?? "#b9c3cd";
    this.group = new THREE.Group();
    this.material = new THREE.MeshStandardMaterial({ color: this.color, roughness: 0.85 });
    this.edgeMaterial = new THREE.LineBasicMaterial({ color: 0x2b323b });
    this.label = makeLabel("", "item");
    this.group.add(this.label);
    this.rebuild();
  }

  rebuild() {
    if (this.mesh) {
      this.group.remove(this.mesh, this.edges);
      this.mesh.geometry.dispose();
      this.edges.geometry.dispose();
    }
    const geo = new THREE.BoxGeometry(this.w, this.h, this.d);
    geo.translate(0, this.h / 2, 0);
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.userData.item = this;
    this.edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), this.edgeMaterial);
    this.group.add(this.mesh, this.edges);
    this.label.position.set(0, this.h + 0.1, 0);
    this.sync();
  }

  sync() {
    this.group.position.set(this.x, 0, this.z);
    this.group.rotation.y = this.rot;
    this.label.element.textContent = `${this.name} ${fmtCm(this.w)}×${fmtCm(this.d)}×${fmtCm(this.h)}`;
  }

  setState(selected, bad) {
    this.material.color.set(this.color);
    if (bad) this.material.color.lerp(BAD, 0.65);
    this.material.emissive.copy(selected ? SELECT : new THREE.Color(0));
    this.material.emissiveIntensity = selected ? 0.25 : 0;
    this.edgeMaterial.color.set(selected ? SELECT : bad ? BAD : new THREE.Color(0x2b323b));
    this.label.element.className = `label item${selected ? " selected" : ""}${bad && !selected ? " bad" : ""}`;
  }

  toJSON() {
    const r = (v) => Math.round(v * 1000) / 1000;
    return { id: this.id, name: this.name, w: r(this.w), d: r(this.d), h: r(this.h), x: r(this.x), z: r(this.z),
      rot: Math.round(this.rot * 1e4) / 1e4, color: this.color };
  }

  dispose() {
    this.label.element.remove();
    this.mesh.geometry.dispose();
    this.edges.geometry.dispose();
    this.material.dispose();
    this.edgeMaterial.dispose();
  }
}

/**
 * Furniture boxes on the floor (y = 0), in calibrated meters. Not part of the
 * calibrated scene root: a 2 m sofa stays 2 m when the room scale is corrected.
 */
export class FurnitureManager {
  constructor(root, { getBounds, onChange, onSelect }) {
    this.root = root;
    this.getBounds = getBounds;
    this.onChange = onChange;
    this.onSelect = onSelect;
    this.items = [];
    this.selected = null;
    this.drag = null;
    this.guides = new THREE.Group();
    this.root.add(this.guides);
    this.floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    this.conflicts = [];
    this.gaps = null;
  }

  get dragging() {
    return this.drag !== null;
  }

  meshes() {
    return this.items.map((it) => it.mesh);
  }

  add(spec, { select = true } = {}) {
    const b = this.getBounds();
    const it = new Item({
      ...spec,
      x: spec.x ?? (b.x0 + b.x1) / 2,
      z: spec.z ?? (b.z0 + b.z1) / 2,
    });
    this.items.push(it);
    this.root.add(it.group);
    if (select) this.select(it);
    this.changed();
    return it;
  }

  addPreset(p) {
    return this.add({ name: p.name, w: p.w / 100, d: p.d / 100, h: p.h / 100, color: p.color });
  }

  remove(it) {
    this.items = this.items.filter((x) => x !== it);
    this.root.remove(it.group);
    it.dispose();
    if (this.selected === it) this.select(null);
    this.changed();
  }

  duplicate(it) {
    const { id, ...spec } = it.toJSON();
    const copy = this.add({ ...spec, x: it.x + 0.3, z: it.z + 0.3 });
    this.clampInside(copy);
    copy.sync();
    this.changed();
  }

  clear() {
    for (const it of [...this.items]) {
      this.root.remove(it.group);
      it.dispose();
    }
    this.items = [];
    this.select(null);
    this.changed();
  }

  load(list) {
    for (const it of this.items) {
      this.root.remove(it.group);
      it.dispose();
    }
    this.items = [];
    this.selected = null;
    for (const spec of list || []) {
      if (!(spec.w > 0 && spec.d > 0 && spec.h > 0)) continue;
      const it = new Item(spec);
      this.items.push(it);
      this.root.add(it.group);
    }
    this.select(null);
    this.refresh();
  }

  toJSON() {
    return this.items.map((it) => it.toJSON());
  }

  select(it) {
    this.selected = it;
    this.refresh();
    this.onSelect?.(it);
  }

  rotate(it, deg) {
    it.rot = (it.rot + (deg * Math.PI) / 180) % (2 * Math.PI);
    it.sync();
    this.changed();
  }

  resize(it, { name, w, d, h }) {
    if (name !== undefined) it.name = name || "가구";
    if (w > 0) it.w = w;
    if (d > 0) it.d = d;
    if (h > 0) it.h = h;
    it.rebuild();
    this.changed();
  }

  pick(raycaster) {
    const hit = raycaster.intersectObjects(this.meshes(), false)[0];
    return hit ? hit.object.userData.item : null;
  }

  beginDrag(it, raycaster) {
    const p = raycaster.ray.intersectPlane(this.floor, new THREE.Vector3());
    if (!p) return false;
    this.drag = { it, dx: it.x - p.x, dz: it.z - p.z };
    return true;
  }

  dragTo(raycaster) {
    const p = raycaster.ray.intersectPlane(this.floor, new THREE.Vector3());
    if (!p || !this.drag) return;
    const { it, dx, dz } = this.drag;
    it.x = Math.round((p.x + dx) * 100) / 100; // 1 cm steps
    it.z = Math.round((p.z + dz) * 100) / 100;
    this.snapToWalls(it);
    it.sync();
    this.refresh();
  }

  endDrag() {
    this.drag = null;
    this.changed();
  }

  snapToWalls(it) {
    const b = this.getBounds();
    const e = extent(it);
    if (Math.abs(e.minX - b.x0) < SNAP) it.x += b.x0 - e.minX;
    else if (Math.abs(b.x1 - e.maxX) < SNAP) it.x += b.x1 - e.maxX;
    if (Math.abs(e.minZ - b.z0) < SNAP) it.z += b.z0 - e.minZ;
    else if (Math.abs(b.z1 - e.maxZ) < SNAP) it.z += b.z1 - e.maxZ;
  }

  clampInside(it) {
    const b = this.getBounds();
    const e = extent(it);
    if (e.minX < b.x0) it.x += b.x0 - e.minX;
    if (e.maxX > b.x1) it.x -= e.maxX - b.x1;
    if (e.minZ < b.z0) it.z += b.z0 - e.minZ;
    if (e.maxZ > b.z1) it.z -= e.maxZ - b.z1;
  }

  changed() {
    this.refresh();
    this.onChange?.();
  }

  /** Recheck collisions, recolor, redraw wall-distance guides for the selection. */
  refresh() {
    const b = this.getBounds();
    const bad = new Set();
    const msgs = [];
    for (const it of this.items) {
      const e = extent(it);
      if (e.minX < b.x0 - EPS || e.maxX > b.x1 + EPS || e.minZ < b.z0 - EPS || e.maxZ > b.z1 + EPS) {
        bad.add(it);
        msgs.push(`${it.name}: 벽 밖으로 나가요`);
      }
      if (it.h > b.h + EPS) {
        bad.add(it);
        msgs.push(`${it.name}: 천장(${fmtLen(b.h)})보다 높아요`);
      }
    }
    for (let i = 0; i < this.items.length; i++) {
      for (let j = i + 1; j < this.items.length; j++) {
        const a = this.items[i];
        const c = this.items[j];
        if (overlaps(a, c)) {
          bad.add(a);
          bad.add(c);
          msgs.push(`${a.name} ↔ ${c.name}: 겹쳐요`);
        }
      }
    }
    for (const it of this.items) it.setState(it === this.selected, bad.has(it));
    this.conflicts = msgs;
    this.drawGuides(b);
  }

  drawGuides(b) {
    for (const child of [...this.guides.children]) {
      this.guides.remove(child);
      child.geometry?.dispose();
      child.material?.dispose();
      child.element?.remove();
    }
    this.gaps = null;
    const it = this.selected;
    if (!it) return;
    const e = extent(it);
    const xc = (e.minX + e.maxX) / 2;
    const zc = (e.minZ + e.maxZ) / 2;
    const y = 0.015;
    const segs = {
      left: { a: [b.x0, zc], b: [e.minX, zc], gap: e.minX - b.x0 },
      right: { a: [e.maxX, zc], b: [b.x1, zc], gap: b.x1 - e.maxX },
      up: { a: [xc, b.z0], b: [xc, e.minZ], gap: e.minZ - b.z0 },
      down: { a: [xc, e.maxZ], b: [xc, b.z1], gap: b.z1 - e.maxZ },
    };
    for (const s of Object.values(segs)) {
      const neg = s.gap < -EPS;
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(s.a[0], y, s.a[1]),
        new THREE.Vector3(s.b[0], y, s.b[1]),
      ]);
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: neg ? 0xe25555 : 0x2f6fed, depthTest: false }));
      line.renderOrder = 5;
      this.guides.add(line);
      if (Math.abs(s.gap) >= 0.01) {
        const label = makeLabel(fmtLen(s.gap), neg ? "guide bad" : "guide");
        label.position.set((s.a[0] + s.b[0]) / 2, y, (s.a[1] + s.b[1]) / 2);
        this.guides.add(label);
      }
    }
    this.gaps = Object.fromEntries(Object.entries(segs).map(([k, v]) => [k, v.gap]));
  }
}
