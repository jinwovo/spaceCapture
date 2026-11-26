import * as THREE from "three";
import { fmtLen, makeLabel } from "./labels.js";

const DOT = new THREE.SphereGeometry(0.015, 16, 12);

function dot(p, color) {
  const m = new THREE.Mesh(DOT, new THREE.MeshBasicMaterial({ color, depthTest: false }));
  m.renderOrder = 10;
  m.position.copy(p);
  return m;
}

function segment(a, b, color) {
  const g = new THREE.BufferGeometry().setFromPoints([a, b]);
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, depthTest: false }));
  l.renderOrder = 10;
  return l;
}

/**
 * Two-click distance tool. kind = "measure" keeps a list of results,
 * kind = "calibrate" hands the measured pair to the app to rescale the scene.
 */
export class MeasureTool {
  constructor(root, { onList, onPair, onState }) {
    this.root = root;
    this.onList = onList;
    this.onPair = onPair;
    this.onState = onState;
    this.kind = null;
    this.first = null;
    this.temp = [];
    this.preview = null;
    this.results = [];
    this.pending = null;
    this.nextId = 1;
  }

  get active() {
    return this.kind !== null && this.pending === null;
  }

  start(kind) {
    this.cancel();
    this.kind = kind;
    this.onState?.(kind);
  }

  cancel() {
    this.clearTemp();
    this.kind = null;
    this.first = null;
    this.pending = null;
    this.onState?.(null);
  }

  clearTemp() {
    for (const o of this.temp) this.dispose(o);
    this.temp = [];
    if (this.preview) {
      for (const o of this.preview) this.dispose(o);
      this.preview = null;
    }
  }

  dispose(o) {
    this.root.remove(o);
    if (o.geometry && o.geometry !== DOT) o.geometry.dispose();
    o.material?.dispose();
    o.element?.remove();
  }

  hover(p) {
    if (!this.first || this.pending) return;
    if (this.preview) for (const o of this.preview) this.dispose(o);
    this.preview = null;
    if (!p) return;
    const label = makeLabel(fmtLen(this.first.distanceTo(p)), "measure");
    label.position.copy(this.first).lerp(p, 0.5);
    this.preview = [segment(this.first, p, 0x1b2129), label];
    this.root.add(...this.preview);
  }

  click(p) {
    if (!this.active) return false;
    const color = this.kind === "calibrate" ? 0xd97706 : 0x1b2129;
    if (!this.first) {
      this.first = p.clone();
      this.temp.push(dot(p, color));
      this.root.add(...this.temp);
      return true;
    }
    const a = this.first;
    const b = p.clone();
    const dist = a.distanceTo(b);
    if (this.preview) for (const o of this.preview) this.dispose(o);
    this.preview = null;
    const label = makeLabel(fmtLen(dist), "measure");
    label.position.copy(a).lerp(b, 0.5);
    const objs = [dot(b, color), segment(a, b, color), label];
    this.root.add(...objs);
    if (this.kind === "measure") {
      this.results.push({ id: this.nextId++, dist, objs: [...this.temp, ...objs] });
      this.temp = [];
      this.first = null;
      this.onList?.(this.results);
    } else {
      this.temp.push(...objs);
      this.pending = dist;
      this.onPair?.(dist);
    }
    return true;
  }

  remove(id) {
    const r = this.results.find((x) => x.id === id);
    if (!r) return;
    for (const o of r.objs) this.dispose(o);
    this.results = this.results.filter((x) => x !== r);
    this.onList?.(this.results);
  }

  clearAll() {
    for (const r of this.results) for (const o of r.objs) this.dispose(o);
    this.results = [];
    this.onList?.(this.results);
  }
}
