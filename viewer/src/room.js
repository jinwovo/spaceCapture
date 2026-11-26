import * as THREE from "three";
import { fmtLen, makeLabel } from "./labels.js";

/**
 * Room box in the raw scene frame (child of the calibrated scene root):
 * pickable planes for measuring, outline, translucent shell (when there is no
 * photo-real splat), floor grid and dimension labels.
 */
export class Room {
  constructor(parent) {
    this.group = new THREE.Group();
    this.outline = new THREE.Group();
    this.shell = new THREE.Group();
    this.grid = new THREE.Group();
    this.planes = [];
    this.labels = {};
    this.group.add(this.outline, this.shell, this.grid);
    parent.add(this.group);
  }

  clear() {
    for (const g of [this.outline, this.shell, this.grid]) {
      for (const child of [...g.children]) {
        g.remove(child);
        child.geometry?.dispose();
        child.material?.dispose();
      }
    }
    for (const p of this.planes) {
      this.group.remove(p);
      p.geometry.dispose();
    }
    this.planes = [];
    this.labels = {};
  }

  build(room) {
    this.clear();
    this.raw = room;
    const [x0, , z0] = room.min;
    const [x1, h, z1] = room.max;
    const w = x1 - x0;
    const d = z1 - z0;
    const cx = (x0 + x1) / 2;
    const cz = (z0 + z1) / 2;

    // invisible planes: targets for measuring and furniture dragging
    const pickMat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    const addPlane = (pw, ph, pos, rot) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(pw, ph), pickMat);
      m.position.set(...pos);
      m.rotation.set(...rot);
      m.visible = false;
      this.group.add(m);
      this.planes.push(m);
    };
    addPlane(w, d, [cx, 0, cz], [-Math.PI / 2, 0, 0]);
    addPlane(w, d, [cx, h, cz], [Math.PI / 2, 0, 0]);
    addPlane(d, h, [x0, h / 2, cz], [0, Math.PI / 2, 0]);
    addPlane(d, h, [x1, h / 2, cz], [0, -Math.PI / 2, 0]);
    addPlane(w, h, [cx, h / 2, z0], [0, 0, 0]);
    addPlane(w, h, [cx, h / 2, z1], [0, Math.PI, 0]);

    // outline
    const box = new THREE.Box3(new THREE.Vector3(x0, 0, z0), new THREE.Vector3(x1, h, z1));
    const helper = new THREE.Box3Helper(box, 0x2f6fed);
    helper.material.transparent = true;
    helper.material.opacity = 0.75;
    this.outline.add(helper);

    // translucent shell (used when only the sparse point cloud is available)
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(w, d),
      new THREE.MeshStandardMaterial({ color: 0xd9dde3, roughness: 1, transparent: true, opacity: 0.9 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(cx, -0.002, cz);
    this.shell.add(floor);
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0xf3f4f6, roughness: 1, transparent: true, opacity: 0.35, side: THREE.BackSide, depthWrite: false,
    });
    const walls = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), wallMat);
    walls.position.set(cx, h / 2, cz);
    this.shell.add(walls);

    // 10 cm grid with 1 m lines, clipped to the room footprint
    const fine = [];
    const major = [];
    for (let x = Math.ceil(x0 * 10) / 10; x <= x1 + 1e-6; x += 0.1) {
      (Math.abs(x - Math.round(x)) < 1e-6 ? major : fine).push(x, 0.002, z0, x, 0.002, z1);
    }
    for (let z = Math.ceil(z0 * 10) / 10; z <= z1 + 1e-6; z += 0.1) {
      (Math.abs(z - Math.round(z)) < 1e-6 ? major : fine).push(x0, 0.002, z, x1, 0.002, z);
    }
    const lines = (arr, color, opacity) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
      return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
    };
    this.grid.add(lines(fine, 0x5a6575, 0.18), lines(major, 0x2f6fed, 0.45));

    // dimension labels on the edges facing the default camera corner (x0, z1)
    this.labels.width = makeLabel("", "dim");
    this.labels.width.position.set(cx, 0.02, z0);
    this.labels.depth = makeLabel("", "dim");
    this.labels.depth.position.set(x1, 0.02, cz);
    this.labels.height = makeLabel("", "dim");
    this.labels.height.position.set(x1, h / 2, z0);
    this.outline.add(this.labels.width, this.labels.depth, this.labels.height);
  }

  updateLabels(k) {
    if (!this.raw) return;
    const r = this.raw;
    this.labels.width.element.textContent = fmtLen((r.max[0] - r.min[0]) * k);
    this.labels.depth.element.textContent = fmtLen((r.max[2] - r.min[2]) * k);
    const hText = fmtLen(r.height * k);
    this.labels.height.element.textContent = r.height_source === "detected" ? hText : `${hText} (가정)`;
  }

  /** bounds in calibrated world meters */
  bounds(k) {
    const r = this.raw;
    return { x0: r.min[0] * k, x1: r.max[0] * k, z0: r.min[2] * k, z1: r.max[2] * k, h: r.height * k };
  }
}
