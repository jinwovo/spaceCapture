import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";

export const EYE_HEIGHT = 1.6;

/** First-person walking at eye height, kept inside the room box. */
export class Walker {
  constructor(camera, dom) {
    this.camera = camera;
    this.controls = new PointerLockControls(camera, dom);
    this.keys = new Set();
    this.active = false;
    window.addEventListener("keydown", (e) => {
      if (this.active && !e.target.matches?.("input, select, textarea")) this.keys.add(e.code);
    });
    window.addEventListener("keyup", (e) => this.keys.delete(e.code));
    window.addEventListener("blur", () => this.keys.clear());
  }

  get locked() {
    return this.controls.isLocked;
  }

  lock() {
    this.controls.lock();
  }

  unlock() {
    if (this.controls.isLocked) this.controls.unlock();
    this.keys.clear();
  }

  update(dt, b) {
    if (!this.active || !this.controls.isLocked) return;
    const k = this.keys;
    const speed = (k.has("ShiftLeft") || k.has("ShiftRight") ? 2.8 : 1.4) * dt;
    const forward = (k.has("KeyW") || k.has("ArrowUp") ? 1 : 0) - (k.has("KeyS") || k.has("ArrowDown") ? 1 : 0);
    const right = (k.has("KeyD") || k.has("ArrowRight") ? 1 : 0) - (k.has("KeyA") || k.has("ArrowLeft") ? 1 : 0);
    if (forward) this.controls.moveForward(forward * speed);
    if (right) this.controls.moveRight(right * speed);
    const p = this.camera.position;
    const m = 0.15;
    p.x = Math.min(Math.max(p.x, b.x0 + m), b.x1 - m);
    p.z = Math.min(Math.max(p.z, b.z0 + m), b.z1 - m);
    p.y = Math.min(EYE_HEIGHT, b.h - 0.1);
  }
}
