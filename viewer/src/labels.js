import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

export function makeLabel(text, className = "") {
  const el = document.createElement("div");
  el.className = `label ${className}`.trim();
  el.textContent = text;
  const obj = new CSS2DObject(el);
  obj.center.set(0.5, 0.5);
  return obj;
}

/** 0.32 -> "32 cm", 2.4 -> "2.40 m" */
export function fmtLen(m) {
  const cm = m * 100;
  return Math.abs(cm) < 100 ? `${Math.round(cm)} cm` : `${m.toFixed(2)} m`;
}

export function fmtCm(m) {
  return `${Math.round(m * 100)}`;
}
