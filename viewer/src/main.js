import {
  SparkRenderer,
  SplatEdit,
  SplatEditRgbaBlendMode,
  SplatEditSdf,
  SplatEditSdfType,
  SplatMesh,
} from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { FurnitureManager, PRESETS } from "./furniture.js";
import { fmtLen } from "./labels.js";
import { MeasureTool } from "./measure.js";
import { Room } from "./room.js";
import { initIntake, renderAssessment } from "./intake.js";
import { renderProperty, propertyOverlay, structureModel } from "./property.js";
import * as store from "./storage.js";
import { EYE_HEIGHT, Walker } from "./walk.js";

const $ = (id) => document.getElementById(id);
const PYEONG = 3.3058;
const DEMO = {
  name: "__demo__",
  room: { min: [-2, 0, -1.75], max: [2, 2.4, 1.75], height: 2.4, height_source: "assumed" },
  files: {},
  scale: { method: "demo" },
  quality: { warnings: [] },
  cameras: [],
};

// ------------------------------------------------------------------ renderer & scene
const viewport = $("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.localClippingEnabled = true;
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.className = "labels";
viewport.prepend(renderer.domElement, labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe9ecf0);
scene.add(new SparkRenderer({ renderer }));
scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa5b1, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.3);
sun.position.set(3, 6, 2);
scene.add(sun);

const world = new THREE.Group(); // reconstruction, scaled by the user's calibration
scene.add(world);
const room = new Room(world);
const furnitureRoot = new THREE.Group();
const measureRoot = new THREE.Group();
scene.add(furnitureRoot, measureRoot);

// ------------------------------------------------------------------ cameras & controls
const cam = new THREE.PerspectiveCamera(65, 1, 0.03, 200);
const topCam = new THREE.PerspectiveCamera(20, 1, 0.1, 500); // narrow FOV from high up ~ plan view
topCam.up.set(0, 0, -1);
const orbit = new OrbitControls(cam, renderer.domElement);
orbit.enableDamping = true;
orbit.maxDistance = 30;
const topOrbit = new OrbitControls(topCam, renderer.domElement);
topOrbit.enableRotate = false;
topOrbit.screenSpacePanning = true;
topOrbit.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
topOrbit.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN };
topOrbit.enabled = false;
const walker = new Walker(cam, renderer.domElement);
let mode = "orbit";
let activeCam = cam;
let orbitPose = null;

// ------------------------------------------------------------------ state
let current = { name: null, data: null, k: 1 };
let splat = null;
let ceilingCut = null;
let points = null;
let camMarkers = null;
let observedWalls = null;
let sceneLoadVersion = 0;
let houseOverlay = null;
let structure = null;
const pointsClip = new THREE.Plane(new THREE.Vector3(0, -1, 0), 100);

const bounds = () => room.bounds(current.k);

const furniture = new FurnitureManager(furnitureRoot, {
  getBounds: bounds,
  onChange: () => {
    if (current.name && current.data?.layout?.room_box_supported !== false) store.save(current.name, "layout", furniture.toJSON());
    updateFurnitureUI();
  },
  onSelect: () => updateFurnitureUI(),
});

const measure = new MeasureTool(measureRoot, {
  onList: (list) => renderMeasureList(list),
  onPair: () => {
    $("calib-box").hidden = false;
    $("calib-input").value = "";
    $("calib-input").focus();
    setHint("실제 길이를 cm로 입력하고 '적용'을 누르세요");
  },
  onState: (kind) => {
    $("tool-measure").classList.toggle("on", kind === "measure");
    $("tool-calibrate").classList.toggle("on", kind === "calibrate");
    if (!kind) $("calib-box").hidden = true;
    setHint();
  },
});

// ------------------------------------------------------------------ helpers
let toastTimer = 0;
function toast(text, ms = 3500) {
  const el = $("toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), ms);
}

const HINTS = {
  orbit: "드래그 회전 · 휠 확대 · 우클릭 드래그 이동 · 가구는 끌어서 옮기기",
  walk: "WASD·방향키 이동 · Shift 빠르게 · ESC 멈춤",
  top: "드래그로 화면 이동 · 휠 확대 · 가구는 끌어서 옮기기 · Q/E 회전",
  measure: "길이를 잴 두 점을 차례로 클릭 · ESC 종료",
  calibrate: "길이를 아는 두 점(문 폭, 침대 길이 등)을 클릭 · ESC 취소",
};
function setHint(text) {
  $("hint").textContent = text ?? (measure.kind ? HINTS[measure.kind] : HINTS[mode]);
}

function showLoading(text, frac) {
  $("loading").hidden = false;
  $("loading-text").textContent = text;
  $("loading-bar").style.width = `${Math.round((frac ?? 0) * 100)}%`;
}

function disposeObject(o) {
  if (!o) return;
  o.parent?.remove(o);
  o.traverse?.((c) => {
    c.geometry?.dispose?.();
    c.material?.dispose?.();
  });
  o.dispose?.();
}

// ------------------------------------------------------------------ scene loading
async function fetchScenes() {
  try {
    const r = await fetch("/scenes/");
    return r.ok ? await r.json() : [];
  } catch {
    return [];
  }
}

// 천장 높이를 측정(detected)했거나 사용자가 직접 입력(user)한 경우만 확정값으로 본다.
const heightKnown = (r) => r.height_source === "detected" || r.height_source === "user";

function cutHeightLocal(data) {
  const r = data.room;
  return heightKnown(r) ? r.height - 0.12 : Math.min(2.1, r.height - 0.12);
}

async function loadScene(name) {
  const version = ++sceneLoadVersion;
  showLoading("공간 불러오는 중…", 0);
  $("show-splat").disabled = true;
  measure.cancel();
  measure.clearAll();
  for (const o of [splat, points, camMarkers, observedWalls, houseOverlay, structure]) disposeObject(o);
  houseOverlay = null;
  structure = null;
  observedWalls = null;
  splat = points = camMarkers = ceilingCut = null;

  let data = DEMO;
  if (name !== DEMO.name) {
    const r = await fetch(`/scenes/${encodeURIComponent(name)}/scene.json`);
    if (!r.ok) throw new Error(`장면을 불러오지 못했어요: ${name}`);
    data = await r.json();
  }
  if (version !== sceneLoadVersion) return;
  current = { name, data, k: store.load(name, "calibration", 1) };
  world.scale.setScalar(current.k);
  const userHeight = store.load(name, "ceiling", null);  // 사용자가 직접 입력한 천장 높이(m)
  if (typeof userHeight === "number" && userHeight > 1.5 && userHeight < 6) {
    data.room = { ...data.room, height: userHeight, height_source: "user", max: [data.room.max[0], data.room.min[1] + userHeight, data.room.max[2]] };
  }
  room.build(data.room);
  room.updateLabels(current.k);
  if (data.property_graph) {
    houseOverlay=propertyOverlay(THREE,data.property_graph);
    world.add(houseOverlay);
    if (data.property_graph.spaces?.length) {
      structure=structureModel(THREE,data.property_graph,data.room?.height??2.4,data.floorplan,data.objects?.instances);
      world.add(structure);
    }
  }
  renderProperty(data,name,(photo)=>{
    const camera=data.cameras.find(c=>c.name===photo); if (!camera) return;
    setMode('orbit');
    cam.position.set(...camera.position).multiplyScalar(current.k);
    const forward=new THREE.Vector3(0,0,-1).applyQuaternion(new THREE.Quaternion(...camera.quaternion));
    orbit.target.copy(cam.position).addScaledVector(forward,1.5*current.k);
    const damping=orbit.enableDamping; orbit.enableDamping=false; orbit.update(); orbit.enableDamping=damping;
    $("property-active-photo").textContent=`현재 촬영 위치: ${camera.source_photo??photo}`;
    toast(`${camera.source_photo??photo} 촬영 위치`);
  });
  const boxSupported = data.layout?.room_box_supported !== false;
  if (!boxSupported && mode === "walk") setMode("orbit");
  document.querySelector('[data-mode="walk"]').disabled = !boxSupported;
  for (const input of $("furniture-section").querySelectorAll("button,input")) input.disabled = !boxSupported;
  if (!boxSupported) {
    const vertices = [];
    for (const plane of data.layout.planes) {
      const [a, b] = [plane.start, plane.end];
      vertices.push(a[0], 0.03, a[1], b[0], 0.03, b[1]);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    observedWalls = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xd17b21 }));
    world.add(observedWalls);
  }

  const base = `/scenes/${encodeURIComponent(name)}/`;
  if (data.files?.points) {
    const geo = await new PLYLoader().loadAsync(base + data.files.points);
    if (version !== sceneLoadVersion) { geo.dispose(); return; }
    const mat = new THREE.PointsMaterial({ size: 0.025, vertexColors: true, sizeAttenuation: true });
    points = new THREE.Points(geo, mat);
    world.add(points);
  }
  if (data.files?.splat) {
    showLoading("실사 3D 불러오는 중…", 0);
    const loadedSplat = new SplatMesh({
      url: base + data.files.splat,
      raycastable: true,
      onProgress: (e) => version === sceneLoadVersion && e.total && showLoading("실사 3D 불러오는 중…", e.loaded / e.total),
    });
    splat = loadedSplat;
    world.add(splat);
    try {
      await loadedSplat.initialized;
      if (version !== sceneLoadVersion) { disposeObject(loadedSplat); return; }
    } catch (e) {
      if (version !== sceneLoadVersion) return;
      toast(`실사 3D를 불러오지 못했어요: ${e.message ?? e}`);
      disposeObject(splat);
      splat = null;
    }
    $("loading").hidden = true;
    if (splat) {
      // hide everything above the ceiling line so plan / dollhouse views can see in
      ceilingCut = new SplatEdit({ rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY });
      const box = new SplatEditSdf({ type: SplatEditSdfType.BOX, opacity: 0 });
      box.position.set(0, cutHeightLocal(data) + 50, 0);
      box.scale.set(500, 50, 500); // half extents
      ceilingCut.addSdf(box);
      splat.add(ceilingCut);
    }
  }
  if (data.cameras?.length) {
    camMarkers = new THREE.Group();
    const dotGeo = new THREE.SphereGeometry(0.03, 10, 8);
    const dotMat = new THREE.MeshBasicMaterial({ color: 0xd97706 });
    const dirs = [];
    for (const c of data.cameras) {
      const p = new THREE.Vector3(...c.position);
      const m = new THREE.Mesh(dotGeo, dotMat);
      m.position.copy(p);
      camMarkers.add(m);
      const fwd = new THREE.Vector3(0, 0, -0.18).applyQuaternion(new THREE.Quaternion(...c.quaternion));
      dirs.push(p.x, p.y, p.z, p.x + fwd.x, p.y + fwd.y, p.z + fwd.z);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(dirs, 3));
    camMarkers.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xd97706 })));
    world.add(camMarkers);
  }

  $("show-points").checked = !splat;
  $("show-splat").disabled = !splat;
  furniture.load(boxSupported ? store.load(name, "layout", []) : []);
  applyDisplay();
  updateRoomInfo();
  resetViews();
  $("loading").hidden = true;
  const sceneOption = Array.from($("scene-select").options).find(option => option.value === name);
  if (sceneOption && data.source?.title) sceneOption.textContent = data.source.title;
  if (name === DEMO.name) toast("데모 방(4.0 × 3.5 m)이에요. 가구 배치를 먼저 시험해 보세요.");
}

// ------------------------------------------------------------------ display & views
function applyDisplay() {
  const boxSupported = current.data.layout?.room_box_supported !== false;
  const showSplat = $("show-splat").checked && !!splat;
  if (splat) splat.visible = showSplat;
  if (points) points.visible = $("show-points").checked;
  room.outline.visible = $("show-room").checked && boxSupported;
  room.shell.visible = $("show-room").checked && !showSplat && boxSupported;
  room.grid.visible = boxSupported && ($("show-grid").checked || mode === "top");
  if (observedWalls) observedWalls.visible = $("show-room").checked;
  if (houseOverlay) houseOverlay.visible = $("show-property").checked;
  if (structure) structure.visible = $("show-structure").checked;
  if (camMarkers) camMarkers.visible = $("show-cams").checked;
  const cut = $("cut-ceiling").checked && mode !== "walk";
  if (ceilingCut) ceilingCut.visible = cut;
  if (points) {
    pointsClip.constant = cutHeightLocal(current.data) * current.k;
    points.material.clippingPlanes = cut ? [pointsClip] : [];
    points.material.needsUpdate = true;
  }
}

function resetViews() {
  const b = bounds();
  const cx = (b.x0 + b.x1) / 2;
  const cz = (b.z0 + b.z1) / 2;
  const w = b.x1 - b.x0;
  const d = b.z1 - b.z0;
  orbitPose = {
    pos: new THREE.Vector3(b.x0 + 0.12 * w, Math.min(1.75, b.h - 0.2), b.z1 - 0.12 * d),
    target: new THREE.Vector3(cx, 0.7, cz),
  };
  fitTop();
  if (mode === "orbit") restoreOrbit();
  if (mode === "walk") enterWalk();
}

function restoreOrbit() {
  cam.position.copy(orbitPose.pos);
  orbit.target.copy(orbitPose.target);
  orbit.update();
}

function fitTop() {
  const b = bounds();
  const margin = 0.6;
  const half = Math.tan(THREE.MathUtils.degToRad(topCam.fov / 2));
  const aspect = Math.max(topCam.aspect, 0.1);
  const dist = Math.max((b.x1 - b.x0) / 2 + margin, ((b.z1 - b.z0) / 2 + margin) * aspect) / (half * aspect);
  const cx = (b.x0 + b.x1) / 2;
  const cz = (b.z0 + b.z1) / 2;
  topCam.position.set(cx, dist, cz);
  topCam.far = dist + 20;
  topCam.updateProjectionMatrix();
  topOrbit.target.set(cx, 0, cz);
  topOrbit.update();
}

function enterWalk() {
  const b = bounds();
  cam.position.set((b.x0 + b.x1) / 2, Math.min(EYE_HEIGHT, b.h - 0.1), (b.z0 + b.z1) / 2);
  cam.rotation.set(0, 0, 0, "YXZ");
}

function setMode(m) {
  if (m === "walk" && current.data.layout?.room_box_supported === false) {
    toast("벽 경계가 확실하지 않아 걷기는 보류했어요. 3D와 관측 벽 후보를 확인해 주세요.");
    return;
  }
  if (m === mode) return;
  measure.cancel();
  if (mode === "orbit") orbitPose = { pos: cam.position.clone(), target: orbit.target.clone() };
  if (mode === "walk") {
    walker.active = false;
    walker.unlock();
    $("walk-overlay").hidden = true;
  }
  mode = m;
  orbit.enabled = m === "orbit";
  topOrbit.enabled = m === "top";
  activeCam = m === "top" ? topCam : cam;
  if (m === "orbit") restoreOrbit();
  if (m === "walk") {
    furniture.select(null);
    enterWalk();
    walker.active = true;
    $("walk-overlay").hidden = false;
  }
  for (const btn of $("modes").querySelectorAll("button")) btn.classList.toggle("on", btn.dataset.mode === m);
  applyDisplay();
  setHint();
}

// ------------------------------------------------------------------ panel UI
function updateRoomInfo() {
  const { data, k } = current;
  const b = bounds();
  const w = b.x1 - b.x0;
  const d = b.z1 - b.z0;
  $("dim-w").textContent = `${w.toFixed(2)}m`;
  $("dim-d").textContent = `${d.toFixed(2)}m`;
  $("dim-h").textContent = `${b.h.toFixed(2)}m${heightKnown(data.room) ? (data.room.height_source === "user" ? " (입력)" : "") : "*"}`;
  $("ceiling-h").value = Math.round(data.room.height * 100);
  $("ceiling-reset").hidden = data.room.height_source !== "user";
  const area = w * d;
  const boxSupported = data.layout?.room_box_supported !== false;
  $("space-info-title").textContent = boxSupported ? "공간 정보 · 구조 후보" : "관측 범위 · 도면 확정 보류";
  $("dim-w-label").textContent = boxSupported ? "가로" : "범위 가로";
  $("dim-d-label").textContent = boxSupported ? "세로" : "범위 세로";
  $("dim-area").textContent =
    `${data.layout ? "직사각형 가정 면적" : "바닥 면적"} ${area.toFixed(1)}㎡ (${(area / PYEONG).toFixed(1)}평)` +
    (heightKnown(data.room) ? "" : " · *천장 높이는 가정값 (아래에서 직접 입력 가능)");
  if (!boxSupported) $("dim-area").textContent = "바닥 면적 확정 보류 · 관측된 벽 후보만 표시합니다.";

  const el = $("scale-info");
  const method = data.scale?.method;
  el.className = `scale ${method === "marker" || k !== 1 ? "ok" : method === "demo" ? "" : "est"}`;
  el.textContent =
    method === "marker" ? "✓ 인쇄 마커 기준 실측 크기" :
    method === "demo" ? "데모 방 (실제 장면 아님)" :
    "촬영 높이를 가정한 크기 · 실제 길이로 보정 필요";
  if (k !== 1) {
    el.textContent += ` · 보정 ×${k.toFixed(3)}`;
    const reset = document.createElement("button");
    reset.className = "btn small";
    reset.textContent = "보정 취소";
    reset.onclick = () => setCalibration(1);
    el.append(reset);
  }
  const notes = $("warnings");
  const warnings = [...(data.quality?.warnings ?? [])];
  if (data.assessment?.message) warnings.push(data.assessment.message);
  notes.replaceChildren(...warnings.map((t) => Object.assign(document.createElement("li"), { textContent: t })));
  renderAssessment($("assessment-checks"), data.assessment);
  const plan = $("floorplan-download");
  plan.hidden = !data.files?.floorplan;
  if (data.files?.floorplan) {
    plan.href = `/scenes/${encodeURIComponent(current.name)}/${encodeURIComponent(data.files.floorplan)}`;
    plan.textContent = k !== 1 ? "분석 원본 도면 SVG (보정 전)" : "분석 도면 SVG 저장";
  }
}

function setCalibration(k) {
  current.k = k;
  store.save(current.name, "calibration", k);
  world.scale.setScalar(k);
  room.updateLabels(k);
  measure.clearAll();
  applyDisplay();
  updateRoomInfo();
  furniture.refresh();
  updateFurnitureUI();
  fitTop();
}

function renderMeasureList(list) {
  $("measure-list").replaceChildren(
    ...list.map((r, i) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>측정 ${i + 1}: <b>${fmtLen(r.dist)}</b></span>`;
      const del = document.createElement("button");
      del.className = "btn small";
      del.textContent = "지우기";
      del.onclick = () => measure.remove(r.id);
      li.append(del);
      return li;
    }),
  );
}

function updateFurnitureUI() {
  const it = furniture.selected;
  $("selected").hidden = !it;
  if (it) {
    const active = document.activeElement;
    const set = (id, v) => {
      if ($(id) !== active) $(id).value = v;
    };
    set("s-name", it.name);
    set("s-w", Math.round(it.w * 100));
    set("s-d", Math.round(it.d * 100));
    set("s-h", Math.round(it.h * 100));
    const g = furniture.gaps;
    $("s-gaps").textContent = g
      ? `벽까지 ← ${fmtLen(g.left)} · → ${fmtLen(g.right)} · ↑ ${fmtLen(g.up)} · ↓ ${fmtLen(g.down)} (평면도 기준)`
      : "";
  }
  const list = $("conflicts");
  if (furniture.conflicts.length) {
    list.replaceChildren(...furniture.conflicts.map((t) => Object.assign(document.createElement("li"), { textContent: t })));
  } else if (furniture.items.length) {
    list.replaceChildren(Object.assign(document.createElement("li"), { className: "ok", textContent: "✓ 모든 가구가 방 안에 겹치지 않게 들어가요" }));
  } else {
    list.replaceChildren();
  }
}

function buildPresets() {
  $("presets").replaceChildren(
    ...PRESETS.map((p) => {
      const b = document.createElement("button");
      b.className = "btn";
      b.innerHTML = `<span>${p.name}</span><small>${p.w}×${p.d}×${p.h}</small>`;
      b.onclick = () => furniture.addPreset(p);
      return b;
    }),
  );
}

// ------------------------------------------------------------------ pointer input
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let down = null;

function setRay(e) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, activeCam);
}

function pickPoint(includeSplat) {
  const safePlanes = current.data.layout?.room_box_supported === false
    ? (current.data.floor_found ? room.planes.slice(0, 1) : []) : room.planes;
  const targets = [...furniture.meshes(), ...safePlanes];
  if (includeSplat && splat?.visible) targets.push(splat);
  if (includeSplat && points?.visible) targets.push(points);
  // Spark returns every splat along the ray and ignores edits: skip what the ceiling cut hides
  const cutting = $("cut-ceiling").checked && mode !== "walk";
  const cut = cutting ? cutHeightLocal(current.data) * current.k : Infinity;
  const hit = raycaster.intersectObjects(targets, false).find((h) => h.point.y < cut || furniture.meshes().includes(h.object));
  return hit ? hit.point.clone() : null;
}

viewport.addEventListener(
  "pointerdown",
  (e) => {
    if (e.target !== renderer.domElement || mode === "walk") return;
    setRay(e);
    down = { x: e.clientX, y: e.clientY };
    if (measure.active || e.button !== 0) return;
    const it = furniture.pick(raycaster);
    if (it) {
      furniture.select(it);
      if (furniture.beginDrag(it, raycaster)) {
        orbit.enabled = topOrbit.enabled = false;
        e.stopPropagation(); // keep the camera still while dragging furniture
      }
    }
  },
  { capture: true },
);

viewport.addEventListener(
  "pointermove",
  (e) => {
    if (mode === "walk") return;
    setRay(e);
    if (furniture.dragging) {
      furniture.dragTo(raycaster);
      updateFurnitureUI();
    } else if (measure.active && measure.first) {
      measure.hover(pickPoint(false));
    }
  },
  { capture: true },
);

window.addEventListener("pointerup", (e) => {
  if (furniture.dragging) {
    furniture.endDrag();
    orbit.enabled = mode === "orbit";
    topOrbit.enabled = mode === "top";
    down = null;
    return;
  }
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  down = null;
  if (moved > 5 || e.target !== renderer.domElement) return;
  setRay(e);
  if (measure.active) {
    const p = pickPoint(true);
    if (p) measure.click(p);
    else toast("그 위치에서는 표면을 찾지 못했어요");
    return;
  }
  if (!furniture.pick(raycaster)) furniture.select(null);
});

$("walk-overlay").addEventListener("click", () => walker.lock());
walker.controls.addEventListener("lock", () => ($("walk-overlay").hidden = true));
walker.controls.addEventListener("unlock", () => {
  if (mode === "walk") $("walk-overlay").hidden = false;
});

window.addEventListener("keydown", (e) => {
  if (e.target.matches?.("input, select, textarea")) return;
  if (e.key === "Escape") {
    if (measure.kind) measure.cancel();
    else furniture.select(null);
    return;
  }
  if (mode === "walk") return;
  if (e.key === "1") setMode("orbit");
  if (e.key === "2") setMode("walk");
  if (e.key === "3") setMode("top");
  const it = furniture.selected;
  if (!it) return;
  const step = e.shiftKey ? 1 : 15;
  if (e.code === "KeyQ") furniture.rotate(it, step);
  else if (e.code === "KeyE") furniture.rotate(it, -step);
  else if (e.code === "KeyR") furniture.rotate(it, -90);
  else if (e.key === "Delete" || e.key === "Backspace") furniture.remove(it);
});

// ------------------------------------------------------------------ panel wiring
for (const btn of $("modes").querySelectorAll("button")) btn.onclick = () => setMode(btn.dataset.mode);
for (const id of ["show-splat", "show-points", "show-room", "show-grid", "cut-ceiling", "show-cams", "show-property", "show-structure"]) {
  $(id).onchange = applyDisplay;
}
// 구조 모형을 켜면 집 전체를 비스듬히 내려다보는 시점(모형 집 보기)으로 옮긴다.
$("show-structure").addEventListener("change", () => {
  if (!$("show-structure").checked || !structure) return;
  const box = new THREE.Box3().setFromObject(structure), c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  const r = Math.max(size.x, size.z);
  setMode("orbit");
  cam.position.set(c.x + 0.3 * r, 1.15 * r, c.z + 0.85 * r);
  orbit.target.set(c.x, 0, c.z);
  orbit.update();
});
$("tool-measure").onclick = () => (measure.kind === "measure" ? measure.cancel() : (setMode(mode === "walk" ? "orbit" : mode), measure.start("measure")));
$("tool-calibrate").onclick = () => {
  if (measure.kind === "calibrate") return measure.cancel();
  if (mode === "walk") setMode("orbit");
  measure.start("calibrate");
  toast("길이를 아는 두 점을 클릭하세요. 예: 문 폭, 침대 길이, 마커 한 변");
};
$("calib-apply").onclick = () => {
  const real = parseFloat($("calib-input").value) / 100;
  const measured = measure.pending;
  if (!(real > 0) || !(measured > 0)) return toast("실제 길이를 cm로 입력해 주세요");
  const k = current.k * (real / measured);
  measure.cancel();
  setCalibration(k);
  toast(`크기를 ×${(real / measured).toFixed(3)} 보정했어요`);
};
$("calib-input").addEventListener("keydown", (e) => e.key === "Enter" && $("calib-apply").click());
$("calib-cancel").onclick = () => measure.cancel();

$("c-add").onclick = () => {
  const w = parseFloat($("c-w").value) / 100;
  const d = parseFloat($("c-d").value) / 100;
  const h = parseFloat($("c-h").value) / 100;
  if (!(w > 0 && d > 0 && h > 0)) return toast("가로·깊이·높이를 cm로 입력해 주세요");
  furniture.add({ name: $("c-name").value.trim() || "가구", w, d, h, color: "#b9c3cd" });
};
for (const id of ["s-name", "s-w", "s-d", "s-h"]) {
  $(id).addEventListener("change", () => {
    const it = furniture.selected;
    if (!it) return;
    furniture.resize(it, {
      name: $("s-name").value.trim(),
      w: parseFloat($("s-w").value) / 100,
      d: parseFloat($("s-d").value) / 100,
      h: parseFloat($("s-h").value) / 100,
    });
  });
}
$("s-rot-l").onclick = () => furniture.selected && furniture.rotate(furniture.selected, 15);
$("s-rot-r").onclick = () => furniture.selected && furniture.rotate(furniture.selected, -15);
$("s-rot-90").onclick = () => furniture.selected && furniture.rotate(furniture.selected, -90);
$("s-dup").onclick = () => furniture.selected && furniture.duplicate(furniture.selected);
$("s-del").onclick = () => furniture.selected && furniture.remove(furniture.selected);

$("layout-save").onclick = () =>
  store.downloadJSON(`layout-${current.name}.json`, {
    scene: current.name,
    calibration: current.k,
    furniture: furniture.toJSON(),
  });
$("layout-load").onclick = async () => {
  try {
    const data = await store.pickJSON();
    furniture.load(Array.isArray(data) ? data : data.furniture);
    store.save(current.name, "layout", furniture.toJSON());
    updateFurnitureUI();
    toast("배치를 불러왔어요");
  } catch {
    toast("배치 파일을 읽지 못했어요");
  }
};
let clearArmed = 0;
$("layout-clear").onclick = () => {
  const btn = $("layout-clear");
  if (Date.now() - clearArmed < 3000) {
    furniture.clear();
    btn.textContent = "모두 지우기";
    clearArmed = 0;
    return;
  }
  clearArmed = Date.now();
  btn.textContent = "한 번 더 누르면 삭제";
  setTimeout(() => (btn.textContent = "모두 지우기"), 3000);
};

$("scene-select").onchange = async (e) => {
  const name = e.target.value;
  const url = new URL(location.href);
  url.searchParams.set("scene", name);
  history.replaceState(null, "", url);
  try {
    await loadScene(name);
  } catch (err) {
    toast(err.message);
  }
};

// ------------------------------------------------------------------ boot
function resize() {
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  for (const c of [cam, topCam]) {
    c.aspect = w / h;
    c.updateProjectionMatrix();
  }
}
new ResizeObserver(resize).observe(viewport);
resize();

let last = performance.now();
let processingJob = false, lastDraw = 0;
window.addEventListener("spacecapture-processing", event => { processingJob = event.detail; });
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  // The 780M shares its GPU with Brush. Keep the preview responsive at 5 fps
  // while a background analysis runs, leaving compute time for training.
  if (processingJob && now - lastDraw < 200) return;
  lastDraw = now;
  if (mode === "walk") walker.update(dt, bounds());
  else if (mode === "top") topOrbit.update();
  else orbit.update();
  renderer.render(scene, activeCam);
  labelRenderer.render(scene, activeCam);
});

if (import.meta.env.DEV) {
  // debugging handle for the dev server only (not in production builds)
  window.__spacecapture = { THREE, scene, world, room, furniture, measure, walker, raycaster, cam, topCam, get splat() { return splat; }, get activeCam() { return activeCam; }, pickPoint };
}

(async function boot() {
  buildPresets();
  initIntake(async (name, title) => {
    const select = $("scene-select");
    if (!Array.from(select.options).some(option => option.value === name)) select.add(new Option(title || name, name), 0);
    select.value = name;
    const url = new URL(location.href); url.searchParams.set("scene", name);
    history.replaceState(null, "", url);
    await loadScene(name);
  });
  const scenes = await fetchScenes();
  const select = $("scene-select");
  for (const s of scenes) {
    const r = s.room;
    const opt = new Option(`${s.name}  ${r.width.toFixed(1)}×${r.depth.toFixed(1)}m${s.splat ? "" : " (점)"}`, s.name);
    select.add(opt);
  }
  select.add(new Option("데모: 빈 방", DEMO.name));
  const wanted = new URLSearchParams(location.search).get("scene");
  const name = scenes.some((s) => s.name === wanted) || wanted === DEMO.name ? wanted : scenes[0]?.name ?? DEMO.name;
  select.value = name;
  setHint();
  try {
    await loadScene(name);
  } catch (err) {
    toast(err.message);
    await loadScene(DEMO.name);
  }
  if (!scenes.length) toast("처리된 장면이 없어요. python -m spacecapture process 영상.mp4 로 만들어 보세요.", 6000);
})();

// 천장 높이 직접 입력: 천장이 충분히 보이지 않아 가정값(2.4m)일 때 줄자로 잰 값을 장면별로 저장한다.
$("ceiling-save").onclick = () => {
  const cm = Number($("ceiling-h").value);
  if (!current.name || !(cm >= 180 && cm <= 500)) { toast("천장 높이는 180~500cm 사이로 입력해 주세요."); return; }
  store.save(current.name, "ceiling", cm / 100);
  toast(`천장 높이 ${cm}cm를 이 장면에 저장했어요.`);
  loadScene(current.name);
};
$("ceiling-reset").onclick = () => {
  if (!current.name) return;
  store.save(current.name, "ceiling", null);
  toast("입력한 천장 높이를 지우고 분석값으로 되돌렸어요.");
  loadScene(current.name);
};
