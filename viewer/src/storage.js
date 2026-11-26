// Per-scene settings (furniture layout, scale calibration) kept in this browser only.
const key = (scene, what) => `spacecapture:${scene}:${what}`;

export function load(scene, what, fallback) {
  try {
    const raw = localStorage.getItem(key(scene, what));
    return raw == null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function save(scene, what, value) {
  try {
    localStorage.setItem(key(scene, what), JSON.stringify(value));
  } catch {
    /* private mode / quota: settings just won't persist */
  }
}

export function downloadJSON(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export function pickJSON() {
  return new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json,.json";
    input.onchange = async () => {
      try {
        resolve(JSON.parse(await input.files[0].text()));
      } catch (e) {
        reject(e);
      }
    };
    input.click();
  });
}
